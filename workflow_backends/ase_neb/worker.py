from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
from workflow_backends.sampling import (
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingSeedSource,
)
from workflow_backends.sampling_models import OrderedSamplingModels
from workflow_core.common.sampling_plan import (
    ExplicitSamplingPlan,
    NebSamplingParameters,
    SamplingConditionGroup,
    SeedRole,
)
from workflow_core.structures.formats import read_extxyz_structures


REQUEST_SCHEMA = "dice.native.ase-neb-request"
REQUEST_REVISION = 4
POLICY_IDENTITY = {"name": "committee-force-deviation", "revision": 1}


class _DeepMDCalculatorProvider:
    policy_identity = POLICY_IDENTITY

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        from deepmd.calculator import DP

        del DP
        if not model_paths:
            raise ValueError("DeepMD NEB requires at least one ordered model")

    def create(self, model_path: Path):
        from deepmd.calculator import DP

        return DP(model=model_path)


class _NepCalculatorProvider:
    policy_identity = POLICY_IDENTITY

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        from torchnep.ase_calculator import NEP

        del NEP
        if not model_paths:
            raise ValueError("NEP NEB requires at least one ordered model")

    def create(self, model_path: Path):
        from torchnep.ase_calculator import NEP

        return NEP(model_file=model_path)


def execute_manifest(path: str | Path) -> None:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ASE NEB worker request is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_revision",
        "run_json_path",
        "iter",
        "round",
        "sample_dir",
        "model_family",
        "operation_role",
        "operation_profile_id",
        "allow_empty_observations",
        "max_workers",
        "models",
        "seeds",
        "groups",
    }:
        raise ValueError("ASE NEB worker request fields are malformed")
    if (
        payload["schema"] != REQUEST_SCHEMA
        or payload["schema_revision"] != REQUEST_REVISION
    ):
        raise ValueError("ASE NEB worker request schema is unsupported")
    max_workers = payload["max_workers"]
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 64
    ):
        raise ValueError("ASE NEB worker max_workers is unsupported")
    model_family = str(payload["model_family"])
    if model_family not in {"deepmd-tf", "deepmd-pt", "nep"}:
        raise ValueError("ASE NEB worker model family is unsupported")
    raw_models = payload["models"]
    raw_seeds = payload["seeds"]
    raw_groups = payload["groups"]
    if (
        not isinstance(raw_models, list)
        or not raw_models
        or not isinstance(raw_seeds, list)
        or not raw_seeds
        or not isinstance(raw_groups, list)
        or not raw_groups
    ):
        raise ValueError("ASE NEB worker requires models, seeds, and groups")
    models: list[Path] = []
    member_ids: list[str] = []
    for model in raw_models:
        if not isinstance(model, dict) or set(model) != {"member_id", "path"}:
            raise ValueError("ASE NEB worker model inventory is malformed")
        member_ids.append(str(model["member_id"]))
        models.append(Path(model["path"]).resolve())
    seeds: list[SeedRole] = []
    systems: list[SamplingSeedSource] = []
    for seed in raw_seeds:
        if not isinstance(seed, dict) or set(seed) != {
            "sampling_role_id",
            "selection_group_id",
            "path",
        }:
            raise ValueError("ASE NEB worker seed inventory is malformed")
        role_id = str(seed["sampling_role_id"])
        seeds.append(SeedRole(role_id, str(seed["selection_group_id"])))
        systems.append(
            SamplingSeedSource(
                role_id,
                tuple(
                    read_extxyz_structures(
                        Path(seed["path"]).resolve(), allow_empty=False
                    )
                ),
            )
        )
    groups: list[SamplingConditionGroup] = []
    for group in raw_groups:
        required_group_fields = {
            "condition_group_id",
            "task_id",
            "sampling_role_id",
            "images",
            "fmax_ev_a",
            "max_steps",
            "include_history",
            "init",
            "method",
            "climb",
            "fixed_atoms",
        }
        optional_group_fields = {"relax", "climb_preconverge_fmax_ev_a"}
        if (
            not isinstance(group, dict)
            or not required_group_fields <= set(group)
            or not set(group) - required_group_fields <= optional_group_fields
            or ("relax" in group and group["relax"] is not True)
        ):
            raise ValueError("ASE NEB worker group inventory is malformed")
        if not isinstance(group["fixed_atoms"], list):
            raise ValueError("ASE NEB worker fixed_atoms inventory is malformed")
        groups.append(
            SamplingConditionGroup(
                condition_group_id=str(group["condition_group_id"]),
                kind="neb",
                sampling_role_ids=(str(group["sampling_role_id"]),),
                neb=NebSamplingParameters(
                    images=group["images"],
                    fmax_ev_a=group["fmax_ev_a"],
                    max_steps=group["max_steps"],
                    include_history=group["include_history"],
                    init=group["init"],
                    method=group["method"],
                    climb=group["climb"],
                    climb_preconverge_fmax_ev_a=group.get(
                        "climb_preconverge_fmax_ev_a"
                    ),
                    fixed_atoms=tuple(group["fixed_atoms"]),
                    relax=group.get("relax", False),
                ),
            )
        )
    round_index = int(payload["round"])
    plan = ExplicitSamplingPlan(
        seeds=tuple(seeds),
        condition_groups=tuple(groups),
        round_count=round_index + 1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=Path(payload["run_json_path"]),
        iter_index=int(payload["iter"]),
        round_index=round_index,
        sample_dir=Path(payload["sample_dir"]),
        plan=plan,
        systems=tuple(systems),
        models=OrderedSamplingModels(
            tuple(models), member_ids=tuple(member_ids)
        ),
        steps=None,
    )
    provider = (
        _NepCalculatorProvider()
        if model_family == "nep"
        else _DeepMDCalculatorProvider()
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=provider,
        model_family=model_family,
        operation_role=str(payload["operation_role"]),
        operation_profile_id=str(payload["operation_profile_id"]),
        allow_empty_observations=bool(payload["allow_empty_observations"]),
        max_workers=max_workers,
        task_id_overrides={
            str(group["condition_group_id"]): str(group["task_id"])
            for group in raw_groups
        },
    )
    backend.execute(SamplingExecutionRequest(evidence))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    execute_manifest(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
