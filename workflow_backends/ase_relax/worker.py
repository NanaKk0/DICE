from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps
from workflow_backends.sampling import (
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingSeedSource,
)
from workflow_backends.sampling_models import OrderedSamplingModels
from workflow_core.common.sampling_plan import (
    ExplicitSamplingPlan,
    RelaxSamplingParameters,
    SamplingConditionGroup,
    SeedRole,
)
from workflow_core.structures.formats import read_extxyz_structures


REQUEST_SCHEMA = "dice.native.ase-relax-request"
REQUEST_REVISION = 1
POLICY_IDENTITY = {"name": "committee-force-deviation", "revision": 1}


class _DeepMDCalculatorProvider:
    policy_identity = POLICY_IDENTITY

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        from deepmd.calculator import DP

        del DP
        if not model_paths:
            raise ValueError("DeepMD relax requires at least one ordered model")

    def create(self, model_path: Path):
        from deepmd.calculator import DP

        return DP(model=model_path)


class _NepCalculatorProvider:
    policy_identity = POLICY_IDENTITY

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        from torchnep.ase_calculator import NEP

        del NEP
        if not model_paths:
            raise ValueError("NEP relax requires at least one ordered model")

    def create(self, model_path: Path):
        from torchnep.ase_calculator import NEP

        return NEP(model_file=model_path)


def load_manifest(path: str | Path):
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ASE relax worker request is unreadable") from exc
    expected = {
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
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("ASE relax worker request fields are malformed")
    if (
        payload["schema"] != REQUEST_SCHEMA
        or payload["schema_revision"] != REQUEST_REVISION
    ):
        raise ValueError("ASE relax worker request schema is unsupported")
    max_workers = payload["max_workers"]
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 64
    ):
        raise ValueError("ASE relax worker max_workers is unsupported")
    model_family = str(payload["model_family"])
    if model_family not in {"deepmd-tf", "deepmd-pt", "nep"}:
        raise ValueError("ASE relax worker model family is unsupported")
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
        raise ValueError("ASE relax worker requires models, seeds, and groups")
    models: list[Path] = []
    member_ids: list[str] = []
    for model in raw_models:
        if not isinstance(model, dict) or set(model) != {"member_id", "path"}:
            raise ValueError("ASE relax worker model inventory is malformed")
        member_ids.append(str(model["member_id"]))
        models.append(Path(model["path"]).resolve())
    source_index_by_role = {
        str(group["sampling_role_id"]): int(group["source_frame_index"])
        for group in raw_groups
        if isinstance(group, dict)
        and "sampling_role_id" in group
        and "source_frame_index" in group
    }
    seeds: list[SeedRole] = []
    systems: list[SamplingSeedSource] = []
    for seed in raw_seeds:
        if not isinstance(seed, dict) or set(seed) != {
            "sampling_role_id",
            "selection_group_id",
            "path",
        }:
            raise ValueError("ASE relax worker seed inventory is malformed")
        role_id = str(seed["sampling_role_id"])
        seeds.append(
            SeedRole(
                role_id,
                str(seed["selection_group_id"]),
                source_index_by_role.get(role_id, 0),
            )
        )
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
    required_group_fields = {
        "condition_group_id",
        "task_id",
        "sampling_role_id",
        "source_frame_index",
        "fmax_ev_a",
        "max_steps",
        "max_step_a",
        "include_history",
        "history_stride",
        "fixed_atoms",
    }
    groups: list[SamplingConditionGroup] = []
    for group in raw_groups:
        group_fields = set(group) if isinstance(group, dict) else set()
        if (
            not isinstance(group, dict)
            or not required_group_fields.issubset(group)
            or group_fields - required_group_fields not in (
                set(),
                {"accept_unconverged"},
            )
        ):
            raise ValueError("ASE relax worker group inventory is malformed")
        if not isinstance(group["fixed_atoms"], list):
            raise ValueError("ASE relax worker fixed_atoms inventory is malformed")
        groups.append(
            SamplingConditionGroup(
                condition_group_id=str(group["condition_group_id"]),
                kind="relax",
                sampling_role_ids=(str(group["sampling_role_id"]),),
                relax=RelaxSamplingParameters(
                    fmax_ev_a=group["fmax_ev_a"],
                    max_steps=group["max_steps"],
                    max_step_a=group["max_step_a"],
                    include_history=group["include_history"],
                    history_stride=group["history_stride"],
                    accept_unconverged=group.get("accept_unconverged", False),
                    fixed_atoms=tuple(group["fixed_atoms"]),
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
        models=OrderedSamplingModels(tuple(models), member_ids=tuple(member_ids)),
        steps=None,
    )
    provider = _NepCalculatorProvider() if model_family == "nep" else _DeepMDCalculatorProvider()
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=provider,
        model_family=model_family,
        operation_role=str(payload["operation_role"]),
        operation_profile_id=str(payload["operation_profile_id"]),
        allow_empty_observations=bool(payload["allow_empty_observations"]),
        max_workers=max_workers,
        task_id_overrides={
            (
                str(group["condition_group_id"]),
                str(group["sampling_role_id"]),
                int(group["source_frame_index"]),
            ): str(group["task_id"])
            for group in raw_groups
        },
    )
    return backend, evidence


def execute_manifest(path: str | Path) -> None:
    backend, evidence = load_manifest(path)
    backend.execute(SamplingExecutionRequest(evidence))


class PersistentRelaxAdapter:
    def __init__(self, path):
        self.backend, self.evidence = load_manifest(path)
        from workflow_backends.ase_relax.model_adapters import model_adapter
        self.backend.calculator_provider = model_adapter(self.backend.model_family)
        self.backend.calculator_provider.initialize()
        self.backend.retain_prefix = True
        self.tasks = {task.task_id: task for task in self.backend._tasks(self.evidence)}
        self.models = tuple(Path(p).resolve() for p in self.evidence.models.paths)
        self.identity = {'family': self.backend.model_family, 'models': [
            {'member_id': member, 'path': str(path)}
            for member, path in zip(self.evidence.models.member_ids, self.models, strict=True)]}
        self.backend.calculator_provider.preflight(self.models)
        for model in self.models:
            self.backend._calculator(model)

    def execute(self, task_id):
        task = self.tasks[task_id]
        outcome = self.backend.execute_individual(self.evidence, task)
        return {'task_id': task_id, 'status': outcome.status, 'diagnostic': outcome.diagnostic}

    def close(self):
        self.backend.calculator_provider.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--channel", type=int)
    args = parser.parse_args(argv)
    if args.channel is None:
        execute_manifest(args.manifest)
    else:
        from workflow_backends.common.persistent_worker import serve
        serve(args.channel, lambda: PersistentRelaxAdapter(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
