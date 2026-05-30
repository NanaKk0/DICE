from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.teacher.base import TeacherBackend, TeacherModelArtifact, TeacherRelabelRequest, TeacherRelabelResult
from workflow_backends.teacher.committee_labeling import (
    compute_force_uncertainty,
    ensure_stress_matrix,
    label_with_committee,
)
from workflow_backends.teacher.mace.backend import build_mace_calculator, print_backend_banner, resolve_mace_backend
from workflow_backends.teacher.mace.env import ensure_backend_environment, probe_mace_environment
from workflow_backends.teacher.mace.finetune import load_mace_committee_finetune_config, run_mace_committee_finetune
from workflow_backends.teacher.mace.foundation import run_mace_foundation_train
from workflow_backends.teacher.mace.model_check import (
    ensure_committee_backend_compatible,
    infer_mace_backend,
    load_mace_model_metadata,
)
from workflow_backends.teacher.mace.relabel_worker import run_mace_relabel_worker
from workflow_runtime.environment import teacher_backend_role

_ensure_stress_matrix = ensure_stress_matrix


def _normalize_teacher_num(value: Any) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    return normalized if normalized > 0 else None


def _metadata_teacher_num(teacher_dir: Path) -> int | None:
    metadata = load_mace_model_metadata(teacher_dir)
    if not metadata:
        return None
    value = metadata.get("teacher_num")
    if value is None:
        return None
    normalized = int(value)
    return normalized if normalized > 0 else None


def _resolve_mace_model_paths(teacher_dir: Path, teacher_num: int | None) -> list[Path]:
    if teacher_num is None:
        raise RuntimeError(
            f"MACE teacher relabel requires teacher_num for committee {teacher_dir}. "
            "Use the teacher committee finetune config snapshot or mace_model_metadata.json."
        )

    model_paths = [teacher_dir / f"teacher_{i:02d}" / f"mace_teacher_{i:02d}.model" for i in range(1, teacher_num + 1)]
    missing = [path for path in model_paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        if len(missing) == len(model_paths):
            raise RuntimeError(
                f"No MACE teacher models found in {teacher_dir}. "
                "Expected files like teacher_01/mace_teacher_01.model."
            )
        raise RuntimeError(
            f"MACE teacher committee appears incomplete in {teacher_dir}: "
            f"expected teacher_num={teacher_num}; missing model files: {formatted}"
        )
    return model_paths


def _load_mace_committee(cfg: dict[str, Any]):
    teacher_dir = Path(cfg["teacher_dir"])
    teacher_num = _normalize_teacher_num(cfg.get("teacher_num")) or _metadata_teacher_num(teacher_dir)
    backend = cfg["mace_backend"]
    ensure_committee_backend_compatible(teacher_dir, backend, teacher_num=teacher_num)
    model_paths = _resolve_mace_model_paths(teacher_dir, teacher_num)

    calculators = []

    for model in model_paths:
        calc = build_mace_calculator(
            model,
            backend=backend,
            device=cfg.get("device", "cpu"),
            default_dtype=cfg.get("default_dtype", "float32"),
        )

        calculators.append(calc)

    print("Loaded teacher models:", len(calculators))

    return calculators


def _committee_label(atoms_list, calculators):
    return label_with_committee(atoms_list, calculators)


class MaceTeacherBackend(TeacherBackend):
    backend_name = "mace"
    aliases = ("mace-les", "mace_les")

    def collect_models(self, model_root: str) -> list[TeacherModelArtifact]:
        root = Path(model_root).resolve()
        artifacts: list[TeacherModelArtifact] = []
        for member_dir in sorted(root.glob("teacher_*")):
            if not member_dir.is_dir():
                continue
            model_files = sorted(member_dir.glob("mace_teacher_*.model"))
            if not model_files:
                continue
            member_text = member_dir.name.rsplit("_", 1)[-1]
            try:
                member_index = int(member_text) - 1
            except ValueError:
                member_index = len(artifacts)
            artifacts.append(
                TeacherModelArtifact(
                    backend=self.backend_name,
                    member_index=member_index,
                    model_path=str(model_files[0].resolve()),
                    metadata={"member_dir": str(member_dir.resolve())},
                )
            )
        return artifacts

    def relabel_structures(self, atoms_list: list[Any], request: TeacherRelabelRequest) -> TeacherRelabelResult:
        teacher_dir = Path(request.committee_dir).resolve()
        options = dict(request.backend_options or {})
        backend = resolve_mace_backend(
            options.get("model_family"),
            options.get("mace_backend"),
            infer_mace_backend(teacher_dir, fallback="standard"),
        )
        execution = normalize_backend_request_context(request, role=teacher_backend_role(self.backend_name, "relabel")).execution
        env_report = ensure_backend_environment(backend, probe_mace_environment(execution.context))
        print_backend_banner(env_report, backend)
        core_cfg = {
            "teacher_dir": str(teacher_dir),
            "teacher_num": _normalize_teacher_num(request.teacher_num or options.get("teacher_num")),
            "device": request.device,
            "default_dtype": request.default_dtype,
            "mace_backend": backend,
        }
        if execution.configured:
            return run_mace_relabel_worker(atoms_list, request, execution, core_cfg)
        calculators = _load_mace_committee(core_cfg)
        return _committee_label(atoms_list, calculators)

    def load_committee_finetune_config(self, config_path: str | Path) -> dict[str, Any]:
        return load_mace_committee_finetune_config(config_path)

    def finetune_committee(self, request) -> dict[str, Any]:
        return run_mace_committee_finetune(request)

    def train_foundation(self, request, *, runner=None) -> dict[str, Any]:
        if runner is None:
            runner = request.context.get("foundation_runner")
        return run_mace_foundation_train(request, runner=runner)
