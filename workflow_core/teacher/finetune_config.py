from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json


@dataclass(frozen=True, slots=True)
class TeacherFinetuneWorkflowConfig:
    model_family: str
    teacher_num: int
    teacher_seeds: list[int]
    training_data: dict[str, Any]
    init_mode: str | None = None
    foundation_model: str | None = None
    type_map: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TeacherFinetuneConfig:
    backend_name: str
    workflow: TeacherFinetuneWorkflowConfig
    backend_native: dict[str, Any]
    source_path: Path

    def as_snapshot(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "workflow": asdict(self.workflow),
            "backend_native": dict(self.backend_native),
            "source_path": str(self.source_path),
        }


KNOWN_TEACHER_NATIVE_BLOCKS = {"mace", "deepmd"}


def load_teacher_finetune_config_boundary(config_path: str | Path, *, backend_name: str) -> TeacherFinetuneConfig:
    resolved = Path(config_path).resolve()
    payload = read_json(resolved)
    return normalize_teacher_finetune_config_boundary(payload, backend_name=backend_name, source_path=resolved)


def normalize_teacher_finetune_config_boundary(
    payload: dict[str, Any],
    *,
    backend_name: str,
    source_path: str | Path,
) -> TeacherFinetuneConfig:
    source = Path(source_path).resolve()
    backend = str(backend_name or "").strip().lower()
    if not backend:
        raise ValueError("teacher finetune config requires a canonical teacher backend name")
    if not isinstance(payload, dict):
        raise ValueError(f"teacher finetune config must be an object: {source}")

    if "workflow" not in payload:
        if backend == "mace":
            raise ValueError(
                f"MACE teacher finetune flat config is retired; use 'workflow' and 'mace' blocks: {source}"
            )
        raise ValueError(f"teacher finetune config requires a 'workflow' block: {source}")
    workflow_payload = payload.get("workflow")
    if not isinstance(workflow_payload, dict):
        raise ValueError(f"teacher finetune config workflow block must be an object: {source}")

    if backend not in payload:
        raise ValueError(f"teacher_backend={backend!r} requires a '{backend}' block in teacher finetune config: {source}")
    native_payload = payload.get(backend)
    if not isinstance(native_payload, dict):
        raise ValueError(f"teacher finetune config '{backend}' block must be an object: {source}")

    mismatched = sorted((set(payload) & KNOWN_TEACHER_NATIVE_BLOCKS) - {backend})
    if mismatched:
        raise ValueError(
            f"teacher finetune config for backend {backend!r} contains mismatched native block(s) {mismatched}: {source}"
        )

    model_family = str(workflow_payload.get("model_family", "")).strip().lower()
    if not model_family:
        raise ValueError(f"teacher finetune workflow.model_family is required: {source}")
    teacher_num = int(workflow_payload.get("teacher_num", 0) or 0)
    if teacher_num <= 0:
        raise ValueError(f"teacher finetune workflow.teacher_num must be positive: {source}")
    raw_seeds = workflow_payload.get("teacher_seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError(
            f"teacher finetune workflow.teacher_seeds must be an explicit list with exactly "
            f"teacher_num={teacher_num} entries: {source}"
        )
    teacher_seeds = [int(item) for item in raw_seeds]
    if len(teacher_seeds) != teacher_num:
        raise ValueError(
            f"Invalid {backend.upper()} finetune config {source}: teacher_seeds has {len(teacher_seeds)} entries, "
            f"but teacher_num={teacher_num}"
        )

    training_data = workflow_payload.get("training_data")
    if not isinstance(training_data, dict):
        raise ValueError(f"teacher finetune workflow.training_data must be an object: {source}")
    base_datasets = training_data.get("base_datasets")
    if not isinstance(base_datasets, list) or not base_datasets:
        raise ValueError(f"teacher finetune workflow.training_data.base_datasets must be a non-empty list: {source}")

    type_map = [str(item) for item in workflow_payload.get("type_map", []) or []]
    init_mode = workflow_payload.get("init_mode")
    if init_mode is not None:
        init_mode = str(init_mode).strip().lower()
        if not init_mode:
            raise ValueError(f"teacher finetune workflow.init_mode must be a non-empty string: {source}")
    foundation_model = workflow_payload.get("foundation_model")
    if foundation_model is not None:
        foundation_model = str(foundation_model).strip()
        if not foundation_model:
            raise ValueError(f"teacher finetune workflow.foundation_model must be a non-empty path string: {source}")

    workflow = TeacherFinetuneWorkflowConfig(
        model_family=model_family,
        teacher_num=teacher_num,
        teacher_seeds=teacher_seeds,
        training_data=dict(training_data),
        init_mode=init_mode,
        foundation_model=foundation_model,
        type_map=type_map,
    )
    return TeacherFinetuneConfig(
        backend_name=backend,
        workflow=workflow,
        backend_native=dict(native_payload),
        source_path=source,
    )


def flatten_teacher_finetune_config(config: TeacherFinetuneConfig) -> dict[str, Any]:
    workflow = config.workflow
    payload = dict(config.backend_native)
    payload.update(
        {
            "backend_name": config.backend_name,
            "workflow": {
                "model_family": workflow.model_family,
                "teacher_num": workflow.teacher_num,
                "teacher_seeds": list(workflow.teacher_seeds),
                "training_data": dict(workflow.training_data),
                "type_map": list(workflow.type_map),
            },
            "backend_native": dict(config.backend_native),
            "source_path": str(config.source_path),
            "model_family": workflow.model_family,
            "teacher_num": workflow.teacher_num,
            "teacher_seeds": list(workflow.teacher_seeds),
            "training_data": dict(workflow.training_data),
        }
    )
    if workflow.init_mode is not None:
        payload["workflow"]["init_mode"] = workflow.init_mode
        payload["init_mode"] = workflow.init_mode
    if workflow.foundation_model is not None:
        payload["workflow"]["foundation_model"] = workflow.foundation_model
        payload["foundation_model"] = workflow.foundation_model
    if workflow.type_map:
        payload["type_map"] = list(workflow.type_map)
    return payload
