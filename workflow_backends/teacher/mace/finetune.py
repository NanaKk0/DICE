from __future__ import annotations

import json
import os
from pathlib import Path

from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.teacher.base import TeacherCommitteeFinetuneRequest
from workflow_backends.teacher.mace.backend import (
    apply_backend_to_train_config,
    normalize_mace_backend,
    print_backend_banner,
    resolve_mace_backend,
)
from workflow_backends.teacher.mace.env import ensure_backend_environment, probe_mace_environment
from workflow_backends.teacher.mace.model_check import (
    ensure_committee_backend_compatible,
    ensure_foundation_backend_compatible,
    write_mace_model_metadata,
)
from workflow_backends.teacher.mace.train import run_mace_training, write_mace_rendered_yaml
from workflow_core.teacher.finetune_config import flatten_teacher_finetune_config, load_teacher_finetune_config_boundary
from workflow_runtime.environment import teacher_backend_role


def _teacher_work_dir(cfg, teacher_index: int) -> Path:
    return Path(cfg["work_root"]) / f"teacher_{teacher_index:02d}"


def _teacher_model_path(cfg, teacher_index: int) -> Path:
    return _teacher_work_dir(cfg, teacher_index) / f"mace_teacher_{teacher_index:02d}.model"


def _teacher_seed(cfg, teacher_index: int) -> int:
    teacher_seeds = cfg.get("teacher_seeds")
    if not isinstance(teacher_seeds, list):
        raise ValueError("MACE committee training requires explicit teacher_seeds in the training config")
    seed_index = teacher_index - 1
    if seed_index < 0 or seed_index >= len(teacher_seeds):
        raise ValueError(
            f"MACE committee training requires exactly one teacher seed per member; "
            f"missing seed for teacher_{teacher_index:02d}"
        )
    return int(teacher_seeds[seed_index])


def teacher_training_is_complete(cfg, teacher_index: int) -> bool:
    model_path = _teacher_model_path(cfg, teacher_index)
    return model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0


def build_teacher_command(cfg, teacher_index):
    rendered_yaml = cfg.get("_rendered_yaml_by_teacher", {}).get(int(teacher_index))
    if rendered_yaml is None:
        rendered_yaml = _teacher_work_dir(cfg, teacher_index) / "rendered_mace.yaml"
    command_cfg = {
        "rendered_yaml": str(Path(rendered_yaml).resolve()),
        "execution": cfg.get("execution"),
    }
    if cfg.get("mace_run_train_executable"):
        command_cfg["mace_run_train_executable"] = cfg["mace_run_train_executable"]
    from workflow_backends.teacher.mace.train import build_mace_command

    return build_mace_command(command_cfg)


def render_teacher_train_config(cfg, teacher_index: int) -> dict:
    teacher_name = f"mace_teacher_{teacher_index:02d}"
    teacher_seed = _teacher_seed(cfg, teacher_index)
    work_dir = _teacher_work_dir(cfg, teacher_index)
    work_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = dict(cfg)
    train_cfg.update(
        {
            "name": teacher_name,
            "foundation_model": cfg["foundation_model"],
            "work_dir": str(work_dir),
            "train_file": cfg["train_file"],
            "seed": teacher_seed,
        }
    )
    if train_cfg.get("freeze_interactions"):
        train_cfg["freeze"] = train_cfg["freeze_interactions"]
    if "ema" not in train_cfg:
        train_cfg["ema"] = True
    return train_cfg


def run_committee_training(cfg, env=None, log_dir=None, quiet=None):
    teacher_num = cfg["teacher_num"]
    skip_completed_teachers = bool(cfg.get("skip_completed_teachers", True))

    if env is None:
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for i in range(1, teacher_num + 1):
        model_path = _teacher_model_path(cfg, i)
        if skip_completed_teachers and teacher_training_is_complete(cfg, i):
            print("=" * 90)
            print(f"Skipping completed MACE committee teacher {i:02d}")
            print(str(model_path))
            print("=" * 90)
            continue

        teacher_cfg = render_teacher_train_config(cfg, i)
        rendered_yaml = write_mace_rendered_yaml(teacher_cfg, _teacher_work_dir(cfg, i) / "rendered_mace.yaml")
        train_request = {
            "rendered_yaml": str(rendered_yaml.resolve()),
            "execution": cfg.get("execution"),
        }
        if cfg.get("mace_run_train_executable"):
            train_request["mace_run_train_executable"] = cfg["mace_run_train_executable"]

        print("=" * 90)
        print(f"Running MACE committee teacher {i:02d}")
        print(f"rendered YAML: {rendered_yaml}")
        print("=" * 90)

        if log_dir is None:
            run_mace_training(train_request, env=env)
            continue

        log_path = Path(log_dir) / f"teacher_{i:02d}.log"
        run_mace_training(train_request, env=env, log_file=log_path, quiet=quiet)


def collect_existing_mace_committee_models(work_root: Path, teacher_num: int) -> list[Path]:
    models = []
    for teacher_index in range(1, teacher_num + 1):
        model_path = work_root / f"teacher_{teacher_index:02d}" / f"mace_teacher_{teacher_index:02d}.model"
        if model_path.exists():
            models.append(model_path.resolve())
    return models


def find_mace_foundation_model(paths, teacher_round: int) -> Path:
    user_foundation_dir = paths.user_foundation_models
    if user_foundation_dir.exists():
        user_candidates = sorted(user_foundation_dir.rglob("*stagetwo*.model"))
        if user_candidates:
            return user_candidates[0].resolve()

        user_candidates = sorted(user_foundation_dir.rglob("*.model"))
        if user_candidates:
            return user_candidates[0].resolve()

    foundation_root = paths.teacher_foundation_train_artifacts_dir(teacher_round)
    for search_root in (foundation_root / "checkpoints", foundation_root):
        ckpt = sorted(search_root.glob("*stagetwo*.model"))
        if ckpt:
            return ckpt[0].resolve()
        ckpt = sorted(search_root.glob("*.model"))
        if ckpt:
            return ckpt[0].resolve()

    raise RuntimeError(
        "No foundation model found in "
        f"{user_foundation_dir} or {foundation_root}"
    )


def _resolve_configured_foundation_model(raw_value: str | Path, paths) -> Path:
    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        if raw_path.name != str(raw_path):
            raise ValueError("foundation_model must be an absolute path or a bare filename")
        resolved = (Path(paths.user_foundation_models) / raw_path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"foundation_model not found: {resolved}")
    return resolved


def validate_mace_finetune_config(train_cfg, config_path: Path) -> None:
    if bool(train_cfg.get("swa")):
        max_num_epochs = int(train_cfg["max_num_epochs"])
        start_swa = train_cfg.get("start_swa")
        if start_swa is not None and int(start_swa) >= max_num_epochs:
            raise ValueError(
                f"Invalid MACE finetune config {config_path}: start_swa={start_swa} must be smaller than "
                f"max_num_epochs={max_num_epochs}"
            )


def validate_teacher_seed_config(train_cfg: dict, *, config_path: str | Path) -> dict:
    teacher_num = int(train_cfg.get("teacher_num", 0) or 0)
    teacher_seeds_raw = train_cfg.get("teacher_seeds")
    if not isinstance(teacher_seeds_raw, list):
        raise ValueError(
            f"Invalid MACE finetune config {config_path}: teacher_seeds must be an explicit list with "
            f"exactly teacher_num={teacher_num} entries"
        )
    teacher_seeds = [int(item) for item in teacher_seeds_raw]
    if len(teacher_seeds) != teacher_num:
        raise ValueError(
            f"Invalid MACE finetune config {config_path}: teacher_seeds has {len(teacher_seeds)} entries, "
            f"but teacher_num={teacher_num}"
        )
    train_cfg["teacher_seeds"] = teacher_seeds
    return train_cfg


MACE_WORKFLOW_OWNED_NATIVE_FIELDS = {
    "name",
    "work_dir",
    "train_file",
    "seed",
    "foundation_model",
    "model",
    "mace_backend",
}


def validate_mace_native_ownership(train_cfg: dict, *, config_path: str | Path) -> None:
    native = dict(train_cfg.get("backend_native", {}) or {})
    misplaced = sorted(set(native) & MACE_WORKFLOW_OWNED_NATIVE_FIELDS)
    if misplaced:
        raise ValueError(
            f"Invalid MACE finetune config {config_path}: MACE native block contains workflow-owned/generated "
            f"fields {misplaced}; move them under 'workflow' or let the workflow generate them"
        )


def load_mace_committee_finetune_config(config_path: str | Path) -> dict:
    resolved = Path(config_path).resolve()
    bounded = load_teacher_finetune_config_boundary(resolved, backend_name="mace")
    train_cfg = flatten_teacher_finetune_config(bounded)
    train_cfg["mace_backend"] = str(train_cfg["model_family"]).strip().lower()
    validate_mace_native_ownership(train_cfg, config_path=resolved)
    train_cfg = validate_teacher_seed_config(train_cfg, config_path=resolved)
    validate_mace_finetune_config(train_cfg, resolved)
    return train_cfg


def run_mace_committee_finetune(request: TeacherCommitteeFinetuneRequest) -> dict:
    train_cfg = dict(request.train_cfg)
    backend = resolve_mace_backend(
        request.context.get("mace_backend"),
        train_cfg.get("mace_backend"),
    )
    execution = normalize_backend_request_context(request, role=teacher_backend_role("mace", "finetune")).execution
    env_report = ensure_backend_environment(backend, probe_mace_environment(execution.context))
    print_backend_banner(env_report, backend)

    train_cfg["train_file"] = str(request.merged_xyz)
    train_cfg["work_root"] = str(request.work_root)

    foundation_model = (
        _resolve_configured_foundation_model(train_cfg["foundation_model"], request.paths)
        if train_cfg.get("foundation_model") is not None
        else find_mace_foundation_model(request.paths, request.teacher_round)
    )
    ensure_foundation_backend_compatible(foundation_model, backend)
    train_cfg["foundation_model"] = str(foundation_model)
    train_cfg = apply_backend_to_train_config(train_cfg, backend)
    train_cfg["execution"] = execution
    rendered_yaml_by_teacher: dict[int, str] = {}
    for teacher_index in range(1, int(train_cfg["teacher_num"]) + 1):
        teacher_cfg = render_teacher_train_config(train_cfg, teacher_index)
        rendered_yaml = write_mace_rendered_yaml(
            teacher_cfg,
            _teacher_work_dir(train_cfg, teacher_index) / "rendered_mace.yaml",
        )
        rendered_yaml_by_teacher[teacher_index] = str(rendered_yaml.resolve())
    train_cfg["_rendered_yaml_by_teacher"] = dict(rendered_yaml_by_teacher)
    training_env = execution.merge_env(
        os.environ,
        {
            "MKL_THREADING_LAYER": "GNU",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

    runtime_execution_args = dict(train_cfg)
    runtime_execution_args.pop("execution", None)
    runtime_execution_args.pop("_rendered_yaml_by_teacher", None)
    request.runtime_json.parent.mkdir(parents=True, exist_ok=True)
    with open(request.runtime_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "backend": "mace",
                "model_family": backend,
                "teacher_num": int(train_cfg["teacher_num"]),
                "user_config": {
                    "workflow": dict(train_cfg.get("workflow", {}) or {}),
                    "mace": dict(train_cfg.get("backend_native", {}) or {}),
                },
                "execution_args": runtime_execution_args,
                "rendered_yaml_by_teacher": dict(rendered_yaml_by_teacher),
                "execution_context": execution.redacted_diagnostics(),
            },
            handle,
            indent=2,
        )

    teacher_num = int(train_cfg["teacher_num"])
    if request.reuse_existing_committee:
        existing_models = collect_existing_mace_committee_models(request.work_root, teacher_num)
        if len(existing_models) == teacher_num:
            ensure_committee_backend_compatible(request.work_root, backend, teacher_num=teacher_num)
            print("\nReusing existing MACE committee:")
            for model_path in existing_models:
                print(" ", model_path)
            return {
                "backend": "mace",
                "mace_backend": backend,
                "teacher_num": teacher_num,
                "reused_existing_committee": True,
            }

    request.backend_log_dir.mkdir(parents=True, exist_ok=True)
    run_committee_training(train_cfg, env=training_env, log_dir=request.backend_log_dir)
    write_mace_model_metadata(
        request.work_root,
        backend=backend,
        model_role="committee_root",
        env_report=env_report,
        extra={"teacher_num": teacher_num, "foundation_model": str(foundation_model)},
    )
    for teacher_index in range(1, teacher_num + 1):
        write_mace_model_metadata(
            request.work_root / f"teacher_{teacher_index:02d}",
            backend=backend,
            model_role="committee_teacher",
            env_report=env_report,
            extra={"teacher_index": teacher_index, "foundation_model": str(foundation_model)},
        )
    return {
        "backend": "mace",
        "mace_backend": backend,
        "teacher_num": teacher_num,
        "reused_existing_committee": False,
    }
