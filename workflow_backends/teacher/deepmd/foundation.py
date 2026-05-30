from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.deepmd_common.training_watchdog import run_dp_train_with_watchdog
from workflow_backends.teacher.base import TeacherFoundationTrainRequest
from workflow_backends.teacher.deepmd.config import load_deepmd_foundation_train_config
from workflow_backends.teacher.deepmd.family import get_deepmd_family_adapter
from workflow_backends.teacher.deepmd.finetune import materialize_deepmd_teacher_dataset
from workflow_core.structures.datasets import merge_xyz_files
from workflow_core.training_data_assets import validate_path_only_xyz_list
from workflow_runtime.environment import teacher_backend_role
from workflow_runtime.logging.runtime_logging import run_logged_command


DeepMDFoundationRunner = Callable[[dict[str, Any]], None]


def _write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def _workspace_root(request: TeacherFoundationTrainRequest, config_path: Path) -> Path:
    if request.context.get("workspace_root"):
        return Path(str(request.context["workspace_root"])).resolve()
    if request.paths is not None and getattr(request.paths, "workspace_root", None) is not None:
        return Path(request.paths.workspace_root).resolve()
    return config_path.parent.resolve()


def _resolve_training_datasets(train_cfg: dict[str, Any], *, workspace_root: Path) -> list[Path]:
    training_data = dict(train_cfg.get("training_data", {}) or {})
    return validate_path_only_xyz_list(
        training_data.get("base_datasets"),
        field_name="teacher_foundation_train.training_data.base_datasets",
        bare_filename_root=workspace_root / "user_assets" / "teacher_foundation_train",
    )


def _render_foundation_input(
    native_config: dict[str, Any],
    *,
    adapter,
    type_map: list[str],
    train_systems: list[str],
    valid_systems: list[str],
    seed: int,
) -> dict[str, Any]:
    return adapter.render_member_input(
        native_config,
        type_map=type_map,
        train_systems=train_systems,
        valid_systems=valid_systems,
        seed=seed,
    )


def _default_deepmd_foundation_runner(run_cfg: dict[str, Any]) -> None:
    work_root = Path(run_cfg["artifact_root"]).resolve()
    input_json = Path(run_cfg["input_json"]).resolve()
    train_log = Path(run_cfg["train_log"]).resolve()
    freeze_log = Path(run_cfg["freeze_log"]).resolve()
    watch_log = Path(run_cfg["watch_log"]).resolve()
    execution = dict(run_cfg["execution"])
    backend_execution = run_cfg["backend_execution"]
    env = backend_execution.merge_env(os.environ)
    train_command = list(run_cfg["train_command"])
    watchdog = dict(execution.get("nan_watchdog", {}) or {})
    if bool(watchdog.get("enabled", True)):
        result = run_dp_train_with_watchdog(
            command=train_command,
            workdir=work_root,
            log_file=train_log,
            watch_log_file=watch_log,
            input_json=input_json,
            attempt=1,
            seed=int(run_cfg["seed"]),
            env=env,
            terminate_timeout_seconds=float(watchdog.get("terminate_timeout_seconds", 10.0)),
            watch_lcurve_fallback=bool(watchdog.get("watch_lcurve_fallback", True)),
            lcurve_poll_interval_seconds=float(watchdog.get("lcurve_poll_interval_seconds", 1.5)),
            quiet=True,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"DeepMD teacher foundation train failed: {result}")
    else:
        run_logged_command(train_command, log_file=train_log, cwd=work_root, env=env, quiet=True)

    run_logged_command(list(run_cfg["freeze_command"]), log_file=freeze_log, cwd=work_root, env=env, quiet=True)


def run_deepmd_foundation_train(
    request: TeacherFoundationTrainRequest,
    *,
    runner: DeepMDFoundationRunner | None = None,
) -> dict[str, Any]:
    config_path = Path(str(request.cfg.get("config", ""))).resolve()
    if not str(config_path):
        raise ValueError("DeepMD teacher foundation train requires cfg.config")
    train_cfg = load_deepmd_foundation_train_config(config_path)
    workspace_root = _workspace_root(request, config_path)
    artifact_root = Path(request.artifact_root).resolve()
    input_root = artifact_root / "input"
    dataset_root = input_root / "deepmd_dataset"
    foundation_root = Path(request.foundation_model_root).resolve()
    foundation_root.mkdir(parents=True, exist_ok=True)
    request.backend_log_dir.mkdir(parents=True, exist_ok=True)

    source_files = _resolve_training_datasets(train_cfg, workspace_root=workspace_root)
    merged_xyz = input_root / "train_merged.xyz"
    merge_xyz_files(source_files, merged_xyz)
    train_systems, valid_systems, split_meta = materialize_deepmd_teacher_dataset(
        merged_xyz,
        dataset_root,
        type_map=list(train_cfg["type_map"]),
        valid_fraction=float(train_cfg["valid_fraction"]),
        split_seed=int(train_cfg["split_seed"]),
    )

    adapter = get_deepmd_family_adapter(str(train_cfg["model_family"]))
    execution = normalize_backend_request_context(request, role=teacher_backend_role("deepmd", "foundation_train")).execution
    dp_executable = execution.required_executable("dp")
    rendered = _render_foundation_input(
        copy.deepcopy(train_cfg["native_config"]),
        adapter=adapter,
        type_map=list(train_cfg["type_map"]),
        train_systems=train_systems,
        valid_systems=valid_systems,
        seed=int(train_cfg["seed"]),
    )
    input_json = _write_json(artifact_root / "input.json", rendered)
    checkpoint_path = artifact_root / "model.ckpt.pt"
    raw_model_path = artifact_root / adapter.freeze_output_model_name()
    foundation_model = foundation_root / adapter.foundation_model_name()
    train_command = adapter.build_train_command(
        adapter.train_command(dp_executable),
        input_json.name,
        init_mode="scratch",
    )
    train_command = execution.shell_command(train_command)
    freeze_command = execution.shell_command(adapter.freeze_command(dp_executable))
    run_cfg = {
        "backend": "deepmd",
        "model_family": adapter.name,
        "artifact_root": str(artifact_root),
        "input_json": str(input_json),
        "seed": int(train_cfg["seed"]),
        "checkpoint_path": str(checkpoint_path),
        "raw_model_path": str(raw_model_path),
        "foundation_model": str(foundation_model),
        "train_command": train_command,
        "freeze_command": freeze_command,
        "train_log": str((request.backend_log_dir / "foundation_train.log").resolve()),
        "freeze_log": str((request.backend_log_dir / "foundation_freeze.log").resolve()),
        "watch_log": str((request.backend_log_dir / "foundation_watch.jsonl").resolve()),
        "execution": copy.deepcopy(train_cfg["execution"]),
        "execution_context": None if execution.context is None else execution.context.to_payload(),
        "execution_diagnostics": execution.redacted_diagnostics(),
        "backend_execution": execution,
    }
    (runner or _default_deepmd_foundation_runner)(run_cfg)

    if not checkpoint_path.exists() or checkpoint_path.stat().st_size <= 0:
        raise RuntimeError(f"DeepMD foundation checkpoint not found after train: {checkpoint_path}")
    if not raw_model_path.exists() or raw_model_path.stat().st_size <= 0:
        raise RuntimeError(f"DeepMD foundation freeze output not found: {raw_model_path}")
    shutil.copyfile(raw_model_path, foundation_model)
    if not foundation_model.exists() or foundation_model.stat().st_size <= 0:
        raise RuntimeError(f"DeepMD foundation model not published: {foundation_model}")

    runtime_payload = {
        "backend": "deepmd",
        "model_family": adapter.name,
        "type_map": list(train_cfg["type_map"]),
        "source_config": str(config_path),
        "source_datasets": [str(path.resolve()) for path in source_files],
        "merged_xyz": str(merged_xyz.resolve()),
        "deepmd_dataset": str(dataset_root.resolve()),
        "split_meta": str(split_meta.resolve()),
        "input_json": str(input_json.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "raw_model": str(raw_model_path.resolve()),
        "foundation_model": str(foundation_model.resolve()),
        "train_command": train_command,
        "freeze_command": freeze_command,
        "execution": copy.deepcopy(train_cfg["execution"]),
        "execution_context": execution.redacted_diagnostics(),
    }
    _write_json(request.runtime_json, runtime_payload)
    _write_json(
        foundation_root / "deepmd_teacher_foundation_metadata.json",
        {
            "backend": "deepmd",
            "model_family": adapter.name,
            "type_map": list(train_cfg["type_map"]),
            "foundation_model": str(foundation_model.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        },
    )
    return {
        "backend": "deepmd",
        "model_family": adapter.name,
        "foundation_model": str(foundation_model.resolve()),
        "foundation_checkpoint": str(checkpoint_path.resolve()),
    }
