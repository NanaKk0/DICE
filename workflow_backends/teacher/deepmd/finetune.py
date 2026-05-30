from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable

from workflow_backends.deepmd_common.data_format import write_deepmd_dataset
from workflow_backends.deepmd_common.training_watchdog import run_dp_train_with_watchdog
from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.teacher.base import TeacherCommitteeFinetuneRequest
from workflow_backends.teacher.deepmd.family import (
    get_deepmd_family_adapter,
)
from workflow_runtime.environment import teacher_backend_role
from workflow_runtime.logging.runtime_logging import run_logged_command


DeepMDTeacherRunner = Callable[[dict[str, Any]], None]


def teacher_member_dir(work_root: str | Path, teacher_index: int) -> Path:
    return Path(work_root).resolve() / f"teacher_{teacher_index:02d}"


def teacher_primary_model_path(work_root: str | Path, teacher_index: int, *, model_family: str) -> Path:
    adapter = get_deepmd_family_adapter(model_family)
    return teacher_member_dir(work_root, teacher_index) / adapter.primary_model_name(teacher_index)


def collect_existing_deepmd_committee_models(work_root: str | Path, teacher_num: int, *, model_family: str) -> list[Path]:
    models = []
    for teacher_index in range(1, int(teacher_num) + 1):
        model_path = teacher_primary_model_path(work_root, teacher_index, model_family=model_family)
        if model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0:
            models.append(model_path.resolve())
    return models


def materialize_deepmd_teacher_dataset(
    merged_xyz: str | Path,
    dataset_root: str | Path,
    *,
    type_map: list[str],
    valid_fraction: float,
    split_seed: int,
) -> tuple[list[str], list[str], Path]:
    dataset_root = Path(dataset_root).resolve()
    train_dir = dataset_root / "train" / "system.000"
    valid_dir = dataset_root / "valid" / "system.000"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    source = Path(merged_xyz).resolve()
    export_mode = "deepmd_npy_raw"
    export_error = None
    nframes = None
    train_indices: list[int] = []
    valid_indices: list[int] = []
    try:
        from ase.io import read

        atoms_list = list(read(str(source), index=":"))
        nframes = len(atoms_list)
        train_atoms, valid_atoms, train_indices, valid_indices = _split_atoms(
            atoms_list,
            valid_fraction=valid_fraction,
            split_seed=split_seed,
        )
        write_deepmd_dataset(train_atoms, train_dir, type_map)
        write_deepmd_dataset(valid_atoms, valid_dir, type_map)
    except Exception as exc:
        export_mode = "placeholder_extxyz_dataset"
        export_error = str(exc)
        train_source = train_dir / "source.extxyz"
        valid_source = valid_dir / "source.extxyz"
        shutil.copyfile(source, train_source)
        shutil.copyfile(source, valid_source)
    split_meta = dataset_root / "split_meta.json"
    split_meta.write_text(
        json.dumps(
            {
                "mode": export_mode,
                "export_error": export_error,
                "source_extxyz": str(source),
                "source_sha256": _file_sha256(source),
                "type_map": list(type_map),
                "valid_fraction": float(valid_fraction),
                "split_seed": int(split_seed),
                "nframes": nframes,
                "train_indices": train_indices,
                "valid_indices": valid_indices,
                "train_systems": [str(train_dir.resolve())],
                "valid_systems": [str(valid_dir.resolve())],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return [str(train_dir.resolve())], [str(valid_dir.resolve())], split_meta


def _split_atoms(atoms_list: list[Any], *, valid_fraction: float, split_seed: int) -> tuple[list[Any], list[Any], list[int], list[int]]:
    nframes = len(atoms_list)
    if nframes <= 0:
        raise ValueError("Cannot materialize DeePMD teacher dataset from empty merged_xyz")
    indices = list(range(nframes))
    rng = random.Random(int(split_seed))
    rng.shuffle(indices)
    valid_count = max(1, int(round(nframes * float(valid_fraction)))) if nframes > 1 else 1
    valid_count = min(valid_count, nframes)
    valid_indices = sorted(indices[:valid_count])
    train_indices = sorted(indices[valid_count:])
    if not train_indices:
        train_indices = list(valid_indices)
    train_atoms = [atoms_list[index] for index in train_indices]
    valid_atoms = [atoms_list[index] for index in valid_indices]
    return train_atoms, valid_atoms, train_indices, valid_indices


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path).resolve(), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def _runtime_member_payload(member_meta: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(member_meta)
    payload.pop("backend_execution", None)
    return payload


def _default_deepmd_teacher_runner(member_cfg: dict[str, Any]) -> None:
    member_dir = Path(member_cfg["member_dir"]).resolve()
    input_json = Path(member_cfg["input_json"]).resolve()
    train_log = Path(member_cfg["train_log"]).resolve()
    freeze_log = Path(member_cfg["freeze_log"]).resolve()
    watch_log = Path(member_cfg["watch_log"]).resolve()
    execution = dict(member_cfg["execution"])
    adapter = get_deepmd_family_adapter(str(member_cfg["model_family"]))
    backend_execution = member_cfg["backend_execution"]
    env = backend_execution.merge_env(os.environ)
    train_command = list(member_cfg["train_command"])

    watchdog = dict(execution.get("nan_watchdog", {}) or {})
    if bool(watchdog.get("enabled", True)):
        result = run_dp_train_with_watchdog(
            command=train_command,
            workdir=member_dir,
            log_file=train_log,
            watch_log_file=watch_log,
            input_json=input_json,
            attempt=1,
            seed=int(member_cfg["seed"]),
            env=env,
            terminate_timeout_seconds=float(watchdog.get("terminate_timeout_seconds", 10.0)),
            watch_lcurve_fallback=bool(watchdog.get("watch_lcurve_fallback", True)),
            lcurve_poll_interval_seconds=float(watchdog.get("lcurve_poll_interval_seconds", 1.5)),
            quiet=True,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"DeepMD teacher train failed: {result}")
    else:
        run_logged_command(train_command, log_file=train_log, cwd=member_dir, env=env, quiet=True)

    run_logged_command(list(member_cfg["freeze_command"]), log_file=freeze_log, cwd=member_dir, env=env, quiet=True)
    raw_model = member_dir / adapter.freeze_output_model_name()
    primary_model = Path(member_cfg["primary_model_path"]).resolve()
    if raw_model.exists() and raw_model.resolve() != primary_model:
        shutil.copyfile(raw_model, primary_model)
    if not primary_model.exists() or primary_model.stat().st_size <= 0:
        raise RuntimeError(f"DeepMD teacher primary model not found after freeze: {primary_model}")


def run_deepmd_committee_finetune(
    request: TeacherCommitteeFinetuneRequest,
    *,
    runner: DeepMDTeacherRunner | None = None,
) -> dict[str, Any]:
    train_cfg = copy.deepcopy(request.train_cfg)
    model_family = str(train_cfg["model_family"])
    adapter = get_deepmd_family_adapter(model_family)
    teacher_num = int(train_cfg["teacher_num"])
    init_mode = str(train_cfg.get("init_mode", "scratch"))
    foundation_model = train_cfg.get("foundation_model")

    if request.reuse_existing_committee:
        existing_models = collect_existing_deepmd_committee_models(
            request.work_root,
            teacher_num,
            model_family=model_family,
        )
        if len(existing_models) == teacher_num:
            return {
                "backend": "deepmd",
                "model_family": model_family,
                "teacher_num": teacher_num,
                "reused_existing_committee": True,
            }

    dataset_root = request.work_root / "input" / "deepmd_dataset"
    train_systems, valid_systems, split_meta = materialize_deepmd_teacher_dataset(
        request.merged_xyz,
        dataset_root,
        type_map=list(train_cfg["type_map"]),
        valid_fraction=float(train_cfg["valid_fraction"]),
        split_seed=int(train_cfg["split_seed"]),
    )
    native_config = copy.deepcopy(train_cfg["native_config"])

    request.backend_log_dir.mkdir(parents=True, exist_ok=True)
    members = []
    runner = runner or _default_deepmd_teacher_runner
    execution = normalize_backend_request_context(request, role=teacher_backend_role("deepmd", "finetune")).execution
    dp_executable = execution.required_executable("dp")

    for teacher_index, seed in enumerate(train_cfg["teacher_seeds"], start=1):
        member_dir = teacher_member_dir(request.work_root, teacher_index)
        member_dir.mkdir(parents=True, exist_ok=True)
        primary_model = teacher_primary_model_path(request.work_root, teacher_index, model_family=model_family)
        rendered = adapter.render_member_input(
            native_config,
            type_map=list(train_cfg["type_map"]),
            train_systems=train_systems,
            valid_systems=valid_systems,
            seed=int(seed),
        )
        input_json = _write_json(member_dir / "input.json", rendered)
        train_command = adapter.build_train_command(
            adapter.train_command(dp_executable),
            input_json.name,
            init_mode=init_mode,
            foundation_model=foundation_model if init_mode == "foundation_finetune" else None,
        )
        train_command = execution.shell_command(train_command)
        freeze_command = execution.shell_command(adapter.freeze_command(dp_executable))
        member_meta = {
            "backend": "deepmd",
            "model_family": model_family,
            "init_mode": init_mode,
            "foundation_model": str(Path(foundation_model).resolve()) if foundation_model else None,
            "teacher_index": teacher_index,
            "seed": int(seed),
            "member_dir": str(member_dir.resolve()),
            "input_json": str(input_json),
            "primary_model_path": str(primary_model.resolve()),
            "train_log": str((request.backend_log_dir / f"teacher_{teacher_index:02d}_train.log").resolve()),
            "freeze_log": str((request.backend_log_dir / f"teacher_{teacher_index:02d}_freeze.log").resolve()),
            "watch_log": str((request.backend_log_dir / f"teacher_{teacher_index:02d}_watch.jsonl").resolve()),
            "train_command": train_command,
            "freeze_command": freeze_command,
            "execution": copy.deepcopy(train_cfg["execution"]),
            "execution_context": None if execution.context is None else execution.context.to_payload(),
            "execution_diagnostics": execution.redacted_diagnostics(),
            "backend_execution": execution,
        }
        metadata_payload = copy.deepcopy(member_meta)
        metadata_payload.pop("backend_execution", None)
        _write_json(member_dir / "deepmd_teacher_metadata.json", metadata_payload)
        runner(member_meta)
        if not primary_model.exists() or primary_model.stat().st_size <= 0:
            raise RuntimeError(f"DeepMD teacher runner did not produce primary model: {primary_model}")
        members.append(member_meta)

    runtime_payload = {
        "backend": "deepmd",
        "model_family": model_family,
        "init_mode": init_mode,
        "foundation_model": str(Path(foundation_model).resolve()) if foundation_model else None,
        "teacher_num": teacher_num,
        "type_map": list(train_cfg["type_map"]),
        "valid_fraction": float(train_cfg["valid_fraction"]),
        "split_seed": int(train_cfg["split_seed"]),
        "user_config": {
            "workflow": copy.deepcopy(train_cfg.get("workflow", {})),
            "deepmd": copy.deepcopy(train_cfg.get("backend_native", {})),
        },
        "deepmd_dataset": str(dataset_root.resolve()),
        "split_meta": str(split_meta.resolve()),
        "members": [_runtime_member_payload(member) for member in members],
        "execution": copy.deepcopy(train_cfg["execution"]),
        "execution_context": execution.redacted_diagnostics(),
    }
    _write_json(request.runtime_json, runtime_payload)
    _write_json(
        request.work_root / "deepmd_teacher_metadata.json",
        {
            "backend": "deepmd",
            "model_family": model_family,
            "teacher_num": teacher_num,
            "type_map": list(train_cfg["type_map"]),
        },
    )

    return {
        "backend": "deepmd",
        "model_family": model_family,
        "teacher_num": teacher_num,
        "reused_existing_committee": False,
    }
