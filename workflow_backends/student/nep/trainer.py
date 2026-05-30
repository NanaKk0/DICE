from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from workflow_backends.student.nep.data_writer import split_canonical_extxyz_to_train_test
from workflow_backends.student.nep.env_check import resolve_nep_command
from workflow_backends.student.nep.input_writer import write_nep_input
from workflow_backends.student.nep.model_registry import write_nep_committee_manifest
from workflow_backends.student.config import _normalize_nep_config
from workflow_core.common.io import read_json, write_json

_LOSS_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_GENERATION_MODEL_RE = re.compile(r"nep_gen(\d+)\.txt$")
_TIMESTAMP_MODEL_RE = re.compile(r"generation(\d+)\.txt$")


def _safe_resolve_existing(path: str | Path, *, label: str) -> Path:
    raw_path = Path(path)
    try:
        resolved = raw_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} not found: {raw_path}") from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{label} cannot be resolved, likely due to a symlink loop: {raw_path}") from exc
    return resolved


def _copy_regular_file(source: str | Path, target: str | Path, *, label: str) -> Path:
    source_path = _safe_resolve_existing(source, label=label)
    if not source_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {source_path}")
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        try:
            if target_path.exists() and source_path.samefile(target_path):
                raise RuntimeError(f"Refusing to copy {label} onto itself: {source_path}")
        except OSError:
            pass
        target_path.unlink()
    shutil.copy2(source_path, target_path)
    return _safe_resolve_existing(target_path, label=label)


def _native_recipe_source(nep_cfg: dict[str, Any]) -> Path:
    recipe_path = nep_cfg.get("native_recipe_path") or nep_cfg.get("recipe_path") or nep_cfg.get("nep_in")
    if not recipe_path:
        raise ValueError("NEP training requires native_recipe_path pointing at a user-maintained nep.in")
    return _safe_resolve_existing(recipe_path, label="NEP native recipe")


def _stage_native_nep_recipe(
    nep_cfg: dict[str, Any],
    member_dir: Path,
    *,
    inject_fine_tune: bool = True,
) -> Path:
    target = member_dir / "nep.in"
    staged = _copy_regular_file(_native_recipe_source(nep_cfg), target, label="NEP native recipe")
    if not inject_fine_tune:
        return staged
    fine_tune_model = nep_cfg.get("fine_tune_model_path")
    fine_tune_restart = nep_cfg.get("fine_tune_restart_path")
    if fine_tune_model or fine_tune_restart:
        if not fine_tune_model or not fine_tune_restart:
            raise ValueError("NEP fine_tune requires both fine_tune_model_path and fine_tune_restart_path")
        _inject_fine_tune_line(staged, fine_tune_model=fine_tune_model, fine_tune_restart=fine_tune_restart)
    return _safe_resolve_existing(staged, label="NEP staged native recipe")


def _inject_fine_tune_line(path: Path, *, fine_tune_model: str | Path, fine_tune_restart: str | Path) -> None:
    member_dir = path.parent
    model_path = _copy_regular_file(
        fine_tune_model,
        member_dir / "foundation_nep.txt",
        label="NEP foundation model",
    )
    restart_path = _copy_regular_file(
        fine_tune_restart,
        member_dir / "foundation_nep.restart",
        label="NEP foundation restart",
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    filtered = [line for line in lines if not line.strip().lower().startswith("fine_tune")]
    filtered.append(f"fine_tune    {model_path} {restart_path}")
    path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def _path_exists_without_loop(path: str | Path) -> bool:
    try:
        _safe_resolve_existing(path, label="path")
    except (FileNotFoundError, RuntimeError):
        return False
    return True


def _default_work_root(paths, outer_round: int, dp_round: int | None) -> Path:
    raise RuntimeError("NEP target-mode training requires explicit work_root")


def _parse_loss_metrics(loss_path: Path) -> tuple[dict[str, Any], list[str]]:
    if not loss_path.exists():
        return {
            "final_loss": None,
            "force_rmse": None,
            "energy_rmse": None,
        }, [f"loss.out missing: {loss_path.name}"]
    lines = [line.strip() for line in loss_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return {
            "final_loss": None,
            "force_rmse": None,
            "energy_rmse": None,
        }, [f"loss.out empty: {loss_path.name}"]
    last_line = lines[-1]
    numbers = [float(item) for item in _LOSS_NUMBER_RE.findall(last_line)]
    metrics = {
        "final_loss": numbers[-1] if numbers else None,
        "force_rmse": numbers[-2] if len(numbers) >= 2 else None,
        "energy_rmse": numbers[-3] if len(numbers) >= 3 else None,
    }
    warnings: list[str] = []
    if not numbers:
        warnings.append(f"unable to parse numeric metrics from loss.out: {loss_path.name}")
    return metrics, warnings


def _resolve_trained_model_path(member_dir: Path) -> Path | None:
    direct_path = member_dir / "nep.txt"
    if direct_path.exists() or direct_path.is_symlink():
        return _safe_resolve_existing(direct_path, label="NEP trained model")

    candidates = [
        path
        for path in member_dir.glob("nep*.txt")
        if path.name != "nep.txt" and not path.name.startswith("nep_stage_")
    ]
    if not candidates:
        return None

    def _sort_key(path: Path) -> tuple[int, int, str]:
        generation = -1
        match = _GENERATION_MODEL_RE.fullmatch(path.name)
        if match:
            generation = int(match.group(1))
        else:
            match = _TIMESTAMP_MODEL_RE.search(path.name)
            if match:
                generation = int(match.group(1))
        mtime_ns = _safe_resolve_existing(path, label="NEP candidate model").stat().st_mtime_ns
        return generation, mtime_ns, path.name

    return _safe_resolve_existing(max(candidates, key=_sort_key), label="NEP trained model")


def _resolve_restart_path(member_dir: Path) -> Path | None:
    direct_path = member_dir / "nep.restart"
    if direct_path.exists() or direct_path.is_symlink():
        return _safe_resolve_existing(direct_path, label="NEP restart")
    candidates = [
        path
        for path in member_dir.glob("nep*.restart")
        if path.name != "nep.restart" and not path.name.startswith("nep_stage_")
    ]
    if not candidates:
        return None

    def _sort_key(path: Path) -> tuple[int, int, str]:
        generation = -1
        match = _GENERATION_MODEL_RE.fullmatch(path.with_suffix(".txt").name)
        if match:
            generation = int(match.group(1))
        else:
            match = _TIMESTAMP_MODEL_RE.search(path.name)
            if match:
                generation = int(match.group(1))
        return generation, _safe_resolve_existing(path, label="NEP candidate restart").stat().st_mtime_ns, path.name

    return _safe_resolve_existing(max(candidates, key=_sort_key), label="NEP restart")


def _summary_path_from_previous_root(previous_model_root: str | Path) -> Path:
    root = Path(previous_model_root).resolve()
    return root if root.is_file() else root / "summary.json"


def _restart_path_from_model_entry(model_path: str | Path) -> Path | None:
    raw_model_path = Path(str(model_path))
    member_dir = raw_model_path.parent
    restart_path = member_dir / "nep.restart"
    if restart_path.exists() or restart_path.is_symlink():
        return _safe_resolve_existing(restart_path, label="NEP restart continuation source")
    snapshot_candidates = sorted(member_dir.glob("nep_stage_*.restart"))
    for snapshot in reversed(snapshot_candidates):
        if snapshot.exists() or snapshot.is_symlink():
            return _safe_resolve_existing(snapshot, label="NEP restart continuation snapshot")
    return None


def _resolve_previous_restart_sources(
    previous_model_root: str | Path,
    *,
    committee_size: int,
) -> list[dict[str, Any]]:
    summary_path = _summary_path_from_previous_root(previous_model_root)
    sources_by_index: dict[int, dict[str, Any]] = {}
    if summary_path.exists():
        payload = read_json(summary_path)
        for fallback_index, member in enumerate(payload.get("members", []) or []):
            status = str(member.get("status", "success")).strip().lower()
            if status not in {"success", "warning", "prepared"}:
                continue
            member_index = int(member.get("member_index", fallback_index))
            model_path = member.get("model_path")
            member_dir = Path(str(member.get("member_dir", ""))).resolve() if member.get("member_dir") else None
            restart_path = member.get("restart_path")
            if restart_path is None and member_dir is not None:
                resolved_restart = _resolve_restart_path(member_dir)
                restart_path = None if resolved_restart is None else str(resolved_restart)
            if restart_path:
                restart_path = _safe_resolve_existing(restart_path, label="NEP restart continuation source")
                sources_by_index[member_index] = {
                    "member_index": member_index,
                    "source_restart_path": str(restart_path),
                }
        for member_index, model_path in enumerate(payload.get("models", []) or []):
            restart_path = _restart_path_from_model_entry(model_path)
            if restart_path is not None:
                sources_by_index.setdefault(
                    member_index,
                    {
                        "member_index": member_index,
                        "source_restart_path": str(restart_path),
                    },
                )

    root = Path(previous_model_root).resolve()
    for member_index in range(int(committee_size)):
        if member_index in sources_by_index:
            continue
        member_dir = root / f"student_{member_index + 1:02d}"
        restart_path = _resolve_restart_path(member_dir)
        if restart_path is not None:
            sources_by_index[member_index] = {
                "member_index": member_index,
                "source_restart_path": str(restart_path),
            }

    sources: list[dict[str, Any]] = []
    for member_index in range(int(committee_size)):
        source = sources_by_index.get(member_index)
        if source is None:
            raise FileNotFoundError(
                f"NEP restart continuation source missing for committee member {member_index}: "
                f"previous root={Path(previous_model_root).resolve()}"
            )
        restart_raw = source.get("source_restart_path")
        restart_path = _safe_resolve_existing(restart_raw, label="NEP restart continuation source") if restart_raw else None
        if restart_path is None:
            raise FileNotFoundError(
                f"NEP restart continuation source restart not specified for member {member_index}"
            )
        if not restart_path.exists():
            raise FileNotFoundError(
                f"NEP restart continuation source restart not found for member {member_index}: {restart_path}. "
                "Cross-round NEP restart continuation requires previous nep.restart."
            )
        sources.append(
            {
                "member_index": member_index,
                "source_restart_path": str(restart_path),
            }
        )
    return sources


def _stage_restart_source(member_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    staged_restart = member_dir / "nep.restart"
    restart_source = _safe_resolve_existing(source["source_restart_path"], label="NEP restart continuation source")
    staged_restart_resolved = _copy_regular_file(restart_source, staged_restart, label="staged NEP restart")
    return {
        **dict(source),
        "source_restart_path": str(restart_source),
        "staged_restart_path": str(staged_restart_resolved),
    }


def _normalize_nep_training_stage(training_stage: str | None, *, continue_training: bool = False) -> str:
    stage = str(training_stage or "bootstrap").strip().lower()
    if bool(continue_training) or stage in {"finetune", "fine_tune", "restart_continue", "previous_restart_continue"}:
        return "previous_restart_continue"
    return stage


def _nep_run_stage_configs(nep_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if nep_cfg.get("native_recipe_path") or nep_cfg.get("recipe_path") or nep_cfg.get("nep_in"):
        return [("main", dict(nep_cfg))]
    has_stage_2 = isinstance(nep_cfg.get("loss_stage_2"), dict) or isinstance(nep_cfg.get("training_stage_2"), dict)
    if not has_stage_2:
        return [("main", _normalize_nep_config(nep_cfg, run_stage="stage_1"))]
    return [
        ("stage_1", _normalize_nep_config(nep_cfg, run_stage="stage_1")),
        ("stage_2", _normalize_nep_config(nep_cfg, run_stage="stage_2")),
    ]


def _member_stage_status_path(member_dir: Path) -> Path:
    return member_dir / "stage_status.json"


def _load_member_stage_status(member_dir: Path) -> dict[str, Any]:
    path = _member_stage_status_path(member_dir)
    if not path.exists():
        return {"stages": {}}
    try:
        payload = read_json(path)
    except Exception:
        return {"stages": {}}
    if not isinstance(payload, dict):
        return {"stages": {}}
    payload.setdefault("stages", {})
    return payload


def _write_member_stage_status(member_dir: Path, payload: dict[str, Any]) -> None:
    payload.setdefault("stages", {})
    write_json(_member_stage_status_path(member_dir), payload)


def _stage_status_entry(member_dir: Path, stage_name: str) -> dict[str, Any]:
    payload = _load_member_stage_status(member_dir)
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    entry = stages.get(stage_name, {})
    return dict(entry or {}) if isinstance(entry, dict) else {}


def _stage_completed(member_dir: Path, stage_name: str) -> bool:
    entry = _stage_status_entry(member_dir, stage_name)
    if entry.get("status") != "success":
        return False
    model_path = entry.get("model_path") or entry.get("snapshot_model_path")
    restart_path = entry.get("restart_path") or entry.get("snapshot_restart_path")
    return bool(model_path and _path_exists_without_loop(model_path) and restart_path and _path_exists_without_loop(restart_path))


def _set_stage_status(member_dir: Path, stage_name: str, updates: dict[str, Any]) -> None:
    payload = _load_member_stage_status(member_dir)
    stages = payload.setdefault("stages", {})
    current = dict(stages.get(stage_name, {}) or {})
    current.update(dict(updates))
    stages[stage_name] = current
    _write_member_stage_status(member_dir, payload)


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        pass


def _clear_nep_runtime_outputs(member_dir: Path) -> None:
    protected = {"nep.in", "train.xyz", "test.xyz", "stage_status.json"}
    for path in member_dir.iterdir() if member_dir.exists() else []:
        if path.name in protected or path.name.startswith("nep_stage_"):
            continue
        if path.is_dir():
            continue
        if (
            path.name == "loss.out"
            or path.name == "nep.txt"
            or path.name == "nep.restart"
            or path.name.startswith("nep_gen")
            or path.name.startswith("generation")
        ):
            _unlink_if_exists(path)


def _stage_snapshot_paths(member_dir: Path, stage_name: str) -> tuple[Path, Path]:
    safe_stage_name = stage_name.replace("/", "_")
    return member_dir / f"nep_{safe_stage_name}.txt", member_dir / f"nep_{safe_stage_name}.restart"


def _snapshot_stage_outputs(member_dir: Path, stage_name: str) -> dict[str, str]:
    model_path = _resolve_trained_model_path(member_dir)
    restart_path = _resolve_restart_path(member_dir)
    if model_path is None or restart_path is None:
        raise RuntimeError(f"Cannot snapshot NEP {stage_name}: model or restart missing in {member_dir}")
    snapshot_model, snapshot_restart = _stage_snapshot_paths(member_dir, stage_name)
    snapshot_model_resolved = _copy_regular_file(model_path, snapshot_model, label=f"NEP {stage_name} model")
    snapshot_restart_resolved = _copy_regular_file(restart_path, snapshot_restart, label=f"NEP {stage_name} restart")
    return {
        "model_path": str(_safe_resolve_existing(model_path, label=f"NEP {stage_name} model")),
        "restart_path": str(_safe_resolve_existing(restart_path, label=f"NEP {stage_name} restart")),
        "snapshot_model_path": str(snapshot_model_resolved),
        "snapshot_restart_path": str(snapshot_restart_resolved),
    }


def _restore_stage_start(member_dir: Path, stage_name: str, *, initial_restart_source: str | Path | None = None) -> None:
    if stage_name == "stage_2" and _stage_completed(member_dir, "stage_1"):
        stage_1 = _stage_status_entry(member_dir, "stage_1")
        _clear_nep_runtime_outputs(member_dir)
        _copy_regular_file(stage_1["snapshot_restart_path"], member_dir / "nep.restart", label="NEP stage_1 restart snapshot")
        return
    _clear_nep_runtime_outputs(member_dir)
    if initial_restart_source is not None and stage_name in {"main", "stage_1"}:
        source = Path(initial_restart_source)
        if source.exists() or source.is_symlink():
            _copy_regular_file(source, member_dir / "nep.restart", label="NEP initial restart source")


def prepare_nep_committee_training(
    dataset_path: str | Path,
    nep_cfg: dict[str, Any],
    *,
    work_root: str | Path,
    random_seed: int = 42,
    training_stage: str = "bootstrap",
    restart_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    nep_cfg = dict(nep_cfg)
    if not (nep_cfg.get("native_recipe_path") or nep_cfg.get("recipe_path") or nep_cfg.get("nep_in")):
        nep_cfg = _normalize_nep_config(nep_cfg, run_stage="stage_1")
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    committee_size = int(nep_cfg["committee_size"])
    training_stage = _normalize_nep_training_stage(training_stage)
    restart_sources_by_index = {
        int(item["member_index"]): dict(item)
        for item in (restart_sources or [])
    }
    if training_stage == "previous_restart_continue" and len(restart_sources_by_index) < committee_size:
        raise ValueError(
            f"NEP restart continuation requires one source per committee member; "
            f"got {len(restart_sources_by_index)} for committee_size={committee_size}"
        )
    split_seed = int(nep_cfg.get("split_seed", random_seed))

    members: list[dict[str, Any]] = []
    for member_index in range(committee_size):
        member_dir = work_root / f"student_{member_index + 1:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        restart_source = None
        if training_stage == "previous_restart_continue":
            restart_source = _stage_restart_source(member_dir, restart_sources_by_index[member_index])
        member_split_seed = split_seed + member_index
        split_info = split_canonical_extxyz_to_train_test(
            dataset_path,
            member_dir / "train.xyz",
            member_dir / "test.xyz",
            train_fraction=float(nep_cfg.get("train_test_split", 0.95)),
            random_seed=member_split_seed,
        )
        nep_input = _stage_native_nep_recipe(nep_cfg, member_dir)
        members.append(
            {
                "member_index": member_index,
                "split_seed": member_split_seed,
                "member_dir": str(member_dir.resolve()),
                "train_xyz": str((member_dir / "train.xyz").resolve()),
                "test_xyz": str((member_dir / "test.xyz").resolve()),
                "nep_in": str(nep_input.resolve()),
                "model_path": str((member_dir / "nep.txt").absolute()),
                "loss_out": str((member_dir / "loss.out").absolute()),
                "split": split_info,
                "training_stage": training_stage,
                "restart_source": restart_source,
                "status": "prepared",
            }
        )

    manifest = {
        "backend": "nep",
        "training_stage": training_stage,
        "dataset_path": str(Path(dataset_path).resolve()),
        "committee_size": committee_size,
        "work_root": str(work_root.resolve()),
        "members": members,
    }
    write_json(work_root / "prepare_manifest.json", manifest)
    return manifest


def run_nep_committee_training(
    dataset_path: str | Path,
    nep_cfg: dict[str, Any],
    *,
    work_root: str | Path,
    summary_path: str | Path | None = None,
    random_seed: int = 42,
    env: dict[str, str] | None = None,
    previous_model_root: str | Path | None = None,
    continue_training: bool = False,
    training_stage: str = "bootstrap",
    resolved_train_decision: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
    log_prefix: str = "nep_train",
) -> str:
    nep_cfg = dict(nep_cfg)
    if not (nep_cfg.get("native_recipe_path") or nep_cfg.get("recipe_path") or nep_cfg.get("nep_in")):
        nep_cfg = _normalize_nep_config(nep_cfg, run_stage="stage_1")
    committee_size = int(nep_cfg["committee_size"])
    stage_norm = _normalize_nep_training_stage(training_stage, continue_training=continue_training)
    restart_sources = None
    if stage_norm == "previous_restart_continue":
        if previous_model_root is None:
            raise ValueError("NEP restart continuation requires previous_model_root")
        restart_sources = _resolve_previous_restart_sources(
            previous_model_root,
            committee_size=committee_size,
        )
    manifest = prepare_nep_committee_training(
        dataset_path,
        nep_cfg,
        work_root=work_root,
        random_seed=random_seed,
        training_stage=stage_norm,
        restart_sources=restart_sources,
    )
    command = resolve_nep_command(nep_cfg)
    run_stage_configs = _nep_run_stage_configs(nep_cfg)
    members_payload: list[dict[str, Any]] = []
    models: list[str] = []
    restarts: list[str] = []
    metrics: dict[str, Any] = {}
    warnings: list[str] = []

    for member in manifest["members"]:
        member_dir = Path(member["member_dir"]).resolve()
        result = dict(member)
        stage_logs: list[str] = []
        central_stage_logs: list[str] = []
        returncode = 0
        central_log_path = None
        initial_restart_source = None
        if isinstance(member.get("restart_source"), dict):
            initial_restart_source = member["restart_source"].get("source_restart_path")
        for run_stage_index, (run_stage_name, run_stage_cfg) in enumerate(run_stage_configs):
            run_stage_cfg = dict(run_stage_cfg)
            if run_stage_index > 0:
                run_stage_cfg.pop("fine_tune_model_path", None)
                run_stage_cfg.pop("fine_tune_restart_path", None)
            if _stage_completed(member_dir, run_stage_name):
                continue
            _restore_stage_start(member_dir, run_stage_name, initial_restart_source=initial_restart_source)
            if run_stage_index > 0:
                if run_stage_cfg.get("native_recipe_path") or run_stage_cfg.get("recipe_path") or run_stage_cfg.get("nep_in"):
                    _stage_native_nep_recipe(run_stage_cfg, member_dir)
                else:
                    write_nep_input(member_dir / "nep.in", run_stage_cfg)
            log_path = member_dir / ("nep.stdout.log" if len(run_stage_configs) == 1 else f"nep_{run_stage_name}.stdout.log")
            central_log_path = None
            if log_dir is not None:
                central_log_name = (
                    f"{log_prefix}_{member_dir.name}.log"
                    if len(run_stage_configs) == 1
                    else f"{log_prefix}_{member_dir.name}_{run_stage_name}.log"
                )
                central_log_path = Path(log_dir).resolve() / central_log_name
                central_log_path.parent.mkdir(parents=True, exist_ok=True)
            _set_stage_status(
                member_dir,
                run_stage_name,
                {
                    "status": "running",
                    "stage_index": int(run_stage_index),
                    "generation": int(run_stage_cfg.get("generation", 0) or 0),
                    "nep_in": str((member_dir / "nep.in").resolve()),
                },
            )
            with open(log_path, "w", encoding="utf-8") as member_handle:
                central_handle = open(central_log_path, "w", encoding="utf-8") if central_log_path is not None else None
                try:
                    try:
                        proc = subprocess.Popen(
                            list(command),
                            cwd=member_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=env,
                        )
                    except FileNotFoundError as exc:
                        raise RuntimeError(f"cannot find nep executable: {command[0]}") from exc
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        member_handle.write(line)
                        member_handle.flush()
                        if central_handle is not None:
                            central_handle.write(line)
                            central_handle.flush()
                    returncode = proc.wait()
                finally:
                    if central_handle is not None:
                        central_handle.close()
            stage_logs.append(str(log_path.resolve()))
            if central_log_path is not None:
                central_stage_logs.append(str(central_log_path.resolve()))
            snapshot_payload = _snapshot_stage_outputs(member_dir, run_stage_name)
            _set_stage_status(
                member_dir,
                run_stage_name,
                {
                    "status": "success" if int(returncode) == 0 else "warning",
                    "returncode": int(returncode),
                    "stdout_log": str(log_path.resolve()),
                    **snapshot_payload,
                },
            )
        result["returncode"] = int(returncode)
        result["stdout_log"] = str(Path(stage_logs[-1]).resolve()) if stage_logs else str((member_dir / "nep.stdout.log").resolve())
        result["stage_stdout_logs"] = stage_logs
        if central_log_path is not None:
            result["central_stdout_log"] = str(Path(central_stage_logs[-1]).resolve()) if central_stage_logs else str(central_log_path.resolve())
        if central_stage_logs:
            result["central_stage_stdout_logs"] = central_stage_logs
        model_path = _resolve_trained_model_path(member_dir)
        if model_path is None or not model_path.exists():
            generation = int(run_stage_configs[-1][1].get("generation", nep_cfg.get("generation", 100000)))
            hint = ""
            if generation < 100:
                hint = f" current generation={generation}; this NEP build writes nep.txt on 100-generation reporting intervals."
            raise RuntimeError(
                f"NEP training did not produce a usable model file for member {member_dir.name}; "
                f"expected nep.txt or nep*.txt, see {log_path.resolve()}.{hint}"
            )
        result["model_path"] = str(model_path.resolve())
        restart_path = _resolve_restart_path(member_dir)
        if restart_path is None or not restart_path.exists():
            raise RuntimeError(f"NEP training did not produce nep.restart for member {member_dir.name}; see {log_path.resolve()}")
        result["restart_path"] = str(restart_path.resolve())
        member_metrics, member_warnings = _parse_loss_metrics(Path(member["loss_out"]).resolve())
        result.update(member_metrics)
        result["warnings"] = member_warnings
        result["stage_status_path"] = str(_member_stage_status_path(member_dir).resolve())
        result["status"] = "success" if int(returncode) == 0 else "warning"
        members_payload.append(result)
        models.append(str(model_path.resolve()))
        restarts.append(str(restart_path.resolve()))
        metrics[member_dir.name] = member_metrics
        warnings.extend(member_warnings)

    decision_payload = dict(resolved_train_decision or {})
    payload = {
        "backend": "nep",
        "models": models,
        "restarts": restarts,
        "metrics": metrics,
        "members": members_payload,
        "warnings": warnings,
        "nep_config": dict(nep_cfg),
    }
    if decision_payload:
        payload["effective_train_recipe_branch"] = str(decision_payload.get("effective_train_recipe_branch", ""))
        payload["dataset_mode"] = str(decision_payload.get("dataset_mode", ""))
        payload["initialization_mode"] = str(decision_payload.get("initialization_mode", ""))
    else:
        payload["training_stage"] = stage_norm
        payload["initialization_mode"] = "foundation_finetune" if nep_cfg.get("foundation_bootstrap") else (
            "incremental_finetune" if stage_norm == "previous_restart_continue" else "bootstrap"
        )
    if nep_cfg.get("foundation_bootstrap"):
        payload["foundation_bootstrap"] = dict(nep_cfg.get("foundation_bootstrap") or {})
    if summary_path is None:
        summary_path = Path(work_root).resolve() / "summary.json"
    write_nep_committee_manifest(summary_path, payload)
    return str(Path(summary_path).resolve())


def load_nep_committee_summary(summary_path: str | Path) -> dict[str, Any]:
    return read_json(summary_path)
