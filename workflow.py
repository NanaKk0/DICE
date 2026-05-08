import json
import re
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from workflow_core.common.io import read_json
from workflow_runtime.paths.path_manager import PathManager
from workflow_runtime.monitoring.resource_monitor import ResourceSummary
from workflow_runtime.state.run_state import RunStateManager
from workflow_runtime.logging.runtime_logging import print_exception_banner
from workflow_runtime.state.signal_handler import install_signal_handlers
from workflow_runtime.state.step_runner import execute_step
from workflow_core.dft.external_import import discover_external_dft_xyz_files
from workflow_core.entrypoints.registry import TASK_REGISTRY

EPOCH_LINE_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+INFO:\s+"
    r"(?P<label>Initial|Epoch\s+(?P<epoch>\d+)):\s+head:\s+\S+,\s+loss=(?P<loss>[0-9.eE+-]+),\s+"
    r"RMSE_E_per_atom=\s*(?P<rmse_e>[0-9.eE+-]+)\s+meV,\s+RMSE_F=\s*(?P<rmse_f>[0-9.eE+-]+)\s+meV\s*/\s*A,\s+"
    r"RMSE_stress=\s*(?P<rmse_s>[0-9.eE+-]+)\s+meV\s*/\s*A\^3"
)
LEARNING_RATE_RE = re.compile(r"Learning rate:\s*([0-9.eE+-]+)")
TRAIN_TOTAL_RE = re.compile(r"Total training set \[(\d+) configs")
VALID_TOTAL_RE = re.compile(r"Validation set contains (\d+) configurations")
TEST_TABLE_RE = re.compile(r"\|\s+Default_default\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|.*\|\s+([0-9.]+)\s+\|")


class NullMetricsLogger:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _empty_resource_summary() -> ResourceSummary:
    return ResourceSummary()


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_resume_config(wf: dict[str, Any]) -> dict[str, Any]:
    payload = dict(wf.get("resume", {}) or {})
    payload.setdefault("enabled", True)
    payload.setdefault("mode", "auto")
    payload.setdefault("verify_success_markers", True)
    payload.setdefault("rerun_incomplete_step", True)
    payload.setdefault("backup_partial_outputs", True)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return read_json(path)


def _resolve_workspace_path(paths: PathManager, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (paths.workspace_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_previous_round_dft_source(paths: PathManager, prev_round: int) -> Path | None:
    external_files = discover_external_dft_xyz_files(paths, prev_round)
    if external_files:
        return external_files[0]
    runtime_labeled = paths.dft_submit_labeled_xyz(prev_round)
    if runtime_labeled.exists() and runtime_labeled.is_file() and runtime_labeled.stat().st_size > 0:
        return runtime_labeled.resolve()
    return None


def _resolve_previous_round_all_dft_candidate_sources(paths: PathManager, prev_round: int) -> list[Path]:
    status_path = paths.dp_subworkflow_status_json(prev_round)
    max_dp_round = 0
    if status_path.exists():
        try:
            status_payload = _load_json(status_path)
            max_dp_round = int(status_payload.get("current_dp_round", 0) or 0)
        except Exception:
            max_dp_round = 0
    sources: list[Path] = []
    for dp_round in range(max_dp_round + 1):
        candidate = paths.dp_subworkflow_round_active_learning_dir(prev_round, dp_round) / "dft.xyz"
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            sources.append(candidate.resolve())
    return sources


def _resolve_preflight_config_context(paths: PathManager, wf: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "committee_cycle_config_path": None,
        "dp_subworkflow_config_path": None,
    }
    committee_cycle_cfg_path = _resolve_workspace_path(paths, (wf.get("committee_cycle") or {}).get("config"))
    context["committee_cycle_config_path"] = str(committee_cycle_cfg_path) if committee_cycle_cfg_path else None
    if committee_cycle_cfg_path is None or not committee_cycle_cfg_path.exists():
        return context

    committee_cycle_cfg = _load_json(committee_cycle_cfg_path)
    dp_cfg_path = _resolve_workspace_path(
        paths,
        (committee_cycle_cfg.get("dp_subworkflow") or {}).get("config"),
    )
    context["dp_subworkflow_config_path"] = str(dp_cfg_path) if dp_cfg_path else None
    if dp_cfg_path is None or not dp_cfg_path.exists():
        return context

    return context


def _append_preflight_check(
    payload: dict[str, Any],
    *,
    key: str,
    ok: bool,
    required: bool = True,
    detail: str | None = None,
    path: Path | None = None,
) -> None:
    entry = {
        "key": key,
        "ok": bool(ok),
        "required": bool(required),
        "detail": detail or "",
        "path": str(path.resolve()) if path is not None else None,
    }
    payload["checks"].append(entry)
    if path is not None:
        payload["assets"][key] = str(path.resolve())
    if required and not ok:
        label = key if detail is None else f"{key}: {detail}"
        payload["missing"].append(label)


def _write_preflight_report(paths: PathManager, round_id: int, payload: dict[str, Any]) -> None:
    return None


def _preflight_outer_round_start(paths: PathManager, wf: dict[str, Any], round_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": paths.runtime_root.name,
        "round": int(round_id),
        "runtime_root": str(paths.runtime_root),
        "pipeline": list(wf.get("pipeline", [])),
        "previous_round": int(round_id - 1),
        "ok": True,
        "checks": [],
        "missing": [],
        "warnings": [],
        "assets": {},
        "config": {},
    }
    if round_id <= 0:
        payload["warnings"].append("preflight skipped for round 0")
        _write_preflight_report(paths, round_id, payload)
        return payload

    prev_round = round_id - 1
    config_ctx = _resolve_preflight_config_context(paths, wf)
    payload["config"] = config_ctx

    committee_cycle_status = paths.committee_cycle_status_json(prev_round)
    dp_status_path = paths.dp_subworkflow_status_json(prev_round)
    _append_preflight_check(
        payload,
        key="previous_committee_cycle_status",
        ok=committee_cycle_status.exists(),
        detail="previous outer_round committee_cycle status must exist",
        path=committee_cycle_status,
    )
    _append_preflight_check(
        payload,
        key="previous_dp_subworkflow_status",
        ok=dp_status_path.exists(),
        detail="previous outer_round dp_subworkflow status must exist",
        path=dp_status_path,
    )

    dp_status_payload: dict[str, Any] | None = None
    if dp_status_path.exists():
        try:
            dp_status_payload = _load_json(dp_status_path)
        except Exception as exc:
            payload["warnings"].append(f"failed to parse previous dp status: {exc}")

    prev_stopped = bool((dp_status_payload or {}).get("stopped"))
    _append_preflight_check(
        payload,
        key="previous_dp_subworkflow_stopped",
        ok=prev_stopped,
        detail="previous outer_round must be in stopped terminal state",
    )

    final_dp_model_path_raw = (dp_status_payload or {}).get("final_dp_model_path")
    final_dp_model_path = Path(final_dp_model_path_raw).resolve() if final_dp_model_path_raw else None
    _append_preflight_check(
        payload,
        key="previous_final_dp_model",
        ok=bool(final_dp_model_path and final_dp_model_path.exists()),
        detail="previous outer_round final DP committee summary/path must exist",
        path=final_dp_model_path,
    )

    prev_round_dft_source = _resolve_previous_round_dft_source(paths, prev_round)
    _append_preflight_check(
        payload,
        key="previous_round_historical_dft_source",
        ok=prev_round_dft_source is not None,
        detail="teacher finetune historical DFT source for previous outer_round must exist",
        path=prev_round_dft_source,
    )

    use_cross_outer_entry = round_id > 0
    payload["config"]["use_cross_outer_entry_sampling"] = use_cross_outer_entry
    if use_cross_outer_entry:
        source_mode = "previous_outer_all_dft_candidates"
        payload["config"]["cross_outer_round0_source_mode"] = source_mode
        if source_mode == "previous_outer_all_dft_candidates":
            previous_dft_candidate_sources = _resolve_previous_round_all_dft_candidate_sources(paths, prev_round)
            first_candidate = previous_dft_candidate_sources[0] if previous_dft_candidate_sources else None
            _append_preflight_check(
                payload,
                key="previous_round_all_dft_candidate_sources",
                ok=bool(previous_dft_candidate_sources),
                detail="cross-outer dp_r0 source requires previous outer_round dft candidate xyz files",
                path=first_candidate,
            )
            payload["assets"]["previous_round_all_dft_candidate_source_count"] = str(len(previous_dft_candidate_sources))
        else:
            _append_preflight_check(
                payload,
                key="cross_outer_round0_source_mode_supported",
                ok=False,
                detail=f"unsupported cross_outer source_mode '{source_mode}'",
            )

    payload["ok"] = not payload["missing"]
    _write_preflight_report(paths, round_id, payload)
    if not payload["ok"]:
        missing_summary = "; ".join(payload["missing"])
        raise RuntimeError(
            f"Preflight failed for outer_round r{round_id}. Missing required assets: {missing_summary}."
        )
    return payload


def _maybe_migrate_legacy_workflow_state(paths: PathManager, round_id: int) -> dict[str, Any]:
    target_state = paths.workflow_round_state_json(round_id)
    target_markers = paths.workflow_round_marker_dir(round_id)
    legacy_state = paths.runtime_meta_dir() / "workflow_state.json"
    legacy_markers = paths.runtime_meta_dir() / "markers"
    result = {
        "migrated": False,
        "legacy_state_path": str(legacy_state.resolve()),
        "target_state_path": str(target_state.resolve()),
    }
    if target_state.exists() or not legacy_state.exists():
        return result
    try:
        legacy_payload = _load_json(legacy_state)
    except Exception:
        return result
    if int(legacy_payload.get("round", -1)) != int(round_id):
        return result
    target_state.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_state, target_state)
    if legacy_markers.exists():
        shutil.copytree(legacy_markers, target_markers, dirs_exist_ok=True)
    result["migrated"] = True
    return result


def build_runtime_root(workspace_root: Path, workflow_path: Path, wf: dict) -> Path:
    explicit_run_name = wf.get("run_name")
    if explicit_run_name:
        safe_name = str(explicit_run_name).strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{workflow_path.stem}_{timestamp}"
    return (workspace_root / "runs" / safe_name).resolve()


def initialize_runtime_layout(paths: PathManager, workflow_path: Path, wf: dict) -> None:
    for directory in (
        paths.runtime_root,
        paths.runtime_data_dir(),
        paths.runtime_results_dir(),
        paths.runtime_logs_dir(),
        paths.runtime_meta_dir(),
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _existing_path_str(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _safe_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(value)
    except Exception:
        return None


def count_extxyz_frames(path: Path | None) -> int | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "Properties=" in line:
                    count += 1
        return count
    except Exception:
        return None


def gather_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }
    for module_name, key in (
        ("torch", "torch_version"),
        ("mace", "mace_version"),
        ("deepmd", "deepmd_version"),
    ):
        try:
            module = __import__(module_name)
            info[key] = getattr(module, "__version__", None)
        except Exception:
            info[key] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda_available"] = None
        info["gpu_name"] = None
    return info


def train_mace_success_check(paths: PathManager, round_id: int) -> bool:
    model_dir = paths.mace_model_dir()
    if not model_dir.exists():
        return False
    final_models = sorted(model_dir.glob("*.model"))
    checkpoint_models = sorted((model_dir / "checkpoints").glob("*.model"))
    candidates = final_models + checkpoint_models
    return any(_is_nonempty_file(path) for path in candidates)


def committee_cycle_success_check(paths: PathManager, round_id: int) -> bool:
    status_path = paths.committee_cycle_status_json(round_id)
    dp_status_path = paths.dp_subworkflow_status_json(round_id)
    if not status_path.exists() or not dp_status_path.exists():
        return False
    try:
        cycle_status = read_json(status_path)
        dp_status = read_json(dp_status_path)
    except Exception:
        return False
    return bool(cycle_status.get("mace_committee_dir")) and bool(dp_status.get("stopped"))


def dft_success_check(paths: PathManager, round_id: int) -> bool:
    status_path = paths.dft_submit_status_json(round_id)
    if not status_path.exists():
        return False
    try:
        payload = read_json(status_path)
    except Exception:
        return False
    return payload.get("status") == "completed"


def build_top_level_steps(
    wf: dict[str, Any],
    paths: PathManager,
    round_id: int,
) -> list[dict[str, Any]]:
    success_checks: dict[str, Callable[[], bool]] = {
        "train_mace": lambda: train_mace_success_check(paths, round_id),
        "committee_cycle": lambda: committee_cycle_success_check(paths, round_id),
        "dft": lambda: dft_success_check(paths, round_id),
    }
    output_dirs: dict[str, Path] = {
        "train_mace": paths.mace_model_dir(),
        "committee_cycle": paths.committee_cycle_results_dir(round_id),
        "dft": paths.dft_submit_results_dir(round_id),
    }
    backup_flags = {
        "train_mace": False,
        "committee_cycle": False,
        "dft": False,
    }
    steps: list[dict[str, Any]] = []
    for index, step_name in enumerate(wf.get("pipeline", [])):
        if step_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown step {step_name}")
        cfg = dict(wf.get(step_name, {}))
        steps.append(
            {
                "id": f"r{round_id}_{step_name}",
                "name": step_name,
                "round": round_id,
                "index": index,
                "output_dir": str(output_dirs.get(step_name)) if step_name in output_dirs else None,
                "backup_partial_outputs": backup_flags.get(step_name, False),
                "run": lambda step_name=step_name, cfg=cfg: TASK_REGISTRY[step_name](cfg, paths, round_id),
                "success_check": success_checks.get(step_name),
            }
        )
    return steps


def write_run_summary(
    paths: PathManager,
    wf: dict,
    round_id: int,
    step_states: dict[str, dict],
    overall_status: str,
    current_step: str | None = None,
    error: dict | None = None,
) -> None:
    return None


def parse_mace_training_log(
    log_path: Path,
    step_id: str,
    round_id: int,
    metrics: Any,
    cfg: dict[str, Any],
    paths: PathManager,
) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    lr = None
    train_size = None
    valid_size = None
    best_rmse_e = None
    best_rmse_f = None
    best_rmse_s = None
    step_start_dt = None
    eval_test_metrics: dict[str, Any] = {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception:
        return {}

    for line in lines:
        lr_match = LEARNING_RATE_RE.search(line)
        if lr_match:
            try:
                lr = float(lr_match.group(1))
            except Exception:
                pass
        train_match = TRAIN_TOTAL_RE.search(line)
        if train_match:
            train_size = int(train_match.group(1))
        valid_match = VALID_TOTAL_RE.search(line)
        if valid_match:
            valid_size = int(valid_match.group(1))
        epoch_match = EPOCH_LINE_RE.search(line)
        if not epoch_match:
            continue
        ts = epoch_match.group("timestamp")
        try:
            current_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            current_dt = None
        if current_dt is not None and step_start_dt is None:
            step_start_dt = current_dt
        epoch_label = epoch_match.group("label")
        epoch_value = -1 if epoch_label == "Initial" else int(epoch_match.group("epoch"))
        train_loss = float(epoch_match.group("loss"))
        rmse_e = float(epoch_match.group("rmse_e"))
        rmse_f = float(epoch_match.group("rmse_f"))
        rmse_s = float(epoch_match.group("rmse_s"))
        if best_rmse_e is None or rmse_e < best_rmse_e:
            best_rmse_e = rmse_e
        if best_rmse_f is None or rmse_f < best_rmse_f:
            best_rmse_f = rmse_f
        if best_rmse_s is None or rmse_s < best_rmse_s:
            best_rmse_s = rmse_s
        wall_time = None
        if current_dt is not None and step_start_dt is not None:
            wall_time = round((current_dt - step_start_dt).total_seconds(), 6)
        row = {
            "run_name": metrics.run_name,
            "round_idx": round_id,
            "step_id": step_id,
            "model_family": cfg.get("model", "MACE"),
            "phase": "stage_two" if epoch_value >= int(cfg.get("start_swa", 0) or 0) else "stage_one",
            "epoch": epoch_value,
            "global_step": "",
            "timestamp": ts.replace(" ", "T") + "Z",
            "wall_time_from_step_start_sec": wall_time,
            "train_loss": "",
            "valid_loss": train_loss,
            "rmse_e_mev_per_atom": rmse_e,
            "rmse_f_mev_per_a": rmse_f,
            "rmse_stress_mev_per_a3": rmse_s,
            "learning_rate": lr,
            "train_batch_size": cfg.get("batch_size"),
            "valid_batch_size": cfg.get("valid_batch_size"),
            "num_train_structures": train_size,
            "num_valid_structures": valid_size,
            "epoch_time_sec": "",
            "cumulative_train_time_sec": wall_time,
            "gpu_util_percent": "",
            "gpu_mem_mb": "",
            "cpu_percent": "",
            "ram_mb": "",
        }
        rows.append(row)
        metrics.log_training_metric(row)

    for index, line in enumerate(lines):
        if "Error-table on TEST" not in line:
            continue
        for next_line in lines[index + 1:index + 10]:
            table_match = TEST_TABLE_RE.search(next_line)
            if table_match:
                eval_test_metrics = {
                    "best_rmse_e": float(table_match.group(1)),
                    "best_rmse_f": float(table_match.group(2)),
                    "best_rmse_stress": float(table_match.group(3)),
                }
                break
        if eval_test_metrics:
            break

    updates = {
        "epoch_count": len(rows),
        "best_rmse_e": eval_test_metrics.get("best_rmse_e", best_rmse_e),
        "best_rmse_f": eval_test_metrics.get("best_rmse_f", best_rmse_f),
        "best_rmse_stress": eval_test_metrics.get("best_rmse_stress", best_rmse_s),
        "input_structure_count": train_size,
        "output_structure_count": train_size,
        "train_batch_count": len(rows),
        "valid_batch_count": len(rows),
        "model_path": _existing_path_str(paths.mace_model_dir() / f"{cfg.get('name', 'model')}.model"),
    }
    return updates


def collect_step_metric_updates(
    step_name: str,
    step_id: str,
    paths: PathManager,
    round_id: int,
    cfg: dict[str, Any],
    metrics: Any,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if step_name == "train_mace":
        log_path = paths.runtime_logs_dir() / f"r{round_id}" / "train_mace.log"
        updates["log_path"] = str(log_path) if log_path.exists() else ""
        parsed = parse_mace_training_log(log_path, step_id, round_id, metrics, cfg, paths)
        updates.update(parsed)
        updates["output_path"] = str(paths.mace_model_dir())
        model_candidates = sorted(paths.mace_model_dir().glob("*stagetwo.model")) or sorted(paths.mace_model_dir().glob("*.model"))
        if model_candidates:
            updates["model_path"] = str(model_candidates[-1])
            metrics.update_run_summary({"final_mace_model_path": str(model_candidates[-1])})
    elif step_name == "committee_cycle":
        status_path = paths.committee_cycle_status_json(round_id)
        updates["output_path"] = str(paths.committee_cycle_results_dir(round_id))
        updates["input_path"] = str(paths.committee_cycle_finetune_merged_xyz(round_id))
        updates["input_structure_count"] = count_extxyz_frames(paths.committee_cycle_finetune_merged_xyz(round_id))
        if status_path.exists():
            try:
                payload = read_json(status_path)
                updates["model_path"] = payload.get("mace_committee_dir", "")
            except Exception:
                pass
    elif step_name == "dft":
        status_path = paths.dft_submit_status_json(round_id)
        updates["output_path"] = str(paths.dft_submit_results_dir(round_id))
        updates["input_path"] = str(paths.dft_submit_source_xyz(round_id))
        updates["input_structure_count"] = count_extxyz_frames(paths.dft_submit_source_xyz(round_id))
        if status_path.exists():
            try:
                payload = read_json(status_path)
                updates["output_structure_count"] = count_extxyz_frames(paths.dft_submit_labeled_xyz(round_id))
                metrics.update_run_summary({"total_labeled_structures": count_extxyz_frames(paths.dft_submit_labeled_xyz(round_id)) or 0})
                if payload.get("status") == "completed":
                    updates["model_path"] = ""
            except Exception:
                pass
    return updates


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python workflow.py workflow.yaml")

    workflow_path = Path(sys.argv[1]).resolve()
    wf = load_yaml(workflow_path)
    round_id = wf["round"]
    resume_cfg = load_resume_config(wf)

    workspace_root = Path(__file__).resolve().parent
    runtime_root = build_runtime_root(workspace_root, workflow_path, wf)
    paths = PathManager(workspace_root=workspace_root, runtime_root=runtime_root)
    initialize_runtime_layout(paths, workflow_path, wf)
    migration_info = _maybe_migrate_legacy_workflow_state(paths, round_id)
    round_state_path = paths.workflow_round_state_json(round_id)
    resumed_existing_round = round_state_path.exists()
    if round_id > 0 and not resumed_existing_round:
        _preflight_outer_round_start(paths, wf, round_id)

    steps = build_top_level_steps(wf, paths, round_id)
    step_states: dict[str, dict] = {}
    write_run_summary(paths, wf, round_id, step_states, overall_status="initialized")

    state_mgr = RunStateManager(
        round_state_path,
        paths.workflow_round_marker_dir(round_id),
        backup_partial_outputs=bool(resume_cfg.get("backup_partial_outputs", True)),
        verify_success_markers=bool(resume_cfg.get("verify_success_markers", True)),
    )
    resumed = state_mgr.exists()
    if resumed:
        state_mgr.load()
    else:
        state_mgr.init_state(run_name=paths.runtime_root.name, round_id=round_id, steps=steps)

    success_checks = {step["id"]: step["success_check"] for step in steps if callable(step.get("success_check"))}
    state_mgr.reconcile(success_checks=success_checks)
    resume_index = state_mgr.get_resume_index() if bool(resume_cfg.get("enabled", True)) else 0
    current_step_id: dict[str, str | None] = {"value": None}
    install_signal_handlers(state_mgr, lambda: current_step_id["value"])

    print(f"[INFO] Workspace root: {paths.workspace_root}")
    print(f"[INFO] Runtime root:   {paths.runtime_root}")

    state_mgr.mark_workflow_running(
        current_step_index=resume_index if resume_index < len(steps) else None,
        current_step_id=steps[resume_index]["id"] if resume_index < len(steps) else None,
    )

    try:
        for step in steps[resume_index:]:
            step_name = step["name"]
            step_id = step["id"]

            print("\n==============================")
            print(f"RUN STEP: {step_name} (r{round_id})")
            print("==============================")

            try:
                current_step_id["value"] = step_id
                state_mgr.mark_workflow_running(current_step_index=step["index"], current_step_id=step_id)
                step_states[step_name] = {"status": "running", "step_id": step_id}
                write_run_summary(paths, wf, round_id, step_states, overall_status="running", current_step=step_name)
                execute_step(step, state_mgr)

                step_states[step_name] = {"status": "completed", "step_id": step_id}
                write_run_summary(paths, wf, round_id, step_states, overall_status="running", current_step=step_name)

                if step_name == "dft":
                    count_extxyz_frames(paths.dft_submit_labeled_xyz(round_id))
                elif step_name == "committee_cycle":
                    dp_status_path = paths.dp_subworkflow_status_json(round_id)
                    if dp_status_path.exists():
                        try:
                            payload = read_json(dp_status_path)
                        except Exception:
                            pass
            except KeyboardInterrupt as exc:
                current_step_id["value"] = None
                state_mgr.mark_workflow_interrupted(str(exc) or "Interrupted by KeyboardInterrupt")
                step_states[step_name] = {
                    "status": "interrupted",
                    "step_id": step_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
                write_run_summary(
                    paths,
                    wf,
                    round_id,
                    step_states,
                    overall_status="interrupted",
                    current_step=step_name,
                    error={"step": step_name, "type": exc.__class__.__name__, "message": str(exc)},
                )
                raise
            except SystemExit as exc:
                current_step_id["value"] = None
                state_mgr.mark_workflow_interrupted(f"Interrupted by SystemExit({exc.code})")
                step_states[step_name] = {
                    "status": "interrupted",
                    "step_id": step_id,
                    "error_type": "SystemExit",
                    "error_message": str(exc),
                }
                write_run_summary(
                    paths,
                    wf,
                    round_id,
                    step_states,
                    overall_status="interrupted",
                    current_step=step_name,
                    error={"step": step_name, "type": "SystemExit", "message": str(exc)},
                )
                raise
            except Exception as exc:
                current_step_id["value"] = None
                state_mgr.mark_workflow_failed(str(exc))
                step_states[step_name] = {
                    "status": "failed",
                    "step_id": step_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
                write_run_summary(
                    paths,
                    wf,
                    round_id,
                    step_states,
                    overall_status="failed",
                    current_step=step_name,
                    error={
                        "step": step_name,
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                print_exception_banner(step_name, paths.runtime_root, exc)
                raise

        current_step_id["value"] = None
        state_mgr.mark_workflow_success()
        write_run_summary(paths, wf, round_id, step_states, overall_status="completed")
    except Exception:
        raise


if __name__ == "__main__":
    main()


