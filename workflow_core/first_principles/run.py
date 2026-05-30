from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from workflow_core.common.io import ensure_dir, read_json, write_json
from workflow_core.first_principles.substep_harness import FirstPrinciplesSubstepContext, run_first_principles_substep


def load_dft_run_config(config_path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(config_path).resolve())
    payload.setdefault("skip_finished", True)
    payload.setdefault("retry_failed", False)
    payload.setdefault("dry_run", False)
    payload.setdefault("required_outputs", ["cp2k.out"])
    return payload


load_first_principles_run_config = load_dft_run_config


def run_dft_jobs(cfg: dict[str, Any]) -> dict[str, Any]:
    return run_first_principles_substep(cfg, substep="run", execute=_run_dft_jobs)


run_first_principles_jobs = run_dft_jobs


def _run_dft_jobs(context: FirstPrinciplesSubstepContext) -> dict[str, Any]:
    cfg = context.cfg
    backend_name = context.backend_name
    backend = context.backend
    task_root = Path(cfg["task_root"]).resolve()
    output_root = ensure_dir(cfg["output_root"])
    record_path = Path(cfg.get("record_path", output_root / "record.dft.json")).resolve()
    status_path = Path(cfg["status_path"]).resolve() if cfg.get("status_path") else None

    summary = backend.run_tasks({**cfg, "task_root": str(task_root), "output_root": str(output_root)})
    _publish_backend_task_evidence(task_root / str(cfg["jobid"]), output_root)

    record_payload = {}
    if record_path.exists():
        record_payload = read_json(record_path)
    record_payload["run"] = {
        "submit_log": summary["log_path"],
        "submit_summary_json": summary["submit_summary_json"],
        "n_total": summary["n_total"],
        "n_submitted": summary["n_submitted"],
        "n_skipped": summary["n_skipped"],
        "n_success": summary["n_success"],
        "n_failure": summary["n_failure"],
        "dry_run": bool(cfg.get("dry_run", False)),
        "first_principles_backend": backend_name,
        "first_principles_runtime_resolution": cfg.get("first_principles_runtime_resolution"),
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    write_json(record_path, record_payload)

    payload = {
        "status": "ran" if not bool(cfg.get("dry_run", False)) else "prepared_only",
        "jobid": cfg["jobid"],
        "task_root": str(task_root),
        "output_root": str(output_root),
        "submit_summary_json": summary["submit_summary_json"],
        "first_principles_backend": backend_name,
        "first_principles_runtime_resolution": cfg.get("first_principles_runtime_resolution"),
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    if status_path is not None:
        write_json(status_path, payload)
    return payload


def _publish_backend_task_evidence(job_root: Path, output_root: Path) -> None:
    if not job_root.is_dir():
        return
    for task_dir in sorted(path for path in job_root.iterdir() if path.is_dir()):
        target_task_dir = output_root / task_dir.name
        for source in (task_dir / "task_status.json", *sorted(task_dir.glob("*.out"))):
            if not source.is_file():
                continue
            target_task_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_task_dir / source.name)

