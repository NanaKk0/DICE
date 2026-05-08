from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_core.dft.runtime import resolve_dft_runtime
from workflow_core.dft.submit import submit_module
from workflow_core.common.io import ensure_dir, read_json, write_json


def load_dft_run_config(config_path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(config_path).resolve())
    payload.setdefault("skip_finished", True)
    payload.setdefault("retry_failed", False)
    payload.setdefault("dry_run", False)
    payload.setdefault("required_outputs", ["cp2k.out"])
    return payload


def run_dft_jobs(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_dft_runtime(cfg)
    task_root = Path(cfg["task_root"]).resolve()
    output_root = ensure_dir(cfg["output_root"])
    record_path = Path(cfg.get("record_path", output_root / "record.dft.json")).resolve()
    status_path = Path(cfg.get("status_path", output_root / "status.json")).resolve()

    submit_cfg = dict(cfg)
    submit_cfg["input_dir"] = str(task_root)
    submit_cfg["output_dir"] = str(output_root)
    summary = submit_module(submit_cfg)

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
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    write_json(record_path, record_payload)

    payload = {
        "status": "ran" if not bool(cfg.get("dry_run", False)) else "prepared_only",
        "jobid": cfg["jobid"],
        "task_root": str(task_root),
        "output_root": str(output_root),
        "submit_summary_json": summary["submit_summary_json"],
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    write_json(status_path, payload)
    return payload

