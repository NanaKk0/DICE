from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from workflow_core.common.io import read_json, write_json


def trim_uncertainty_only_sampling_summary(summary_path: Path) -> Path:
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        return summary_path
    if str(summary.get("sampling_policy_mode", "")).strip().lower() != "uncertainty_only":
        return summary_path
    for key in (
        "failed_geometry_count",
        "suspicious_candidate_count",
        "high_tail_buffer_count",
        "selected_tail_counts",
        "selected_per_job_top",
        "selected_jsonl",
        "decision_metric",
        "level_f_lo",
        "level_f_hi",
        "stop_high_unc_threshold",
        "sampling_backend",
        "failed_task_count",
        "f_trust_lo",
        "f_trust_hi",
        "sampling_policy_mode",
        "candidate_train_selection_mode",
        "candidate_train_random_seed",
        "candidate_by_system_dir",
        "failed_by_system_dir",
        "max_parallel_tasks",
        "device_policy",
        "execution_backend",
        "task_group_size",
        "max_parallel_groups",
        "group_parallel_degree",
    ):
        summary.pop(key, None)
    write_json(summary_path, summary)
    return summary_path


def append_sampling_summary_to_progress_log(progress_log_path: str | Path | None, summary_payload: dict) -> None:
    if not progress_log_path:
        return
    progress_log_path = Path(progress_log_path).resolve()
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] === Final Sampling Summary ===\n")
        handle.write(json.dumps(summary_payload, indent=2, ensure_ascii=False))
        handle.write("\n")
