from __future__ import annotations

import csv
import json
import socket
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json


RUN_SUMMARY_FIELDS = [
    "run_name",
    "workflow_version",
    "start_time",
    "end_time",
    "total_wall_time_sec",
    "status",
    "total_rounds_completed",
    "total_steps_completed",
    "total_steps_failed",
    "total_steps_interrupted",
    "total_retries",
    "total_candidate_structures",
    "total_selected_structures",
    "total_labeled_structures",
    "total_discarded_structures",
    "total_training_structures_added",
    "final_train_dataset_size",
    "final_dp_model_path",
    "final_mace_model_path",
    "best_eval_rmse_e_mev_per_atom",
    "best_eval_rmse_f_mev_per_a",
    "best_eval_rmse_stress_mev_per_a3",
    "total_gpu_time_sec",
    "avg_gpu_util_percent",
    "max_gpu_mem_mb",
    "avg_cpu_percent",
    "max_ram_mb",
    "manual_interrupt_count",
    "auto_resume_count",
    "time_per_labeled_structure",
    "time_per_selected_structure",
    "gpu_time_per_labeled_structure",
    "selection_precision",
    "label_efficiency",
]

ROUND_FIELDS = [
    "run_name",
    "round_idx",
    "start_time",
    "end_time",
    "wall_time_sec",
    "status",
    "input_train_size",
    "output_train_size",
    "new_train_structures",
    "candidate_structures",
    "sampled_structures",
    "selected_structures",
    "labeled_structures",
    "discarded_structures",
    "selection_ratio",
    "label_ratio",
    "discard_ratio",
    "rmse_e_before",
    "rmse_e_after",
    "rmse_f_before",
    "rmse_f_after",
    "rmse_stress_before",
    "rmse_stress_after",
    "delta_rmse_e",
    "delta_rmse_f",
    "delta_rmse_stress",
    "rmse_f_gain_per_label",
    "rmse_f_gain_per_hour",
    "gpu_time_sec",
    "avg_gpu_util",
    "max_gpu_mem_mb",
    "avg_cpu_percent",
    "max_ram_mb",
    "retry_count",
    "manual_intervention_count",
]

STEP_FIELDS = [
    "run_name",
    "round_idx",
    "step_id",
    "step_name",
    "status",
    "attempt",
    "start_time",
    "end_time",
    "wall_time_sec",
    "input_structure_count",
    "output_structure_count",
    "input_dataset_size",
    "output_dataset_size",
    "input_path",
    "output_path",
    "model_path",
    "log_path",
    "epoch_count",
    "train_batch_count",
    "valid_batch_count",
    "avg_epoch_time_sec",
    "last_epoch_time_sec",
    "best_rmse_e",
    "best_rmse_f",
    "best_rmse_stress",
    "gpu_time_sec",
    "avg_gpu_util",
    "max_gpu_util",
    "max_gpu_mem_mb",
    "avg_cpu_percent",
    "max_cpu_percent",
    "max_ram_mb",
    "error_type",
    "error_message",
]

TRAINING_FIELDS = [
    "run_name",
    "round_idx",
    "step_id",
    "model_family",
    "phase",
    "epoch",
    "global_step",
    "timestamp",
    "wall_time_from_step_start_sec",
    "train_loss",
    "valid_loss",
    "rmse_e_mev_per_atom",
    "rmse_f_mev_per_a",
    "rmse_stress_mev_per_a3",
    "learning_rate",
    "train_batch_size",
    "valid_batch_size",
    "num_train_structures",
    "num_valid_structures",
    "epoch_time_sec",
    "cumulative_train_time_sec",
    "gpu_util_percent",
    "gpu_mem_mb",
    "cpu_percent",
    "ram_mb",
]

SAMPLING_FIELDS = [
    "run_name",
    "round_idx",
    "step_id",
    "sampler_type",
    "trajectory_id",
    "seed_id",
    "start_time",
    "end_time",
    "wall_time_sec",
    "status",
    "temperature",
    "time_step_fs",
    "n_steps_requested",
    "n_steps_completed",
    "n_frames_generated",
    "n_frames_kept",
    "n_frames_rejected",
    "min_pair_distance_min",
    "min_pair_distance_avg",
    "max_force_max",
    "max_force_avg",
    "energy_min",
    "energy_max",
    "energy_avg",
    "uncertainty_min",
    "uncertainty_max",
    "uncertainty_avg",
    "termination_reason",
    "error_message",
    "gpu_time_sec",
    "avg_gpu_util",
    "max_gpu_mem_mb",
    "avg_cpu_percent",
    "max_ram_mb",
]

LABELING_FIELDS = [
    "run_name",
    "round_idx",
    "step_id",
    "labeler_type",
    "batch_id",
    "start_time",
    "end_time",
    "wall_time_sec",
    "status",
    "input_structure_count",
    "success_structure_count",
    "failed_structure_count",
    "avg_time_per_structure_sec",
    "energy_mean",
    "energy_std",
    "force_norm_mean",
    "force_norm_std",
    "uncertainty_mean",
    "uncertainty_std",
    "output_path",
    "error_message",
]

RESOURCE_FIELDS = [
    "timestamp",
    "run_name",
    "round_idx",
    "step_id",
    "process_pid",
    "cpu_percent",
    "ram_mb",
    "ram_percent",
    "gpu_util_percent",
    "gpu_mem_mb",
    "gpu_power_w",
    "gpu_temp_c",
    "disk_read_mb_s",
    "disk_write_mb_s",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    n = _safe_float(numerator)
    d = _safe_float(denominator)
    if n is None or d in (None, 0.0):
        return None
    return n / d


@dataclass
class StepContext:
    step_id: str
    step_name: str
    round_idx: int
    attempt: int
    start_time: str
    start_perf: float
    payload: dict[str, Any]


@dataclass
class RoundContext:
    round_idx: int
    start_time: str
    start_perf: float
    payload: dict[str, Any]


class MetricsLogger:
    def __init__(self, metrics_dir: str | Path, run_name: str, *, workflow_version: str = "metrics_v1") -> None:
        self.metrics_dir = Path(metrics_dir).resolve()
        self.run_name = run_name
        self.workflow_version = workflow_version
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.run_summary_path = self.metrics_dir / "run_summary.json"
        self.run_summary_csv_path = self.metrics_dir / "run_summary.csv"
        self.round_metrics_path = self.metrics_dir / "round_metrics.csv"
        self.step_metrics_path = self.metrics_dir / "step_metrics.csv"
        self.training_metrics_path = self.metrics_dir / "training_metrics.csv"
        self.sampling_metrics_path = self.metrics_dir / "sampling_metrics.csv"
        self.labeling_metrics_path = self.metrics_dir / "labeling_metrics.csv"
        self.resource_metrics_path = self.metrics_dir / "resource_metrics.csv"
        self.event_log_path = self.metrics_dir / "event_log.jsonl"
        self.config_snapshot_path = self.metrics_dir / "config_snapshot.json"

        self.run_summary: dict[str, Any] = {field: None for field in RUN_SUMMARY_FIELDS}
        self.run_summary.update(
            {
                "run_name": run_name,
                "workflow_version": workflow_version,
                "status": "initialized",
                "total_rounds_completed": 0,
                "total_steps_completed": 0,
                "total_steps_failed": 0,
                "total_steps_interrupted": 0,
                "total_retries": 0,
                "total_candidate_structures": 0,
                "total_selected_structures": 0,
                "total_labeled_structures": 0,
                "total_discarded_structures": 0,
                "total_training_structures_added": 0,
                "manual_interrupt_count": 0,
                "auto_resume_count": 0,
                "total_gpu_time_sec": 0.0,
            }
        )
        if self.run_summary_path.exists():
            try:
                existing = read_json(self.run_summary_path)
                if isinstance(existing, dict):
                    self.run_summary.update(existing)
                    self.run_summary["run_name"] = run_name
                    self.run_summary["workflow_version"] = workflow_version
            except Exception as exc:
                self._warn(f"Failed to load existing run summary for resume: {exc}")
        self.step_contexts: dict[str, StepContext] = {}
        self.round_contexts: dict[int, RoundContext] = {}

    def _warn(self, message: str) -> None:
        warnings.warn(f"[metrics] {message}", RuntimeWarning, stacklevel=2)

    def _write_csv_row(self, path: Path, fields: list[str], row: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = path.exists()
            with open(path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if not file_exists or path.stat().st_size == 0:
                    writer.writeheader()
                payload = {field: row.get(field, "") for field in fields}
                writer.writerow(payload)
        except Exception as exc:
            self._warn(f"Failed to write CSV row to {path}: {exc}")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            self._warn(f"Failed to write JSON to {path}: {exc}")

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._warn(f"Failed to append JSONL to {path}: {exc}")

    def _flush_run_summary(self) -> None:
        summary = dict(self.run_summary)
        summary["time_per_labeled_structure"] = _safe_div(
            summary.get("total_wall_time_sec"), summary.get("total_labeled_structures")
        )
        summary["time_per_selected_structure"] = _safe_div(
            summary.get("total_wall_time_sec"), summary.get("total_selected_structures")
        )
        summary["gpu_time_per_labeled_structure"] = _safe_div(
            summary.get("total_gpu_time_sec"), summary.get("total_labeled_structures")
        )
        summary["selection_precision"] = _safe_div(
            summary.get("total_selected_structures"), summary.get("total_candidate_structures")
        )
        summary["label_efficiency"] = _safe_div(
            summary.get("total_training_structures_added"), summary.get("total_labeled_structures")
        )
        self._write_json(self.run_summary_path, summary)
        self._write_csv_row(self.run_summary_csv_path, RUN_SUMMARY_FIELDS, summary)

    def init_run(self, workflow_config: dict[str, Any], env_info: dict[str, Any] | None = None) -> None:
        if not self.run_summary.get("start_time"):
            self.run_summary["start_time"] = utc_now_iso()
        self.run_summary["status"] = "running"
        snapshot = {
            "workflow_config": workflow_config,
            "env_info": env_info or {},
            "hostname": socket.gethostname(),
            "python_version": sys.version,
        }
        self._write_json(self.config_snapshot_path, snapshot)
        self._flush_run_summary()
        self.log_event("workflow_started", payload={"workflow_version": self.workflow_version})

    def finalize_run(self, status: str, summary_updates: dict[str, Any] | None = None) -> None:
        self.run_summary["status"] = status
        self.run_summary["end_time"] = utc_now_iso()
        self.run_summary.update(summary_updates or {})
        start_time = self.run_summary.get("start_time")
        if start_time:
            try:
                start_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(self.run_summary["end_time"]).replace("Z", "+00:00"))
                self.run_summary["total_wall_time_sec"] = round((end_dt - start_dt).total_seconds(), 6)
            except Exception:
                pass
        self._flush_run_summary()
        event_type = {
            "SUCCESS": "workflow_finished",
            "FAILED": "workflow_failed",
            "INTERRUPTED": "workflow_interrupted",
        }.get(status, "workflow_finished")
        self.log_event(event_type, status=status)

    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        round_idx: int | None = None,
        step_id: str | None = None,
        status: str | None = None,
    ) -> None:
        event = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "run_name": self.run_name,
            "round_idx": round_idx,
            "step_id": step_id,
            "status": status,
            "payload": payload or {},
        }
        self._append_jsonl(self.event_log_path, event)

    def start_round(self, round_idx: int, payload: dict[str, Any] | None = None) -> None:
        import time

        ctx = RoundContext(round_idx=round_idx, start_time=utc_now_iso(), start_perf=time.perf_counter(), payload=payload or {})
        self.round_contexts[round_idx] = ctx
        self.log_event("round_started", payload=payload, round_idx=round_idx, status="RUNNING")

    def finish_round(self, round_idx: int, status: str, updates: dict[str, Any] | None = None) -> None:
        import time

        ctx = self.round_contexts.get(round_idx)
        row = {field: "" for field in ROUND_FIELDS}
        row["run_name"] = self.run_name
        row["round_idx"] = round_idx
        row["status"] = status
        if ctx is not None:
            row["start_time"] = ctx.start_time
            row["end_time"] = utc_now_iso()
            row["wall_time_sec"] = round(time.perf_counter() - ctx.start_perf, 6)
            row.update(ctx.payload)
        row.update(updates or {})
        row["selection_ratio"] = _safe_div(row.get("selected_structures"), row.get("candidate_structures"))
        row["label_ratio"] = _safe_div(row.get("labeled_structures"), row.get("selected_structures"))
        row["discard_ratio"] = _safe_div(row.get("discarded_structures"), row.get("candidate_structures"))
        row["delta_rmse_e"] = (
            _safe_float(row.get("rmse_e_before")) - _safe_float(row.get("rmse_e_after"))
            if _safe_float(row.get("rmse_e_before")) is not None and _safe_float(row.get("rmse_e_after")) is not None
            else None
        )
        row["delta_rmse_f"] = (
            _safe_float(row.get("rmse_f_before")) - _safe_float(row.get("rmse_f_after"))
            if _safe_float(row.get("rmse_f_before")) is not None and _safe_float(row.get("rmse_f_after")) is not None
            else None
        )
        row["delta_rmse_stress"] = (
            _safe_float(row.get("rmse_stress_before")) - _safe_float(row.get("rmse_stress_after"))
            if _safe_float(row.get("rmse_stress_before")) is not None and _safe_float(row.get("rmse_stress_after")) is not None
            else None
        )
        row["rmse_f_gain_per_label"] = _safe_div(row.get("delta_rmse_f"), row.get("labeled_structures"))
        row["rmse_f_gain_per_hour"] = _safe_div(row.get("delta_rmse_f"), _safe_div(row.get("wall_time_sec"), 3600))
        self._write_csv_row(self.round_metrics_path, ROUND_FIELDS, row)
        if status == "SUCCESS":
            self.run_summary["total_rounds_completed"] = int(self.run_summary.get("total_rounds_completed", 0) or 0) + 1
        for key in ("candidate_structures", "selected_structures", "labeled_structures", "discarded_structures", "new_train_structures"):
            summary_key = {
                "candidate_structures": "total_candidate_structures",
                "selected_structures": "total_selected_structures",
                "labeled_structures": "total_labeled_structures",
                "discarded_structures": "total_discarded_structures",
                "new_train_structures": "total_training_structures_added",
            }[key]
            value = row.get(key)
            if value not in ("", None):
                self.run_summary[summary_key] = int(self.run_summary.get(summary_key, 0) or 0) + int(value)
        self._flush_run_summary()
        self.log_event("round_finished", payload=row, round_idx=round_idx, status=status)

    def start_step(self, step_id: str, step_name: str, round_idx: int, payload: dict[str, Any] | None = None) -> None:
        import time

        payload = dict(payload or {})
        attempt = int(payload.get("attempt", 1))
        ctx = StepContext(
            step_id=step_id,
            step_name=step_name,
            round_idx=round_idx,
            attempt=attempt,
            start_time=utc_now_iso(),
            start_perf=time.perf_counter(),
            payload=payload,
        )
        self.step_contexts[step_id] = ctx
        self.log_event("step_started", payload=payload, round_idx=round_idx, step_id=step_id, status="RUNNING")

    def finish_step(self, step_id: str, status: str, updates: dict[str, Any] | None = None) -> None:
        import time

        ctx = self.step_contexts.get(step_id)
        row = {field: "" for field in STEP_FIELDS}
        if ctx is not None:
            row["run_name"] = self.run_name
            row["round_idx"] = ctx.round_idx
            row["step_id"] = ctx.step_id
            row["step_name"] = ctx.step_name
            row["attempt"] = ctx.attempt
            row["start_time"] = ctx.start_time
            row["end_time"] = utc_now_iso()
            row["wall_time_sec"] = round(time.perf_counter() - ctx.start_perf, 6)
            row.update(ctx.payload)
        row["status"] = status
        row.update(updates or {})
        self._write_csv_row(self.step_metrics_path, STEP_FIELDS, row)
        if status == "SUCCESS":
            self.run_summary["total_steps_completed"] = int(self.run_summary.get("total_steps_completed", 0) or 0) + 1
        elif status == "FAILED":
            self.run_summary["total_steps_failed"] = int(self.run_summary.get("total_steps_failed", 0) or 0) + 1
        elif status == "INTERRUPTED":
            self.run_summary["total_steps_interrupted"] = int(self.run_summary.get("total_steps_interrupted", 0) or 0) + 1
            self.run_summary["manual_interrupt_count"] = int(self.run_summary.get("manual_interrupt_count", 0) or 0) + 1
        attempt = row.get("attempt")
        if attempt not in ("", None):
            self.run_summary["total_retries"] = int(self.run_summary.get("total_retries", 0) or 0) + max(0, int(attempt) - 1)
        for key in ("best_rmse_e", "best_rmse_f", "best_rmse_stress"):
            run_key = {
                "best_rmse_e": "best_eval_rmse_e_mev_per_atom",
                "best_rmse_f": "best_eval_rmse_f_mev_per_a",
                "best_rmse_stress": "best_eval_rmse_stress_mev_per_a3",
            }[key]
            value = _safe_float(row.get(key))
            current = _safe_float(self.run_summary.get(run_key))
            if value is not None and (current is None or value < current):
                self.run_summary[run_key] = value
        for key in ("gpu_time_sec",):
            value = _safe_float(row.get(key))
            if value is not None:
                self.run_summary["total_gpu_time_sec"] = round(float(self.run_summary.get("total_gpu_time_sec", 0.0) or 0.0) + value, 6)
        self._flush_run_summary()
        event_type = {
            "SUCCESS": "step_finished",
            "FAILED": "step_failed",
            "INTERRUPTED": "step_interrupted",
        }.get(status, "step_finished")
        self.log_event(event_type, payload=row, round_idx=row.get("round_idx"), step_id=step_id, status=status)

    def log_training_metric(self, row: dict[str, Any]) -> None:
        payload = {field: row.get(field, "") for field in TRAINING_FIELDS}
        payload.setdefault("run_name", self.run_name)
        self._write_csv_row(self.training_metrics_path, TRAINING_FIELDS, payload)
        self.log_event("training_epoch_logged", payload=payload, round_idx=payload.get("round_idx"), step_id=payload.get("step_id"))

    def log_sampling_metric(self, row: dict[str, Any]) -> None:
        payload = {field: row.get(field, "") for field in SAMPLING_FIELDS}
        payload.setdefault("run_name", self.run_name)
        self._write_csv_row(self.sampling_metrics_path, SAMPLING_FIELDS, payload)
        self.log_event("sampling_finished", payload=payload, round_idx=payload.get("round_idx"), step_id=payload.get("step_id"), status=payload.get("status"))

    def log_labeling_metric(self, row: dict[str, Any]) -> None:
        payload = {field: row.get(field, "") for field in LABELING_FIELDS}
        payload.setdefault("run_name", self.run_name)
        self._write_csv_row(self.labeling_metrics_path, LABELING_FIELDS, payload)
        self.log_event("labeling_finished", payload=payload, round_idx=payload.get("round_idx"), step_id=payload.get("step_id"), status=payload.get("status"))

    def log_resource_metric(self, row: dict[str, Any]) -> None:
        payload = {field: row.get(field, "") for field in RESOURCE_FIELDS}
        payload.setdefault("run_name", self.run_name)
        self._write_csv_row(self.resource_metrics_path, RESOURCE_FIELDS, payload)

    def update_run_summary(self, updates: dict[str, Any]) -> None:
        self.run_summary.update(updates or {})
        self._flush_run_summary()

    def note_resume_triggered(self, payload: dict[str, Any] | None = None) -> None:
        self.run_summary["auto_resume_count"] = int(self.run_summary.get("auto_resume_count", 0) or 0) + 1
        self._flush_run_summary()
        self.log_event("resume_triggered", payload=payload or {})
