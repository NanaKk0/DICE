from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

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


def parse_mace_training_log(
    log_path: Path,
    step_id: str,
    round_id: int,
    metrics: Any,
    cfg: dict[str, Any],
    paths,
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

    return {
        "epoch_count": len(rows),
        "best_rmse_e": eval_test_metrics.get("best_rmse_e", best_rmse_e),
        "best_rmse_f": eval_test_metrics.get("best_rmse_f", best_rmse_f),
        "best_rmse_stress": eval_test_metrics.get("best_rmse_stress", best_rmse_s),
        "input_structure_count": train_size,
        "output_structure_count": train_size,
        "train_batch_count": len(rows),
        "valid_batch_count": len(rows),
    }
