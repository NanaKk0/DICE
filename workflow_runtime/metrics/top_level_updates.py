from __future__ import annotations

from pathlib import Path
from typing import Any


class NullMetricsLogger:
    def __init__(self, run_name: str = "") -> None:
        self.run_name = run_name

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None

        return _noop


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


def collect_step_metric_updates(
    step_name: str,
    step_id: str,
    paths,
    round_id: int,
    cfg: dict[str, Any],
    metrics: Any,
) -> dict[str, Any]:
    del step_name, step_id, paths, round_id, cfg, metrics
    return {}
