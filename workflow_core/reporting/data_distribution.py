from __future__ import annotations

from pathlib import Path
from typing import Any


def load_distribution_plot_config(paths, config_path: str | Path | None = None) -> dict[str, Any]:
    del paths, config_path
    return {"enabled": False}


def plot_outer_round_dft_increment(paths, outer_round: int, config_path: str | Path | None = None) -> None:
    del paths, outer_round, config_path
    return None


def plot_student_round_increment(paths, outer_round: int, student_round: int, config_path: str | Path | None = None) -> None:
    del paths, outer_round, student_round, config_path
    return None
