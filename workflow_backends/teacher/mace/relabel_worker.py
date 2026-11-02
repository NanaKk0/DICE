from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from workflow_backends.teacher.mace.label_operation import label_structures
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


def worker_main(argv: list[str] | None = None) -> int:
    from workflow_runtime.lifecycle.progress.worker_observations import optional_observation_writer

    parser = argparse.ArgumentParser(description="Run MACE teacher relabel in the selected Python runtime.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncertainties", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress-log")
    parser.add_argument("--observation-channel")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    atoms_list = read_extxyz_structures(args.input, allow_empty=True)
    progress_log = _progress_log_path(args.progress_log or config.get("relabel_progress_log_path"))
    _append_relabel_progress(progress_log, f"mace relabel worker started: frames={len(atoms_list)}")
    if atoms_list:
        if progress_log is not None:
            config["relabel_progress_log_path"] = str(progress_log)
        result = label_structures(
            config,
            atoms_list,
            progress_callback=_relabel_progress_callback(progress_log, len(atoms_list)),
        )
        labeled = result.labeled_structures
        uncertainties = np.asarray(result.uncertainties, dtype=float)
    else:
        labeled = []
        uncertainties = np.asarray([], dtype=float)

    write_extxyz_structures(args.output, labeled, allow_empty=True)
    Path(args.uncertainties).write_text(json.dumps([float(value) for value in uncertainties]), encoding="utf-8")
    writer = optional_observation_writer(args.observation_channel)
    if writer is not None:
        chunk_size = int(config.get("chunk_size_frames", 64))
        for start in range(0, len(atoms_list), chunk_size):
            chunk = atoms_list[start : start + chunk_size]
            writer.label_chunk(
                tuple(str(frame.info.get("structure_id", "")) for frame in chunk),
                tuple(float(value) for value in uncertainties[start : start + chunk_size]),
            )
    _append_relabel_progress(progress_log, f"mace relabel worker completed: frames={len(labeled)}")
    return 0


def _progress_log_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).resolve()


def _append_relabel_progress(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _relabel_progress_callback(progress_log: Path | None, total_frames: int) -> Callable[[int, int], None] | None:
    if progress_log is None:
        return None
    total = max(0, int(total_frames))
    interval = 1 if total <= 100 else max(1, total // 100)

    def _record(completed_frames: int, callback_total: int) -> None:
        completed = int(completed_frames)
        effective_total = max(total, int(callback_total))
        if completed == 1 or completed >= effective_total or completed % interval == 0:
            percent = 100.0 if effective_total <= 0 else 100.0 * float(completed) / float(effective_total)
            _append_relabel_progress(
                progress_log,
                f"mace relabel progress: {completed}/{effective_total} frames ({percent:.1f}%)",
            )

    return _record


if __name__ == "__main__":
    raise SystemExit(worker_main())
