from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from workflow_backends.teacher.label_result import TeacherLabelResult
from workflow_backends.teacher.committee_labeling import (
    compute_force_uncertainty_metrics,
    ensure_stress_matrix,
    label_with_committee,
)
from workflow_backends.teacher.mace.backend import build_mace_calculator_from_declared_models


VALID_COMMITTEE_EVALUATION_MODES = {"native", "separate"}
DEFAULT_RELABEL_CHUNK_SIZE_FRAMES = 50
ProgressCallback = Callable[[int, int], None]


def label_structures(
    config: dict[str, Any],
    frames: list[Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> TeacherLabelResult:
    """Label ordered frames with the MACE committee described by ``config``."""
    mode = _normalize_committee_evaluation_mode(config.get("committee_evaluation_mode"))
    config["chunk_size_frames"] = _normalize_chunk_size_frames(config.get("chunk_size_frames"))
    progress_log = _relabel_progress_log_path(config)

    if mode == "separate":
        calculators = _load_mace_committee(config)
        _append_relabel_progress(
            progress_log,
            f"mace relabel models loaded: teacher_count={len(calculators)} mode=separate",
        )
        return _label_with_separate_mace_committee_chunked(
            frames,
            calculators,
            config,
            progress_callback=progress_callback,
        )

    calculator = _load_native_mace_committee(config)
    _append_relabel_progress(progress_log, "mace relabel models loaded: mode=native")
    return _label_with_native_mace_committee_chunked(
        frames,
        calculator,
        config,
        progress_callback=progress_callback,
    )


def _resolved_mace_model_paths_for_relabel(config: dict[str, Any]) -> list[Path]:
    explicit = config.get("model_paths")
    if not isinstance(explicit, list) or not explicit:
        raise ValueError("MACE relabel model_paths must be a non-empty list")
    paths = [Path(value).resolve() for value in explicit]
    missing = [path for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise ValueError(f"MACE relabel model_paths contain missing/empty assets: {missing}")
    return paths


def _load_mace_committee(config: dict[str, Any]) -> list[Any]:
    model_paths = _resolved_mace_model_paths_for_relabel(config)
    calculators = []

    for model in model_paths:
        calculator = build_mace_calculator_from_declared_models(
            model_path=model,
            device=config.get("device", "cpu"),
            default_dtype=config.get("default_dtype", "float32"),
        )
        calculators.append(calculator)

    print("Loaded teacher models:", len(calculators))
    return calculators


def _load_native_mace_committee(config: dict[str, Any]) -> Any:
    model_paths = _resolved_mace_model_paths_for_relabel(config)
    calculator = build_mace_calculator_from_declared_models(
        model_paths=[str(path.resolve()) for path in model_paths],
        device=config.get("device", "cpu"),
        default_dtype=config.get("default_dtype", "float32"),
    )
    print("Loaded native teacher committee models:", len(model_paths))
    return calculator


def _normalize_committee_evaluation_mode(value: Any) -> str:
    mode = str(value or "native").strip().lower()
    if mode not in VALID_COMMITTEE_EVALUATION_MODES:
        raise ValueError(
            "Unsupported MACE committee_evaluation_mode="
            f"{value!r}. Expected one of: {sorted(VALID_COMMITTEE_EVALUATION_MODES)}"
        )
    return mode


def _normalize_chunk_size_frames(value: Any) -> int:
    if value in (None, ""):
        return DEFAULT_RELABEL_CHUNK_SIZE_FRAMES
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"MACE relabel chunk_size_frames must be positive; got {value!r}")
    return normalized


def _append_relabel_progress(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _relabel_progress_log_path(config: dict[str, Any]) -> Path | None:
    value = config.get("relabel_progress_log_path")
    if value in (None, ""):
        return None
    return Path(str(value)).resolve()


def _frame_chunks(frames: list[Any], chunk_size_frames: int) -> list[list[Any]]:
    if not frames:
        return []
    return [
        frames[index : index + chunk_size_frames]
        for index in range(0, len(frames), chunk_size_frames)
    ]


def _label_with_native_mace_committee(
    frames: list[Any],
    calculator: Any,
    *,
    progress_callback: ProgressCallback | None = None,
) -> TeacherLabelResult:
    labeled = []
    uncertainties = []
    total = len(frames)
    for atoms in frames:
        working = atoms.copy()
        working.calc = calculator
        energy = float(working.get_potential_energy())
        forces = np.asarray(working.get_forces(), dtype=float)
        stress = np.clip(
            np.asarray(ensure_stress_matrix(working.get_stress()), dtype=float),
            -100,
            100,
        )
        raw_results = dict(getattr(calculator, "results", {}) or {})
        if "forces_comm" not in raw_results:
            raise RuntimeError(
                "Native MACE committee relabel requires calculator results['forces_comm']; "
                "verify the installed MACE version supports MACECalculator(model_paths=[...]) "
                "or set committee_evaluation_mode='separate' in the explicit worker request "
                "for debug comparison."
            )
        metrics = compute_force_uncertainty_metrics(raw_results["forces_comm"])

        output = atoms.copy()
        output.calc = SinglePointCalculator(output, energy=energy, forces=forces, stress=stress)
        uncertainty = float(metrics["force_std_max"])
        output.info["force_uncertainty"] = uncertainty
        output.info["force_std_max"] = float(metrics["force_std_max"])
        output.info["force_std_avg"] = float(metrics["force_std_avg"])
        output.info["force_std_p90"] = float(metrics["force_std_p90"])
        labeled.append(output)
        uncertainties.append(uncertainty)
        if progress_callback is not None:
            progress_callback(len(labeled), total)

    return TeacherLabelResult(
        labeled_structures=labeled,
        uncertainties=np.asarray(uncertainties, dtype=float),
    )


def _label_with_native_mace_committee_chunked(
    frames: list[Any],
    calculator: Any,
    config: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> TeacherLabelResult:
    return _label_in_chunks(
        frames,
        config,
        mode="native",
        label_chunk=lambda chunk, callback: _label_with_native_mace_committee(
            chunk,
            calculator,
            progress_callback=callback,
        ),
        progress_callback=progress_callback,
    )


def _label_with_separate_mace_committee_chunked(
    frames: list[Any],
    calculators: list[Any],
    config: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> TeacherLabelResult:
    return _label_in_chunks(
        frames,
        config,
        mode="separate",
        label_chunk=lambda chunk, callback: label_with_committee(
            chunk,
            calculators,
            progress_callback=callback,
        ),
        progress_callback=progress_callback,
    )


def _label_in_chunks(
    frames: list[Any],
    config: dict[str, Any],
    *,
    mode: str,
    label_chunk: Callable[[list[Any], ProgressCallback | None], TeacherLabelResult],
    progress_callback: ProgressCallback | None,
) -> TeacherLabelResult:
    chunk_size = _normalize_chunk_size_frames(config.get("chunk_size_frames"))
    chunks = _frame_chunks(frames, chunk_size)
    progress_log = _relabel_progress_log_path(config)
    labeled: list[Any] = []
    uncertainties: list[float] = []
    total_chunks = len(chunks)

    for chunk_index, chunk in enumerate(chunks, start=1):
        completed_before_chunk = len(labeled)
        _append_relabel_progress(
            progress_log,
            f"mace relabel chunk {chunk_index}/{total_chunks}: frames={len(chunk)} mode={mode}",
        )

        def _chunk_progress(completed_in_chunk: int, _chunk_total: int) -> None:
            if progress_callback is not None:
                progress_callback(completed_before_chunk + int(completed_in_chunk), len(frames))

        result = label_chunk(
            chunk,
            _chunk_progress if progress_callback is not None else None,
        )
        labeled.extend(result.labeled_structures)
        uncertainties.extend(float(value) for value in result.uncertainties)

    return TeacherLabelResult(
        labeled_structures=labeled,
        uncertainties=np.asarray(uncertainties, dtype=float),
    )
