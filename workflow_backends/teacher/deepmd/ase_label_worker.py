from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from workflow_backends.teacher.deepmd.evaluator import DeepMDPythonEvaluator as _DeepMDPythonEvaluator
from workflow_core.structures.canonical_identity import copy_atoms_preserving_singlepoint_results
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


class DeepMDPythonEvaluator(_DeepMDPythonEvaluator):
    """Family-free lifecycle evaluator surface over the shared native DeepPot loader."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int | None = None,
        deepmd_auto_batch_size: int | str | None = None,
        type_map: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_path,
            device=device,
            batch_size=batch_size,
            deepmd_auto_batch_size=deepmd_auto_batch_size,
            type_map=type_map,
        )
        self.model_path = str(Path(model_path).absolute())


def label_with_ase_models(
    models: Sequence[str | Path],
    frames: list[Atoms],
    *,
    type_map: Sequence[str],
    device: str,
    evaluator_factory: Callable[..., Any] = DeepMDPythonEvaluator,
) -> tuple[list[Atoms], list[float]]:
    """Label lifecycle frames through DeePMD's Python/ASE-facing inference route only."""
    paths = tuple(Path(model).absolute() for model in models)
    if not paths:
        raise ValueError("DeepMD lifecycle ASE label worker requires at least one model path")
    normalized_type_map = [str(item) for item in type_map]
    if not normalized_type_map or any(not item for item in normalized_type_map):
        raise ValueError("DeepMD lifecycle ASE label worker requires a non-empty type_map")
    evaluators = [
        evaluator_factory(
            path,
            device=device,
            type_map=normalized_type_map,
        )
        for path in paths
    ]
    for evaluator in evaluators:
        preflight = getattr(evaluator, "preflight", None)
        if not callable(preflight):
            raise ValueError("DeepMD lifecycle ASE label worker evaluator must expose preflight")
        preflight()

    labeled: list[Atoms] = []
    uncertainties: list[float] = []
    for frame_index, atoms in enumerate(frames):
        payloads = [dict(evaluator.evaluate(atoms)) for evaluator in evaluators]
        result, uncertainty = _aggregate_payloads(payloads, atom_count=len(atoms), frame_index=frame_index)
        item = copy_atoms_preserving_singlepoint_results(atoms)
        item.calc = SinglePointCalculator(item, **result)
        item.info["force_uncertainty"] = uncertainty
        item.info["force_std_max"] = uncertainty
        labeled.append(item)
        uncertainties.append(uncertainty)
    return labeled, uncertainties


def _aggregate_payloads(payloads: list[dict[str, Any]], *, atom_count: int, frame_index: int) -> tuple[dict[str, Any], float]:
    if not payloads:
        raise ValueError("DeepMD lifecycle ASE label worker received no model payloads")
    try:
        energies = np.asarray([payload["energy"] for payload in payloads], dtype=float)
        forces = np.asarray([payload["forces"] for payload in payloads], dtype=float)
    except KeyError as exc:
        raise ValueError(f"DeepMD lifecycle ASE label worker frame {frame_index} is missing {exc.args[0]}") from exc
    if forces.shape != (len(payloads), atom_count, 3):
        raise ValueError(f"DeepMD lifecycle ASE label worker frame {frame_index} returned invalid force shape {forces.shape!r}")
    if not np.isfinite(energies).all() or not np.isfinite(forces).all():
        raise ValueError(f"DeepMD lifecycle ASE label worker frame {frame_index} returned non-finite labels")

    result: dict[str, Any] = {"energy": float(energies.mean()), "forces": forces.mean(axis=0)}
    stresses = [payload.get("stress") for payload in payloads]
    if all(stress is not None for stress in stresses):
        stress_array = np.asarray(stresses, dtype=float)
        if not np.isfinite(stress_array).all():
            raise ValueError(f"DeepMD lifecycle ASE label worker frame {frame_index} returned non-finite stress")
        result["stress"] = stress_array.mean(axis=0)

    component_std = np.std(forces, axis=0)
    uncertainty = float(np.linalg.norm(component_std, axis=1).max(initial=0.0))
    return result, uncertainty


def worker_main(argv: list[str] | None = None) -> int:
    from workflow_runtime.lifecycle.progress.worker_observations import optional_observation_writer

    parser = argparse.ArgumentParser(description="Run lifecycle DeePMD teacher labels through the ASE/Python inference route.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncertainties", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--observation-channel")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    models = config.get("model_paths")
    if not isinstance(models, list) or not models:
        raise ValueError("DeepMD lifecycle ASE label worker config requires non-empty model_paths")
    frames = read_extxyz_structures(args.input, allow_empty=True)
    labeled, uncertainties = label_with_ase_models(
        models,
        frames,
        type_map=config.get("type_map", []),
        device=str(config.get("device", "cuda")),
    )
    write_extxyz_structures(args.output, labeled, allow_empty=True)
    Path(args.uncertainties).write_text(json.dumps(uncertainties), encoding="utf-8")
    writer = optional_observation_writer(args.observation_channel)
    if writer is not None:
        for start in range(0, len(frames), 64):
            chunk = frames[start : start + 64]
            writer.label_chunk(
                tuple(str(frame.info.get("structure_id", "")) for frame in chunk),
                tuple(float(value) for value in uncertainties[start : start + 64]),
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(worker_main())
