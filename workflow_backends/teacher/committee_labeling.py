from __future__ import annotations

from typing import Any, Callable, Protocol

from workflow_backends.committee_evaluation import canonical_committee_evaluation_result
from workflow_backends.teacher.label_result import TeacherLabelResult


class TeacherStructureEvaluator(Protocol):
    def evaluate(self, atoms: Any) -> dict[str, Any]:
        ...


def ensure_stress_matrix(stress: Any) -> Any:
    import numpy as np

    arr = np.asarray(stress, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (6,):
        xx, yy, zz, yz, xz, xy = arr.tolist()
        return np.array(
            [
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported stress shape: {arr.shape!r}")


def compute_force_uncertainty(forces: Any) -> float:
    import numpy as np

    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(
            "force uncertainty requires forces shaped as "
            f"(n_committee_models, n_atoms, 3); got {forces.shape!r}"
        )
    f_mean = forces.mean(axis=0)
    diff = forces - f_mean
    sq = np.sum(diff**2, axis=2)
    mean_sq = np.mean(sq, axis=0)
    atom_std = np.sqrt(mean_sq)
    return float(np.max(atom_std))


def compute_force_uncertainty_metrics(forces: Any) -> dict[str, Any]:
    import numpy as np

    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(
            "force uncertainty requires forces shaped as "
            f"(n_committee_models, n_atoms, 3); got {forces.shape!r}"
        )
    f_mean = forces.mean(axis=0)
    diff = forces - f_mean
    sq = np.sum(diff**2, axis=2)
    mean_sq = np.mean(sq, axis=0)
    atom_std = np.sqrt(mean_sq)
    return {
        "force_std_max": float(np.max(atom_std)) if atom_std.size else 0.0,
        "force_std_avg": float(np.mean(atom_std)) if atom_std.size else 0.0,
        "force_std_p90": float(np.percentile(atom_std, 90)) if atom_std.size else 0.0,
        "per_atom_force_std": atom_std.astype(float).tolist(),
    }


def evaluate_structure(evaluator: Any, atoms: Any) -> dict[str, Any]:
    import numpy as np

    if hasattr(evaluator, "evaluate"):
        result = evaluator.evaluate(atoms)
        if not isinstance(result, dict):
            raise TypeError("teacher evaluator evaluate() must return a mapping")
        if "stress" not in result:
            result = dict(result)
            result["stress"] = np.zeros(6)
            result["_stress_fallback"] = True
        return result

    atoms.calc = evaluator
    payload: dict[str, Any] = {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(),
    }
    try:
        payload["stress"] = atoms.get_stress()
    except Exception:
        payload["stress"] = np.zeros(6)
        payload["_stress_fallback"] = True
    return payload


def label_with_committee(
    atoms_list: list[Any],
    evaluators: list[Any],
    *,
    progress: bool = True,
    desc: str = "Committee labeling",
    progress_callback: Callable[[int, int], None] | None = None,
) -> TeacherLabelResult:
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator
    from tqdm import tqdm

    if not evaluators:
        raise ValueError("teacher committee labeling requires at least one evaluator")

    labeled = []
    uncertainties = []
    iterable = tqdm(atoms_list, desc=desc) if progress else atoms_list

    total = len(atoms_list)
    for atoms in iterable:
        energies = []
        forces = []
        stresses = []
        stress_fallback_count = 0

        for evaluator in evaluators:
            result = evaluate_structure(evaluator, atoms)
            energies.append(result["energy"])
            forces.append(result["forces"])
            stresses.append(result.get("stress", np.zeros(6)))
            if result.get("_stress_fallback"):
                stress_fallback_count += 1

        energies = np.array(energies)
        forces = np.array(forces)
        stresses = np.array([ensure_stress_matrix(stress) for stress in stresses], dtype=float)

        energy_mean = energies.mean()
        force_mean = forces.mean(axis=0)
        stress_mean = np.clip(stresses.mean(axis=0), -100, 100)
        metrics = compute_force_uncertainty_metrics(forces)
        uncertainty = float(metrics["force_std_max"])

        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy_mean,
            forces=force_mean,
            stress=stress_mean,
        )
        atoms.info["force_uncertainty"] = float(uncertainty)
        atoms.info["force_std_max"] = float(metrics["force_std_max"])
        atoms.info["force_std_avg"] = float(metrics["force_std_avg"])
        atoms.info["force_std_p90"] = float(metrics["force_std_p90"])
        if stress_fallback_count:
            atoms.info["teacher_stress_fallback_count"] = int(stress_fallback_count)

        labeled.append(atoms)
        uncertainties.append(uncertainty)
        if progress_callback is not None:
            progress_callback(len(labeled), total)

    return TeacherLabelResult(labeled_structures=labeled, uncertainties=np.array(uncertainties))


def score_structures_with_committee(
    atoms_list: list[Any],
    evaluators: list[Any],
    *,
    backend: str,
    progress: bool = True,
    desc: str = "Committee scoring",
) -> dict[str, Any]:
    import numpy as np
    from tqdm import tqdm

    if not evaluators:
        raise ValueError("teacher committee scoring requires at least one evaluator")

    frame_metrics = []
    iterable = tqdm(atoms_list, desc=desc) if progress else atoms_list
    stress_fallback_seen = False

    for frame_index, atoms in enumerate(iterable):
        energies = []
        forces = []
        stresses = []
        stress_fallback_count = 0

        for evaluator in evaluators:
            result = evaluate_structure(evaluator, atoms)
            energies.append(result["energy"])
            forces.append(result["forces"])
            stresses.append(result.get("stress", np.zeros(6)))
            if result.get("_stress_fallback"):
                stress_fallback_count += 1

        energies = np.array(energies)
        forces = np.array(forces)
        stresses = np.array([ensure_stress_matrix(stress) for stress in stresses], dtype=float)
        metrics = compute_force_uncertainty_metrics(forces)
        metrics.update(
            {
                "frame_index": int(frame_index),
                "natoms": int(len(atoms)),
                "energy_mean": float(energies.mean()),
                "forces_mean": forces.mean(axis=0).astype(float).tolist(),
                "stress_mean": np.clip(stresses.mean(axis=0), -100, 100).astype(float).tolist(),
            }
        )
        if stress_fallback_count:
            stress_fallback_seen = True
            metrics["stress_fallback_count"] = int(stress_fallback_count)
        frame_metrics.append(metrics)

    return canonical_committee_evaluation_result(
        backend=backend,
        committee_size=len(evaluators),
        evaluation_mode="committee_mean",
        capabilities={
            "force_deviation": True,
            "committee_mean_labels": True,
            "mean_energy": True,
            "mean_forces": True,
            "mean_stress": not stress_fallback_seen,
        },
        frame_metrics=frame_metrics,
    )
