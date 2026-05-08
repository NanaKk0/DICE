from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def compute_min_pair_distance(atoms: Atoms) -> float:
    natoms = len(atoms)
    if natoms < 2:
        return float("inf")

    distances = atoms.get_all_distances(mic=True)
    iu = np.triu_indices(natoms, k=1)
    if iu[0].size == 0:
        return float("inf")
    return float(np.min(distances[iu]))


def compute_min_pair_distance_scaled(atoms: Atoms) -> tuple[float, float]:
    natoms = len(atoms)
    if natoms < 2:
        return float("inf"), float("inf")

    distances = atoms.get_all_distances(mic=True)
    numbers = atoms.get_atomic_numbers()
    radii = np.array([covalent_radii[z] for z in numbers], dtype=float)
    radius_sums = radii[:, None] + radii[None, :]

    iu = np.triu_indices(natoms, k=1)
    pair_distances = distances[iu]
    pair_radius_sums = radius_sums[iu]
    valid = np.isfinite(pair_distances) & np.isfinite(pair_radius_sums) & (pair_radius_sums > 0)
    if not np.any(valid):
        return float("inf"), float("inf")

    pair_distances = pair_distances[valid]
    pair_radius_sums = pair_radius_sums[valid]
    scaled = pair_distances / pair_radius_sums
    min_pos = int(np.argmin(scaled))
    return float(scaled[min_pos]), float(pair_distances[min_pos])


def compute_rmsd(atoms_a: Atoms, atoms_b: Atoms | None) -> float:
    if atoms_b is None or len(atoms_a) != len(atoms_b):
        return float("inf")
    if atoms_a.get_chemical_symbols() != atoms_b.get_chemical_symbols():
        return float("inf")

    pos_a = atoms_a.get_positions()
    pos_b = atoms_b.get_positions()
    return float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=1))))


def key_geometry_changed_enough(
    atoms: Atoms,
    ref_atoms: Atoms | None,
    bond_thr: float,
    height_thr: float,
) -> bool:
    if ref_atoms is None or len(atoms) != len(ref_atoms):
        return False

    symbols = atoms.get_chemical_symbols()
    if symbols != ref_atoms.get_chemical_symbols():
        return True

    ads_idx = [i for i, sym in enumerate(symbols) if sym != "Pt"]
    if not ads_idx:
        return False

    pos = atoms.get_positions()
    ref = ref_atoms.get_positions()

    height_delta = np.max(np.abs(pos[ads_idx, 2] - ref[ads_idx, 2]))
    if height_delta > height_thr:
        return True

    if len(ads_idx) >= 2:
        curr = atoms.get_all_distances(mic=True)[np.ix_(ads_idx, ads_idx)]
        prev = ref_atoms.get_all_distances(mic=True)[np.ix_(ads_idx, ads_idx)]
        if np.max(np.abs(curr - prev)) > bond_thr:
            return True

    return False


def estimate_novelty_threshold(
    ref_features: list[np.ndarray],
    novelty_cfg: dict[str, Any],
) -> float:
    floor = float(novelty_cfg.get("novelty_threshold_floor", 0.10))
    if len(ref_features) < 2:
        return floor

    dmins = []
    for i, feat in enumerate(ref_features):
        best = np.inf
        for j, other in enumerate(ref_features):
            if i == j:
                continue
            dist = np.linalg.norm(feat - other)
            if dist < best:
                best = dist
        if np.isfinite(best):
            dmins.append(float(best))

    if not dmins:
        return floor

    quantile = float(novelty_cfg.get("novelty_quantile", 0.80))
    return max(floor, float(np.quantile(dmins, quantile)))


@dataclass
class NoveltyTracker:
    ref_features: list[np.ndarray]

    def nearest_distance(self, feat: np.ndarray) -> float:
        if not self.ref_features:
            return float("inf")
        return float(min(np.linalg.norm(feat - x) for x in self.ref_features))

    def add_reference(self, feat: np.ndarray) -> None:
        self.ref_features.append(feat)


def get_consistent_energy_forces(atoms: Atoms) -> tuple[float, np.ndarray, dict[str, Any]]:
    calc = atoms.calc
    if calc is None:
        raise RuntimeError("Atoms calculator is not set")

    calc.calculate(atoms)
    results = getattr(calc, "results", {})

    if "energy" not in results and "free_energy" not in results:
        raise RuntimeError(f"Calculator {type(calc).__name__} did not return energy")
    if "forces" not in results:
        raise RuntimeError(f"Calculator {type(calc).__name__} did not return forces")

    energy_raw = results.get("energy")
    free_energy_raw = results.get("free_energy")
    energy = float(energy_raw) if energy_raw is not None else float("nan")
    free_energy = float(free_energy_raw) if free_energy_raw is not None else float("nan")
    forces = np.asarray(results["forces"], dtype=float)

    if forces.ndim != 2 or forces.shape[1] != 3:
        raise RuntimeError(f"Invalid forces shape from calculator {type(calc).__name__}: {forces.shape}")
    if not np.all(np.isfinite(forces)):
        raise RuntimeError(f"Non-finite forces from calculator {type(calc).__name__}")

    selected_energy_key = "energy"
    selected_energy = energy
    disagreement_thr = 100.0

    if np.isfinite(energy) and np.isfinite(free_energy):
        if abs(energy - free_energy) > disagreement_thr:
            selected_energy_key = "free_energy"
            selected_energy = free_energy
    elif np.isfinite(free_energy):
        selected_energy_key = "free_energy"
        selected_energy = free_energy
    elif not np.isfinite(energy):
        raise RuntimeError(f"Non-finite energy from calculator {type(calc).__name__}: {energy}")

    calc_meta = {
        "calc_class": type(calc).__name__,
        "calc_id": id(calc),
        "calc_results_keys": sorted(results.keys()),
        "energy_raw": energy if np.isfinite(energy) else None,
        "free_energy_raw": free_energy if np.isfinite(free_energy) else None,
        "selected_energy_key": selected_energy_key,
    }
    return selected_energy, forces, calc_meta


@dataclass
class CommitteeTrajectoryMonitor:
    hard_stop_cfg: dict[str, Any]
    save_policy_cfg: dict[str, Any]
    early_stop_cfg: dict[str, Any]
    novelty_threshold: float
    saved_frames: list[Atoms] = field(default_factory=list)
    novelty_hist: list[float] = field(default_factory=list)
    eval_rmsd_hist: list[float] = field(default_factory=list)
    last_eval_atoms: Atoms | None = None
    last_save_step: int = -10**9
    initial_eatom: float | None = None

    def check_hard_stop(self, atoms: Atoms) -> tuple[bool, str | None, dict[str, float | None]]:
        min_pair_scale, min_pair_distance = compute_min_pair_distance_scaled(atoms)
        energy, forces, _ = get_consistent_energy_forces(atoms)
        max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(atoms) else 0.0
        eatom = energy / max(1, len(atoms))

        if self.initial_eatom is None:
            self.initial_eatom = eatom

        metrics = {
            "min_pair_scale": float(min_pair_scale),
            "min_pair_distance": float(min_pair_distance),
            "max_force": float(max_force),
            "potential_energy": float(energy),
            "energy_per_atom": float(eatom),
            "energy_per_atom_jump": float(eatom - self.initial_eatom),
        }

        hard_min_pair = self.hard_stop_cfg.get("hard_min_pair_distance")
        if hard_min_pair is not None and min_pair_distance < float(hard_min_pair):
            return True, "hard_stop_pair_too_close", metrics

        if max_force > float(self.hard_stop_cfg.get("max_force_hard", 20.0)):
            return True, "hard_stop_force_explosion", metrics

        if (eatom - self.initial_eatom) > float(self.hard_stop_cfg.get("max_eatom_jump", 10.0)):
            return True, "hard_stop_energy_explosion", metrics

        return False, None, metrics

    def evaluate_online_save(
        self,
        atoms: Atoms,
        feat: np.ndarray,
        novelty: float,
        step: int,
    ) -> tuple[bool, list[str], dict[str, float]]:
        rmsd_to_last_saved = compute_rmsd(atoms, self.saved_frames[-1] if self.saved_frames else None)
        rmsd_to_prev_eval = compute_rmsd(atoms, self.last_eval_atoms)
        geometry_changed = key_geometry_changed_enough(
            atoms,
            self.saved_frames[-1] if self.saved_frames else None,
            bond_thr=float(self.save_policy_cfg.get("bond_change_save", 0.08)),
            height_thr=float(self.save_policy_cfg.get("height_change_save", 0.1)),
        )

        reasons: list[str] = []
        if novelty > self.novelty_threshold:
            reasons.append("novelty")
        if rmsd_to_last_saved > float(self.save_policy_cfg.get("rmsd_save_threshold", 0.1)):
            reasons.append("rmsd")
        if geometry_changed:
            reasons.append("geometry")

        if (step - self.last_save_step) < int(self.save_policy_cfg.get("save_interval", 20)):
            reasons = []
        if len(self.saved_frames) >= int(self.save_policy_cfg.get("max_saved_per_traj", 20)):
            reasons = []

        if reasons:
            self.saved_frames.append(atoms.copy())
            self.last_save_step = step
            if bool(self.save_policy_cfg.get("dynamic_reference", True)):
                # Caller controls whether to update tracker; this branch is only used for state symmetry.
                pass

        self.novelty_hist.append(float(novelty))
        self.eval_rmsd_hist.append(float(rmsd_to_prev_eval))
        self.last_eval_atoms = atoms.copy()

        return bool(reasons), reasons, {
            "novelty_score": float(novelty),
            "rmsd_to_last_saved": float(rmsd_to_last_saved),
            "rmsd_to_prev_eval": float(rmsd_to_prev_eval),
            "geometry_changed": float(1.0 if geometry_changed else 0.0),
        }

    def should_stop(self, prod_step: int) -> tuple[bool, str | None, dict[str, float | bool | None]]:
        debug = {
            "prod_step": int(prod_step),
            "cond_struct": False,
            "cond_novelty": False,
            "mean_rmsd_recent": None,
            "max_rmsd_recent": None,
            "novelty_low_frac_recent": None,
        }

        if prod_step < int(self.early_stop_cfg.get("min_prod_steps", 0)):
            return False, None, debug

        cond_struct = False
        cond_novelty = False

        w_struct = int(self.early_stop_cfg.get("window_struct", 20))
        if len(self.eval_rmsd_hist) >= w_struct:
            xs = np.asarray(self.eval_rmsd_hist[-w_struct:], dtype=float)
            debug["mean_rmsd_recent"] = float(xs.mean())
            debug["max_rmsd_recent"] = float(xs.max())
            if (
                xs.mean() < float(self.early_stop_cfg.get("mean_rmsd_stop", 0.08))
                and xs.max() < float(self.early_stop_cfg.get("max_rmsd_stop", 0.15))
            ):
                cond_struct = True

        w_novelty = int(self.early_stop_cfg.get("window_novelty", 20))
        if len(self.novelty_hist) >= w_novelty:
            xs = np.asarray(self.novelty_hist[-w_novelty:], dtype=float)
            low_frac = float(np.mean(xs < self.novelty_threshold))
            debug["novelty_low_frac_recent"] = low_frac
            if low_frac >= float(self.early_stop_cfg.get("novelty_low_frac", 0.8)):
                cond_novelty = True

        debug["cond_struct"] = cond_struct
        debug["cond_novelty"] = cond_novelty

        if cond_struct and cond_novelty:
            return True, "info_gain_saturated", debug

        return False, None, debug
