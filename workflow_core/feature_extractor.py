from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms


@dataclass
class FeatureExtractor:
    cfg: dict[str, Any]

    def fit(self, atoms_list: list[Atoms]) -> "FeatureExtractor":
        return self

    def transform(self, atoms: Atoms) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, atoms_list: list[Atoms]) -> np.ndarray:
        self.fit(atoms_list)
        return np.asarray([self.transform(atoms) for atoms in atoms_list], dtype=float)


class GeometricFeatureExtractor(FeatureExtractor):
    def transform(self, atoms: Atoms) -> np.ndarray:
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        if len(atoms) == 0:
            return np.zeros(16, dtype=float)

        center = positions.mean(axis=0)
        radial = np.linalg.norm(positions - center, axis=1)

        ads_idx = [i for i, sym in enumerate(symbols) if sym != "Pt"]
        pt_idx = [i for i, sym in enumerate(symbols) if sym == "Pt"]
        ads_z = positions[ads_idx, 2] if ads_idx else np.array([0.0])

        feature = [
            float(len(atoms)),
            float(symbols.count("Pt")),
            float(symbols.count("C")),
            float(symbols.count("O")),
            float(symbols.count("H")),
            float(radial.mean()),
            float(radial.std()),
            float(np.ptp(positions[:, 2])),
            float(ads_z.mean()),
            float(ads_z.std()),
            _min_pair_distance(positions, symbols, "Pt", "Pt"),
            _min_pair_distance(positions, symbols, "Pt", "C"),
            _min_pair_distance(positions, symbols, "Pt", "O"),
            _min_pair_distance(positions, symbols, "Pt", "H"),
            _min_pair_distance(positions, symbols, "C", "O"),
            _mean_nearest_pt_distance(positions, pt_idx, ads_idx),
        ]
        return np.asarray(feature, dtype=float)


def build_feature_extractor(cfg: dict[str, Any]) -> FeatureExtractor:
    feature_type = str(cfg.get("feature_type", "geometric")).lower()
    if feature_type == "geometric":
        return GeometricFeatureExtractor(cfg)
    raise ValueError(f"Unsupported feature_type for sampling: {feature_type}")


def _min_pair_distance(
    positions: np.ndarray,
    symbols: list[str],
    s1: str,
    s2: str,
) -> float:
    idx1 = [i for i, sym in enumerate(symbols) if sym == s1]
    idx2 = [i for i, sym in enumerate(symbols) if sym == s2]
    if not idx1 or not idx2:
        return 999.0

    best = np.inf
    for i in idx1:
        for j in idx2:
            if s1 == s2 and i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < best:
                best = dist
    return float(best if np.isfinite(best) else 999.0)


def _mean_nearest_pt_distance(
    positions: np.ndarray,
    pt_idx: list[int],
    ads_idx: list[int],
) -> float:
    if not pt_idx or not ads_idx:
        return 999.0

    mins = []
    for idx in ads_idx:
        dists = [np.linalg.norm(positions[idx] - positions[j]) for j in pt_idx]
        mins.append(min(dists))
    return float(np.mean(mins))
