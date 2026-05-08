from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from workflow_core.feature_extractor import build_feature_extractor
from workflow_core.common.io import ensure_dir, read_json, write_json


def _warn(message: str) -> None:
    warnings.warn(f"[distribution_plot] {message}", RuntimeWarning, stacklevel=2)


def load_distribution_plot_config(paths, config_path: str | Path | None = None) -> dict[str, Any]:
    defaults = {
        "enabled": False,
        "feature_type": "geometric",
        "use_pca": True,
        "pca_dim": 32,
        "normalize_mode": "per_plot",
        "max_structures_per_group": 2000,
        "random_state": 42,
        "save_csv": True,
        "save_embeddings": True,
        "figure": {
            "dpi": 160,
            "figsize": [7.0, 6.0],
            "point_size": 10,
            "alpha_base": 0.35,
            "alpha_new": 0.75,
        },
        "colors": {
            "base": "#7f8c8d",
            "increment": "#d35400",
        },
    }
    allowed_keys = {
        "enabled",
        "feature_type",
        "use_pca",
        "pca_dim",
        "normalize_mode",
        "max_structures_per_group",
        "random_state",
        "save_csv",
        "save_embeddings",
    }
    cfg_path = (
        Path(config_path)
        if config_path is not None
        else Path(paths.configs) / "reporting" / "data_distribution_plot.json"
    )
    if not cfg_path.exists():
        return defaults
    try:
        payload = read_json(cfg_path)
    except Exception as exc:
        _warn(f"Failed to load distribution plot config from {cfg_path}: {exc}")
        return defaults

    merged = dict(defaults)
    merged.update({k: v for k, v in payload.items() if k in allowed_keys})
    merged["figure"] = dict(defaults["figure"])
    merged["figure"].update(payload.get("figure", {}))
    merged["colors"] = dict(defaults["colors"])
    merged["colors"].update(payload.get("colors", {}))
    return merged


def _load_atoms_from_xyz(path: Path) -> list[Atoms]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    atoms = read(str(path), index=":")
    if isinstance(atoms, Atoms):
        return [atoms]
    return list(atoms)


def _load_group_atoms(xyz_files: list[Path], max_structures: int, random_state: int) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for xyz in xyz_files:
        atoms_list.extend(_load_atoms_from_xyz(xyz))
    if max_structures > 0 and len(atoms_list) > max_structures:
        rng = np.random.default_rng(int(random_state))
        indices = np.sort(rng.choice(len(atoms_list), size=max_structures, replace=False))
        atoms_list = [atoms_list[int(idx)] for idx in indices]
    return atoms_list


def _project_groups(groups: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    all_atoms: list[Atoms] = []
    for group in groups:
        all_atoms.extend(group["atoms"])
    if not all_atoms:
        raise RuntimeError("No structures available for data distribution plotting")

    extractor = build_feature_extractor(cfg)
    extractor.fit(all_atoms)

    features_by_group: list[np.ndarray] = []
    for group in groups:
        atoms_list = group["atoms"]
        if not atoms_list:
            features_by_group.append(np.zeros((0, 1), dtype=float))
            continue
        features = np.asarray([extractor.transform(atoms) for atoms in atoms_list], dtype=float)
        features_by_group.append(features)

    normalize_mode = str(cfg.get("normalize_mode", "per_plot")).lower()
    use_pca = bool(cfg.get("use_pca", True))
    pca_dim = int(cfg.get("pca_dim", 32))

    if normalize_mode == "global_reference" and len(features_by_group) >= 1 and len(features_by_group[0]) > 0:
        fit_features = features_by_group[0]
    else:
        fit_features = np.vstack([item for item in features_by_group if item.size > 0])

    scaler = StandardScaler()
    scaled_fit = scaler.fit_transform(fit_features)
    scaled_groups = [scaler.transform(item) if item.size > 0 else np.zeros((0, scaled_fit.shape[1]), dtype=float) for item in features_by_group]

    pca = None
    fit_projected = scaled_fit
    if use_pca and scaled_fit.ndim == 2 and scaled_fit.shape[0] > 1 and scaled_fit.shape[1] > 1 and not np.allclose(scaled_fit, scaled_fit[0]):
        pca_dim_eff = min(max(2, pca_dim), scaled_fit.shape[0] - 1, scaled_fit.shape[1])
        if pca_dim_eff >= 2:
            pca = PCA(n_components=pca_dim_eff, random_state=int(cfg.get("random_state", 42)))
            fit_projected = pca.fit_transform(scaled_fit)
            projected_groups = [pca.transform(item) if item.size > 0 else np.zeros((0, pca_dim_eff), dtype=float) for item in scaled_groups]
        else:
            projected_groups = scaled_groups
    else:
        projected_groups = scaled_groups

    coords_2d: list[np.ndarray] = []
    for projected in projected_groups:
        if projected.size == 0:
            coords_2d.append(np.zeros((0, 2), dtype=float))
        elif projected.shape[1] == 1:
            coords_2d.append(np.column_stack([projected[:, 0], np.zeros(projected.shape[0], dtype=float)]))
        else:
            coords_2d.append(np.asarray(projected[:, :2], dtype=float))

    if normalize_mode in {"per_plot", "global_reference"}:
        if normalize_mode == "global_reference" and len(coords_2d) >= 1 and len(coords_2d[0]) > 0:
            reference_coords = coords_2d[0]
        else:
            reference_coords = np.vstack([item for item in coords_2d if item.size > 0])
        center = reference_coords.mean(axis=0) if len(reference_coords) > 0 else np.zeros(2, dtype=float)
        centered = [coords - center for coords in coords_2d]
        if len(reference_coords) > 0:
            scale_source = reference_coords - center
            scale = float(np.max(np.abs(scale_source)))
        else:
            scale = 1.0
        if scale <= 0.0:
            scale = 1.0
        coords_2d = [coords / scale for coords in centered]
        normalization = {"mode": normalize_mode, "center": center.tolist(), "scale": scale}
    else:
        normalization = {"mode": "none", "center": [0.0, 0.0], "scale": 1.0}

    metadata = {
        "normalize_mode": normalize_mode,
        "normalization": normalization,
        "explained_variance_ratio": [] if pca is None else pca.explained_variance_ratio_[:2].tolist(),
        "n_features": int(fit_features.shape[1]),
    }
    return coords_2d, metadata


def _write_embedding_csv(output_csv: Path, groups: list[dict[str, Any]], coords_2d: list[np.ndarray]) -> None:
    ensure_dir(output_csv.parent)
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "label", "point_index", "x", "y", "natoms"])
        writer.writeheader()
        for group, coords in zip(groups, coords_2d):
            for index, (x_coord, y_coord) in enumerate(coords):
                natoms = len(group["atoms"][index]) if index < len(group["atoms"]) else ""
                writer.writerow(
                    {
                        "group": group["name"],
                        "label": group["label"],
                        "point_index": index,
                        "x": float(x_coord),
                        "y": float(y_coord),
                        "natoms": natoms,
                    }
                )


def _save_plot(output_png: Path, groups: list[dict[str, Any]], coords_2d: list[np.ndarray], cfg: dict[str, Any], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _warn(f"matplotlib is unavailable, skipped PNG generation for {output_png.name}: {exc}")
        return

    ensure_dir(output_png.parent)
    figure_cfg = cfg.get("figure", {})
    fig, ax = plt.subplots(figsize=tuple(figure_cfg.get("figsize", [7.0, 6.0])), dpi=int(figure_cfg.get("dpi", 160)))
    for group, coords in zip(groups, coords_2d):
        if len(coords) == 0:
            continue
        alpha = figure_cfg.get("alpha_new", 0.75) if group.get("is_increment", False) else figure_cfg.get("alpha_base", 0.35)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(figure_cfg.get("point_size", 10)),
            alpha=float(alpha),
            c=group["color"],
            label=f"{group['label']} (n={len(coords)})",
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("PCA 1 (normalized)")
    ax.set_ylabel("PCA 2 (normalized)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def _render_distribution_plot(groups: list[dict[str, Any]], output_png: Path, output_json: Path, output_csv: Path, cfg: dict[str, Any], title: str, metadata_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    coords_2d, metadata = _project_groups(groups, cfg)
    if bool(cfg.get("save_csv", True)) and bool(cfg.get("save_embeddings", True)):
        _write_embedding_csv(output_csv, groups, coords_2d)
    _save_plot(output_png, groups, coords_2d, cfg, title)
    metadata.update(metadata_extra or {})
    metadata["groups"] = [
        {
            "name": group["name"],
            "label": group["label"],
            "n_structures": len(group["atoms"]),
            "xyz_files": [str(path.resolve()) for path in group["xyz_files"]],
        }
        for group in groups
    ]
    metadata["output_png"] = str(output_png.resolve())
    metadata["output_csv"] = str(output_csv.resolve())
    write_json(output_json, metadata)
    return {
        "png": str(output_png.resolve()),
        "json": str(output_json.resolve()),
        "csv": str(output_csv.resolve()),
        "group_counts": {group["name"]: len(group["atoms"]) for group in groups},
    }


def _safe_plot(groups: list[dict[str, Any]], output_png: Path, output_json: Path, output_csv: Path, cfg: dict[str, Any], title: str, metadata_extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        return _render_distribution_plot(groups, output_png, output_json, output_csv, cfg, title, metadata_extra)
    except Exception as exc:
        _warn(f"Failed to build distribution plot {output_png.name}: {exc}")
        return None


def plot_outer_round_dft_increment(paths, outer_round: int, config_path: str | Path | None = None) -> dict[str, Any] | None:
    cfg = load_distribution_plot_config(paths, config_path=config_path)
    if not bool(cfg.get("enabled", False)):
        return None

    base_xyz = paths.committee_cycle_finetune_merged_xyz(outer_round)
    increment_xyz = paths.dft_submit_labeled_xyz(outer_round)
    if not base_xyz.exists() or not increment_xyz.exists() or increment_xyz.stat().st_size == 0:
        return None

    max_structures = int(cfg.get("max_structures_per_group", 2000))
    random_state = int(cfg.get("random_state", 42))
    groups = [
        {
            "name": "base",
            "label": "existing_train",
            "color": cfg.get("colors", {}).get("base", "#7f8c8d"),
            "is_increment": False,
            "xyz_files": [base_xyz],
            "atoms": _load_group_atoms([base_xyz], max_structures, random_state),
        },
        {
            "name": "increment",
            "label": "new_dft_labeled",
            "color": cfg.get("colors", {}).get("increment", "#d35400"),
            "is_increment": True,
            "xyz_files": [increment_xyz],
            "atoms": _load_group_atoms([increment_xyz], max_structures, random_state),
        },
    ]
    if not groups[0]["atoms"] or not groups[1]["atoms"]:
        return None

    return _safe_plot(
        groups,
        paths.outer_round_dft_distribution_png(outer_round),
        paths.outer_round_dft_distribution_json(outer_round),
        paths.outer_round_dft_distribution_csv(outer_round),
        cfg,
        title=f"Outer Round {outer_round}: New DFT-Labeled vs Existing Training Distribution",
        metadata_extra={"outer_round": int(outer_round), "plot_type": "outer_round_dft_increment"},
    )


def plot_dp_round_increment(paths, outer_round: int, dp_round: int, config_path: str | Path | None = None) -> dict[str, Any] | None:
    if dp_round <= 0:
        return None

    cfg = load_distribution_plot_config(paths, config_path=config_path)
    if not bool(cfg.get("enabled", False)):
        return None

    base_xyz_files: list[Path] = []
    for previous_round in range(0, dp_round):
        candidate = paths.dp_subworkflow_round_train_xyz(outer_round, previous_round)
        if candidate.exists() and candidate.stat().st_size > 0:
            base_xyz_files.append(candidate)
    increment_xyz = paths.dp_subworkflow_round_train_xyz(outer_round, dp_round)
    if not base_xyz_files or not increment_xyz.exists() or increment_xyz.stat().st_size == 0:
        return None

    max_structures = int(cfg.get("max_structures_per_group", 2000))
    random_state = int(cfg.get("random_state", 42))
    groups = [
        {
            "name": "base",
            "label": f"dp_r{dp_round - 1}_train_cumulative",
            "color": cfg.get("colors", {}).get("base", "#7f8c8d"),
            "is_increment": False,
            "xyz_files": base_xyz_files,
            "atoms": _load_group_atoms(base_xyz_files, max_structures, random_state),
        },
        {
            "name": "increment",
            "label": f"dp_r{dp_round}_new_train",
            "color": cfg.get("colors", {}).get("increment", "#d35400"),
            "is_increment": True,
            "xyz_files": [increment_xyz],
            "atoms": _load_group_atoms([increment_xyz], max_structures, random_state),
        },
    ]
    if not groups[0]["atoms"] or not groups[1]["atoms"]:
        return None

    return _safe_plot(
        groups,
        paths.dp_round_distribution_png(outer_round, dp_round),
        paths.dp_round_distribution_json(outer_round, dp_round),
        paths.dp_round_distribution_csv(outer_round, dp_round),
        cfg,
        title=f"Outer Round {outer_round} / DP Round {dp_round}: New Train Increment vs Previous Train Distribution",
        metadata_extra={"outer_round": int(outer_round), "dp_round": int(dp_round), "plot_type": "dp_round_increment"},
    )

