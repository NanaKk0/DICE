from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from workflow_core.dft.defaults import DEFAULT_DFT_STRUCTURE_NAME, DEFAULT_DFT_TASK_DIR_PREFIX
from workflow_core.common.io import ensure_dir, load_atoms_list, read_json, write_candidate_xyz, write_json
from workflow_core.dft.runtime import resolve_dft_runtime


def _load_dft_config(config_path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(config_path).resolve())
    payload.setdefault("jobid", None)
    payload.setdefault("overwrite_existing", False)
    payload.setdefault("deduplicate", False)
    payload.setdefault("sort_by", None)
    payload.setdefault("missing_cell_policy", "recover_or_fail")
    payload.setdefault("default_cell", None)
    payload["structure_name"] = DEFAULT_DFT_STRUCTURE_NAME
    payload["task_dir_prefix"] = DEFAULT_DFT_TASK_DIR_PREFIX
    return payload


def load_dft_prepare_config(config_path: str | Path) -> dict[str, Any]:
    return _load_dft_config(config_path)


def _frame_meta_from_atoms(atoms: Atoms, frame_idx: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"frame_idx": int(frame_idx)}
    for key, value in (atoms.info or {}).items():
        if isinstance(value, np.integer):
            payload[key] = int(value)
        elif isinstance(value, np.floating):
            payload[key] = float(value)
        elif isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload


def _load_optional_meta(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    meta_path = Path(path).resolve()
    if not meta_path.exists():
        return {}
    payload = read_json(meta_path)
    if isinstance(payload, list):
        return {int(item["frame_idx"]): item for item in payload if "frame_idx" in item}
    if isinstance(payload, dict):
        if all(str(key).isdigit() for key in payload.keys()):
            return {int(key): value for key, value in payload.items()}
    return {}


def _cell_is_valid(atoms: Atoms) -> bool:
    cell = np.asarray(atoms.get_cell())
    if cell.shape != (3, 3):
        return False
    lengths = np.linalg.norm(cell, axis=1)
    return bool(np.isfinite(cell).all() and np.all(lengths > 1e-8))


def _recover_cell_from_source(frame_meta: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates: list[tuple[str | None, int | None]] = [
        (frame_meta.get("source_saved_file"), frame_meta.get("source_saved_frame_index")),
        (frame_meta.get("source_file"), frame_meta.get("source_frame")),
    ]
    for source_file, source_frame in candidates:
        if not source_file:
            continue
        path = Path(str(source_file))
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            continue
        index = int(source_frame or 0)
        try:
            source_atoms = read(str(path), index=index)
        except Exception:
            continue
        if _cell_is_valid(source_atoms):
            return source_atoms.get_cell().array.copy(), {
                "status": "recovered_from_source",
                "source_file": str(path),
                "source_frame": index,
            }
    return None, {"status": "missing"}


def _apply_default_cell(default_cell: Any) -> np.ndarray | None:
    if default_cell is None:
        return None
    arr = np.asarray(default_cell, dtype=float)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"default_cell must be length-3 or 3x3, got shape {arr.shape}")


def resolve_structure_cell(atoms: Atoms, frame_meta: dict[str, Any], cfg: dict[str, Any]) -> tuple[Atoms, dict[str, Any]]:
    atoms = atoms.copy()
    if _cell_is_valid(atoms):
        return atoms, {"status": "present_in_input"}

    meta_cell = frame_meta.get("cell")
    if meta_cell is not None:
        cell = _apply_default_cell(meta_cell)
        if cell is not None:
            atoms.set_cell(cell)
            atoms.set_pbc(True)
            return atoms, {"status": "restored_from_meta"}

    recovered_cell, detail = _recover_cell_from_source(frame_meta)
    if recovered_cell is not None:
        atoms.set_cell(recovered_cell)
        atoms.set_pbc(True)
        return atoms, detail

    policy = str(cfg.get("missing_cell_policy", "recover_or_fail")).lower()
    if policy == "default_cell":
        default_cell = _apply_default_cell(cfg.get("default_cell"))
        if default_cell is None:
            raise RuntimeError("missing_cell_policy=default_cell but default_cell is not configured")
        atoms.set_cell(default_cell)
        atoms.set_pbc(True)
        return atoms, {"status": "default_cell"}
    if policy == "skip":
        return atoms, {"status": "skipped_missing_cell"}
    raise RuntimeError("Missing cell information and failed to recover from source/meta")


def _dedup_signature(atoms: Atoms) -> str:
    positions = np.round(atoms.get_positions(), 4)
    cell = np.round(atoms.get_cell().array, 4)
    symbols = atoms.get_chemical_symbols()
    composition = ",".join(sorted(symbols))
    payload = {
        "natoms": len(atoms),
        "composition": composition,
        "positions": positions.tolist(),
        "cell": cell.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_dft_jobs(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_dft_runtime(cfg)
    input_xyz = Path(cfg["input_xyz"]).resolve()
    input_meta = cfg.get("input_meta")
    task_root = ensure_dir(cfg["task_root"])
    output_root = ensure_dir(cfg["output_root"])
    structure_name = DEFAULT_DFT_STRUCTURE_NAME
    task_dir_prefix = DEFAULT_DFT_TASK_DIR_PREFIX
    overwrite_existing = bool(cfg.get("overwrite_existing", False))
    deduplicate = bool(cfg.get("deduplicate", False))
    sort_by = cfg.get("sort_by")
    jobid = str(cfg["jobid"])

    if not input_xyz.exists():
        raise FileNotFoundError(f"DFT input xyz not found: {input_xyz}")

    atoms_list = load_atoms_list(input_xyz)
    meta_map = _load_optional_meta(input_meta)

    source_dir = ensure_dir(task_root / "source")
    source_snapshot = source_dir / input_xyz.name
    source_snapshot.write_text(input_xyz.read_text(encoding="utf-8"), encoding="utf-8")
    source_meta_path = source_dir / "input_meta.json"
    write_json(source_meta_path, meta_map)

    job_root = ensure_dir(task_root / jobid)
    task_map_path = Path(cfg.get("task_map_path", output_root / "task_map.json")).resolve()
    record_path = Path(cfg.get("record_path", output_root / "record.dft.json")).resolve()
    failed_prepare_path = output_root / "failed_prepare.xyz"

    if task_map_path.exists() and record_path.exists() and not overwrite_existing:
        existing = read_json(task_map_path)
        return {
            "jobid": jobid,
            "task_root": str(job_root.resolve()),
            "n_tasks": len(existing),
            "task_map_path": str(task_map_path),
            "record_path": str(record_path),
            "reused": True,
        }

    failed_prepare_atoms: list[Atoms] = []
    task_entries: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    indexed_frames = list(enumerate(atoms_list))
    if isinstance(sort_by, str) and sort_by:
        indexed_frames.sort(key=lambda item: str((item[1].info or {}).get(sort_by, "")))

    task_idx = 0
    for frame_idx, atoms in indexed_frames:
        frame_meta = _frame_meta_from_atoms(atoms, frame_idx)
        frame_meta.update(meta_map.get(frame_idx, {}))

        try:
            resolved_atoms, cell_resolution = resolve_structure_cell(atoms, frame_meta, cfg)
        except Exception as exc:
            failed = atoms.copy()
            failed.info["dft_prepare_error"] = str(exc)
            failed_prepare_atoms.append(failed)
            continue

        if cell_resolution.get("status") == "skipped_missing_cell":
            failed = atoms.copy()
            failed.info["dft_prepare_error"] = "missing_cell_skipped"
            failed_prepare_atoms.append(failed)
            continue

        if deduplicate:
            signature = _dedup_signature(resolved_atoms)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

        task_name = f"{task_dir_prefix}{task_idx:06d}"
        task_dir = ensure_dir(job_root / task_name)
        structure_path = task_dir / structure_name
        meta_path = task_dir / "meta.json"
        frame_idx_path = task_dir / "frame_idx.txt"

        write(structure_path, resolved_atoms, format="extxyz")
        meta_payload = {
            **frame_meta,
            "task_name": task_name,
            "task_idx": int(task_idx),
            "cell_resolution": cell_resolution,
        }
        write_json(meta_path, meta_payload)
        frame_idx_path.write_text(str(frame_idx), encoding="utf-8")

        task_entries.append(
            {
                "task_name": task_name,
                "task_idx": int(task_idx),
                "frame_idx": int(frame_idx),
                "structure_file": str(structure_path.relative_to(job_root)),
                "meta_file": str(meta_path.relative_to(job_root)),
                "source_round": frame_meta.get("source_round"),
                "source_traj": frame_meta.get("source_saved_file") or frame_meta.get("source_file"),
                "source_frame": frame_meta.get("source_saved_frame_index", frame_meta.get("source_frame")),
                "selection_score": frame_meta.get("selection_score"),
                "committee_uncertainty": frame_meta.get("committee_uncertainty"),
            }
        )
        task_idx += 1

    write_json(task_map_path, task_entries)
    record_payload = {
        "jobid": jobid,
        "input_xyz": str(input_xyz),
        "input_meta": str(input_meta) if input_meta else None,
        "task_root": str(job_root.resolve()),
        "n_tasks": len(task_entries),
        "n_failed_prepare": len(failed_prepare_atoms),
        "task_map_path": str(task_map_path),
        "failed_prepare_xyz": str(failed_prepare_path.resolve()),
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    write_json(record_path, record_payload)
    write_candidate_xyz(failed_prepare_path, failed_prepare_atoms)
    return {
        "jobid": jobid,
        "task_root": str(job_root.resolve()),
        "n_tasks": len(task_entries),
        "task_map_path": str(task_map_path),
        "record_path": str(record_path),
        "failed_prepare_xyz": str(failed_prepare_path.resolve()),
        "reused": False,
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }

