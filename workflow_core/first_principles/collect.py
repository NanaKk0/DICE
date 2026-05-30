from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from workflow_core.common.io import read_json, write_candidate_xyz, write_json
from workflow_core.first_principles.substep_harness import FirstPrinciplesSubstepContext, run_first_principles_substep


TASK_DIR_PATTERN = re.compile(r"^(?:task\.)?(\d+)$")


def load_dft_collect_config(config_path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(config_path).resolve())
    payload.setdefault("required_outputs", ["cp2k.out"])
    return payload


load_first_principles_collect_config = load_dft_collect_config


def _iter_task_dirs(job_root: Path) -> list[tuple[int, Path]]:
    tasks = []
    for child in sorted(job_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        match = TASK_DIR_PATTERN.match(child.name)
        if not match:
            continue
        tasks.append((int(match.group(1)), child))
    return tasks


def _task_success(task_dir: Path, cfg: dict[str, Any], backend) -> bool:
    status_path = task_dir / "task_status.json"
    if status_path.exists():
        try:
            payload = read_json(status_path)
            if payload.get("status") == "success":
                return True
        except Exception:
            pass
    return bool(backend.task_is_successful(task_dir, cfg))


def collect_dft_results(cfg: dict[str, Any]) -> dict[str, Any]:
    return run_first_principles_substep(cfg, substep="collect", execute=_collect_dft_results)


collect_first_principles_results = collect_dft_results


def _collect_dft_results(context: FirstPrinciplesSubstepContext) -> dict[str, Any]:
    cfg = context.cfg
    backend_name = context.backend_name
    backend = context.backend
    structure_name = str(cfg.get("structure_name", backend.structure_name))
    job_root = Path(cfg["task_root"]).resolve() / cfg["jobid"]
    output_root = Path(cfg.get("output_root", Path(cfg["task_root"]).resolve().parent)).resolve()
    labeled_xyz = Path(cfg.get("labeled_xyz", output_root / "labeled.xyz")).resolve()
    failed_xyz = Path(cfg.get("failed_xyz", output_root / "failed.xyz")).resolve()
    summary_path = Path(cfg["summary_path"]).resolve() if cfg.get("summary_path") else None
    task_map_path = Path(cfg.get("task_map_path", output_root / "task_map.json")).resolve()
    task_map = read_json(task_map_path) if task_map_path.exists() else []
    task_map_by_idx = {int(item["task_idx"]): item for item in task_map}

    labeled_atoms: list[Atoms] = []
    failed_atoms: list[Atoms] = []
    failed_tasks: list[str] = []

    for task_idx, task_dir in _iter_task_dirs(job_root):
        structure_path = task_dir / structure_name
        base_atoms = read(str(structure_path), index=0)
        if not _task_success(task_dir, cfg, backend):
            base_atoms.info["dft_collect_error"] = "task_not_successful"
            failed_atoms.append(base_atoms)
            failed_tasks.append(task_dir.name)
            continue
        try:
            atoms = backend.parse_labeled_task(task_dir, cfg)
        except Exception as exc:
            failed = base_atoms.copy()
            failed.info["dft_collect_error"] = str(exc)
            failed_atoms.append(failed)
            failed_tasks.append(task_dir.name)
            continue
        meta_path = task_dir / "meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            for key, value in meta.items():
                if key == "cell_resolution":
                    atoms.info[key] = value
                    continue
                atoms.info[key] = value
        atoms.info["first_principles_backend"] = backend_name
        atoms.info["dft_jobid"] = cfg["jobid"]
        atoms.info["dft_task_idx"] = int(task_idx)
        mapped = task_map_by_idx.get(task_idx, {})
        for key in ("source_round", "source_traj", "source_frame", "selection_score", "committee_uncertainty"):
            if key in mapped and key not in atoms.info:
                atoms.info[key] = mapped[key]
        labeled_atoms.append(atoms)

    write_candidate_xyz(labeled_xyz, labeled_atoms)
    write_candidate_xyz(failed_xyz, failed_atoms)
    summary = {
        "jobid": cfg["jobid"],
        "first_principles_backend": backend_name,
        "n_total": len(task_map) if task_map else len(labeled_atoms) + len(failed_atoms),
        "n_success": len(labeled_atoms),
        "n_failed": len(failed_atoms),
        "success_ratio": (len(labeled_atoms) / max(1, len(labeled_atoms) + len(failed_atoms))),
        "failed_tasks": failed_tasks,
        "labeled_xyz": str(labeled_xyz),
        "failed_xyz": str(failed_xyz),
        "first_principles_runtime_resolution": cfg.get("first_principles_runtime_resolution"),
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    if summary_path is not None:
        write_json(summary_path, summary)
    return summary
