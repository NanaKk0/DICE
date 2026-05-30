from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from workflow_backends.student.registry import get_student_evaluation_backend, get_student_model_catalog_backend
from workflow_core.common.io import read_json, write_candidate_xyz
from workflow_core.seed_pool.metadata import ensure_minimal_seed_metadata, sanitize_system_id
from workflow_runtime.environment import context_with_execution_context, student_backend_role


def committee_models_from_summary(summary_path: str | Path) -> tuple[str, list[str]]:
    payload = read_json(summary_path)
    backend_name = str(payload.get("backend", "deepmd")).strip().lower()
    backend = get_student_model_catalog_backend(backend_name)
    artifacts = backend.collect_models(str(Path(summary_path).resolve()))
    members = [str(Path(item.model_path).resolve()) for item in artifacts if Path(item.model_path).resolve().exists()]
    if len(members) < 2:
        raise RuntimeError(
            f"Usable {backend_name} committee size is {len(members)}; at least 2 successful committee models are required"
        )
    return backend_name, members


def _resolve_backend_eval_cfg(cfg: dict[str, Any], backend_name: str) -> dict[str, Any] | str:
    if backend_name == "nep":
        nep_cfg = dict(cfg.get("nep", {}) or {})
        student_model = dict(cfg.get("student_model", {}) or {})
        for key in ("type_map", "committee_size"):
            if student_model.get(key) is not None:
                nep_cfg[key] = student_model[key]
        return {
            "nep": nep_cfg,
            "committee_summary_path": None,
        }
    return dict(cfg or {})


def rescore_atoms_with_student_backend(
    atoms_list: list[Atoms],
    *,
    cfg: dict[str, Any],
    committee_summary_path: str | Path,
    temp_root: str | Path,
    prefix: str,
) -> list[Atoms]:
    atoms_items = [ensure_minimal_seed_metadata(item) for item in atoms_list]
    if not atoms_items:
        return []
    temp_root = Path(temp_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    input_xyz = temp_root / f"{prefix}.extxyz"
    write_candidate_xyz(input_xyz, atoms_items)
    summary_path = Path(committee_summary_path).resolve()
    summary_payload = read_json(summary_path)
    backend_name = str(summary_payload.get("backend", cfg.get("student_backend", "deepmd"))).strip().lower()
    backend = get_student_evaluation_backend(backend_name)
    eval_cfg = _resolve_backend_eval_cfg(cfg, backend_name)
    extra_context = {
        "committee_summary_path": str(summary_path.resolve()),
        "committee_summary_override": str(summary_path.resolve()),
    }
    extra_context = context_with_execution_context(
        cfg,
        role=student_backend_role(backend_name, "evaluate"),
        context=extra_context,
    )
    request = backend.build_evaluation_request(
        eval_cfg,
        dataset_path=str(input_xyz.resolve()),
        committee_summary_path=str(summary_path.resolve()),
        work_root=str((temp_root / f"{prefix}_eval").resolve()),
        extra_context=extra_context,
    )
    infer_result = backend.evaluate_committee(None, request)
    frame_metrics = list(infer_result.get("frame_metrics", []) or [])
    if len(frame_metrics) != len(atoms_items):
        raise RuntimeError(
            f"Committee rescoring frame count mismatch for backend={backend_name}: "
            f"expected {len(atoms_items)}, got {len(frame_metrics)}"
        )
    rescored: list[Atoms] = []
    for frame_index, atoms in enumerate(atoms_items):
        metrics = dict(frame_metrics[frame_index] or {})
        item = atoms.copy()
        item.info["force_std_max"] = float(metrics["force_std_max"])
        item.info["decision_metric"] = float(metrics["force_std_max"])
        item.info["student_uncertainty"] = float(metrics["force_std_max"])
        item.info["student_backend"] = backend_name
        item.info["uncertainty_metric"] = "force_std_max"
        item.info["uncertainty_source"] = "committee_eval"
        rescored.append(item)
    return rescored


def rescore_seed_pool_grouped(
    grouped_atoms: dict[str, list[Atoms]],
    *,
    cfg: dict[str, Any],
    committee_summary_path: str | Path,
    temp_root: str | Path,
) -> tuple[str, dict[str, list[Atoms]]]:
    backend_name, _ = committee_models_from_summary(committee_summary_path)
    rescored_grouped: dict[str, list[Atoms]] = {}
    for system_id in sorted(grouped_atoms):
        rescored_grouped[system_id] = rescore_atoms_with_student_backend(
            grouped_atoms[system_id],
            cfg=cfg,
            committee_summary_path=committee_summary_path,
            temp_root=temp_root,
            prefix=f"{backend_name}_{sanitize_system_id(system_id)}_seed_pool",
        )
    return backend_name, rescored_grouped
