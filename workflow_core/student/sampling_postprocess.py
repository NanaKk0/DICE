from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from workflow_artifacts.sampling_export import export_selected_candidates_xyz_only, write_sampling_summary_fast
from workflow_artifacts.sampling_summary import (
    append_sampling_summary_to_progress_log,
    trim_uncertainty_only_sampling_summary,
)
from workflow_backends.sampling.base import SamplingBackendResult, SamplingObservation
from workflow_core.common.io import read_json, write_candidate_xyz, write_json
from workflow_core.common.uncertainty import UncertaintyRecord, append_uncertainty_records, route_uncertainty
from workflow_core.seed_pool.uncertainty import group_atoms_by_system, write_system_extxyz_dir
from workflow_core.structures.formats import extract_lammpstrj_selected_frames
from workflow_policies.sampling_training_selection import select_training_candidates_by_system


def _trust_thresholds(cfg: dict[str, Any]) -> tuple[float, float]:
    lammps_cfg = dict(cfg.get("lammps", {}) or {})
    uncertainty_cfg = dict(cfg.get("uncertainty", {}) or {})
    trust_lo = float(lammps_cfg.get("f_trust_lo", uncertainty_cfg.get("level_f_lo", uncertainty_cfg.get("trust_lo", 0.10))))
    trust_hi = float(lammps_cfg.get("f_trust_hi", uncertainty_cfg.get("level_f_hi", uncertainty_cfg.get("trust_hi", 0.40))))
    return trust_lo, trust_hi


def _system_id_from_atoms(atoms) -> str:
    info = atoms.info or {}
    return str(info.get("system_id") or info.get("seed_system_id") or atoms.get_chemical_formula(mode="hill"))


def _materialize_observation_atoms(observations: list[SamplingObservation], cfg: dict[str, Any]) -> dict[int, Any]:
    materialized: dict[int, Any] = {}
    pending_by_traj: dict[tuple[str, str], list[tuple[int, SamplingObservation]]] = defaultdict(list)
    for index, observation in enumerate(observations):
        if observation.atoms is not None:
            materialized[index] = observation.atoms
            continue
        ref = observation.trajectory_ref
        if ref is None:
            continue
        pending_by_traj[(str(Path(ref.trajectory_path).resolve()), str(ref.trajectory_format))].append((index, observation))

    lammps_cfg = dict(cfg.get("lammps", {}) or {})
    type_map = list(lammps_cfg.get("type_map", ["Pt", "O", "H"]))
    for (trajectory_path, trajectory_format), items in pending_by_traj.items():
        if trajectory_format != "lammpstrj":
            continue
        target_steps: set[int] = set()
        for _, observation in items:
            if observation.step is not None:
                target_steps.add(int(observation.step))
            ref = observation.trajectory_ref
            if ref is not None and ref.fallback_step is not None:
                target_steps.add(int(ref.fallback_step))
        frame_map = extract_lammpstrj_selected_frames(Path(trajectory_path), type_map=type_map, target_steps=target_steps)
        for index, observation in items:
            atoms = frame_map.get(int(observation.step)) if observation.step is not None else None
            ref = observation.trajectory_ref
            if atoms is None and ref is not None and ref.fallback_step is not None:
                atoms = frame_map.get(int(ref.fallback_step))
            if atoms is not None:
                materialized[index] = atoms
    return materialized


def _annotate_atoms(atoms, observation: SamplingObservation, route: str):
    item = atoms.copy()
    metadata = dict(observation.metadata or {})
    item.info["source_job_id"] = str(observation.task_id)
    item.info["source_task"] = str(observation.task_id)
    item.info["source_frame_index"] = int(observation.frame_index or 0)
    item.info["driver_model_index"] = int(metadata.get("driver_model_index", 0) or 0)
    item.info["uncertainty_source"] = str(metadata.get("uncertainty_source", "sampling_backend"))
    item.info["student_uncertainty"] = float(observation.force_std_max)
    item.info["force_std_max"] = float(observation.force_std_max)
    item.info["force_std_avg"] = float(observation.force_std_avg if observation.force_std_avg is not None else observation.force_std_max)
    item.info["force_std_p90"] = float(observation.force_std_p90 if observation.force_std_p90 is not None else observation.force_std_max)
    item.info["decision_metric"] = float(observation.force_std_max)
    if observation.step is not None:
        item.info["md_step"] = int(observation.step)
    if "source_structure_id" in metadata:
        item.info["source_structure_id"] = metadata["source_structure_id"]
    if "source_system_id" in metadata:
        item.info["system_id"] = metadata["source_system_id"]
        item.info["seed_system_id"] = metadata["source_system_id"]
    item.info["frame_class"] = route
    return item


def _selected_record(atoms, observation: SamplingObservation, route: str) -> dict[str, Any]:
    annotated = _annotate_atoms(atoms, observation, route)
    return {
        "atoms": annotated,
        "source_job_id": str(observation.task_id),
        "source_frame_index": int(observation.frame_index or 0),
        "driver_model_index": int((observation.metadata or {}).get("driver_model_index", 0) or 0),
        "decision_metric": float(observation.force_std_max),
        "metrics": {
            "force_std_max": float(observation.force_std_max),
            "force_std_avg": float(observation.force_std_avg if observation.force_std_avg is not None else observation.force_std_max),
            "force_std_p90": float(observation.force_std_p90 if observation.force_std_p90 is not None else observation.force_std_max),
        },
        "geometry": {"min_pair_distance": float((observation.metadata or {}).get("min_pair_distance", 0.0) or 0.0)},
        "frame_class": route,
    }


def _sampling_system_stats(system_route_counts: dict[str, Counter[str]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"group_by": "formula", "systems": {}, "totals": {}}
    totals = Counter()
    for system_id in sorted(system_route_counts):
        counts = system_route_counts[system_id]
        total = int(counts.get("accurate", 0) + counts.get("candidate", 0) + counts.get("failed", 0))
        denom = max(1, total)
        stats["systems"][system_id] = {
            "accurate": int(counts.get("accurate", 0)),
            "candidate": int(counts.get("candidate", 0)),
            "failed": int(counts.get("failed", 0)),
            "accurate_ratio": float(counts.get("accurate", 0) / denom),
            "candidate_ratio": float(counts.get("candidate", 0) / denom),
            "failed_ratio": float(counts.get("failed", 0) / denom),
        }
        totals.update(
            {
                "accurate": int(counts.get("accurate", 0)),
                "candidate": int(counts.get("candidate", 0)),
                "failed": int(counts.get("failed", 0)),
            }
        )
    total_frames = int(totals.get("accurate", 0) + totals.get("candidate", 0) + totals.get("failed", 0))
    denom = max(1, total_frames)
    stats["totals"] = {
        "accurate": int(totals.get("accurate", 0)),
        "candidate": int(totals.get("candidate", 0)),
        "failed": int(totals.get("failed", 0)),
        "accurate_ratio": float(totals.get("accurate", 0) / denom),
        "candidate_ratio": float(totals.get("candidate", 0) / denom),
        "failed_ratio": float(totals.get("failed", 0) / denom),
    }
    return stats


def _task_summaries(observations: list[SamplingObservation], routes_by_index: dict[int, str], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for index, observation in enumerate(observations):
        route = routes_by_index.get(index)
        if route is None:
            continue
        grouped[str(observation.task_id)][route] += 1
    summaries = list(existing or [])
    existing_ids = {str(item.get("task_id", "")) for item in summaries}
    for task_id in sorted(grouped):
        if task_id in existing_ids:
            continue
        counts = grouped[task_id]
        total = int(counts.get("accurate", 0) + counts.get("candidate", 0) + counts.get("failed_uncertainty", 0))
        summaries.append(
            {
                "task_id": task_id,
                "status": "finished",
                "record_count": total,
                "accurate_count": int(counts.get("accurate", 0)),
                "candidate_count": int(counts.get("candidate", 0)),
                "failed_count": int(counts.get("failed_uncertainty", 0)),
                "failed_ratio": float(counts.get("failed_uncertainty", 0) / total) if total else 0.0,
            }
        )
    return summaries


def postprocess_student_sampling_result(
    *,
    backend_result: SamplingBackendResult,
    cfg: dict[str, Any],
    work_root: str | Path,
    summary: dict[str, Any],
    context: dict[str, Any],
) -> Path:
    started_at = time.perf_counter()
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    failed_tasks = [
        dict(item)
        for item in list(backend_result.task_summaries or [])
        if str(item.get("status", "")).strip().lower() not in {"", "finished", "success"}
    ]
    if failed_tasks:
        task_ids = ", ".join(str(item.get("task_id", "<unknown>")) for item in failed_tasks)
        raise RuntimeError(f"sampling backend tasks failed: {task_ids}")
    observations = list(backend_result.observations or [])
    trust_lo, trust_hi = _trust_thresholds(cfg)
    high_unc_threshold = context.get("stop_high_unc_threshold")
    high_unc_threshold = None if high_unc_threshold is None else float(high_unc_threshold)

    materialized_atoms = _materialize_observation_atoms(observations, cfg)
    route_counts = Counter()
    routes_by_index: dict[int, str] = {}
    system_route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_records: list[dict[str, Any]] = []
    candidate_atoms: list[Any] = []
    failed_atoms: list[Any] = []
    high_unc_count = 0

    for index, observation in enumerate(observations):
        route = route_uncertainty(observation.force_std_max, trust_lo=trust_lo, trust_hi=trust_hi)
        routes_by_index[index] = route
        route_counts[route] += 1
        if high_unc_threshold is not None and float(observation.force_std_max) >= high_unc_threshold:
            high_unc_count += 1
        atoms = materialized_atoms.get(index)
        system_id = str((observation.metadata or {}).get("source_system_id") or "UNKNOWN")
        if atoms is not None:
            system_id = _system_id_from_atoms(atoms)
        system_bucket = "failed" if route == "failed_uncertainty" else route
        system_route_counts[system_id][system_bucket] += 1
        if atoms is None or route == "accurate":
            continue
        if route == "failed_uncertainty":
            failed_atoms.append(_annotate_atoms(atoms, observation, route))
            continue
        record = _selected_record(atoms, observation, route)
        candidate_records.append(record)
        candidate_atoms.append(record["atoms"])

    candidate_xyz = work_root / "candidate.xyz"
    failed_xyz = work_root / "failed.xyz"
    candidate_by_system_dir = work_root / "candidate_by_system"
    failed_by_system_dir = work_root / "failed_by_system"
    write_candidate_xyz(candidate_xyz, candidate_atoms)
    write_candidate_xyz(failed_xyz, failed_atoms)
    write_system_extxyz_dir(candidate_by_system_dir, group_atoms_by_system(candidate_atoms))
    write_system_extxyz_dir(failed_by_system_dir, group_atoms_by_system(failed_atoms))

    candidate_train_cfg = dict(cfg.get("candidate_train_selection", {}) or {})
    selected_records = (
        select_training_candidates_by_system(
            candidate_records,
            max_train_candidates_per_system=int(candidate_train_cfg.get("max_train_candidates_per_system", 200)),
            selection_mode=str(candidate_train_cfg.get("selection_mode", "random_per_system")).strip().lower(),
            random_seed=int(candidate_train_cfg.get("random_seed", 20260502)),
            teacher_round=int((context.get("summary_extra", {}) or {}).get("teacher_round", -1)),
            student_round=int((context.get("summary_extra", {}) or {}).get("student_round", -1)),
        )
        if bool(candidate_train_cfg.get("enabled", True))
        else list(candidate_records)
    )
    selected_xyz = work_root / "selected_candidates.xyz"
    export_selected_candidates_xyz_only(selected_records, selected_xyz)
    selected_count_per_system: Counter[str] = Counter()
    selected_decision_values: list[float] = []
    selected_per_job_counts: Counter[str] = Counter()
    for record in selected_records:
        selected_decision_values.append(float(record["decision_metric"]))
        selected_per_job_counts[str(record["source_job_id"])] += 1
        selected_count_per_system[_system_id_from_atoms(record["atoms"])] += 1

    uncertainty_jsonl = work_root / "uncertainty_records.jsonl"
    uncertainty_jsonl.write_text("", encoding="utf-8")
    uncertainty_records: list[UncertaintyRecord] = []
    for index, observation in enumerate(observations):
        route = routes_by_index.get(index)
        if route is None:
            continue
        atoms = materialized_atoms.get(index)
        uncertainty_records.append(
            UncertaintyRecord(
                backend=backend_result.backend,
                sampling_backend=backend_result.sampling_backend,
                source=str((observation.metadata or {}).get("uncertainty_source", backend_result.sampling_backend)),
                task_id=str(observation.task_id),
                frame_id=f"{observation.task_id}:{int(observation.frame_index or 0)}",
                step=observation.step,
                temperature=observation.temperature,
                force_std_max=float(observation.force_std_max),
                route=route,
                structure_path=str(Path((observation.metadata or {}).get("structure_path")).resolve())
                if (observation.metadata or {}).get("structure_path")
                else None,
                metadata={
                    **dict(observation.metadata or {}),
                    "source_system_id": (
                        _system_id_from_atoms(atoms)
                        if atoms is not None
                        else str((observation.metadata or {}).get("source_system_id", "UNKNOWN"))
                    ),
                },
            )
        )
    append_uncertainty_records(uncertainty_jsonl, uncertainty_records)

    write_json(work_root / "jobs.json", [dict(item) for item in backend_result.jobs])
    summary_output = write_sampling_summary_fast(
        summary_path=work_root / "sampling_summary.json",
        committee_size=int(summary.get("committee_size", len(summary.get("members", [])))),
        usable_committee_size=int(backend_result.runtime_artifacts.get("usable_committee_size", summary.get("usable_committee_size", summary.get("committee_size", 0))) or 0),
        total_jobs=len(backend_result.jobs),
        total_frames=len(observations),
        accurate_count=int(route_counts.get("accurate", 0)),
        candidate_count=int(route_counts.get("candidate", 0)),
        failed_uncertainty_count=int(route_counts.get("failed_uncertainty", 0)),
        selected_count=int(len(selected_records)),
        decision_metric="force_std_max",
        level_f_lo=trust_lo,
        level_f_hi=trust_hi,
        selected_xyz=selected_xyz,
        selected_jsonl=None,
        stop_high_unc_threshold=high_unc_threshold,
        high_unc_count=int(high_unc_count),
        selected_decision_values=selected_decision_values,
        selected_per_job_counts=selected_per_job_counts,
        sampling_system_stats=_sampling_system_stats(system_route_counts),
        extra_summary={
            **dict(context.get("summary_extra") or {}),
            "backend": backend_result.backend,
            "sampling_backend": backend_result.sampling_backend,
            "candidate_xyz": str(candidate_xyz.resolve()),
            "failed_xyz": str(failed_xyz.resolve()),
            "candidate_by_system_dir": str(candidate_by_system_dir.resolve()),
            "failed_by_system_dir": str(failed_by_system_dir.resolve()),
            "uncertainty_records_jsonl": str(uncertainty_jsonl.resolve()),
            "task_summaries": _task_summaries(observations, routes_by_index, backend_result.task_summaries),
            "high_tail_buffer_count": 0,
            "sampling_policy_mode": "uncertainty_only",
            "candidate_train_selected_count_per_system": dict(sorted(selected_count_per_system.items())),
            **dict(backend_result.runtime_artifacts or {}),
            "postprocess_phase_wall_time_sec": float(time.perf_counter() - started_at),
        },
    )
    trim_uncertainty_only_sampling_summary(Path(summary_output).resolve())
    try:
        summary_payload = read_json(summary_output)
        if isinstance(summary_payload, dict):
            append_sampling_summary_to_progress_log(context.get("sampling_progress_log_path"), summary_payload)
    except Exception:
        pass
    return Path(summary_output).resolve()
