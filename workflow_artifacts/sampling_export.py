from __future__ import annotations

from collections import Counter
from pathlib import Path

from ase.io import write

from workflow_core.common.io import append_jsonl, write_candidate_xyz, write_json
from workflow_core.labeling.split_stats import summarize_sampling_records_by_system


def _compact_sampling_system_stats(summary: dict | None) -> dict | None:
    if not isinstance(summary, dict):
        return None
    systems = summary.get("systems")
    totals = summary.get("totals")
    if not isinstance(systems, dict) or not isinstance(totals, dict):
        return None

    def _project(payload: dict) -> dict:
        return {
            "accurate": int(payload.get("accurate", 0) or 0),
            "candidate": int(payload.get("candidate", 0) or 0),
            "failed": int(payload.get("failed", 0) or 0),
            "accurate_ratio": float(payload.get("accurate_ratio", 0.0) or 0.0),
            "candidate_ratio": float(payload.get("candidate_ratio", 0.0) or 0.0),
            "failed_ratio": float(payload.get("failed_ratio", 0.0) or 0.0),
        }

    return {
        "group_by": "formula",
        "systems": {key: _project(value) for key, value in sorted(systems.items())},
        "totals": _project(totals),
    }


def export_selected_candidates(selected_records, output_xyz, output_jsonl):
    output_xyz = Path(output_xyz).resolve()
    output_jsonl = Path(output_jsonl).resolve()
    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    output_xyz.write_text("", encoding="utf-8")
    output_jsonl.write_text("", encoding="utf-8")

    for record in selected_records:
        atoms = record["atoms"].copy()
        metadata = {
            "source_job_id": record["source_job_id"],
            "source_frame_index": int(record["source_frame_index"]),
            "driver_model_index": int(record["driver_model_index"]),
            "decision_metric": float(record["decision_metric"]),
            "force_std_max": float(record["metrics"]["force_std_max"]),
            "force_std_avg": float(record["metrics"]["force_std_avg"]),
            "force_std_p90": float(record["metrics"]["force_std_p90"]),
            "min_pair_distance": float(record["geometry"]["min_pair_distance"]),
            "frame_class": record["frame_class"],
            "novelty_score": float(record.get("novelty_score", 0.0) or 0.0),
            "rmsd_to_last_saved": float(record.get("rmsd_to_last_saved", float("inf"))),
            "rmsd_to_prev_eval": float(record.get("rmsd_to_prev_eval", float("inf"))),
            "geometry_changed": bool(record.get("geometry_changed", False)),
            "save_reason": record.get("save_reason"),
        }
        if "selection_band" in record:
            metadata["selection_band"] = record.get("selection_band")
        if "selection_score" in record:
            metadata["selection_score"] = float(record.get("selection_score"))
        if "tail_rank" in record:
            metadata["tail_rank"] = int(record.get("tail_rank"))
        if "tail_quantile" in record:
            metadata["tail_quantile"] = float(record.get("tail_quantile"))
        if "persistent_bonus" in record:
            metadata["persistent_bonus"] = float(record.get("persistent_bonus"))
        if "score_breakdown" in record:
            metadata["score_breakdown"] = record.get("score_breakdown")
        for key, value in metadata.items():
            atoms.info[key] = value

        write(
            output_xyz,
            atoms,
            format="extxyz",
            append=output_xyz.stat().st_size > 0,
        )
        append_jsonl(output_jsonl, metadata)

    return output_xyz, output_jsonl


def export_selected_candidates_xyz_only(selected_records, output_xyz):
    output_xyz = Path(output_xyz).resolve()
    output_xyz.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    for record in selected_records:
        atoms = record["atoms"].copy()
        metadata = {
            "source_job_id": record["source_job_id"],
            "source_frame_index": int(record["source_frame_index"]),
            "driver_model_index": int(record["driver_model_index"]),
            "decision_metric": float(record["decision_metric"]),
            "force_std_max": float(record["metrics"]["force_std_max"]),
            "force_std_avg": float(record["metrics"]["force_std_avg"]),
            "force_std_p90": float(record["metrics"]["force_std_p90"]),
            "frame_class": record["frame_class"],
        }
        for key, value in metadata.items():
            atoms.info[key] = value
        atoms_list.append(atoms)

    return write_candidate_xyz(output_xyz, atoms_list)


def write_sampling_summary_fast(
    summary_path,
    committee_size,
    usable_committee_size,
    total_jobs,
    total_frames,
    accurate_count,
    candidate_count,
    failed_uncertainty_count,
    selected_count,
    decision_metric,
    level_f_lo,
    level_f_hi,
    selected_xyz,
    selected_jsonl=None,
    stop_high_unc_threshold=None,
    high_unc_count=0,
    selected_decision_values=None,
    selected_per_job_counts=None,
    sampling_system_stats=None,
    extra_summary=None,
):
    selected_decision_values = list(selected_decision_values or [])
    effective_sample_count = int(total_frames)

    summary = {
        "committee_size": int(committee_size),
        "usable_committee_size": int(usable_committee_size),
        "total_jobs": int(total_jobs),
        "total_frames": int(total_frames),
        "accurate_count": int(accurate_count),
        "candidate_count": int(candidate_count),
        "failed_uncertainty_count": int(failed_uncertainty_count),
        "selected_count": int(selected_count),
        "effective_sample_count": effective_sample_count,
        "high_unc_count": int(high_unc_count),
        "high_unc_ratio": float(high_unc_count / effective_sample_count) if effective_sample_count else 0.0,
        "candidate_ratio": float(candidate_count / total_frames) if total_frames else 0.0,
        "failed_ratio": float(failed_uncertainty_count / total_frames) if total_frames else 0.0,
        "sampling_system_stats": _compact_sampling_system_stats(sampling_system_stats),
    }
    if extra_summary:
        summary.update(extra_summary)
    return write_json(summary_path, summary)


def write_sampling_summary(
    summary_path,
    committee_size,
    usable_committee_size,
    total_jobs,
    total_frames,
    frame_records,
    selected_records,
    decision_metric,
    level_f_lo,
    level_f_hi,
    selected_xyz,
    selected_jsonl,
    stop_high_unc_threshold=None,
    extra_summary=None,
):
    counts = Counter(record["frame_class"] for record in frame_records)
    decision_values = [
        float(record["decision_metric"])
        for record in frame_records
        if "decision_metric" in record
    ]
    high_unc_count = 0
    if stop_high_unc_threshold is not None:
        threshold = float(stop_high_unc_threshold)
        high_unc_count = sum(1 for value in decision_values if value >= threshold)
    effective_sample_count = int(len(decision_values))
    system_stats = _compact_sampling_system_stats(summarize_sampling_records_by_system(frame_records, selected_records))
    summary = {
        "committee_size": int(committee_size),
        "usable_committee_size": int(usable_committee_size),
        "total_jobs": int(total_jobs),
        "total_frames": int(total_frames),
        "accurate_count": int(counts.get("accurate", 0)),
        "candidate_count": int(counts.get("candidate", 0)),
        "failed_uncertainty_count": int(counts.get("failed_uncertainty", 0)),
        "failed_geometry_count": int(counts.get("failed_geometry", 0)),
        "suspicious_candidate_count": int(counts.get("suspicious_candidate", 0)),
        "selected_count": int(len(selected_records)),
        "effective_sample_count": effective_sample_count,
        "high_unc_count": int(high_unc_count),
        "high_unc_ratio": float(high_unc_count / effective_sample_count) if effective_sample_count else 0.0,
        "candidate_ratio": float((counts.get("candidate", 0) + counts.get("suspicious_candidate", 0)) / total_frames) if total_frames else 0.0,
        "failed_ratio": float((counts.get("failed_uncertainty", 0) + counts.get("failed_geometry", 0)) / total_frames) if total_frames else 0.0,
        "sampling_system_stats": system_stats,
    }
    if extra_summary:
        summary.update(extra_summary)
    return write_json(summary_path, summary)

