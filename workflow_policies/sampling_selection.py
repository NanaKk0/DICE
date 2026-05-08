from __future__ import annotations

from collections import defaultdict

import numpy as np

from workflow_backends.deepmd.online_monitor import compute_rmsd


def _quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    idx = float(q) * (len(sorted_values) - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _compute_tail_statistics(candidate_records, selection_cfg):
    decision_metric = str(selection_cfg.get("decision_metric", "force_std_max"))
    values = sorted(float(record["metrics"][decision_metric]) for record in candidate_records)

    if not values:
        return {
            "decision_metric": decision_metric,
            "tail_core_threshold": 0.0,
            "tail_shoulder_threshold": 0.0,
            "value_min": 0.0,
            "value_max": 0.0,
        }

    use_quantiles = bool(selection_cfg.get("tail_quantile_mode", False))
    if use_quantiles:
        tail_core_threshold = _quantile(values, float(selection_cfg.get("tail_core_quantile", 0.90)))
        tail_shoulder_threshold = _quantile(values, float(selection_cfg.get("tail_shoulder_quantile", 0.70)))
    else:
        tail_core_threshold = float(selection_cfg.get("tail_core_min_metric", 0.20))
        tail_shoulder_threshold = float(selection_cfg.get("tail_shoulder_min_metric", 0.14))

    tail_shoulder_threshold = min(float(tail_shoulder_threshold), float(tail_core_threshold))
    return {
        "decision_metric": decision_metric,
        "tail_core_threshold": float(tail_core_threshold),
        "tail_shoulder_threshold": float(tail_shoulder_threshold),
        "value_min": float(values[0]),
        "value_max": float(values[-1]),
    }


def _compute_job_persistent_stats(candidate_records, decision_metric, selection_cfg):
    grouped = defaultdict(list)
    for record in candidate_records:
        grouped[str(record["source_job_id"])].append(record)

    persistent_cfg = dict(selection_cfg.get("persistent_bonus", {}) or {})
    mean_threshold = float(persistent_cfg.get("mean_metric_threshold", 0.20))
    candidate_threshold = int(persistent_cfg.get("candidate_count_threshold", 10))
    failed_threshold = int(persistent_cfg.get("failed_count_threshold", 1))

    stats = {}
    for job_id, records in grouped.items():
        values = [float(item["metrics"][decision_metric]) for item in records]
        failed_like = sum(1 for item in records if item.get("frame_class") == "failed_uncertainty")
        mean_value = float(np.mean(values)) if values else 0.0
        max_value = float(np.max(values)) if values else 0.0
        persistent = (
            mean_value >= mean_threshold
            or len(records) >= candidate_threshold
            or failed_like >= failed_threshold
        )
        stats[job_id] = {
            "mean_metric": mean_value,
            "max_metric": max_value,
            "candidate_count": len(records),
            "failed_count": failed_like,
            "persistent": persistent,
        }
    return stats


def _normalized_tail_score(metric, tail_stats):
    lo = float(tail_stats["tail_shoulder_threshold"])
    hi = float(tail_stats["tail_core_threshold"])
    top = max(float(tail_stats["value_max"]), hi)
    metric = float(metric)

    if top <= lo:
        return 1.0 if metric >= hi else 0.0
    if metric >= hi:
        return 1.0 + (metric - hi) / max(top - hi, 1e-12)
    return max(0.0, (metric - lo) / max(hi - lo, 1e-12))


def _geometry_score(record, selection_cfg):
    geometry_cfg = dict(selection_cfg)
    soft_floor = float(geometry_cfg.get("geometry_soft_floor", 0.90))
    hard_floor = float(geometry_cfg.get("geometry_hard_floor", 0.85))
    min_pair = float(record["geometry"].get("min_pair_distance", float("inf")))
    if not np.isfinite(min_pair):
        return 0.0
    if min_pair <= hard_floor:
        return 1.0
    if min_pair >= soft_floor:
        return 0.0
    return max(0.0, (soft_floor - min_pair) / max(soft_floor - hard_floor, 1e-12))


def _assign_selection_band(metric, tail_stats):
    if metric >= float(tail_stats["tail_core_threshold"]):
        return "tail_core"
    if metric >= float(tail_stats["tail_shoulder_threshold"]):
        return "tail_shoulder"
    return "coverage"


def _annotate_records(candidate_records, selection_cfg):
    if not candidate_records:
        return [], {"decision_metric": str(selection_cfg.get("decision_metric", "force_std_max"))}

    tail_stats = _compute_tail_statistics(candidate_records, selection_cfg)
    decision_metric = str(tail_stats["decision_metric"])
    job_stats = _compute_job_persistent_stats(candidate_records, decision_metric, selection_cfg)
    score_weights = dict(selection_cfg.get("score_weights", {}) or {})
    w_tail = float(score_weights.get("tail", 3.0))
    w_geometry = float(score_weights.get("geometry", 1.0))
    w_persistent = float(score_weights.get("persistent", 1.5))
    w_novelty = float(score_weights.get("novelty", 0.5))
    persistent_cfg = dict(selection_cfg.get("persistent_bonus", {}) or {})
    persistent_enabled = bool(persistent_cfg.get("enabled", True))
    persistent_bonus_value = float(persistent_cfg.get("bonus_value", 0.15))

    values_desc = sorted(
        (float(record["metrics"][decision_metric]) for record in candidate_records),
        reverse=True,
    )
    rank_lookup = {}
    for idx, value in enumerate(values_desc):
        rank_lookup.setdefault(value, idx + 1)

    annotated = []
    total = max(len(candidate_records), 1)
    for record in candidate_records:
        item = dict(record)
        metric = float(item["metrics"][decision_metric])
        job_id = str(item["source_job_id"])
        job_info = job_stats.get(job_id, {})
        tail_score = _normalized_tail_score(metric, tail_stats)
        geometry_score = _geometry_score(item, selection_cfg)
        novelty_score = float(item.get("novelty_score", 0.0) or 0.0)
        persistent_bonus = 0.0
        if persistent_enabled and bool(job_info.get("persistent", False)):
            persistent_bonus = persistent_bonus_value

        selection_score = (
            w_tail * tail_score
            + w_geometry * geometry_score
            + w_persistent * persistent_bonus
            + w_novelty * novelty_score
        )
        item["selection_band"] = _assign_selection_band(metric, tail_stats)
        item["selection_score"] = float(selection_score)
        item["tail_rank"] = int(rank_lookup.get(metric, total))
        item["tail_quantile"] = float(1.0 - (item["tail_rank"] - 1) / total)
        item["persistent_bonus"] = float(persistent_bonus)
        item["score_breakdown"] = {
            "tail": float(tail_score),
            "geometry": float(geometry_score),
            "persistent": float(persistent_bonus),
            "novelty": float(novelty_score),
            "total": float(selection_score),
        }
        annotated.append(item)

    return annotated, tail_stats


def _select_with_traj_gap_priority(candidate_records, selection_cfg):
    min_frame_gap = int(selection_cfg.get("min_frame_gap", 5))
    max_per_traj = int(selection_cfg.get("max_candidates_per_traj", 20))

    grouped = defaultdict(list)
    for record in candidate_records:
        grouped[str(record["source_job_id"])].append(record)

    filtered = []

    for records in grouped.values():
        records = sorted(
            records,
            key=lambda item: (
                -float(item.get("selection_score", item.get("decision_metric", 0.0))),
                -float(item.get("decision_metric", 0.0)),
                int(item["source_frame_index"]),
            ),
        )
        kept = []
        for record in records:
            frame_index = int(record["source_frame_index"])
            if any(abs(frame_index - int(existing["source_frame_index"])) < min_frame_gap for existing in kept):
                continue
            kept.append(record)
            if len(kept) >= max_per_traj:
                break
        filtered.extend(kept)

    return filtered


def _allocate_band_budgets(filtered_records, selection_cfg):
    max_total = int(selection_cfg.get("max_candidates_total", 300))
    counts = defaultdict(int)
    for record in filtered_records:
        counts[str(record.get("selection_band", "coverage"))] += 1

    core_budget = int(round(max_total * float(selection_cfg.get("tail_core_fraction", 0.70))))
    shoulder_budget = int(round(max_total * float(selection_cfg.get("tail_shoulder_fraction", 0.20))))
    coverage_budget = max_total - core_budget - shoulder_budget
    budgets = {
        "tail_core": max(0, core_budget),
        "tail_shoulder": max(0, shoulder_budget),
        "coverage": max(0, coverage_budget),
    }

    unused = 0
    for band in ("tail_core", "tail_shoulder", "coverage"):
        available = counts.get(band, 0)
        if budgets[band] > available:
            unused += budgets[band] - available
            budgets[band] = available

    for band in ("tail_core", "tail_shoulder", "coverage"):
        if unused <= 0:
            break
        available_extra = max(0, counts.get(band, 0) - budgets[band])
        if available_extra <= 0:
            continue
        take = min(unused, available_extra)
        budgets[band] += take
        unused -= take

    return budgets


def _deduplicate_by_rmsd_high_score(records, selection_cfg):
    threshold = float(selection_cfg.get("rmsd_threshold", 0.15))
    selected = []

    ordered = sorted(
        records,
        key=lambda item: (
            -float(item.get("selection_score", item.get("decision_metric", 0.0))),
            -float(item.get("decision_metric", 0.0)),
            int(item.get("source_frame_index", 0)),
        ),
    )

    for record in ordered:
        atoms = record["atoms"]
        duplicate = False
        for existing in selected:
            if len(existing["atoms"]) != len(atoms):
                continue
            if existing["atoms"].get_chemical_symbols() != atoms.get_chemical_symbols():
                continue
            if compute_rmsd(existing["atoms"], atoms) < threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(record)

    return selected


def select_candidate_frames(candidate_records, selection_cfg):
    if not candidate_records:
        return []

    annotated, tail_stats = _annotate_records(candidate_records, selection_cfg)
    filtered = _select_with_traj_gap_priority(annotated, selection_cfg)
    budgets = _allocate_band_budgets(filtered, selection_cfg)

    grouped = defaultdict(list)
    for record in filtered:
        grouped[str(record.get("selection_band", "coverage"))].append(record)

    chosen = []
    for band in ("tail_core", "tail_shoulder", "coverage"):
        records = sorted(
            grouped.get(band, []),
            key=lambda item: (
                -float(item.get("selection_score", 0.0)),
                -float(item.get("decision_metric", 0.0)),
                int(item.get("source_frame_index", 0)),
            ),
        )
        chosen.extend(records[: int(budgets.get(band, 0))])

    deduped = _deduplicate_by_rmsd_high_score(chosen, selection_cfg)
    max_total = int(selection_cfg.get("max_candidates_total", 300))
    return sorted(
        deduped[:max_total],
        key=lambda item: (
            -float(item.get("selection_score", 0.0)),
            -float(item.get("decision_metric", 0.0)),
            str(item.get("source_job_id", "")),
            int(item.get("source_frame_index", 0)),
        ),
    )

