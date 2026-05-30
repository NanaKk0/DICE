from __future__ import annotations

from typing import Any


def resolve_dft_screening_policy(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    policy = dict(raw_cfg.get("dft_screening_policy", {}) or {})
    budget_per_system = int(policy.get("budget_per_system", 20))
    total_budget = max(1, budget_per_system) * 8
    stop_current = int(policy.get("stop_student_round_when_current_screened_count_gte", total_budget))
    stop_cumulative = int(policy.get("stop_student_round_when_cumulative_screened_count_gte", total_budget))
    return {
        "enabled": bool(policy.get("enabled", False)),
        "selection_mode": str(policy.get("selection_mode", "bucket_random_per_system")),
        "budget_per_system": budget_per_system,
        "random_seed": int(policy.get("random_seed", 20260503)),
        "signature_neighbor_order": max(1, int(policy.get("signature_neighbor_order", 3))),
        "signature_quantiles": [float(item) for item in policy.get("signature_quantiles", [0.10, 0.50, 0.90])],
        "bucket_width": float(policy.get("bucket_width", 4.0)),
        "keep_top_per_bucket": max(1, int(policy.get("keep_top_per_bucket", 1))),
        "stop_student_round_when_current_screened_count_gte": stop_current,
        "stop_student_round_when_cumulative_screened_count_gte": stop_cumulative,
    }
