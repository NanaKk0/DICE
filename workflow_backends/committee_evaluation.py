from __future__ import annotations

from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _aggregate_frame_metrics(frame_metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not frame_metrics:
        return {
            "force_std_max": 0.0,
            "force_std_avg": 0.0,
            "force_std_p90": 0.0,
        }
    max_values = [_as_float(metrics.get("force_std_max")) for metrics in frame_metrics]
    avg_values = [_as_float(metrics.get("force_std_avg", metrics.get("force_std_max"))) for metrics in frame_metrics]
    p90_values = [_as_float(metrics.get("force_std_p90", metrics.get("force_std_max"))) for metrics in frame_metrics]
    return {
        "force_std_max": float(max(max_values)),
        "force_std_avg": float(sum(avg_values) / len(avg_values)),
        "force_std_p90": float(max(p90_values)),
    }


def canonical_committee_evaluation_result(
    *,
    backend: str,
    committee_size: int,
    frame_metrics: list[dict[str, Any]],
    evaluation_mode: str,
    capabilities: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    normalized_capabilities = {
        "force_deviation": True,
        "committee_mean_labels": False,
        "mean_energy": False,
        "mean_forces": False,
        "mean_stress": False,
        "mean_virial": False,
    }
    normalized_capabilities.update(dict(capabilities or {}))
    normalized_frame_metrics = [dict(metrics or {}) for metrics in frame_metrics]
    result: dict[str, Any] = {
        **extra,
        "backend": str(backend),
        "committee_size": int(committee_size),
        "evaluation_mode": str(evaluation_mode),
        "capabilities": normalized_capabilities,
        "frame_metrics": normalized_frame_metrics,
    }
    result.update(_aggregate_frame_metrics(normalized_frame_metrics))
    return result


__all__ = ["canonical_committee_evaluation_result"]
