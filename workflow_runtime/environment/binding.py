from __future__ import annotations

from typing import Any

from workflow_runtime.environment.execution_context import execution_context_payload_for_role


def context_with_execution_context(
    cfg: dict[str, Any],
    *,
    role: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(context or {})
    if "execution_context" in merged:
        return merged
    raw_profile = dict(cfg or {}).get("environment_profile")
    payload = execution_context_payload_for_role(raw_profile, role)
    if payload is not None:
        merged["execution_context"] = payload
    return merged
