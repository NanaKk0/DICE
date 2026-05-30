from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from workflow_core.common.io import write_json
from workflow_runtime.plan import PreflightPlan


def _append_preflight_result(
    payload: dict[str, Any],
    *,
    key: str,
    ok: bool,
    required: bool,
    detail: str,
    path: str | Path | None,
    assets: dict[str, Any],
    warnings: list[str],
) -> None:
    resolved_path = None if path is None else str(Path(path).resolve())
    payload["checks"].append(
        {
            "key": key,
            "ok": bool(ok),
            "required": bool(required),
            "detail": detail,
            "path": resolved_path,
        }
    )
    if resolved_path is not None:
        payload["assets"][key] = resolved_path
    payload["assets"].update(dict(assets or {}))
    payload["warnings"].extend(str(item) for item in warnings)
    if required and not ok:
        label = key if not detail else f"{key}: {detail}"
        payload["missing"].append(label)


def run_preflight_plan(plan: PreflightPlan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": plan.run_name,
        "round": int(plan.round_id),
        "runtime_root": None if plan.runtime_root is None else str(plan.runtime_root),
        "pipeline": list(plan.pipeline),
        "previous_round": int(plan.previous_round) if plan.previous_round is not None else None,
        "ok": True,
        "checks": [],
        "missing": [],
        "warnings": [],
        "assets": {},
        "config": dict(plan.config or {}),
    }
    for check in plan.checks:
        result = check.run()
        _append_preflight_result(
            payload,
            key=check.key,
            ok=bool(result.ok),
            required=bool(check.required),
            detail=str(result.detail or ""),
            path=result.path,
            assets=dict(result.assets or {}),
            warnings=list(result.warnings or []),
        )

    payload["ok"] = not payload["missing"]
    if plan.report_path is not None:
        write_json(Path(plan.report_path), payload)
    if not payload["ok"]:
        missing_summary = "; ".join(payload["missing"])
        raise RuntimeError(
            f"Preflight failed for round {plan.round_id}. Missing required assets: {missing_summary}."
        )
    return payload
