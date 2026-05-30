from __future__ import annotations

from pathlib import Path
from typing import Callable

ArtifactCheckFactory = Callable[[Path], bool]

_CONTRACTS: dict[str, ArtifactCheckFactory] = {}
_DEFAULTS_BOOTSTRAPPED = False


def register_artifact_contract(
    contract_id: str,
    check_fn: ArtifactCheckFactory,
    *,
    replace: bool = False,
) -> None:
    resolved_id = str(contract_id).strip()
    if not resolved_id:
        raise ValueError("artifact contract id must be non-empty")
    if not callable(check_fn):
        raise TypeError("artifact contract check must be callable")
    if resolved_id in _CONTRACTS and not replace:
        raise ValueError(f"artifact contract already registered: {resolved_id}")
    _CONTRACTS[resolved_id] = check_fn


def artifact_check_for(contract_id: str, root: str | Path) -> Callable[[], bool]:
    _ensure_default_contracts_registered()
    resolved_id = str(contract_id).strip()
    try:
        check_fn = _CONTRACTS[resolved_id]
    except KeyError as exc:
        raise ValueError(f"no artifact contract registered for {resolved_id!r}") from exc
    root_path = Path(root)
    return lambda: bool(check_fn(root_path))


def registered_artifact_contract_ids() -> tuple[str, ...]:
    _ensure_default_contracts_registered()
    return tuple(sorted(_CONTRACTS))


def _ensure_default_contracts_registered() -> None:
    global _DEFAULTS_BOOTSTRAPPED
    if _DEFAULTS_BOOTSTRAPPED:
        return
    _DEFAULTS_BOOTSTRAPPED = True
    try:
        from workflow_core.artifact_contracts import ensure_runtime_artifact_contracts_registered
    except ModuleNotFoundError as exc:
        if exc.name == "workflow_core":
            return
        raise

    ensure_runtime_artifact_contracts_registered()
