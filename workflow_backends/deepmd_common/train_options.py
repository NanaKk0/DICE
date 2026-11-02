from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_SKIP_NEIGHBOR_STAT = "skip_neighbor_stat"
_NATIVE_SKIP_NEIGHBOR_STAT = "--skip-neighbor-stat"


def append_train_options(
    arguments: list[str], options: Mapping[str, Any] | None
) -> None:
    """Translate the closed, normalized DeepMD train option at the native seam."""

    if options is None:
        return
    unknown = sorted(set(options) - {_SKIP_NEIGHBOR_STAT})
    if unknown:
        raise ValueError(
            "DeepMD train options contain unsupported field(s): "
            + ", ".join(unknown)
        )
    enabled = options.get(_SKIP_NEIGHBOR_STAT, False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "DeepMD train option skip_neighbor_stat must be a boolean"
        )
    if not enabled:
        return
    if _NATIVE_SKIP_NEIGHBOR_STAT in arguments:
        raise ValueError(
            "DeepMD native train arguments contain --skip-neighbor-stat more than once"
        )
    arguments.append(_NATIVE_SKIP_NEIGHBOR_STAT)


def reject_automatic_sel_for_skip_neighbor_stat(
    rendered: Mapping[str, Any], options: Mapping[str, Any] | None
) -> None:
    """Reject native automatic ``sel`` values without changing rendered input."""

    if not options or options.get(_SKIP_NEIGHBOR_STAT) is not True:
        return
    paths: list[tuple[str, str]] = []
    _collect_automatic_sel(rendered.get("model"), "model", paths)
    if paths:
        locations = ", ".join(
            f"{path}={value!r}" for path, value in paths
        )
        raise ValueError(
            "DeepMD train with skip_neighbor_stat rejects automatic native "
            f"sel at JSON path(s): {locations}"
        )


def _collect_automatic_sel(
    value: Any, path: str, found: list[tuple[str, str]]
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "sel" and isinstance(child, str) and child in {"auto", "auto:factor"}:
                found.append((child_path, child))
            _collect_automatic_sel(child, child_path, found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_automatic_sel(child, f"{path}[{index}]", found)


__all__ = [
    "append_train_options",
    "reject_automatic_sel_for_skip_neighbor_stat",
]
