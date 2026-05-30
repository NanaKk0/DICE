from __future__ import annotations

import shlex
from typing import Any


def _normalize_command(value: Any, *, default_executable: str) -> list[str]:
    if value in (None, "", "auto"):
        return [default_executable]
    if isinstance(value, str):
        command = shlex.split(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        command = list(value)
    else:
        raise TypeError(f"Invalid command specification: {value!r}")
    if not command:
        raise RuntimeError(f"cannot build {default_executable} command from an empty command specification")
    return command


def resolve_nep_command(nep_cfg: dict) -> list[str]:
    return _normalize_command(dict(nep_cfg or {}).get("command"), default_executable="nep")


def resolve_gpumd_command(gpumd_cfg: dict) -> list[str]:
    return _normalize_command(dict(gpumd_cfg or {}).get("command"), default_executable="gpumd")
