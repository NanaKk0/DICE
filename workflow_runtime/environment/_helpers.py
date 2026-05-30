from __future__ import annotations

import os
import re
from typing import Any


ENV_VAR_RE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def string_mapping(value: Any, *, field_name: str) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return {str(key): str(item) for key, item in value.items()}


def string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value in (None, "", []):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings")
    return tuple(value)


def merge_env_values(target: dict[str, str], values: dict[str, str]) -> None:
    for key, value in values.items():
        target[str(key)] = expand_env_value(str(value), target)


def expand_env_value(value: str, env: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain") or ""
        return env.get(name, os.environ.get(name, ""))

    return ENV_VAR_RE.sub(replace, value)
