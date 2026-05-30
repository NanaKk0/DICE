from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeepMDNanWatchdogConfig:
    enabled: bool = True
    terminate_timeout_seconds: float = 10.0
    watch_lcurve_fallback: bool = True
    lcurve_poll_interval_seconds: float = 1.5


@dataclass(frozen=True, slots=True)
class DeepMDExecutionConfig:
    train_command: list[str]
    freeze_command: list[str]
    env: dict[str, str]
    max_retries_per_member: int = 2
    retry_seed_stride: int = 1000
    nan_watchdog: DeepMDNanWatchdogConfig = DeepMDNanWatchdogConfig()


def normalize_command(value: str | list[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError(f"Invalid command value: {value!r}")


def normalize_dp_module_command(command: list[str]) -> list[str]:
    if not command:
        return command
    if command[0] == "dp":
        return [sys.executable, "-m", "deepmd", *command[1:]]
    return list(command)


def normalize_env(value: Any, *, field_name: str = "execution.env") -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping of environment variables")
    return {str(key): str(item) for key, item in value.items()}


def merge_execution_env(
    base_env: dict[str, str] | None = None,
    *overrides: dict[str, Any] | None,
) -> dict[str, str]:
    merged = dict(os.environ if base_env is None else base_env)
    for override in overrides:
        if not override:
            continue
        merged.update({str(key): str(value) for key, value in override.items()})
    return merged


def normalize_nan_watchdog_config(
    payload: dict[str, Any],
    *,
    default_enabled: bool = True,
    field_name: str = "execution.nan_watchdog",
) -> DeepMDNanWatchdogConfig:
    watchdog = payload.get("nan_watchdog", {})
    if watchdog in (None, {}):
        return DeepMDNanWatchdogConfig(enabled=default_enabled)
    if not isinstance(watchdog, dict):
        raise TypeError(f"{field_name} must be an object")
    enabled = watchdog.get("enabled", default_enabled)
    if not isinstance(enabled, bool):
        raise TypeError(f"{field_name}.enabled must be a boolean")
    return DeepMDNanWatchdogConfig(
        enabled=enabled,
        terminate_timeout_seconds=float(watchdog.get("terminate_timeout_seconds", 10.0)),
        watch_lcurve_fallback=bool(watchdog.get("watch_lcurve_fallback", True)),
        lcurve_poll_interval_seconds=float(watchdog.get("lcurve_poll_interval_seconds", 1.5)),
    )


def normalize_deepmd_execution(
    payload: dict[str, Any] | None,
    *,
    default_train_command: list[str],
    default_freeze_command: list[str],
    legacy_command_keys: bool = True,
    env_field_name: str = "execution.env",
    watchdog_field_name: str = "execution.nan_watchdog",
) -> DeepMDExecutionConfig:
    payload = dict(payload or {})
    raw_execution = payload.get("execution", {})
    if raw_execution is None:
        raw_execution = {}
    if not isinstance(raw_execution, dict):
        raise TypeError("execution must be an object")
    execution = dict(raw_execution)

    def _get(*names: str) -> Any:
        for name in names:
            if name in execution:
                return execution[name]
            if legacy_command_keys and name in payload:
                return payload[name]
        return None

    train_command = normalize_dp_module_command(
        normalize_command(_get("train_command", "dp_command"), default_train_command)
    )
    freeze_command = normalize_dp_module_command(
        normalize_command(_get("freeze_command"), default_freeze_command)
    )
    env_value = execution.get("env", payload.get("env") if legacy_command_keys else None)
    retries_value = execution.get(
        "max_retries_per_member",
        payload.get("max_retries_per_member", 2) if legacy_command_keys else 2,
    )
    seed_stride_value = execution.get(
        "retry_seed_stride",
        payload.get("retry_seed_stride", 1000) if legacy_command_keys else 1000,
    )
    watchdog_payload = dict(payload)
    if "nan_watchdog" in execution:
        watchdog_payload["nan_watchdog"] = execution["nan_watchdog"]

    return DeepMDExecutionConfig(
        train_command=train_command,
        freeze_command=freeze_command,
        env=normalize_env(env_value, field_name=env_field_name),
        max_retries_per_member=max(0, int(retries_value)),
        retry_seed_stride=max(1, int(seed_stride_value)),
        nan_watchdog=normalize_nan_watchdog_config(
            watchdog_payload,
            field_name=watchdog_field_name,
        ),
    )
