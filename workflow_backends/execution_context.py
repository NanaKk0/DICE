from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from workflow_runtime.environment.execution_context import ExecutionContext, coerce_execution_context
from workflow_runtime.environment.shell_rendering import render_inline_shell_command


@dataclass(frozen=True, slots=True)
class BackendExecution:
    context: ExecutionContext | None
    role: str | None = None

    @property
    def configured(self) -> bool:
        return self.context is not None

    def optional_executable(self, name: str) -> str | None:
        if self.context is None:
            return None
        return self.context.executable(name, required=False)

    def required_executable(self, name: str) -> str | None:
        if self.context is None:
            logical_name = str(name).strip()
            raise KeyError(
                f"backend role {self.role!r} requires executable {logical_name!r}, "
                "but no script-first execution context was provided"
            )
        return self.context.executable(name, required=True)

    def python_executable(self) -> str | None:
        if self.context is None:
            return None
        return self.context.python_executable

    def source_list(self) -> tuple[str, ...]:
        if self.context is None:
            return ()
        return tuple(self.context.source_list)

    def shell_command(self, command: list[str] | tuple[str, ...]) -> list[str]:
        command_items = [str(item) for item in command]
        source_list = self.source_list()
        if not source_list:
            return command_items
        return ["bash", "-lc", render_inline_shell_command(command_items, source_list=source_list)]

    def merge_env(
        self,
        base_env: dict[str, str] | None = None,
        *overrides: dict[str, Any] | None,
    ) -> dict[str, str]:
        base = dict(os.environ if base_env is None else base_env)
        if self.context is not None:
            merged = self.context.merge_env(base)
        else:
            merged = base
        for override in overrides:
            _merge_env_values(merged, override or {})
        return merged

    def redacted_diagnostics(self) -> dict[str, Any]:
        if self.context is None:
            payload: dict[str, Any] = {"configured": False}
            if self.role is not None:
                payload["role"] = str(self.role)
            return payload
        return self.context.redacted_diagnostics(role=self.role)


@dataclass(frozen=True, slots=True)
class BackendRequestContext:
    context: dict[str, Any]
    execution: BackendExecution


def resolve_backend_execution(request: Any, *, role: str | None = None) -> BackendExecution:
    explicit_context = getattr(request, "execution_context", None)
    if explicit_context not in (None, ""):
        return BackendExecution(coerce_execution_context(explicit_context), role=role)
    request_context = getattr(request, "context", None)
    if isinstance(request_context, dict):
        return BackendExecution(coerce_execution_context(request_context.get("execution_context")), role=role)
    return BackendExecution(None, role=role)


def normalize_backend_request_context(request: Any, *, role: str | None = None) -> BackendRequestContext:
    request_context = getattr(request, "context", None)
    context = dict(request_context or {}) if isinstance(request_context, dict) else {}
    explicit_context = getattr(request, "execution_context", None)
    if explicit_context not in (None, ""):
        context["execution_context"] = explicit_context
    return BackendRequestContext(
        context=context,
        execution=resolve_backend_execution(request, role=role),
    )


def _merge_env_values(target: dict[str, str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] = _expand_env_value(str(value), target)


def _expand_env_value(value: str, env: dict[str, str]) -> str:
    import re

    pattern = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain") or ""
        return env.get(name, os.environ.get(name, ""))

    return pattern.sub(replace, value)
