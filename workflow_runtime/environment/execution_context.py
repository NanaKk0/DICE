from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow_runtime.environment._helpers import merge_env_values, optional_string, string_mapping, string_tuple
from workflow_runtime.environment.resolution import LocalTaskExecution, resolve_local_task_execution
from workflow_runtime.environment.schema import EnvironmentProfile, parse_environment_profile


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    environment_id: str
    provider: str = "local"
    python_executable: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    executables: dict[str, str] = field(default_factory=dict)
    source_list: tuple[str, ...] = ()
    probes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | "ExecutionContext") -> "ExecutionContext":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise TypeError("execution context payload must be an object")
        return cls(
            environment_id=str(payload.get("environment_id", "")),
            provider=str(payload.get("provider", "local")),
            python_executable=optional_string(payload.get("python_executable")),
            env=string_mapping(payload.get("env", {}), field_name="execution_context.env"),
            executables=string_mapping(payload.get("executables", {}), field_name="execution_context.executables"),
            source_list=string_tuple(
                payload.get("source_list"),
                field_name="execution_context.source_list",
            ),
            probes=dict(payload.get("probes", {}) or {}),
        )

    def executable(self, name: str, *, required: bool = True) -> str | None:
        logical_name = str(name).strip()
        executable = self.executables.get(logical_name)
        if executable:
            return executable
        if not required:
            return None
        raise KeyError(
            f"execution context environment {self.environment_id!r} provider {self.provider!r} "
            f"does not define executable {logical_name!r}"
        )

    def merge_env(
        self,
        base_env: dict[str, str] | None = None,
        *overrides: dict[str, str] | None,
    ) -> dict[str, str]:
        merged = dict(base_env or {})
        merge_env_values(merged, self.env)
        for override in overrides:
            merge_env_values(merged, override or {})
        return merged

    def redacted_diagnostics(self, *, role: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "environment_id": self.environment_id,
            "provider": self.provider,
            "python_executable": "<redacted>" if self.python_executable else None,
            "env": {key: "<redacted>" for key in sorted(self.env)},
            "executables": {key: "<redacted>" for key in sorted(self.executables)},
            "source_list": ["<redacted>" for _ in self.source_list],
            "probes": dict(self.probes),
        }
        if role is not None:
            payload["role"] = str(role)
        return payload

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "environment_id": self.environment_id,
            "provider": self.provider,
            "executables": dict(self.executables),
            "source_list": list(self.source_list),
        }
        if self.python_executable is not None:
            payload["python_executable"] = self.python_executable
        if self.env:
            payload["env"] = dict(self.env)
        if self.probes:
            payload["probes"] = dict(self.probes)
        return payload


def local_task_execution_to_execution_context(task: LocalTaskExecution) -> ExecutionContext:
    return ExecutionContext(
        environment_id=task.profile_id,
        provider="local",
        env={},
        executables=dict(task.executables),
        source_list=tuple(task.environment.source_list),
        probes={
            "local_task": {
                "role": task.role,
                "profile_id": task.profile_id,
            }
        },
    )


def resolve_execution_context(profile: EnvironmentProfile, role: str) -> ExecutionContext:
    return local_task_execution_to_execution_context(resolve_local_task_execution(profile, role))


def execution_context_payload_for_role(raw_profile: Any, role: str) -> dict[str, Any] | None:
    if not isinstance(raw_profile, dict):
        return None
    task = resolve_local_task_execution(parse_environment_profile(raw_profile), role)
    context = local_task_execution_to_execution_context(task)
    payload = context.to_payload()
    payload["role"] = str(role)
    payload["profile_id"] = task.profile_id
    return payload


def coerce_execution_context(payload: Any) -> ExecutionContext | None:
    if payload in (None, ""):
        return None
    if isinstance(payload, ExecutionContext):
        return payload
    if isinstance(payload, dict):
        return ExecutionContext.from_payload(payload)
    raise TypeError("execution_context must be an ExecutionContext or object payload")
