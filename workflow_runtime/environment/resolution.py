from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow_runtime.environment.schema import EnvironmentProfile, LocalTaskEnvironment


@dataclass(frozen=True, slots=True)
class LocalTaskExecution:
    role: str
    profile_id: str
    environment: LocalTaskEnvironment = field(default_factory=LocalTaskEnvironment)
    executables: dict[str, str] = field(default_factory=dict)

    def to_execution_context(self) -> "ExecutionContext":
        from workflow_runtime.environment.execution_context import local_task_execution_to_execution_context

        return local_task_execution_to_execution_context(self)

    def to_payload(self) -> dict[str, Any]:
        from workflow_runtime.environment.payloads import local_task_environment_payload

        return {
            "role": self.role,
            "profile_id": self.profile_id,
            "environment": local_task_environment_payload(self.environment),
            "executables": dict(self.executables),
        }


def resolve_local_task_execution(profile: EnvironmentProfile, role: str) -> LocalTaskExecution:
    normalized_role = str(role).strip()
    profile_id = profile.bindings.get(normalized_role)
    if profile_id is None:
        raise ValueError(f"no environment binding for role {normalized_role!r}")
    software_profile = profile.software_profiles.get(profile_id)
    if software_profile is None:
        raise ValueError(f"role {normalized_role!r} references missing software profile {profile_id!r}")
    return LocalTaskExecution(
        role=normalized_role,
        profile_id=software_profile.profile_id,
        environment=software_profile.environment,
        executables=dict(software_profile.executables),
    )
