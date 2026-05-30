from __future__ import annotations

from typing import Any

from workflow_runtime.environment.schema import (
    EnvironmentProfile,
    LocalTaskEnvironment,
    SoftwareProfile,
)


def summarize_environment_profile(profile: EnvironmentProfile) -> dict[str, Any]:
    return {
        "path": None if profile.path is None else str(profile.path),
        "schema_version": profile.schema_version,
        "software_profiles": {
            profile_id: {
                "source_count": len(spec.environment.source_list),
                "executables": sorted(spec.executables),
            }
            for profile_id, spec in sorted(profile.software_profiles.items())
        },
        "bindings": dict(sorted(profile.bindings.items())),
    }


def environment_profile_payload(profile: EnvironmentProfile) -> dict[str, Any]:
    return {
        "schema_version": profile.schema_version,
        "software_profiles": {
            profile_id: _software_profile_payload(spec)
            for profile_id, spec in sorted(profile.software_profiles.items())
        },
        "bindings": dict(sorted(profile.bindings.items())),
    }


def _software_profile_payload(spec: SoftwareProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    environment = local_task_environment_payload(spec.environment)
    if environment:
        payload["environment"] = environment
    if spec.executables:
        payload["executables"] = dict(spec.executables)
    return payload


def local_task_environment_payload(environment: LocalTaskEnvironment) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if environment.source_list:
        payload["source_list"] = list(environment.source_list)
    return payload
