from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_runtime.environment._helpers import string_mapping, string_tuple


RETIRED_TOP_LEVEL_FIELDS = frozenset({"environments", "stages"})
RETIRED_PROFILE_FIELDS = frozenset(
    {
        "backend",
        "command",
        "check_commands",
        "conda",
        "envs",
        "python_executable",
        "prepend_script",
        "append_script",
        "plugin_path",
    }
)
RETIRED_ENVIRONMENT_FIELDS = frozenset(
    {
        "conda",
        "envs",
        "python_executable",
        "prepend_script",
        "append_script",
        "check_commands",
        "command",
        "plugin_path",
    }
)
RETIRED_BINDING_FIELDS = frozenset(
    {
        "environment",
        "executables",
        "command",
        "envs",
        "conda",
        "python_executable",
        "check_commands",
        "prepend_script",
        "append_script",
        "plugin_path",
    }
)


@dataclass(frozen=True, slots=True)
class LocalTaskEnvironment:
    source_list: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SoftwareProfile:
    profile_id: str
    environment: LocalTaskEnvironment = field(default_factory=LocalTaskEnvironment)
    executables: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    schema_version: int
    software_profiles: dict[str, SoftwareProfile]
    bindings: dict[str, str]
    path: Path | None = None


def load_environment_profile(path: str | Path) -> EnvironmentProfile:
    resolved_path = Path(path).resolve()
    payload = read_json(resolved_path)
    return parse_environment_profile(payload, path=resolved_path)


def parse_environment_profile(payload: dict[str, Any], *, path: str | Path | None = None) -> EnvironmentProfile:
    if not isinstance(payload, dict):
        raise TypeError("environment_profile must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != 3:
        raise ValueError(
            "environment_profile requires schema_version: 3 with software_profiles and bindings; "
            "v1/v2 environment_profile shapes are retired"
        )
    _reject_fields(payload, RETIRED_TOP_LEVEL_FIELDS, "environment_profile")

    raw_profiles = payload.get("software_profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("environment_profile.software_profiles must be a non-empty object")
    raw_bindings = payload.get("bindings")
    if not isinstance(raw_bindings, dict) or not raw_bindings:
        raise ValueError("environment_profile.bindings must be a non-empty object")

    software_profiles = {
        str(profile_id): _parse_software_profile(str(profile_id), spec)
        for profile_id, spec in raw_profiles.items()
    }
    bindings = _parse_bindings(raw_bindings, software_profiles=software_profiles)
    return EnvironmentProfile(
        schema_version=3,
        software_profiles=software_profiles,
        bindings=bindings,
        path=None if path is None else Path(path).resolve(),
    )


def _parse_software_profile(profile_id: str, spec: Any) -> SoftwareProfile:
    if not profile_id.strip():
        raise ValueError("environment_profile.software_profiles contains an empty profile id")
    if not isinstance(spec, dict):
        raise ValueError(f"environment_profile.software_profiles.{profile_id} must be an object")
    _reject_fields(spec, RETIRED_PROFILE_FIELDS, f"environment_profile.software_profiles.{profile_id}")
    allowed = {"environment", "executables"}
    _reject_unknown_fields(spec, allowed, f"environment_profile.software_profiles.{profile_id}")
    return SoftwareProfile(
        profile_id=profile_id,
        environment=_parse_environment(
            spec.get("environment", {}),
            field_name=f"environment_profile.software_profiles.{profile_id}.environment",
        ),
        executables=string_mapping(
            spec.get("executables", {}),
            field_name=f"environment_profile.software_profiles.{profile_id}.executables",
        ),
        raw=dict(spec),
    )


def _parse_bindings(raw_bindings: dict[Any, Any], *, software_profiles: dict[str, SoftwareProfile]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for role, value in raw_bindings.items():
        normalized_role = str(role).strip()
        if not normalized_role:
            raise ValueError("environment_profile.bindings contains an empty role")
        if isinstance(value, dict):
            _reject_fields(value, RETIRED_BINDING_FIELDS, f"environment_profile.bindings.{normalized_role}")
            if set(value) == {"profile"}:
                raise ValueError(
                    f"environment_profile.bindings.{normalized_role} must map directly to a software profile id string, "
                    "not an object with profile"
                )
            raise TypeError(f"environment_profile.bindings.{normalized_role} must be a software profile id string")
        profile_id = str(value).strip()
        if not profile_id:
            raise ValueError(f"environment_profile.bindings.{normalized_role} must be a non-empty string")
        if profile_id not in software_profiles:
            raise ValueError(f"role {normalized_role!r} references missing software profile {profile_id!r}")
        bindings[normalized_role] = profile_id
    return bindings


def _parse_environment(value: Any, *, field_name: str) -> LocalTaskEnvironment:
    if value in (None, {}):
        return LocalTaskEnvironment()
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    _reject_fields(value, RETIRED_ENVIRONMENT_FIELDS, field_name)
    _reject_unknown_fields(value, {"source_list"}, field_name)
    return LocalTaskEnvironment(
        source_list=string_tuple(value.get("source_list"), field_name=f"{field_name}.source_list"),
    )


def _reject_fields(payload: dict[str, Any], retired: frozenset[str], field_name: str) -> None:
    present = sorted(str(key) for key in payload if str(key) in retired)
    if present:
        raise ValueError(
            f"{field_name} contains retired environment_profile field(s): {', '.join(present)}; "
            "use schema_version: 3 with environment.source_list and profile-level executables"
        )


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported field(s): {', '.join(unknown)}")
