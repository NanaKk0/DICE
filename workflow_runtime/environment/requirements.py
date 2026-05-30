from __future__ import annotations

from typing import Any, Iterable

from workflow_runtime.environment.schema import EnvironmentProfile, parse_environment_profile


STATIC_EXECUTABLE_KEYS_BY_ROLE = {
    "teacher_backend.mace.foundation_train": ("mace_run_train",),
    "teacher_backend.mace.finetune": ("mace_run_train",),
    "teacher_backend.deepmd.foundation_train": ("dp",),
    "teacher_backend.deepmd.finetune": ("dp",),
    "student_backend.deepmd.train": ("dp",),
    "student_backend.nep.train": ("nep",),
    "student_backend.nep.evaluate": ("nep",),
    "sampling_backend.lammps.sample": ("lmp",),
    "sampling_backend.gpumd.sample": ("gpumd",),
}


def required_executable_keys_for_role(role: str) -> tuple[str, ...]:
    return STATIC_EXECUTABLE_KEYS_BY_ROLE.get(str(role).strip(), ())


def validate_environment_bindings(profile_or_payload: EnvironmentProfile | dict[str, Any], roles: Iterable[str]) -> None:
    profile = _coerce_environment_profile(profile_or_payload)
    selected_roles = tuple(sorted({str(role).strip() for role in roles if str(role).strip()}))
    missing_bindings = [role for role in selected_roles if role not in profile.bindings]
    missing_executables: list[tuple[str, str, str]] = []
    for role in selected_roles:
        profile_id = profile.bindings.get(role)
        if profile_id is None:
            continue
        software_profile = profile.software_profiles[profile_id]
        for executable_key in required_executable_keys_for_role(role):
            if executable_key not in software_profile.executables:
                missing_executables.append((role, profile_id, executable_key))
    messages: list[str] = []
    if missing_bindings:
        messages.append("missing environment binding(s): " + ", ".join(missing_bindings))
    if missing_executables:
        details = ", ".join(
            f"{role} via profile {profile_id!r} missing executable key {key!r}"
            for role, profile_id, key in missing_executables
        )
        messages.append("missing executable key(s): " + details)
    if messages:
        raise ValueError("; ".join(messages))


def _coerce_environment_profile(profile_or_payload: EnvironmentProfile | dict[str, Any]) -> EnvironmentProfile:
    if isinstance(profile_or_payload, EnvironmentProfile):
        return profile_or_payload
    if isinstance(profile_or_payload, dict):
        return parse_environment_profile(profile_or_payload)
    raise TypeError("environment binding validation requires an EnvironmentProfile or v3 payload")
