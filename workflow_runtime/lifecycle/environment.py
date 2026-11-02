from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Mapping

from workflow_runtime.execution.allocation import Allocation


_RUNTIME_KEYS = frozenset(
    {
        "descriptor_chunk_frames", "mace_workers", "mace_batch_size", "fp_workers",
        "mace_threads", "cpu_threads", "tf_gpu_allocator", "gmm_reference_spill",
        "batch_size",
    }
)


_HISTORICAL_CANONICAL_OPERATION_ROLES = (
    "teacher.mace.train",
    "teacher.mace.label",
    "teacher.mace.gmm_reference",
    "teacher.mace.gmm_score",
    "teacher.deepmd.tf.train",
    "teacher.deepmd.tf.label",
    "teacher.deepmd.pt.train",
    "teacher.deepmd.pt.label",
    "teacher.deepmd.pt.gmm_reference",
    "teacher.deepmd.pt.gmm_score",
    "student.deepmd.tf.train",
    "student.deepmd.tf.compress",
    "student.deepmd.tf.gmm_reference",
    "student.deepmd.tf.gmm_score",
    "student.deepmd.pt.train",
    "student.deepmd.pt.compress",
    "sampler.lammps.tf",
    "sampler.lammps.pt",
    "student.nep.train",
    "student.nep.gmm_reference",
    "student.nep.gmm_score",
    "sampler.gpumd",
)
CANONICAL_OPERATION_ROLES = (
    *_HISTORICAL_CANONICAL_OPERATION_ROLES,
    "student.deepmd.pt.gmm_screen",
    "student.deepmd.tf.neb",
    "student.deepmd.pt.neb",
    "student.nep.neb",
    "student.deepmd.tf.relax",
    "student.deepmd.pt.relax",
    "student.nep.relax",
    "first_principles.abacus.label",
    "first_principles.cp2k.label",
)
_REVISION_FOUR_OPERATION_ROLES = (
    *_HISTORICAL_CANONICAL_OPERATION_ROLES,
    "student.deepmd.pt.gmm_reference",
    "student.deepmd.pt.gmm_score",
    "first_principles.abacus.label",
    "first_principles.cp2k.label",
)
HISTORICAL_OPERATION_ROLE_ALIASES = {
    f"v2.{role}": role for role in _HISTORICAL_CANONICAL_OPERATION_ROLES
}
_CANONICAL_OPERATION_ROLE_SET = frozenset(CANONICAL_OPERATION_ROLES)
_REVISION_FOUR_OPERATION_ROLE_SET = frozenset(_REVISION_FOUR_OPERATION_ROLES)


@dataclass(frozen=True, slots=True)
class SoftwareProfile:
    profile_id: str
    source_list: tuple[str, ...]
    executables: dict[str, str]
    runtime: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    path: Path
    profiles: dict[str, SoftwareProfile]
    bindings: dict[str, str]


@dataclass(frozen=True, slots=True)
class EnvironmentPreflight:
    operation_roles: tuple[str, ...]
    evidence: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class LammpsExecutionPolicy:
    parallel_tasks: int
    active_thread_percentage: int

    @property
    def mps_enabled(self) -> bool:
        return self.parallel_tasks > 1


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Backend-facing resolved operation data; lifecycle state remains elsewhere."""

    role: str
    profile_id: str
    source_scripts: tuple[dict[str, str], ...]
    executables: dict[str, str]
    runtime: dict[str, Any]
    lammps_execution_policy: LammpsExecutionPolicy | None = None
    allocation: Allocation = field(default_factory=Allocation.local_default)

    def executable(self, logical_name: str) -> str:
        try:
            return self.executables[logical_name]
        except KeyError as exc:
            raise ValueError(f"lifecycle operation {self.role!r} has no executable {logical_name!r}") from exc


def load_environment_profile(path: str | Path) -> EnvironmentProfile:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read lifecycle environment profile {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"lifecycle environment profile {resolved} is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("environment profile must be a JSON object")
    canonical_envelope = (
        payload.get("schema") == "dice.environment-profile"
        and "schema_revision" not in payload
        and "schema_version" not in payload
    )
    revision_envelope = (
        payload.get("schema") == "dice.environment-profile"
        and payload.get("schema_revision") in {4, 5}
        and not isinstance(payload.get("schema_revision"), bool)
    )
    historical_envelope = (
        payload.get("schema_version") == 4
        and not isinstance(payload.get("schema_version"), bool)
    )
    if canonical_envelope:
        expected_fields = {"schema", "software_profiles", "bindings"}
    elif revision_envelope:
        expected_fields = {"schema", "schema_revision", "software_profiles", "bindings"}
    elif historical_envelope:
        expected_fields = {"schema_version", "software_profiles", "bindings"}
    else:
        raise ValueError(
            "environment profile requires stable schema 'dice.environment-profile' "
            "or an explicitly supported historical revision envelope"
        )
    if set(payload) != expected_fields:
        raise ValueError("environment profile contains unsupported top-level field(s)")
    raw_profiles = payload["software_profiles"]
    raw_bindings = payload["bindings"]
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("lifecycle environment profile software_profiles must be a non-empty object")
    if not isinstance(raw_bindings, dict):
        raise ValueError("lifecycle environment profile bindings must be an object")
    profiles = {str(profile_id): _parse_profile(str(profile_id), value) for profile_id, value in raw_profiles.items()}
    accepted_roles = (
        _REVISION_FOUR_OPERATION_ROLE_SET
        if historical_envelope
        or (revision_envelope and payload["schema_revision"] == 4)
        else _CANONICAL_OPERATION_ROLE_SET
    )
    bindings: dict[str, str] = {}
    for raw_role, raw_profile in raw_bindings.items():
        declared_role = str(raw_role).strip()
        role = HISTORICAL_OPERATION_ROLE_ALIASES.get(declared_role, declared_role)
        profile_id = str(raw_profile).strip()
        if ".coverage" in declared_role:
            raise ValueError(
                f"retired coverage binding {declared_role!r}; use the reference-tail "
                "with gmm_reference/gmm_score operation roles"
            )
        if role not in accepted_roles:
            raise ValueError(f"unknown environment operation role {declared_role!r}")
        if not role or not profile_id or profile_id not in profiles:
            raise ValueError(f"invalid environment binding {raw_role!r}: {raw_profile!r}")
        if role in bindings:
            raise ValueError(
                f"duplicate semantic environment role {role!r} declared as {declared_role!r}"
            )
        bindings[role] = profile_id
    if (
        revision_envelope and payload["schema_revision"] == 4
    ) or historical_envelope:
        profiles, bindings = _normalize_revision_four_pt_student_gmm(
            profiles, bindings
        )
    return EnvironmentProfile(path=resolved, profiles=profiles, bindings=bindings)


def _normalize_revision_four_pt_student_gmm(
    profiles: dict[str, SoftwareProfile], bindings: dict[str, str]
) -> tuple[dict[str, SoftwareProfile], dict[str, str]]:
    reference_role = "student.deepmd.pt.gmm_reference"
    score_role = "student.deepmd.pt.gmm_score"
    reference_profile = bindings.get(reference_role)
    score_profile = bindings.get(score_role)
    normalized_bindings = dict(bindings)
    normalized_profiles = dict(profiles)
    atom_budget_profiles = {
        bindings[role]
        for role in (
            "teacher.deepmd.pt.gmm_reference",
            "teacher.deepmd.pt.gmm_score",
        )
        if role in bindings
    }
    if reference_profile is not None and reference_profile == score_profile:
        normalized_bindings.pop(reference_role)
        normalized_bindings.pop(score_role)
        normalized_bindings["student.deepmd.pt.gmm_screen"] = reference_profile
        atom_budget_profiles.add(reference_profile)
    for profile_id in atom_budget_profiles:
        software = normalized_profiles[profile_id]
        runtime = dict(software.runtime)
        runtime.setdefault("batch_size", 9472)
        normalized_profiles[profile_id] = replace(software, runtime=runtime)
    return normalized_profiles, normalized_bindings


def _parse_profile(profile_id: str, value: Any) -> SoftwareProfile:
    if not profile_id or not isinstance(value, dict) or set(value) - {"environment", "executables", "runtime"}:
        raise ValueError(f"invalid lifecycle software profile {profile_id!r}")
    environment = value.get("environment", {})
    if not isinstance(environment, dict) or set(environment) - {"source_list"}:
        raise ValueError(f"invalid environment declaration for lifecycle profile {profile_id!r}")
    source_list = _string_list(environment.get("source_list", []), f"{profile_id}.environment.source_list")
    executables = value.get("executables", {})
    if not isinstance(executables, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in executables.items()):
        raise ValueError(f"lifecycle profile {profile_id!r} executables must be a string mapping")
    runtime = value.get("runtime", {})
    if not isinstance(runtime, dict) or set(runtime) - _RUNTIME_KEYS or any(isinstance(v, (dict, list)) for v in runtime.values()):
        raise ValueError(f"lifecycle profile {profile_id!r} runtime contains unsupported tuning")
    if "gmm_reference_spill" in runtime and not isinstance(runtime["gmm_reference_spill"], bool):
        raise ValueError(f"lifecycle profile {profile_id!r} runtime gmm_reference_spill must be boolean")
    if "batch_size" in runtime and (
        not isinstance(runtime["batch_size"], int)
        or isinstance(runtime["batch_size"], bool)
        or runtime["batch_size"] < 1
    ):
        raise ValueError(
            f"lifecycle profile {profile_id!r} runtime.batch_size must be a positive integer "
            "total-atom budget per same-topology DeepMD call, not a frame count"
        )
    return SoftwareProfile(profile_id, source_list, dict(executables), dict(runtime))


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def preflight_environment(
    profile: EnvironmentProfile,
    operation_roles: tuple[str, ...],
) -> EnvironmentPreflight:
    evidence: dict[str, dict[str, Any]] = {}
    profile_cache: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for role in operation_roles:
        profile_id = profile.bindings.get(role)
        if profile_id is None:
            if role == "student.deepmd.pt.gmm_screen" and {
                "student.deepmd.pt.gmm_reference",
                "student.deepmd.pt.gmm_score",
            } <= set(profile.bindings):
                raise ValueError(
                    "revision-4 PT student GMM migration requires gmm_reference "
                    "and gmm_score to bind the same software profile before they can "
                    "normalize to student.deepmd.pt.gmm_screen"
                )
            raise ValueError(f"no lifecycle environment binding for required role {role!r}")
        software = profile.profiles[profile_id]
        if role in {
            "student.deepmd.pt.gmm_screen",
            "teacher.deepmd.pt.gmm_reference",
            "teacher.deepmd.pt.gmm_score",
        }:
            batch_size = software.runtime.get("batch_size")
            if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or batch_size < 1
            ):
                raise ValueError(
                    f"lifecycle operation {role!r} requires positive integer "
                    "runtime.batch_size counting total atoms per same-topology "
                    "DeepMD call, not frames"
                )
        if role.startswith("first_principles."):
            _validate_first_principles_resources(software.runtime, role=role)
        if profile_id not in profile_cache:
            profile_cache[profile_id] = _probe_profile(profile.path.parent, software)
        common, activated = profile_cache[profile_id]
        role_evidence = {
            "profile_id": common["profile_id"],
            "source_scripts": [dict(item) for item in common["source_scripts"]],
            "executables": dict(common["executables"]),
            "runtime": dict(common["runtime"]),
        }
        if role in {"sampler.lammps.tf", "sampler.lammps.pt"}:
            role_evidence["lammps_execution_policy"] = (
                _resolve_lammps_execution_policy(activated, role=role)
            )
        evidence[role] = role_evidence
    return EnvironmentPreflight(operation_roles=operation_roles, evidence=evidence)


def _validate_first_principles_resources(
    runtime: dict[str, Any], *, role: str
) -> None:
    values: dict[str, int] = {}
    for name in ("fp_workers", "cpu_threads"):
        value = runtime.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"lifecycle operation {role!r} requires positive integer runtime.{name}"
            )
        values[name] = value
    available = len(os.sched_getaffinity(0))
    allocation = values["fp_workers"] * values["cpu_threads"]
    if allocation > available:
        raise ValueError(
            f"lifecycle operation {role!r} requests {allocation} FP CPUs but current affinity/cpuset exposes {available}"
        )


def operation_contexts(
    preflight: EnvironmentPreflight,
    *,
    allocations: Mapping[str, Allocation] | None = None,
) -> dict[str, OperationContext]:
    """Project resolved per-role execution evidence into backend-only contexts."""
    bound_allocations = dict(allocations or {})
    inherited_raw = os.environ.get("DICE_ALLOCATION_JSON")
    inherited = (
        None if inherited_raw in {None, ""} else Allocation.from_json(inherited_raw)
    )
    contexts: dict[str, OperationContext] = {}
    for role in preflight.operation_roles:
        evidence = preflight.evidence[role]
        contexts[role] = OperationContext(
            role=role,
            profile_id=str(evidence["profile_id"]),
            source_scripts=tuple(dict(item) for item in evidence["source_scripts"]),
            executables=dict(evidence["executables"]),
            runtime=dict(evidence["runtime"]),
            lammps_execution_policy=_project_lammps_execution_policy(evidence),
            allocation=bound_allocations.get(
                role,
                inherited if inherited is not None else Allocation.local_default(),
            ),
        )
    return contexts


def activated_operation_environment(context: OperationContext) -> dict[str, str]:
    """Resolve one operation's post-source environment in an isolated shell."""
    source_command = " ".join(
        f"source {shlex.quote(str(item['path']))};" for item in context.source_scripts
    )
    try:
        return _activated_environment(source_command)
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ValueError(f"lifecycle operation {context.role!r} environment activation failed") from exc


def _probe_profile(
    profile_root: Path,
    profile: SoftwareProfile,
) -> tuple[dict[str, Any], dict[str, str]]:
    scripts: list[Path] = []
    for raw_script in profile.source_list:
        script = Path(raw_script)
        script = script if script.is_absolute() else (profile_root / script)
        script = script.resolve()
        if not script.is_file():
            raise ValueError(f"lifecycle source script does not exist: {script}")
        scripts.append(script)
    source_command = " ".join(
        f"source {shlex.quote(str(script))};" for script in scripts
    )
    try:
        activated = _activated_environment(source_command)
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ValueError(
            f"lifecycle software profile {profile.profile_id!r} activation failed"
        ) from exc
    executable_evidence: dict[str, str] = {}
    search_path = activated.get("PATH")
    for name, executable in profile.executables.items():
        resolved = shutil.which(executable, path=search_path)
        if resolved is None:
            raise ValueError(
                f"lifecycle executable {name!r} ({executable!r}) is unavailable "
                "after profile activation"
            )
        executable_evidence[name] = resolved
    evidence = {
        "profile_id": profile.profile_id,
        "source_scripts": [
            {"path": str(path), "digest": _file_digest(path)} for path in scripts
        ],
        "executables": executable_evidence,
        "runtime": dict(profile.runtime),
    }
    return evidence, activated


def _resolve_lammps_execution_policy(
    activated: Mapping[str, str], *, role: str
) -> dict[str, int | str]:
    supported_dice_variables = {"DICE_LAMMPS_PARALLEL_TASKS"}
    unknown = sorted(
        name
        for name in activated
        if name.startswith("DICE_LAMMPS_") and name not in supported_dice_variables
    )
    if unknown:
        raise ValueError(
            f"lifecycle operation {role!r} has unsupported LAMMPS execution variable {unknown[0]!r}"
        )
    parallel_raw = activated.get("DICE_LAMMPS_PARALLEL_TASKS", "")
    if not parallel_raw.isdigit() or int(parallel_raw) < 1:
        raise ValueError(
            f"lifecycle operation {role!r} requires DICE_LAMMPS_PARALLEL_TASKS to be a positive integer"
        )
    percentage_raw = activated.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "")
    if percentage_raw and (
        not percentage_raw.isdigit() or not 1 <= int(percentage_raw) <= 100
    ):
        raise ValueError(
            f"lifecycle operation {role!r} requires CUDA_MPS_ACTIVE_THREAD_PERCENTAGE "
            "to be an integer from 1 through 100"
        )
    percentage = 100 if percentage_raw == "" else int(percentage_raw)
    return {
        "mode": "mps" if int(parallel_raw) > 1 else "serial",
        "parallel_tasks": int(parallel_raw),
        "active_thread_percentage": percentage,
    }


def _activated_environment(
    source_command: str,
    *,
    export_variables: tuple[str, ...] = (),
) -> dict[str, str]:
    marker = b"__DICE_ACTIVATED_ENVIRONMENT__\0"
    export_command = " ".join(
        f"export {shlex.quote(variable)};" for variable in export_variables
    )
    command = f"set -e; {source_command} {export_command} printf '__DICE_ACTIVATED_ENVIRONMENT__\\0'; /usr/bin/env -0"
    result = subprocess.run(["bash", "-c", command], capture_output=True, check=True)
    _, separator, payload = result.stdout.rpartition(marker)
    if not separator:
        raise ValueError("lifecycle operation environment activation did not produce environment evidence")
    environment: dict[str, str] = {}
    for entry in payload.split(b"\0"):
        if not entry:
            continue
        raw_name, equals, raw_value = entry.partition(b"=")
        if not equals:
            continue
        environment[raw_name.decode("utf-8")] = raw_value.decode("utf-8")
    return environment


def _project_lammps_execution_policy(evidence: dict[str, Any]) -> LammpsExecutionPolicy | None:
    raw = evidence.get("lammps_execution_policy")
    if raw is None:
        return None
    return LammpsExecutionPolicy(
        parallel_tasks=int(raw["parallel_tasks"]),
        active_thread_percentage=int(raw["active_thread_percentage"]),
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
