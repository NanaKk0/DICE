from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TOP_LEVEL_FIELDS = {
    "schema",
    "schema_revision",
    "targets",
    "target_pools",
    "resource_classes",
    "bindings",
}
_TARGET_FIELDS = {
    "adapter",
    "max_in_flight",
    "host_alias",
    "remote_root",
    "stage_mode",
    "working_root",
    "partition",
    "account",
}
_CANONICAL_TARGET_FIELDS = _TARGET_FIELDS | {"qos", "resource_policy"}
_RESOURCE_POLICY_FIELDS = {
    "cpu",
    "memory",
    "expected_cpus_per_gpu",
    "expected_memory_mb_per_gpu",
}
_POOL_FIELDS = {"targets"}
_RESOURCE_FIELDS = {
    "nodes",
    "processes_per_node",
    "gpus_per_node",
    "cpus_per_process",
    "memory_mb",
    "wall_time_seconds",
    "launcher",
}
_CANONICAL_RESOURCE_FIELDS = _RESOURCE_FIELDS | {"mode", "per_gpu"}
_LAUNCHER_FIELDS = {"kind", "rdzv_backend"}
_BINDING_FIELDS = {"operation_role", "resource_class", "target_pool"}
_SECRET_KEY_PARTS = ("credential", "password", "passwd", "secret", "token", "private_key")
_COMMAND_KEYS = ("command", "cmd", "argv", "shell")


@dataclass(frozen=True, slots=True)
class SchedulerResourcePolicy:
    cpu: str
    memory: str
    expected_cpus_per_gpu: int | None = None
    expected_memory_mb_per_gpu: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    target_id: str
    adapter: str
    max_in_flight: int
    host_alias: str | None = None
    remote_root: str | None = None
    stage_mode: str | None = None
    working_root: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    resource_policy: SchedulerResourcePolicy | None = None


@dataclass(frozen=True, slots=True)
class TargetPool:
    pool_id: str
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LauncherSpec:
    kind: str
    rdzv_backend: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceClass:
    resource_class_id: str
    nodes: int
    processes_per_node: int
    gpus_per_node: int
    cpus_per_process: int
    memory_mb: int
    wall_time_seconds: int
    launcher: LauncherSpec
    mode: str
    per_gpu: int | None


@dataclass(frozen=True, slots=True)
class ExecutionBinding:
    operation_role: str
    resource_class: str
    target_pool: str


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    path: Path
    targets: Mapping[str, ExecutionTarget]
    target_pools: Mapping[str, TargetPool]
    resource_classes: Mapping[str, ResourceClass]
    bindings: tuple[ExecutionBinding, ...]
    digest: str
    historical: bool = False

    def binding_for(self, operation_role: str, resource_class: str) -> ExecutionBinding:
        matching = tuple(
            binding
            for binding in self.bindings
            if binding.operation_role == operation_role
            and binding.resource_class == resource_class
        )
        if len(matching) != 1:
            raise ValueError(
                "execution profile requires exactly one binding for "
                f"{operation_role!r}/{resource_class!r}"
            )
        return matching[0]


TRAINING_MPS_ROLES = frozenset({
    "teacher.deepmd.tf.train", "teacher.deepmd.pt.train",
    "student.deepmd.tf.train", "student.deepmd.pt.train",
})

INFERENCE_POOL_ROLES = frozenset({
    "teacher.mace.label", "teacher.mace.gmm_score",
    "teacher.deepmd.tf.label", "teacher.deepmd.pt.label",
    "teacher.deepmd.pt.gmm_score", "student.deepmd.tf.gmm_score",
    "student.deepmd.pt.gmm_screen", "student.nep.gmm_score",
})


def validate_operation_resource_shape(
    operation_role: str,
    *,
    gpus_per_node: int,
    mode: str,
    per_gpu: int | None,
) -> None:
    """Validate the canonical backend/resource capability matrix at profile load."""

    if gpus_per_node == 0:
        if mode != "pool" or per_gpu != 1:
            raise ValueError("CPU execution supports only pool mode with per_gpu 1")
        return
    pool_one = {
        "teacher.mace.train",
        "teacher.deepmd.tf.train",
        "teacher.deepmd.pt.train",
        "student.deepmd.tf.train",
        "student.deepmd.pt.train",
        "student.nep.train",
        "student.deepmd.tf.neb",
        "student.deepmd.pt.neb",
        "student.nep.neb",
        "student.deepmd.tf.relax",
        "student.deepmd.pt.relax",
        "student.nep.relax",
        "sampler.gpumd",
    }
    single_gpu_pool = {
        "teacher.mace.gmm_reference",
    }
    lammps = {"sampler.lammps.tf", "sampler.lammps.pt"}
    if mode == "distributed":
        if operation_role != "teacher.mace.train":
            raise ValueError(
                f"operation role {operation_role!r} does not support distributed mode"
            )
        return
    if operation_role in {"student.deepmd.tf.relax", "student.deepmd.pt.relax"} and per_gpu != 1:
        if gpus_per_node != 1:
            raise ValueError("relax MPS requires exactly one GPU")
        return
    if operation_role in TRAINING_MPS_ROLES:
        return
    if operation_role in INFERENCE_POOL_ROLES:
        if per_gpu != 1:
            raise ValueError("inference pool supports per_gpu 1 only")
        return
    if operation_role in pool_one:
        if per_gpu != 1:
            raise ValueError(
                f"operation role {operation_role!r} supports pool per_gpu 1 only"
            )
        return
    if operation_role in single_gpu_pool:
        if gpus_per_node != 1 or per_gpu != 1:
            raise ValueError(
                f"operation role {operation_role!r} supports exactly one GPU "
                "in pool mode"
            )
        return
    if operation_role in lammps:
        return
    raise ValueError(
        f"operation role {operation_role!r} has no supported GPU allocation capability"
    )


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read execution profile {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("execution profile must be a JSON object")
    _reject_authority_leaks(raw)
    _closed(raw, _TOP_LEVEL_FIELDS, "execution profile")
    if raw.get("schema") != "dice.execution-profile":
        raise ValueError("execution profile requires schema 'dice.execution-profile'")
    if "schema_revision" in raw and raw["schema_revision"] != 1:
        raise ValueError(
            "historical execution profile schema_revision must be exactly 1"
        )
    historical = "schema_revision" in raw

    targets_raw = _object(raw.get("targets"), "execution profile targets")
    pools_raw = _object(raw.get("target_pools"), "execution profile target_pools")
    resources_raw = _object(raw.get("resource_classes"), "execution profile resource_classes")
    bindings_raw = raw.get("bindings")
    if not isinstance(bindings_raw, list) or not bindings_raw:
        raise ValueError("execution profile bindings must be a non-empty array")

    targets: dict[str, ExecutionTarget] = {}
    for target_id, value in targets_raw.items():
        _safe_id(target_id, "execution target id")
        item = _object(value, f"execution target {target_id!r}")
        _closed(
            item,
            _TARGET_FIELDS if historical else _CANONICAL_TARGET_FIELDS,
            f"execution target {target_id!r}",
        )
        adapter = item.get("adapter")
        if adapter not in {"local", "ssh", "slurm"}:
            raise ValueError(f"execution target {target_id!r} adapter is unsupported: {adapter!r}")
        host_alias = item.get("host_alias")
        remote_root = item.get("remote_root")
        stage_mode = item.get("stage_mode")
        working_root = item.get("working_root")
        partition = item.get("partition")
        account = item.get("account")
        qos = item.get("qos")
        resource_policy = _scheduler_resource_policy(
            item.get("resource_policy"),
            target_id=target_id,
            adapter=adapter,
        )
        if adapter == "local":
            if any(value is not None for value in (host_alias, remote_root, stage_mode, working_root, partition, account, qos, resource_policy)):
                raise ValueError(f"local execution target {target_id!r} forbids SSH fields")
        elif adapter == "ssh":
            _safe_id(host_alias, f"execution target {target_id}.host_alias")
            remote_root = str(
                validate_safe_remote_root(
                    remote_root, field=f"execution target {target_id}.remote_root"
                )
            )
            if stage_mode != "copy":
                raise ValueError(f"execution target {target_id}.stage_mode must be 'copy'")
            if any(value is not None for value in (working_root, partition, account, qos, resource_policy)):
                raise ValueError(f"SSH execution target {target_id!r} forbids scheduler fields")
        else:
            if any(value is not None for value in (host_alias, remote_root)):
                raise ValueError(f"Slurm execution target {target_id!r} forbids SSH fields")
            working_root = _safe_working_root(
                working_root, field=f"execution target {target_id}.working_root"
            )
            if stage_mode != "shared":
                raise ValueError(f"execution target {target_id}.stage_mode must be 'shared'")
            if partition is not None:
                _safe_id(partition, f"execution target {target_id}.partition")
            if account is not None:
                _safe_id(account, f"execution target {target_id}.account")
            if qos is not None:
                _safe_id(qos, f"execution target {target_id}.qos")
        targets[target_id] = ExecutionTarget(
            target_id=target_id,
            adapter=adapter,
            max_in_flight=_positive_int(item.get("max_in_flight"), f"target {target_id}.max_in_flight"),
            host_alias=None if host_alias is None else str(host_alias),
            remote_root=None if remote_root is None else str(remote_root),
            stage_mode=None if stage_mode is None else str(stage_mode),
            working_root=None if working_root is None else str(working_root),
            partition=None if partition is None else str(partition),
            account=None if account is None else str(account),
            qos=None if qos is None else str(qos),
            resource_policy=resource_policy,
        )

    pools: dict[str, TargetPool] = {}
    for pool_id, value in pools_raw.items():
        _safe_id(pool_id, "target pool id")
        item = _object(value, f"target pool {pool_id!r}")
        _closed(item, _POOL_FIELDS, f"target pool {pool_id!r}")
        listed = item.get("targets")
        if not isinstance(listed, list) or not listed or not all(isinstance(value, str) for value in listed):
            raise ValueError(f"target pool {pool_id!r}.targets must be a non-empty string array")
        target_ids = tuple(listed)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"target pool {pool_id!r} contains duplicate targets")
        missing = [target for target in target_ids if target not in targets]
        if missing:
            raise ValueError(f"target pool {pool_id!r} references unknown target {missing[0]!r}")
        pools[pool_id] = TargetPool(pool_id=pool_id, targets=target_ids)

    resources: dict[str, ResourceClass] = {}
    for resource_id, value in resources_raw.items():
        _safe_id(resource_id, "resource class id")
        item = _object(value, f"resource class {resource_id!r}")
        _closed(
            item,
            _RESOURCE_FIELDS if historical else _CANONICAL_RESOURCE_FIELDS,
            f"resource class {resource_id!r}",
        )
        launcher_raw = _object(item.get("launcher"), f"resource class {resource_id!r}.launcher")
        _closed(launcher_raw, _LAUNCHER_FIELDS, f"resource class {resource_id!r}.launcher")
        launcher = launcher_raw.get("kind")
        if launcher not in {"direct", "torchrun"}:
            raise ValueError(f"resource class {resource_id!r} launcher is unsupported: {launcher!r}")
        rendezvous_backend = launcher_raw.get("rdzv_backend")
        if launcher == "direct":
            if rendezvous_backend is not None:
                raise ValueError("direct launcher forbids rendezvous fields")
        elif historical and rendezvous_backend != "c10d":
            raise ValueError("historical torchrun launcher requires rdzv_backend 'c10d'")
        elif not historical and rendezvous_backend not in {None, "c10d"}:
            raise ValueError("torchrun launcher rdzv_backend must be 'c10d' when supplied")
        nodes = _positive_int(item.get("nodes"), f"resource class {resource_id}.nodes")
        processes = _positive_int(
            item.get("processes_per_node"),
            f"resource class {resource_id}.processes_per_node",
        )
        gpus = _nonnegative_int(item.get("gpus_per_node"), f"resource class {resource_id}.gpus_per_node")
        if historical:
            mode = "pool" if launcher == "direct" else "distributed"
            per_gpu = 1 if mode == "pool" else None
            if launcher == "direct" and (nodes != 1 or processes != 1 or gpus > 1):
                raise ValueError("direct launcher supports exactly one node/process and at most one GPU")
            if launcher == "torchrun" and (nodes != 2 or processes != 1 or gpus != 1):
                raise ValueError(
                    "torchrun launcher supports exactly two nodes with one process/GPU per node"
                )
        else:
            mode = item.get("mode", "pool")
            if mode not in {"pool", "distributed"}:
                raise ValueError(
                    f"resource class {resource_id!r} mode is unsupported: {mode!r}"
                )
            if mode == "pool":
                per_gpu = _positive_int(
                    item.get("per_gpu", 1),
                    f"resource class {resource_id}.per_gpu",
                )
                if nodes != 1 or processes != 1 or launcher != "direct":
                    raise ValueError(
                        "pool mode requires one node/process and launcher kind 'direct'"
                    )
                if gpus == 0 and per_gpu != 1:
                    raise ValueError("CPU pool resources require per_gpu 1")
            else:
                if "per_gpu" in item:
                    raise ValueError("distributed mode forbids per_gpu")
                per_gpu = None
                if (
                    nodes != 1
                    or processes != 1
                    or gpus < 2
                    or launcher != "torchrun"
                ):
                    raise ValueError(
                        "distributed mode requires one node/process, at least two GPUs, and torchrun"
                    )
        resources[resource_id] = ResourceClass(
            resource_class_id=resource_id,
            nodes=nodes,
            processes_per_node=processes,
            gpus_per_node=gpus,
            cpus_per_process=_positive_int(
                item.get("cpus_per_process"),
                f"resource class {resource_id}.cpus_per_process",
            ),
            memory_mb=_positive_int(item.get("memory_mb"), f"resource class {resource_id}.memory_mb"),
            wall_time_seconds=_positive_int(
                item.get("wall_time_seconds"),
                f"resource class {resource_id}.wall_time_seconds",
            ),
            launcher=LauncherSpec(kind=launcher, rdzv_backend=rendezvous_backend),
            mode=str(mode),
            per_gpu=per_gpu,
        )

    bindings: list[ExecutionBinding] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(bindings_raw):
        item = _object(value, f"execution binding {index}")
        _closed(item, _BINDING_FIELDS, f"execution binding {index}")
        operation_role = item.get("operation_role")
        resource_class = item.get("resource_class")
        target_pool = item.get("target_pool")
        if not isinstance(operation_role, str) or not operation_role:
            raise ValueError(f"execution binding {index}.operation_role is invalid")
        if resource_class not in resources:
            raise ValueError(f"execution binding {index} references unknown resource class")
        if target_pool not in pools:
            raise ValueError(f"execution binding {index} references unknown target pool")
        key = (operation_role, resource_class)
        if key in seen:
            raise ValueError(f"duplicate execution binding for {operation_role!r}/{resource_class!r}")
        seen.add(key)
        if resources[str(resource_class)].launcher.kind == "torchrun":
            supported_role = (
                "student.deepmd.pt.train"
                if historical
                else "teacher.mace.train"
            )
            if operation_role != supported_role:
                raise ValueError(
                    f"torchrun launcher is supported only for {supported_role}"
                )
        if not historical:
            resource = resources[str(resource_class)]
            validate_operation_resource_shape(
                operation_role,
                gpus_per_node=resource.gpus_per_node,
                mode=resource.mode,
                per_gpu=resource.per_gpu,
            )
        resource = resources[str(resource_class)]
        if operation_role in TRAINING_MPS_ROLES and resource.mode == "pool" and resource.per_gpu != 1:
            selected_targets = pools[str(target_pool)].targets
            if historical or len(selected_targets) != 1 or targets[selected_targets[0]].adapter != "local":
                raise ValueError("training MPS requires exactly one explicit local target")
        if operation_role in {"student.deepmd.tf.relax", "student.deepmd.pt.relax"} and resource.per_gpu != 1:
            if historical or any(targets[target].adapter != "local"
                                 for target in pools[str(target_pool)].targets):
                raise ValueError("relax MPS requires an explicit local execution binding")
            if resource.cpus_per_process < resource.per_gpu:
                raise ValueError("relax MPS requires at least one CPU per worker")
        bindings.append(
            ExecutionBinding(
                operation_role=operation_role,
                resource_class=resource_class,
                target_pool=target_pool,
            )
        )

    _validate_scheduler_resource_expectations(
        targets=targets,
        pools=pools,
        resources=resources,
        bindings=bindings,
    )

    normalized = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ExecutionProfile(
        path=source,
        targets=targets,
        target_pools=pools,
        resource_classes=resources,
        bindings=tuple(bindings),
        digest=hashlib.sha256(normalized).hexdigest(),
        historical=historical,
    )


def _scheduler_resource_policy(
    value: object,
    *,
    target_id: str,
    adapter: object,
) -> SchedulerResourcePolicy | None:
    if value is None:
        return None
    if adapter != "slurm":
        raise ValueError(
            f"execution target {target_id!r} resource_policy requires Slurm"
        )
    item = _object(value, f"execution target {target_id!r}.resource_policy")
    _closed(
        item,
        _RESOURCE_POLICY_FIELDS,
        f"execution target {target_id!r}.resource_policy",
    )
    allowed = {"explicit", "scheduler-gpu-default"}
    cpu = item.get("cpu", "explicit")
    memory = item.get("memory", "explicit")
    if cpu not in allowed:
        raise ValueError(
            f"execution target {target_id!r}.resource_policy.cpu is unsupported: {cpu!r}"
        )
    if memory not in allowed:
        raise ValueError(
            f"execution target {target_id!r}.resource_policy.memory is unsupported: {memory!r}"
        )
    expected_cpus = item.get("expected_cpus_per_gpu")
    expected_memory = item.get("expected_memory_mb_per_gpu")
    if cpu == "scheduler-gpu-default":
        expected_cpus = _positive_int(
            expected_cpus,
            f"execution target {target_id}.resource_policy.expected_cpus_per_gpu",
        )
    elif expected_cpus is not None:
        raise ValueError(
            "expected_cpus_per_gpu requires cpu 'scheduler-gpu-default'"
        )
    if memory == "scheduler-gpu-default":
        expected_memory = _positive_int(
            expected_memory,
            f"execution target {target_id}.resource_policy.expected_memory_mb_per_gpu",
        )
    elif expected_memory is not None:
        raise ValueError(
            "expected_memory_mb_per_gpu requires memory 'scheduler-gpu-default'"
        )
    return SchedulerResourcePolicy(
        cpu=str(cpu),
        memory=str(memory),
        expected_cpus_per_gpu=expected_cpus,
        expected_memory_mb_per_gpu=expected_memory,
    )


def _validate_scheduler_resource_expectations(
    *,
    targets: Mapping[str, ExecutionTarget],
    pools: Mapping[str, TargetPool],
    resources: Mapping[str, ResourceClass],
    bindings: list[ExecutionBinding],
) -> None:
    for binding in bindings:
        resource = resources[binding.resource_class]
        for target_id in pools[binding.target_pool].targets:
            policy = targets[target_id].resource_policy
            if policy is None:
                continue
            uses_gpu_default = (
                policy.cpu == "scheduler-gpu-default"
                or policy.memory == "scheduler-gpu-default"
            )
            if uses_gpu_default and resource.gpus_per_node <= 0:
                raise ValueError(
                    f"execution target {target_id!r} scheduler GPU defaults require "
                    "a positive GPU resource class"
                )
            if policy.cpu == "scheduler-gpu-default":
                expected = int(policy.expected_cpus_per_gpu) * resource.gpus_per_node
                declared = resource.cpus_per_process * resource.processes_per_node
                if declared != expected:
                    raise ValueError(
                        f"resource class {resource.resource_class_id!r} declares "
                        f"{declared} CPUs per node but target {target_id!r} expects "
                        f"{expected} from scheduler GPU defaults"
                    )
            if policy.memory == "scheduler-gpu-default":
                expected = int(policy.expected_memory_mb_per_gpu) * resource.gpus_per_node
                if resource.memory_mb != expected:
                    raise ValueError(
                        f"resource class {resource.resource_class_id!r} declares "
                        f"{resource.memory_mb} MiB per node but target {target_id!r} "
                        f"expects {expected} from scheduler GPU defaults"
                    )


def _reject_authority_leaks(value: object, *, path: str = "execution profile") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SECRET_KEY_PARTS):
                raise ValueError(f"credential or secret field is forbidden at {path}.{key}")
            if lowered in _COMMAND_KEYS or lowered.endswith("_command"):
                raise ValueError(f"command field is forbidden at {path}.{key}")
            _reject_authority_leaks(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_authority_leaks(child, path=f"{path}[{index}]")


def validate_safe_remote_root(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("~") or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path


def _safe_working_root(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or value.startswith("~"):
        raise ValueError(f"{field} is unsafe")
    path = Path(value)
    if not path.is_absolute() or any(part == ".." for part in path.parts) or path == Path("/"):
        raise ValueError(f"{field} must be one safe non-root absolute path")
    return str(path)


def _closed(value: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field {unknown[0]!r}")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{label} is invalid: {value!r}")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value
