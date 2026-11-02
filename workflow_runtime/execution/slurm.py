from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Mapping, Protocol

from workflow_runtime.execution.planning import PlannedExecution
from workflow_runtime.execution.profile import ExecutionTarget
from workflow_runtime.execution.protocol import ExternalJobReceipt
from workflow_runtime.execution.store import _atomic_write_json


_SLURM_STATES = {
    "PENDING": "queued",
    "CONFIGURING": "queued",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "succeeded",
    "FAILED": "failed",
    "TIMEOUT": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "CANCELLED": "cancelled",
}


class SlurmCommands(Protocol):
    def submit(self, *, script: Path, cwd: Path, job_name: str) -> str: ...
    def observe(self, job_id: str) -> tuple[str, Mapping[str, object]]: ...
    def state(self, job_id: str) -> str: ...
    def allocation(self, job_id: str) -> Mapping[str, object]: ...
    def cancel(self, job_id: str) -> None: ...


_SLURM_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_SLURM_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class NativeSlurmCommands:
    """Narrow detached Slurm command surface; DICE retains state ownership."""

    def __init__(self, *, runner=subprocess.run, command_timeout_seconds: int = 30):
        if command_timeout_seconds <= 0:
            raise ValueError("Slurm command timeout must be positive")
        self.runner = runner
        self.command_timeout_seconds = command_timeout_seconds

    def submit(self, *, script: Path, cwd: Path, job_name: str) -> str:
        if not _SLURM_JOB_NAME.fullmatch(job_name):
            raise ValueError("Slurm job name is unsafe")
        result = self._run(
            [
                "sbatch",
                "--parsable",
                f"--job-name={job_name}",
                str(Path(script).resolve()),
            ],
            cwd=Path(cwd).resolve(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not _SLURM_JOB_ID.fullmatch(job_id):
            raise RuntimeError("sbatch did not return one exact Slurm job id")
        return job_id

    def observe(self, job_id: str) -> tuple[str, Mapping[str, object]]:
        """Observe state and allocation with one scheduler command."""

        self._validate_job_id(job_id)
        live = self._run(["scontrol", "show", "job", "-o", job_id])
        parsed = (
            _parse_scontrol_observation(live.stdout)
            if live.returncode == 0
            else None
        )
        if parsed is not None:
            return parsed
        return self._accounting_observation(job_id)

    def state(self, job_id: str) -> str:
        return self.observe(job_id)[0]

    def allocation(self, job_id: str) -> Mapping[str, object]:
        return self.observe(job_id)[1]

    def _accounting_observation(
        self,
        job_id: str,
    ) -> tuple[str, Mapping[str, object]]:
        result = self._run(
            [
                "sacct",
                f"--jobs={job_id}",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--format=State,AllocTRES,NodeList,ExitCode",
            ]
        )
        if result.returncode != 0:
            return "INDETERMINATE", {}
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not rows:
            return "MISSING", {}
        fields = rows[0].split("|")
        if len(fields) < 4 or not fields[0].strip():
            return "INDETERMINATE", {}
        raw = fields[0].strip()
        return raw, {
            "scheduler_state_raw": raw,
            "observed_alloc_tres": fields[1].strip(),
            "observed_node_list": fields[2].strip(),
            "observed_exit_code": fields[3].strip(),
        }

    def cancel(self, job_id: str) -> None:
        self._validate_job_id(job_id)
        result = self._run(["scancel", f"--jobid={job_id}"])
        if result.returncode != 0:
            raise RuntimeError(f"scancel failed: {result.stderr.strip()}")

    def _run(self, argv: list[str], *, cwd: Path | None = None):
        kwargs: dict[str, object] = {
            "text": True,
            "capture_output": True,
            "check": False,
            "timeout": self.command_timeout_seconds,
        }
        if cwd is not None:
            kwargs["cwd"] = cwd
        return self.runner(argv, **kwargs)

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if not isinstance(job_id, str) or not _SLURM_JOB_ID.fullmatch(job_id):
            raise ValueError("Slurm job id is invalid")


def _parse_scontrol_observation(
    stdout: str,
) -> tuple[str, Mapping[str, object]] | None:
    fields: dict[str, str] = {}
    for token in shlex.split(stdout):
        name, separator, value = token.partition("=")
        if separator and name:
            fields[name] = value
    raw = fields.get("JobState", "").strip()
    if not raw:
        return None
    return raw, {
        "scheduler_state_raw": raw,
        "observed_alloc_tres": fields.get(
            "AllocTRES", fields.get("TRES", fields.get("TresPerNode", ""))
        ),
        "observed_node_list": fields.get("NodeList", ""),
        "observed_exit_code": fields.get("ExitCode", ""),
    }


class FakeSlurmCommands:
    def __init__(self):
        self.submit_count = 0
        self._states: dict[str, str] = {}
        self._allocations: dict[str, dict[str, object]] = {}

    def submit(self, *, script: Path, cwd: Path, job_name: str) -> str:
        del script, cwd, job_name
        self.submit_count += 1
        job_id = str(1000 + self.submit_count)
        self._states[job_id] = "PENDING"
        self._allocations[job_id] = {}
        return job_id

    def observe(self, job_id: str) -> tuple[str, Mapping[str, object]]:
        return (
            self._states.get(job_id, "MISSING"),
            dict(self._allocations.get(job_id, {})),
        )

    def state(self, job_id: str) -> str:
        return self.observe(job_id)[0]

    def allocation(self, job_id: str) -> Mapping[str, object]:
        return self.observe(job_id)[1]

    def cancel(self, job_id: str) -> None:
        if job_id not in self._states:
            raise RuntimeError("Slurm job identity is missing")
        self._states[job_id] = "CANCELLED"

    def set_state(self, job_id: str, state: str) -> None:
        self._states[job_id] = state

    def set_allocation(self, job_id: str, **evidence: object) -> None:
        self._allocations[job_id] = dict(evidence)


class SlurmExecutionAdapter:
    adapter_kind = "slurm"

    def __init__(self, *, target: ExecutionTarget, commands: SlurmCommands):
        if target.adapter != "slurm" or target.working_root is None:
            raise ValueError("Slurm execution adapter requires one Slurm target")
        self.target = target
        self.target_id = target.target_id
        self.commands = commands

    def submit(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt:
        job_root = self._job_root(planned.unit.unit_id, planned.attempt_id)
        receipt_path = job_root / "adapter_receipt.json"
        if receipt_path.exists():
            raise RuntimeError("Slurm adapter refuses duplicate submit")
        self._stage(job_root, launch_intent=launch_intent)
        script = self._write_script(job_root, planned, launch_intent=launch_intent)
        job_name = f"dice-{hashlib.sha256(str(launch_intent['launch_key']).encode()).hexdigest()[:16]}"
        job_id = self.commands.submit(script=script, cwd=job_root / "workspace", job_name=job_name)
        receipt = self._receipt(planned, job_id=job_id, state="queued", intent=launch_intent)
        _atomic_write_json(receipt_path, receipt.as_dict())
        return receipt

    def recover_submission(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt | None:
        value = _read_json(
            self._job_root(planned.unit.unit_id, planned.attempt_id)
            / "adapter_receipt.json"
        )
        if value is None:
            return None
        receipt = ExternalJobReceipt(**value)
        if receipt.immutable_spec_digest != launch_intent["immutable_spec_digest"]:
            raise ValueError("Slurm immutable submission evidence changed")
        return self.poll(receipt)

    def poll(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        observer = getattr(self.commands, "observe", None)
        if callable(observer):
            raw_state, observed_allocation = observer(receipt.external_id)
        else:
            raw_state = self.commands.state(receipt.external_id)
            observed_allocation = self.commands.allocation(receipt.external_id)
        raw = _normalize_slurm_state(raw_state)
        state = _SLURM_STATES.get(
            raw,
            "missing" if raw == "MISSING" else "indeterminate",
        )
        evidence = dict(receipt.allocation_evidence or {})
        evidence.update(observed_allocation)
        evidence.update(
            _runtime_allocation_evidence(
                self._job_root(receipt.unit_id, receipt.attempt_id)
            )
        )
        _enrich_allocation_evidence(evidence)
        _enrich_topology_evidence(evidence)
        return ExternalJobReceipt(
            **{
                **receipt.as_dict(),
                "state": state,
                "allocation_evidence": evidence,
            }
        )

    def cancel(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        self.commands.cancel(receipt.external_id)
        observed = self.poll(receipt)
        if observed.state != "cancelled":
            raise RuntimeError("Slurm cancellation did not reach exact cancelled state")
        return observed

    def collect(
        self,
        receipt: ExternalJobReceipt,
        *,
        destination: Path,
    ) -> Mapping[str, str]:
        if receipt.state != "succeeded":
            raise RuntimeError("Slurm collection requires terminal success")
        if destination.exists():
            raise ValueError("Slurm collection destination must be fresh")
        job_root = self._job_root(receipt.unit_id, receipt.attempt_id)
        spec = _required_json(job_root / "immutable_spec.json")
        prepared = _prepared(spec)
        destination.mkdir(parents=True)
        inventory: dict[str, str] = {}
        for relative in prepared["output_paths"]:
            source = job_root / "workspace" / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Slurm output is missing or unsafe: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inventory[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        return inventory

    def workspace_for(self, planned: PlannedExecution) -> Path:
        return self._job_root(planned.unit.unit_id, planned.attempt_id) / "workspace"

    def _stage(self, job_root: Path, *, launch_intent: Mapping[str, object]) -> None:
        spec_path = job_root / "immutable_spec.json"
        if spec_path.exists():
            if _required_json(spec_path) != dict(launch_intent):
                raise ValueError("Slurm immutable spec changed")
            return
        prepared = _prepared(launch_intent)
        source_root = Path(str(prepared["source_root"])).resolve()
        inputs = launch_intent.get("input_inventory")
        if not isinstance(inputs, dict):
            raise ValueError("Slurm input inventory is invalid")
        workspace = job_root / "workspace"
        workspace.mkdir(parents=True)
        for relative, expected in sorted(inputs.items()):
            source = source_root / str(relative)
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Slurm input is missing or unsafe: {relative}")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Slurm input digest changed: {relative}")
            target = workspace / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        _atomic_write_json(spec_path, dict(launch_intent))

    def _write_script(
        self,
        job_root: Path,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> Path:
        prepared = _prepared(launch_intent)
        shape = planned.unit.resource_shape
        seconds = int(shape["wall_time_seconds"])
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        lines = [
            "#!/bin/sh",
            "#SBATCH --nodes=" + str(shape["nodes"]),
            "#SBATCH --ntasks-per-node=" + str(shape["processes_per_node"]),
        ]
        policy = self.target.resource_policy
        if policy is None or policy.cpu == "explicit":
            lines.append("#SBATCH --cpus-per-task=" + str(shape["cpus_per_process"]))
        if policy is None or policy.memory == "explicit":
            lines.append("#SBATCH --mem=" + str(shape["memory_mb"]) + "M")
        lines.append(f"#SBATCH --time={hours:02d}:{minutes:02d}:{secs:02d}")
        if int(shape["gpus_per_node"]):
            lines.append("#SBATCH --gpus-per-node=" + str(shape["gpus_per_node"]))
        if self.target.partition is not None:
            lines.append("#SBATCH --partition=" + self.target.partition)
        if self.target.account is not None:
            lines.append("#SBATCH --account=" + self.target.account)
        if self.target.qos is not None:
            lines.append("#SBATCH --qos=" + self.target.qos)
        lines.append("set -eu")
        lines.extend(_topology_probe_lines(job_root))
        lines.append(shlex.join(tuple(str(value) for value in prepared["argv"])))
        path = job_root / "job.sh"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _job_root(self, unit_id: str, attempt_id: str) -> Path:
        return Path(str(self.target.working_root)).resolve() / unit_id / attempt_id

    def _initial_allocation_evidence(
        self,
        planned: PlannedExecution,
    ) -> dict[str, object]:
        shape = planned.unit.resource_shape
        policy = self.target.resource_policy
        evidence: dict[str, object] = {
            "scheduler": "slurm",
            "requested_gpus_per_node": shape["gpus_per_node"],
            "declared_cpus_per_process": shape["cpus_per_process"],
            "declared_memory_mb": shape["memory_mb"],
            "device_visibility_source": "scheduler",
            "cpu_request_policy": "explicit" if policy is None else policy.cpu,
            "memory_request_policy": "explicit" if policy is None else policy.memory,
        }
        if policy is not None:
            if policy.expected_cpus_per_gpu is not None:
                evidence["expected_cpus_per_gpu"] = policy.expected_cpus_per_gpu
            if policy.expected_memory_mb_per_gpu is not None:
                evidence["expected_memory_mb_per_gpu"] = (
                    policy.expected_memory_mb_per_gpu
                )
        return evidence

    def _receipt(
        self,
        planned: PlannedExecution,
        *,
        job_id: str,
        state: str,
        intent: Mapping[str, object],
    ) -> ExternalJobReceipt:
        return ExternalJobReceipt(
            schema="dice.execution-receipt", schema_revision=1,
            unit_id=planned.unit.unit_id, attempt_id=planned.attempt_id,
            target_id=planned.target_id, adapter=self.adapter_kind,
            external_id=job_id, state=state,
            immutable_spec_digest=str(intent["immutable_spec_digest"]),
            allocation_evidence=self._initial_allocation_evidence(planned),
        )


def _normalize_slurm_state(value: object) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return "INDETERMINATE"
    return raw.split(None, 1)[0].split("+", 1)[0]


def _topology_probe_lines(job_root: Path) -> tuple[str, ...]:
    observation = shlex.quote(str(job_root / "slurm_topology.env"))
    gpu_inventory = shlex.quote(str(job_root / "slurm_gpu_inventory.csv"))
    gpu_topology = shlex.quote(str(job_root / "slurm_gpu_topology.txt"))
    numa_policy = shlex.quote(str(job_root / "slurm_numa_policy.txt"))
    return (
        f"DICE_TOPOLOGY_TMP={observation}.$$",
        "{",
        "  printf 'slurm_job_id=%s\\n' \"${SLURM_JOB_ID-}\"",
        "  printf 'hostname=%s\\n' \"$(hostname 2>/dev/null || :)\"",
        "  printf 'slurm_cpus_on_node=%s\\n' \"${SLURM_CPUS_ON_NODE-}\"",
        "  printf 'slurm_job_cpus_per_node=%s\\n' \"${SLURM_JOB_CPUS_PER_NODE-}\"",
        "  printf 'slurm_mem_per_gpu=%s\\n' \"${SLURM_MEM_PER_GPU-}\"",
        "  printf 'cuda_visible_devices=%s\\n' \"${CUDA_VISIBLE_DEVICES-}\"",
        "  sed -n 's/^Cpus_allowed_list:[[:space:]]*/cpus_allowed_list=/p; "
        "s/^Mems_allowed_list:[[:space:]]*/mems_allowed_list=/p' /proc/self/status",
        "} > \"$DICE_TOPOLOGY_TMP\"",
        f"mv \"$DICE_TOPOLOGY_TMP\" {observation}",
        "if command -v nvidia-smi >/dev/null 2>&1; then",
        f"  nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits > {gpu_inventory}.tmp 2>/dev/null || :",
        f"  mv {gpu_inventory}.tmp {gpu_inventory}",
        f"  nvidia-smi topo -m > {gpu_topology}.tmp 2>/dev/null || :",
        f"  mv {gpu_topology}.tmp {gpu_topology}",
        "fi",
        "if command -v numactl >/dev/null 2>&1; then",
        f"  numactl --show > {numa_policy}.tmp 2>&1 || :",
        f"  mv {numa_policy}.tmp {numa_policy}",
        "fi",
    )


def _runtime_allocation_evidence(job_root: Path) -> dict[str, object]:
    evidence: dict[str, object] = {}
    observation = job_root / "slurm_topology.env"
    if observation.is_file() and observation.stat().st_size <= 16_384:
        allowed = {
            "slurm_job_id",
            "hostname",
            "slurm_cpus_on_node",
            "slurm_job_cpus_per_node",
            "slurm_mem_per_gpu",
            "cuda_visible_devices",
            "cpus_allowed_list",
            "mems_allowed_list",
        }
        for line in observation.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator and key in allowed:
                evidence[key] = value.strip()
    gpu_inventory = job_root / "slurm_gpu_inventory.csv"
    if gpu_inventory.is_file() and gpu_inventory.stat().st_size <= 16_384:
        devices: list[dict[str, str]] = []
        for line in gpu_inventory.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                devices.append(
                    {"index": fields[0], "uuid": fields[1], "pci_bus_id": fields[2]}
                )
        evidence["visible_gpu_inventory"] = devices
    for name, key, limit in (
        ("slurm_gpu_topology.txt", "gpu_topology_matrix", 32_768),
        ("slurm_numa_policy.txt", "numa_policy", 8_192),
    ):
        source = job_root / name
        if source.is_file() and source.stat().st_size <= limit:
            evidence[key] = source.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
    return evidence


def _enrich_allocation_evidence(evidence: dict[str, object]) -> None:
    raw = evidence.get("observed_alloc_tres")
    if not isinstance(raw, str) or not raw:
        return
    tres: dict[str, str] = {}
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if separator:
            tres[key.strip()] = value.strip()
    try:
        observed_cpus = int(tres["cpu"])
    except (KeyError, ValueError):
        observed_cpus = None
    try:
        observed_gpus = int(tres["gres/gpu"])
    except (KeyError, ValueError):
        observed_gpus = None
    observed_memory = _slurm_memory_mb(tres.get("mem"))
    if observed_cpus is not None:
        evidence["observed_cpus"] = observed_cpus
    if observed_gpus is not None:
        evidence["observed_gpus"] = observed_gpus
    if observed_memory is not None:
        evidence["observed_memory_mb"] = observed_memory
    requested_gpus = evidence.get("requested_gpus_per_node")
    cpu_per_gpu = evidence.get("expected_cpus_per_gpu")
    memory_per_gpu = evidence.get("expected_memory_mb_per_gpu")
    checks: list[bool] = []
    if isinstance(requested_gpus, int) and isinstance(cpu_per_gpu, int):
        checks.append(observed_cpus == requested_gpus * cpu_per_gpu)
    if isinstance(requested_gpus, int) and isinstance(memory_per_gpu, int):
        checks.append(observed_memory == requested_gpus * memory_per_gpu)
    if isinstance(requested_gpus, int) and observed_gpus is not None:
        checks.append(observed_gpus == requested_gpus)
    if checks:
        evidence["matches_resource_expectations"] = all(checks)


def _enrich_topology_evidence(evidence: dict[str, object]) -> None:
    matrix = evidence.get("gpu_topology_matrix")
    if not isinstance(matrix, str) or not matrix:
        return
    clean = re.sub(r"\x1b\[[0-9;]*m", "", matrix)
    lines = [line for line in clean.splitlines() if line.strip()]
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "CPU Affinity" in line and "NUMA Affinity" in line
        ),
        None,
    )
    if header_index is None:
        return
    header = [field.strip() for field in lines[header_index].split("\t")]
    try:
        cpu_column = header.index("CPU Affinity") + 1
        numa_column = header.index("NUMA Affinity") + 1
    except ValueError:
        return
    rows: list[dict[str, object]] = []
    for line in lines[header_index + 1 :]:
        fields = [field.strip() for field in line.split("\t")]
        if not fields or re.fullmatch(r"GPU[0-9]+", fields[0]) is None:
            continue
        if len(fields) <= max(cpu_column, numa_column):
            continue
        numa_values = sorted(_integer_set(fields[numa_column]))
        rows.append(
            {
                "gpu": fields[0],
                "cpu_affinity": fields[cpu_column],
                "numa_affinity": numa_values,
            }
        )
    if not rows:
        return
    evidence["visible_gpu_affinity"] = rows
    allowed = evidence.get("cpus_allowed_list")
    if isinstance(allowed, str) and allowed:
        allowed_cpus = _integer_set(allowed)
        evidence["matches_gpu_cpu_affinity"] = all(
            _integer_set(str(row["cpu_affinity"])) == allowed_cpus for row in rows
        )
    policy = evidence.get("numa_policy")
    if isinstance(policy, str):
        policy_values: dict[str, str] = {}
        for line in policy.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                policy_values[key.strip()] = value.strip()
        cpu_nodes = sorted(_integer_set(policy_values.get("cpubind", "")))
        node_nodes = sorted(_integer_set(policy_values.get("nodebind", "")))
        memory_nodes = sorted(_integer_set(policy_values.get("membind", "")))
        if cpu_nodes:
            evidence["cpu_numa_nodes"] = cpu_nodes
        if node_nodes:
            evidence["nodebind_numa_nodes"] = node_nodes
        if memory_nodes:
            evidence["memory_allowed_numa_nodes"] = memory_nodes
        gpu_nodes = {
            node
            for row in rows
            for node in row["numa_affinity"]
            if isinstance(node, int)
        }
        if gpu_nodes and cpu_nodes:
            evidence["matches_gpu_numa_affinity"] = gpu_nodes.issubset(cpu_nodes)


def _integer_set(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.replace(" ", ",").split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start, separator, stop = token.partition("-")
            if separator and start.isdigit() and stop.isdigit():
                result.update(range(int(start), int(stop) + 1))
        elif token.isdigit():
            result.add(int(token))
    return result


def _slurm_memory_mb(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", value.upper())
    if match is None:
        return None
    magnitude = float(match.group(1))
    multiplier = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[
        match.group(2)
    ]
    return int(round(magnitude * multiplier))


def _prepared(intent: Mapping[str, object]) -> dict[str, object]:
    value = intent.get("prepared")
    if not isinstance(value, dict):
        raise ValueError("Slurm execution requires a prepared command")
    source_root, argv, outputs = value.get("source_root"), value.get("argv"), value.get("output_paths")
    if not isinstance(source_root, str) or not isinstance(argv, list) or not isinstance(outputs, list):
        raise ValueError("Slurm prepared execution shape is invalid")
    return {"source_root": source_root, "argv": tuple(argv), "output_paths": tuple(outputs)}


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Slurm evidence is malformed: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Slurm evidence must be an object: {path.name}")
    return value


def _required_json(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if value is None:
        raise ValueError(f"Slurm evidence is missing: {path.name}")
    return value
