from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Mapping, Protocol

from workflow_runtime.execution.planning import PlannedExecution
from workflow_runtime.execution.profile import ExecutionTarget
from workflow_runtime.execution.protocol import ExternalJobReceipt
from workflow_runtime.execution.store import _atomic_write_json


@dataclass(frozen=True, slots=True)
class TransportJob:
    external_id: str
    state: str
    immutable_spec_digest: str
    allocation_evidence: dict[str, object] | None = None


class SSHTransport(Protocol):
    def stage(
        self,
        remote_dir: PurePosixPath,
        *,
        launch_intent: Mapping[str, object],
    ) -> None: ...

    def submit_once(
        self,
        remote_dir: PurePosixPath,
        *,
        wall_time_seconds: int,
    ) -> TransportJob: ...

    def inspect(
        self,
        remote_dir: PurePosixPath,
        *,
        immutable_spec_digest: str,
    ) -> TransportJob | None: ...

    def cancel(
        self,
        remote_dir: PurePosixPath,
        *,
        immutable_spec_digest: str,
    ) -> TransportJob: ...

    def fetch(
        self,
        remote_dir: PurePosixPath,
        *,
        immutable_spec_digest: str,
        destination: Path,
    ) -> Mapping[str, str]: ...


class SSHExecutionAdapter:
    adapter_kind = "ssh"

    def __init__(self, *, target: ExecutionTarget, transport: SSHTransport):
        if target.adapter != "ssh" or target.remote_root is None:
            raise ValueError("SSH execution adapter requires one SSH target")
        self.target = target
        self.target_id = target.target_id
        self.transport = transport

    def submit(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt:
        remote_dir = self._remote_dir(planned.unit.unit_id, planned.attempt_id)
        self.transport.stage(remote_dir, launch_intent=launch_intent)
        job = self.transport.submit_once(
            remote_dir,
            wall_time_seconds=int(planned.unit.resource_shape["wall_time_seconds"]),
        )
        return self._receipt(planned, job)

    def recover_submission(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt | None:
        job = self.transport.inspect(
            self._remote_dir(planned.unit.unit_id, planned.attempt_id),
            immutable_spec_digest=str(launch_intent["immutable_spec_digest"]),
        )
        return None if job is None else self._receipt(planned, job)

    def poll(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        job = self.transport.inspect(
            self._remote_dir(receipt.unit_id, receipt.attempt_id),
            immutable_spec_digest=receipt.immutable_spec_digest,
        )
        if job is None:
            return ExternalJobReceipt(**{**receipt.as_dict(), "state": "indeterminate"})
        return ExternalJobReceipt(**{**receipt.as_dict(), "state": job.state})

    def cancel(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        job = self.transport.cancel(
            self._remote_dir(receipt.unit_id, receipt.attempt_id),
            immutable_spec_digest=receipt.immutable_spec_digest,
        )
        return ExternalJobReceipt(**{**receipt.as_dict(), "state": job.state})

    def collect(
        self,
        receipt: ExternalJobReceipt,
        *,
        destination: Path,
    ) -> Mapping[str, str]:
        if receipt.state != "succeeded":
            raise RuntimeError("SSH execution collection requires terminal success")
        return self.transport.fetch(
            self._remote_dir(receipt.unit_id, receipt.attempt_id),
            immutable_spec_digest=receipt.immutable_spec_digest,
            destination=destination,
        )

    def _remote_dir(self, unit_id: str, attempt_id: str) -> PurePosixPath:
        return PurePosixPath(str(self.target.remote_root)) / unit_id / attempt_id

    def _receipt(
        self,
        planned: PlannedExecution,
        job: TransportJob,
    ) -> ExternalJobReceipt:
        return ExternalJobReceipt(
            schema="dice.execution-receipt",
            schema_revision=1,
            unit_id=planned.unit.unit_id,
            attempt_id=planned.attempt_id,
            target_id=planned.target_id,
            adapter=self.adapter_kind,
            external_id=job.external_id,
            state=job.state,
            immutable_spec_digest=job.immutable_spec_digest,
            allocation_evidence=job.allocation_evidence,
        )


class LoopbackSSHTransport:
    """Filesystem/process transport used to exercise SSH semantics without a host."""

    def __init__(self, *, root: str | Path):
        self.root = Path(root).resolve()
        self.submit_count = 0
        self.staged_inventory: tuple[str, ...] = ()
        self.fetched_inventory: tuple[str, ...] = ()

    def stage(
        self,
        remote_dir: PurePosixPath,
        *,
        launch_intent: Mapping[str, object],
    ) -> None:
        root = self._resolve(remote_dir)
        spec_path = root / "immutable_spec.json"
        if spec_path.exists():
            existing = _read_json(spec_path)
            if existing != dict(launch_intent):
                raise ValueError("SSH remote immutable spec changed")
            return
        prepared = _prepared(launch_intent)
        source_root = Path(str(prepared["source_root"])).resolve()
        inputs = launch_intent.get("input_inventory")
        if not isinstance(inputs, dict):
            raise ValueError("SSH input inventory is invalid")
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        staged: list[str] = []
        for relative, expected in sorted(inputs.items()):
            source = source_root / str(relative)
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"SSH stage input is missing or unsafe: {relative}")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"SSH stage input digest changed: {relative}")
            target = workspace / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            staged.append(str(relative))
        self.staged_inventory = tuple(staged)
        _atomic_write_json(spec_path, dict(launch_intent))

    def submit_once(
        self,
        remote_dir: PurePosixPath,
        *,
        wall_time_seconds: int,
    ) -> TransportJob:
        root = self._resolve(remote_dir)
        if (root / "terminal.json").exists() or (root / "submitted.json").exists():
            raise RuntimeError("SSH transport refuses duplicate submit")
        spec = _required_json(root / "immutable_spec.json")
        prepared = _prepared(spec)
        self.submit_count += 1
        external_id = f"loopback-{self.submit_count:06d}"
        _atomic_write_json(root / "submitted.json", {"external_id": external_id})
        try:
            completed = subprocess.run(
                list(prepared["argv"]), cwd=root / "workspace", env=os.environ.copy(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=wall_time_seconds, check=False,
            )
            state = "succeeded" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            state = "failed"
        (root / "stdout.log").write_bytes(getattr(completed, "stdout", b"") if "completed" in locals() else b"")
        (root / "stderr.log").write_bytes(getattr(completed, "stderr", b"") if "completed" in locals() else b"")
        job = TransportJob(
            external_id=external_id,
            state=state,
            immutable_spec_digest=str(spec["immutable_spec_digest"]),
            allocation_evidence={
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_visibility_source": "transport-environment",
            },
        )
        _atomic_write_json(root / "terminal.json", {
            "external_id": job.external_id, "state": job.state,
            "immutable_spec_digest": job.immutable_spec_digest,
            "allocation_evidence": job.allocation_evidence,
        })
        return job

    def inspect(
        self,
        remote_dir: PurePosixPath,
        *,
        immutable_spec_digest: str,
    ) -> TransportJob | None:
        root = self._resolve(remote_dir)
        value = _read_json(root / "terminal.json")
        if value is None:
            submitted = _read_json(root / "submitted.json")
            if submitted is None:
                return None
            spec = _required_json(root / "immutable_spec.json")
            return TransportJob(
                external_id=str(submitted["external_id"]), state="indeterminate",
                immutable_spec_digest=str(spec["immutable_spec_digest"]),
            )
        job = TransportJob(**value)
        if job.immutable_spec_digest != immutable_spec_digest:
            raise ValueError("SSH remote immutable spec digest changed")
        return job

    def cancel(
        self,
        remote_dir: PurePosixPath,
        *,
        immutable_spec_digest: str,
    ) -> TransportJob:
        job = self.inspect(remote_dir, immutable_spec_digest=immutable_spec_digest)
        if job is None:
            raise RuntimeError("SSH transport has no exact job to cancel")
        if job.state in {"succeeded", "failed", "cancelled"}:
            return job
        cancelled = TransportJob(
            job.external_id,
            "cancelled",
            job.immutable_spec_digest,
            job.allocation_evidence,
        )
        _atomic_write_json(self._resolve(remote_dir) / "terminal.json", {
            "external_id": cancelled.external_id, "state": cancelled.state,
            "immutable_spec_digest": cancelled.immutable_spec_digest,
            "allocation_evidence": cancelled.allocation_evidence,
        })
        return cancelled

    def fetch(
        self,
        remote_dir: PurePosixPath,
        *,
        immutable_spec_digest: str,
        destination: Path,
    ) -> Mapping[str, str]:
        job = self.inspect(remote_dir, immutable_spec_digest=immutable_spec_digest)
        if job is None or job.state != "succeeded":
            raise RuntimeError("SSH fetch requires exact terminal success")
        if destination.exists():
            raise ValueError("SSH fetch destination must be fresh")
        root = self._resolve(remote_dir)
        prepared = _prepared(_required_json(root / "immutable_spec.json"))
        destination.mkdir(parents=True)
        inventory: dict[str, str] = {}
        for relative in prepared["output_paths"]:
            source = root / "workspace" / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"SSH output is missing or unsafe: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inventory[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.fetched_inventory = tuple(sorted(inventory))
        return inventory

    def _resolve(self, remote_dir: PurePosixPath) -> Path:
        if remote_dir.is_absolute() or any(part in {"", ".", ".."} for part in remote_dir.parts):
            raise ValueError("SSH remote directory is unsafe")
        return self.root.joinpath(*remote_dir.parts)


def _prepared(intent: Mapping[str, object]) -> dict[str, object]:
    value = intent.get("prepared")
    if not isinstance(value, dict):
        raise ValueError("SSH execution requires a prepared command")
    source_root, argv, outputs = value.get("source_root"), value.get("argv"), value.get("output_paths")
    if not isinstance(source_root, str) or not isinstance(argv, list) or not isinstance(outputs, list):
        raise ValueError("SSH prepared execution shape is invalid")
    return {"source_root": source_root, "argv": tuple(argv), "output_paths": tuple(outputs)}


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"SSH transport evidence is malformed: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"SSH transport evidence must be an object: {path.name}")
    return value


def _required_json(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if value is None:
        raise ValueError(f"SSH transport evidence is missing: {path.name}")
    return value
