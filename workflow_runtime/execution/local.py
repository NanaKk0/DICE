from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from workflow_runtime.execution.planning import PlannedExecution
from workflow_runtime.execution.gpu_lease import LocalGPULeaseManager
from workflow_runtime.execution.protocol import ExternalJobReceipt
from workflow_runtime.execution.store import _atomic_write_json


class LocalExecutionAdapter:
    adapter_kind = "local"

    def __init__(self, *, target_id: str, runtime_root: str | Path):
        self.target_id = target_id
        self.runtime_root = Path(runtime_root).resolve()
        self.gpu_leases = LocalGPULeaseManager(self.runtime_root / "gpu-leases")

    def submit(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt:
        job_root = self._job_root(planned)
        if (job_root / "immutable_spec.json").exists():
            raise RuntimeError("local execution adapter refuses duplicate submit")
        prepared = _prepared(launch_intent)
        gpu_count = int(planned.unit.resource_shape["gpus_per_node"])
        lease = (
            None
            if gpu_count == 0
            else self.gpu_leases.acquire(
                owner=f"{planned.unit.unit_id}:{planned.attempt_id}",
                count=gpu_count,
            )
        )
        workspace = job_root / "workspace"
        try:
            workspace.mkdir(parents=True)
            _stage_inputs(prepared, launch_intent, workspace=workspace)
            _atomic_write_json(job_root / "immutable_spec.json", dict(launch_intent))
            external_id = f"local-{hashlib.sha256(str(launch_intent['launch_key']).encode()).hexdigest()[:16]}"
            stdout_path = job_root / "stdout.log"
            stderr_path = job_root / "stderr.log"
            environment = os.environ.copy()
            if lease is not None:
                environment.update(lease.environment_override)
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    completed = subprocess.run(
                        list(prepared["argv"]),
                        cwd=workspace,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=int(planned.unit.resource_shape["wall_time_seconds"]),
                        check=False,
                    )
                state = "succeeded" if completed.returncode == 0 else "failed"
                return_code: int | None = completed.returncode
            except subprocess.TimeoutExpired:
                state = "failed"
                return_code = None
        finally:
            if lease is not None:
                lease.release()
        receipt = ExternalJobReceipt(
            schema="dice.execution-receipt",
            schema_revision=1,
            unit_id=planned.unit.unit_id,
            attempt_id=planned.attempt_id,
            target_id=planned.target_id,
            adapter=self.adapter_kind,
            external_id=external_id,
            state=state,
            immutable_spec_digest=str(launch_intent["immutable_spec_digest"]),
            allocation_evidence=(
                None
                if lease is None
                else {
                    "cuda_visible_devices": list(lease.visible_devices),
                    "exclusive_local_lease": True,
                }
            ),
        )
        _atomic_write_json(
            job_root / "terminal.json",
            {**receipt.as_dict(), "return_code": return_code},
        )
        return receipt

    def recover_submission(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt | None:
        job_root = self._job_root(planned)
        spec = _read_json(job_root / "immutable_spec.json")
        if spec is None:
            return None
        if spec != dict(launch_intent):
            raise ValueError("local execution immutable spec changed")
        return self._terminal_receipt(job_root)

    def poll(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        planned_root = self._receipt_job_root(receipt)
        terminal = self._terminal_receipt(planned_root)
        if terminal is None:
            return receipt
        return terminal

    def cancel(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        terminal = self.poll(receipt)
        if terminal.state in {"succeeded", "failed", "cancelled"}:
            return terminal
        raise RuntimeError("synchronous local execution has no unowned live process to cancel")

    def collect(
        self,
        receipt: ExternalJobReceipt,
        *,
        destination: Path,
    ) -> Mapping[str, str]:
        terminal = self.poll(receipt)
        if terminal.state != "succeeded":
            raise RuntimeError("local execution collection requires terminal success")
        if destination.exists():
            raise ValueError("execution collection destination must be fresh")
        job_root = self._receipt_job_root(receipt)
        spec = _read_json(job_root / "immutable_spec.json")
        if spec is None:
            raise ValueError("local execution immutable spec is missing")
        prepared = _prepared(spec)
        workspace = job_root / "workspace"
        destination.mkdir(parents=True)
        result: dict[str, str] = {}
        for relative in prepared["output_paths"]:
            source = workspace / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"local execution output is missing or unsafe: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            result[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        return result

    def _job_root(self, planned: PlannedExecution) -> Path:
        return self.runtime_root / planned.unit.unit_id / planned.attempt_id

    def _receipt_job_root(self, receipt: ExternalJobReceipt) -> Path:
        return self.runtime_root / receipt.unit_id / receipt.attempt_id

    @staticmethod
    def _terminal_receipt(job_root: Path) -> ExternalJobReceipt | None:
        value = _read_json(job_root / "terminal.json")
        if value is None:
            return None
        value.pop("return_code", None)
        try:
            return ExternalJobReceipt(**value)
        except TypeError as exc:
            raise ValueError("local execution terminal receipt is malformed") from exc


def _prepared(intent: Mapping[str, object]) -> dict[str, object]:
    value = intent.get("prepared")
    if not isinstance(value, dict):
        raise ValueError("local execution requires a prepared command")
    source_root = value.get("source_root")
    argv = value.get("argv")
    outputs = value.get("output_paths")
    if not isinstance(source_root, str) or not isinstance(argv, list) or not isinstance(outputs, list):
        raise ValueError("local prepared execution shape is invalid")
    return {"source_root": source_root, "argv": tuple(argv), "output_paths": tuple(outputs)}


def _stage_inputs(
    prepared: Mapping[str, object],
    intent: Mapping[str, object],
    *,
    workspace: Path,
) -> None:
    source_root = Path(str(prepared["source_root"])).resolve()
    inventory = intent.get("input_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("local execution input inventory is invalid")
    for relative, expected_digest in sorted(inventory.items()):
        source = source_root / str(relative)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"local execution input is missing or unsafe: {relative}")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise ValueError(f"local execution input digest changed: {relative}")
        target = workspace / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"local execution evidence is malformed: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"local execution evidence must be an object: {path.name}")
    return value
