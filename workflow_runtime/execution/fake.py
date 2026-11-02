from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Mapping

from workflow_runtime.execution.planning import PlannedExecution
from workflow_runtime.execution.protocol import ExternalJobReceipt


class FakeExecutionAdapter:
    adapter_kind = "fake"

    def __init__(self, *, target_id: str):
        self.target_id = target_id
        self.submit_count = 0
        self._jobs_by_launch_key: dict[str, ExternalJobReceipt] = {}
        self._outputs_by_launch_key: dict[str, dict[str, bytes]] = {}

    def submit(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt:
        launch_key = str(launch_intent["launch_key"])
        if launch_key in self._jobs_by_launch_key:
            raise RuntimeError("fake adapter detected a duplicate submit")
        self.submit_count += 1
        receipt = ExternalJobReceipt(
            schema="dice.execution-receipt",
            schema_revision=1,
            unit_id=planned.unit.unit_id,
            attempt_id=planned.attempt_id,
            target_id=planned.target_id,
            adapter=self.adapter_kind,
            external_id=f"fake-job-{self.submit_count:06d}",
            state="queued",
            immutable_spec_digest=str(launch_intent["immutable_spec_digest"]),
        )
        self._jobs_by_launch_key[launch_key] = receipt
        return receipt

    def recover_submission(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt | None:
        receipt = self._jobs_by_launch_key.get(str(launch_intent["launch_key"]))
        if receipt is None:
            return None
        if (
            receipt.unit_id != planned.unit.unit_id
            or receipt.attempt_id != planned.attempt_id
            or receipt.immutable_spec_digest != launch_intent["immutable_spec_digest"]
        ):
            raise ValueError("fake adapter immutable submission evidence changed")
        return receipt

    def poll(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        return self._jobs_by_launch_key[_launch_key(receipt)]

    def cancel(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt:
        current = replace(receipt, state="cancelled")
        self._jobs_by_launch_key[_launch_key(receipt)] = current
        return current

    def set_state(self, receipt: ExternalJobReceipt, state: str) -> ExternalJobReceipt:
        current = replace(receipt, state=state)
        self._jobs_by_launch_key[_launch_key(receipt)] = current
        return current

    def set_outputs(
        self,
        receipt: ExternalJobReceipt,
        outputs: Mapping[str, bytes],
    ) -> None:
        self._outputs_by_launch_key[_launch_key(receipt)] = dict(outputs)

    def collect(
        self,
        receipt: ExternalJobReceipt,
        *,
        destination: Path,
    ) -> Mapping[str, str]:
        if receipt.state != "succeeded":
            raise RuntimeError("fake execution collection requires terminal success")
        if destination.exists():
            raise ValueError("execution collection destination must be fresh")
        outputs = self._outputs_by_launch_key.get(_launch_key(receipt), {})
        destination.mkdir(parents=True)
        inventory: dict[str, str] = {}
        for relative, content in outputs.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            inventory[relative] = hashlib.sha256(content).hexdigest()
        return inventory


def _launch_key(receipt: ExternalJobReceipt) -> str:
    return f"{receipt.unit_id}:{receipt.attempt_id}"
