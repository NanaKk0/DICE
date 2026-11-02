from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

from workflow_runtime.execution.planning import PlannedExecution
from workflow_runtime.execution.protocol import (
    EXECUTION_STATES,
    ExecutionAdapter,
    ExternalJobReceipt,
    PreparedExecution,
)
from workflow_runtime.execution.store import ExecutionStore


class DurableExecutionController:
    def __init__(
        self,
        *,
        store: ExecutionStore,
        adapters: Mapping[str, ExecutionAdapter],
    ):
        self.store = store
        self.adapters = dict(adapters)

    def submit(
        self,
        planned: PlannedExecution,
        *,
        input_inventory: Mapping[str, str] | None = None,
        prepared: PreparedExecution | None = None,
        inject_crash_after_submit: bool = False,
    ) -> ExternalJobReceipt:
        adapter = self._adapter(planned)
        binding = {
            "schema": "dice.execution-binding",
            "schema_revision": 1,
            "unit_id": planned.unit.unit_id,
            "target_id": planned.target_id,
            "target_pool": planned.target_pool,
            "pool_digest": planned.pool_digest,
            "profile_digest": planned.profile_digest,
            "attempt_id": planned.attempt_id,
        }
        self.store.publish_exact(planned, "binding.json", binding)
        if prepared is not None and input_inventory is not None:
            raise ValueError("execution submit accepts prepared or input_inventory, not both")
        inputs = prepared.input_inventory if prepared is not None else input_inventory
        if inputs is None:
            raise ValueError("execution submit requires an exact input inventory")
        if prepared is not None:
            _validate_prepared_inputs(prepared)
        intent = _launch_intent(
            planned,
            adapter.adapter_kind,
            inputs,
            prepared=prepared,
        )
        existing_intent = self.store.read(planned, "launch_intent.json")
        if existing_intent is not None and existing_intent != intent:
            raise ValueError("durable execution launch intent changed")
        self.store.publish_exact(planned, "launch_intent.json", intent)

        existing_receipt = self.store.read(planned, "external_job.json")
        if existing_receipt is not None:
            return _receipt(existing_receipt, planned=planned, intent=intent)
        if existing_intent is not None:
            recovered = adapter.recover_submission(planned, launch_intent=intent)
            if recovered is None:
                raise RuntimeError(
                    "execution submission is indeterminate; refusing duplicate launch"
                )
            self._publish_receipt(planned, recovered, intent=intent)
            return recovered

        receipt = adapter.submit(planned, launch_intent=intent)
        if inject_crash_after_submit:
            raise RuntimeError("injected crash after submit")
        self._publish_receipt(planned, receipt, intent=intent)
        return receipt

    def poll(self, planned: PlannedExecution) -> ExternalJobReceipt:
        intent = self._required_intent(planned)
        current = self._required_receipt(planned, intent=intent)
        observed = self._adapter(planned).poll(current)
        _validate_receipt(observed, planned=planned, intent=intent)
        self.store.replace(planned, "external_job.json", observed.as_dict())
        if observed.state in {"succeeded", "failed", "cancelled"}:
            self.store.publish_exact(
                planned, "terminal_receipt.json", observed.as_dict()
            )
        return observed

    def cancel(self, planned: PlannedExecution) -> ExternalJobReceipt:
        intent = self._required_intent(planned)
        current = self._required_receipt(planned, intent=intent)
        cancelled = self._adapter(planned).cancel(current)
        _validate_receipt(cancelled, planned=planned, intent=intent)
        if cancelled.state != "cancelled":
            raise RuntimeError("execution adapter did not prove cancellation")
        self.store.replace(planned, "external_job.json", cancelled.as_dict())
        self.store.publish_exact(
            planned, "terminal_receipt.json", cancelled.as_dict()
        )
        return cancelled

    def collect_and_validate(
        self,
        planned: PlannedExecution,
        *,
        destination: str | Path,
        expected_output_inventory: Mapping[str, str],
        backend_validator,
        canonical_validator,
    ) -> Path:
        intent = self._required_intent(planned)
        receipt = self._required_receipt(planned, intent=intent)
        if receipt.state != "succeeded":
            raise RuntimeError("execution collection requires terminal success")
        output = Path(destination).resolve()
        collected_inventory = dict(
            self._adapter(planned).collect(receipt, destination=output)
        )
        expected = dict(sorted(expected_output_inventory.items()))
        actual = _inventory(output)
        if collected_inventory != expected or actual != expected:
            raise ValueError("collected execution output inventory is not exact")
        collection = {
            "schema": "dice.execution-collection-receipt",
            "schema_revision": 1,
            "unit_id": planned.unit.unit_id,
            "attempt_id": planned.attempt_id,
            "external_id": receipt.external_id,
            "output_inventory": expected,
        }
        self.store.publish_exact(planned, "collection_receipt.json", collection)
        if backend_validator(output) is not True:
            raise ValueError("backend-native execution output validation failed")
        if canonical_validator(output) is not True:
            raise ValueError("canonical execution output validation failed")
        self.store.publish_exact(
            planned,
            "validation_receipt.json",
            {
                "schema": "dice.execution-validation-receipt",
                "schema_revision": 1,
                "unit_id": planned.unit.unit_id,
                "attempt_id": planned.attempt_id,
                "external_id": receipt.external_id,
                "backend_native": "passed",
                "canonical": "passed",
                "output_inventory": expected,
            },
        )
        return output

    def _required_intent(self, planned: PlannedExecution) -> dict[str, object]:
        intent = self.store.read(planned, "launch_intent.json")
        if intent is None:
            raise ValueError("durable execution launch intent is missing")
        return intent

    def _required_receipt(
        self,
        planned: PlannedExecution,
        *,
        intent: Mapping[str, object],
    ) -> ExternalJobReceipt:
        value = self.store.read(planned, "external_job.json")
        if value is None:
            raise ValueError("durable external job receipt is missing")
        return _receipt(value, planned=planned, intent=intent)

    def _adapter(self, planned: PlannedExecution) -> ExecutionAdapter:
        try:
            adapter = self.adapters[planned.target_id]
        except KeyError as exc:
            raise ValueError(
                f"no execution adapter registered for target {planned.target_id!r}"
            ) from exc
        if adapter.target_id != planned.target_id:
            raise ValueError("execution adapter target identity mismatch")
        return adapter

    def _publish_receipt(
        self,
        planned: PlannedExecution,
        receipt: ExternalJobReceipt,
        *,
        intent: Mapping[str, object],
    ) -> None:
        _validate_receipt(receipt, planned=planned, intent=intent)
        self.store.publish_exact(planned, "external_job.json", receipt.as_dict())


def _launch_intent(
    planned: PlannedExecution,
    adapter_kind: str,
    input_inventory: Mapping[str, str],
    *,
    prepared: PreparedExecution | None = None,
) -> dict[str, object]:
    inputs = dict(sorted(input_inventory.items()))
    for path, digest in inputs.items():
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"execution input inventory path is unsafe: {path!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"execution input digest is invalid for {path!r}")
    immutable = {
        "unit": asdict(planned.unit),
        "target_id": planned.target_id,
        "adapter": adapter_kind,
        "input_inventory": inputs,
    }
    prepared_payload: dict[str, object] | None = None
    if prepared is not None:
        if not prepared.argv or not all(isinstance(value, str) and value for value in prepared.argv):
            raise ValueError("prepared execution argv must be a non-empty string tuple")
        output_paths = tuple(_safe_relative(path, label="execution output") for path in prepared.output_paths)
        if not output_paths or len(set(output_paths)) != len(output_paths):
            raise ValueError("prepared execution output inventory must be non-empty and unique")
        prepared_payload = {
            "source_root": str(Path(prepared.source_root).resolve()),
            "argv": list(prepared.argv),
            "output_paths": list(output_paths),
        }
        immutable["prepared"] = prepared_payload
    spec_digest = _digest(immutable)
    result = {
        "schema": "dice.execution-launch-intent",
        "schema_revision": 1,
        "unit_id": planned.unit.unit_id,
        "attempt_id": planned.attempt_id,
        "target_id": planned.target_id,
        "adapter": adapter_kind,
        "launch_key": f"{planned.unit.unit_id}:{planned.attempt_id}",
        "immutable_spec_digest": spec_digest,
        "input_inventory": inputs,
    }
    if prepared_payload is not None:
        result["prepared"] = prepared_payload
    return result


def _receipt(
    value: Mapping[str, object],
    *,
    planned: PlannedExecution,
    intent: Mapping[str, object],
) -> ExternalJobReceipt:
    try:
        receipt = ExternalJobReceipt(**value)
    except TypeError as exc:
        raise ValueError("durable external job receipt shape is invalid") from exc
    _validate_receipt(receipt, planned=planned, intent=intent)
    return receipt


def _validate_receipt(
    receipt: ExternalJobReceipt,
    *,
    planned: PlannedExecution,
    intent: Mapping[str, object],
) -> None:
    if receipt.schema != "dice.execution-receipt" or receipt.schema_revision != 1:
        raise ValueError("execution receipt schema is invalid")
    if receipt.state not in EXECUTION_STATES:
        raise ValueError(f"execution receipt state is invalid: {receipt.state!r}")
    if (
        receipt.unit_id != planned.unit.unit_id
        or receipt.attempt_id != planned.attempt_id
        or receipt.target_id != planned.target_id
        or receipt.immutable_spec_digest != intent["immutable_spec_digest"]
    ):
        raise ValueError("execution receipt does not match immutable launch intent")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError("execution collection root is missing")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("execution collection may not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError(f"{label} path is unsafe: {value!r}")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} path is unsafe: {value!r}")
    return path.as_posix()


def _validate_prepared_inputs(prepared: PreparedExecution) -> None:
    source_root = Path(prepared.source_root).resolve()
    if not source_root.is_dir():
        raise ValueError("prepared execution source root is missing")
    for raw_relative, expected_digest in sorted(prepared.input_inventory.items()):
        relative = _safe_relative(raw_relative, label="execution input")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"execution input digest is invalid for {relative!r}")
        source = source_root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"execution input is missing or unsafe: {relative}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected_digest:
            raise ValueError(f"execution input digest changed: {relative}")
