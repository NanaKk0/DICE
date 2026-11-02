from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DistributedRankReceipt:
    schema: str
    schema_revision: int
    unit_id: str
    attempt_id: str
    target_id: str
    expected_rank: int
    actual_rank: int
    local_rank: int
    expected_world_size: int
    actual_world_size: int
    visible_device: str
    rendezvous_id: str
    rendezvous_backend: str
    rendezvous_endpoint: str
    input_digests: Mapping[str, str]
    terminal_outcome: str


@dataclass(frozen=True, slots=True)
class SingleNodeRankReceipt:
    schema: str
    schema_revision: int
    rank: int
    local_rank: int
    world_size: int
    visible_devices: tuple[str, ...]
    visible_device: str
    assignment_id: str
    pid: int
    command: tuple[str, ...]
    terminal_outcome: str
    return_code: int


def load_and_validate_single_node_rank_receipts(
    root: str | Path,
    *,
    expected_world_size: int,
    expected_visible_devices: tuple[str, ...],
    expected_assignment_id: str,
    expected_command: tuple[str, ...],
) -> tuple[SingleNodeRankReceipt, ...]:
    receipt_root = Path(root)
    expected_paths = tuple(
        receipt_root / f"rank-{rank}.json" for rank in range(expected_world_size)
    )
    if tuple(sorted(receipt_root.glob("rank-*.json"))) != expected_paths:
        raise ValueError("single-node distributed rank receipt inventory is invalid")
    receipts: list[SingleNodeRankReceipt] = []
    for rank in range(expected_world_size):
        path = receipt_root / f"rank-{rank}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"single-node distributed rank receipt is missing or invalid: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("single-node distributed rank receipt must be an object")
        if set(payload) != {
            "schema",
            "schema_revision",
            "rank",
            "local_rank",
            "world_size",
            "visible_devices",
            "visible_device",
            "assignment_id",
            "pid",
            "command",
            "terminal_outcome",
            "return_code",
        }:
            raise ValueError("single-node distributed rank receipt shape is invalid")
        try:
            receipt = SingleNodeRankReceipt(
                schema=payload["schema"],
                schema_revision=payload["schema_revision"],
                rank=payload["rank"],
                local_rank=payload["local_rank"],
                world_size=payload["world_size"],
                visible_devices=tuple(payload["visible_devices"]),
                visible_device=payload["visible_device"],
                assignment_id=payload["assignment_id"],
                pid=payload["pid"],
                command=tuple(payload["command"]),
                terminal_outcome=payload["terminal_outcome"],
                return_code=payload["return_code"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("single-node distributed rank receipt shape is invalid") from exc
        receipts.append(receipt)
    validate_single_node_rank_receipts(
        tuple(receipts),
        expected_world_size=expected_world_size,
        expected_visible_devices=expected_visible_devices,
        expected_assignment_id=expected_assignment_id,
        expected_command=expected_command,
    )
    return tuple(receipts)


def validate_single_node_rank_receipts(
    receipts: tuple[SingleNodeRankReceipt, ...],
    *,
    expected_world_size: int,
    expected_visible_devices: tuple[str, ...],
    expected_assignment_id: str,
    expected_command: tuple[str, ...],
) -> None:
    if expected_world_size < 2 or len(receipts) != expected_world_size:
        raise ValueError("single-node distributed rank receipt count is invalid")
    if (
        len(expected_visible_devices) != expected_world_size
        or len(set(expected_visible_devices)) != expected_world_size
        or any(not device for device in expected_visible_devices)
    ):
        raise ValueError("single-node distributed expected device inventory is invalid")
    ordered = tuple(sorted(receipts, key=lambda receipt: receipt.rank))
    for rank, receipt in enumerate(ordered):
        if (
            type(receipt.rank) is not int
            or type(receipt.local_rank) is not int
            or type(receipt.world_size) is not int
            or type(receipt.pid) is not int
            or type(receipt.return_code) is not int
            or any(not isinstance(item, str) or not item for item in receipt.visible_devices)
            or any(not isinstance(item, str) or not item for item in receipt.command)
            or not isinstance(receipt.visible_device, str)
            or not isinstance(receipt.assignment_id, str)
            or not receipt.assignment_id
        ):
            raise ValueError("single-node distributed rank receipt values are invalid")
        if receipt.schema != "dice.single-node-rank-receipt" or receipt.schema_revision != 1:
            raise ValueError("single-node distributed rank receipt schema is invalid")
        if receipt.rank != rank or receipt.local_rank != rank:
            raise ValueError("single-node distributed actual rank shape is invalid")
        if receipt.world_size != expected_world_size:
            raise ValueError("single-node distributed actual world size is invalid")
        if receipt.visible_devices != expected_visible_devices:
            raise ValueError("single-node distributed visible-device inventory drifted")
        if receipt.visible_device != expected_visible_devices[rank]:
            raise ValueError("single-node distributed rank/device mapping is invalid")
        if receipt.assignment_id != expected_assignment_id:
            raise ValueError("single-node distributed assignment identity drifted")
        if receipt.command != expected_command:
            raise ValueError("single-node distributed native command drifted")
        if receipt.pid <= 0 or receipt.terminal_outcome != "succeeded" or receipt.return_code != 0:
            raise ValueError("single-node distributed rank did not succeed")
    if len({receipt.pid for receipt in ordered}) != expected_world_size:
        raise ValueError("single-node distributed rank process identities are not unique")


def validate_two_node_rank_receipts(
    receipts: tuple[DistributedRankReceipt, ...],
) -> None:
    if len(receipts) != 2:
        raise ValueError("two-node execution requires exactly two rank receipts")
    if any(
        receipt.schema != "dice.distributed-rank-receipt"
        or receipt.schema_revision != 1
        for receipt in receipts
    ):
        raise ValueError("distributed rank receipt schema is invalid")
    if len({receipt.unit_id for receipt in receipts}) != 1 or len(
        {receipt.attempt_id for receipt in receipts}
    ) != 1:
        raise ValueError("distributed rank receipts do not bind one unit attempt")
    if len({receipt.target_id for receipt in receipts}) != 2:
        raise ValueError("distributed rank receipts require two distinct targets")
    ordered = tuple(sorted(receipts, key=lambda receipt: receipt.expected_rank))
    if tuple(receipt.expected_rank for receipt in ordered) != (0, 1):
        raise ValueError("distributed expected rank inventory is invalid")
    if any(receipt.actual_rank != receipt.expected_rank for receipt in ordered):
        raise ValueError("distributed actual rank does not match its planned rank")
    if any(receipt.local_rank != 0 for receipt in ordered):
        raise ValueError("two-node execution requires actual local rank zero on each node")
    if any(
        receipt.expected_world_size != 2 or receipt.actual_world_size != 2
        for receipt in ordered
    ):
        raise ValueError("distributed actual world size must equal two")
    rendezvous = {
        (
            receipt.rendezvous_id,
            receipt.rendezvous_backend,
            receipt.rendezvous_endpoint,
        )
        for receipt in ordered
    }
    if len(rendezvous) != 1 or next(iter(rendezvous))[1] != "c10d":
        raise ValueError("distributed rank receipts do not share one c10d rendezvous")
    if any(not receipt.visible_device for receipt in ordered):
        raise ValueError("distributed rank receipt is missing actual visible device")
    digests = [dict(receipt.input_digests) for receipt in ordered]
    if not digests[0] or digests[0] != digests[1]:
        raise ValueError("distributed rank input digests are missing or unequal")
    if any(
        not path
        or not isinstance(digest, str)
        or len(digest) != 64
        for path, digest in digests[0].items()
    ):
        raise ValueError("distributed rank input digests are invalid")
    if any(receipt.terminal_outcome != "succeeded" for receipt in ordered):
        raise ValueError(
            "distributed terminal outcome requires actual successful rank execution"
        )
