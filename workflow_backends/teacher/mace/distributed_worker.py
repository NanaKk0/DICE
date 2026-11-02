from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description="MACE single-node rank worker")
    parser.add_argument("--receipt-root", required=True)
    parser.add_argument("--expected-world-size", required=True, type=int)
    parser.add_argument("--visible-devices-json", required=True)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if not command:
        raise ValueError("MACE distributed rank worker requires a native command")
    visible_devices_raw = json.loads(args.visible_devices_json)
    if (
        not isinstance(visible_devices_raw, list)
        or len(visible_devices_raw) != args.expected_world_size
        or any(not isinstance(item, str) or not item for item in visible_devices_raw)
        or len(set(visible_devices_raw)) != len(visible_devices_raw)
    ):
        raise ValueError("MACE distributed rank worker device inventory is invalid")
    visible_devices = tuple(visible_devices_raw)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if (
        world_size != args.expected_world_size
        or rank != local_rank
        or rank < 0
        or rank >= world_size
        or tuple(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != visible_devices
        or os.environ.get("DICE_ASSIGNMENT_ID") != args.assignment_id
    ):
        raise ValueError("MACE distributed rank runtime does not match its assignment")
    root = Path(args.receipt_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stdout_path = root / f"rank-{rank}.stdout.log"
    stderr_path = root / f"rank-{rank}.stderr.log"
    native_environment = dict(os.environ)
    # MACE relies on the torchrun LOCAL_RANK to select one device from the complete
    # Slurm allocation. Keep the full allocation visible to every native rank;
    # the receipt below records the resulting rank-to-device mapping.
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        bootstrap = (
            f"DICE_DDP rank={rank} local_rank={local_rank} Processes: {world_size} "
            f"assigned_device={visible_devices[local_rank]} "
            f"visible_devices={','.join(visible_devices)}\n"
        ).encode("utf-8")
        stdout.write(bootstrap)
        stdout.flush()
        result = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
            env=native_environment,
        )
    outcome = "succeeded" if result.returncode == 0 else "failed"
    _atomic_json(
        root / f"rank-{rank}.json",
        {
            "schema": "dice.single-node-rank-receipt",
            "schema_revision": 1,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "visible_devices": list(visible_devices),
            "visible_device": visible_devices[local_rank],
            "assignment_id": args.assignment_id,
            "pid": os.getpid(),
            "command": list(command),
            "terminal_outcome": outcome,
            "return_code": result.returncode,
        },
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
