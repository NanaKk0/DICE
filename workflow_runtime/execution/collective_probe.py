from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded DICE distributed collective preflight")
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--fail-rank", type=int)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("collective preflight timeout must be positive")

    import torch
    import torch.distributed as distributed

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = {
        "schema": "dice.collective-rank-receipt",
        "schema_revision": 1,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": args.backend,
        "rendezvous_id": os.environ.get("TORCHELASTIC_RUN_ID", ""),
        "rendezvous_endpoint": (
            f"{os.environ.get('MASTER_ADDR', '')}:{os.environ.get('MASTER_PORT', '')}"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _atomic_json(output_root / f"started-rank-{rank}.json", {**base, "outcome": "started"})
    initialized = False
    try:
        if args.fail_rank is not None and rank == args.fail_rank:
            raise RuntimeError(f"injected collective rank failure: {rank}")
        if args.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL collective preflight requires CUDA")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            visible_device = str(torch.cuda.current_device())
        else:
            device = torch.device("cpu")
            visible_device = "cpu"
        distributed.init_process_group(
            backend=args.backend,
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        initialized = True
        value = torch.tensor(float(rank + 1), device=device)
        distributed.all_reduce(value, op=distributed.ReduceOp.SUM)
        expected = float(world_size * (world_size + 1) // 2)
        actual = float(value.cpu().item())
        if actual != expected:
            raise RuntimeError(
                f"collective result mismatch: expected {expected}, got {actual}"
            )
        receipt = {
            **base,
            "visible_device": visible_device,
            "all_reduce_result": actual,
            "expected_result": expected,
            "outcome": "succeeded",
        }
        _atomic_json(output_root / f"rank-{rank}.json", receipt)
    except BaseException as exc:
        _atomic_json(
            output_root / f"rank-{rank}.json",
            {**base, "outcome": "failed", "error": str(exc)[:500]},
        )
        raise
    finally:
        if initialized:
            distributed.destroy_process_group()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
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
