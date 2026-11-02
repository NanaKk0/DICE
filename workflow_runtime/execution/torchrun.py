from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from workflow_runtime.execution.planning import PlannedExecution


_ENDPOINT = re.compile(r"^[A-Za-z0-9_.:-]+:[0-9]{1,5}$")


@dataclass(frozen=True, slots=True)
class TorchrunNodeLaunch:
    unit_id: str
    attempt_id: str
    target_id: str
    node_rank: int
    nnodes: int
    nproc_per_node: int
    rendezvous_id: str
    rendezvous_backend: str
    rendezvous_endpoint: str
    argv: tuple[str, ...]


def build_single_node_torchrun_arguments(
    *,
    nproc_per_node: int,
    worker_module: str,
    worker_arguments: tuple[str, ...],
) -> tuple[str, ...]:
    """Build one explicit, non-elastic single-node torchrun invocation."""

    if (
        isinstance(nproc_per_node, bool)
        or not isinstance(nproc_per_node, int)
        or nproc_per_node < 2
    ):
        raise ValueError("single-node torchrun requires at least two processes")
    if not isinstance(worker_module, str) or not worker_module.strip():
        raise ValueError("single-node torchrun requires one worker module")
    if not worker_arguments or any(not isinstance(item, str) for item in worker_arguments):
        raise ValueError("single-node torchrun requires worker arguments")
    return (
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc_per_node}",
        "--module",
        worker_module,
        *worker_arguments,
    )


def build_torchrun_node_launches(
    planned: PlannedExecution,
    *,
    target_ids: tuple[str, ...],
    rendezvous_endpoint: str,
    backend_argv: tuple[str, ...],
) -> tuple[TorchrunNodeLaunch, ...]:
    unit = planned.unit
    shape = unit.resource_shape
    if unit.launcher != "torchrun":
        raise ValueError("distributed launch requires explicit launcher: torchrun")
    if (
        shape["nodes"] != 2
        or shape["processes_per_node"] != 1
        or shape["gpus_per_node"] != 1
    ):
        raise ValueError("torchrun requires two nodes with one process/GPU per node")
    if len(target_ids) != 2 or len(set(target_ids)) != 2:
        raise ValueError("torchrun requires two distinct bound execution targets")
    if not isinstance(rendezvous_endpoint, str) or not _ENDPOINT.fullmatch(rendezvous_endpoint):
        raise ValueError("torchrun rendezvous endpoint is invalid")
    port = int(rendezvous_endpoint.rsplit(":", 1)[1])
    if port <= 0 or port > 65535:
        raise ValueError("torchrun rendezvous port is invalid")
    if tuple(backend_argv[:3]) != ("dp", "--pt-expt", "train"):
        raise ValueError("DPA4C distributed backend must own dp --pt-expt train")
    rendezvous_id = "dice-" + hashlib.sha256(
        f"{unit.unit_id}\0{planned.attempt_id}".encode("utf-8")
    ).hexdigest()[:24]
    launches: list[TorchrunNodeLaunch] = []
    for node_rank, target_id in enumerate(target_ids):
        prefix = (
            "torchrun",
            "--nnodes=2",
            "--nproc-per-node=1",
            f"--node-rank={node_rank}",
            f"--rdzv-id={rendezvous_id}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rendezvous_endpoint}",
        )
        launches.append(
            TorchrunNodeLaunch(
                unit_id=unit.unit_id,
                attempt_id=planned.attempt_id,
                target_id=target_id,
                node_rank=node_rank,
                nnodes=2,
                nproc_per_node=1,
                rendezvous_id=rendezvous_id,
                rendezvous_backend="c10d",
                rendezvous_endpoint=rendezvous_endpoint,
                argv=prefix + tuple(backend_argv),
            )
        )
    return tuple(launches)
