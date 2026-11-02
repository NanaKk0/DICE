from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from workflow_app.composition import (
    BackendComposition,
    dispatch_slurm_lifecycle_block,
    is_slurm_lifecycle_dispatch,
)
from workflow_core.lifecycle.launch import LifecycleLaunch
from workflow_core.lifecycle.state_model import PublicBlockId, PublicLifecyclePlan
from workflow_core.lifecycle.runner import (
    execute_public_lifecycle_block,
)


def suffix_uses_public_slurm(
    launch: LifecycleLaunch,
    *,
    first_block: str,
) -> bool:
    """Return whether the prepared invocation suffix needs block dispatch."""

    block_ids = tuple(block.block_id for block in launch.plan.blocks)
    if first_block not in block_ids:
        raise ValueError(f"invalid lifecycle execution block {first_block!r}")
    return any(
        is_slurm_lifecycle_dispatch(launch, block_id)
        for block_id in block_ids[block_ids.index(first_block) :]
    )


def execute_public_lifecycle_suffix(
    launch: LifecycleLaunch,
    *,
    first_block: str,
    backend_composition: BackendComposition | None,
    composition_factory: Callable[[], BackendComposition],
    slurm_commands: object | None = None,
) -> None:
    """Advance one prepared public suffix through block-scoped providers.

    Core remains the sequencing and completion authority. Invocation entry has
    already reconciled the durable prefix, so the healthy success path advances
    directly through the fixed plan. A later controller invocation performs a
    fresh full reconciliation before entering this loop. Slurm workers still
    execute exactly one block and therefore cannot recursively submit scheduler
    work.
    """

    current_launch = launch
    block_id: str | None = first_block
    semantic_plan = PublicLifecyclePlan(
        tuple(PublicBlockId(block.block_id) for block in launch.plan.blocks)
    )
    first = PublicBlockId(first_block)
    semantic_plan.index(first)
    composition = backend_composition
    while block_id is not None:
        if is_slurm_lifecycle_dispatch(current_launch, block_id):
            result = dispatch_slurm_lifecycle_block(
                current_launch,
                block_id=block_id,
                commands=slurm_commands,
                backend_composition=backend_composition,
            )
        else:
            if composition is None:
                composition = composition_factory()
            execute_public_lifecycle_block(
                current_launch,
                block_id=block_id,
                backend_composition=composition,
            )

        # A public --resume target is an invocation-entry intent. Once that
        # exact block has succeeded, continuation must follow durable state
        # instead of selecting and invalidating the same target again.
        if current_launch.resume_target is not None:
            current_launch = replace(current_launch, resume_target=None)
        completed = PublicBlockId(block_id)
        successor = semantic_plan.successor(completed)
        block_id = None if successor is None else str(successor)


__all__ = ["execute_public_lifecycle_suffix", "suffix_uses_public_slurm"]
