"""Compatibility import path for runtime-owned lifecycle execution.

The durable implementation is owned by ``workflow_runtime.execution``.  This
module remains only until Child G removes compatibility import surfaces.
"""

from workflow_runtime.execution.lifecycle import (
    LifecycleExecutionResult,
    PlannedUnitWork,
    execute_planned_block,
    execute_planned_unit,
)


__all__ = [
    "LifecycleExecutionResult",
    "PlannedUnitWork",
    "execute_planned_block",
    "execute_planned_unit",
]
