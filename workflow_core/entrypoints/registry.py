from __future__ import annotations

from workflow_core.entrypoints.dft import DFT_TASKS
from workflow_core.entrypoints.dp import DP_TASKS
from workflow_core.entrypoints.labeling import LABELING_TASKS
from workflow_core.entrypoints.mace import MACE_TASKS
from workflow_core.entrypoints.pipeline import PIPELINE_TASKS
from workflow_core.entrypoints.seed_pool import SEED_POOL_TASKS


TASK_REGISTRY = {}
TASK_REGISTRY.update(MACE_TASKS)
TASK_REGISTRY.update(PIPELINE_TASKS)
TASK_REGISTRY.update(LABELING_TASKS)
TASK_REGISTRY.update(DP_TASKS)
TASK_REGISTRY.update(DFT_TASKS)
TASK_REGISTRY.update(SEED_POOL_TASKS)
