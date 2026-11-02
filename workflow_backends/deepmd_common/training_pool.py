"""DeepMD opts in to owned MPS without overriding its native allocator policy."""
from workflow_backends.common.training_pool import TrainingPoolCapability, owned_training_sessions

DEEPMD_TRAINING_POOL = TrainingPoolCapability(open_sessions=owned_training_sessions)
