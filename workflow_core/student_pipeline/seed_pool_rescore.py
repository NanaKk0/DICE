"""Compatibility shim for student pipeline seed-pool-rescore service imports."""

from __future__ import annotations

import sys

from workflow_core.student_pipeline.services import seed_pool_rescore as _impl

sys.modules[__name__] = _impl
