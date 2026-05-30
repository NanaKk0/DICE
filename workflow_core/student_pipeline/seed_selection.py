"""Compatibility shim for student pipeline seed-selection service imports."""

from __future__ import annotations

import sys

from workflow_core.student_pipeline.services import seed_selection as _impl

sys.modules[__name__] = _impl
