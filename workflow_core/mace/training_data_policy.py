"""Compatibility shim for teacher training-data policy imports."""

from __future__ import annotations

import sys

from workflow_core.teacher import training_data_policy as _impl

sys.modules[__name__] = _impl
