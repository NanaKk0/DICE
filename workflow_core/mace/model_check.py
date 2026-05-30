"""Compatibility shim for MACE teacher model metadata checks."""

from __future__ import annotations

import sys

from workflow_backends.teacher.mace import model_check as _impl

sys.modules[__name__] = _impl
