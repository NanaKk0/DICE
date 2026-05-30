"""Compatibility shim for MACE teacher committee training helpers."""

from __future__ import annotations

import sys

from workflow_backends.teacher.mace import finetune as _impl

sys.modules[__name__] = _impl

