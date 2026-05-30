"""Compatibility shim for teacher committee-finetune service imports."""

from __future__ import annotations

import sys

from workflow_core.teacher import committee_finetune as _impl

sys.modules[__name__] = _impl
