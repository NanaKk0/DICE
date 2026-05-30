"""Compatibility shim for MACE teacher backend helpers."""

from __future__ import annotations

import sys

from workflow_backends.teacher.mace import backend as _impl

sys.modules[__name__] = _impl
