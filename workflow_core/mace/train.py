"""Compatibility shim for MACE backend training imports."""

from __future__ import annotations

import sys

from workflow_backends.teacher.mace import train as _impl

sys.modules[__name__] = _impl
