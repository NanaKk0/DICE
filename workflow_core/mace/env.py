"""Compatibility shim for MACE backend environment imports."""

from __future__ import annotations

import sys

from workflow_backends.teacher.mace import env as _impl

sys.modules[__name__] = _impl
