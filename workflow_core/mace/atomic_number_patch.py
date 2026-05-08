from __future__ import annotations

from typing import Any


def _normalize_atomic_number(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def apply_atomic_number_table_patch() -> None:
    try:
        from mace.tools import utils as mace_utils
    except Exception:
        return

    table_cls = getattr(mace_utils, "AtomicNumberTable", None)
    if table_cls is None:
        return
    if getattr(table_cls, "_workflow_atomic_number_patch_applied", False):
        return

    original_init = table_cls.__init__
    original_z_to_index = table_cls.z_to_index

    def patched_init(self, zs):
        normalized_zs = [_normalize_atomic_number(item) for item in zs]
        original_init(self, normalized_zs)
        try:
            self.zs = [_normalize_atomic_number(item) for item in self.zs]
        except Exception:
            pass

    def patched_z_to_index(self, atomic_number):
        normalized_atomic_number = _normalize_atomic_number(atomic_number)
        try:
            self.zs = [_normalize_atomic_number(item) for item in self.zs]
            return self.zs.index(normalized_atomic_number)
        except Exception:
            return original_z_to_index(self, normalized_atomic_number)

    table_cls.__init__ = patched_init
    table_cls.z_to_index = patched_z_to_index
    table_cls._workflow_atomic_number_patch_applied = True
