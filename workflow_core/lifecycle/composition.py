"""Lazy compatibility exports for the application composition root.

Child G removes this module after all callers use ``workflow_app.composition``.
No provider selection or concrete wiring belongs here during the migration.
"""

from __future__ import annotations


_COMPATIBILITY_EXPORTS = frozenset(
    {
        "BackendComposition",
        "SelectedBackendAdapters",
        "bind_lifecycle_screen",
        "compose_backend_roles",
        "sampling_backend_binding",
        "student_backend_binding",
        "teacher_backend_binding",
    }
)


def __getattr__(name: str) -> object:
    if name not in _COMPATIBILITY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from workflow_app import composition as app_composition

    return getattr(app_composition, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_COMPATIBILITY_EXPORTS})


__all__ = sorted(_COMPATIBILITY_EXPORTS)
