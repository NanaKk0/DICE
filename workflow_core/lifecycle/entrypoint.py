from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_composition: object | None = None,
    executor_factory: Callable[[Mapping[str, Any]], object] | None = None,
    slurm_commands: object | None = None,
) -> object:
    """Forward to the application entrypoint until Child G removes this shim."""

    from workflow_app.entrypoint import main as app_main

    return app_main(
        argv,
        backend_composition=backend_composition,
        executor_factory=executor_factory,
        slurm_commands=slurm_commands,
    )


def console_main() -> None:
    """Forward the legacy console hook until Child G removes this shim."""

    from workflow_app.entrypoint import console_main as app_console_main

    app_console_main()


__all__ = ["console_main", "main"]
