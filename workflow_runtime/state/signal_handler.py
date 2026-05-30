from __future__ import annotations

import signal
from types import FrameType
from typing import Any, Callable


def install_signal_handlers(
    state_mgr: Any,
    get_current_step_id: Callable[[], str | None],
) -> None:
    def _handle(signum: int, frame: FrameType | None) -> None:
        step_id = get_current_step_id()
        message = f"Interrupted by signal {signum}"
        if step_id is not None:
            try:
                state_mgr.mark_step_interrupted(step_id, error_message=message)
            except Exception:
                pass
        try:
            state_mgr.mark_workflow_interrupted(message)
        except Exception:
            pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

