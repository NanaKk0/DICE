"""Task isolation without backend-specific scientific eligibility rules."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    status: str
    observations: tuple[str, ...] = ()
    diagnostic: str = ''

    def __post_init__(self):
        if not self.task_id or self.status not in {'completed', 'failed'}:
            raise ValueError('invalid terminal task outcome')


def execute_task(
    task_id: str, *, run: Callable[[], TaskOutcome],
    retain_failure: Callable[[Exception], TaskOutcome],
) -> TaskOutcome:
    """Only adapters can certify a prefix; cancellation is never salvaged."""
    try:
        outcome = run()
    except Exception as error:
        outcome = retain_failure(error)
        if outcome.status != 'failed':
            raise ValueError('failure retention must preserve failed status')
    if outcome.task_id != task_id:
        raise ValueError('terminal task identity does not match inventory')
    return outcome
