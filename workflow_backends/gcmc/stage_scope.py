"""Local single-writer scope, not a resumable sampling checkpoint."""
from contextvars import ContextVar
import fcntl
from functools import wraps
import os


_owned = ContextVar('gcmc_sample_writers', default=frozenset())


def single_sample_writer(execute):
    """Hold one lock across nested native branches and all worker lifetimes."""
    @wraps(execute)
    def guarded(self, request):
        evidence = request.evidence
        if not any(t.kind in {'gcmc_md', 'widom'} for t in evidence.plan.tasks_for_round(evidence.round_index)):
            return execute(self, request)
        root = evidence.sample_dir.resolve()
        key = (os.getpid(), str(root))
        if key in _owned.get():
            return execute(self, request)
        root.mkdir(parents=True, exist_ok=True)
        with (root/'.gcmc-stage.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('GCMC sample already has an active local writer') from exc
            token = _owned.set(_owned.get() | {key})
            try:
                return execute(self, request)
            finally:
                _owned.reset(token)
    return guarded
