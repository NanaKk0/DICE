"""Controller cancellation distinct from native task errors."""
from contextlib import contextmanager
import threading
import signal


@contextmanager
def handled_sigterm(operation="sampling"):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt(f"{operation} interrupted by SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
