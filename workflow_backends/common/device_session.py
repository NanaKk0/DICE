"""Device-local admission barrier; a lease is never retried across generations."""
from contextlib import contextmanager
from threading import Condition, Thread
from time import monotonic


class DeviceSessionSupervisor:
    def __init__(self, *, start, stop, poll, probe, interval=0.5,
                 max_recoveries=3, drain_timeout=60.0, on_fault=None):
        if interval <= 0 or drain_timeout <= 0 or max_recoveries < 0:
            raise ValueError('device recovery bounds must be nonnegative with positive timeouts')
        self.start, self.stop, self.poll, self.probe = start, stop, poll, probe
        self.interval, self.max_recoveries = interval, max_recoveries
        self.drain_timeout = drain_timeout
        self.on_fault = on_fault
        self.condition = Condition()
        self.generation = 0
        self.active = 0
        self.fault = None
        self.error = None
        self.closed = False
        self.environment = None
        self.thread = None

    def open(self):
        self.environment = self.start(0)
        self.thread = Thread(target=self._monitor, name='dice-device-session', daemon=True)
        self.thread.start()
        return self

    def report(self, kind, detail):
        with self.condition:
            if not self.closed and self.fault is None:
                self.fault = (kind, detail)
                self.condition.notify_all()
                if self.on_fault is not None:
                    self.on_fault(self.generation, kind, detail)

    @contextmanager
    def admit(self):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.error or self.fault is None)
            if self.error:
                raise RuntimeError(str(self.error)) from self.error
            if self.closed:
                raise RuntimeError('device session is closed')
            environment = self.environment
            self.active += 1
        try:
            yield environment
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def _monitor(self):
        try:
            while True:
                with self.condition:
                    if self.closed:
                        return
                    fault = self.fault
                if fault is None:
                    observed = self.poll()
                    if observed:
                        self.report(*observed)
                    with self.condition:
                        self.condition.wait(timeout=self.interval)
                    continue
                self.environment.cancel_active()
                deadline = monotonic() + self.drain_timeout
                with self.condition:
                    while self.active and not self.closed:
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise RuntimeError('device session clients did not drain safely')
                        self.condition.wait(timeout=remaining)
                    if self.closed:
                        return
                if self.generation >= self.max_recoveries:
                    raise RuntimeError('device session recovery budget exhausted')
                self.stop()
                self.environment = None
                self.generation += 1
                self.environment = self.start(self.generation)
                self.probe()
                with self.condition:
                    self.fault = None
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=self.drain_timeout + 60)
            if self.thread.is_alive():
                raise RuntimeError('device session monitor failed to stop; ownership retained')
        if self.environment is not None:
            self.stop()
            self.environment = None
