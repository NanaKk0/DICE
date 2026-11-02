"""Exec-based workers; adapters own initialization and native task semantics.

The dedicated socket keeps native stdout/stderr out of the bounded protocol.
Commands are explicitly composed by backend owners, never loaded from config.
"""
import json
import socket
import subprocess
from pathlib import Path
from threading import Lock


LIMIT = 65536


class WorkerLost(RuntimeError):
    pass


class TaskExecutionError(RuntimeError):
    pass


def _send(channel, message):
    data = json.dumps(message, allow_nan=False).encode() + b'\n'
    if len(data) > LIMIT:
        raise ValueError('worker message exceeds protocol bound')
    channel.write(data)
    channel.flush()


def _receive(channel):
    data = channel.readline(LIMIT + 1)
    if not data:
        raise WorkerLost('worker exited before its terminal reply')
    if len(data) > LIMIT or not data.endswith(b'\n'):
        raise ValueError('worker sent an invalid bounded reply')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('worker reply must be an object')
    return value


class PersistentWorker:
    def __init__(self, command, *, environment, cwd, log, registry, startup_timeout=120,
                 expected_identity=None):
        self.registry = registry
        self.identity = expected_identity
        self.socket, child = socket.socketpair()
        self.channel = self.socket.makefile('rwb')
        self.process = None
        try:
            with open(log, 'ab') as diagnostic:
                self.process = subprocess.Popen(
                    [*command, str(child.fileno())], cwd=cwd, env=environment,
                    pass_fds=(child.fileno(),), start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=diagnostic, stderr=diagnostic,
                )
            registry.register(self.process)
            child.close()
            self.socket.settimeout(startup_timeout)
            if _receive(self.channel) != {'ready': 1, 'identity': self.identity}:
                raise ValueError('persistent worker startup identity mismatch')
            self.socket.settimeout(None)
        except BaseException:
            child.close()
            self.close()
            raise

    def execute(self, task_id):
        try:
            _send(self.channel, {'task_id': task_id, 'identity': self.identity})
            result = _receive(self.channel)
        except OSError as error:
            raise WorkerLost('worker connection lost') from error
        if result.get('task_id') != task_id:
            raise ValueError('worker terminal identity does not match claimed task')
        if result.pop('identity', object()) != self.identity:
            raise ValueError('worker terminal model identity mismatch')
        if 'error' in result:
            raise TaskExecutionError(result['error'])
        return result

    def close(self):
        process = self.process
        try:
            if process is not None and process.poll() is None:
                try:
                    _send(self.channel, {'stop': True})
                    process.wait(timeout=5)
                except (OSError, RuntimeError, subprocess.TimeoutExpired):
                    # Cancel just this generation, never peers or foreign clients.
                    from workflow_backends.execution import ProcessGroupRegistry
                    owned = ProcessGroupRegistry(grace_seconds=2)
                    owned.register(process)
                    owned.cancel_all()
            if process is not None:
                self.registry.unregister(process)
        finally:
            try:
                self.channel.close()
            except OSError:
                pass  # A dead child cannot consume buffered protocol bytes.
            finally:
                self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def serve(fd, create_adapter):
    with socket.socket(fileno=fd) as connection, connection.makefile('rwb') as channel:
        adapter = create_adapter()
        identity = getattr(adapter, 'identity', None)
        try:
            _send(channel, {'ready': 1, 'identity': identity})
            while True:
                message = _receive(channel)
                if message == {'stop': True}:
                    return
                if (set(message) != {'task_id', 'identity'} or message['identity'] != identity
                        or not isinstance(message['task_id'], str)):
                    raise ValueError('invalid task claim')
                task_id = message['task_id']
                try:
                    result = adapter.execute(task_id)
                except Exception as error:
                    result = {'task_id': task_id, 'error': f'{type(error).__name__}: {error}'[-4096:]}
                _send(channel, dict(result, identity=identity))
        finally:
            release = getattr(adapter, 'close', None)
            if release is not None:
                release()


class PersistentWorkers:
    """One process per allocation slot and session generation; dynamic refill."""
    def __init__(self, command, *, environment, cwd, expected_identity=None):
        self.command, self.environment, self.cwd = command, environment, cwd
        self.identity = expected_identity
        self.workers = {}
        self.lock = Lock()

    def execute(self, task_id):
        from workflow_backends.execution import (
            current_operation_environment_overrides, current_operation_process_registry,
        )
        overlay = current_operation_environment_overrides()
        key = (overlay['DICE_GPU_SLOT'], overlay.get('CUDA_MPS_PIPE_DIRECTORY', ''))
        # Only this slot uses this worker; unrelated slots initialize concurrently.
        worker = self.workers.get(key)
        if worker is None:
            log_dir = Path(self.cwd) / 'backend_logs' / 'workers'
            log_dir.mkdir(parents=True, exist_ok=True)
            registry = current_operation_process_registry()
            if registry is None:
                raise ValueError('persistent workers require owned allocation scope')
            worker = PersistentWorker(self.command,
                environment={**self.environment, **overlay}, cwd=self.cwd,
                log=log_dir / (key[0].replace(':', '_') + '.log'), registry=registry,
                expected_identity=self.identity)
            with self.lock:
                self.workers[key] = worker
        try:
            return worker.execute(task_id)
        except (WorkerLost, TaskExecutionError):
            worker.close()
            with self.lock:
                del self.workers[key]
            raise

    def close(self):
        for worker in self.workers.values():
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
