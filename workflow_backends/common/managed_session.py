"""Bounded recovery for explicitly owned MPS sessions; never GPU-reset."""
from contextlib import contextmanager
import json
import subprocess
from time import time

from workflow_backends.common.device_fault import classify_device_fault
from workflow_backends.common.device_session import DeviceSessionSupervisor


@contextmanager
def managed_device_session(context, policy, root, device, index, *, session_factory, probe_callback):
    session = None
    offsets = {}
    diagnostic = root / 'backend_logs' / 'mps' / f'gpu_{index:03d}'
    diagnostic.mkdir(parents=True, exist_ok=True)

    def event(name, **fields):
        with (diagnostic / 'recovery.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(time=time(), event=name, device=device, **fields)) + '\n')

    def start(generation):
        nonlocal session
        offsets.clear()
        session = session_factory(context, policy, root, device=device,
            diagnostic_namespace=f'gpu_{index:03d}/generation_{generation:03d}', preflight=False)
        environment = session.__enter__()
        event('started', generation=generation, pipe=environment['CUDA_MPS_PIPE_DIRECTORY'])
        return environment

    def stop():
        if session is not None:
            session.__exit__(None, None, None)
            event('stopped')

    def poll():
        # Read only new bounded data, retaining overlap for split messages.
        directory = session._transient_root
        if directory is None:
            return None
        for path in (directory / 'logs').glob('*.log'):
            offset = offsets.get(path, 0)
            size = path.stat().st_size
            if size < offset:
                offset = 0
            with path.open('rb') as handle:
                handle.seek(offset)
                data = handle.read(256 * 1024)
            offsets[path] = offset + max(0, len(data) - 128)
            detail = data.decode('utf-8', errors='replace')
            kind = classify_device_fault(detail)
            if kind:
                return kind, detail[-4096:]
        return None

    def probe():
        environment = dict(session._environment)
        environment['CUDA_VISIBLE_DEVICES'] = '0'
        result = probe_callback(context, environment)
        event('probe', returncode=result.returncode, stderr=result.stderr[-4096:])
        if result.returncode:
            raise RuntimeError('recovered MPS CUDA health probe failed')

    supervisor = DeviceSessionSupervisor(start=start, stop=stop, poll=poll, probe=probe,
        on_fault=lambda generation, kind, detail: event('fault', generation=generation,
            kind=kind, detail=detail[-4096:]))
    supervisor.open()
    try:
        yield supervisor
    finally:
        try:
            supervisor.close()
        finally:
            event('closed', generation=supervisor.generation, error=str(supervisor.error or ''))
        if supervisor.error is not None:
            raise RuntimeError(f'MPS device {device} quarantined: {supervisor.error}') from supervisor.error
