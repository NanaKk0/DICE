"""Bounded intrinsic file digests with a validation-pass-local snapshot cache."""
import hashlib
from pathlib import Path

from workflow_backends.common.frame_index import file_snapshot


def file_digest(path, *, cache=None):
    path = Path(path)
    stamp = file_snapshot(path)
    if cache is not None and stamp in cache:
        return cache[stamp]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if file_snapshot(path) != stamp:
        raise ValueError('native evidence changed during integrity validation')
    value = digest.hexdigest()
    if cache is not None:
        cache[stamp] = value
    return value
