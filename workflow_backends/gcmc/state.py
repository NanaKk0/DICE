"""Lossless plain phase state; deliberately not a lifecycle completion marker."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from ase import Atoms


def _pack(value):
    if isinstance(value, Atoms):
        if value.constraints:
            raise ValueError('GCMC phase state cannot serialize constraints')
        return {'type': 'atoms', 'value': _pack(value.todict())}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'O':
            # ASE extxyz stores per-atom string labels as object arrays. Keep
            # their primitive values and shape without enabling pickle.
            return {'type': 'object-array', 'value': {
                'shape': list(value.shape), 'items': [_pack(item) for item in value.flat]}}
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {'type': 'array', 'value': base64.b64encode(stream.getvalue()).decode('ascii')}
    if isinstance(value, np.generic):
        return _pack(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('phase state keys must be strings')
        return {'type': 'mapping', 'value': {key: _pack(item) for key, item in value.items()}}
    if isinstance(value, (tuple, list)):
        return {'type': 'sequence', 'value': [_pack(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f'unsupported plain phase state: {type(value).__name__}')


def _unpack(value):
    if not isinstance(value, dict):
        return value
    kind, data = value['type'], value['value']
    if kind == 'mapping':
        return {key: _unpack(item) for key, item in data.items()}
    if kind == 'sequence':
        return [_unpack(item) for item in data]
    if kind == 'array':
        return np.load(io.BytesIO(base64.b64decode(data, validate=True)), allow_pickle=False)
    if kind == 'object-array':
        result = np.empty(tuple(data['shape']), dtype=object)
        if len(data['items']) != result.size:
            raise ValueError('object array shape does not match its primitive payload')
        for index, item in enumerate(data['items']):
            result.flat[index] = _unpack(item)
        return result
    if kind == 'atoms':
        return Atoms.fromdict(_unpack(data))
    raise ValueError('unsupported phase state encoding')


def state_identity(value):
    return hashlib.sha256(json.dumps(_pack(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_state(path, value):
    """Atomic, fsynced publication; never pickle evaluators or controller objects."""
    path = Path(path)
    payload = json.dumps(_pack(value), sort_keys=True, allow_nan=False)
    envelope = json.dumps({'schema': 'dice.gcmc.phase-state', 'revision': 1,
                           'sha256': hashlib.sha256(payload.encode()).hexdigest(), 'payload': payload})
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(envelope+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_state(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected regular phase state file')
    envelope = json.loads(path.read_text(encoding='utf-8'))
    if envelope.get('schema') != 'dice.gcmc.phase-state' or envelope.get('revision') != 1:
        raise ValueError('unsupported phase state schema')
    payload = envelope['payload']
    if hashlib.sha256(payload.encode()).hexdigest() != envelope['sha256']:
        raise ValueError('phase state checksum mismatch')
    return _unpack(json.loads(payload))
