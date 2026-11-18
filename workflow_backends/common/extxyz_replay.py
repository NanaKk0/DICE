"""Bounded geometry transport for validated canonical extxyz observations."""
from io import StringIO
from pathlib import Path

from ase.io import read

from workflow_backends.common.frame_index import index_extxyz_frames


class IndexedExtxyzReplay:
    """An invocation-local index, with no cached Atoms or completion authority."""

    def __init__(self, sample_dir, records):
        self._records = {row.structure_id: row for row in records}
        if len(self._records) != len(records):
            raise ValueError('extxyz observation identities collide')
        tasks = {}
        for row in records:
            tasks.setdefault(row.task_index, []).append(row)
        self._indices = {}
        for task_index, rows in tasks.items():
            path = Path(sample_dir)/'tasks'/f'task_{task_index:06d}'/'observations.extxyz'
            if [row.frame_index for row in rows] != list(range(len(rows))):
                raise ValueError('extxyz observation ordinals are not contiguous')
            self._indices[task_index] = index_extxyz_frames(path, expected_count=len(rows))

    def replay(self, identities, *, chunk_size):
        identities = tuple(identities)
        if (type(chunk_size) is not int or chunk_size <= 0
            or len(set(identities)) != len(identities)
            or any(identity not in self._records for identity in identities)):
            raise ValueError('extxyz replay requires unique known identities and positive chunk size')
        for start in range(0, len(identities), chunk_size):
            wanted, tasks, loaded = identities[start:start + chunk_size], {}, {}
            for identity in wanted:
                row = self._records[identity]
                tasks.setdefault(row.task_index, []).append(row)
            for task_index, rows in tasks.items():
                for row, raw in zip(rows, self._indices[task_index].read(row.frame_index for row in rows), strict=True):
                    frame = read(StringIO(raw.decode('utf-8')), format='extxyz')
                    if frame.info.get('structure_id') != row.structure_id:
                        raise ValueError('extxyz replay identity/order mismatch')
                    loaded[row.structure_id] = frame
            yield tuple(loaded[identity] for identity in wanted)
