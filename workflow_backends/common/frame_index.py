"""Invocation-local byte locations in an already validated native snapshot."""
from dataclasses import dataclass
import os
from pathlib import Path


def file_snapshot(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class FrameByteIndex:
    path: Path
    snapshot: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]

    def read(self, ordinals):
        """Read bounded frame bytes; an index never authorizes changed content."""
        if file_snapshot(self.path) != self.snapshot:
            raise ValueError('native observation changed after validation; reopen sample')
        with self.path.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != self.snapshot:
                raise ValueError('native observation changed while opening sample')
            for ordinal in ordinals:
                start, end = self.spans[ordinal]
                stream.seek(start)
                raw = stream.read(end - start)
                if len(raw) != end - start:
                    raise ValueError('native observation frame was truncated')
                yield raw
        if file_snapshot(self.path) != self.snapshot:
            raise ValueError('native observation changed during replay')


def index_extxyz_frames(path, *, expected_count=None):
    """Locate complete text frames without materializing their atom arrays."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('extxyz replay requires regular evidence')
    stamp, spans = file_snapshot(path), []
    with path.open('rb') as stream:
        while True:
            start, header = stream.tell(), stream.readline()
            if not header:
                break
            try:
                count = int(header)
            except ValueError as exc:
                raise ValueError('extxyz replay atom count is invalid') from exc
            if count <= 0:
                raise ValueError('extxyz replay atom count is invalid')
            for _ in range(count + 1):
                if not stream.readline().endswith(b'\n'):
                    raise ValueError('extxyz replay frame is truncated')
            spans.append((start, stream.tell()))
    if expected_count is not None and len(spans) != expected_count:
        raise ValueError('extxyz replay observation count is inconsistent')
    if file_snapshot(path) != stamp:
        raise ValueError('extxyz observation changed during indexing')
    return FrameByteIndex(path, stamp, tuple(spans))
