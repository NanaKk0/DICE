from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from workflow_runtime.lifecycle.progress import SamplingSnapshot
from workflow_runtime.lifecycle.progress.operation import ProbeUpdate


class LammpsSamplingProgressProbe:
    """Read only complete rows from the latest usable LAMMPS thermo section."""

    def __init__(self, path: str | Path, *, base: SamplingSnapshot) -> None:
        self.path = Path(path)
        self.base = base
        self._last_identity: tuple[int, int] | None = None
        self._generation = 0
        self._last_token: str | None = None

    def poll(self) -> ProbeUpdate | None:
        try:
            stat = self.path.stat()
            identity = (int(stat.st_ino), int(stat.st_size))
            if self._last_identity is not None and (
                identity[0] != self._last_identity[0] or identity[1] < self._last_identity[1]
            ):
                self._generation += 1
            self._last_identity = identity
            step = _latest_complete_thermo_step(self.path)
        except OSError:
            return None
        if step is None:
            return None
        token = f"{self._generation}:{step}"
        if token == self._last_token:
            return None
        self._last_token = token
        return ProbeUpdate(token, replace(self.base, active_step=step))


def _latest_complete_thermo_step(path: Path) -> int | None:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    complete = [line.decode("utf-8", errors="replace").strip() for line in lines if line.endswith((b"\n", b"\r"))]
    headers = [index for index, line in enumerate(complete) if line.split()[:1] == ["Step"]]
    for header_index in reversed(headers):
        width = len(complete[header_index].split())
        if width < 2:
            continue
        latest: int | None = None
        for line in complete[header_index + 1 :]:
            fields = line.split()
            if fields[:1] == ["Step"]:
                break
            if len(fields) != width:
                if latest is not None:
                    break
                continue
            try:
                values = tuple(float(value) for value in fields)
            except ValueError:
                if latest is not None:
                    break
                continue
            step = values[0]
            if not step.is_integer() or step < 0:
                if latest is not None:
                    break
                continue
            latest = int(step)
        if latest is not None:
            return latest
    return None


__all__ = ["LammpsSamplingProgressProbe"]
