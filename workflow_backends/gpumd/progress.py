from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import re

from workflow_runtime.lifecycle.progress import SamplingSnapshot
from workflow_runtime.lifecycle.progress.operation import ProbeUpdate


_MILESTONE = re.compile(r"^\s*(\d+)\s+steps completed\.\s*$")
_SPEED = re.compile(r"^\s*Speed of this run =\s*([0-9.eE+-]+)\s+atom\*step/second\.\s*$")


class GpumdSamplingProgressProbe:
    """Project flushed GPUMD milestones and optional user-configured thermo rows."""

    def __init__(
        self,
        stdout_path: str | Path,
        *,
        thermo_path: str | Path,
        thermo_cadence: int | None,
        base: SamplingSnapshot,
    ) -> None:
        self.stdout_path = Path(stdout_path)
        self.thermo_path = Path(thermo_path)
        self.thermo_cadence = thermo_cadence
        self.base = base
        self._last_sizes: dict[Path, tuple[int, int]] = {}
        self._generation = 0
        self._last_token: str | None = None

    def poll(self) -> ProbeUpdate | None:
        reset = False
        for path in (self.stdout_path, self.thermo_path):
            try:
                stat = path.stat()
            except OSError:
                continue
            current = (int(stat.st_ino), int(stat.st_size))
            previous = self._last_sizes.get(path)
            if previous is not None and (current[0] != previous[0] or current[1] < previous[1]):
                reset = True
            self._last_sizes[path] = current
        if reset:
            self._generation += 1

        stdout_step, speed = _stdout_progress(self.stdout_path)
        thermo_step = _thermo_progress(self.thermo_path, cadence=self.thermo_cadence)
        candidates = [step for step in (stdout_step, thermo_step) if step is not None]
        if not candidates:
            return None
        step = min(max(candidates), self.base.steps_per_task)
        token = f"{self._generation}:{step}:{speed}"
        if token == self._last_token:
            return None
        self._last_token = token
        snapshot = replace(self.base, active_step=step)
        if speed is not None:
            snapshot = replace(snapshot, speed=speed, speed_unit="atom·step/s")
        return ProbeUpdate(token, snapshot)


def _stdout_progress(path: Path) -> tuple[int | None, float | None]:
    try:
        lines = _complete_lines(path)
    except OSError:
        return None, None
    step = None
    speed = None
    for line in lines:
        milestone = _MILESTONE.match(line)
        if milestone:
            step = int(milestone.group(1))
        speed_match = _SPEED.match(line)
        if speed_match:
            value = float(speed_match.group(1))
            if math.isfinite(value) and value > 0:
                speed = value
    return step, speed


def _thermo_progress(path: Path, *, cadence: int | None) -> int | None:
    if cadence is None or cadence <= 0:
        return None
    try:
        lines = _complete_lines(path)
    except OSError:
        return None
    count = 0
    for line in lines:
        fields = line.split()
        if len(fields) != 18:
            break
        try:
            values = tuple(float(value) for value in fields)
        except ValueError:
            break
        if any(not math.isfinite(value) for value in values):
            break
        count += 1
    return None if count == 0 else count * cadence


def _complete_lines(path: Path) -> tuple[str, ...]:
    raw = path.read_bytes()
    return tuple(
        line.decode("utf-8", errors="replace").strip()
        for line in raw.splitlines(keepends=True)
        if line.endswith((b"\n", b"\r"))
    )


__all__ = ["GpumdSamplingProgressProbe"]
