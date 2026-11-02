from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
import time
from typing import Any, Mapping

from workflow_runtime.lifecycle.progress import ProbeUpdate, TrainingMetricTrend, TrainingSnapshot


_METRICS = (
    ("rmse_e", "E RMSE", "energy_rmse", "meV/atom"),
    ("mae_e", "E MAE", "energy_mae", "meV/atom"),
    ("rmse_f", "F RMSE", "force_rmse", "meV/angstrom"),
    ("mae_f", "F MAE", "force_mae", "meV/angstrom"),
    ("rmse_v", "virial RMSE", "virial_rmse", "meV"),
    ("mae_v", "virial MAE", "virial_mae", "meV"),
)


class DeepMDTrainingProgressProbe:
    """Parse one effective TF/PT DeePMD learning curve without fixed columns."""

    def __init__(
        self,
        *,
        curve_path: Path,
        total_steps: int,
        member_id: str,
        member_index: int,
        total_members: int,
        cli: str,
    ) -> None:
        if cli not in {"tf", "pt"}:
            raise ValueError("DeePMD progress requires cli='tf' or cli='pt'")
        if total_steps < 1:
            raise ValueError("DeePMD progress requires positive training.numb_steps")
        self.curve_path = Path(curve_path).resolve()
        self.total_steps = total_steps
        self.member_id = member_id
        self.member_index = member_index
        self.total_members = total_members
        self.cli = cli
        self._started = time.monotonic()
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._header: tuple[str, ...] | None = None
        self._halted = False
        self._generation = 0
        self._warning_emitted = False
        self._position: int | None = None
        self._learning_rate: float | None = None
        self._trends: dict[str, OrderedDict[int, float]] = {}
        self._last_token: str | None = None

    @classmethod
    def from_rendered_input(
        cls,
        rendered: Mapping[str, Any],
        *,
        member_dir: Path,
        member_id: str,
        member_index: int,
        total_members: int,
        cli: str,
    ) -> "DeepMDTrainingProgressProbe":
        training = rendered.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("DeePMD progress requires a rendered training object")
        total = training.get("numb_steps")
        if isinstance(total, bool) or not isinstance(total, int) or total < 1:
            raise ValueError("DeePMD progress requires positive integer training.numb_steps")
        raw_curve = training.get("disp_file", "lcurve.out")
        if not isinstance(raw_curve, str) or not raw_curve.strip():
            raise ValueError("DeePMD progress requires training.disp_file to be a path")
        curve = Path(raw_curve)
        if not curve.is_absolute():
            curve = Path(member_dir) / curve
        return cls(
            curve_path=curve,
            total_steps=total,
            member_id=member_id,
            member_index=member_index,
            total_members=total_members,
            cli=cli,
        )

    def poll(self) -> ProbeUpdate | None:
        try:
            stat = self.curve_path.stat()
        except FileNotFoundError:
            return None
        identity = (stat.st_dev, stat.st_ino)
        if self._identity is None:
            self._identity = identity
        elif identity != self._identity or stat.st_size < self._offset:
            self._reset(identity)
        if self._halted:
            return None

        warning: str | None = None
        with self.curve_path.open("rb") as handle:
            handle.seek(self._offset)
            while True:
                line_start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    handle.seek(line_start)
                    break
                self._offset = handle.tell()
                stripped = raw.decode("utf-8").strip()
                if not stripped:
                    continue
                try:
                    if self._header is None:
                        header = tuple(stripped.lstrip("#").split())
                        if not header or "step" not in header or len(set(header)) != len(header):
                            raise ValueError("first complete line is not a unique step header")
                        self._header = header
                        continue
                    if stripped.startswith("#") and self._position is None:
                        continue
                    if stripped.startswith("#"):
                        raise ValueError("unexpected interior header")
                    self._accept_values(stripped.split())
                except Exception as exc:
                    self._halted = True
                    if not self._warning_emitted:
                        self._warning_emitted = True
                        warning = (
                            f"DeePMD progress stopped at malformed learning-curve row byte {line_start}: {exc}"
                        )
                    break

        snapshot = self._snapshot()
        if snapshot is None:
            if warning is not None:
                raise ValueError(warning)
            return None
        token = _snapshot_token(snapshot)
        if token == self._last_token and warning is None:
            return None
        self._last_token = token
        return ProbeUpdate(semantic_token=token, snapshot=snapshot, warning=warning)

    def _accept_values(self, values: list[str]) -> None:
        assert self._header is not None
        if len(values) != len(self._header):
            raise ValueError("column count does not match the dynamic header")
        raw = dict(zip(self._header, values, strict=True))
        try:
            step_number = float(raw["step"])
        except (KeyError, ValueError) as exc:
            raise ValueError("step is not numeric") from exc
        if not math.isfinite(step_number) or step_number < 0 or not step_number.is_integer():
            raise ValueError("step must be a non-negative integer")
        step = min(int(step_number), self.total_steps)
        self._position = step
        lr = _number(raw.get("lr"))
        if lr is not None and lr >= 0:
            self._learning_rate = lr

        split = "validation" if any(f"{base}_val" in raw for base, *_ in _METRICS) else "train"
        suffix = "val" if split == "validation" else "trn"
        for base, short_label, quantity, unit in _METRICS:
            value = _number(raw.get(f"{base}_{suffix}"))
            if value is None:
                continue
            series_id = f"deepmd:g{self._generation}:{split}:{quantity}"
            points = self._trends.setdefault(series_id, OrderedDict())
            points[step] = value * 1000.0
            points.move_to_end(step)
            while len(points) > 4:
                points.popitem(last=False)

    def _snapshot(self) -> TrainingSnapshot | None:
        if self._position is None:
            return None
        validation = any(
            series_id.startswith(f"deepmd:g{self._generation}:validation:")
            for series_id in self._trends
        )
        split = "validation" if validation else "train"
        trends: list[TrainingMetricTrend] = []
        for _base, short_label, quantity, unit in _METRICS:
            series_id = f"deepmd:g{self._generation}:{split}:{quantity}"
            points = self._trends.get(series_id)
            if points:
                trends.append(
                    TrainingMetricTrend(
                        label=f"{split} {short_label}",
                        semantic_quantity=quantity,
                        split=split,
                        unit=unit,
                        values=tuple(points.values()),
                        series_id=series_id,
                    )
                )
        elapsed = time.monotonic() - self._started
        eta = None
        if self._position > 0 and self._position < self.total_steps:
            eta = elapsed / self._position * (self.total_steps - self._position)
        return TrainingSnapshot(
            member_id=self.member_id,
            member_index=self.member_index,
            total_members=self.total_members,
            unit="step",
            position=self._position,
            total=self.total_steps,
            stage="training",
            trends=tuple(trends),
            learning_rate=self._learning_rate,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            series_generation=self._generation,
        )

    def _reset(self, identity: tuple[int, int]) -> None:
        self._identity = identity
        self._offset = 0
        self._header = None
        self._halted = False
        self._generation += 1
        self._warning_emitted = False
        self._position = None
        self._learning_rate = None
        self._trends.clear()


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _snapshot_token(snapshot: TrainingSnapshot) -> str:
    return repr(
        (
            snapshot.series_generation,
            snapshot.position,
            snapshot.learning_rate,
            tuple((trend.series_id, trend.values) for trend in snapshot.trends),
        )
    )
