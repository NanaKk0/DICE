from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
import time

from workflow_runtime.lifecycle.progress import ProbeUpdate, TrainingMetricTrend, TrainingSnapshot


class NepTrainingProgressProbe:
    """Normalize GPUMD/NEP's headerless loss.out using effective nep.in truth."""

    def __init__(
        self,
        *,
        loss_path: Path,
        total_generations: int,
        model_type: int,
        charge_mode: int,
        member_id: str,
        member_index: int,
        total_members: int,
    ) -> None:
        self.loss_path = Path(loss_path).resolve()
        self.total_generations = total_generations
        self.model_type = model_type
        self.charge_mode = charge_mode
        self.member_id = member_id
        self.member_index = member_index
        self.total_members = total_members
        self._started = time.monotonic()
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._halted = False
        self._generation = 0
        self._warning_emitted = False
        self._position: int | None = None
        self._trends: dict[str, OrderedDict[int, float]] = {}
        self._last_token: str | None = None

    @classmethod
    def from_nep_input(
        cls,
        nep_input: Path,
        *,
        member_dir: Path,
        member_id: str,
        member_index: int,
        total_members: int,
    ) -> "NepTrainingProgressProbe":
        config = _parse_nep_input(Path(nep_input))
        total = _config_int(config, "generation", required=True)
        if total < 1:
            raise ValueError("NEP progress requires positive generation")
        model_type = _config_int(config, "model_type", alias="mode", default=0)
        if model_type not in {0, 1, 2, 3}:
            raise ValueError("NEP progress model_type must be 0, 1, 2, or 3")
        charge_mode = _config_int(config, "charge_mode", default=0)
        if charge_mode not in {0, 1, 2}:
            raise ValueError("NEP progress charge_mode must be 0, 1, or 2")
        return cls(
            loss_path=Path(member_dir) / "loss.out",
            total_generations=total,
            model_type=model_type,
            charge_mode=charge_mode,
            member_id=member_id,
            member_index=member_index,
            total_members=total_members,
        )

    def poll(self) -> ProbeUpdate | None:
        try:
            stat = self.loss_path.stat()
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
        with self.loss_path.open("rb") as handle:
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
                    self._accept_values(stripped.split())
                except Exception as exc:
                    self._halted = True
                    if not self._warning_emitted:
                        self._warning_emitted = True
                        warning = f"NEP progress stopped at malformed loss row byte {line_start}: {exc}"
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

    def _accept_values(self, raw_values: list[str]) -> None:
        layout = _layout(self.model_type, self.charge_mode)
        if len(raw_values) != layout.column_count:
            raise ValueError(
                f"expected {layout.column_count} columns for {layout.stage}, got {len(raw_values)}"
            )
        try:
            values = [float(value) for value in raw_values]
        except ValueError as exc:
            raise ValueError("loss row contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("loss row contains a non-finite value")
        position_value = values[0]
        if not position_value.is_integer():
            raise ValueError("generation must be an integer")
        position = int(position_value)
        if position < 0 or position > self.total_generations:
            raise ValueError("generation is outside the effective nep.in range")
        self._position = position
        for index, label, quantity, unit in layout.metrics:
            series_id = f"nep:g{self._generation}:{layout.stage}:{quantity}"
            points = self._trends.setdefault(series_id, OrderedDict())
            points[position] = values[index]
            points.move_to_end(position)
            while len(points) > 4:
                points.popitem(last=False)

    def _snapshot(self) -> TrainingSnapshot | None:
        if self._position is None:
            return None
        layout = _layout(self.model_type, self.charge_mode)
        trends: list[TrainingMetricTrend] = []
        for _index, label, quantity, unit in layout.metrics:
            series_id = f"nep:g{self._generation}:{layout.stage}:{quantity}"
            points = self._trends.get(series_id)
            if points:
                trends.append(
                    TrainingMetricTrend(
                        label=f"same-data evaluation {label}",
                        semantic_quantity=quantity,
                        split="same-data evaluation",
                        unit=unit,
                        values=tuple(points.values()),
                        series_id=series_id,
                    )
                )
        elapsed = time.monotonic() - self._started
        eta = None
        if self._position > 0 and self._position < self.total_generations:
            eta = elapsed / self._position * (self.total_generations - self._position)
        return TrainingSnapshot(
            member_id=self.member_id,
            member_index=self.member_index,
            total_members=self.total_members,
            unit="generation",
            position=self._position,
            total=self.total_generations,
            stage=layout.stage,
            trends=tuple(trends),
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            series_generation=self._generation,
        )

    def _reset(self, identity: tuple[int, int]) -> None:
        self._identity = identity
        self._offset = 0
        self._halted = False
        self._generation += 1
        self._warning_emitted = False
        self._position = None
        self._trends.clear()


class _Layout:
    def __init__(self, stage: str, column_count: int, metrics: tuple[tuple[int, str, str, str], ...]) -> None:
        self.stage = stage
        self.column_count = column_count
        self.metrics = metrics


def _layout(model_type: int, charge_mode: int) -> _Layout:
    if model_type in {1, 2}:
        label = "dipole RMSE" if model_type == 1 else "polarizability RMSE"
        return _Layout(
            "tensorial",
            6,
            ((5, label, "tensor_rmse", "native"),),
        )
    test_offset = 9 if charge_mode else 7
    stage = "qNEP" if charge_mode else "temperature-dependent" if model_type == 3 else "potential"
    return _Layout(
        stage,
        14 if charge_mode else 10,
        (
            (test_offset, "E RMSE", "energy_rmse", "eV/atom"),
            (test_offset + 1, "F RMSE", "force_rmse", "eV/angstrom"),
            (test_offset + 2, "virial RMSE", "virial_rmse", "eV/atom"),
        ),
    )


def _parse_nep_input(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"NEP progress cannot read effective nep.in: {path}") from exc
    config: dict[str, tuple[str, ...]] = {}
    for line in lines:
        values = line.split("#", 1)[0].split()
        if values:
            config[values[0]] = tuple(values[1:])
    return config


def _config_int(
    config: dict[str, tuple[str, ...]],
    key: str,
    *,
    alias: str | None = None,
    required: bool = False,
    default: int | None = None,
) -> int:
    values = config.get(key)
    if values is None and alias is not None:
        values = config.get(alias)
    if values is None:
        if required:
            raise ValueError(f"NEP progress requires {key} in effective nep.in")
        assert default is not None
        return default
    if len(values) != 1:
        raise ValueError(f"NEP progress {key} requires exactly one integer")
    try:
        return int(values[0])
    except ValueError as exc:
        raise ValueError(f"NEP progress {key} must be an integer") from exc


def _snapshot_token(snapshot: TrainingSnapshot) -> str:
    return repr(
        (
            snapshot.series_generation,
            snapshot.position,
            tuple((trend.series_id, trend.values) for trend in snapshot.trends),
        )
    )
