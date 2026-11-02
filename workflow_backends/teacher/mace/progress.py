from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from workflow_runtime.lifecycle.progress import ProbeUpdate, TrainingMetricTrend, TrainingSnapshot


class MaceTrainingProgressProbe:
    """Normalize MACE's append-only training metrics JSONL into common snapshots."""

    def __init__(
        self,
        *,
        metrics_path: Path,
        member_id: str,
        member_index: int,
        total_members: int,
        total_epochs: int,
        validation_head: str | None,
        error_table: str,
        swa: bool,
        start_swa: int | None,
    ) -> None:
        if total_epochs < 1:
            raise ValueError("MACE progress requires positive max_num_epochs")
        self.metrics_path = Path(metrics_path)
        self.member_id = member_id
        self.member_index = member_index
        self.total_members = total_members
        self.total_epochs = total_epochs
        self.validation_head = validation_head
        self.error_table = error_table
        self.swa = swa
        self.start_swa = start_swa if swa else None
        self._started = time.monotonic()
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._halted = False
        self._generation = 0
        self._warning_emitted = False
        self._position = 0
        self._last_token: str | None = None
        self._trends: dict[str, OrderedDict[int, float]] = {}
        self._epoch_times: dict[int, float] = {}
        self._evaluated_epochs: set[int] = set()

    @classmethod
    def from_rendered_recipe(
        cls,
        rendered: Mapping[str, Any],
        *,
        work_dir: Path,
        member_id: str,
        member_index: int,
        total_members: int,
    ) -> "MaceTrainingProgressProbe":
        name = rendered.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MACE progress requires the resolved non-empty name")
        total_epochs = _positive_int(rendered.get("max_num_epochs"), "max_num_epochs")
        metrics_path = _resolved_metrics_path(rendered, cwd=Path(work_dir), name=name)
        swa = rendered.get("swa", False)
        if not isinstance(swa, bool):
            raise ValueError("MACE progress requires rendered swa to be boolean")
        raw_start_swa = rendered.get("start_swa")
        start_swa = None if raw_start_swa is None else _nonnegative_int(raw_start_swa, "start_swa")
        if swa and start_swa is None:
            # Match MACE's resolved native default in get_swa().
            start_swa = max(1, total_epochs // 4 * 3)
        return cls(
            metrics_path=metrics_path,
            member_id=member_id,
            member_index=member_index,
            total_members=total_members,
            total_epochs=total_epochs,
            validation_head=_validation_head(rendered, cwd=Path(work_dir)),
            error_table=str(rendered.get("error_table", "PerAtomRMSE")),
            swa=swa,
            start_swa=start_swa,
        )

    def poll(self) -> ProbeUpdate | None:
        try:
            stat = self.metrics_path.stat()
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
        with self.metrics_path.open("rb") as handle:
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
                try:
                    row = json.loads(raw.decode("utf-8"))
                    if not isinstance(row, dict):
                        raise ValueError("row is not a JSON object")
                    self._accept_row(row)
                except Exception as exc:
                    self._halted = True
                    if not self._warning_emitted:
                        self._warning_emitted = True
                        warning = (
                            f"MACE progress stopped at malformed metrics row byte {line_start}: {exc}"
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

    def _accept_row(self, row: Mapping[str, Any]) -> None:
        mode = row.get("mode")
        epoch = row.get("epoch")
        if mode not in {"opt", "eval"} or epoch is None:
            return
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer or null")
        if mode == "opt":
            self._position = max(self._position, min(epoch + 1, self.total_epochs))
            duration = row.get("time")
            if _finite_number(duration):
                self._epoch_times[epoch] = self._epoch_times.get(epoch, 0.0) + float(duration)
            return
        if self.validation_head is None or row.get("head") != self.validation_head:
            return
        self._position = max(self._position, min(epoch + 1, self.total_epochs))
        self._evaluated_epochs.add(epoch)
        stage = self._stage(epoch)
        for field, label, quantity, unit in _metric_specs(self.error_table):
            value = row.get(field)
            if not _finite_number(value):
                continue
            series_id = (
                f"mace:g{self._generation}:{stage}:{self.validation_head}:{quantity}"
            )
            points = self._trends.setdefault(series_id, OrderedDict())
            points[epoch] = float(value) * 1000.0
            points.move_to_end(epoch)
            while len(points) > 4:
                points.popitem(last=False)

    def _snapshot(self) -> TrainingSnapshot | None:
        if self._position < 1:
            return None
        epoch = self._position - 1
        stage = self._stage(epoch)
        trends: list[TrainingMetricTrend] = []
        for _field, label, quantity, unit in _metric_specs(self.error_table):
            series_id = f"mace:g{self._generation}:{stage}:{self.validation_head}:{quantity}"
            points = self._trends.get(series_id)
            if points:
                trends.append(
                    TrainingMetricTrend(
                        label=label,
                        semantic_quantity=quantity,
                        split="validation",
                        unit=unit,
                        values=tuple(points.values()),
                        series_id=series_id,
                    )
                )
        return TrainingSnapshot(
            member_id=self.member_id,
            member_index=self.member_index,
            total_members=self.total_members,
            unit="epoch",
            position=self._position,
            total=self.total_epochs,
            stage=stage,
            trends=tuple(trends),
            elapsed_seconds=time.monotonic() - self._started,
            eta_seconds=self._eta(stage),
            series_generation=self._generation,
        )

    def _stage(self, epoch: int) -> str:
        if self.start_swa is not None and epoch >= self.start_swa:
            return "stage-two/SWA"
        return "stage-one"

    def _eta(self, stage: str) -> float | None:
        completed = [
            self._epoch_times[epoch]
            for epoch in sorted(self._evaluated_epochs)
            if self._stage(epoch) == stage and self._epoch_times.get(epoch, 0.0) > 0.0
        ]
        if not completed:
            return None
        return (sum(completed) / len(completed)) * (self.total_epochs - self._position)

    def _reset(self, identity: tuple[int, int]) -> None:
        self._identity = identity
        self._offset = 0
        self._halted = False
        self._generation += 1
        self._warning_emitted = False
        self._position = 0
        self._trends.clear()
        self._epoch_times.clear()
        self._evaluated_epochs.clear()


def _resolved_metrics_path(rendered: Mapping[str, Any], *, cwd: Path, name: str) -> Path:
    raw_results = rendered.get("results_dir")
    if raw_results is None:
        raw_work = rendered.get("work_dir", ".")
        if not isinstance(raw_work, str) or not raw_work.strip():
            raise ValueError("MACE progress requires rendered work_dir to be a path")
        native_work = Path(raw_work)
        if not native_work.is_absolute():
            native_work = cwd / native_work
        results = native_work / "results"
    else:
        if not isinstance(raw_results, str) or not raw_results.strip():
            raise ValueError("MACE progress requires rendered results_dir to be a path")
        results = Path(raw_results)
        if not results.is_absolute():
            results = cwd / results
    return (results / f"{name}_train.txt").resolve()


def _validation_head(rendered: Mapping[str, Any], *, cwd: Path) -> str | None:
    heads = rendered.get("heads")
    if heads is None:
        return "Default"
    if not isinstance(heads, Mapping) or not heads:
        return None
    names = tuple(str(name) for name in heads)
    if len(names) == 1:
        return names[0]
    injected = rendered.get("valid_file")
    if not isinstance(injected, str) or not injected:
        return None
    injected_path = _resolve_user_path(injected, cwd)
    matches: list[str] = []
    for name, config in heads.items():
        if not isinstance(config, Mapping):
            continue
        values = config.get("valid_file")
        candidates = values if isinstance(values, (list, tuple)) else (values,)
        if any(
            isinstance(value, str) and _resolve_user_path(value, cwd) == injected_path
            for value in candidates
        ):
            matches.append(str(name))
    return matches[0] if len(matches) == 1 else None


def _resolve_user_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def _metric_specs(error_table: str) -> tuple[tuple[str, str, str, str], ...]:
    mae = "MAE" in error_table
    total = error_table.startswith("Total")
    prefix = "mae" if mae else "rmse"
    statistic = "MAE" if mae else "RMSE"
    energy_field = f"{prefix}_e" if total else f"{prefix}_e_per_atom"
    energy_unit = "meV" if total else "meV/atom"
    specs: list[tuple[str, str, str, str]] = [
        (energy_field, f"validation E {statistic}", f"energy_{prefix}", energy_unit),
        (f"{prefix}_f", f"validation F {statistic}", f"force_{prefix}", "meV/angstrom"),
    ]
    if "stressvirials" in error_table:
        specs.extend(
            (
                (f"{prefix}_stress", f"validation stress {statistic}", f"stress_{prefix}", "meV/angstrom^3"),
                (f"{prefix}_virials_per_atom", f"validation virial {statistic}", f"virial_{prefix}", "meV/atom"),
            )
        )
    return tuple(specs)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result < 1:
        raise ValueError(f"MACE progress requires positive {label}")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"MACE progress requires non-negative integer {label}")
    return value


def _snapshot_token(snapshot: TrainingSnapshot) -> str:
    trend_token = tuple((trend.series_id, trend.values) for trend in snapshot.trends)
    return repr((snapshot.series_generation, snapshot.position, snapshot.stage, trend_token))
