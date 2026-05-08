from __future__ import annotations

import csv
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


@dataclass
class ResourceSummary:
    gpu_time_sec: float | None = None
    avg_gpu_util: float | None = None
    max_gpu_util: float | None = None
    max_gpu_mem_mb: float | None = None
    avg_cpu_percent: float | None = None
    max_cpu_percent: float | None = None
    max_ram_mb: float | None = None


class ResourceMonitor:
    def __init__(
        self,
        callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        interval_sec: float = 5.0,
        run_name: str | None = None,
    ) -> None:
        self.callback = callback
        self.interval_sec = interval_sec
        self.run_name = run_name
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rows: list[dict[str, Any]] = []
        self._step_id: str | None = None
        self._round_idx: int | None = None
        self._pid: int | None = None
        self._last_disk = None
        self._last_disk_ts = None

    def _gpu_snapshot(self) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return {}
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [chunk.strip() for chunk in first_line.split(",")]
            if len(parts) < 4:
                return {}
            return {
                "gpu_util_percent": _safe_float(parts[0]),
                "gpu_mem_mb": _safe_float(parts[1]),
                "gpu_power_w": _safe_float(parts[2]),
                "gpu_temp_c": _safe_float(parts[3]),
            }
        except Exception:
            return {}

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        row: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "run_name": self.run_name,
            "round_idx": self._round_idx,
            "step_id": self._step_id,
            "process_pid": self._pid,
            "cpu_percent": None,
            "ram_mb": None,
            "ram_percent": None,
            "gpu_util_percent": None,
            "gpu_mem_mb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
            "disk_read_mb_s": None,
            "disk_write_mb_s": None,
        }
        if psutil is not None:
            try:
                proc = psutil.Process(self._pid) if self._pid else psutil.Process()
                mem = proc.memory_info()
                row["cpu_percent"] = proc.cpu_percent(interval=None)
                row["ram_mb"] = round(mem.rss / (1024 * 1024), 6)
                row["ram_percent"] = proc.memory_percent()
            except Exception:
                pass
            try:
                disk = psutil.disk_io_counters()
                if disk is not None and self._last_disk is not None and self._last_disk_ts is not None:
                    elapsed = max(now - self._last_disk_ts, 1e-6)
                    row["disk_read_mb_s"] = round((disk.read_bytes - self._last_disk.read_bytes) / (1024 * 1024) / elapsed, 6)
                    row["disk_write_mb_s"] = round((disk.write_bytes - self._last_disk.write_bytes) / (1024 * 1024) / elapsed, 6)
                self._last_disk = disk
                self._last_disk_ts = now
            except Exception:
                pass
        row.update(self._gpu_snapshot())
        return row

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            row = self.snapshot()
            self._rows.append(row)
            if self.callback is not None:
                try:
                    self.callback(row)
                except Exception:
                    pass

    def start(self, step_id: str | None = None, round_idx: int | None = None, pid: int | None = None) -> None:
        self.stop()
        self._stop_event.clear()
        self._rows = []
        self._step_id = step_id
        self._round_idx = round_idx
        self._pid = pid
        self._last_disk = None
        self._last_disk_ts = None
        initial = self.snapshot()
        self._rows.append(initial)
        if self.callback is not None:
            try:
                self.callback(initial)
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, name=f"resource-monitor-{step_id or 'run'}", daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._rows:
            final_row = self.snapshot()
            self._rows.append(final_row)
            if self.callback is not None:
                try:
                    self.callback(final_row)
                except Exception:
                    pass
        return self._summarize()

    def _summarize(self) -> ResourceSummary:
        if not self._rows:
            return ResourceSummary()
        gpu_utils = [_safe_float(row.get("gpu_util_percent")) for row in self._rows if _safe_float(row.get("gpu_util_percent")) is not None]
        gpu_mems = [_safe_float(row.get("gpu_mem_mb")) for row in self._rows if _safe_float(row.get("gpu_mem_mb")) is not None]
        cpu_vals = [_safe_float(row.get("cpu_percent")) for row in self._rows if _safe_float(row.get("cpu_percent")) is not None]
        ram_vals = [_safe_float(row.get("ram_mb")) for row in self._rows if _safe_float(row.get("ram_mb")) is not None]
        gpu_time_sec = None
        if gpu_utils:
            gpu_time_sec = round(sum(self.interval_sec for _ in gpu_utils) / 100.0 * sum(gpu_utils) / len(gpu_utils), 6)
        return ResourceSummary(
            gpu_time_sec=gpu_time_sec,
            avg_gpu_util=round(sum(gpu_utils) / len(gpu_utils), 6) if gpu_utils else None,
            max_gpu_util=max(gpu_utils) if gpu_utils else None,
            max_gpu_mem_mb=max(gpu_mems) if gpu_mems else None,
            avg_cpu_percent=round(sum(cpu_vals) / len(cpu_vals), 6) if cpu_vals else None,
            max_cpu_percent=max(cpu_vals) if cpu_vals else None,
            max_ram_mb=max(ram_vals) if ram_vals else None,
        )
