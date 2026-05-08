from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread

from workflow_backends.lammps.task_layout import SamplingTask, SamplingTaskResult


class SamplingProgressLogger:
    def __init__(self, log_path: str | Path | None, *, poll_interval_sec: float = 2.0) -> None:
        self.log_path = Path(log_path).resolve() if log_path else None
        self.poll_interval_sec = max(0.5, float(poll_interval_sec))
        self.total_tasks = 0
        self._finished_count = 0
        self._last_logged_finished_count = -1
        self._lock = Lock()
        self._poll_stop = Event()
        self._poll_thread: Thread | None = None

    def start(
        self,
        *,
        total_tasks: int,
        backend: str,
        max_parallel_tasks: int,
        max_parallel_groups: int,
        group_parallel_degree: int,
    ) -> None:
        self.total_tasks = max(0, int(total_tasks))
        self._finished_count = 0
        self._last_logged_finished_count = -1
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        header = (
            f"sampling started: total_tasks={self.total_tasks} backend={backend} "
            f"max_parallel_tasks={int(max_parallel_tasks)} "
            f"max_parallel_groups={int(max_parallel_groups)} "
            f"group_parallel_degree={int(group_parallel_degree)}"
        )
        self._write_line(header)
        self.record_finished_count(0, force=True)

    def record_result(self, result: SamplingTaskResult) -> None:
        with self._lock:
            self._finished_count += 1
            finished_count = self._finished_count
        self.record_finished_count(finished_count)

    def record_finished_count(self, finished_count: int, *, force: bool = False) -> None:
        finished_count = max(0, min(int(finished_count), int(self.total_tasks)))
        with self._lock:
            if not force and finished_count == self._last_logged_finished_count:
                return
            self._finished_count = finished_count
            self._last_logged_finished_count = finished_count
        self._write_progress_line(finished_count)

    def start_task_polling(self, tasks: list[SamplingTask], artifact_paths_fn) -> None:
        if self.log_path is None or not tasks:
            return
        self.stop_task_polling()
        self._poll_stop.clear()

        def _worker() -> None:
            while True:
                finished_count = 0
                for task in tasks:
                    task_paths = artifact_paths_fn(task)
                    if Path(task_paths["end_marker"]).exists():
                        finished_count += 1
                self.record_finished_count(finished_count)
                if finished_count >= self.total_tasks:
                    return
                if self._poll_stop.wait(self.poll_interval_sec):
                    return

        self._poll_thread = Thread(target=_worker, name="sampling-progress-poll", daemon=True)
        self._poll_thread.start()

    def stop_task_polling(self) -> None:
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=max(1.0, self.poll_interval_sec + 1.0))
            self._poll_thread = None
        self._poll_stop.clear()

    def finalize(self, task_results: list[SamplingTaskResult]) -> None:
        self.stop_task_polling()
        self.record_finished_count(len(task_results), force=True)
        if self.log_path is None:
            return
        finished_ok = sum(1 for item in task_results if item.status == "finished")
        failed_count = max(0, len(task_results) - finished_ok)
        self._write_line(
            f"sampling tasks completed: finished={len(task_results)}/{self.total_tasks} "
            f"successful={finished_ok} failed={failed_count}"
        )

    def _write_progress_line(self, finished_count: int) -> None:
        if self.log_path is None:
            return
        percent = 100.0 if self.total_tasks <= 0 else (100.0 * float(finished_count) / float(self.total_tasks))
        bar = self._render_bar(finished_count)
        self._write_line(f"{bar} {finished_count}/{self.total_tasks} tasks finished ({percent:.1f}%)")

    def _render_bar(self, finished_count: int, width: int = 20) -> str:
        if self.total_tasks <= 0:
            return "[" + "-" * width + "]"
        filled = min(width, max(0, int(round(width * float(finished_count) / float(self.total_tasks)))))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _write_line(self, message: str) -> None:
        if self.log_path is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
