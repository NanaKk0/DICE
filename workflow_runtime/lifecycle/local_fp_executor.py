from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time
from typing import Callable, Sequence

from workflow_runtime.lifecycle.environment import OperationContext


_TASK_ID = re.compile(r"^task_[0-9]{6}$")
FP_LOG_LIMIT_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FPLocalTaskCommand:
    task_id: str
    workspace: Path
    input_digest: str
    logical_executable: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FPLocalCompletedTask:
    task_id: str
    workspace: Path
    input_digest: str
    command_digest: str
    returncode: int
    stdout_path: Path
    stderr_path: Path


class FPProcessIndeterminateError(RuntimeError):
    pass


class LocalFPExecutor:
    """Run opaque calculator task commands in one bounded same-host CPU pool."""

    def __init__(
        self,
        operation_context: OperationContext,
        *,
        termination_grace_seconds: float = 1.0,
        fault_hook: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        if not operation_context.role.startswith("first_principles."):
            raise ValueError("local FP executor requires a first-principles operation role")
        self.operation_context = operation_context
        self.fp_workers = _positive_runtime_int(operation_context, "fp_workers")
        self.cpu_threads = _positive_runtime_int(operation_context, "cpu_threads")
        if termination_grace_seconds <= 0:
            raise ValueError("local FP termination grace interval must be positive")
        self.termination_grace_seconds = float(termination_grace_seconds)
        self._fault_hook = fault_hook
        self._process_lock = threading.Lock()
        self._owned_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._reattached_processes: dict[
            str, tuple[dict[str, object], Path]
        ] = {}
        self._interrupted_launches: set[str] = set()
        self._cancelled = False

    def run(
        self, commands: Sequence[FPLocalTaskCommand]
    ) -> tuple[FPLocalCompletedTask, ...]:
        normalized = tuple(commands)
        _validate_commands(normalized)
        if not normalized:
            return ()
        pool = ThreadPoolExecutor(
            max_workers=self.fp_workers, thread_name_prefix="dice-local-fp"
        )
        try:
            futures = [pool.submit(self._run_one, command) for command in normalized]
            return tuple(future.result() for future in futures)
        except BaseException:
            self._terminate_owned_processes()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _run_one(self, command: FPLocalTaskCommand) -> FPLocalCompletedTask:
        workspace = Path(command.workspace).resolve()
        executable = self.operation_context.executable(command.logical_executable)
        stdout_path = workspace / "fp.stdout.log"
        stderr_path = workspace / "fp.stderr.log"
        command_digest = _command_digest(
            executable=executable,
            arguments=command.arguments,
            input_digest=command.input_digest,
        )
        workspace_identity = _workspace_identity(workspace)
        started_path = workspace / "supervisor_started.json"
        exit_path = workspace / "supervisor_exit.json"
        receipt_path = workspace / "process_receipt.json"
        status_path = workspace / "observed_status.json"
        previous_intent = _read_json(workspace / "launch_intent.json")
        previous_status = _read_json(status_path)
        if previous_intent is not None:
            _validate_intent_record(previous_intent, workspace=workspace)
            reconciled = self._reconcile_existing_launch(
                command=command,
                workspace=workspace,
                command_digest=command_digest,
                workspace_identity=workspace_identity,
                intent=previous_intent,
                status=previous_status,
            )
            if reconciled is not None:
                return reconciled
        else:
            orphan_paths = tuple(
                path
                for path in (started_path, exit_path, receipt_path, status_path)
                if path.is_file()
            )
            orphan_pids = _find_workspace_fp_processes(workspace_identity)
            if orphan_paths or orphan_pids:
                raise FPProcessIndeterminateError(
                    "local FP process evidence exists without launch intent; "
                    f"task={command.task_id} workspace={workspace} "
                    f"pids={orphan_pids}"
                )
        attempt = int(previous_intent.get("attempt", 0)) + 1 if previous_intent else 1
        launch_key = _launch_key(
            task_id=command.task_id,
            attempt=attempt,
            command_digest=command_digest,
            input_digest=command.input_digest,
            workspace_identity=workspace_identity,
        )
        for stale in (
            started_path,
            exit_path,
            receipt_path,
            status_path,
        ):
            stale.unlink(missing_ok=True)
        source_commands = " ".join(
            f"source {shlex.quote(str(item['path']))};"
            for item in self.operation_context.source_scripts
        )
        thread_exports = " ".join(
            f"export {name}={self.cpu_threads};"
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        )
        wrapper_command = (
            f"{shlex.quote(executable)} "
            + " ".join(shlex.quote(str(item)) for item in command.arguments)
        )
        started_tmp = started_path.with_suffix(".json.tmp")
        exit_tmp = exit_path.with_suffix(".json.tmp")
        shell_command = (
            "set +e; "
            "dice_fp_pid=$$; "
            "dice_fp_start_ticks=$(awk '{print $22}' /proc/$$/stat); "
            "dice_fp_boot_id=$(cat /proc/sys/kernel/random/boot_id); "
            f"printf '{{\"schema\":\"dice.local-fp-supervisor-start\",\"schema_revision\":1,\"launch_key\":\"%s\",\"pid\":%s,\"process_start_identity\":\"%s:%s\",\"state\":\"started\"}}\\n' "
            f"{shlex.quote(launch_key)} \"$dice_fp_pid\" \"$dice_fp_boot_id\" \"$dice_fp_start_ticks\" > {shlex.quote(str(started_tmp))}; "
            f"mv {shlex.quote(str(started_tmp))} {shlex.quote(str(started_path))}; "
            f"{source_commands} {thread_exports} {wrapper_command}; dice_fp_rc=$?; "
            f"printf '{{\"schema\":\"dice.local-fp-supervisor-exit\",\"schema_revision\":1,\"launch_key\":\"%s\",\"returncode\":%s,\"state\":\"terminal\"}}\\n' "
            f"{shlex.quote(launch_key)} \"$dice_fp_rc\" > {shlex.quote(str(exit_tmp))}; "
            f"mv {shlex.quote(str(exit_tmp))} {shlex.quote(str(exit_path))}; "
            "exit \"$dice_fp_rc\""
        )
        process_command_digest = hashlib.sha256(
            b"bash\0-c\0" + shell_command.encode("utf-8") + b"\0"
        ).hexdigest()
        intent = {
            "schema": "dice.local-fp-launch-intent",
            "schema_revision": 1,
            "task_id": command.task_id,
            "attempt": attempt,
            "launch_key": launch_key,
            "command_digest": command_digest,
            "process_command_digest": process_command_digest,
            "input_digest": command.input_digest,
            "workspace_identity": workspace_identity,
            "resources": {
                "fp_workers": self.fp_workers,
                "cpu_threads": self.cpu_threads,
            },
            "stdout_path": stdout_path.name,
            "stderr_path": stderr_path.name,
        }
        _atomic_write_json(workspace / "launch_intent.json", intent)
        self._emit_fault("after_intent", intent)
        child_environment = os.environ.copy()
        child_environment["DICE_FP_LAUNCH_KEY"] = launch_key
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                ["bash", "-c", shell_command],
                cwd=workspace,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=child_environment,
            )
            self._register_process(launch_key, process)
            self._emit_fault(
                "after_spawn", {**intent, "pid": process.pid}
            )
            receipt = {
                "schema": "dice.local-fp-process-receipt",
                "schema_revision": 1,
                "task_id": command.task_id,
                "attempt": attempt,
                "launch_key": launch_key,
                "pid": process.pid,
                "process_start_identity": _process_start_identity(process.pid),
                "command_digest": command_digest,
                "process_command_digest": process_command_digest,
                "input_digest": command.input_digest,
                "workspace_identity": workspace_identity,
                "resources": intent["resources"],
                "stdout_path": stdout_path.name,
                "stderr_path": stderr_path.name,
            }
            _atomic_write_json(workspace / "process_receipt.json", receipt)
            self._emit_fault("after_receipt", receipt)
            try:
                returncode = process.wait()
            finally:
                interrupted = self._unregister_process(launch_key)
        _atomic_write_json(
            workspace / "observed_status.json",
            {
                "schema": "dice.local-fp-observed-status",
                "schema_revision": 1,
                "task_id": command.task_id,
                "attempt": attempt,
                "launch_key": launch_key,
                "state": "interrupted" if interrupted else "terminal",
                "returncode": returncode,
            },
        )
        _bound_log(stdout_path)
        _bound_log(stderr_path)
        return FPLocalCompletedTask(
            task_id=command.task_id,
            workspace=workspace,
            input_digest=command.input_digest,
            command_digest=command_digest,
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def _emit_fault(self, stage: str, evidence: dict[str, object]) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage, dict(evidence))

    def _reconcile_existing_launch(
        self,
        *,
        command: FPLocalTaskCommand,
        workspace: Path,
        command_digest: str,
        workspace_identity: dict[str, str | int],
        intent: dict[str, object],
        status: dict[str, object] | None,
    ) -> FPLocalCompletedTask | None:
        if (
            intent.get("task_id") != command.task_id
            or intent.get("command_digest") != command_digest
            or intent.get("input_digest") != command.input_digest
            or intent.get("workspace_identity") != workspace_identity
            or not isinstance(intent.get("launch_key"), str)
            or not intent["launch_key"]
        ):
            raise FPProcessIndeterminateError(
                f"local FP launch evidence mismatch for {command.task_id}; workspace={workspace}"
            )
        launch_key = str(intent["launch_key"])
        receipt = _read_json(workspace / "process_receipt.json")
        terminal = _read_supervisor_exit(workspace, launch_key=launch_key)
        if status is not None:
            _validate_status(intent, status, workspace=workspace)

        if _is_confirmed_terminal(intent, status):
            returncode = int(status["returncode"])
            if status.get("state") == "terminal" and returncode == 0:
                if terminal != 0 or receipt is None:
                    raise FPProcessIndeterminateError(
                        f"local FP success evidence is incomplete for {command.task_id}; workspace={workspace}"
                    )
                _validate_receipt(intent, receipt, workspace=workspace)
                return _completed_task(command, command_digest, returncode)
            return None

        if receipt is None:
            receipt = self._recover_missing_receipt(
                intent, workspace=workspace, terminal=terminal
            )
            if receipt is not None:
                _atomic_write_json(workspace / "process_receipt.json", receipt)

        if terminal is not None:
            if receipt is None:
                raise FPProcessIndeterminateError(
                    f"local FP terminal evidence lacks exact process identity for {command.task_id}; workspace={workspace}"
                )
            _validate_receipt(intent, receipt, workspace=workspace)
            _write_observed_status(
                workspace,
                intent=intent,
                state="terminal",
                returncode=terminal,
            )
            _bound_log(workspace / "fp.stdout.log")
            _bound_log(workspace / "fp.stderr.log")
            return _completed_task(command, command_digest, terminal)

        if receipt is None:
            # The supervisor contract writes its handshake before invoking the
            # wrapper. With neither handshake nor matching process, this
            # intent stopped before scientific execution and a new attempt is safe.
            return None
        _validate_receipt(intent, receipt, workspace=workspace)
        pid = int(receipt["pid"])
        self._validate_live_receipt_process(receipt, workspace=workspace)
        self._register_reattached(receipt, workspace=workspace)
        try:
            while True:
                terminal = _read_supervisor_exit(workspace, launch_key=launch_key)
                if terminal is not None:
                    _write_observed_status(
                        workspace,
                        intent=intent,
                        state="terminal",
                        returncode=terminal,
                    )
                    _bound_log(workspace / "fp.stdout.log")
                    _bound_log(workspace / "fp.stderr.log")
                    return _completed_task(command, command_digest, terminal)
                observed = _read_json(workspace / "observed_status.json")
                if observed is not None:
                    _validate_status(intent, observed, workspace=workspace)
                    if observed.get("state") == "interrupted":
                        return _completed_task(
                            command, command_digest, int(observed["returncode"])
                        )
                if not Path(f"/proc/{pid}").exists():
                    raise FPProcessIndeterminateError(
                        f"local FP process {pid} disappeared without exact terminal evidence for {command.task_id}; workspace={workspace}"
                    )
                self._validate_live_receipt_process(receipt, workspace=workspace)
                time.sleep(0.05)
        finally:
            self._unregister_reattached(launch_key)

    def _recover_missing_receipt(
        self,
        intent: dict[str, object],
        *,
        workspace: Path,
        terminal: int | None,
    ) -> dict[str, object] | None:
        launch_key = str(intent["launch_key"])
        deadline = time.monotonic() + 0.25
        while True:
            started = _read_json(workspace / "supervisor_started.json")
            if started is not None:
                pid = started.get("pid")
                start_identity = started.get("process_start_identity")
                if (
                    started.get("schema") != "dice.local-fp-supervisor-start"
                    or started.get("schema_revision") != 1
                    or started.get("launch_key") != launch_key
                    or started.get("state") != "started"
                    or not isinstance(pid, int)
                    or isinstance(pid, bool)
                    or pid < 1
                    or not isinstance(start_identity, str)
                    or not start_identity
                ):
                    raise FPProcessIndeterminateError(
                        f"invalid local FP supervisor handshake; task={intent.get('task_id')} workspace={workspace}"
                    )
                receipt = _recovered_receipt(
                    intent, pid=pid, process_start_identity=start_identity
                )
                if terminal is None:
                    self._validate_live_receipt_process(receipt, workspace=workspace)
                return receipt

            tagged = _find_launch_key_processes(launch_key)
            candidates = [
                pid for pid in tagged if _process_matches_intent(pid, intent)
            ]
            if len(candidates) > 1:
                raise FPProcessIndeterminateError(
                    f"multiple local processes match launch key for {intent.get('task_id')}; workspace={workspace}"
                )
            if len(candidates) == 1:
                pid = candidates[0]
                receipt = _recovered_receipt(
                    intent,
                    pid=pid,
                    process_start_identity=_process_start_identity(pid),
                )
                self._validate_live_receipt_process(receipt, workspace=workspace)
                return receipt
            if tagged:
                raise FPProcessIndeterminateError(
                    "local FP launch-key process exists without its exact "
                    f"supervisor owner; task={intent.get('task_id')} "
                    f"workspace={workspace} pids={tagged}"
                )
            if terminal is not None:
                raise FPProcessIndeterminateError(
                    f"local FP exit receipt exists without its required start handshake; workspace={workspace}"
                )
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    @staticmethod
    def _validate_live_receipt_process(
        receipt: dict[str, object], *, workspace: Path
    ) -> None:
        pid = int(receipt["pid"])
        try:
            start_identity = _process_start_identity(pid)
            cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
            cwd_identity = _workspace_identity(cwd)
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError) as exc:
            raise FPProcessIndeterminateError(
                f"cannot inspect local FP process {pid}; workspace={workspace}"
            ) from exc
        expected_environment = (
            f"DICE_FP_LAUNCH_KEY={receipt['launch_key']}".encode("utf-8")
        )
        if (
            start_identity != receipt.get("process_start_identity")
            or cwd_identity != receipt.get("workspace_identity")
            or expected_environment not in environment
            or hashlib.sha256(cmdline).hexdigest()
            != receipt.get("process_command_digest")
        ):
            raise FPProcessIndeterminateError(
                f"local FP PID/start/command/input/workspace evidence mismatch; task={receipt.get('task_id')} pid={pid} workspace={workspace}"
            )

    def _register_process(
        self, launch_key: str, process: subprocess.Popen[bytes]
    ) -> None:
        with self._process_lock:
            cancelled = self._cancelled
            if cancelled:
                self._interrupted_launches.add(launch_key)
            else:
                self._owned_processes[launch_key] = process
        if cancelled:
            _signal_process_group(process, signal.SIGTERM)

    def _unregister_process(self, launch_key: str) -> bool:
        with self._process_lock:
            self._owned_processes.pop(launch_key, None)
            return launch_key in self._interrupted_launches

    def _register_reattached(
        self, receipt: dict[str, object], *, workspace: Path
    ) -> None:
        launch_key = str(receipt["launch_key"])
        with self._process_lock:
            cancelled = self._cancelled
            if not cancelled:
                self._reattached_processes[launch_key] = (dict(receipt), workspace)
        if cancelled:
            self._terminate_reattached_process(receipt, workspace=workspace)

    def _unregister_reattached(self, launch_key: str) -> None:
        with self._process_lock:
            self._reattached_processes.pop(launch_key, None)

    def _terminate_owned_processes(self) -> None:
        with self._process_lock:
            self._cancelled = True
            owned = tuple(self._owned_processes.items())
            reattached = tuple(self._reattached_processes.values())
            self._interrupted_launches.update(key for key, _ in owned)
        live = [process for _, process in owned if process.poll() is None]
        for process in live:
            _signal_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + self.termination_grace_seconds
        while live and time.monotonic() < deadline:
            live = [process for process in live if process.poll() is None]
            if live:
                time.sleep(0.02)
        for process in live:
            _signal_process_group(process, signal.SIGKILL)
        for receipt, workspace in reattached:
            self._terminate_reattached_process(receipt, workspace=workspace)

    def _terminate_reattached_process(
        self, receipt: dict[str, object], *, workspace: Path
    ) -> None:
        pid = int(receipt["pid"])
        if not self._receipt_process_is_live(receipt, workspace=workspace):
            return
        try:
            if os.getpgid(pid) != pid:
                raise FPProcessIndeterminateError(
                    f"local FP reattached process group mismatch; pid={pid} workspace={workspace}"
                )
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + self.termination_grace_seconds
        while (
            self._receipt_process_identity_is_live(receipt, workspace=workspace)
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        if self._receipt_process_identity_is_live(receipt, workspace=workspace):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + self.termination_grace_seconds
            while (
                self._receipt_process_identity_is_live(
                    receipt, workspace=workspace
                )
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
        if self._receipt_process_identity_is_live(receipt, workspace=workspace):
            raise FPProcessIndeterminateError(
                f"local FP reattached process did not terminate; pid={pid} workspace={workspace}"
            )
        terminal = _read_supervisor_exit(
            workspace, launch_key=str(receipt["launch_key"])
        )
        _write_observed_status(
            workspace,
            intent=receipt,
            state="terminal" if terminal is not None else "interrupted",
            returncode=terminal if terminal is not None else -int(signal.SIGTERM),
        )

    def _receipt_process_is_live(
        self, receipt: dict[str, object], *, workspace: Path
    ) -> bool:
        if not self._receipt_process_identity_is_live(
            receipt, workspace=workspace
        ):
            return False
        self._validate_live_receipt_process(receipt, workspace=workspace)
        return True

    @staticmethod
    def _receipt_process_identity_is_live(
        receipt: dict[str, object], *, workspace: Path
    ) -> bool:
        pid = int(receipt["pid"])
        try:
            identity, state = _process_identity_and_state(pid)
        except (FileNotFoundError, ProcessLookupError, OSError):
            return False
        if identity != receipt.get("process_start_identity"):
            raise FPProcessIndeterminateError(
                f"local FP process identity changed during termination; pid={pid} workspace={workspace}"
            )
        return state not in {"X", "Z"}


def _signal_process_group(
    process: subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _positive_runtime_int(context: OperationContext, name: str) -> int:
    value = context.runtime.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            f"local FP executor requires positive integer runtime.{name}"
        )
    return value


def _validate_commands(commands: tuple[FPLocalTaskCommand, ...]) -> None:
    task_ids: set[str] = set()
    workspaces: set[Path] = set()
    for command in commands:
        workspace = Path(command.workspace).resolve()
        if (
            not _TASK_ID.fullmatch(command.task_id)
            or command.task_id in task_ids
            or workspace in workspaces
            or not workspace.is_dir()
            or not command.input_digest
            or not command.logical_executable
            or not isinstance(command.arguments, tuple)
            or not all(isinstance(item, str) for item in command.arguments)
        ):
            raise ValueError("invalid or duplicate local FP task command")
        task_ids.add(command.task_id)
        workspaces.add(workspace)


def _command_digest(
    *, executable: str, arguments: tuple[str, ...], input_digest: str
) -> str:
    payload = {
        "schema": "dice.local-fp-command",
        "schema_revision": 1,
        "executable": executable,
        "arguments": list(arguments),
        "input_digest": input_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _workspace_identity(workspace: Path) -> dict[str, str | int]:
    stat = workspace.stat()
    return {
        "path": str(workspace),
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _launch_key(
    *,
    task_id: str,
    attempt: int,
    command_digest: str,
    input_digest: str,
    workspace_identity: dict[str, str | int],
) -> str:
    payload = {
        "task_id": task_id,
        "attempt": attempt,
        "command_digest": command_digest,
        "input_digest": input_digest,
        "workspace_identity": workspace_identity,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _process_start_identity(pid: int) -> str:
    identity, _state = _process_identity_and_state(pid)
    return identity


def _process_identity_and_state(pid: int) -> tuple[str, str]:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    state = fields[0]
    start_ticks = fields[19]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8"
    ).strip()
    return f"{boot_id}:{start_ticks}", state


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    staged.replace(path)


def _write_observed_status(
    workspace: Path,
    *,
    intent: dict[str, object],
    state: str,
    returncode: int,
) -> None:
    _atomic_write_json(
        workspace / "observed_status.json",
        {
            "schema": "dice.local-fp-observed-status",
            "schema_revision": 1,
            "task_id": intent["task_id"],
            "attempt": intent["attempt"],
            "launch_key": intent["launch_key"],
            "state": state,
            "returncode": returncode,
        },
    )


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        label = _evidence_label(path)
        raise FPProcessIndeterminateError(
            f"invalid local FP {label} evidence at {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise FPProcessIndeterminateError(
            f"invalid local FP {_evidence_label(path)} evidence at {path}"
        )
    return payload


def _evidence_label(path: Path) -> str:
    return {
        "launch_intent.json": "launch intent",
        "process_receipt.json": "process receipt",
        "observed_status.json": "observed status",
        "supervisor_started.json": "supervisor start",
        "supervisor_exit.json": "supervisor exit",
    }.get(path.name, path.name)


def _validate_intent_record(
    intent: dict[str, object], *, workspace: Path
) -> None:
    workspace_identity = intent.get("workspace_identity")
    resources = intent.get("resources")
    attempt = intent.get("attempt")
    if (
        intent.get("schema") != "dice.local-fp-launch-intent"
        or intent.get("schema_revision") != 1
        or not isinstance(intent.get("task_id"), str)
        or not _TASK_ID.fullmatch(str(intent.get("task_id")))
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
        or not all(
            isinstance(intent.get(field), str) and bool(intent.get(field))
            for field in (
                "launch_key",
                "command_digest",
                "process_command_digest",
                "input_digest",
                "stdout_path",
                "stderr_path",
            )
        )
        or not isinstance(workspace_identity, dict)
        or not isinstance(resources, dict)
        or set(resources) != {"fp_workers", "cpu_threads"}
        or not all(
            isinstance(resources.get(field), int)
            and not isinstance(resources.get(field), bool)
            and int(resources[field]) > 0
            for field in ("fp_workers", "cpu_threads")
        )
    ):
        raise FPProcessIndeterminateError(
            f"invalid local FP launch intent evidence; workspace={workspace}"
        )


def _validate_status(
    intent: dict[str, object],
    status: dict[str, object],
    *,
    workspace: Path,
) -> None:
    returncode = status.get("returncode")
    if (
        status.get("schema") != "dice.local-fp-observed-status"
        or status.get("schema_revision") != 1
        or status.get("task_id") != intent.get("task_id")
        or status.get("attempt") != intent.get("attempt")
        or status.get("launch_key") != intent.get("launch_key")
        or status.get("state") not in {"terminal", "interrupted"}
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or not -255 <= returncode <= 255
    ):
        raise FPProcessIndeterminateError(
            f"invalid local FP observed status evidence; task={intent.get('task_id')} workspace={workspace}"
        )


def _is_confirmed_terminal(
    intent: dict[str, object], status: dict[str, object] | None
) -> bool:
    return bool(
        status is not None
        and status.get("state") in {"terminal", "interrupted"}
        and status.get("launch_key") == intent.get("launch_key")
        and isinstance(status.get("returncode"), int)
        and not isinstance(status.get("returncode"), bool)
    )


def _read_supervisor_exit(workspace: Path, *, launch_key: str) -> int | None:
    payload = _read_json(workspace / "supervisor_exit.json")
    if payload is None:
        return None
    returncode = payload.get("returncode")
    if (
        payload.get("schema") != "dice.local-fp-supervisor-exit"
        or payload.get("schema_revision") != 1
        or payload.get("launch_key") != launch_key
        or payload.get("state") != "terminal"
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or not 0 <= returncode <= 255
    ):
        raise FPProcessIndeterminateError(
            f"invalid local FP supervisor exit evidence in {workspace}"
        )
    return returncode


def _validate_receipt(
    intent: dict[str, object], receipt: dict[str, object], *, workspace: Path
) -> None:
    matching_fields = (
        "task_id",
        "attempt",
        "launch_key",
        "command_digest",
        "process_command_digest",
        "input_digest",
        "workspace_identity",
        "resources",
        "stdout_path",
        "stderr_path",
    )
    pid = receipt.get("pid")
    if (
        receipt.get("schema") != "dice.local-fp-process-receipt"
        or receipt.get("schema_revision") != 1
        or any(receipt.get(field) != intent.get(field) for field in matching_fields)
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid < 1
        or not isinstance(receipt.get("process_start_identity"), str)
        or not receipt["process_start_identity"]
    ):
        raise FPProcessIndeterminateError(
            f"local FP process receipt mismatch; task={intent.get('task_id')} pid={pid} workspace={workspace}"
        )


def _recovered_receipt(
    intent: dict[str, object], *, pid: int, process_start_identity: str
) -> dict[str, object]:
    return {
        "schema": "dice.local-fp-process-receipt",
        "schema_revision": 1,
        **{
            field: intent[field]
            for field in (
                "task_id",
                "attempt",
                "launch_key",
                "command_digest",
                "process_command_digest",
                "input_digest",
                "workspace_identity",
                "resources",
                "stdout_path",
                "stderr_path",
            )
        },
        "pid": pid,
        "process_start_identity": process_start_identity,
        "recovered": True,
    }


def _find_launch_key_processes(launch_key: str) -> list[int]:
    expected = f"DICE_FP_LAUNCH_KEY={launch_key}".encode("utf-8")
    matches: list[int] = []
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            if expected in (process_root / "environ").read_bytes().split(b"\0"):
                matches.append(int(process_root.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
    return matches


def _process_matches_intent(pid: int, intent: dict[str, object]) -> bool:
    workspace_identity = intent.get("workspace_identity")
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        cwd_identity = _workspace_identity(
            Path(f"/proc/{pid}/cwd").resolve(strict=True)
        )
        return bool(
            isinstance(workspace_identity, dict)
            and cwd_identity == workspace_identity
            and os.getpgid(pid) == pid
            and hashlib.sha256(cmdline).hexdigest()
            == intent.get("process_command_digest")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False


def _find_workspace_fp_processes(
    workspace_identity: dict[str, str | int],
) -> list[int]:
    prefix = b"DICE_FP_LAUNCH_KEY="
    matches: list[int] = []
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            environment = (process_root / "environ").read_bytes().split(b"\0")
            if not any(entry.startswith(prefix) for entry in environment):
                continue
            cwd_stat = (process_root / "cwd").stat()
            if (
                cwd_stat.st_dev == workspace_identity["device"]
                and cwd_stat.st_ino == workspace_identity["inode"]
            ):
                matches.append(int(process_root.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
    return matches


def _completed_task(
    command: FPLocalTaskCommand, command_digest: str, returncode: int
) -> FPLocalCompletedTask:
    workspace = Path(command.workspace).resolve()
    return FPLocalCompletedTask(
        task_id=command.task_id,
        workspace=workspace,
        input_digest=command.input_digest,
        command_digest=command_digest,
        returncode=returncode,
        stdout_path=workspace / "fp.stdout.log",
        stderr_path=workspace / "fp.stderr.log",
    )


def _bound_log(path: Path, *, limit_bytes: int = FP_LOG_LIMIT_BYTES) -> None:
    size = path.stat().st_size
    if size <= limit_bytes:
        return
    prefix = f"[DICE truncated {size - limit_bytes} or more leading bytes]\n".encode(
        "utf-8"
    )
    keep = limit_bytes - len(prefix)
    with path.open("rb") as handle:
        handle.seek(size - keep)
        tail = handle.read(keep)
    staged = path.with_suffix(path.suffix + ".tmp")
    with staged.open("wb") as handle:
        handle.write(prefix)
        handle.write(tail)
    staged.replace(path)


__all__ = [
    "FP_LOG_LIMIT_BYTES",
    "FPProcessIndeterminateError",
    "FPLocalCompletedTask",
    "FPLocalTaskCommand",
    "LocalFPExecutor",
]
