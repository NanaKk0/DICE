from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4


_TEXT_DIAGNOSTIC_NAMES = {
    "lcurve.out",
    "loss.out",
    "thermo.out",
    "log.lammps",
}
_SMALL_INPUT_SUFFIXES = {".json", ".yaml", ".yml", ".in"}
_SECRET_PATTERN = re.compile(
    r"(?im)^([ \t]*(?:token|password|secret|api[_-]?key|authorization)[ \t]*[:=][ \t]*).*$"
)


@dataclass(frozen=True, slots=True)
class AttemptSeal:
    attempt_id: str
    attempt_dir: Path
    diagnostic_ref: str
    pruned_attempt_ids: tuple[str, ...] = ()


class AttemptDiagnosticManager:
    """Seal bounded diagnostic evidence without turning staging into recovery state."""

    def __init__(self, *, iter_root: str | Path, retention: int = 3) -> None:
        if retention < 1:
            raise ValueError("lifecycle failed-attempt retention must be positive")
        self.iter_root = Path(iter_root)
        self.retention = retention

    def seal(
        self,
        *,
        block: str,
        staging_path: str | Path,
        attempt_id: str,
        outcome: str,
        error: BaseException | None = None,
        apply_retention: bool = True,
    ) -> AttemptSeal:
        source_root = Path(staging_path)
        block_root = self.iter_root / block
        attempts_root = block_root / ".attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        destination = attempts_root / attempt_id
        if destination.exists():
            raise ValueError(f"lifecycle diagnostic attempt already exists: {attempt_id}")
        destination.mkdir(parents=True)

        copied: list[str] = []
        candidates = _gcmc_candidate_files(source_root) if block.endswith('/sample') else set()
        for source in sorted(set(_diagnostic_files(source_root)) | candidates):
            relative = source.relative_to(source_root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source not in candidates and source.suffix.lower() in _SMALL_INPUT_SUFFIXES:
                value = source.read_text(encoding="utf-8", errors="replace")
                target.write_text(_redact(value), encoding="utf-8")
            else:
                shutil.copyfile(source, target)
            copied.append(relative.as_posix())

        primary = _primary_diagnostic(destination, copied)
        sealed_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        manifest: dict[str, Any] = {
            "schema": "dice.lifecycle.failed-attempt",
            "schema_revision": 1,
            "attempt_id": attempt_id,
            "block": block,
            "outcome": outcome,
            "sealed_at": sealed_at,
            "files": copied,
            "primary_diagnostic": primary,
            "error_type": None if error is None else type(error).__name__,
            "error": None if error is None else str(error),
        }
        (destination / "attempt.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        diagnostic = destination / primary if primary is not None else destination / "attempt.json"
        diagnostic_ref = diagnostic.relative_to(self.iter_root).as_posix()
        pruned = self._apply_retention(attempts_root, newest=destination) if apply_retention else ()
        return AttemptSeal(
            attempt_id=attempt_id,
            attempt_dir=destination,
            diagnostic_ref=diagnostic_ref,
            pruned_attempt_ids=pruned,
        )

    def rotate(self, *, block: str, newest_attempt_id: str) -> tuple[str, ...]:
        attempts_root = self.iter_root / block / ".attempts"
        newest = attempts_root / newest_attempt_id
        if not newest.is_dir():
            return ()
        return self._apply_retention(attempts_root, newest=newest)

    def archive_stale(self, *, block: str, staging_path: str | Path) -> AttemptSeal:
        attempt_id = (
            "stale-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-"
            f"{uuid4().hex[:8]}"
        )
        return self.seal(
            block=block,
            staging_path=staging_path,
            attempt_id=attempt_id,
            outcome="stale",
            apply_retention=False,
        )

    def _apply_retention(self, attempts_root: Path, *, newest: Path) -> tuple[str, ...]:
        attempts = [
            path
            for path in attempts_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        ]
        attempts.sort(key=_attempt_sort_key)
        if newest in attempts:
            attempts.remove(newest)
            attempts.append(newest)
        prune_count = max(0, len(attempts) - self.retention)
        pruned = tuple(path.name for path in attempts[:prune_count])
        for path in attempts[:prune_count]:
            shutil.rmtree(path, ignore_errors=True)
        for path in attempts[prune_count:]:
            if path != newest:
                _compress_attempt(path)
        return pruned


def _gcmc_candidate_files(root: Path) -> set[Path]:
    """Preserve acquired frames on controller failure, not resumable state.

    Keep only the named candidate inventory, byte-for-byte and without the
    small-input size cap. It remains untrusted attempt evidence, never a
    completion marker or permission to skip the ordinary consumer validator.
    """
    tasks = root/'tasks'
    if not tasks.is_dir() or tasks.is_symlink():
        return set()
    files = set()
    for task in tasks.iterdir():
        if (task.is_symlink() or not task.is_dir()
            or not re.fullmatch(r'(?:task_\d+|\.task_\d+\.gcmc-[\w-]+)', task.name)):
            continue
        if not (task/'gcmc_result.json').is_file() and not (task/'observations.extxyz').is_file():
            continue
        for name in ('gcmc_result.json', 'observations.extxyz', 'final_structure.extxyz'):
            path = task/name
            if path.is_file() and not path.is_symlink():
                files.add(path)
        # Raw failed MD evidence is diagnostic-only, but must survive an outer
        # device-quarantine/stage failure without the small-input size cap.
        for base in (task, task/'failed-attempt-diagnostics'):
            if base.is_symlink():
                continue
            for path in base.glob('md_*/trajectory.dump'):
                if path.is_file() and not path.is_symlink() and not path.parent.is_symlink():
                    files.add(path)
    seed = root/'seeds.xyz'
    if files and seed.is_file() and not seed.is_symlink():
        files.add(seed)
    return files


def _diagnostic_files(root: Path):
    if not root.is_dir():
        return ()
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.name.lower()
        if (
            name.endswith(".stdout.log")
            or name.endswith(".stderr.log")
            or name.endswith(".log")
            or name in _TEXT_DIAGNOSTIC_NAMES
            or (name == 'recovery.jsonl' and 'mps' in path.relative_to(root).parts)
        ):
            result.append(path)
        elif path.suffix.lower() in _SMALL_INPUT_SUFFIXES and path.stat().st_size <= 1024 * 1024:
            result.append(path)
    return tuple(result)


def _primary_diagnostic(root: Path, copied: list[str]) -> str | None:
    stderr = [name for name in copied if name.lower().endswith(".stderr.log")]
    for name in stderr:
        if (root / name).stat().st_size:
            return name
    stdout = [name for name in copied if name.lower().endswith(".stdout.log")]
    for name in stdout:
        if (root / name).stat().st_size:
            return name
    return copied[0] if copied else None


def _redact(value: str) -> str:
    return _SECRET_PATTERN.sub(r"\1<redacted>", value)


def _attempt_sort_key(path: Path) -> tuple[str, int, str]:
    try:
        payload = json.loads((path / "attempt.json").read_text(encoding="utf-8"))
        sealed_at = str(payload.get("sealed_at", "")) if isinstance(payload, dict) else ""
    except (OSError, json.JSONDecodeError):
        sealed_at = ""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return sealed_at, mtime, path.name


def _compress_attempt(attempt_dir: Path) -> None:
    manifest_path = attempt_dir / "attempt.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        return
    primary = manifest.get("primary_diagnostic")
    compressed_files = set(str(item) for item in manifest.get("compressed_files", []))
    for relative in tuple(str(item) for item in manifest.get("files", [])):
        if relative in compressed_files:
            continue
        source = attempt_dir / relative
        if not source.is_file() or source.is_symlink():
            continue
        compressed = source.with_suffix(source.suffix + ".gz")
        with source.open("rb") as input_handle, gzip.open(compressed, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
        source.unlink()
        if relative == primary:
            source.write_text(
                f"compressed diagnostic: {compressed.name}\n", encoding="utf-8"
            )
        compressed_files.add(relative)
    manifest["compressed_files"] = sorted(compressed_files)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
