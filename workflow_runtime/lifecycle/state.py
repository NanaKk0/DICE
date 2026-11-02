from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any


ArtifactCheck = Callable[[Path], bool]
BlockPublisher = Callable[[Path], Any]


@dataclass(frozen=True, slots=True)
class LifecycleBlockObservation:
    schema: str
    schema_revision: int
    state: str
    block: str
    iter: int
    reason: str
    summary: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecycleReconciliationObservation:
    """Storage-facing evidence with no durable location exposed to callers."""

    status: str
    evidence: str
    reason: str


class LifecycleState:
    """Canonical-result lifecycle state for one iter."""

    def __init__(
        self,
        *,
        root: str | Path,
        iter_index: int,
        blocks: tuple[str, ...],
        artifact_checks: dict[str, ArtifactCheck],
    ) -> None:
        self.root = Path(root).resolve()
        self.iter_index = iter_index
        self.blocks = blocks
        self.artifact_checks = artifact_checks
        if len(set(blocks)) != len(blocks) or any(
            block not in artifact_checks for block in blocks
        ):
            raise ValueError("every lifecycle block needs one artifact check")

    @property
    def iter_root(self) -> Path:
        return self.root / f"iter{self.iter_index}"

    @property
    def state_root(self) -> Path:
        return self.root / ".dice_state"

    def block_path(self, block: str) -> Path:
        self._require_block(block)
        return self.iter_root / block

    def staging_path(self, block: str) -> Path:
        block_path = self.block_path(block)
        return block_path.with_name(f".{block_path.name}.staging")

    def temporary_index_path(self, block: str, name: str) -> Path:
        """Return bounded lifecycle-local temporary state owned by the runtime layer."""
        self._require_block(block)
        if not name or "/" in name or "\\" in name:
            raise ValueError("lifecycle temporary index name must be one filename")
        return self.state_root / "indexes" / f"iter{self.iter_index}" / f"{block.replace('/', '__')}.{name}"

    def remove_retired_sample_observation_index(self, screen_block: str) -> None:
        """Remove the retired full-geometry index only after screen publication succeeds."""
        self._require_block(screen_block)
        if not screen_block.endswith("/screen"):
            raise ValueError("retired sample observation cleanup requires an rN/screen block")
        sample_block = f"{screen_block.rsplit('/', 1)[0]}/sample"
        self._require_block(sample_block)
        path = self.state_root / "indexes" / f"iter{self.iter_index}" / f"{sample_block.replace('/', '__')}.observations.xyz"
        path.unlink(missing_ok=True)

    def is_complete(self, block: str) -> bool:
        self._require_block(block)
        return bool(self.artifact_checks[block](self.block_path(block)))

    def observe(self, block: str) -> LifecycleBlockObservation:
        """Read one block's canonical result without reconciliation."""
        self._require_block(block)
        if not self.artifact_checks[block](self.block_path(block)):
            return self._observation(
                block,
                state="pending",
                reason="result_unavailable",
                summary=(
                    f"lifecycle block {block} has no consumer-runnable "
                    "canonical result"
                ),
            )
        return self._observation(
            block,
            state="succeeded",
            reason="result_reusable",
            summary=(
                f"lifecycle block {block} has a consumer-runnable canonical result"
            ),
        )

    def _observation(
        self,
        block: str,
        *,
        state: str,
        reason: str,
        summary: str,
    ) -> LifecycleBlockObservation:
        return LifecycleBlockObservation(
            schema="dice.lifecycle.result-observation",
            schema_revision=1,
            state=state,
            block=block,
            iter=self.iter_index,
            reason=reason,
            summary=summary,
        )

    def publish(
        self,
        block: str,
        publisher: BlockPublisher,
        *,
        diagnostics: dict[str, Any] | None = None,
        phase_callback: Callable[[str], None] | None = None,
        failure_callback: Callable[[Path, BaseException], None] | None = None,
    ) -> None:
        """Publish one synthetic or backend-complete block only after its artifact check passes."""
        block_path = self.block_path(block)
        staged_path = self.staging_path(block)
        backup_path = block_path.with_name(f".{block_path.name}.previous")
        shutil.rmtree(staged_path, ignore_errors=True)
        staged_path.mkdir(parents=True, exist_ok=True)
        try:
            publisher(staged_path)
            if phase_callback is not None:
                phase_callback("validating")
            if not self.artifact_checks[block](staged_path):
                raise ValueError(
                    f"lifecycle block {block} artifact check failed; "
                    "no canonical result was published"
                )
            if phase_callback is not None:
                phase_callback("publishing")
            self._move_attempts(block_path, staged_path)
            shutil.rmtree(backup_path, ignore_errors=True)
            if block_path.exists():
                block_path.replace(backup_path)
            try:
                staged_path.replace(block_path)
            except Exception:
                if backup_path.exists():
                    backup_path.replace(block_path)
                self._move_attempts(staged_path, block_path)
                raise
            shutil.rmtree(backup_path, ignore_errors=True)
        except BaseException as exc:
            try:
                if failure_callback is not None:
                    failure_callback(staged_path, exc)
            finally:
                shutil.rmtree(staged_path, ignore_errors=True)
            raise

    def reconcile(self, *, resume: str | None = None) -> str | None:
        """Private compatibility route using canonical results only."""
        self._require_no_higher_round_evidence()
        if resume is None:
            for index, block in enumerate(self.blocks):
                if not self.is_complete(block):
                    self._prepare_append_only_extension(index)
                    self._cleanup_from(index)
                    return block
            return None
        self._require_block(resume)
        target_index = self.blocks.index(resume)
        for block in self.blocks[:target_index]:
            if not self.is_complete(block):
                raise ValueError(f"cannot resume {resume}: preceding lifecycle block {block} is not complete")
        self._prepare_append_only_extension(target_index)
        self._cleanup_from(target_index)
        return resume

    def reconciliation_observation(
        self,
        block: str,
    ) -> LifecycleReconciliationObservation:
        """Classify reconciliation evidence without cleanup or path disclosure."""

        self._require_block(block)
        if not self.artifact_checks[block](self.block_path(block)):
            return LifecycleReconciliationObservation(
                "rebuildable-incomplete",
                "artifact-invalid",
                "canonical artifact is incomplete",
            )
        return LifecycleReconciliationObservation(
            "complete",
            "canonical-complete",
            "validated canonical completion",
        )

    def prepare_reconciliation(self, first_run: str | None) -> None:
        """Apply one already validated Process Manager suffix decision."""

        self._require_no_higher_round_evidence()
        if first_run is None:
            return
        self._require_block(first_run)
        index = self.blocks.index(first_run)
        self._prepare_append_only_extension(index)
        self._cleanup_from(index)

    def archive_higher_round_evidence(self, *, after_round: int) -> Path:
        """Explicitly archive evidence above the configured terminal round.

        This is the narrow, opt-in counterpart to ordinary reconciliation's
        fail-closed round-shrink guard.  Renames stay on the run filesystem and
        a durable manifest makes an interrupted archive resumable.
        """

        if (
            not isinstance(after_round, int)
            or isinstance(after_round, bool)
            or after_round < 0
        ):
            raise ValueError("round truncation target must be a non-negative integer")
        configured_rounds = self._configured_rounds()
        if not configured_rounds or max(configured_rounds) != after_round:
            raise ValueError(
                "round truncation target must equal the configured terminal round"
            )
        if "fp" not in self.blocks:
            raise ValueError("round truncation requires a terminal fp block")
        fp_index = self.blocks.index("fp")
        for block in self.blocks[:fp_index]:
            if not self.is_complete(block):
                raise ValueError(
                    f"cannot truncate after r{after_round}: preceding lifecycle "
                    f"block {block} is not complete"
                )
        self._require_terminal_execution_quiescent()

        archive_parent = (
            self.state_root / "round_truncations" / f"iter{self.iter_index}"
        )
        archive = archive_parent / f"after-r{after_round}"
        staged = archive_parent / f".after-r{after_round}.staging"
        candidates = self._higher_round_evidence(after_round=after_round)
        if archive.exists() or archive.is_symlink():
            if candidates:
                raise ValueError(
                    "round truncation archive already exists while new higher-round "
                    "evidence is present"
                )
            self._validate_round_truncation_manifest(
                archive / "manifest.json", after_round=after_round, status="complete"
            )
            return archive

        relative_sources = tuple(
            str(path.relative_to(self.root)) for path in candidates
        )
        manifest = {
            "schema": "dice.lifecycle.round-truncation",
            "schema_revision": 1,
            "status": "preparing",
            "iter": self.iter_index,
            "after_round": after_round,
            "evidence": [
                {
                    "source": relative,
                    "archive": f"evidence/{relative}",
                }
                for relative in relative_sources
            ],
        }
        archive_parent.mkdir(parents=True, exist_ok=True)
        if staged.exists() or staged.is_symlink():
            existing = self._validate_round_truncation_manifest(
                staged / "manifest.json",
                after_round=after_round,
                status="preparing",
            )
            if existing["evidence"] != manifest["evidence"]:
                # The source inventory shrinks as an interrupted transaction
                # advances, so accept only the original manifest whose every
                # item is present at exactly one of source or archive paths.
                manifest = existing
        else:
            staged.mkdir()
            self._write_json_atomic(staged / "manifest.json", manifest)

        moved: list[tuple[Path, Path]] = []
        try:
            for item in manifest["evidence"]:
                source = self.root / item["source"]
                destination = staged / item["archive"]
                source_present = source.exists() or source.is_symlink()
                destination_present = destination.exists() or destination.is_symlink()
                if source_present and not destination_present:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(destination)
                    moved.append((source, destination))
                    continue
                if destination_present and not source_present:
                    continue
                raise ValueError(
                    "round truncation transaction has ambiguous evidence ownership"
                )
            manifest["status"] = "complete"
            self._write_json_atomic(staged / "manifest.json", manifest)
            staged.replace(archive)
        except BaseException:
            for source, destination in reversed(moved):
                if destination.exists() or destination.is_symlink():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            raise
        self._require_no_higher_round_evidence()
        return archive

    def _prepare_append_only_extension(self, index: int) -> None:
        block = self.blocks[index]
        prefix, separator, phase = block.partition("/")
        if (
            separator != "/"
            or phase != "train"
            or not prefix.startswith("r")
            or not prefix[1:].isdigit()
            or int(prefix[1:]) <= 0
            or not self._has_terminal_evidence()
        ):
            return
        self._require_terminal_execution_quiescent()
        for path in self._terminal_runtime_roots():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink(missing_ok=True)
        stale_label_marker = (
            self.state_root
            / "markers"
            / f"iter{self.iter_index}"
            / "fp__label.json"
        )
        if "fp/label" not in self.blocks:
            stale_label_marker.unlink(missing_ok=True)

    def _require_no_higher_round_evidence(self) -> None:
        configured_rounds = self._configured_rounds()
        if not configured_rounds:
            return
        higher = sorted(
            path.name
            for path in self._higher_round_evidence(
                after_round=max(configured_rounds)
            )
        )
        if higher:
            raise ValueError(
                "round shrink or higher-round orphan evidence is not append-only: "
                + ", ".join(higher)
            )

    def _configured_rounds(self) -> tuple[int, ...]:
        return tuple(
            int(block.split("/", 1)[0][1:])
            for block in self.blocks
            if block.startswith("r")
        )

    def _higher_round_evidence(self, *, after_round: int) -> tuple[Path, ...]:
        def round_index(name: str) -> int | None:
            prefix = name.split("__", 1)[0]
            if not prefix.startswith("r") or not prefix[1:].isdigit():
                return None
            return int(prefix[1:])

        candidates: list[Path] = []
        if self.iter_root.is_dir() and not self.iter_root.is_symlink():
            candidates.extend(self.iter_root.iterdir())
        elif self.iter_root.exists() or self.iter_root.is_symlink():
            raise ValueError("round shrink found damaged lifecycle evidence ownership")
        execution_root = self.state_root / "executions" / f"iter{self.iter_index}"
        index_root = self.state_root / "indexes" / f"iter{self.iter_index}"
        for root in (execution_root, index_root):
            if root.is_dir() and not root.is_symlink():
                candidates.extend(root.iterdir())
            elif root.exists() or root.is_symlink():
                raise ValueError(
                    "round shrink found damaged lifecycle evidence ownership"
                )
        return tuple(
            sorted(
                (
                    path
                    for path in candidates
                    if (index := round_index(path.name)) is not None
                    and index > after_round
                ),
                key=lambda path: str(path.relative_to(self.root)),
            )
        )

    def _validate_round_truncation_manifest(
        self,
        path: Path,
        *,
        after_round: int,
        status: str,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("round truncation archive manifest is missing or invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "dice.lifecycle.round-truncation"
            or payload.get("schema_revision") != 1
            or payload.get("status") != status
            or payload.get("iter") != self.iter_index
            or payload.get("after_round") != after_round
            or not isinstance(payload.get("evidence"), list)
            or any(
                not isinstance(item, dict)
                or set(item) != {"source", "archive"}
                or not isinstance(item["source"], str)
                or not isinstance(item["archive"], str)
                or not item["archive"].startswith("evidence/")
                for item in payload.get("evidence", [])
            )
        ):
            raise ValueError("round truncation archive manifest is missing or invalid")
        return payload

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        staged = path.with_suffix(path.suffix + ".tmp")
        staged.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        staged.replace(path)

    def _has_terminal_evidence(self) -> bool:
        return any(
            path.exists() or path.is_symlink()
            for path in (
                self.iter_root / "fp",
                *self._terminal_runtime_roots(),
            )
        )

    def _terminal_runtime_roots(self) -> tuple[Path, ...]:
        execution_root = (
            self.state_root / "executions" / f"iter{self.iter_index}"
        )
        return (
            execution_root / "fp",
            execution_root / "fp__label",
            self.state_root / "fp_runs" / f"iter{self.iter_index}",
        )

    def _require_terminal_execution_quiescent(self) -> None:
        execution_root = (
            self.state_root / "executions" / f"iter{self.iter_index}"
        )
        for block_name in ("fp", "fp__label"):
            block_root = execution_root / block_name
            if not block_root.is_dir() or block_root.is_symlink():
                if block_root.exists() or block_root.is_symlink():
                    raise ValueError(
                        "append-only extension found indeterminate terminal execution evidence"
                    )
                continue
            for attempt in block_root.glob("*/attempts/*"):
                if not attempt.is_dir() or attempt.is_symlink():
                    raise ValueError(
                        "append-only extension found indeterminate terminal execution evidence"
                    )
                intent = attempt / "launch_intent.json"
                receipt = attempt / "terminal_receipt.json"
                if not intent.is_file():
                    if any(attempt.iterdir()):
                        raise ValueError(
                            "append-only extension found unattributable terminal execution evidence"
                        )
                    continue
                terminal = self._read_terminal_json(receipt)
                if terminal.get("state") not in {"succeeded", "failed", "cancelled"}:
                    raise ValueError(
                        "append-only extension found live or indeterminate terminal execution"
                    )

        fp_runs = self.state_root / "fp_runs" / f"iter{self.iter_index}"
        if not fp_runs.exists():
            return
        if not fp_runs.is_dir() or fp_runs.is_symlink():
            raise ValueError(
                "append-only extension found indeterminate terminal execution evidence"
            )
        process_names = {
            "supervisor_started.json",
            "supervisor_exit.json",
            "process_receipt.json",
            "observed_status.json",
        }
        for task_root in fp_runs.glob("*/tasks/task_*"):
            if not task_root.is_dir() or task_root.is_symlink():
                raise ValueError(
                    "append-only extension found indeterminate terminal execution evidence"
                )
            intent_path = task_root / "launch_intent.json"
            present_process_evidence = {
                name for name in process_names if (task_root / name).exists()
            }
            if not intent_path.is_file():
                if present_process_evidence:
                    raise ValueError(
                        "append-only extension found unattributable terminal execution evidence"
                    )
                continue
            intent = self._read_terminal_json(intent_path)
            launch_key = intent.get("launch_key")
            status_path = task_root / "observed_status.json"
            if status_path.is_file():
                status = self._read_terminal_json(status_path)
                if (
                    status.get("launch_key", launch_key) == launch_key
                    and status.get("state") in {"terminal", "interrupted"}
                    and isinstance(status.get("returncode"), int)
                ):
                    continue
            exit_path = task_root / "supervisor_exit.json"
            receipt_path = task_root / "process_receipt.json"
            if exit_path.is_file() and receipt_path.is_file():
                terminal = self._read_terminal_json(exit_path)
                receipt = self._read_terminal_json(receipt_path)
                if (
                    terminal.get("launch_key") == launch_key
                    and receipt.get("launch_key") == launch_key
                    and terminal.get("state") == "terminal"
                    and isinstance(terminal.get("returncode"), int)
                ):
                    continue
            raise ValueError(
                "append-only extension found live or indeterminate terminal execution"
            )

    @staticmethod
    def _read_terminal_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "append-only extension found damaged terminal execution evidence"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "append-only extension found damaged terminal execution evidence"
            )
        return payload

    def _cleanup_from(self, index: int) -> None:
        suffix = self.blocks[index:]
        for block in suffix:
            nested_prefix = f"{block}/"
            preserve_children = {
                descendant.removeprefix(nested_prefix).split("/", 1)[0]
                for descendant in suffix
                if descendant.startswith(nested_prefix)
            }
            self._remove_canonical_block_content(
                self.block_path(block),
                preserve_children=preserve_children,
            )
            (
                self.state_root
                / "markers"
                / f"iter{self.iter_index}"
                / f"{block.replace('/', '__')}.json"
            ).unlink(missing_ok=True)
            self._remove_legacy_validated_prefix_receipts(block)
            for index_path in (self.state_root / "indexes" / f"iter{self.iter_index}").glob(f"{block.replace('/', '__')}.*"):
                index_path.unlink(missing_ok=True)

    def _remove_legacy_validated_prefix_receipts(self, block: str) -> None:
        """Remove only the retired receipt name in its enumerated attempt layout."""

        block_root = (
            self.state_root
            / "executions"
            / f"iter{self.iter_index}"
            / block.replace("/", "__")
        )
        if not block_root.is_dir() or block_root.is_symlink():
            return
        for unit_root in block_root.iterdir():
            if not unit_root.is_dir() or unit_root.is_symlink():
                continue
            attempts_root = unit_root / "attempts"
            if not attempts_root.is_dir() or attempts_root.is_symlink():
                continue
            for attempt_root in attempts_root.iterdir():
                if not attempt_root.is_dir() or attempt_root.is_symlink():
                    continue
                (attempt_root / "validated_prefix_receipt.json").unlink(
                    missing_ok=True
                )

    @staticmethod
    def _move_attempts(source_block: Path, destination_block: Path) -> None:
        source = source_block / ".attempts"
        if not source.exists() and not source.is_symlink():
            return
        destination = destination_block / ".attempts"
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"lifecycle block staging must not publish reserved diagnostic path {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)

    @staticmethod
    def _remove_canonical_block_content(
        block_path: Path,
        *,
        preserve_children: set[str] | None = None,
    ) -> None:
        """Remove block-owned canonical content while preserving diagnostic attempts."""
        if not block_path.exists() and not block_path.is_symlink():
            return
        if not block_path.is_dir() or block_path.is_symlink():
            block_path.unlink(missing_ok=True)
            return
        has_attempts = False
        for child in block_path.iterdir():
            if child.name == ".attempts" or child.name in (preserve_children or set()):
                has_attempts = True
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        if not has_attempts:
            shutil.rmtree(block_path, ignore_errors=True)

    def _require_block(self, block: str) -> None:
        if block not in self.blocks:
            raise ValueError(f"invalid lifecycle lifecycle block {block!r}")
