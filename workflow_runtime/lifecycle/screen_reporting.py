from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from workflow_artifacts.lifecycle.screen_report import (
    load_screen_report,
    render_report_header,
    render_screen_report_sections,
)


class ScreenReportCoordinator:
    """Runtime owner for the diagnostic current-iter ``dice.log`` report."""

    def __init__(
        self,
        *,
        iter_root: str | Path,
        iter_index: int,
        warning: Callable[[Exception], None] | None = None,
        expected_route: Mapping[str, object] | None = None,
        system_ids: tuple[str, ...] | Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.iter_root = Path(iter_root)
        self.iter_index = iter_index
        self.dice_path = self.iter_root / "dice.log"
        self._warning = warning
        self._expected_route = None if expected_route is None else dict(expected_route)
        self._system_ids = (
            None
            if system_ids is None
            else {block: tuple(ids) for block, ids in system_ids.items()}
            if isinstance(system_ids, Mapping)
            else tuple(system_ids)
        )

    def ensure_header(self) -> None:
        if self.dice_path.exists():
            return
        self._atomic_write(render_report_header(self.iter_index))

    def append_snapshot(self, snapshot_path: str | Path, *, block: str) -> None:
        report = load_screen_report(
            snapshot_path,
            expected_iter=self.iter_index,
            expected_block=block,
            expected_route=self._expected_route,
            expected_roles=_expected_roles(block),
            expected_system_ids=self._expected_system_ids(block),
        )
        self.ensure_header()
        _append_section(self.dice_path, render_screen_report_sections(report))

    def append_unavailable(self, *, block: str, reason: str) -> None:
        self.ensure_header()
        detail = " ".join(str(reason).split())
        if len(detail) > 240:
            detail = detail[:239].rstrip() + "…"
        section = (
            f"[{block}]\n"
            "report unavailable: "
            f"{detail}\n"
            f"[/{block}]\n"
        )
        _append_section(self.dice_path, section)

    def regenerate_prefix(
        self,
        *,
        blocks: Sequence[str],
        target: str | None,
        report_blocks: Sequence[str],
    ) -> None:
        """Atomically rebuild the diagnostic screen prefix before ``target``."""
        ordered_blocks = tuple(blocks)
        if target is None:
            prefix = ordered_blocks
        else:
            if target not in ordered_blocks:
                raise ValueError(f"screen report regeneration target is not in lifecycle plan: {target}")
            prefix = ordered_blocks[: ordered_blocks.index(target)]
        report_block_set = set(report_blocks)
        fallback, invalid = _read_fallback_sections(self.dice_path, self.iter_index)
        for detail in invalid:
            self._warn(ValueError(detail))

        content = render_report_header(self.iter_index)
        for block in prefix:
            if block not in report_block_set:
                continue
            section: str | None = None
            snapshot_path = self.iter_root / block / "screen_report.json"
            try:
                report = load_screen_report(
                    snapshot_path,
                    expected_iter=self.iter_index,
                    expected_block=block,
                    expected_route=self._expected_route,
                    expected_roles=_expected_roles(block),
                    expected_system_ids=self._expected_system_ids(block),
                )
                section = render_screen_report_sections(report)
            except Exception as exc:
                self._warn(exc)
                if block in fallback and block not in invalid:
                    section = fallback[block]
            if section is None:
                section = _unavailable_section(
                    block,
                    "valid screen_report.json or complete dice section unavailable",
                )
            content = _join_sections(content, section)
        self._atomic_write(content)

    def _atomic_write(self, value: str) -> None:
        self.dice_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.dice_path.with_name(f".{self.dice_path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.dice_path)

    def _warn(self, exc: Exception) -> None:
        if self._warning is not None:
            self._warning(exc)

    def _expected_system_ids(self, block: str) -> tuple[str, ...] | None:
        if self._system_ids is None:
            return None
        if isinstance(self._system_ids, dict):
            return () if block == "teach/screen" else self._system_ids.get(block)
        return () if block == "teach/screen" else self._system_ids


def _append_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _append_section(path: Path, section: str) -> None:
    separator = ""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 2))
            tail = handle.read()
        if not tail.endswith(b"\n\n"):
            separator = "\n"
    _append_text(path, separator + section)


def _join_sections(prefix: str, section: str) -> str:
    if prefix and not prefix.endswith("\n\n"):
        prefix += "\n"
    return prefix + section


def _unavailable_section(block: str, reason: str) -> str:
    detail = " ".join(str(reason).split())
    if len(detail) > 240:
        detail = detail[:239].rstrip() + "…"
    return (
        f"[{block}]\n"
        f"report unavailable: {detail}\n"
        f"[/{block}]\n"
    )


def _expected_roles(block: str) -> tuple[str, ...]:
    return ("teacher",) if block == "teach/screen" else ("teacher", "student")


_SCREEN_START = re.compile(r"^\[(teach|r[0-9]+/screen)\]$")
_SCREEN_END = re.compile(r"^\[/((?:teach|r[0-9]+)/screen)\]$")


def _read_fallback_sections(path: Path, iter_index: int) -> tuple[dict[str, str], tuple[str, ...]]:
    if not path.exists():
        return {}, ()
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {}, (f"cannot read dice report fallback: {exc}",)
    header = render_report_header(iter_index)
    if not text.startswith(header):
        return {}, ("dice report header is invalid; fallback unavailable",)
    sections: dict[str, str] = {}
    invalid: set[str] = set()
    current: str | None = None
    captured: list[str] = []
    for raw_line in text[len(header) :].splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        start = _SCREEN_START.fullmatch(line)
        end = _SCREEN_END.fullmatch(line)
        if current is None:
            if start is not None:
                current = start.group(1)
                captured = [raw_line]
            continue
        captured.append(raw_line)
        if start is not None:
            invalid.add(current)
            current = start.group(1)
            captured = [raw_line]
            continue
        if end is None:
            continue
        if end.group(1) != current or "published: train=" not in "".join(captured):
            invalid.add(current)
        elif current in sections:
            invalid.add(current)
        else:
            sections[current] = "".join(captured)
        current = None
        captured = []
    if current is not None:
        invalid.add(current)
    for block in invalid:
        sections.pop(block, None)
    return sections, tuple(sorted(invalid))


__all__ = ["ScreenReportCoordinator"]
