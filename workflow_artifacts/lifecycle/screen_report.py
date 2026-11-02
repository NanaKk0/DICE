from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping
import unicodedata

from workflow_artifacts.identities import RevisionedIdentity


SCREEN_REPORT_IDENTITY = RevisionedIdentity("dice.lifecycle.screen-report", 1)
SCREEN_REPORT_SCHEMA = SCREEN_REPORT_IDENTITY.name
SCREEN_REPORT_SCHEMA_REVISION = SCREEN_REPORT_IDENTITY.revision
ROUTE_REPORT_REVISION = 1
GMM_EVIDENCE = {"name": "gmm-reference-tail", "revision": 4}
COMMITTEE_EVIDENCE = {"name": "committee-force-deviation", "revision": 1}
JOINT_DIRECT_EVIDENCE = {"name": "joint-direct", "revision": 2}
GMM_COLUMNS = ("covered", "candidate", "discard")
COMMITTEE_COLUMNS = ("accurate", "candidate", "failed")
JOINT_DIRECT_COLUMNS = ("candidate", "proposal", "underfill")
_PUBLIC_SCREEN = re.compile(r"^(?:teach|r[0-9]+)/screen$")


@dataclass(frozen=True, slots=True)
class ScreenReportRow:
    system_id: str
    denominator: int
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.system_id, str) or not self.system_id:
            raise ValueError("screen report row requires a non-empty system id")
        _require_integer(self.denominator, "screen report denominator", minimum=0)
        counts = dict(self.counts)
        if any(not isinstance(key, str) or not key for key in counts):
            raise ValueError("screen report count keys must be non-empty strings")
        for key, value in counts.items():
            _require_integer(value, f"screen report count {key!r}", minimum=0)
        object.__setattr__(self, "counts", counts)

    def as_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id,
            "denominator": self.denominator,
            "counts": dict(self.counts),
        }


@dataclass(frozen=True, slots=True)
class ScreenReportRole:
    role: str
    evidence: Mapping[str, object]
    thresholds: Mapping[str, object]
    systems: tuple[ScreenReportRow, ...]
    total: ScreenReportRow
    denominator: int | None = None

    def __post_init__(self) -> None:
        if self.role not in {"teacher", "student"}:
            raise ValueError("screen report role must be teacher or student")
        evidence = dict(self.evidence)
        expected_evidence = _evidence_for(str(evidence.get("name", "")))
        if evidence != expected_evidence:
            raise ValueError("screen report evidence identity is invalid")
        thresholds = _normalize_thresholds(
            str(evidence["name"]), self.thresholds, role=self.role
        )
        systems = tuple(self.systems)
        if len({row.system_id for row in systems}) != len(systems):
            raise ValueError("screen report system rows must be unique")
        if any(row.system_id == "TOTAL" for row in systems):
            raise ValueError("screen report system rows must not contain TOTAL")
        expected_columns = _columns_for(str(evidence["name"]))
        for row in (*systems, self.total):
            if set(row.counts) != set(expected_columns):
                raise ValueError("screen report counts do not match evidence kind")
            if any(value > row.denominator for value in row.counts.values()):
                raise ValueError("screen report count exceeds its denominator")
            if str(evidence["name"]) == COMMITTEE_EVIDENCE["name"]:
                if sum(row.counts.values()) != row.denominator:
                    raise ValueError("committee screen report counts must partition N")
        if self.total.system_id != "TOTAL":
            raise ValueError("screen report total row must use TOTAL")
        denominator = self.total.denominator if self.denominator is None else self.denominator
        _require_integer(denominator, "screen report role denominator", minimum=0)
        if denominator != self.total.denominator:
            raise ValueError("screen report role denominator disagrees with total row")
        if systems and sum(row.denominator for row in systems) != denominator:
            raise ValueError("screen report system denominators do not sum to TOTAL")
        if systems:
            for column in expected_columns:
                if sum(row.counts[column] for row in systems) != self.total.counts[column]:
                    raise ValueError("screen report system counts do not sum to TOTAL")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "systems", systems)
        object.__setattr__(self, "denominator", denominator)

    @property
    def evidence_kind(self) -> str:
        return str(dict(self.evidence).get("name", ""))

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "evidence": dict(self.evidence),
            "thresholds": dict(self.thresholds),
            "denominator": self.denominator,
            "systems": [row.as_dict() for row in self.systems],
            "total": self.total.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScreenReport:
    iter_index: int
    block: str
    route: Mapping[str, object]
    roles: tuple[ScreenReportRole, ...]
    published_train: int
    published_fp_candidates: int

    def __post_init__(self) -> None:
        _require_integer(self.iter_index, "screen report iter", minimum=0)
        if not _PUBLIC_SCREEN.fullmatch(self.block):
            raise ValueError(f"invalid public screen block {self.block!r}")
        route = dict(self.route)
        base_fields = {"name", "revision"}
        extended_fields = {"name", "revision", "directional_policy"}
        route_fields = set(route)
        if (route_fields != base_fields and route_fields != extended_fields) or not isinstance(route["name"], str) or not route["name"]:
            raise ValueError("screen report route identity is invalid")
        if route["revision"] != ROUTE_REPORT_REVISION:
            raise ValueError("screen report route revision is unsupported")
        if route_fields == extended_fields:
            policy = route["directional_policy"]
            if (
                not isinstance(policy, dict)
                or policy
                not in (
                    {
                        "name": "student-reference-tail_teacher-reference-tail",
                        "revision": 1,
                    },
                    {
                        "name": "student-force-deviation_teacher-reference-tail",
                        "revision": 1,
                    },
                )
            ):
                raise ValueError("screen report directional route identity is invalid")
        roles = tuple(self.roles)
        if len({role.role for role in roles}) != len(roles):
            raise ValueError("screen report roles must be unique")
        _require_integer(self.published_train, "published train count", minimum=0)
        _require_integer(
            self.published_fp_candidates,
            "published FP candidate count",
            minimum=0,
        )
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "roles", roles)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCREEN_REPORT_SCHEMA,
            "schema_revision": SCREEN_REPORT_SCHEMA_REVISION,
            "iter": self.iter_index,
            "block": self.block,
            "route": dict(self.route),
            "roles": [role.as_dict() for role in self.roles],
            "published": {
                "train": self.published_train,
                "fp_candidate": self.published_fp_candidates,
            },
        }


def publish_screen_report(path: str | Path, report: ScreenReport) -> Path:
    """Atomically publish one diagnostic screen fact projection."""
    if not isinstance(report, ScreenReport):
        raise TypeError("screen report publisher requires ScreenReport")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    encoded = json.dumps(report.as_dict(), sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def load_screen_report(
    path: str | Path,
    *,
    expected_iter: int | None = None,
    expected_block: str | None = None,
    expected_route: Mapping[str, object] | None = None,
    expected_roles: tuple[str, ...] | None = None,
    expected_system_ids: tuple[str, ...] | None = None,
) -> ScreenReport:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid screen report diagnostic: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError("screen report diagnostic must be an object")
    report = _from_dict(payload)
    if expected_iter is not None and report.iter_index != expected_iter:
        raise ValueError("screen report iter context is invalid")
    if expected_block is not None and report.block != expected_block:
        raise ValueError("screen report block context is invalid")
    if expected_route is not None and dict(report.route) != dict(expected_route):
        raise ValueError("screen report route context is invalid")
    if expected_roles is not None and tuple(role.role for role in report.roles) != tuple(expected_roles):
        raise ValueError("screen report role set or order is invalid")
    if expected_system_ids is not None:
        expected = tuple(expected_system_ids)
        for role in report.roles:
            if tuple(row.system_id for row in role.systems) != expected:
                raise ValueError("screen report system order is invalid")
    return report


def render_report_header(iter_index: int) -> str:
    _require_integer(iter_index, "screen report iter", minimum=0)
    return (
        f"DICE iter{iter_index} screen report\n"
        "screen-report-format=1\n"
        "percent denominator: role N | GMM route sets may overlap\n\n"
    )


def render_screen_report(report: ScreenReport) -> str:
    return render_report_header(report.iter_index) + render_screen_report_sections(report)


def render_screen_report_sections(report: ScreenReport) -> str:
    lines = [f"[{report.block}]"]
    if "directional_policy" in report.route:
        policy = dict(report.route["directional_policy"])
        lines.append(
            "route: "
            f"directional_policy={policy['name']} (revision {policy['revision']})"
        )
    for role in report.roles:
        lines.extend(_render_role(role))
        lines.append("")
    lines.append(
        f"published: train={report.published_train:,} | "
        f"fp_candidate={report.published_fp_candidates:,}"
    )
    lines.append(f"[/{report.block}]")
    return "\n".join(lines) + "\n"


def _from_dict(payload: Mapping[str, object]) -> ScreenReport:
    expected = {"schema", "schema_revision", "iter", "block", "route", "roles", "published"}
    if set(payload) != expected:
        raise ValueError("screen report diagnostic fields are invalid")
    if payload["schema"] != SCREEN_REPORT_SCHEMA or payload["schema_revision"] != SCREEN_REPORT_SCHEMA_REVISION:
        raise ValueError("screen report diagnostic schema is unsupported")
    if not isinstance(payload["route"], dict) or not isinstance(payload["roles"], list) or not isinstance(payload["published"], dict):
        raise ValueError("screen report diagnostic shape is invalid")
    published = payload["published"]
    if set(published) != {"train", "fp_candidate"}:
        raise ValueError("screen report published counts are invalid")
    roles: list[ScreenReportRole] = []
    for raw_role in payload["roles"]:
        if not isinstance(raw_role, dict) or set(raw_role) != {
            "role", "evidence", "thresholds", "denominator", "systems", "total"
        }:
            raise ValueError("screen report role fields are invalid")
        if (
            not isinstance(raw_role["evidence"], dict)
            or not isinstance(raw_role["thresholds"], dict)
            or not isinstance(raw_role["systems"], list)
            or not isinstance(raw_role["total"], dict)
        ):
            raise ValueError("screen report role shape is invalid")
        systems = tuple(_row_from_dict(item) for item in raw_role["systems"])
        roles.append(
            ScreenReportRole(
                role=raw_role["role"],
                evidence=raw_role["evidence"],
                thresholds=raw_role["thresholds"],
                denominator=raw_role["denominator"],
                systems=systems,
                total=_row_from_dict(raw_role["total"]),
            )
        )
    return ScreenReport(
        iter_index=payload["iter"],
        block=payload["block"],
        route=payload["route"],
        roles=tuple(roles),
        published_train=published["train"],
        published_fp_candidates=published["fp_candidate"],
    )


def _row_from_dict(payload: object) -> ScreenReportRow:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"system_id", "denominator", "counts"}
        or not isinstance(payload["counts"], dict)
    ):
        raise ValueError("screen report row fields are invalid")
    return ScreenReportRow(
        system_id=payload["system_id"],
        denominator=payload["denominator"],
        counts=payload["counts"],
    )


def _columns_for(evidence_kind: str) -> tuple[str, ...]:
    if evidence_kind == GMM_EVIDENCE["name"]:
        return GMM_COLUMNS
    if evidence_kind == COMMITTEE_EVIDENCE["name"]:
        return COMMITTEE_COLUMNS
    if evidence_kind == JOINT_DIRECT_EVIDENCE["name"]:
        return JOINT_DIRECT_COLUMNS
    raise ValueError("screen report evidence kind is invalid")


def _evidence_for(evidence_kind: str) -> dict[str, object]:
    if evidence_kind == GMM_EVIDENCE["name"]:
        return dict(GMM_EVIDENCE)
    if evidence_kind == COMMITTEE_EVIDENCE["name"]:
        return dict(COMMITTEE_EVIDENCE)
    if evidence_kind == JOINT_DIRECT_EVIDENCE["name"]:
        return dict(JOINT_DIRECT_EVIDENCE)
    raise ValueError("screen report evidence kind is invalid")


def _normalize_thresholds(
    evidence_kind: str,
    value: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    if evidence_kind == GMM_EVIDENCE["name"]:
        expected = (
            ("floor", "trust", "ceiling")
            if role == "teacher"
            else ("floor", "ceiling")
        )
    elif evidence_kind == COMMITTEE_EVIDENCE["name"]:
        expected = ("lo", "hi")
    else:
        expected = ("pool",)
    if evidence_kind == GMM_EVIDENCE["name"]:
        allowed = (
            ({"floor", "trust"}, {"floor", "trust", "ceiling"})
            if role == "teacher"
            else ({"floor"}, {"floor", "ceiling"})
        )
        if set(value) not in allowed:
            raise ValueError("GMM screen report thresholds are invalid")
    elif evidence_kind == COMMITTEE_EVIDENCE["name"] and set(value) != set(expected):
        raise ValueError("committee screen report thresholds are invalid")
    elif evidence_kind == JOINT_DIRECT_EVIDENCE["name"] and set(value) != {"pool"}:
        raise ValueError("joint DIRECT screen report threshold is invalid")
    result = dict(value)
    for key in expected:
        if key not in result:
            result[key] = None
        item = result[key]
        if item is not None and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError("screen report threshold is not finite")
        if (
            evidence_kind != JOINT_DIRECT_EVIDENCE["name"]
            and item is not None
            and not 0 <= float(item) <= 1
        ):
            raise ValueError("screen report threshold is outside [0, 1]")
    if evidence_kind == GMM_EVIDENCE["name"]:
        if not 0 <= float(result["floor"]) < 1:
            raise ValueError("GMM screen report thresholds are invalid")
        if role == "teacher" and not 0 <= float(result["trust"]) <= 1:
            raise ValueError("GMM screen report thresholds are invalid")
        ceiling = result["ceiling"]
        if ceiling is not None and not float(result["floor"]) < float(ceiling) <= 1:
            raise ValueError("GMM screen report ceiling is invalid")
    elif evidence_kind == COMMITTEE_EVIDENCE["name"] and not float(result["lo"]) < float(result["hi"]):
        raise ValueError("committee screen report thresholds are invalid")
    elif evidence_kind == JOINT_DIRECT_EVIDENCE["name"] and (
        isinstance(result["pool"], bool)
        or int(result["pool"]) != result["pool"]
        or int(result["pool"]) <= 0
    ):
        raise ValueError("joint DIRECT screen report pool is invalid")
    return {key: result[key] for key in expected}


def _render_role(role: ScreenReportRole) -> list[str]:
    if role.evidence_kind == GMM_EVIDENCE["name"]:
        evidence = "GMM reference-tail"
        columns = GMM_COLUMNS
        overlap = " | route sets may overlap"
    elif role.evidence_kind == COMMITTEE_EVIDENCE["name"]:
        evidence = "committee force deviation"
        columns = COMMITTEE_COLUMNS
        overlap = ""
    else:
        evidence = "joint DIRECT prescreen"
        columns = JOINT_DIRECT_COLUMNS
        overlap = ""
    thresholds = " ".join(
        f"{key}={_threshold(value)}" for key, value in role.thresholds.items()
    )
    header = (
        f"{role.role}: {evidence} | N={role.denominator:,} | "
        f"thresholds: {thresholds}{overlap}"
    )
    rows = (*role.systems, role.total)
    display_systems = tuple(_truncate_system(row.system_id, 32) for row in rows)
    rendered_cells = tuple(
        tuple(_render_cell(row.counts[column], row.denominator) for column in columns)
        for row in rows
    )
    system_width = max(_display_width("system"), *(_display_width(value) for value in display_systems))
    widths = tuple(
        max(_display_width(column), *(_display_width(cells[index]) for cells in rendered_cells))
        for index, column in enumerate(columns)
    )
    lines = [header]
    lines.append(
        _pad_left("system", system_width)
        + "  "
        + "  ".join(_pad_right(column, width) for column, width in zip(columns, widths))
    )
    lines.extend(
        _pad_left(display, system_width)
        + "  "
        + "  ".join(_pad_right(cell, width) for cell, width in zip(cells, widths))
        for display, cells in zip(display_systems, rendered_cells)
    )
    return lines


def _render_cell(count: int, denominator: int) -> str:
    if denominator == 0:
        return f"{count:,} (n/a)"
    return f"{count:,} ({count * 100 / denominator:.2f}%)"


def _threshold(value: object) -> str:
    return "none" if value is None else f"{float(value):g}"


def _truncate_system(value: str, limit: int) -> str:
    if _display_width(value) <= limit:
        return value
    budget = limit - _display_width("…")
    left_budget = budget // 2
    right_budget = budget - left_budget
    return _take_width(value, left_budget, from_right=False) + "…" + _take_width(value, right_budget, from_right=True)


def _take_width(value: str, limit: int, *, from_right: bool) -> str:
    characters = list(reversed(value)) if from_right else list(value)
    result: list[str] = []
    width = 0
    for character in characters:
        character_width = _display_width(character)
        if width + character_width > limit:
            break
        result.append(character)
        width += character_width
    return "".join(reversed(result)) if from_right else "".join(result)


def _display_width(value: str) -> int:
    return sum(
        0
        if unicodedata.combining(character)
        else 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def _pad_left(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def _pad_right(value: str, width: int) -> str:
    return " " * max(0, width - _display_width(value)) + value


def _require_integer(value: object, label: str, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


__all__ = [
    "COMMITTEE_COLUMNS",
    "COMMITTEE_EVIDENCE",
    "GMM_COLUMNS",
    "GMM_EVIDENCE",
    "JOINT_DIRECT_COLUMNS",
    "JOINT_DIRECT_EVIDENCE",
    "SCREEN_REPORT_IDENTITY",
    "SCREEN_REPORT_SCHEMA",
    "SCREEN_REPORT_SCHEMA_REVISION",
    "ScreenReport",
    "ScreenReportRole",
    "ScreenReportRow",
    "load_screen_report",
    "publish_screen_report",
    "render_report_header",
    "render_screen_report",
    "render_screen_report_sections",
]
