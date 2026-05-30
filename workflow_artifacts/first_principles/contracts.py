from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_artifacts.first_principles.handoff import (
    first_principles_handoff_input_complete,
    first_principles_handoff_input_path_from_prepare_dir,
)
from workflow_runtime.artifacts.checks import xyz_file_complete


@dataclass(frozen=True, slots=True)
class FirstPrinciplesPrepareArtifacts:
    prepare_dir: Path
    source_input_xyz: Path
    tasks_dir: Path
    task_map_path: Path
    failed_prepare_xyz: Path


@dataclass(frozen=True, slots=True)
class FirstPrinciplesRunArtifacts:
    run_dir: Path


@dataclass(frozen=True, slots=True)
class FirstPrinciplesCollectArtifacts:
    collect_dir: Path
    labeled_xyz: Path
    failed_xyz: Path


@dataclass(frozen=True, slots=True)
class FirstPrinciplesArtifactContract:
    root: Path
    prepare: FirstPrinciplesPrepareArtifacts
    run: FirstPrinciplesRunArtifacts
    collect: FirstPrinciplesCollectArtifacts

    @classmethod
    def from_teacher_round_dir(cls, teacher_round_dir: str | Path) -> "FirstPrinciplesArtifactContract":
        return cls.from_first_principles_dir(Path(teacher_round_dir) / "first_principles")

    @classmethod
    def from_first_principles_dir(cls, first_principles_dir: str | Path) -> "FirstPrinciplesArtifactContract":
        root = Path(first_principles_dir).resolve()
        prepare_dir = root / "prepare"
        run_dir = root / "run"
        collect_dir = root / "collect"
        return cls(
            root=root,
            prepare=FirstPrinciplesPrepareArtifacts(
                prepare_dir=prepare_dir,
                source_input_xyz=first_principles_handoff_input_path_from_prepare_dir(prepare_dir),
                tasks_dir=prepare_dir / "tasks",
                task_map_path=prepare_dir / "task_map.json",
                failed_prepare_xyz=prepare_dir / "failed_prepare.xyz",
            ),
            run=FirstPrinciplesRunArtifacts(run_dir=run_dir),
            collect=FirstPrinciplesCollectArtifacts(
                collect_dir=collect_dir,
                labeled_xyz=collect_dir / "labeled.xyz",
                failed_xyz=collect_dir / "failed.xyz",
            ),
        )

    @classmethod
    def from_prepare_dir(cls, prepare_dir: str | Path) -> "FirstPrinciplesArtifactContract":
        prepare_path = Path(prepare_dir).resolve()
        return cls.from_first_principles_dir(prepare_path.parent)

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "FirstPrinciplesArtifactContract":
        run_path = Path(run_dir).resolve()
        return cls.from_first_principles_dir(run_path.parent)

    @classmethod
    def from_collect_dir(cls, collect_dir: str | Path) -> "FirstPrinciplesArtifactContract":
        collect_path = Path(collect_dir).resolve()
        return cls.from_first_principles_dir(collect_path.parent)


def prepare_complete(prepare_dir: str | Path) -> bool:
    contract = FirstPrinciplesArtifactContract.from_prepare_dir(prepare_dir)
    return first_principles_handoff_input_complete(
        contract.prepare.source_input_xyz,
        allow_empty=False,
    ) and _prepared_task_tree_complete(contract.prepare)


def run_complete(run_dir: str | Path) -> bool:
    contract = FirstPrinciplesArtifactContract.from_run_dir(run_dir)
    return _run_task_evidence_complete(contract.run)


def collect_complete(collect_dir: str | Path) -> bool:
    contract = FirstPrinciplesArtifactContract.from_collect_dir(collect_dir)
    return xyz_file_complete(contract.collect.labeled_xyz, allow_empty=True) and xyz_file_complete(
        contract.collect.failed_xyz,
        allow_empty=True,
    )


def first_principles_complete(root: str | Path) -> bool:
    contract = FirstPrinciplesArtifactContract.from_first_principles_dir(root)
    return (
        prepare_complete(contract.prepare.prepare_dir)
        and run_complete(contract.run.run_dir)
        and collect_complete(contract.collect.collect_dir)
    )


def dft_submit_complete(root: str | Path) -> bool:
    base = Path(root)
    return xyz_file_complete(base / "labeled.xyz", allow_empty=True) and xyz_file_complete(
        base / "failed.xyz",
        allow_empty=True,
    )


def _prepared_task_tree_complete(artifacts: FirstPrinciplesPrepareArtifacts) -> bool:
    if not artifacts.tasks_dir.is_dir():
        return False
    task_dirs = [path for path in artifacts.tasks_dir.iterdir() if path.is_dir()]
    if not task_dirs:
        return False
    return all(_prepared_task_dir_complete(path) for path in task_dirs)


def _prepared_task_dir_complete(task_dir: Path) -> bool:
    task_files = [path for path in task_dir.iterdir() if path.is_file()]
    if not task_files:
        return False
    return any(path.suffix.lower() in {".inp", ".xyz"} for path in task_files)


def _run_task_evidence_complete(artifacts: FirstPrinciplesRunArtifacts) -> bool:
    if not artifacts.run_dir.is_dir():
        return False
    status_files = sorted(artifacts.run_dir.rglob("task_status.json"))
    if status_files:
        return all(_task_status_success(path) for path in status_files)
    return any(_cp2k_output_success(path) for path in artifacts.run_dir.rglob("*.out"))


def _task_status_success(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status = str(payload.get("status", payload.get("state", ""))).strip().lower()
    return status in {"success", "succeeded", "complete", "completed", "done"}


def _cp2k_output_success(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    lowered = text.lower()
    return "program ended" in lowered or "cp2k ended normally" in lowered
