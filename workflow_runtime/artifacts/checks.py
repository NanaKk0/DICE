from __future__ import annotations

from pathlib import Path
from typing import Iterable


def nonempty_file_complete(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0


def xyz_file_complete(path: str | Path, *, allow_empty: bool = False) -> bool:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return False
    try:
        from workflow_core.structures.formats import validate_extxyz_file

        return validate_extxyz_file(file_path, allow_empty=allow_empty)
    except ModuleNotFoundError:
        if file_path.stat().st_size == 0:
            return bool(allow_empty)
        return _basic_extxyz_complete(file_path)
    except Exception:
        return False


def all_xyz_complete(paths: Iterable[str | Path], *, allow_empty: bool = False) -> bool:
    return all(xyz_file_complete(path, allow_empty=allow_empty) for path in paths)


def model_artifacts_complete(root: str | Path, *, minimum_count: int = 1) -> bool:
    from workflow_core.artifact_contracts import model_artifacts_complete as check

    return check(root, minimum_count=minimum_count)


def student_stage_complete(stage_type: str, stage_dir: str | Path) -> bool:
    from workflow_runtime.artifacts.registry import artifact_check_for

    stage = str(stage_type).strip()
    try:
        return artifact_check_for(f"student_stage:{stage}", stage_dir)()
    except ValueError as exc:
        raise ValueError(f"Unsupported student stage artifact check: {stage_type!r}") from exc


def student_round_complete(
    student_round_dir: str | Path,
    planned_stage_types: Iterable[str] | None = None,
) -> bool:
    from workflow_core.artifact_contracts import student_round_complete as check

    return check(student_round_dir, planned_stage_types=planned_stage_types)


def student_subworkflow_complete(root: str | Path) -> bool:
    from workflow_core.artifact_contracts import student_subworkflow_complete as check

    return check(root)


def teacher_foundation_train_complete(root: str | Path) -> bool:
    from workflow_core.artifact_contracts import teacher_foundation_train_complete as check

    return check(root)


def teacher_committee_finetune_complete(root: str | Path, *, teacher_count: int = 1) -> bool:
    from workflow_core.artifact_contracts import teacher_committee_finetune_complete as check

    return check(root, teacher_count=teacher_count)


def first_principles_prepare_complete(prepare_dir: str | Path) -> bool:
    from workflow_core.artifact_contracts import first_principles_prepare_complete as check

    return check(prepare_dir)


def first_principles_run_complete(run_dir: str | Path) -> bool:
    from workflow_core.artifact_contracts import first_principles_run_complete as check

    return check(run_dir)


def first_principles_collect_complete(collect_dir: str | Path) -> bool:
    from workflow_core.artifact_contracts import first_principles_collect_complete as check

    return check(collect_dir)


def first_principles_complete(root: str | Path) -> bool:
    from workflow_core.artifact_contracts import first_principles_complete as check

    return check(root)


def dft_submit_complete(root: str | Path) -> bool:
    from workflow_core.artifact_contracts import dft_submit_complete as check

    return check(root)


def status_publish_complete(root: str | Path) -> bool:
    del root
    raise ValueError("status_publish is retired in marker-only target mode")


def _basic_extxyz_complete(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        found_frame = False
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue
            atom_count = int(lines[index].strip())
            if atom_count < 0:
                return False
            if index + 1 + atom_count >= len(lines):
                return False
            for row in lines[index + 2 : index + 2 + atom_count]:
                fields = row.split()
                if len(fields) < 4:
                    return False
                float(fields[1])
                float(fields[2])
                float(fields[3])
            found_frame = True
            index += atom_count + 2
        return found_frame
    except Exception:
        return False
