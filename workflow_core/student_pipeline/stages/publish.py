from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from workflow_core.student_pipeline.artifacts.stage_contracts import (
    StagePathRole,
    StudentStageContractDescriptor,
    student_stage_contract_descriptor,
)
from workflow_core.student_pipeline.runtime.commit_plan import ArtifactPublication, StagePublishPlan
from workflow_core.structures.formats import copy_extxyz_file


COMMIT_OWNED_STAGE_MANIFEST_FIELDS = frozenset(
    {
        "teacher_round",
        "student_round",
        "stage_run_index",
        "stage",
        "config",
        "completed",
    }
)


def stage_publish_staging_dir(stage_dir: str | Path) -> Path:
    return (Path(stage_dir).resolve() / ".staging" / "publish").resolve()


def build_stage_file_publications(
    stage_type: str,
    staging_dir: str | Path,
    target_root: str | Path,
) -> tuple[ArtifactPublication, ...]:
    descriptor = student_stage_contract_descriptor(stage_type)
    staging_root = Path(staging_dir).resolve()
    target_root_path = Path(target_root).resolve()
    return _build_stage_file_publications(
        descriptor,
        staging_root,
        target_root_path,
        validate_extxyz_roles=None,
    )


def prepare_stage_result_publications(
    stage_type: str,
    *,
    staging_dir: str | Path,
    target_root: str | Path,
    stage_result: Mapping[str, Any],
    role_source_fields: Mapping[str, str],
    extxyz_roles: tuple[str, ...] | None = None,
) -> tuple[ArtifactPublication, ...]:
    role_sources = {
        role: stage_result.get(source_field)
        for role, source_field in dict(role_source_fields or {}).items()
    }
    return prepare_stage_output_publications(
        stage_type,
        staging_dir=staging_dir,
        target_root=target_root,
        role_sources=role_sources,
        extxyz_roles=extxyz_roles,
    )


def prepare_stage_output_publications(
    stage_type: str,
    *,
    staging_dir: str | Path,
    target_root: str | Path,
    role_sources: Mapping[str, Any],
    extxyz_roles: tuple[str, ...] | None = None,
) -> tuple[ArtifactPublication, ...]:
    descriptor = student_stage_contract_descriptor(stage_type)
    staging_root = Path(staging_dir).resolve()
    roles_to_copy_as_extxyz = _extxyz_roles(descriptor, extxyz_roles)
    for role, source in dict(role_sources or {}).items():
        path_role = _published_output_role(descriptor, role)
        target = (staging_root / path_role.relative_path).resolve()
        _stage_output(source, target, copy_as_extxyz=role in roles_to_copy_as_extxyz, allow_empty=path_role.allow_empty_extxyz)
    return _build_stage_file_publications(
        descriptor,
        staging_root,
        Path(target_root).resolve(),
        validate_extxyz_roles=roles_to_copy_as_extxyz,
    )


def build_backend_log_publications(
    staging_dir: str | Path,
    backend_log_dir: str | Path,
) -> tuple[ArtifactPublication, ...]:
    backend_log_root = Path(staging_dir).resolve() / "backend_logs"
    if not backend_log_root.exists():
        return ()
    return tuple(
        ArtifactPublication(
            source,
            Path(backend_log_dir).resolve() / source.relative_to(backend_log_root),
            validate_extxyz=False,
        )
        for source in sorted(path for path in backend_log_root.rglob("*") if path.is_file())
    )


def _build_stage_file_publications(
    descriptor: StudentStageContractDescriptor,
    staging_root: Path,
    target_root_path: Path,
    *,
    validate_extxyz_roles: set[str] | None,
) -> tuple[ArtifactPublication, ...]:
    publications: list[ArtifactPublication] = []
    seen_targets: set[Path] = set()
    for role in descriptor.published_output_roles:
        for publication in _publications_for_role(
            descriptor,
            role,
            staging_root,
            target_root_path,
            validate_extxyz_roles=validate_extxyz_roles,
        ):
            if publication.target in seen_targets:
                continue
            seen_targets.add(publication.target)
            publications.append(publication)
    return tuple(publications)


def _publication_for_role(
    descriptor: StudentStageContractDescriptor,
    role: str,
    staging_root: Path,
    target_root_path: Path,
    *,
    validate_extxyz_roles: set[str] | None,
) -> ArtifactPublication | None:
    target = descriptor.published_output_path(role, target_root_path)
    source = _staged_output_path(descriptor, role, staging_root)
    if source is None:
        return None
    allow_empty = role in descriptor.allow_empty_extxyz_roles
    if validate_extxyz_roles is None:
        validate_extxyz = source.suffix.lower() == ".xyz"
    else:
        validate_extxyz = role in validate_extxyz_roles
    return ArtifactPublication(source, target, allow_empty=allow_empty, validate_extxyz=validate_extxyz)


def _publications_for_role(
    descriptor: StudentStageContractDescriptor,
    role: str,
    staging_root: Path,
    target_root_path: Path,
    *,
    validate_extxyz_roles: set[str] | None,
) -> list[ArtifactPublication]:
    publication = _publication_for_role(
        descriptor,
        role,
        staging_root,
        target_root_path,
        validate_extxyz_roles=validate_extxyz_roles,
    )
    if publication is None:
        return []
    source = publication.source
    if source.is_dir():
        return [
            ArtifactPublication(
                file_path,
                publication.target / file_path.relative_to(source),
                allow_empty=publication.allow_empty,
                validate_extxyz=publication.validate_extxyz and file_path.suffix.lower() == ".xyz",
            )
            for file_path in sorted(path for path in source.rglob("*") if path.is_file())
        ]
    return [publication]


def _staged_output_path(descriptor: StudentStageContractDescriptor, role: str, staging_root: Path) -> Path | None:
    for item in descriptor.published_outputs:
        if item.role != role:
            continue
        return (staging_root / item.relative_path).resolve()
    return None


def _published_output_role(descriptor: StudentStageContractDescriptor, role: str) -> StagePathRole:
    requested = str(role)
    for item in descriptor.published_outputs:
        if item.role == requested:
            return item
    raise ValueError(f"unknown published output role {requested!r} for student stage {descriptor.stage_type!r}")


def _extxyz_roles(
    descriptor: StudentStageContractDescriptor,
    extxyz_roles: tuple[str, ...] | None,
) -> set[str]:
    if extxyz_roles is not None:
        return {str(role) for role in extxyz_roles}
    return {
        role.role
        for role in descriptor.published_outputs
        if role.relative_path.suffix.lower() == ".xyz"
    }


def _stage_output(source: Any, target: Path, *, copy_as_extxyz: bool, allow_empty: bool) -> Path:
    target_path = Path(target).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source in (None, ""):
        if target_path.suffix:
            if not target_path.exists():
                target_path.write_text("", encoding="utf-8")
        else:
            target_path.mkdir(parents=True, exist_ok=True)
        return target_path
    source_path = Path(str(source)).resolve()
    if not source_path.exists():
        if not target_path.exists():
            target_path.write_text("", encoding="utf-8")
        return target_path
    if source_path == target_path:
        return target_path
    if source_path.is_dir():
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        return target_path
    if copy_as_extxyz:
        copy_extxyz_file(source_path, target_path, allow_empty=allow_empty)
    else:
        shutil.copy2(source_path, target_path)
    return target_path


def build_stage_publish_plan(
    *,
    student_round: int,
    stage_run_index: int,
    artifact_publications: tuple[ArtifactPublication, ...] = (),
    backend_log_publications: tuple[ArtifactPublication, ...] = (),
    stage_payload: dict | None = None,
) -> StagePublishPlan:
    return StagePublishPlan(
        student_round=student_round,
        stage_run_index=stage_run_index,
        artifact_publications=tuple(artifact_publications),
        backend_log_publications=tuple(backend_log_publications),
        stage_payload=stage_payload_without_commit_fields(stage_payload),
    )


def stage_payload_without_commit_fields(payload: dict | None) -> dict:
    return {key: value for key, value in dict(payload or {}).items() if key not in COMMIT_OWNED_STAGE_MANIFEST_FIELDS}
