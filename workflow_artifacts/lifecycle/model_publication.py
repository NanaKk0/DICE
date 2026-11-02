from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from workflow_core.lifecycle.artifact_handles import ModelHandle


HistoricalModelResolver = Callable[[str, Path], Path]


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One provider-selected physical terminal for an ordered model member."""

    member_id: str
    terminal_model_path: Path
    evidence: Mapping[str, Any] = field(default_factory=dict)
    backend_asset: object | None = None

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        if not member_id:
            raise ValueError("lifecycle model artifact member_id must be non-empty")
        raw_path = Path(self.terminal_model_path)
        if raw_path.is_symlink():
            raise ValueError(
                f"lifecycle terminal model artifact must not be a symlink: {raw_path}"
            )
        path = raw_path.resolve()
        if not path.is_file() or path.stat().st_size <= 0 or not path.suffix:
            raise ValueError(
                "lifecycle terminal model must be a non-empty regular file with "
                f"a native suffix: {path}"
            )
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "terminal_model_path", path)
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class PublishedModelHandle:
    """Physical resolution of one canonical path-free model member identity.

    The historical name is retained for provider and public-import compatibility;
    lifecycle logic uses ``workflow_core.lifecycle.artifact_handles.ModelHandle``.
    """

    member_id: str
    path: Path

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        raw_path = Path(self.path)
        if raw_path.is_symlink():
            raise ValueError(
                "lifecycle published model handle must be a regular file, not a "
                f"symlink: {raw_path}"
            )
        path = raw_path.resolve()
        if not member_id:
            raise ValueError(
                "lifecycle published model handle member_id must be non-empty"
            )
        if not path.is_file() or path.stat().st_size <= 0 or not path.suffix:
            raise ValueError(
                "lifecycle published model handle must be a non-empty regular "
                f"file with a native suffix: {path}"
            )
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class ResolvedModelArtifact:
    """One path-free model identity paired with validated physical evidence."""

    handle: ModelHandle
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        physical = PublishedModelHandle(self.handle.member_id, self.path)
        digest = str(self.sha256).strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("resolved lifecycle model artifact requires a SHA-256 digest")
        if _file_digest(physical.path) != digest:
            raise ValueError("resolved lifecycle model artifact digest does not match its file")
        object.__setattr__(self, "path", physical.path)
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class ModelPublicationAdapter:
    """Resolve physical model files without exposing locations to lifecycle logic."""

    role: Literal["teacher", "student"]

    def __post_init__(self) -> None:
        if self.role not in {"teacher", "student"}:
            raise ValueError("model publication role must be teacher or student")

    def publish(
        self,
        artifacts: Sequence[ModelArtifact],
        *,
        expected_handles: Sequence[ModelHandle],
        stage_root: str | Path,
    ) -> tuple[ResolvedModelArtifact, ...]:
        expected = _validated_semantic_handles(expected_handles)
        published = publish_model_handles(
            artifacts,
            expected_member_ids=tuple(handle.member_id for handle in expected),
            stage_root=stage_root,
            role=self.role,
        )
        return _pair_semantic_and_physical(expected, published)

    def resolve(
        self,
        *,
        expected_handles: Sequence[ModelHandle],
        stage_root: str | Path,
        legacy_resolver: HistoricalModelResolver | None = None,
    ) -> tuple[ResolvedModelArtifact, ...]:
        expected = _validated_semantic_handles(expected_handles)
        resolved = resolve_model_handles(
            stage_root=stage_root,
            role=self.role,
            expected_member_ids=tuple(handle.member_id for handle in expected),
            legacy_resolver=legacy_resolver,
        )
        return _pair_semantic_and_physical(expected, resolved)


def publish_model_handles(
    artifacts: Sequence[ModelArtifact],
    *,
    expected_member_ids: Sequence[str],
    stage_root: str | Path,
    role: Literal["teacher", "student"],
) -> tuple[PublishedModelHandle, ...]:
    """Publish ordered terminals under stable role names with native suffixes."""

    expected = _validated_member_ids(expected_member_ids)
    ordered_artifacts = tuple(artifacts)
    returned = tuple(artifact.member_id for artifact in ordered_artifacts)
    if returned != expected:
        raise ValueError(
            "lifecycle model artifacts must match the ordered member identities: "
            f"expected {expected!r}, got {returned!r}"
        )
    root = Path(stage_root).resolve()
    handles: list[PublishedModelHandle] = []
    for artifact in ordered_artifacts:
        source = artifact.terminal_model_path
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "lifecycle terminal model must live inside its staged train block: "
                f"{source}"
            ) from exc
        member_root = root / artifact.member_id
        existing = tuple(member_root.glob(f"{role}_model*"))
        if existing:
            raise ValueError(
                f"lifecycle {role} member {artifact.member_id!r} already has "
                "canonical model handles"
            )
        target = member_root / f"{role}_model{source.suffix}"
        _materialize_regular_model_file(source, target)
        handles.append(PublishedModelHandle(artifact.member_id, target))
    return tuple(handles)


def publish_model_alias(native_model: str | Path, alias: str | Path) -> Path:
    """Publish one historical loader alias with hardlink-first copy fallback."""

    source = Path(native_model).resolve()
    target = Path(alias)
    if not source.is_file():
        raise ValueError(f"cannot publish missing native model {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)
    return target


def resolve_model_handles(
    *,
    stage_root: str | Path,
    role: Literal["teacher", "student"],
    expected_member_ids: Sequence[str],
    legacy_resolver: HistoricalModelResolver | None = None,
) -> tuple[PublishedModelHandle, ...]:
    """Resolve exactly one canonical native-suffix file per ordered member."""

    expected = _validated_member_ids(expected_member_ids)
    root = Path(stage_root)
    resolved: list[PublishedModelHandle] = []
    basename = f"{role}_model"
    for member_id in expected:
        member_root = root / member_id
        old_handle = member_root / basename
        old_exists = old_handle.exists() or old_handle.is_symlink()
        suffixed = (
            tuple(
                path
                for path in member_root.iterdir()
                if path.name.startswith(f"{basename}.")
            )
            if member_root.is_dir()
            else ()
        )
        if old_exists and suffixed:
            raise ValueError(
                f"ambiguous lifecycle {role} model handles for member {member_id!r}: "
                "old and native-suffix handles coexist"
            )
        if len(suffixed) > 1:
            raise ValueError(
                f"ambiguous lifecycle {role} model handles for member {member_id!r}: "
                "multiple native-suffix handles"
            )
        if old_exists:
            if legacy_resolver is None:
                raise ValueError(
                    f"legacy no-extension lifecycle {role} model handle requires "
                    f"backend-specific recovery: {old_handle}"
                )
            if not old_handle.is_file() or old_handle.stat().st_size <= 0:
                raise ValueError(
                    f"legacy no-extension lifecycle {role} model handle is missing "
                    f"or empty: {old_handle}"
                )
            recovered = Path(legacy_resolver(member_id, old_handle))
            resolved.append(PublishedModelHandle(member_id, recovered))
            continue
        if not suffixed:
            raise ValueError(
                f"missing lifecycle {role} native-suffix model handle for member "
                f"{member_id!r}"
            )
        resolved.append(PublishedModelHandle(member_id, suffixed[0]))
    return tuple(resolved)


def _validated_member_ids(member_ids: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(str(member_id).strip() for member_id in member_ids)
    if (
        not ordered
        or any(not member_id for member_id in ordered)
        or len(set(ordered)) != len(ordered)
    ):
        raise ValueError(
            "lifecycle model handles require ordered unique member identities"
        )
    return ordered


def _validated_semantic_handles(
    handles: Sequence[ModelHandle],
) -> tuple[ModelHandle, ...]:
    ordered = tuple(handles)
    if not ordered or any(not isinstance(handle, ModelHandle) for handle in ordered):
        raise TypeError("model publication requires path-free ModelHandle values")
    _validated_member_ids(tuple(handle.member_id for handle in ordered))
    model_ids = tuple(handle.model_id for handle in ordered)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model publication requires ordered unique model identities")
    return ordered


def _pair_semantic_and_physical(
    semantic: Sequence[ModelHandle],
    physical: Sequence[PublishedModelHandle],
) -> tuple[ResolvedModelArtifact, ...]:
    ordered_semantic = tuple(semantic)
    ordered_physical = tuple(physical)
    if tuple(handle.member_id for handle in ordered_semantic) != tuple(
        item.member_id for item in ordered_physical
    ):
        raise ValueError(
            "resolved lifecycle model artifacts changed ordered semantic members"
        )
    return tuple(
        ResolvedModelArtifact(
            handle=handle,
            path=item.path,
            sha256=_file_digest(item.path),
        )
        for handle, item in zip(ordered_semantic, ordered_physical, strict=True)
    )


def _materialize_regular_model_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if target.is_symlink() or not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"lifecycle model handle publication failed: {target}")
    return target


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "HistoricalModelResolver",
    "ModelArtifact",
    "ModelPublicationAdapter",
    "PublishedModelHandle",
    "ResolvedModelArtifact",
    "publish_model_alias",
    "publish_model_handles",
    "resolve_model_handles",
]
