from __future__ import annotations

from pathlib import Path
import os
from unittest.mock import patch

import pytest


def test_model_artifact_preserves_native_path_and_rejects_invalid_terminals(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import ModelArtifact

    terminal = tmp_path / "teacher_stagetwo.model"
    terminal.write_bytes(b"mace-stage-two")

    artifact = ModelArtifact(
        member_id="teacher_01",
        terminal_model_path=terminal,
        evidence={"native_phase": "stage_two"},
    )

    assert artifact.member_id == "teacher_01"
    assert artifact.terminal_model_path == terminal.resolve()
    assert artifact.evidence == {"native_phase": "stage_two"}
    assert not hasattr(artifact, "native_suffix")
    assert not hasattr(artifact, "model_family")

    invalid_paths = {
        "missing": tmp_path / "missing.model",
        "empty": tmp_path / "empty.model",
        "suffixless": tmp_path / "terminal",
        "directory": tmp_path / "package.model",
    }
    invalid_paths["empty"].touch()
    invalid_paths["suffixless"].write_bytes(b"model")
    invalid_paths["directory"].mkdir()

    for case, path in invalid_paths.items():
        with pytest.raises(ValueError, match="terminal model"):
            ModelArtifact("teacher", path, {"case": case})

    with pytest.raises(ValueError, match="member_id"):
        ModelArtifact("", terminal)


def test_publish_model_handles_preserves_configured_order_and_native_suffix(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import (
        ModelArtifact,
        publish_model_handles,
    )

    stage = tmp_path / "teach" / "train"
    terminals = []
    for member_id in ("teacher_01", "teacher_02"):
        terminal = stage / member_id / f"{member_id}_stagetwo.model"
        terminal.parent.mkdir(parents=True, exist_ok=True)
        terminal.write_text(member_id, encoding="utf-8")
        terminals.append(terminal)
    artifacts = tuple(
        ModelArtifact(member_id, terminal, {"native_phase": "stage_two"})
        for member_id, terminal in zip(("teacher_01", "teacher_02"), terminals)
    )

    handles = publish_model_handles(
        artifacts,
        expected_member_ids=("teacher_01", "teacher_02"),
        stage_root=stage,
        role="teacher",
    )

    assert tuple(handle.member_id for handle in handles) == (
        "teacher_01",
        "teacher_02",
    )
    assert tuple(handle.path.name for handle in handles) == (
        "teacher_model.model",
        "teacher_model.model",
    )
    for handle, terminal in zip(handles, terminals):
        assert handle.path.is_file()
        assert not handle.path.is_symlink()
        assert os.path.samefile(handle.path, terminal)

    with pytest.raises(ValueError, match="ordered member"):
        publish_model_handles(
            tuple(reversed(artifacts)),
            expected_member_ids=("teacher_01", "teacher_02"),
            stage_root=stage,
            role="teacher",
        )
    with pytest.raises(ValueError, match="ordered member"):
        publish_model_handles(
            artifacts[:1],
            expected_member_ids=("teacher_01", "teacher_02"),
            stage_root=stage,
            role="teacher",
        )
    with pytest.raises(ValueError, match="unique member"):
        publish_model_handles(
            artifacts,
            expected_member_ids=("teacher_01", "teacher_01"),
            stage_root=stage,
            role="teacher",
        )


def test_publish_model_handles_copies_unknown_suffix_when_hardlink_is_unavailable(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import (
        ModelArtifact,
        publish_model_handles,
    )

    stage = tmp_path / "r0" / "train"
    terminal = stage / "student" / "native.futurefmt"
    terminal.parent.mkdir(parents=True)
    terminal.write_bytes(b"future model bytes")

    with patch(
        "workflow_core.lifecycle.model_handles.os.link",
        side_effect=OSError("cross-device link"),
    ):
        handles = publish_model_handles(
            (ModelArtifact("student", terminal),),
            expected_member_ids=("student",),
            stage_root=stage,
            role="student",
        )

    handle = handles[0]
    assert handle.path.name == "student_model.futurefmt"
    assert handle.path.read_bytes() == terminal.read_bytes()
    assert handle.path.is_file()
    assert not handle.path.is_symlink()
    assert not os.path.samefile(handle.path, terminal)


def test_teacher_handle_publication_has_no_parallel_descriptor_asset_contract(
    tmp_path: Path,
) -> None:
    import workflow_core.lifecycle.model_handles as handles

    stage = tmp_path / "iter0" / "teach" / "train"
    member = stage / "teacher"
    member.mkdir(parents=True)
    terminal = member / "native.pt2"
    terminal.write_bytes(b"terminal")

    published = handles.publish_model_handles(
        (handles.ModelArtifact("teacher", terminal),),
        expected_member_ids=("teacher",),
        stage_root=stage,
        role="teacher",
    )

    assert published[0].path.name == "teacher_model.pt2"
    assert not (stage / "teacher_model_assets.json").exists()
    assert not (member / "teacher_descriptor_checkpoint.pt").exists()
    assert not hasattr(handles, "DescriptorModelSidecar")
    assert not hasattr(handles, "PublishedDescriptorSidecar")
    assert not hasattr(handles, "resolve_descriptor_sidecars")
    assert not hasattr(handles, "descriptor_sidecar_fingerprint_evidence")


def test_resolve_model_handles_is_ordered_and_rejects_mixed_or_multiple_handles(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import resolve_model_handles

    stage = tmp_path / "teach" / "train"
    for member_id in ("teacher_01", "teacher_02"):
        member = stage / member_id
        member.mkdir(parents=True)
        (member / "teacher_model.model").write_text(member_id, encoding="utf-8")

    resolved = resolve_model_handles(
        stage_root=stage,
        role="teacher",
        expected_member_ids=("teacher_02", "teacher_01"),
    )
    assert tuple(handle.member_id for handle in resolved) == (
        "teacher_02",
        "teacher_01",
    )
    assert tuple(handle.path.parent.name for handle in resolved) == (
        "teacher_02",
        "teacher_01",
    )

    (stage / "teacher_01" / "teacher_model").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model_handles(
            stage_root=stage,
            role="teacher",
            expected_member_ids=("teacher_01", "teacher_02"),
        )

    (stage / "teacher_01" / "teacher_model").unlink()
    (stage / "teacher_01" / "teacher_model.other").write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model_handles(
            stage_root=stage,
            role="teacher",
            expected_member_ids=("teacher_01", "teacher_02"),
        )


def test_resolver_rejects_loader_facing_symlink_handle(tmp_path: Path) -> None:
    from workflow_core.lifecycle.model_handles import resolve_model_handles

    stage = tmp_path / "teach" / "train"
    member = stage / "teacher"
    member.mkdir(parents=True)
    native = member / "native.model"
    native.write_bytes(b"native")
    canonical = member / "teacher_model.model"
    canonical.symlink_to(native)

    with pytest.raises(ValueError, match="regular file"):
        resolve_model_handles(
            stage_root=stage,
            role="teacher",
            expected_member_ids=("teacher",),
        )


def test_resolver_rejects_missing_and_empty_handles(tmp_path: Path) -> None:
    from workflow_core.lifecycle.model_handles import resolve_model_handles

    stage = tmp_path / "r0" / "train"
    member = stage / "student"
    member.mkdir(parents=True)
    with pytest.raises(ValueError, match="missing"):
        resolve_model_handles(
            stage_root=stage,
            role="student",
            expected_member_ids=("student",),
        )

    (member / "student_model.pb").touch()
    with pytest.raises(ValueError, match="non-empty regular file"):
        resolve_model_handles(
            stage_root=stage,
            role="student",
            expected_member_ids=("student",),
        )


def test_fresh_suffixed_handle_never_invokes_historical_recovery(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import resolve_model_handles

    stage = tmp_path / "r0" / "train"
    member = stage / "student"
    member.mkdir(parents=True)
    terminal = member / "student_model.pb"
    terminal.write_bytes(b"fresh")

    def reject_fallback(_member_id: str, _legacy_handle: Path) -> Path:
        raise AssertionError("fresh model publication must not scan historical forms")

    resolved = resolve_model_handles(
        stage_root=stage,
        role="student",
        expected_member_ids=("student",),
        legacy_resolver=reject_fallback,
    )

    assert tuple(handle.path for handle in resolved) == (terminal.resolve(),)

def test_resolver_reads_legacy_handle_only_through_backend_evidence_without_mutation(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import resolve_model_handles

    stage = tmp_path / "teach" / "train"
    member = stage / "teacher"
    member.mkdir(parents=True)
    old_handle = member / "teacher_model"
    native_terminal = member / "teacher_stagetwo.model"
    old_handle.write_bytes(b"stage-two")
    os.link(old_handle, native_terminal)
    before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in (old_handle, native_terminal)
    }
    calls: list[tuple[str, Path]] = []

    def recover(member_id: str, legacy_handle: Path) -> Path:
        calls.append((member_id, legacy_handle))
        return native_terminal

    resolved = resolve_model_handles(
        stage_root=stage,
        role="teacher",
        expected_member_ids=("teacher",),
        legacy_resolver=recover,
    )

    assert resolved[0].path == native_terminal.resolve()
    assert calls == [("teacher", old_handle)]
    assert not (member / "teacher_model.model").exists()
    assert before == {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in (old_handle, native_terminal)
    }


def test_resolver_rejects_legacy_handle_without_or_with_invalid_backend_evidence(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.model_handles import resolve_model_handles

    stage = tmp_path / "r0" / "train"
    member = stage / "student"
    member.mkdir(parents=True)
    old_handle = member / "student_model"
    old_handle.write_bytes(b"model")

    with pytest.raises(ValueError, match="backend-specific recovery"):
        resolve_model_handles(
            stage_root=stage,
            role="student",
            expected_member_ids=("student",),
        )
    with pytest.raises(ValueError, match="native suffix"):
        resolve_model_handles(
            stage_root=stage,
            role="student",
            expected_member_ids=("student",),
            legacy_resolver=lambda _member_id, legacy: legacy,
        )
