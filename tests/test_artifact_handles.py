from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
from pathlib import Path

import pytest


def test_shared_handles_are_minimal_path_free_ordered_values() -> None:
    import workflow_core.lifecycle.artifact_handles as contracts
    from workflow_core.lifecycle.artifact_handles import (
        DatasetHandle,
        ModelHandle,
        StructureBatchHandle,
    )

    dataset = DatasetHandle(" dataset-0 ", ("structure-2", "structure-0"))
    model = ModelHandle(" member-02 ", " model-02 ")
    batch = StructureBatchHandle(" selected ", ("structure-0", "structure-2"))

    assert dataset == DatasetHandle("dataset-0", ("structure-2", "structure-0"))
    assert model == ModelHandle("member-02", "model-02")
    assert batch == StructureBatchHandle(
        "selected", ("structure-0", "structure-2")
    )
    assert StructureBatchHandle("empty-selection", ()).ordered_structure_ids == ()

    source = inspect.getsource(contracts)
    for forbidden_dependency in (
        "from pathlib import Path",
        "workflow_artifacts",
        "workflow_backends",
        "workflow_runtime",
    ):
        assert forbidden_dependency not in source

    for value_type in (DatasetHandle, ModelHandle, StructureBatchHandle):
        assert not {item.name for item in fields(value_type)} & {
            "path",
            "directory",
            "work_dir",
            "recipe",
            "command",
            "executor",
            "receipt",
            "suffix",
            "digest",
        }

    for value in (
        lambda: DatasetHandle("dataset", ()),
        lambda: DatasetHandle("dataset", ("same", "same")),
        lambda: ModelHandle("", "model"),
        lambda: StructureBatchHandle("batch", ("same", "same")),
    ):
        with pytest.raises(ValueError):
            value()


def test_teacher_public_handles_alias_or_compose_the_shared_values() -> None:
    from workflow_core.lifecycle.artifact_handles import (
        ModelHandle,
        StructureBatchHandle,
    )
    from workflow_core.lifecycle.teacher_capabilities import (
        TeacherDatasetHandle,
        TeacherModelHandle,
        TeacherStructureBatchHandle,
    )

    assert TeacherModelHandle is ModelHandle
    assert TeacherStructureBatchHandle is StructureBatchHandle

    dataset = TeacherDatasetHandle(
        dataset_id="teacher-dataset",
        train_structure_ids=("train-1", "train-0"),
        validation_structure_ids=("valid-0",),
        validation_mode="holdout",
    )
    assert dataset.train_structure_ids == ("train-1", "train-0")
    assert dataset.validation_structure_ids == ("valid-0",)


def test_artifact_owned_model_round_trip_preserves_order_suffix_bytes_and_digest(
    tmp_path: Path,
) -> None:
    import workflow_core.lifecycle.model_handles as compatibility
    from workflow_artifacts.lifecycle import model_publication
    from workflow_core.lifecycle.artifact_handles import ModelHandle

    stage = tmp_path / "iter0" / "teach" / "train"
    member_ids = ("teacher_02", "teacher_01")
    terminals = []
    for index, member_id in enumerate(member_ids):
        terminal = stage / member_id / f"native-{index}.model"
        terminal.parent.mkdir(parents=True)
        terminal.write_bytes(f"model-bytes-{index}".encode())
        terminals.append(terminal)

    artifacts = tuple(
        model_publication.ModelArtifact(member_id, terminal)
        for member_id, terminal in zip(member_ids, terminals, strict=True)
    )
    expected = tuple(
        ModelHandle(member_id, f"teacher-model-{member_id[-2:]}")
        for member_id in member_ids
    )
    adapter = model_publication.ModelPublicationAdapter("teacher")
    published = adapter.publish(
        artifacts,
        expected_handles=expected,
        stage_root=stage,
    )
    resolved = adapter.resolve(
        stage_root=stage,
        expected_handles=expected,
    )
    assert compatibility.ModelArtifact is model_publication.ModelArtifact
    assert compatibility.PublishedModelHandle is model_publication.PublishedModelHandle
    assert tuple(item.handle for item in published) == expected
    assert resolved == published
    assert tuple(item.path.name for item in resolved) == (
        "teacher_model.model",
        "teacher_model.model",
    )
    assert [item.handle.member_id for item in resolved] == list(member_ids)
    assert [item.sha256 for item in resolved] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in terminals
    ]

    compatibility_resolved = compatibility.resolve_model_handles(
        stage_root=stage,
        role="teacher",
        expected_member_ids=member_ids,
    )
    assert tuple(item.path for item in compatibility_resolved) == tuple(
        item.path for item in resolved
    )

    compatibility_source = inspect.getsource(compatibility)
    for physical_implementation in (
        "from pathlib import Path",
        "hashlib",
        "shutil",
        ".is_file(",
        ".read_bytes(",
        ".write_bytes(",
    ):
        assert physical_implementation not in compatibility_source
