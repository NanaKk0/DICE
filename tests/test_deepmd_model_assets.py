from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_publish_and_load_dpa4_member_bundle_exposes_distinct_roles(tmp_path: Path) -> None:
    from workflow_artifacts.lifecycle.deepmd_model_assets import (
        load_deepmd_model_bundles,
        publish_deepmd_model_bundle,
    )

    train_root = tmp_path / "iter0" / "teach" / "train"
    member_root = train_root / "teacher"
    member_root.mkdir(parents=True)
    terminal = member_root / "teacher_model.pt2"
    checkpoint = member_root / "teacher_readout_checkpoint.pt"
    terminal.write_bytes(b"frozen-terminal")
    checkpoint.write_bytes(b"selected-checkpoint")

    published = publish_deepmd_model_bundle(
        member_root=member_root,
        member_id="teacher",
        member_order=1,
        model_family="dpa4",
        native_type_map=("Pt", "H", "O"),
        inference_terminal_path=terminal,
        readout_source_path=checkpoint,
        checkpoint_family="ema",
        selected_checkpoint_argument="model.ckpt-1000_ema.pt",
        freeze_command_profile={"cli": "pt", "role": "teacher.deepmd.pt.train"},
    )

    assert published.inference_terminal == terminal.resolve()
    assert published.readout_source == checkpoint.resolve()
    receipt = member_root / "deepmd_model_assets.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_revision"] == 1
    assert payload["relationship"] == "terminal_frozen_from_checkpoint"
    assert payload["checkpoint_family"] == "ema"
    assert payload["inference_terminal_path"] == "teacher_model.pt2"
    assert payload["readout_source_path"] == "teacher_readout_checkpoint.pt"

    loaded = load_deepmd_model_bundles(
        stage_root=train_root,
        expected_member_ids=("teacher",),
        expected_native_type_maps=(("Pt", "H", "O"),),
    )
    assert loaded[0].inference_terminal == terminal.resolve()
    assert loaded[0].readout_source == checkpoint.resolve()
    assert loaded[0].checkpoint_family == "ema"


def _publish_bundle_for_validation(tmp_path: Path) -> tuple[Path, Path, Path]:
    from workflow_artifacts.lifecycle.deepmd_model_assets import publish_deepmd_model_bundle

    root = tmp_path / "iter0" / "teach" / "train" / "teacher"
    root.mkdir(parents=True)
    terminal = root / "teacher_model.pt2"
    checkpoint = root / "teacher_readout_checkpoint.pt"
    terminal.write_bytes(b"terminal")
    checkpoint.write_bytes(b"checkpoint")
    publish_deepmd_model_bundle(
        member_root=root,
        member_id="teacher",
        member_order=1,
        model_family="dpa4",
        native_type_map=("Pt", "H", "O"),
        inference_terminal_path=terminal,
        readout_source_path=checkpoint,
        checkpoint_family="regular",
        selected_checkpoint_argument="model.ckpt.pt",
        freeze_command_profile={"role": "teacher.deepmd.pt.train"},
    )
    return root, terminal, checkpoint


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing or invalid"),
        ("empty", "wrong suffix or is empty"),
        ("symlink", "regular file"),
        ("escaped", "ambiguous or non-canonical"),
        ("wrong_suffix", "ambiguous or non-canonical"),
        ("ambiguous", "ambiguous or non-canonical"),
        ("digest", "digest drifted"),
        ("family", "not ordinary DPA4"),
        ("order", "member order changed"),
        ("type_map", "native type map changed"),
    ],
)
def test_dpa4_bundle_loader_fails_closed_for_asset_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from workflow_artifacts.lifecycle.deepmd_model_assets import load_deepmd_model_bundles

    root, terminal, checkpoint = _publish_bundle_for_validation(tmp_path)
    manifest = root / "deepmd_model_assets.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected_maps = (("Pt", "H", "O"),)
    if mutation == "missing":
        checkpoint.unlink()
    elif mutation == "empty":
        checkpoint.write_bytes(b"")
    elif mutation == "symlink":
        checkpoint.unlink()
        outside = tmp_path / "outside.pt"
        outside.write_bytes(b"outside")
        checkpoint.symlink_to(outside)
    elif mutation == "escaped":
        payload["readout_source_path"] = "../outside.pt"
    elif mutation == "wrong_suffix":
        payload["inference_terminal_path"] = "teacher_model.pb"
    elif mutation == "ambiguous":
        payload["readout_source_path"] = "teacher_readout_checkpoint.copy.pt"
    elif mutation == "digest":
        terminal.write_bytes(b"changed")
    elif mutation == "family":
        payload["model_family"] = "dpa4c"
    elif mutation == "order":
        payload["member_order"] = 2
    elif mutation == "type_map":
        payload["native_type_map"] = ["H", "Pt", "O"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_deepmd_model_bundles(
            stage_root=root.parent,
            expected_member_ids=("teacher",),
            expected_native_type_maps=expected_maps,
        )


def test_dpa4_bundle_publication_rejects_dpa4c_and_symlinked_canonical_inputs(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.deepmd_model_assets import publish_deepmd_model_bundle

    root = tmp_path / "teacher"
    root.mkdir()
    terminal = root / "teacher_model.pt2"
    checkpoint = root / "teacher_readout_checkpoint.pt"
    outside_terminal = tmp_path / "outside.pt2"
    outside_terminal.write_bytes(b"terminal")
    terminal.symlink_to(outside_terminal)
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="regular file"):
        publish_deepmd_model_bundle(
            member_root=root,
            member_id="teacher",
            member_order=1,
            model_family="dpa4",
            native_type_map=("H",),
            inference_terminal_path=terminal,
            readout_source_path=checkpoint,
            checkpoint_family="regular",
            selected_checkpoint_argument="model.ckpt.pt",
            freeze_command_profile={"role": "teacher.deepmd.pt.train"},
        )

    terminal.unlink()
    terminal.write_bytes(b"terminal")
    with pytest.raises(ValueError, match="only valid for ordinary DPA4"):
        publish_deepmd_model_bundle(
            member_root=root,
            member_id="teacher",
            member_order=1,
            model_family="dpa4c",
            native_type_map=("H",),
            inference_terminal_path=terminal,
            readout_source_path=checkpoint,
            checkpoint_family="regular",
            selected_checkpoint_argument="model.ckpt.pt",
            freeze_command_profile={"role": "teacher.deepmd.pt.train"},
        )


def _publish_completed_dpa4_train(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from workflow_backends.teacher.deepmd.native import publish_terminal_digest_metadata
    from workflow_core.lifecycle.model_handles import ModelArtifact, publish_model_handles
    from workflow_artifacts.lifecycle.deepmd_model_assets import publish_deepmd_model_bundle

    stage = tmp_path / "iter0" / "teach" / "train"
    member = stage / "teacher"
    member.mkdir(parents=True)
    (member / "native.pt2").write_bytes(b"terminal")
    (member / "teacher_readout_checkpoint.pt").write_bytes(b"readout")
    (member / "input.json").write_text(
        json.dumps({"model": {"descriptor": {"type": "dpa4"}, "type_map": ["Pt"]}}),
        encoding="utf-8",
    )
    (member / "deepmd_model_metadata.json").write_text(
        json.dumps(
            {
                "model_family": "dpa4",
                "type_map": ["Pt"],
                "native_model_suffix": ".pt2",
            }
        ),
        encoding="utf-8",
    )
    handle = publish_model_handles(
        (ModelArtifact("teacher", member / "native.pt2"),),
        expected_member_ids=("teacher",),
        stage_root=stage,
        role="teacher",
    )
    publish_terminal_digest_metadata(handle)
    publish_deepmd_model_bundle(
        member_root=member,
        member_id="teacher",
        member_order=1,
        model_family="dpa4",
        native_type_map=("Pt",),
        inference_terminal_path=handle[0].path,
        readout_source_path=member / "teacher_readout_checkpoint.pt",
        checkpoint_family="ema",
        selected_checkpoint_argument="model_ema.ckpt.pt",
        freeze_command_profile={"role": "teacher.deepmd.pt.train", "cli": "pt"},
    )
    spec = {
        "backend": "DeepMD",
        "cli": "pt",
        "model": "single",
        "type_map": ["Pt"],
        "recipe": "teacher.json",
    }
    (tmp_path / "teacher.json").write_text(
        json.dumps({"model": {"descriptor": {"type": "dpa4"}}}),
        encoding="utf-8",
    )
    return stage, spec


def test_dpa4_completed_prefix_requires_bundle_without_downstream_fingerprint_chain(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.adapter import DeepMDTeacherBackendFactory
    from workflow_backends.teacher.role import TeacherBackendBinding
    from workflow_core.lifecycle.data import LabelContract
    from workflow_core.lifecycle.teacher_facade import LifecycleTeacherFacade

    stage, spec = _publish_completed_dpa4_train(tmp_path)

    teacher = LifecycleTeacherFacade(
        DeepMDTeacherBackendFactory(None, cli="pt").bind(  # type: ignore[arg-type]
            TeacherBackendBinding(
                run_json_path=tmp_path / "run.json",
                iter_index=0,
                recipe_path=tmp_path / "teacher.json",
                type_map=("Pt",),
                model_shape="single",
                member_ids=("teacher",),
                policy="from_scratch",
                foundation_model=None,
                train_options={},
                requires_readout=True,
            )
        ),
        LabelContract(),
    )

    assert teacher.validate_published_training(
        stage_root=stage, member_ids=("teacher",)
    )

    class Config:
        root = tmp_path
        iter_index = 0
        round_count = 0
        raw = {
            "teacher": spec,
            "student": {
                "backend": "NEP",
                "model": "single",
                "type_map": ["Pt"],
            },
        }

    first = "base"
    checkpoint = stage / "teacher" / "teacher_readout_checkpoint.pt"
    checkpoint.write_bytes(b"changed-readout")
    assert not teacher.validate_published_training(
        stage_root=stage, member_ids=("teacher",)
    )
    second = "base"
    assert first == second


def test_dpa4_gmm_identity_digest_binds_terminal_checkpoint_and_receipt(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.adapter import DeepMDTeacherAdapter
    from workflow_backends.teacher.readout import TeacherReferenceFitRequest
    from workflow_artifacts.lifecycle.deepmd_model_assets import file_sha256
    from workflow_policies.gmm_reference_tail import GMMReferenceFitResult

    class Executor:
        model_digest: str | None = None

        def fit_gmm_router(self, _model, **request):
            self.model_digest = request["model_digest"]
            return GMMReferenceFitResult(
                router_dir=request["router_dir"],
                policy_id="gmm-reference-tail",
                element_pca_dimensions={},
            )

    stage, _spec = _publish_completed_dpa4_train(tmp_path)
    terminal = stage / "teacher" / "teacher_model.pt2"
    executor = Executor()
    adapter = DeepMDTeacherAdapter(
        executor=executor,  # type: ignore[arg-type]
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        cli="pt",
        work_root=stage,
    )

    reference = tmp_path / "reference.xyz"
    reference.write_bytes(b"reference")
    request = TeacherReferenceFitRequest(
        model=terminal,
        reference_path=reference,
        router_dir=tmp_path / "router",
        work_dir=tmp_path / "work",
        type_map=("Pt",),
    )
    adapter.fit_reference(request)

    assert executor.model_digest != file_sha256(terminal)
    (stage / "teacher" / "teacher_readout_checkpoint.pt").write_bytes(
        b"changed-again"
    )
    with pytest.raises(ValueError, match="digest drifted"):
        adapter.fit_reference(request)
