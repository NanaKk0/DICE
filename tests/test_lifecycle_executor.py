from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import hashlib
import json
import os
import shlex
import sys

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from lifecycle_role_fakes import compose_test_backend_roles


class _FakeTeacher:
    def __init__(self):
        from workflow_runtime.lifecycle.environment import OperationContext

        self.train_context = OperationContext(
            role="teacher.mace.train",
            profile_id="fake",
            source_scripts=(),
            executables={"mace_run_train": "fake"},
            runtime={},
        )

    def train(
        self,
        rendered,
        work_dir,
        *,
        progress_reporter=None,
        member_index=1,
        total_members=1,
    ):
        import yaml

        payload = yaml.safe_load(Path(rendered).read_text(encoding="utf-8"))
        suffix = "_stagetwo.model" if payload.get("swa") is True else ".model"
        model = Path(payload["model_dir"]) / f"{payload['name']}{suffix}"
        model.write_text("teacher", encoding="utf-8")
        return model

    def label(self, models, frames, *args, **kwargs):
        result = []
        for frame in frames:
            item = frame.copy()
            item.calc = SinglePointCalculator(item, energy=0.0, forces=np.zeros((len(item), 3)))
            item.info["teacher_p"] = 1.0
            if len(models) > 1:
                item.info["force_uncertainty"] = 0.0
            result.append(item)
        return result

    def fit_gmm_router(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import (
            GMMReferenceStreamFitRequest,
            fit_gmm_router_stream,
        )

        frames = read_extxyz_structures(request["reference_path"], allow_empty=False)
        header, chunks = _fake_descriptor_stream(
            frames, width=128, backend="mace", model_family="mace"
        )
        return fit_gmm_router_stream(GMMReferenceStreamFitRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=request["model_digest"], reference_digest=request["reference_digest"],
        ))

    def score_gmm_router(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMStreamScoreRequest, score_gmm_router_stream

        by_id = {
            str(frame.info["structure_id"]): frame
            for frame in read_extxyz_structures(request["candidate_path"], allow_empty=False)
        }
        frames = [by_id[identity] for identity in request["frame_ids"]]
        header, chunks = _fake_descriptor_stream(
            frames, width=128, backend="mace", model_family="mace"
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=request["model_digest"], reference_digest=request["reference_digest"],
            emit_frame_means=request["emit_frame_means"],
        ))


class _ElementFakeTeacher(_FakeTeacher):
    def __init__(self):
        super().__init__()
        self.structure_vector_calls = 0

    def structure_vectors(self, *, model, input_path, type_map, work_dir):
        from workflow_core.structures.formats import read_extxyz_structures

        del model, work_dir
        self.structure_vector_calls += 1
        frames = read_extxyz_structures(input_path, allow_empty=False)
        header, _chunks = _fake_descriptor_stream(
            frames, width=128, backend="mace", model_family="mace"
        )
        identities = tuple(str(frame.info["structure_id"]) for frame in frames)
        rows = np.zeros((len(frames), 128), dtype=np.float32)
        for index, frame in enumerate(frames):
            rows[index, :2] = (float(index + 1), float(len(frame)))
        by_element = {}
        for element in type_map:
            element_ids = tuple(
                identity
                for identity, frame in zip(identities, frames, strict=True)
                if element in frame.get_chemical_symbols()
            )
            if element_ids:
                by_element[element] = (
                    element_ids,
                    np.stack(
                        [
                            rows[index]
                            for index, identity in enumerate(identities)
                            if identity in set(element_ids)
                        ]
                    ),
                )
        return (
            dict(header.schema),
            identities,
            rows,
            by_element,
        )


class _CommitteeFPTeacher(_FakeTeacher):
    def label(self, models, frames, *args, **kwargs):
        labeled = super().label(models, frames, *args, **kwargs)
        work_dir = Path(args[0]) if args else Path(kwargs.get("work_dir", "."))
        if len(models) > 1 and "teach" not in work_dir.parts:
            for index, frame in enumerate(labeled):
                frame.info["force_uncertainty"] = 0.05 if index == 0 else 0.15
        return labeled


class _ContextFreeFakeTeacher(_FakeTeacher):
    """Native executor double that does not expose operation context to core."""

    def __init__(self) -> None:
        pass


class _ChangedIdentityFakeTeacher(_FakeTeacher):
    def label(self, models, frames, *args, **kwargs):
        del kwargs
        labeled = super().label(models, frames, *args)
        labeled[0].info["structure_id"] = "sha256:" + "f" * 64
        return labeled


class _FakeDeepmdTeacher:
    def __init__(self, *, cli: str = "pt"):
        from workflow_runtime.lifecycle.environment import OperationContext

        self.train_context = OperationContext(
            role=f"teacher.deepmd.{cli}.train",
            profile_id="fake",
            source_scripts=(),
            executables={"dp": "fake"},
            runtime={},
        )
        self.gmm_score_context = OperationContext(
            role="teacher.deepmd.pt.gmm_score",
            profile_id="fake-dpa4-gmm",
            source_scripts=(),
            executables={"python": "fake"},
            runtime={},
        )
        self.label_context = OperationContext(
            role=f"teacher.deepmd.{cli}.label",
            profile_id="fake-dpa4-label",
            source_scripts=(),
            executables={"python": "fake"},
            runtime={},
        )
        self.reference_calls = 0
        self.stream_score_calls = 0

    def train(
        self,
        input_json,
        work_dir,
        cli,
        *,
        foundation_model=None,
        train_options=None,
        progress_reporter=None,
        member_index=1,
        total_members=1,
    ):
        rendered = json.loads(Path(input_json).read_text(encoding="utf-8"))
        model_config = rendered.get("model", {})
        descriptor = (
            model_config.get("descriptor", {})
            if isinstance(model_config, dict)
            else {}
        )
        if isinstance(descriptor, dict) and descriptor.get("type") == "dpa4":
            model_family = "dpa4"
        elif isinstance(model_config, dict) and model_config.get("type") == "dpa4":
            model_family = "dpa4"
        else:
            model_family = "standard"
        model = work_dir / ("native.pt2" if cli == "pt" else "native.pb")
        model.write_text("teacher", encoding="utf-8")
        native_type_map = tuple(rendered["model"]["type_map"])
        (work_dir / "deepmd_model_metadata.json").write_text(
            json.dumps(
                {
                    "model_family": model_family,
                    "type_map": list(native_type_map),
                    "native_model_suffix": model.suffix,
                    "terminal_model_sha256": hashlib.sha256(
                        model.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        if cli == "pt":
            from workflow_backends.teacher.deepmd.native import DeepMDNativeTrainResult

            if model_family == "dpa4":
                checkpoint = work_dir / "model.ckpt.pt"
                checkpoint.write_bytes(b"selected-checkpoint-" + work_dir.name.encode())
                return DeepMDNativeTrainResult(
                    terminal_model_path=model,
                    freeze_source_checkpoint_path=checkpoint,
                    checkpoint_family="regular",
                    selected_checkpoint_argument=checkpoint.name,
                    freeze_command_profile={
                        "role": "test.deepmd.pt.freeze",
                        "cli": "pt",
                    },
                    model_family=model_family,
                    native_type_map=native_type_map,
                )
            return DeepMDNativeTrainResult(
                terminal_model_path=model,
                model_family=model_family,
                native_type_map=native_type_map,
            )
        return model

    def label(self, models, frames, cli, work_dir, **kwargs):
        result = []
        for frame in frames:
            item = frame.copy()
            item.calc = SinglePointCalculator(item, energy=0.0, forces=np.zeros((len(item), 3)))
            item.info["force_uncertainty"] = 0.0
            result.append(item)
        return result

    def fit_gmm_router(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import (
            GMMReferenceStreamFitRequest,
            fit_gmm_router_stream,
        )

        self.reference_calls += 1
        frames = read_extxyz_structures(request["reference_path"], allow_empty=False)
        header, chunks = _fake_descriptor_stream(
            frames, width=17, backend="deepmd-pt", model_family="dpa4"
        )
        return fit_gmm_router_stream(GMMReferenceStreamFitRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=request["model_digest"], reference_digest=request["reference_digest"],
        ))

    def score_gmm_router(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMStreamScoreRequest, score_gmm_router_stream

        by_id = {
            str(frame.info["structure_id"]): frame
            for frame in read_extxyz_structures(request["candidate_path"], allow_empty=False)
        }
        frames = [by_id[identity] for identity in request["frame_ids"]]
        header, chunks = _fake_descriptor_stream(
            frames, width=17, backend="deepmd-pt", model_family="dpa4"
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=request["model_digest"], reference_digest=request["reference_digest"],
            emit_frame_means=request["emit_frame_means"],
        ))

    def score_gmm_router_chunks(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import (
            GMMStreamScoreRequest,
            score_gmm_router_stream,
        )

        frames = []
        for path, identities in zip(
            request["candidate_paths"], request["frame_ids_by_chunk"], strict=True
        ):
            by_id = {
                str(frame.info["structure_id"]): frame
                for frame in read_extxyz_structures(path, allow_empty=False)
            }
            frames.extend(by_id[identity] for identity in identities)
        header, chunks = _fake_descriptor_stream(
            frames, width=17, backend="deepmd-pt", model_family="dpa4"
        )
        return score_gmm_router_stream(
            GMMStreamScoreRequest(
                router_dir=request["router_dir"],
                header=header,
                chunks=chunks,
                model_digest=request["model_digest"],
                reference_digest=request["reference_digest"],
                emit_frame_means=request["emit_frame_means"],
            )
        )

    def score_gmm_router_stream(self, model, **request):
        from workflow_policies.gmm_reference_tail import (
            GMMStreamScoreRequest,
            score_gmm_router_stream,
        )

        self.stream_score_calls += 1
        frames = [frame for chunk in request["frame_chunks"] for frame in chunk]
        header, chunks = _fake_descriptor_stream(
            frames, width=17, backend="deepmd-pt", model_family="dpa4"
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"],
            header=header,
            chunks=chunks,
            model_digest=request["model_digest"],
            reference_digest=request["reference_digest"],
            emit_frame_means=request["emit_frame_means"],
        ))


class _SingleLookupDeepmdTeacher(_FakeDeepmdTeacher):
    """Reject caller-side discovery before the typed Adapter operation runs."""

    def __init__(self) -> None:
        super().__init__()
        self._stream_transport_pending = False

    @property
    def score_gmm_router_stream(self):
        if self._stream_transport_pending:
            raise AssertionError(
                "lifecycle core discovered DeepMD stream transport before Adapter dispatch"
            )
        self._stream_transport_pending = True

        def invoke(model, **request):
            self._stream_transport_pending = False
            return super(_SingleLookupDeepmdTeacher, self).score_gmm_router_stream(
                model, **request
            )

        return invoke

class _FakeStudent:
    def train(self, input_json, member_dir, *args):
        cli = str(args[0])
        model = member_dir / ("native.pb" if cli == "tf" else "native.pt2")
        model.write_text("student", encoding="utf-8")
        return model

    def gmm_reference(self, *, model, reference_path, router_dir, work_dir, type_map):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMReferenceStreamFitRequest, fit_gmm_router_stream

        frames = read_extxyz_structures(reference_path, allow_empty=False)
        width, backend, family = (
            (256, "deepmd-pt", "dpa4c")
            if Path(model).suffix == ".pt2"
            else (12, "deepmd-tf", "se_e2_a")
        )
        header, chunks = _fake_descriptor_stream(
            frames, width=width, backend=backend, model_family=family
        )
        return fit_gmm_router_stream(GMMReferenceStreamFitRequest(
            router_dir=router_dir, header=header, chunks=chunks,
            model_digest=_digest(model), reference_digest=_digest(reference_path),
        ))

    def structure_vectors(self, *, model, input_path, type_map, work_dir):
        from workflow_core.structures.formats import read_extxyz_structures

        del work_dir
        frames = read_extxyz_structures(input_path, allow_empty=False)
        width, backend, family = (
            (256, "deepmd-pt", "dpa4c")
            if Path(model).suffix == ".pt2"
            else (12, "deepmd-tf", "se_e2_a")
        )
        header, chunks = _fake_descriptor_stream(
            frames, width=width, backend=backend, model_family=family
        )
        identities = []
        means = []
        for chunk in chunks:
            identities.extend(chunk.frame_ids)
            means.append(np.asarray(chunk.descriptors, dtype=np.float32).mean(axis=0))
        del type_map
        return dict(header.schema), tuple(identities), np.stack(means)

    def gmm_score(
        self, *, model, reference_path, observation_path, ordered_identities,
        router_dir, work_dir, emit_frame_means,
    ):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMStreamScoreRequest, score_gmm_router_stream

        by_id = {
            str(frame.info["structure_id"]): frame
            for frame in read_extxyz_structures(observation_path, allow_empty=False)
        }
        frames = [by_id[identity] for identity in ordered_identities]
        width, backend, family = (
            (256, "deepmd-pt", "dpa4c")
            if Path(model).suffix == ".pt2"
            else (12, "deepmd-tf", "se_e2_a")
        )
        header, chunks = _fake_descriptor_stream(
            frames, width=width, backend=backend, model_family=family
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=router_dir, header=header, chunks=chunks,
            model_digest=_digest(model), reference_digest=_digest(reference_path),
            emit_frame_means=emit_frame_means,
        ))

    def gmm_score_chunks(self, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import (
            GMMStreamScoreRequest,
            score_gmm_router_stream,
        )

        frames = []
        for path, identities in zip(
            request["observation_paths"],
            request["ordered_identities_by_chunk"],
            strict=True,
        ):
            by_id = {
                str(frame.info["structure_id"]): frame
                for frame in read_extxyz_structures(path, allow_empty=False)
            }
            frames.extend(by_id[identity] for identity in identities)
        width, backend, family = (
            (256, "deepmd-pt", "dpa4c")
            if Path(request["model"]).suffix == ".pt2"
            else (12, "deepmd-tf", "se_e2_a")
        )
        header, chunks = _fake_descriptor_stream(
            frames, width=width, backend=backend, model_family=family
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"],
            header=header,
            chunks=chunks,
            model_digest=_digest(request["model"]),
            reference_digest=_digest(request["reference_path"]),
            emit_frame_means=request["emit_frame_means"],
        ))


class _ElementFakeStudent(_FakeStudent):
    def structure_vectors(self, **kwargs):
        from workflow_core.structures.formats import read_extxyz_structures

        schema, identities, rows = super().structure_vectors(**kwargs)
        frames = read_extxyz_structures(kwargs["input_path"], allow_empty=False)
        by_element = {}
        for element in kwargs["type_map"]:
            element_ids = tuple(
                identity
                for identity, frame in zip(identities, frames, strict=True)
                if element in frame.get_chemical_symbols()
            )
            if element_ids:
                positions = {
                    identity: index for index, identity in enumerate(identities)
                }
                by_element[element] = (
                    element_ids,
                    np.stack([rows[positions[item]] for item in element_ids]),
                )
        return schema, identities, rows, by_element


def _digest(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fake_descriptor_stream(frames, *, width: int, backend: str, model_family: str):
    from workflow_backends.descriptor_stream import AtomDescriptorChunk, DescriptorStreamHeader

    type_map = tuple(dict.fromkeys(symbol for frame in frames for symbol in frame.get_chemical_symbols()))
    schema = {
        "name": "readout-input",
        "revision": 1,
        "backend": backend,
        "model_family": model_family,
        "definition": "atomic-energy readout inputs",
        "width": width,
        "dtype": "float32",
        "normalized": False,
        "components": [{
            "id": "fake_atomic_energy_readout_input",
            "offset": 0,
            "width": width,
            "native_source": "test fixture readout input",
            "invariance": "scalar",
        }],
        "constant_terms_included": False,
        "deduplication": "unique-native-input",
    }
    header = DescriptorStreamHeader(schema=schema, type_map=type_map, frame_count=len(frames))
    chunks = []
    for ordinal, frame in enumerate(frames):
        identity = str(frame.info.get("structure_id", "")) or f"reference-frame:{ordinal:08d}"
        symbols = tuple(frame.get_chemical_symbols())
        rows = []
        for atom_index, symbol in enumerate(symbols):
            base = float(frame.positions[atom_index, 0]) + ordinal * 0.02 + type_map.index(symbol)
            rows.append(np.linspace(base, base + 0.5, width, dtype=np.float32))
        chunks.append(AtomDescriptorChunk(
            descriptors=np.asarray(rows, dtype=np.float32),
            type_indices=np.asarray([type_map.index(symbol) for symbol in symbols], dtype=np.int16),
            frame_ptr=np.asarray([0, len(symbols)], dtype=np.int64),
            frame_ordinals=np.asarray([ordinal], dtype=np.int64),
            frame_ids=(identity,),
        ))
    return header, tuple(chunks)


class _FakeSampler:
    def run(self, task_dir, task):
        _write_fake_lammpstrj(task_dir, x=0.1)
        if int(task["model_count"]) > 1:
            (task_dir / "model_devi.out").write_text(
                "20 0 0 0 0.05 0 0.01\n", encoding="utf-8"
            )
        if task.get('kind') == 'native':
            (task_dir/'native_protocol_complete.out').write_text('DICE_NATIVE_PROTOCOL_COMPLETE 20\n')
        seed = Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
        sampled = Atoms("H", positions=[[0.1, 0, 0]], cell=[4, 4, 4], pbc=True)
        return [(seed, 0), (sampled, 20)]


class _FakeSingleModelSampler:
    def run(self, task_dir, task):
        _write_fake_lammpstrj(task_dir, x=0.1)
        seed = Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
        sampled = Atoms("H", positions=[[0.1, 0, 0]], cell=[4, 4, 4], pbc=True)
        return [(seed, 0), (sampled, 20)]


class _PairGuardSampler:
    def run(self, task_dir, task):
        distance = 0.5 if float(task["temperature_k"]) == 300.0 else 0.49
        _write_fake_lammpstrj(task_dir, symbols=(1, 1), positions=((0.0, 0.0, 0.0), (distance, 0.0, 0.0)))
        if int(task["model_count"]) > 1:
            (task_dir / "model_devi.out").write_text(
                "20 0 0 0 0.05 0 0.01\n", encoding="utf-8"
            )
        sampled = Atoms("HH", positions=[[0, 0, 0], [distance, 0, 0]], cell=[4, 4, 4], pbc=True)
        return [(sampled, 20)]


def _pair_guard_sampler(root: Path) -> _PairGuardSampler:
    from workflow_core.structures.formats import write_extxyz_structures

    write_extxyz_structures(
        root / "data" / "seed.xyz",
        [
            Atoms(
                "HH",
                positions=[[0, 0, 0], [0.5, 0, 0]],
                cell=[4, 4, 4],
                pbc=True,
            )
        ],
    )
    return _PairGuardSampler()


def _write_fake_lammpstrj(
    task_dir: Path,
    *,
    x: float | None = None,
    symbols: tuple[int, ...] = (1,),
    positions: tuple[tuple[float, float, float], ...] | None = None,
) -> None:
    coords = positions if positions is not None else ((float(x or 0.0), 0.0, 0.0),)
    rows = "".join(
        f"{index} {atom_type} {position[0]} {position[1]} {position[2]}\n"
        for index, (atom_type, position) in enumerate(zip(symbols, coords, strict=True), start=1)
    )
    (task_dir / "traj.lammpstrj").write_text(
        "ITEM: TIMESTEP\n20\n"
        f"ITEM: NUMBER OF ATOMS\n{len(coords)}\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n"
        "ITEM: ATOMS id type x y z\n"
        + rows,
        encoding="utf-8",
    )


class _PairGuardRecordingTeacher(_FakeTeacher):
    def __init__(self):
        super().__init__()
        self.score_identity_calls = []
        self.label_identity_calls = []

    def score_gmm_router(self, model, **request):
        self.score_identity_calls.append(tuple(request["frame_ids"]))
        return super().score_gmm_router(model, **request)

    def label(self, models, frames, *args, **kwargs):
        self.label_identity_calls.append(tuple(str(frame.info.get("structure_id", "")) for frame in frames))
        return super().label(models, frames, *args, **kwargs)


class _PairGuardFailingScoreTeacher(_PairGuardRecordingTeacher):
    def score_gmm_router(self, model, **request):
        self.score_identity_calls.append(tuple(request["frame_ids"]))
        raise RuntimeError("injected guarded screen score failure")


class _PairGuardRecordingStudent(_FakeStudent):
    def __init__(self):
        self.score_identity_calls = []
        self.reference_calls = []

    def gmm_reference(self, **request):
        self.reference_calls.append(str(request["reference_path"]))
        return super().gmm_reference(**request)

    def gmm_score(self, **request):
        self.score_identity_calls.append(tuple(request["ordered_identities"]))
        return super().gmm_score(**request)

    def gmm_score_chunks(self, **request):
        self.score_identity_calls.append(
            tuple(
                identity
                for chunk in request["ordered_identities_by_chunk"]
                for identity in chunk
            )
        )
        return super().gmm_score_chunks(**request)


class _RoutingCommitteeTeacher(_FakeTeacher):
    def __init__(self):
        super().__init__()
        self.label_identity_calls = []

    def label(self, models, frames, *args, **kwargs):
        self.label_identity_calls.append(
            tuple(str(frame.info.get("structure_id", "")) for frame in frames)
        )
        labeled = super().label(models, frames, *args)
        for index, frame in enumerate(labeled):
            frame.info["force_uncertainty"] = (
                0.05 if len(labeled) >= 8 else (0.05, 0.2, 0.6)[index % 3]
            )
        return labeled


class _NoLowRoutingCommitteeTeacher(_RoutingCommitteeTeacher):
    def label(self, models, frames, *args, **kwargs):
        labeled = super().label(models, frames, *args, **kwargs)
        if len(labeled) < 8:
            for index, frame in enumerate(labeled):
                frame.info["force_uncertainty"] = (0.1, 0.499, 0.5)[index % 3]
        return labeled


class _HighDeviationRoutingCommitteeTeacher(_RoutingCommitteeTeacher):
    def label(self, models, frames, *args, **kwargs):
        labeled = super().label(models, frames, *args, **kwargs)
        for frame in labeled:
            frame.info["force_uncertainty"] = 0.6
        return labeled


class _BoundedBridgeTeacher(_FakeTeacher):
    def __init__(self):
        super().__init__()
        self.score_bridge_sizes = []
        self.score_operation_calls = 0

    def score_gmm_router_chunks(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMStreamScoreRequest, score_gmm_router_stream

        self.score_operation_calls += 1
        frames = []
        for candidate_path, frame_ids in zip(
            request["candidate_paths"], request["frame_ids_by_chunk"], strict=True
        ):
            chunk = read_extxyz_structures(candidate_path, allow_empty=False)
            self.score_bridge_sizes.append(len(chunk))
            by_id = {str(frame.info["structure_id"]): frame for frame in chunk}
            frames.extend(by_id[identity] for identity in frame_ids)
        header, chunks = _fake_descriptor_stream(
            frames, width=128, backend="mace", model_family="mace"
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=request["model_digest"], reference_digest=request["reference_digest"],
            emit_frame_means=request["emit_frame_means"],
        ))

    def score_gmm_router(self, model, **request):
        from workflow_core.structures.formats import read_extxyz_structures

        self.score_bridge_sizes.append(
            len(read_extxyz_structures(request["candidate_path"], allow_empty=False))
        )
        return super().score_gmm_router(model, **request)


class _BoundedBridgeStudent(_FakeStudent):
    def __init__(self):
        self.score_bridge_sizes = []
        self.score_operation_calls = 0

    def gmm_score_chunks(self, **request):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMStreamScoreRequest, score_gmm_router_stream

        self.score_operation_calls += 1
        frames = []
        for observation_path, ordered_identities in zip(
            request["observation_paths"], request["ordered_identities_by_chunk"], strict=True
        ):
            chunk = read_extxyz_structures(observation_path, allow_empty=False)
            self.score_bridge_sizes.append(len(chunk))
            by_id = {str(frame.info["structure_id"]): frame for frame in chunk}
            frames.extend(by_id[identity] for identity in ordered_identities)
        width, backend, family = (
            (256, "deepmd-pt", "dpa4c")
            if Path(request["model"]).suffix == ".pt2"
            else (12, "deepmd-tf", "se_e2_a")
        )
        header, chunks = _fake_descriptor_stream(
            frames, width=width, backend=backend, model_family=family
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=_digest(request["model"]), reference_digest=_digest(request["reference_path"]),
            emit_frame_means=request["emit_frame_means"],
        ))

    def gmm_score_rolling(self, **request):
        from workflow_policies.gmm_reference_tail import (
            GMMStreamScoreRequest,
            score_gmm_router_stream,
        )

        self.score_operation_calls += 1
        frames = []
        for chunk in request["frame_chunks"]:
            self.score_bridge_sizes.append(len(chunk))
            frames.extend(chunk)
        header, chunks = _fake_descriptor_stream(
            frames, width=12, backend="deepmd-tf", model_family="se_e2_a"
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"], header=header, chunks=chunks,
            model_digest=_digest(request["model"]),
            reference_digest=_digest(request["reference_path"]),
            emit_frame_means=request["emit_frame_means"],
        ))

    def gmm_score_stream(self, **request):
        return self.gmm_score_rolling(**request)

    def gmm_score(self, **request):
        from workflow_core.structures.formats import read_extxyz_structures

        self.score_bridge_sizes.append(
            len(read_extxyz_structures(request["observation_path"], allow_empty=False))
        )
        return super().gmm_score(**request)


class _FusedBridgeStudent(_FakeStudent):
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.screen_calls = 0
        self.reference_calls = 0
        self.score_calls = 0

    def gmm_screen(self, **request):
        from types import SimpleNamespace
        from workflow_policies.gmm_reference_tail import (
            GMMStreamScoreRequest,
            score_gmm_router_stream,
        )

        self.screen_calls += 1
        self.reference_calls += 1
        fit = super().gmm_reference(
            model=request["model"],
            reference_path=request["reference_path"],
            router_dir=request["router_dir"],
            work_dir=request["work_dir"],
            type_map=request["type_map"],
        )
        frames = [frame for chunk in request["frame_chunks"] for frame in chunk]
        if self.fail:
            raise RuntimeError("injected fused screen failure")
        self.score_calls += 1
        width, backend, family = (
            (256, "deepmd-pt", "dpa4c")
            if Path(request["model"]).suffix == ".pt2"
            else (12, "deepmd-tf", "se_e2_a")
        )
        header, chunks = _fake_descriptor_stream(
            frames, width=width, backend=backend, model_family=family
        )
        scored = score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=request["router_dir"],
            header=header,
            chunks=chunks,
            model_digest=_digest(request["model"]),
            reference_digest=_digest(request["reference_path"]),
            emit_frame_means=request["emit_frame_means"],
        ))

        class View:
            def __init__(self):
                self.identities = tuple(str(item) for item in scored.frame_ids.tolist())
                self.descriptor_schema = scored.descriptor_schema
                self.values = {
                    identity: float(value)
                    for identity, value in zip(self.identities, scored.frame_p)
                }
                self.means = (
                    None
                    if scored.frame_means is None
                    else {
                        identity: row
                        for identity, row in zip(self.identities, scored.frame_means)
                    }
                )

            def close(self, *, cleanup=False):
                del cleanup

        return SimpleNamespace(
            fit=fit,
            score=View(),
            phase_timings={"reference_seconds": 0.1, "score_seconds": 0.2},
        )


class _FakeNepStudent:
    def train(self, recipe, member):
        model = member / "nep.txt"; restart = member / "nep.restart"
        model.write_text("nep", encoding="utf-8"); restart.write_text("restart", encoding="utf-8")
        return model, restart

    def gmm_reference(self, *, model, reference_path, router_dir, work_dir, type_map):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMReferenceStreamFitRequest, fit_gmm_router_stream

        frames = read_extxyz_structures(reference_path, allow_empty=False)
        header, chunks = _fake_descriptor_stream(
            frames, width=9, backend="nep", model_family="nep4"
        )
        return fit_gmm_router_stream(GMMReferenceStreamFitRequest(
            router_dir=router_dir,
            header=header,
            chunks=chunks,
            model_digest=_digest(model),
            reference_digest=_digest(reference_path),
        ))

    def gmm_score(
        self, *, model, reference_path, observation_path, ordered_identities,
        router_dir, work_dir, emit_frame_means,
    ):
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_policies.gmm_reference_tail import GMMStreamScoreRequest, score_gmm_router_stream

        by_id = {
            str(frame.info["structure_id"]): frame
            for frame in read_extxyz_structures(observation_path, allow_empty=False)
        }
        frames = [by_id[identity] for identity in ordered_identities]
        header, chunks = _fake_descriptor_stream(
            frames, width=9, backend="nep", model_family="nep4"
        )
        return score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=router_dir,
            header=header,
            chunks=chunks,
            model_digest=_digest(model),
            reference_digest=_digest(reference_path),
            emit_frame_means=emit_frame_means,
        ))


class _FakeGpumdSampler:
    def run(self, task_dir, task):
        (task_dir / "active.out").write_text("20 0.05\n", encoding="utf-8")
        (task_dir / "active.xyz").write_text(
            "1\nProperties=species:S:1:pos:R:3\nH 0.1 0 0\n",
            encoding="utf-8",
        )
        seed = Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
        sampled = Atoms("H", positions=[[0.1, 0, 0]], cell=[4, 4, 4], pbc=True)
        sampled.info["force_uncertainty"] = 0.06
        return [(seed, 0), (sampled, 20)]


class _FakeFirstPrinciplesLabelService:
    calculator = "CP2K"
    operation_role = "first_principles.cp2k.label"

    def __init__(self):
        self.calls = 0

    def label(self, request):
        from workflow_backends.first_principles.protocols import (
            FPLabelBatchResult,
            FPLabelResult,
        )
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        self.calls += 1
        selected = read_extxyz_structures(request.selected_path, allow_empty=True)
        results = []
        for index, frame in enumerate(selected):
            task_id = f"task_{index:06d}"
            task_path = request.task_root / task_id
            task_path.mkdir(parents=True)
            write_extxyz_structures(task_path / "selected.xyz", [frame])
            input_digest = hashlib.sha256(
                (task_path / "selected.xyz").read_bytes()
            ).hexdigest()
            (task_path / "task_context.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "selected_index": index,
                        "structure_id": frame.info["structure_id"],
                        "system_id": frame.info["system_id"],
                        "calculator": self.calculator,
                        "input_digest": input_digest,
                        "required_labels": ["energy", "forces"],
                        "native_inputs": ["selected.xyz"],
                        "operation": {
                            "role": self.operation_role,
                            "profile_id": "fake-test",
                            "logical_executable": "cp2k",
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            labeled = frame.copy()
            labeled.calc = SinglePointCalculator(
                labeled,
                energy=-float(index + 1),
                forces=np.zeros((len(labeled), 3)),
            )
            results.append(
                FPLabelResult(
                    task_id=task_id,
                    selected_index=index,
                    structure_id=str(frame.info["structure_id"]),
                    system_id=str(frame.info["system_id"]),
                    labeled_atoms=labeled,
                    input_digest=input_digest,
                    parser_evidence={"fake": True},
                )
            )
        return FPLabelBatchResult(tuple(results), {"fake": True})


def _write_public_run(
    root: Path,
    *,
    teacher_backend: str = "MACE",
    teacher_model: str = "single",
    student_backend: str = "DeepMD",
    student_model: str = "single",
) -> Path:
    from workflow_core.structures.formats import write_extxyz_structures

    (root / "data").mkdir()
    (root / "recipes").mkdir()
    labeled_frames = []
    for index in range(8):
        labeled = Atoms("H", positions=[[0.02 * index, 0, 0]], cell=[4, 4, 4], pbc=True)
        labeled.calc = SinglePointCalculator(labeled, energy=0.0, forces=np.zeros((1, 3)))
        labeled_frames.append(labeled)
    write_extxyz_structures(root / "data" / "teacher.xyz", labeled_frames)
    write_extxyz_structures(
        root / "data" / "student.xyz",
        [Atoms("H", positions=[[0.03 * index, 0, 0]], cell=[4, 4, 4], pbc=True) for index in range(8)],
    )
    write_extxyz_structures(root / "data" / "seed.xyz", [Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)])
    (root / "recipes" / "teacher.yaml").write_text("max_num_epochs: 1\n", encoding="utf-8")
    (root / "recipes" / "teacher.json").write_text(
        json.dumps({"model": {"type": "dpa4", "descriptor": {"type": "dpa4"}}}),
        encoding="utf-8",
    )
    (root / "recipes" / "student.json").write_text(
        json.dumps({"model": {"descriptor": {"type": "se_e2_a"}}}), encoding="utf-8"
    )
    (root / "recipes" / "nep.in").write_text("version 4\n", encoding="utf-8")
    (root / "recipes" / "sample.in").write_text("include dice_task_setup.lammps\nrun ${dice_run_steps}\n", encoding="utf-8")
    (root / "recipes" / "run.in").write_text("velocity 300\nensemble nvt_ber 300 300 100\nrun 20\n", encoding="utf-8")
    teacher: dict[str, object] = {"backend": teacher_backend, "model": teacher_model, "type_map": ["H"], "recipe": "recipes/teacher.yaml" if teacher_backend == "MACE" else "recipes/teacher.json", "policy": "from_scratch", "valid_frac": 0.0}
    if teacher_backend == "DeepMD":
        teacher["cli"] = "pt"
    if teacher_model != "single":
        teacher.update({"members": 2, "lo": 0.1, "hi": 0.5})
    student: dict[str, object] = {"backend": student_backend, "model": student_model, "type_map": ["H"], "policy": "from_scratch", "valid_frac": 0.0}
    if student_model == "committee":
        student["members"] = 2
    if student_backend == "DeepMD":
        student.update({"cli": "tf" if student_model == "single" else "pt", "recipe": "recipes/student.json"})
    else:
        student["recipe"] = "recipes/nep.in"
    screen: dict[str, object] = {
        "pool": [1, 1], "keep": [1, 1], "fp_pool": 1,
        "policy": "random" if teacher_model == "single" else "FPS",
    }
    if student_model == "committee":
        screen.update({"lo": 0.0, "hi": 0.25})
    if teacher_model == "committee":
        del screen["fp_pool"]
    payload = {
        "iter": 0,
        "teacher_train_bootstrap": ["data/teacher.xyz"],
        "student_train_bootstrap": ["data/student.xyz"],
        "fp_data": [],
        "teacher": teacher,
        "student": student,
        "sampler": {"backend": "LAMMPS" if student_backend == "DeepMD" else "GPUMD", "recipe": "recipes/sample.in" if student_backend == "DeepMD" else "recipes/run.in"},
        "sample": {"systems": {"water": {"seed": "data/seed.xyz"}}, "steps": [20, 20], "ensemble": "nvt", "temperatures_k": [300, 500]},
        "screen": screen,
        "fp": {"keep": 1, "policy": "random"},
    }
    run = root / "run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run


def _bound_student_role(
    root: Path,
    executor,
    *,
    backend: str = "DeepMD",
    model_shape: str = "single",
):
    from workflow_backends.student.role import StudentBackendBinding

    binding = StudentBackendBinding(
        run_json_path=root / "run.json",
        recipe_path=(
            root / "recipes" / "student.json"
            if backend == "DeepMD"
            else root / "recipes" / "nep.in"
        ),
        type_map=("H",),
        model_shape=model_shape,
        policy="from_scratch",
        foundation_model=None,
        foundation_restart=None,
        train_options={},
        compress_sampling_models=False,
    )
    if backend == "DeepMD":
        from workflow_backends.student.deepmd.adapter import (
            DeepMDStudentBackendFactory,
        )

        return DeepMDStudentBackendFactory(
            executor,
            cli="tf" if model_shape == "single" else "pt",
        ).bind(binding)
    from workflow_backends.student.nep.adapter import NepStudentBackendFactory

    return NepStudentBackendFactory(executor).bind(binding)


def _write_native_fake_environment(root: Path, *, student_backend: str = "DeepMD") -> Path:
    serial_lammps_environment = (
        Path(__file__).resolve().parent / "fixtures" / "lammps_serial.env.sh"
    )
    mace_train = root / "mace_run_train.sh"
    mace_train.write_text(
        "#!/bin/sh\n"
        f"{shlex.quote(sys.executable)} -c \"import pathlib,sys,yaml; "
        "p=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()); "
        "s='_stagetwo.model' if p.get('swa') is True else '.model'; "
        "q=pathlib.Path(p['model_dir'])/(p['name']+s); q.parent.mkdir(parents=True,exist_ok=True); q.write_text('model')\" \"$2\"\n",
        encoding="utf-8",
    )
    mace_train.chmod(0o755)
    mace_python = root / "mace_python.py"
    mace_python.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from ase.calculators.singlepoint import SinglePointCalculator\n"
        "from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures\n"
        "def value(name):\n"
        "    return Path(sys.argv[sys.argv.index(name) + 1])\n"
        "module = sys.argv[sys.argv.index('-m') + 1]\n"
        "if module.endswith('relabel_worker'):\n"
        "    frames = read_extxyz_structures(value('--input'), allow_empty=True)\n"
        "    for atoms in frames:\n"
        "        atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))\n"
        "    write_extxyz_structures(value('--output'), frames, allow_empty=True)\n"
        "    value('--uncertainties').write_text(json.dumps([0.0] * len(frames)), encoding='utf-8')\n"
        "else:\n"
        "    raise SystemExit('unexpected fake MACE module: ' + module)\n",
        encoding="utf-8",
    )
    mace_python.chmod(0o755)
    dp = root / "dp.sh"
    dp.write_text("#!/bin/sh\nif [ \"$2\" = freeze ]; then printf model > \"$4\"; fi\n", encoding="utf-8")
    dp.chmod(0o755)
    lmp = root / "lmp.sh"
    lmp.write_text(
        "#!/bin/sh\n"
        "printf 'ITEM: TIMESTEP\\n20\\nITEM: NUMBER OF ATOMS\\n1\\nITEM: BOX BOUNDS pp pp pp\\n0 4\\n0 4\\n0 4\\nITEM: ATOMS id type x y z\\n1 1 0.1 0 0\\n' > traj.lammpstrj\n"
        "printf '20 0 0 0 0.1 0.1 0.1\\n' > model_devi.out\n",
        encoding="utf-8",
    )
    lmp.chmod(0o755)
    nep = root / "nep.sh"
    nep.write_text("#!/bin/sh\nprintf model > nep.txt\nprintf restart > nep.restart\n", encoding="utf-8")
    nep.chmod(0o755)
    nep_python = root / "nep_python.sh"
    nep_python.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do case $1 in --output) output=$2; shift 2;; *) shift;; esac; done\n"
        f"{shlex.quote(sys.executable)} -c 'import numpy as np, sys; np.savez_compressed(sys.argv[1], frame_000000=np.zeros((1, 35), dtype=np.float32))' \"$output\"\n",
        encoding="utf-8",
    )
    nep_python.chmod(0o755)
    gpumd = root / "gpumd.sh"
    gpumd.write_text(
        "#!/bin/sh\n"
        "printf '20 0.1\\n' > active.out\n"
        "printf '1\\nProperties=species:S:1:pos:R:3\\nH 0.1 0 0\\n' > active.xyz\n",
        encoding="utf-8",
    )
    gpumd.chmod(0o755)
    bindings = {
        "teacher.mace.train": "mace", "teacher.mace.label": "mace",
    }
    if student_backend == "DeepMD":
        bindings.update({"student.deepmd.pt.train": "deepmd", "sampler.lammps.pt": "deepmd"})
    else:
        bindings.update({
            "student.nep.train": "nep",
            "student.nep.gmm_reference": "nep",
            "student.nep.gmm_score": "nep",
            "sampler.gpumd": "nep",
        })
    environment = root / "native-fake-environment.json"
    environment.write_text(json.dumps({
        "schema": "dice.environment-profile", "schema_revision": 4,
        "software_profiles": {
            "mace": {"environment": {"source_list": []}, "executables": {"mace_run_train": str(mace_train), "python": str(mace_python)}},
            "deepmd": {"environment": {"source_list": [str(serial_lammps_environment)]}, "executables": {"dp": str(dp), "lmp": str(lmp)}},
            "nep": {"environment": {"source_list": []}, "executables": {"nep": str(nep), "gpumd": str(gpumd), "python": str(nep_python)}},
        },
        "bindings": bindings,
    }), encoding="utf-8")
    return environment


class LifecycleLifecycleExecutorTests(unittest.TestCase):

    def test_public_lifecycle_runs_final_only_neb_through_normal_sample_and_screen(
        self,
    ) -> None:
        from dataclasses import replace

        from ase.calculators.calculator import Calculator, all_changes

        from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.lifecycle.sampling_coordinator import (
            LifecycleSamplingCoordinator,
        )
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        class Harmonic(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(
                self,
                atoms=None,
                properties=("energy", "forces"),
                system_changes=all_changes,
            ):
                super().calculate(atoms, properties, system_changes)
                positions = np.asarray(self.atoms.positions, dtype=float)
                self.results = {
                    "energy": float(np.sum(positions**2)),
                    "forces": -2.0 * positions,
                }

        class Provider:
            policy_identity = {
                "name": "committee-force-deviation",
                "revision": 1,
            }

            def preflight(self, model_paths):
                assert model_paths

            def create(self, model_path):
                assert Path(model_path).is_file()
                return Harmonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            write_extxyz_structures(
                root / "data" / "endpoints.xyz",
                [
                    Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[4, 4, 4], pbc=True),
                ],
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sampler"] = {"backend": "LAMMPS"}
            payload["sample"] = {
                "round_count": 1,
                "seeds": {
                    "hop": {
                        "source": "data/endpoints.xyz",
                        "selection_group": "paths",
                    }
                },
                "condition_groups": {
                    "neb-a": {
                        "kind": "neb",
                        "seeds": ["hop"],
                        "images": 3,
                        "fmax_ev_a": 0.05,
                        "max_steps": 100,
                        "relax": True,
                        "method": "dyneb",
                    }
                },
            }
            payload["screen"]["pool"] = 3
            payload["screen"]["keep"] = 1
            run.write_text(json.dumps(payload), encoding="utf-8")

            base = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
            )
            native = AseNebSamplingNativeOps(
                calculator_provider=Provider(),
                model_family="deepmd-tf",
                operation_role="student.deepmd.tf.neb",
                operation_profile_id="test-neb",
            )
            composition = replace(
                base,
                sampling=LifecycleSamplingCoordinator(
                    native,
                    base.student.sampling_models,
                ),
            )

            main([str(run)], backend_composition=composition)

            task = root / "iter0" / "r0" / "sample" / "tasks" / "task_000000"
            status = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            self.assertIn(
                "r0/train [RUNNING] phase=training operation=student-training",
                status,
            )
            self.assertIn(
                "r0/sample [SUCCESS] backend=ASE shape=single tasks=1",
                status,
            )
            self.assertTrue((task / "task_completion_receipt.json").is_file())
            self.assertTrue((task / "raw_endpoints.extxyz").is_file())
            self.assertTrue((task / "endpoint_relaxation.json").is_file())
            self.assertEqual(
                len(
                    read_extxyz_structures(
                        root / "iter0" / "r0" / "screen" / "train.xyz",
                        allow_empty=False,
                    )
                ),
                1,
            )
            train_dir = root / "iter0" / "r0" / "train"
            sentinel = train_dir / "preserve-neb-recovery"
            sentinel.write_text("predecessor", encoding="utf-8")
            final_band = task / "final_band.extxyz"
            pristine_band = final_band.read_bytes()
            final_band.write_bytes(pristine_band + b"\n")

            main([str(run)], backend_composition=composition)
            self.assertTrue(sentinel.is_file())
            self.assertEqual(final_band.read_bytes(), pristine_band + b"\n")

            endpoint_relaxation = task / "endpoint_relaxation.json"
            pristine_endpoint_relaxation = endpoint_relaxation.read_bytes()
            corrupted_preparation = json.loads(
                endpoint_relaxation.read_text(encoding="utf-8")
            )
            corrupted_preparation["endpoints"][0]["optimizer_steps"] = 101
            endpoint_relaxation.write_text(
                json.dumps(
                    corrupted_preparation,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            context_path = task / "task_context.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["endpoint_preparation"] = corrupted_preparation
            context_path.write_text(
                json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            result_path = task / "neb_result.json"
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload["endpoint_preparation"] = corrupted_preparation
            result_payload["files"]["endpoint_relaxation.json"] = _digest(
                endpoint_relaxation
            )
            result_payload["files"]["task_context.json"] = _digest(context_path)
            result_path.write_text(
                json.dumps(result_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            success_path = task / "task_completion_receipt.json"
            success_payload = json.loads(success_path.read_text(encoding="utf-8"))
            success_payload["result_sha256"] = _digest(result_path)
            success_path.write_text(
                json.dumps(success_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )

            main([str(run)], backend_composition=composition)
            self.assertTrue(sentinel.is_file())
            self.assertNotEqual(
                endpoint_relaxation.read_bytes(), pristine_endpoint_relaxation
            )

            final_band.write_bytes(pristine_band + b"\n")

            main(
                [str(run), "--resume", "r0/sample"],
                backend_composition=composition,
            )

            self.assertTrue(sentinel.is_file())
            self.assertEqual(final_band.read_bytes(), pristine_band)
            self.assertTrue((task / "task_completion_receipt.json").is_file())

    def test_completed_explicit_two_round_iter_appends_third_round_end_to_end(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            (root/'recipes/sample.in').write_text('units metal\natom_style atomic\ntimestep .001\n'
                'fix bath all nvt temp 300 300 .1\nrun 20\n')
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sampler"] = {"backend": "LAMMPS"}
            payload["sample"] = {
                "seeds": {
                    "water": {
                        "source": "data/seed.xyz",
                        "selection_group": "water",
                    }
                },
                "condition_groups": {
                    "nvt": {
                        "kind": "nvt",
                        "recipe": "recipes/sample.in",
                        "temperatures_k": [300],
                        "steps": 20,
                    }
                },
                "rounds": [
                    {"seeds": ["water"], "condition_groups": ["nvt"]},
                    {"seeds": ["water"], "condition_groups": ["nvt"]},
                ],
            }
            run.write_text(json.dumps(payload), encoding="utf-8")

            def execute() -> None:
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(
                        run,
                        teacher=_FakeTeacher(),
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ),
                )

            execute()
            preserved = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    root / "iter0" / "r0" / "screen" / "train.xyz",
                    root / "iter0" / "r1" / "screen" / "train.xyz",
                )
            }
            old_fp_result = hashlib.sha256(
                (root / "iter0" / "fp" / "selected.xyz").read_bytes()
            ).hexdigest()

            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["rounds"].append(
                {"seeds": ["water"], "condition_groups": ["nvt"]}
            )
            payload["screen"]["pool"].append(1)
            payload["screen"]["keep"].append(1)
            run.write_text(json.dumps(payload), encoding="utf-8")
            execute()

            for path, digest in preserved.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertTrue((root / "iter0" / "r2" / "sample" / "tasks").is_dir())
            self.assertTrue((root / "iter0" / "r2" / "screen" / "train.xyz").is_file())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
            self.assertNotEqual(
                hashlib.sha256(
                    (root / "iter0" / "fp" / "selected.xyz").read_bytes()
                ).hexdigest(),
                old_fp_result,
            )
            self.assertFalse((root / ".dice_state" / "markers").exists())

    def test_public_teacher_train_uses_typed_request_without_executor_context_reachthrough(
        self,
    ) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_ContextFreeFakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            assert (
                root
                / "iter0"
                / "teach"
                / "train"
                / "teacher"
                / "teacher_model.model"
            ).is_file()

    def test_public_teacher_label_rejects_changed_identity_before_atomic_publication(
        self,
    ) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)

            with self.assertRaisesRegex(ValueError, "identities.*canonical"):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(
                        run,
                        teacher=_ChangedIdentityFakeTeacher(),
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ),
                )

            self.assertFalse((root / "iter0" / "teach" / "screen" / "train.xyz").exists())
    def test_lifecycle_routes_lammps_sample_and_screen_through_selected_sampling_role(self) -> None:
        from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
        from workflow_core.lifecycle.composition import BackendComposition
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.lifecycle.sampling_coordinator import (
            LifecycleSamplingCoordinator,
        )
        from workflow_core.lifecycle.student_facade import LifecycleStudentFacade

        class _SamplingRoleSpy:
            def __init__(self, delegate) -> None:
                self.delegate = delegate
                self.execute_requests = []
                self.open_requests = []
                self.validation_requests = []

            @property
            def model_provider(self):
                return self.delegate.model_provider

            @property
            def native_ops(self):
                return self

            @property
            def evidence_observer(self):
                return self

            def execute(self, request):
                self.execute_requests.append(request)
                return self.delegate.native_ops.execute(request)

            def open_current(self, request):
                self.open_requests.append(request)
                return self.delegate.native_ops.open_current(request)

            def validate_current(self, request):
                self.validation_requests.append(request)
                return self.delegate.native_ops.validate_current(request)

            def reconstruct_compatible_legacy(self, request):
                self.validation_requests.append(request)
                return self.delegate.native_ops.reconstruct_compatible_legacy(request)

            def observe_evidence(self, request):
                return self.delegate.evidence_observer.observe_evidence(request)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            selected_student = _bound_student_role(
                root, _FakeStudent(), model_shape="committee"
            )
            student = LifecycleStudentFacade(delegate=selected_student)
            sampling = _SamplingRoleSpy(
                LifecycleSamplingCoordinator(
                    LammpsSamplingNativeOps(
                        executor=_FakeSampler(),
                        type_map=("H",),
                        recipe=root / "recipes" / "sample.in",
                    ),
                    student.sampling_models,
                )
            )

            main(
                [str(run)],
                backend_composition=BackendComposition(
                    teacher=compose_test_backend_roles(
                        run, teacher=_FakeTeacher()
                    ).teacher,
                    student=student,
                    sampling=sampling,
                    screen=compose_test_backend_roles(
                        run,
                        teacher=_FakeTeacher(),
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ).screen,
                ),
            )

            self.assertEqual(len(sampling.execute_requests), 2)
            self.assertEqual(len(sampling.open_requests), 2)
            self.assertEqual(sampling.validation_requests, [])
            self.assertTrue(
                all(
                    request.progress_callback is not None
                    for request in sampling.execute_requests
                )
            )
            self.assertTrue(
                (root / "iter0" / "r0" / "sample" / "seeds.xyz").is_file()
            )
            self.assertFalse((root / ".dice_state" / "markers").exists())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
    def test_explicit_thirteen_roles_map_to_seven_groups_across_fake_lifecycle(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            recipes = {
                "cold": "# cold\nunits metal\natom_style atomic\ntimestep .001\nfix bath all nvt temp 300 300 .1\nrun 20\n",
                "hot": "# hot\nunits metal\natom_style atomic\ntimestep .001\nfix bath all nvt temp 300 300 .1\nrun 20\n",
                "pressure": "# pressure\nunits metal\natom_style atomic\ntimestep .001\nfix bath all npt temp 300 300 .1 iso 1 1 1\nrun 20\n",
                "cycle": (
                    "variable dice_trajectory_every_steps equal 20\n"
                    "include dice_task_setup.lammps\n"
                    "variable dice_tstart file dice_temperature_start.dat\n"
                    "variable dice_tstop file dice_temperature_stop.dat\n"
                    "variable dice_segment_steps file dice_temperature_steps.dat\n"
                    "label dice_segment_loop\n"
                    "fix dice_nvt all nvt temp ${dice_tstart} ${dice_tstop} 0.1\n"
                    "run ${dice_segment_steps}\n"
                    "unfix dice_nvt\n"
                    "next dice_tstart dice_tstop dice_segment_steps\n"
                    "jump SELF dice_segment_loop\n"
                ),
            }
            for name, content in recipes.items():
                (root / "recipes" / f"{name}.lammps").write_text(
                    content, encoding="utf-8"
                )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sampler"] = {"backend": "LAMMPS"}
            payload["sample"] = {
                "round_count": 2,
                "seeds": {
                    f"role-{index:02d}": {
                        "source": "data/seed.xyz",
                        "selection_group": f"group-{min(index // 2, 6):02d}",
                    }
                    for index in range(13)
                },
                "condition_groups": {
                    "cold": {
                        "kind": "nvt",
                        "recipe": "recipes/cold.lammps",
                        "seeds": ["role-00"],
                        "temperatures_k": [300],
                        "steps": [20, 20],
                    },
                    "hot": {
                        "kind": "nvt",
                        "recipe": "recipes/hot.lammps",
                        "temperatures_k": [700],
                        "steps": 20,
                    },
                    "pressure": {
                        "kind": "npt",
                        "recipe": "recipes/pressure.lammps",
                        "seeds": ["role-01"],
                        "temperatures_k": [500],
                        "pressures_bar": [1, 10],
                        "steps": [20, 20],
                    },
                    "cycle": {
                        "kind": "nvt_program",
                        "recipe": "recipes/cycle.lammps",
                        "seeds": ["role-00"],
                        "segments": [
                            {"temperature_k": 300, "steps": 10},
                            {
                                "start_temperature_k": 300,
                                "stop_temperature_k": 700,
                                "steps": 10,
                            },
                        ],
                    },
                },
            }
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            dice_log = (root / "iter0" / "dice.log").read_text(encoding="utf-8")
            self.assertIn("[r0/screen]", dice_log)
            self.assertIn("[r1/screen]", dice_log)
            for group_index in range(7):
                self.assertIn(f"group-{group_index:02d}", dice_log)

            for round_index in range(2):
                sample = root / "iter0" / f"r{round_index}" / "sample"
                contexts = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted((sample / "tasks").glob("*/task_context.json"))
                ]
                self.assertEqual(len(contexts), 17)
                self.assertEqual(
                    {item["sampling_role_id"] for item in contexts},
                    {f"role-{index:02d}" for index in range(13)},
                )
                self.assertEqual(
                    [item["condition_group_id"] for item in contexts],
                    ["cold", *(["hot"] * 13), "pressure", "pressure", "cycle"],
                )
                self.assertEqual(
                    [(item['schema'], item.get('revision',item.get('schema_revision'))) for item in contexts],
                    [('dice.native-protocol-task',1)]*16 + [('dice.lifecycle.sample-task-context',2)],
                )
                model_digests = []
                for item in contexts:
                    if item['schema'] == 'dice.native-protocol-task':
                        result = json.loads((sample/'tasks'/item['task_id']/'protocol_result.json').read_text())
                        model_digests.append(tuple(result['files'][name] for name in sorted(result['files']) if name.startswith('graph_')))
                    else:
                        model_digests.append(tuple(item['model_digests']))
                self.assertEqual(len(set(model_digests)),1)
                self.assertEqual(
                    {item["selection_group_id"] for item in contexts},
                    {f"group-{index:02d}" for index in range(7)},
                )
                self.assertEqual(contexts[-1]["segment_boundaries"], [10, 20])
                self.assertEqual(
                    {
                        item["condition_group_id"]
                        for item in contexts
                        if item["selection_group_id"] == "group-00"
                    },
                    {"cold", "hot", "pressure", "cycle"},
                )
                grouped = json.loads(
                    (
                        root
                        / "iter0"
                        / f"r{round_index}"
                        / "screen"
                        / "selection_group_outcomes.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    grouped["selection_groups"],
                    [f"group-{index:02d}" for index in range(7)],
                )
                self.assertEqual(len(grouped["rows"]), 7)
                keep_value = payload["screen"]["keep"]
                keep_ceiling = (
                    keep_value[round_index]
                    if isinstance(keep_value, list)
                    else keep_value
                )
                self.assertLessEqual(
                    sum(row["final_selected_count"] for row in grouped["rows"]),
                    7 * keep_ceiling,
                )
                self.assertFalse((root / ".dice_state" / "markers").exists())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
            fp_groups = json.loads(
                (
                    root
                    / "iter0"
                    / "fp"
                    / "selection_group_outcomes.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                fp_groups["selection_groups"],
                [f"group-{index:02d}" for index in range(7)],
            )
            self.assertEqual(len(fp_groups["rows"]), 7)
            self.assertLessEqual(
                sum(row["selected_count"] for row in fp_groups["rows"]),
                7 * payload["fp"]["keep"],
            )
            r0_screen_ids = {
                frame.info["structure_id"]
                for frame in read_extxyz_structures(
                    root / "iter0" / "r0" / "screen" / "train.xyz",
                    allow_empty=True,
                )
            }
            r0_train_ids = {
                frame.info["structure_id"]
                for frame in read_extxyz_structures(
                    root / "iter0" / "r0" / "train" / "dataset" / "train.xyz",
                    allow_empty=False,
                )
            }
            r1_train_ids = {
                frame.info["structure_id"]
                for frame in read_extxyz_structures(
                    root / "iter0" / "r1" / "train" / "dataset" / "train.xyz",
                    allow_empty=False,
                )
            }
            self.assertTrue(r0_screen_ids)
            self.assertTrue(r0_screen_ids.isdisjoint(r0_train_ids))
            self.assertTrue(r0_screen_ids.issubset(r1_train_ids))

            mapping_sample_sentinel = (
                root / "iter0" / "r0" / "sample" / "preserve-mapping-mismatch"
            )
            mapping_screen_sentinel = (
                root / "iter0" / "r0" / "screen" / "preserve-mapping-mismatch"
            )
            mapping_sample_sentinel.write_text("sample", encoding="utf-8")
            mapping_screen_sentinel.write_text("screen", encoding="utf-8")
            payload["sample"]["seeds"]["role-01"]["selection_group"] = "group-06"
            run.write_text(json.dumps(payload), encoding="utf-8")
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )
            self.assertTrue(mapping_sample_sentinel.is_file())
            self.assertTrue(mapping_screen_sentinel.is_file())
            payload["sample"]["seeds"]["role-01"]["selection_group"] = "group-00"
            run.write_text(json.dumps(payload), encoding="utf-8")

            payload["screen"]["pool"][0] = 2
            run.write_text(json.dumps(payload), encoding="utf-8")
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )
            self.assertTrue(mapping_sample_sentinel.is_file())
            self.assertTrue(mapping_screen_sentinel.is_file())
            payload["screen"]["pool"][0] = 1
            run.write_text(json.dumps(payload), encoding="utf-8")

            train_sentinel = root / "iter0" / "r0" / "train" / "preserve-on-sample-repair"
            train_sentinel.write_text("train", encoding="utf-8")
            schedule = (
                root
                / "iter0"
                / "r0"
                / "sample"
                / "tasks"
                / "task_000016"
                / "dice_temperature_steps.dat"
            )
            prior_attempt = (
                root
                / "iter0"
                / "r0"
                / "sample"
                / ".attempts"
                / "prior"
                / "attempt.json"
            )
            prior_attempt.parent.mkdir(parents=True)
            prior_attempt.write_text("{}", encoding="utf-8")
            schedule.write_text("10\n9\n", encoding="utf-8")
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )
            self.assertTrue(train_sentinel.is_file())
            self.assertEqual(schedule.read_text(encoding="utf-8"), "10\n9\n")
            self.assertTrue(prior_attempt.is_file())

            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve-before-resume"
            screen_sentinel = root / "iter0" / "r0" / "screen" / "preserve-before-resume"
            sample_sentinel.write_text("sample", encoding="utf-8")
            screen_sentinel.write_text("screen", encoding="utf-8")
            cycle_recipe = root / "recipes" / "cycle.lammps"
            cycle_recipe.write_text(
                cycle_recipe.read_text(encoding="utf-8").replace(
                    "include dice_task_setup.lammps",
                    "# changed thermostat protocol bytes\ninclude dice_task_setup.lammps",
                ),
                encoding="utf-8",
            )
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )
            self.assertTrue(sample_sentinel.is_file())
            self.assertTrue(screen_sentinel.is_file())

            main(
                [str(run), "--resume", "r0/sample"],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )
            self.assertFalse(sample_sentinel.exists())
            self.assertFalse(screen_sentinel.exists())
            self.assertTrue(train_sentinel.is_file())

    def test_lifecycle_routes_gpumd_sample_and_screen_through_selected_sampling_role(self) -> None:
        from workflow_backends.gpumd.adapter import GpumdSamplingNativeOps
        from workflow_core.lifecycle.composition import BackendComposition
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.lifecycle.sampling_coordinator import (
            LifecycleSamplingCoordinator,
        )
        from workflow_core.lifecycle.student_facade import LifecycleStudentFacade

        class _SamplingRoleSpy:
            def __init__(self, delegate) -> None:
                self.delegate = delegate
                self.execute_requests = []
                self.open_requests = []
                self.validation_requests = []

            @property
            def model_provider(self):
                return self.delegate.model_provider

            @property
            def native_ops(self):
                return self

            @property
            def evidence_observer(self):
                return self

            def execute(self, request):
                self.execute_requests.append(request)
                return self.delegate.native_ops.execute(request)

            def open_current(self, request):
                self.open_requests.append(request)
                return self.delegate.native_ops.open_current(request)

            def validate_current(self, request):
                self.validation_requests.append(request)
                return self.delegate.native_ops.validate_current(request)

            def reconstruct_compatible_legacy(self, request):
                self.validation_requests.append(request)
                return self.delegate.native_ops.reconstruct_compatible_legacy(request)

            def observe_evidence(self, request):
                return self.delegate.evidence_observer.observe_evidence(request)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root,
                teacher_model="committee",
                student_backend="NEP",
                student_model="committee",
            )
            selected_student = _bound_student_role(
                root,
                _FakeNepStudent(),
                backend="NEP",
                model_shape="committee",
            )
            student = LifecycleStudentFacade(delegate=selected_student)
            sampling = _SamplingRoleSpy(
                LifecycleSamplingCoordinator(
                    GpumdSamplingNativeOps(
                        executor=_FakeGpumdSampler(),
                        recipe=root / "recipes" / "run.in",
                    ),
                    student.sampling_models,
                )
            )

            main(
                [str(run)],
                backend_composition=BackendComposition(
                    teacher=compose_test_backend_roles(
                        run, teacher=_FakeTeacher()
                    ).teacher,
                    student=student,
                    sampling=sampling,
                    screen=compose_test_backend_roles(
                        run,
                        teacher=_FakeTeacher(),
                        student=_FakeNepStudent(),
                        sampling=_FakeGpumdSampler(),
                    ).screen,
                ),
            )

            self.assertEqual(len(sampling.execute_requests), 2)
            self.assertEqual(len(sampling.open_requests), 2)
            self.assertEqual(sampling.validation_requests, [])
            self.assertTrue(
                all(
                    request.progress_callback is not None
                    for request in sampling.execute_requests
                )
            )
            self.assertTrue(
                (root / "iter0" / "r0" / "sample" / "seeds.xyz").is_file()
            )
            self.assertFalse((root / ".dice_state" / "markers").exists())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_public_single_dpa4c_gmm_uses_uncompressed_train_terminal(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.gmm_reference_tail import validate_gmm_router
        from workflow_runtime.lifecycle.environment import OperationContext

        class _SingleDPA4CStudent(_FusedBridgeStudent):
            def __init__(self) -> None:
                super().__init__()
                self.compress_context = OperationContext(
                    role="student.deepmd.pt.compress",
                    profile_id="fake-dpa4c-compress",
                    source_scripts=(),
                    executables={"dp": "fake"},
                    runtime={},
                )
                self.gmm_models: list[tuple[str, bytes]] = []

            def compress(self, source_model, output_model, cli, **kwargs):
                output_model.parent.mkdir(parents=True, exist_ok=True)
                output_model.write_bytes(b"compressed:" + source_model.read_bytes())
                return output_model

            def gmm_screen(self, **request):
                model = Path(request["model"])
                self.gmm_models.append(("screen", model.read_bytes()))
                return super().gmm_screen(**request)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="single"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"].update({"cli": "pt", "compress": True})
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "student.json").write_text(
                json.dumps({"model": {"descriptor": {"type": "dpa4c"}}}),
                encoding="utf-8",
            )
            student = _SingleDPA4CStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=student,
                    sampling=_FakeSingleModelSampler(),
                ),
            )

            train_terminal = (
                root
                / "iter0"
                / "r0"
                / "train"
                / "student"
                / "student_model.pt2"
            )
            sample = root / "iter0" / "r0" / "sample"
            task_model = sample / "tasks" / "task_000000" / "graph_000.pt2"
            router = root / "iter0" / "r0" / "screen" / "gmm_router"
            manifest = validate_gmm_router(router)

            self.assertEqual(train_terminal.read_bytes(), b"student")
            self.assertEqual(task_model.read_bytes(), b"compressed:student")
            self.assertEqual(student.gmm_models, [("screen", b"student")])
            self.assertEqual(manifest["model_digest"], _digest(train_terminal))
            self.assertEqual(manifest["descriptor_schema"]["model_family"], "dpa4c")
            native_execution = json.loads(
                (
                    sample
                    / "tasks"
                    / "task_000000"
                    / "native_execution.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(native_execution["execution_mode"], "kokkos")
            self.assertEqual(
                native_execution["command"],
                ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-in", "input.lammps"],
            )
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_public_dpa4c_committee_compresses_every_member_before_lammps(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_runtime.lifecycle.environment import OperationContext

        class _CompressingStudent(_FakeStudent):
            def __init__(self) -> None:
                self.compress_context = OperationContext(
                    role="student.deepmd.pt.compress",
                    profile_id="fake-dpa4c-compress",
                    source_scripts=(),
                    executables={"dp": "fake"},
                    runtime={},
                )
                self.compression_calls: list[str] = []

            def compress(
                self,
                source_model,
                output_model,
                cli,
                **kwargs,
            ):
                self.compression_calls.append(source_model.parent.name)
                output_model.parent.mkdir(parents=True, exist_ok=True)
                output_model.write_bytes(
                    b"compressed:" + source_model.read_bytes()
                )
                return output_model

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["compress"] = True
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "student.json").write_text(
                json.dumps({"model": {"descriptor": {"type": "dpa4c"}}}),
                encoding="utf-8",
            )
            student = _CompressingStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=student,
                    sampling=_FakeSampler(),
                ),
            )

            sample = root / "iter0" / "r0" / "sample"
            manifest = json.loads(
                (sample / "sampling_model_assets.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                student.compression_calls,
                ["student_01", "student_02"],
            )
            self.assertEqual(
                [member["member_id"] for member in manifest["members"]],
                ["student_01", "student_02"],
            )
            for member in manifest["members"]:
                rendered = json.loads(
                    (
                        root
                        / "iter0"
                        / member["training_script"]
                    ).read_text(encoding="utf-8")
                )
                systems = rendered["training"]["training_data"]["systems"]
                self.assertTrue(all(Path(path).is_dir() or Path(path).is_file() for path in systems))
                self.assertFalse(any(".train.staging" in path for path in systems))
            for task in sorted((sample / "tasks").iterdir()):
                self.assertEqual(
                    (task / "graph_000.pt2").read_bytes(),
                    b"compressed:student",
                )
                self.assertEqual(
                    (task / "graph_001.pt2").read_bytes(),
                    b"compressed:student",
                )

    def test_compression_failure_never_starts_lammps_or_publishes_partial_assets(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_runtime.lifecycle.environment import OperationContext

        class _FailingCompressionStudent(_FakeStudent):
            def __init__(self) -> None:
                self.compress_context = OperationContext(
                    role="student.deepmd.pt.compress",
                    profile_id="fake-dpa4c-compress",
                    source_scripts=(),
                    executables={"dp": "fake"},
                    runtime={},
                )
                self.calls = 0

            def compress(self, source_model, output_model, cli, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("injected member compression failure")
                output_model.parent.mkdir(parents=True, exist_ok=True)
                output_model.write_bytes(b"partial")
                return output_model

        class _RecordingSampler(_FakeSampler):
            def __init__(self) -> None:
                self.calls = 0

            def run(self, task_dir, task):
                self.calls += 1
                return super().run(task_dir, task)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["compress"] = True
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "student.json").write_text(
                json.dumps({"model": {"descriptor": {"type": "dpa4c"}}}),
                encoding="utf-8",
            )
            student = _FailingCompressionStudent()
            sampler = _RecordingSampler()

            with self.assertRaisesRegex(
                RuntimeError, "injected member compression failure"
            ):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_FakeTeacher(),
                        student=student,
                        sampling=sampler,
                    ),
                )

            sample = root / "iter0" / "r0" / "sample"
            self.assertEqual(student.calls, 2)
            self.assertEqual(sampler.calls, 0)
            self.assertTrue((sample / ".attempts").is_dir())
            self.assertFalse((sample / "sampling_models").exists())
            self.assertFalse((sample / "sampling_model_assets.json").exists())
            self.assertFalse(
                (root / "iter0" / "r0" / ".sample.staging").exists()
            )

    def test_corrupt_irrelevant_compressed_model_is_ignored_behind_terminal(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_runtime.lifecycle.environment import OperationContext

        class _CountingStudent(_FakeStudent):
            def __init__(self) -> None:
                self.compress_context = OperationContext(
                    role="student.deepmd.pt.compress",
                    profile_id="fake-dpa4c-compress",
                    source_scripts=(),
                    executables={"dp": "fake"},
                    runtime={},
                )
                self.train_calls = 0
                self.compress_calls = 0

            def train(self, input_json, member_dir, *args, **kwargs):
                self.train_calls += 1
                return super().train(input_json, member_dir, *args)

            def compress(self, source_model, output_model, cli, **kwargs):
                self.compress_calls += 1
                output_model.parent.mkdir(parents=True, exist_ok=True)
                output_model.write_bytes(
                    b"compressed:" + source_model.read_bytes()
                )
                return output_model

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["compress"] = True
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "student.json").write_text(
                json.dumps({"model": {"descriptor": {"type": "dpa4c"}}}),
                encoding="utf-8",
            )
            student = _CountingStudent()
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=student,
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            self.assertEqual((student.train_calls, student.compress_calls), (2, 2))
            derived = (
                root
                / "iter0"
                / "r0"
                / "sample"
                / "sampling_models"
                / "student_01"
                / "model_compressed.pt2"
            )
            derived.write_bytes(b"corrupt")

            main([str(run)], backend_composition=bundle)

            self.assertEqual((student.train_calls, student.compress_calls), (2, 2))
            self.assertEqual(derived.read_bytes(), b"corrupt")
            self.assertTrue(
                (
                    root
                    / "iter0"
                    / "r0"
                    / "sample"
                    / "sampling_model_assets.json"
                ).is_file()
            )
            task = (
                root
                / "iter0"
                / "r0"
                / "sample"
                / "tasks"
                / "task_000000"
            )
            execution_path = task / "native_execution.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["pair_style"] = "deepmd/kk"
            execution_path.write_text(json.dumps(execution), encoding="utf-8")

            main([str(run)], backend_composition=bundle)

            self.assertEqual((student.train_calls, student.compress_calls), (2, 2))
            repaired_execution = json.loads(
                execution_path.read_text(encoding="utf-8")
            )
            self.assertEqual(repaired_execution["pair_style"], "deepmd/kk")

            (task / "graph_001.pt2").unlink()
            main([str(run)], backend_composition=bundle)
            self.assertEqual((student.train_calls, student.compress_calls), (2, 2))
            self.assertFalse((task / "graph_001.pt2").exists())

    def test_corrupt_single_dpa4c_kokkos_evidence_rebuilds_sample_suffix_only(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        class _CountingStudent(_FusedBridgeStudent):
            def __init__(self) -> None:
                super().__init__()
                self.train_calls = 0

            def train(self, input_json, member_dir, *args):
                self.train_calls += 1
                return super().train(input_json, member_dir, *args)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["cli"] = "pt"
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "student.json").write_text(
                json.dumps({"model": {"descriptor": {"type": "dpa4c"}}}),
                encoding="utf-8",
            )
            student = _CountingStudent()
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=student,
                sampling=_FakeSingleModelSampler(),
            )

            main([str(run)], backend_composition=bundle)

            train = root / "iter0" / "r0" / "train"
            sentinel = train / "preserve.me"
            sentinel.write_text("valid train", encoding="utf-8")
            task = root / "iter0" / "r0" / "sample" / "tasks" / "task_000000"
            receipt_path = task / "native_execution.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["command"] = [
                "lmp", "-sf", "kk", "-k", "on", "g", "1", "-in", "input.lammps"
            ]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            main([str(run), "--resume", "r0/sample"], backend_composition=bundle)

            self.assertEqual(student.train_calls, 1)
            self.assertTrue(sentinel.is_file())
            repaired = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired["execution_mode"], "kokkos")
            self.assertEqual(
                repaired["command"],
                ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-in", "input.lammps"],
            )

            (task / "graph_000.pt2").unlink()
            (task / "graph_000.pt2").write_bytes(b"damaged presentation")
            main([str(run)], backend_composition=bundle)
            self.assertEqual(student.train_calls, 1)
            self.assertEqual(
                (task / "graph_000.pt2").read_bytes(),
                b"damaged presentation",
            )

            (task / "traj.lammpstrj").write_text("damaged trajectory", encoding="utf-8")
            main([str(run)], backend_composition=bundle)
            self.assertEqual(student.train_calls, 1)
            self.assertEqual(
                (task / "traj.lammpstrj").read_text(encoding="utf-8"),
                "damaged trajectory",
            )

    def test_compress_toggle_invalidates_sample_suffix_but_preserves_train(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_runtime.lifecycle.environment import OperationContext

        class _CountingStudent(_FakeStudent):
            def __init__(self) -> None:
                self.compress_context = OperationContext(
                    role="student.deepmd.pt.compress",
                    profile_id="fake-dpa4c-compress",
                    source_scripts=(),
                    executables={"dp": "fake"},
                    runtime={},
                )
                self.train_calls = 0
                self.compress_calls = 0

            def train(self, input_json, member_dir, *args, **kwargs):
                self.train_calls += 1
                return super().train(input_json, member_dir, *args)

            def compress(self, source_model, output_model, cli, **kwargs):
                self.compress_calls += 1
                output_model.parent.mkdir(parents=True, exist_ok=True)
                output_model.write_bytes(
                    b"compressed:" + source_model.read_bytes()
                )
                return output_model

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["compress"] = True
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "student.json").write_text(
                json.dumps({"model": {"descriptor": {"type": "dpa4c"}}}),
                encoding="utf-8",
            )
            student = _CountingStudent()
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=student,
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            train = root / "iter0" / "r0" / "train"
            sentinel = train / "preserve.me"
            sentinel.write_text("train", encoding="utf-8")
            payload["student"]["compress"] = False
            run.write_text(json.dumps(payload), encoding="utf-8")

            main([str(run)], backend_composition=bundle)
            self.assertTrue(sentinel.is_file())

            main(
                [str(run), "--resume", "r0/sample"],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=student,
                    sampling=_FakeSampler(),
                ),
            )

            sample = root / "iter0" / "r0" / "sample"
            self.assertTrue(sentinel.is_file())
            self.assertEqual((student.train_calls, student.compress_calls), (2, 2))
            self.assertFalse((sample / "sampling_model_assets.json").exists())
            task = sample / "tasks" / "task_000000"
            self.assertEqual((task / "graph_000.pt2").read_bytes(), b"student")
            self.assertEqual((task / "graph_001.pt2").read_bytes(), b"student")

    def test_single_gmm_scores_only_bounded_replay_bridges(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _BoundedBridgeTeacher()
            student = _BoundedBridgeStudent()
            from workflow_backends.lammps.observation_source import LammpsObservationSource

            replay_calls: list[tuple[str, ...]] = []
            original_replay = LammpsObservationSource.replay

            def counted_replay(source, ordered_identities, *, chunk_size):
                replay_calls.append(tuple(ordered_identities))
                yield from original_replay(
                    source, ordered_identities, chunk_size=chunk_size
                )

            with mock.patch.object(
                LammpsObservationSource, "replay", counted_replay
            ):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=teacher,
                        student=student,
                        sampling=_FakeSampler(),
                        observation_chunk_size=1,
                        gmm_score_chunk_size=1,
                    ),
                )

            self.assertTrue(teacher.score_bridge_sizes)
            self.assertTrue(student.score_bridge_sizes)
            self.assertEqual(teacher.score_operation_calls, 1)
            self.assertEqual(student.score_operation_calls, 1)
            self.assertEqual(teacher.score_bridge_sizes, [1])
            self.assertGreater(len(student.score_bridge_sizes), 1)
            student_population_size = sum(student.score_bridge_sizes)
            self.assertEqual(
                sum(len(call) == student_population_size for call in replay_calls),
                1,  # Scientific scoring only; visualization must not replay it.
                replay_calls,
            )
            self.assertLessEqual(max(teacher.score_bridge_sizes), 1)
            self.assertLessEqual(max(student.score_bridge_sizes), 1)
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))
            self.assertFalse(any(path.name == "candidates.xyz" for path in root.rglob("*")))
            self.assertFalse(any(path.name == "rolling_extxyz" for path in root.rglob("*")))
            self.assertFalse(any(path.name == "geometry.sock" for path in root.rglob("*")))
            self.assertFalse(any(path.name == "compact_score_result" for path in root.rglob("*")))
            self.assertTrue((root / "iter0" / "r0" / "screen" / "gmm_scores.npz").is_file())

    def test_fused_student_screen_is_one_atomic_call_and_targeted_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            failing = _FusedBridgeStudent(fail=True)
            with self.assertRaisesRegex(RuntimeError, "injected fused screen failure"):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_FakeTeacher(),
                        student=failing,
                        sampling=_FakeSampler(),
                        observation_chunk_size=1,
                    ),
                )
            self.assertEqual((failing.screen_calls, failing.reference_calls, failing.score_calls), (1, 1, 0))
            screen = root / "iter0" / "r0" / "screen"
            self.assertFalse((screen / "gmm_scores.npz").exists())
            self.assertFalse((root / ".dice_state" / "markers").exists())
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            sample_sentinel.write_text("sample", encoding="utf-8")

            successful = _FusedBridgeStudent()
            main(
                [str(run), "--resume", "r0/screen"],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=successful,
                    sampling=_FakeSampler(),
                    observation_chunk_size=1,
                ),
            )
            self.assertEqual((successful.screen_calls, successful.reference_calls, successful.score_calls), (1, 1, 1))
            self.assertTrue(sample_sentinel.is_file())
            self.assertTrue((screen / "gmm_router" / "manifest.json").is_file())
            self.assertTrue((screen / "gmm_scores.npz").is_file())
            self.assertFalse((root / ".dice_state" / "markers").exists())

    def test_lammps_screen_replays_committed_tasks_without_sample_observation_index(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)

            sample = root / "iter0" / "r0" / "sample"
            self.assertTrue((sample / "seeds.xyz").is_file())
            self.assertTrue((sample / "tasks" / "task_000000" / "task_context.json").is_file())
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))
            first_ids = tuple(
                frame.info["structure_id"]
                for frame in read_extxyz_structures(root / "iter0" / "r0" / "screen" / "train.xyz", allow_empty=True)
            )

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            resumed_ids = tuple(
                frame.info["structure_id"]
                for frame in read_extxyz_structures(root / "iter0" / "r0" / "screen" / "train.xyz", allow_empty=True)
            )
            self.assertEqual(resumed_ids, first_ids)
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))

    def test_lammps_sample_artifact_check_rebuilds_an_extra_task_directory(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample = root / "iter0" / "r0" / "sample"
            extra = sample / "tasks" / "task_999999"
            extra.mkdir()
            (extra / "unexpected").write_text("stale", encoding="utf-8")

            main([str(run), "--resume", "r0/sample"], backend_composition=bundle)

            self.assertFalse(extra.exists())
            self.assertEqual(
                sorted(path.name for path in (sample / "tasks").iterdir() if path.is_dir()),
                ["task_000000", "task_000001"],
            )

    def test_gpumd_sample_artifact_check_rebuilds_an_extra_task_directory(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root,
                teacher_model="committee",
                student_backend="NEP",
                student_model="committee",
            )
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeNepStudent(),
                sampling=_FakeGpumdSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample = root / "iter0" / "r0" / "sample"
            extra = sample / "tasks" / "task_999999"
            extra.mkdir()

            main([str(run), "--resume", "r0/sample"], backend_composition=bundle)

            self.assertFalse(extra.exists())
            self.assertEqual(
                sorted(path.name for path in (sample / "tasks").iterdir() if path.is_dir()),
                ["task_000000", "task_000001"],
            )

    def test_gpumd_heterogeneous_plan_runs_through_sample_and_screen_reconstruction(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import write_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_backend="NEP", student_model="committee",
            )
            write_extxyz_structures(
                root / "data" / "B.xyz",
                [Atoms("H", positions=[[0.2, 0, 0]], cell=[4, 4, 4], pbc=True)],
            )
            (root/'recipes/run.in').write_text('velocity 300\nensemble npt_ber 300 300 100 0 200 1000\nrun 20\n')
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"] = {
                "systems": {
                    "A": {"seed": "data/seed.xyz", "ensemble": "nvt", "temperatures_k": [300, 500]},
                    "B": {
                        "seed": "data/B.xyz", "ensemble": "npt",
                        "temperatures_k": [400, 600], "pressures_bar": [1, 10],
                    },
                },
                "steps": [20],
            }
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(), student=_FakeNepStudent(), sampling=_FakeGpumdSampler(),
                ),
            )

            sample = root / "iter0" / "r0" / "sample"
            self.assertEqual(
                [task.name for task in sorted((sample / "tasks").iterdir())],
                [f"task_{index:06d}" for index in range(6)],
            )
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "fp_candidate.xyz").is_file())
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))
            status = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            self.assertIn("r0/sample [SUCCESS] backend=GPUMD shape=committee tasks=6 systems=2 temperatures=4 steps=120", status)

            class FailingGpumd(_FakeGpumdSampler):
                def run(self, task_dir, task):
                    if task["system_id"] == "B" and task["pressure_bar"] == 10.0:
                        raise RuntimeError("planned failure")
                    return super().run(task_dir, task)

            with self.assertRaisesRegex(RuntimeError, "failed atomically"):
                main(
                    [str(run), "--resume", "r0/sample"],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_FakeTeacher(), student=_FakeNepStudent(), sampling=FailingGpumd(),
                    ),
                )
            self.assertFalse((root / ".dice_state" / "markers" / "iter0" / "r0__sample.json").exists())
            self.assertFalse((root / "iter0" / "r0" / "sample" / "tasks").exists())

    def test_lammps_heterogeneous_plan_runs_with_ensemble_recipes_and_screen_replay(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import write_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            write_extxyz_structures(
                root / "data" / "B.xyz",
                [Atoms("H", positions=[[0.2, 0, 0]], cell=[4, 4, 4], pbc=True)],
            )
            for ensemble in ("nvt", "npt"):
                (root / "recipes" / f"{ensemble}.lammps").write_text(
                    "include dice_task_setup.lammps\n"
                    f"# {ensemble.upper()}-RECIPE\n"
                    f"fix protocol all {ensemble} temp 300 300 100\n"
                    "run ${dice_run_steps}\n",
                    encoding="utf-8",
                )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sampler"] = {
                "backend": "LAMMPS",
                "recipes": {"nvt": "recipes/nvt.lammps", "npt": "recipes/npt.lammps"},
            }
            payload["sample"] = {
                "systems": {
                    "A": {"seed": "data/seed.xyz", "ensemble": "nvt", "temperatures_k": [300, 500]},
                    "B": {
                        "seed": "data/B.xyz", "ensemble": "npt",
                        "temperatures_k": [400, 600], "pressures_bar": [1, 10],
                    },
                },
                "steps": [20],
            }
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
                ),
            )

            sample = root / "iter0" / "r0" / "sample"
            tasks = sorted((sample / "tasks").iterdir())
            self.assertEqual([task.name for task in tasks], [f"task_{index:06d}" for index in range(6)])
            self.assertEqual(
                ["NVT-RECIPE" in (task / "input.lammps").read_text(encoding="utf-8") for task in tasks],
                [True, True, False, False, False, False],
            )
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))
            status = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            self.assertIn("r0/sample [SUCCESS] backend=LAMMPS shape=committee tasks=6 systems=2 temperatures=4 steps=120", status)

            class FailingLammps(_FakeSampler):
                def run(self, task_dir, task):
                    if task["system_id"] == "B" and task["pressure_bar"] == 10.0:
                        raise RuntimeError("planned failure")
                    return super().run(task_dir, task)

            main(
                [str(run), "--resume", "r0/sample"],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=FailingLammps(),
                ),
            )
            self.assertTrue((root / "iter0" / "r0" / "sample" / "tasks").is_dir())
            self.assertFalse((root / ".dice_state" / "markers").exists())
            failed_tasks = root / "iter0" / "r0" / "sample" / "tasks"
            self.assertTrue(
                (failed_tasks / "task_000003" / "sampling_failure.json").is_file()
            )
            self.assertTrue(
                (failed_tasks / "task_000005" / "sampling_failure.json").is_file()
            )
            self.assertIn(
                "flow=frames trajectory=4 catalog=4",
                (root / "iter0" / "status.log").read_text(encoding="utf-8"),
            )

            main(
                [str(run), "--resume", "r0/sample"],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )
            self.assertTrue((root / "iter0" / "r0" / "sample" / "tasks").is_dir())
            self.assertFalse((root / ".dice_state" / "markers").exists())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))

    def test_heterogeneous_sample_drift_requires_block_resume_and_corruption_replays_atomically(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import write_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_backend="NEP", student_model="committee",
            )
            write_extxyz_structures(
                root / "data" / "B.xyz",
                [Atoms("H", positions=[[0.2, 0, 0]], cell=[4, 4, 4], pbc=True)],
            )
            (root/'recipes/run.in').write_text('velocity 300\nensemble npt_ber 300 300 100 0 200 1000\nrun 20\n')
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"] = {
                "systems": {
                    "A": {"seed": "data/seed.xyz", "ensemble": "nvt", "temperatures_k": [300, 500]},
                    "B": {
                        "seed": "data/B.xyz", "ensemble": "npt",
                        "temperatures_k": [400, 600], "pressures_bar": [1, 10],
                    },
                },
                "steps": [20],
            }
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeNepStudent(), sampling=_FakeGpumdSampler(),
            )
            main([str(run)], backend_composition=bundle)

            sample = root / "iter0" / "r0" / "sample"
            sample_sentinel = sample / "preserve-before-explicit-resume"
            screen_sentinel = root / "iter0" / "r0" / "screen" / "preserve-before-explicit-resume"
            sample_sentinel.write_text("sample", encoding="utf-8")
            screen_sentinel.write_text("screen", encoding="utf-8")
            payload["sample"]["systems"]["B"]["pressures_bar"] = [1, 11]
            run.write_text(json.dumps(payload), encoding="utf-8")

            main([str(run)], backend_composition=bundle)
            self.assertTrue(sample_sentinel.is_file())
            self.assertTrue(screen_sentinel.is_file())

            main([str(run), "--resume", "r0/sample"], backend_composition=bundle)
            self.assertFalse(sample_sentinel.exists())
            self.assertFalse(screen_sentinel.exists())
            contexts = sorted(sample.glob("tasks/*/task_context.json"))
            self.assertEqual(len(contexts), 6)
            self.assertEqual(json.loads(contexts[-1].read_text(encoding="utf-8"))["pressure_bar"], 11.0)
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())

            train_sentinel = root / "iter0" / "r0" / "train" / "preserve-on-artifact-repair"
            train_sentinel.write_text("train", encoding="utf-8")
            corrupt = json.loads(contexts[-1].read_text(encoding="utf-8"))
            corrupt["pressure_bar"] = 12.0
            contexts[-1].write_text(json.dumps(corrupt, sort_keys=True), encoding="utf-8")
            main([str(run)], backend_composition=bundle)
            self.assertTrue(train_sentinel.is_file())
            repaired = json.loads(contexts[-1].read_text(encoding="utf-8"))
            self.assertEqual(repaired["pressure_bar"], 12.0)
            self.assertFalse(any(path.name == "observations.xyz" for path in root.rglob("*")))

    def test_old_common_only_sample_marker_is_inert_without_rewrite(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            marker = root / ".dice_state" / "markers" / "iter0" / "r0__sample.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(
                json.dumps(
                    {
                        "schema": "dice.v2.block-marker.v1",
                        "status": "success",
                        "block": "r0/sample",
                        "iter": 0,
                        "fingerprint": "retired-common-only-sample",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            context = root / "iter0" / "r0" / "sample" / "tasks" / "task_000000" / "task_context.json"
            before = {"run": run.read_bytes(), "marker": marker.read_bytes(), "context": context.read_bytes()}

            main([str(run)], backend_composition=bundle)

            self.assertEqual(run.read_bytes(), before["run"])
            self.assertEqual(marker.read_bytes(), before["marker"])
            self.assertEqual(context.read_bytes(), before["context"])

    def test_legacy_lammps_sample_resume_is_read_only_and_ignores_retired_index(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample = root / "iter0" / "r0" / "sample"
            (sample / "seeds.xyz").unlink()
            for context in sample.glob("tasks/*/task_context.json"):
                context.unlink()
            retired = root / ".dice_state" / "indexes" / "iter0" / "r0__sample.observations.xyz"
            retired.parent.mkdir(parents=True, exist_ok=True)
            retired.write_text("corrupt retired index", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "required result r0/sample for r0/screen is not consumer-runnable",
            ):
                main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertFalse((sample / "seeds.xyz").exists())
            self.assertFalse(any(sample.glob("tasks/*/task_context.json")))
            self.assertTrue(retired.exists())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_legacy_lammps_mismatched_seed_rejects_resume_before_screen_cleanup(self) -> None:
        from workflow_core.structures.formats import write_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample = root / "iter0" / "r0" / "sample"
            (sample / "seeds.xyz").unlink()
            for context in sample.glob("tasks/*/task_context.json"):
                context.unlink()
            write_extxyz_structures(
                sample / "tasks" / "task_000000" / "seed.xyz",
                [Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[4, 4, 4], pbc=True)],
                allow_empty=False,
            )
            sentinel = root / "iter0" / "r0" / "screen" / "must_survive.txt"
            sentinel.write_text("prefix not cleaned", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "required result r0/sample for r0/screen is not consumer-runnable",
            ):
                main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertTrue(sentinel.is_file())

    def test_legacy_gpumd_sample_resume_is_read_only_and_ignores_retired_index(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root,
                teacher_model="committee",
                student_backend="NEP",
                student_model="committee",
            )
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeNepStudent(),
                sampling=_FakeGpumdSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample = root / "iter0" / "r0" / "sample"
            (sample / "seeds.xyz").unlink()
            for context in sample.glob("tasks/*/task_context.json"):
                context.unlink()
            retired = root / ".dice_state" / "indexes" / "iter0" / "r0__sample.observations.xyz"
            retired.parent.mkdir(parents=True, exist_ok=True)
            retired.write_text("corrupt retired index", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "required result r0/sample for r0/screen is not consumer-runnable",
            ):
                main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertFalse((sample / "seeds.xyz").exists())
            self.assertFalse(any(sample.glob("tasks/*/task_context.json")))
            self.assertTrue(retired.exists())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())

    def test_pair_guard_all_reject_short_circuits_gmm_and_keeps_later_train_sources(self) -> None:
        from workflow_artifacts.lifecycle.gmm_screen import validate_gmm_screen_summary
        from workflow_artifacts.lifecycle.pair_guard import load_pair_guard
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.pair_guard import PairGuardThresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.51}
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()
            student = _PairGuardRecordingStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=student,
                    sampling=_pair_guard_sampler(root),
                ),
            )

            self.assertEqual(teacher.score_identity_calls, [])
            self.assertEqual(student.score_identity_calls, [])
            self.assertEqual(student.reference_calls, [])
            self.assertEqual(len(teacher.label_identity_calls), 1)
            for round_index in range(2):
                screen = root / "iter0" / f"r{round_index}" / "screen"
                guard = load_pair_guard(
                    screen / "pair_guard.npz",
                    thresholds=PairGuardThresholds(minimum=0.51, pairs={}),
                )
                self.assertEqual(guard.passed.tolist(), [False, False])
                self.assertEqual(read_extxyz_structures(screen / "train.xyz", allow_empty=True), [])
                self.assertEqual(read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True), [])
                for name in (
                    "gmm_router",
                    "gmm_scores.npz",
                    "representative_features.npz",
                    "student_representative_features.npz",
                    ".screen_work",
                ):
                    self.assertFalse((screen / name).exists(), name)
                summary = validate_gmm_screen_summary(screen / "screen_summary.json")
                self.assertEqual(
                    {name: summary["counts"][name] for name in ("observations", "guard_pass", "guard_reject", "teacher_scored", "train", "fp_candidate")},
                    {"observations": 2, "guard_pass": 0, "guard_reject": 2, "teacher_scored": 0, "train": 0, "fp_candidate": 0},
                )

            later_dataset = read_extxyz_structures(
                root / "iter0" / "r1" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            self.assertEqual(len(later_dataset), 8)

    def test_pair_guard_partial_pass_precedes_every_single_gmm_operation(self) -> None:
        from workflow_artifacts.lifecycle.pair_guard import load_pair_guard
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.pair_guard import PairGuardThresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()
            student = _PairGuardRecordingStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=student,
                    sampling=_pair_guard_sampler(root),
                ),
            )

            for round_index in range(2):
                screen = root / "iter0" / f"r{round_index}" / "screen"
                guard = load_pair_guard(
                    screen / "pair_guard.npz",
                    thresholds=PairGuardThresholds(minimum=0.5, pairs={}),
                )
                pass_ids = tuple(
                    str(identity)
                    for identity, passed in zip(guard.structure_ids.tolist(), guard.passed.tolist(), strict=True)
                    if passed
                )
                reject_ids = {
                    str(identity)
                    for identity, passed in zip(guard.structure_ids.tolist(), guard.passed.tolist(), strict=True)
                    if not passed
                }
                self.assertEqual(guard.passed.tolist(), [True, False])
                with np.load(screen / "gmm_scores.npz", allow_pickle=False) as scores:
                    self.assertEqual(tuple(scores["structure_ids"].tolist()), pass_ids)
                summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["thresholds"]["pair_guard"], {"min": 0.5, "pairs": {}})
                self.assertEqual(
                    {name: summary["counts"][name] for name in ("observations", "guard_pass", "guard_reject", "teacher_scored")},
                    {"observations": 2, "guard_pass": 1, "guard_reject": 1, "teacher_scored": 1},
                )
                self.assertEqual(summary["per_system"]["water"]["guard_reject"], 1)
                self.assertFalse(any(path.suffix == ".csv" or "event" in path.name for path in screen.rglob("*")))
                self.assertEqual(
                    {path.name for path in screen.glob("*.xyz")},
                    {"train.xyz", "fp_candidate.xyz"},
                )
                self.assertFalse(any(identity in call for identity in reject_ids for call in teacher.score_identity_calls))
                self.assertFalse(any(identity in call for identity in reject_ids for call in student.score_identity_calls))
                self.assertFalse(any(identity in call for identity in reject_ids for call in teacher.label_identity_calls))

    def test_pair_guard_has_own_logged_phase_before_student_scoring(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.49}
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_PairGuardRecordingTeacher(),
                    student=_PairGuardRecordingStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            status = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            phases = (
                "r0/screen [RUNNING] phase=catalog",
                "r0/screen [RUNNING] phase=pair guard",
                "r0/screen [RUNNING] phase=student scoring",
            )
            self.assertTrue(all(phase in status for phase in phases), status)
            self.assertEqual(
                sorted(status.index(phase) for phase in phases),
                [status.index(phase) for phase in phases],
            )

    def test_pair_guard_off_and_all_pass_preserve_single_gmm_route(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        for configured in (False, True):
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run = _write_public_run(root)
                if configured:
                    payload = json.loads(run.read_text(encoding="utf-8"))
                    payload["screen"]["pair_guard"] = {"min": 0.49}
                    run.write_text(json.dumps(payload), encoding="utf-8")
                teacher = _PairGuardRecordingTeacher()
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=teacher,
                        student=_PairGuardRecordingStudent(),
                        sampling=_pair_guard_sampler(root),
                    ),
                )
                for round_index in range(2):
                    screen = root / "iter0" / f"r{round_index}" / "screen"
                    self.assertEqual((screen / "pair_guard.npz").is_file(), configured)
                    with np.load(screen / "gmm_scores.npz", allow_pickle=False) as scores:
                        self.assertEqual(len(scores["structure_ids"]), 2)
                        self.assertTrue(scores["student_scored"].all())
                    summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["counts"]["guard_reject"], 0)
                    self.assertNotIn("gate_order", summary)
                    self.assertEqual(summary["counts"]["normal_teacher_scored"], 1)
                    self.assertEqual(summary["counts"]["teacher_scored"], 1)

    def test_pair_guard_partial_pass_precedes_committee_label_and_keeps_committee_schema(self) -> None:
        from workflow_artifacts.lifecycle.pair_guard import load_pair_guard
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.pair_guard import PairGuardThresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            for round_index in range(2):
                screen = root / "iter0" / f"r{round_index}" / "screen"
                guard = load_pair_guard(
                    screen / "pair_guard.npz",
                    thresholds=PairGuardThresholds(minimum=0.5, pairs={}),
                )
                reject_ids = {
                    str(identity)
                    for identity, passed in zip(guard.structure_ids.tolist(), guard.passed.tolist(), strict=True)
                    if not passed
                }
                self.assertEqual(guard.passed.tolist(), [True, False])
                self.assertFalse((screen / "gmm_scores.npz").exists())
                self.assertTrue((screen / "committee_force_scores.npz").is_file())
                summary = json.loads(
                    (screen / "screen_summary.json").read_text(encoding="utf-8")
                )
                self.assertTrue(summary["fill_underfilled_from_safe_region"])
                self.assertEqual(summary["counts"]["guard_pass"], 1)
                self.assertEqual(summary["counts"]["guard_reject"], 1)
                self.assertFalse(any(identity in call for identity in reject_ids for call in teacher.label_identity_calls))
                from workflow_core.structures.formats import read_extxyz_structures

                self.assertEqual(len(read_extxyz_structures(screen / "train.xyz", allow_empty=False)), 1)
                self.assertEqual(read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True), [])

    def test_pair_guard_all_reject_short_circuits_committee_without_gmm_summary(self) -> None:
        from workflow_artifacts.lifecycle.pair_guard import load_pair_guard
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.pair_guard import PairGuardThresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.51}
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            self.assertEqual(len(teacher.label_identity_calls), 1)
            for round_index in range(2):
                screen = root / "iter0" / f"r{round_index}" / "screen"
                guard = load_pair_guard(
                    screen / "pair_guard.npz",
                    thresholds=PairGuardThresholds(minimum=0.51, pairs={}),
                )
                self.assertEqual(guard.passed.tolist(), [False, False])
                self.assertEqual(read_extxyz_structures(screen / "train.xyz", allow_empty=True), [])
                self.assertEqual(read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True), [])
                for name in (
                    "gmm_router",
                    "gmm_scores.npz",
                    "screen_summary.json",
                    "representative_features.npz",
                    ".screen_work",
                ):
                    self.assertFalse((screen / name).exists(), name)

    def test_pair_guard_all_reject_committee_element_fallback_completes(self) -> None:
        from workflow_artifacts.lifecycle.policy_resolution import (
            load_policy_resolution,
        )
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="committee"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["visualize"] = False
            payload["screen"].update(
                pair_guard={"min": 0.51}, policy="ELEMENT_DIRECT"
            )
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_PairGuardRecordingTeacher(),
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            for round_index in range(2):
                screen = root / "iter0" / f"r{round_index}" / "screen"
                evidence = load_policy_resolution(
                    screen / "policy_resolution.json"
                )
                self.assertEqual(evidence["rows"][0]["candidate_ids"], [])
                self.assertFalse((root / ".dice_state" / "markers").exists())

    def test_gmm_screen_summary_is_diagnostic_only_for_completion(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()
            student = _PairGuardRecordingStudent()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=student,
                sampling=_pair_guard_sampler(root),
            )

            main([str(run)], backend_composition=bundle)
            summary = root / "iter0" / "r0" / "screen" / "screen_summary.json"
            score_calls = list(teacher.score_identity_calls)
            student_calls = list(student.score_identity_calls)
            summary.unlink()

            main([str(run)], backend_composition=bundle)

            self.assertFalse(summary.exists())
            self.assertEqual(teacher.score_identity_calls, score_calls)
            self.assertEqual(student.score_identity_calls, student_calls)

    def test_corrupt_pair_guard_rebuilds_screen_and_preserves_sample_prefix(self) -> None:
        from workflow_artifacts.lifecycle.pair_guard import load_pair_guard
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.pair_guard import PairGuardThresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_PairGuardRecordingStudent(),
                sampling=_pair_guard_sampler(root),
            )
            main([str(run)], backend_composition=bundle)
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            sample_sentinel.write_text("sample", encoding="utf-8")
            score_call_count = len(teacher.score_identity_calls)
            audit = root / "iter0" / "r0" / "screen" / "pair_guard.npz"
            audit.write_bytes(b"corrupt")

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertTrue(sample_sentinel.is_file())
            self.assertGreater(len(teacher.score_identity_calls), score_call_count)
            guard = load_pair_guard(
                audit,
                thresholds=PairGuardThresholds(minimum=0.5, pairs={}),
            )
            self.assertEqual(guard.passed.tolist(), [True, False])

    def test_pair_guard_drift_requires_then_honors_targeted_screen_resume(self) -> None:
        from workflow_artifacts.lifecycle.pair_guard import load_pair_guard
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.pair_guard import PairGuardThresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.49}
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_pair_guard_sampler(root),
            )
            main([str(run)], backend_composition=bundle)
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            screen_sentinel = root / "iter0" / "r0" / "screen" / "replace.me"
            downstream_sentinel = root / "iter0" / "r1" / "sample" / "replace.me"
            for path in (sample_sentinel, screen_sentinel, downstream_sentinel):
                path.write_text(path.name, encoding="utf-8")

            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=bundle)
            self.assertTrue(screen_sentinel.is_file())

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertTrue(sample_sentinel.is_file())
            self.assertFalse(screen_sentinel.exists())
            self.assertFalse(downstream_sentinel.exists())
            guard = load_pair_guard(
                root / "iter0" / "r0" / "screen" / "pair_guard.npz",
                thresholds=PairGuardThresholds(minimum=0.5, pairs={}),
            )
            self.assertEqual(guard.passed.tolist(), [True, False])

    def test_guarded_screen_score_failure_leaves_no_partial_publication_and_can_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "injected guarded screen score failure"):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_PairGuardFailingScoreTeacher(),
                        student=_FakeStudent(),
                        sampling=_pair_guard_sampler(root),
                    ),
                )

            sample = root / "iter0" / "r0" / "sample"
            self.assertTrue(sample.is_dir())
            failed_screen = root / "iter0" / "r0" / "screen"
            self.assertTrue((failed_screen / ".attempts").is_dir())
            self.assertFalse((failed_screen / "train.xyz").exists())
            self.assertFalse((failed_screen / "fp_candidate.xyz").exists())
            self.assertFalse((root / "iter0" / "r0" / ".screen.staging").exists())
            self.assertFalse((root / ".dice_state" / "markers" / "iter0" / "r0__screen.json").exists())
            sentinel = sample / "preserve.me"
            sentinel.write_text("sample", encoding="utf-8")

            main(
                [str(run), "--resume", "r0/screen"],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )
            self.assertTrue(sentinel.is_file())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "pair_guard.npz").is_file())

    def test_later_iter_student_sources_follow_prior_iter_actual_rounds(self) -> None:
        from workflow_core.lifecycle.runner import _prior_iter_student_train_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            prior = Path(tmpdir) / "iter0"
            expected = (
                prior / "r0" / "train" / "dataset" / "train.xyz",
                prior / "r0" / "screen" / "train.xyz",
            )
            for path in expected:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("canonical source", encoding="utf-8")

            self.assertEqual(_prior_iter_student_train_sources(prior), expected)

    def test_enabled_teacher_train_union_is_relabeled_in_teach_screen_and_persists_through_r1(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.exact_identity import exact_structure_hash
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            teacher_input = read_extxyz_structures(
                root / "data" / "teacher.xyz", allow_empty=False
            )
            for index, frame in enumerate(teacher_input):
                frame.calc = SinglePointCalculator(
                    frame,
                    energy=float(index + 10),
                    forces=np.full((len(frame), 3), float(index + 10)),
                )
            write_extxyz_structures(root / "data" / "teacher.xyz", teacher_input)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["include_teacher_train_data"] = True
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            student_bootstrap = read_extxyz_structures(
                root / "data" / "student.xyz", allow_empty=False
            )
            teacher_train = read_extxyz_structures(
                root / "iter0" / "teach" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            expected_hashes = []
            seen_hashes = set()
            for frame in [*student_bootstrap, *teacher_train]:
                geometry = exact_structure_hash(frame)
                if geometry not in seen_hashes:
                    expected_hashes.append(geometry)
                    seen_hashes.add(geometry)

            relabeled = read_extxyz_structures(
                root / "iter0" / "teach" / "screen" / "train.xyz",
                allow_empty=False,
            )
            relabeled_hashes = [exact_structure_hash(frame) for frame in relabeled]
            self.assertEqual(relabeled_hashes, expected_hashes)
            self.assertTrue(
                all(frame.get_potential_energy() == 0.0 for frame in relabeled)
            )

            r0_dataset = read_extxyz_structures(
                root / "iter0" / "r0" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            r0_hashes = [exact_structure_hash(frame) for frame in r0_dataset]
            self.assertEqual(r0_hashes, relabeled_hashes)

            r1_dataset = read_extxyz_structures(
                root / "iter0" / "r1" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            r1_hashes = [exact_structure_hash(frame) for frame in r1_dataset]
            self.assertEqual(r1_hashes[: len(r0_hashes)], r0_hashes)
            self.assertTrue(set(r0_hashes).issubset(r1_hashes))

    def test_teacher_relabel_union_uses_only_the_committed_teacher_train_split(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.exact_identity import exact_structure_hash
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="single"
            )
            teacher_frames = []
            for index in range(8):
                frame = Atoms(
                    "H",
                    positions=[[1.0 + 0.1 * index, 0.0, 0.0]],
                    cell=[4, 4, 4],
                    pbc=True,
                )
                frame.calc = SinglePointCalculator(
                    frame,
                    energy=float(index),
                    forces=np.zeros((1, 3)),
                )
                teacher_frames.append(frame)
            write_extxyz_structures(root / "data" / "teacher.xyz", teacher_frames)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["teacher"]["valid_frac"] = 0.25
            payload["student"]["include_teacher_train_data"] = True
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            teacher_train_hashes = {
                exact_structure_hash(frame)
                for frame in read_extxyz_structures(
                    root / "iter0" / "teach" / "train" / "dataset" / "train.xyz",
                    allow_empty=False,
                )
            }
            teacher_valid_hashes = {
                exact_structure_hash(frame)
                for frame in read_extxyz_structures(
                    root / "iter0" / "teach" / "train" / "dataset" / "valid.xyz",
                    allow_empty=False,
                )
            }
            relabeled_hashes = {
                exact_structure_hash(frame)
                for frame in read_extxyz_structures(
                    root / "iter0" / "teach" / "screen" / "train.xyz",
                    allow_empty=False,
                )
            }
            self.assertTrue(teacher_train_hashes.issubset(relabeled_hashes))
            self.assertTrue(teacher_valid_hashes)
            self.assertTrue(teacher_valid_hashes.isdisjoint(relabeled_hashes))

    def test_omitted_teacher_union_direct_predecessor_matches_legacy_cumulative_bytes(self) -> None:
        from workflow_core.lifecycle.data import DataService, LabelContract
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            service = DataService(
                run_json_path=run,
                iter_index=0,
                teacher_type_map=("H",),
                student_type_map=("H",),
                teacher_contract=LabelContract(),
                student_contract=LabelContract(),
            )
            legacy_expected = service.materialize_student_dataset(
                [
                    read_extxyz_structures(
                        root / "iter0" / "teach" / "screen" / "train.xyz",
                        allow_empty=False,
                    ),
                    read_extxyz_structures(
                        root / "iter0" / "r0" / "screen" / "train.xyz",
                        allow_empty=True,
                    ),
                ],
                root / "legacy-r1-train.xyz",
            )
            current = (
                root / "iter0" / "r1" / "train" / "dataset" / "train.xyz"
            )
            self.assertEqual(current.read_bytes(), legacy_expected.read_bytes())

    def test_later_iter_relabels_union_of_final_student_predecessor_and_current_teacher_train(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.exact_identity import exact_structure_hash
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["include_teacher_train_data"] = True
            run.write_text(json.dumps(payload), encoding="utf-8")
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            prior_final_train = read_extxyz_structures(
                root / "iter0" / "r1" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            prior_final_screen = read_extxyz_structures(
                root / "iter0" / "r1" / "screen" / "train.xyz",
                allow_empty=True,
            )

            current_teacher_only = Atoms(
                "H",
                positions=[[0.77, 0.0, 0.0]],
                cell=[4.0, 4.0, 4.0],
                pbc=True,
            )
            current_teacher_only.calc = SinglePointCalculator(
                current_teacher_only,
                energy=-1.0,
                forces=np.full((1, 3), -1.0),
            )
            write_extxyz_structures(
                root / "data" / "teacher-iter1.xyz",
                [current_teacher_only],
            )
            payload["iter"] = 1
            payload["student_train_bootstrap"] = []
            payload["teacher_train_bootstrap"] = ["data/teacher-iter1.xyz"]
            payload["fp_data"] = ["data/teacher.xyz"]
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            current_teacher_train = read_extxyz_structures(
                root / "iter1" / "teach" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            expected_hashes = []
            seen_hashes = set()
            for frame in [
                *prior_final_train,
                *prior_final_screen,
                *current_teacher_train,
            ]:
                geometry = exact_structure_hash(frame)
                if geometry not in seen_hashes:
                    expected_hashes.append(geometry)
                    seen_hashes.add(geometry)

            relabeled = read_extxyz_structures(
                root / "iter1" / "teach" / "screen" / "train.xyz",
                allow_empty=False,
            )
            relabeled_hashes = [exact_structure_hash(frame) for frame in relabeled]
            self.assertEqual(relabeled_hashes, expected_hashes)
            current_geometry = exact_structure_hash(current_teacher_only)
            current_row = next(
                frame
                for frame in relabeled
                if exact_structure_hash(frame) == current_geometry
            )
            self.assertEqual(current_row.get_potential_energy(), 0.0)

            current_r0 = read_extxyz_structures(
                root / "iter1" / "r0" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            self.assertEqual(
                [exact_structure_hash(frame) for frame in current_r0],
                relabeled_hashes,
            )

    def test_teacher_relabel_union_validation_failure_is_atomic_and_calls_no_backend(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        class _RecordingTeacher(_FakeTeacher):
            def __init__(self) -> None:
                super().__init__()
                self.label_calls = 0

            def label(self, models, frames, *args, **kwargs):
                self.label_calls += 1
                return super().label(models, frames, *args, **kwargs)

        class _RecordingStudent(_FakeStudent):
            def __init__(self) -> None:
                self.train_calls = 0

            def train(self, input_json, member_dir, *args):
                self.train_calls += 1
                return super().train(input_json, member_dir, *args)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="single"
            )
            teacher_frames = read_extxyz_structures(
                root / "data" / "teacher.xyz", allow_empty=False
            )
            incompatible = Atoms(
                "O", positions=[[0.5, 0.0, 0.0]], cell=[4, 4, 4], pbc=True
            )
            incompatible.calc = SinglePointCalculator(
                incompatible,
                energy=0.0,
                forces=np.zeros((1, 3)),
            )
            write_extxyz_structures(
                root / "data" / "teacher.xyz",
                [*teacher_frames, incompatible],
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["teacher"]["type_map"] = ["H", "O"]
            payload["student"]["include_teacher_train_data"] = True
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _RecordingTeacher()
            student = _RecordingStudent()

            with self.assertRaisesRegex(ValueError, "unsupported elements"):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(
                        run,
                        teacher=teacher,
                        student=student,
                        sampling=_FakeSampler(),
                    ),
                )

            self.assertEqual(teacher.label_calls, 0)
            self.assertEqual(student.train_calls, 0)
            self.assertTrue(
                (root / "iter0" / "teach" / "train" / "dataset" / "train.xyz").is_file()
            )
            failed_screen = root / "iter0" / "teach" / "screen"
            self.assertFalse((failed_screen / "train.xyz").exists())
            self.assertFalse((failed_screen / "fp_candidate.xyz").exists())
            self.assertFalse(
                (
                    root
                    / ".dice_state"
                    / "markers"
                    / "iter0"
                    / "teach__screen.json"
                ).exists()
            )
            self.assertFalse((root / "iter0" / "r0").exists())

    def test_teacher_union_toggle_requires_teach_screen_resume_and_preserves_teach_train(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.exact_identity import exact_structure_hash
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)

            def composition():
                return compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                )

            main([str(run)], backend_composition=composition())
            teach_train_sentinel = (
                root / "iter0" / "teach" / "train" / "preserve.me"
            )
            teach_screen_sentinel = (
                root / "iter0" / "teach" / "screen" / "replace.me"
            )
            r0_sentinel = root / "iter0" / "r0" / "train" / "replace.me"
            r1_sentinel = root / "iter0" / "r1" / "train" / "replace.me"
            for sentinel in (
                teach_train_sentinel,
                teach_screen_sentinel,
                r0_sentinel,
                r1_sentinel,
            ):
                sentinel.write_text(sentinel.name, encoding="utf-8")

            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["include_teacher_train_data"] = False
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=composition())
            self.assertTrue(teach_screen_sentinel.is_file())

            payload["student"]["include_teacher_train_data"] = True
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=composition())
            self.assertTrue(teach_train_sentinel.is_file())
            self.assertTrue(teach_screen_sentinel.is_file())
            self.assertTrue(r0_sentinel.is_file())
            self.assertTrue(r1_sentinel.is_file())

            main(
                [str(run), "--resume", "teach/screen"],
                backend_composition=composition(),
            )
            self.assertTrue(teach_train_sentinel.is_file())
            self.assertFalse(teach_screen_sentinel.exists())
            self.assertFalse(r0_sentinel.exists())
            self.assertFalse(r1_sentinel.exists())

            relabeled = read_extxyz_structures(
                root / "iter0" / "teach" / "screen" / "train.xyz",
                allow_empty=False,
            )
            teacher_train = read_extxyz_structures(
                root / "iter0" / "teach" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            relabeled_hashes = {
                exact_structure_hash(frame) for frame in relabeled
            }
            self.assertTrue(
                {
                    exact_structure_hash(frame) for frame in teacher_train
                }.issubset(relabeled_hashes)
            )
            rebuilt = read_extxyz_structures(
                root / "iter0" / "r0" / "train" / "dataset" / "train.xyz",
                allow_empty=False,
            )
            self.assertEqual(
                [exact_structure_hash(frame) for frame in rebuilt],
                [exact_structure_hash(frame) for frame in relabeled],
            )

    def test_single_gmm_student_first_fp_scope_recovers_corrupt_audit(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["fp_pool"] = 2
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            r0 = root / "iter0" / "r0" / "screen"
            self.assertEqual(len(read_extxyz_structures(r0 / "train.xyz", allow_empty=True)), 1)
            self.assertEqual(len(read_extxyz_structures(r0 / "fp_candidate.xyz", allow_empty=True)), 1)
            self.assertTrue((r0 / "gmm_router" / "manifest.json").is_file())
            self.assertTrue((r0 / "gmm_scores.npz").is_file())
            self.assertFalse((r0 / ".screen_work").exists())
            (r0 / "gmm_scores.npz").write_bytes(b"corrupt")
            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)
            with np.load(r0 / "gmm_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(str(audit["schema"].item()), "dice.lifecycle.gmm-screen-scores")
                self.assertEqual(int(audit["schema_revision"].item()), 5)
                self.assertEqual(str(audit["policy"].item()), "gmm-reference-tail")
                self.assertEqual(int(audit["policy_revision"].item()), 4)

    def test_single_teacher_committee_student_mixed_route_runs_full_iter_with_overlap_audit(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"].update({"pool": [1], "keep": [1], "fp_pool": 2})
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            train = read_extxyz_structures(screen / "train.xyz", allow_empty=True)
            fp_candidates = read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)
            self.assertEqual(len(train), 1)
            self.assertEqual(len(fp_candidates), 1)
            self.assertIn(
                train[0].info["structure_id"],
                {frame.info["structure_id"] for frame in fp_candidates},
            )
            self.assertFalse((screen / "gmm_router").exists())
            self.assertFalse((screen / "gmm_scores.npz").exists())
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(str(audit["schema"].item()), "dice.lifecycle.mixed-route-audit")
                self.assertEqual(int(audit["schema_revision"].item()), 3)
                self.assertEqual(
                    str(audit["policy"].item()),
                    "student-force-deviation_teacher-reference-tail",
                )
                identities = tuple(str(item) for item in audit["structure_ids"].tolist())
                self.assertEqual(len(identities), 2)
                self.assertIn(str(train[0].info["structure_id"]), identities)
                self.assertEqual(int(np.count_nonzero(audit["normal_teacher_scored"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["teacher_scored"])), 1)
                self.assertTrue(
                    np.array_equal(
                        audit["teacher_p_applicable"], audit["teacher_scored"]
                    )
                )
                self.assertTrue(audit["student_force_deviation_applicable"].all())
                self.assertTrue(audit["student_in_band"].all())
                self.assertEqual(int(np.count_nonzero(audit["student_pool"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["train_selected"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["fp_candidate"])), 1)
                self.assertFalse(any("descriptor" in field for field in audit.files))
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["policy_identity"],
                {"name": "student-force-deviation_teacher-reference-tail", "revision": 1},
            )
            self.assertEqual(summary["counts"]["train_selected"], 1)
            self.assertEqual(summary["counts"]["fp_candidate"], 1)
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
            history = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            for phase in ("student proposal", "teacher scoring", "post-teacher selection", "finalizing"):
                self.assertIn(f"r0/screen [RUNNING] phase={phase}", history)
            self.assertNotIn("student score", history.lower())
            self.assertNotIn("student gmm", history.lower())
            self.assertFalse(any(path.name == "observations.xyz" for path in screen.rglob("*")))

    def test_single_teacher_committee_student_partial_guard_audits_only_pass_rows_and_counts_rejects(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 500]
            payload["screen"].update({
                "pool": [1],
                "keep": [1],
                "pair_guard": {"min": 0.5},
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            with np.load(screen / "pair_guard.npz", allow_pickle=False) as guard:
                pass_ids = tuple(
                    str(identity)
                    for identity, passed in zip(
                        guard["structure_ids"].tolist(), guard["passed"].tolist(), strict=True
                    )
                    if bool(passed)
                )
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(tuple(str(item) for item in audit["structure_ids"].tolist()), pass_ids)
            self.assertEqual(teacher.score_identity_calls[-1], pass_ids)
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["observations"], 2)
            self.assertEqual(summary["counts"]["guard_pass"], 1)
            self.assertEqual(summary["counts"]["guard_reject"], 1)

    def test_single_teacher_committee_student_all_guard_reject_skips_mixed_work(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"].update({
                "pool": [1],
                "keep": [1],
                "pair_guard": {"min": 0.51},
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            self.assertEqual(teacher.score_identity_calls, [])
            self.assertEqual(read_extxyz_structures(screen / "train.xyz", allow_empty=True), [])
            self.assertEqual(
                read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True), []
            )
            self.assertTrue((screen / "pair_guard.npz").is_file())
            for name in (
                "mixed_route_scores.npz",
                "screen_summary.json",
                "gmm_router",
                "gmm_scores.npz",
                "representative_features.npz",
                "student_representative_features.npz",
            ):
                self.assertFalse((screen / name).exists(), name)
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_teacher_first_mixed_route_is_rejected_before_execution(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["gate_order"] = "teacher-first"
            run.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported field.*gate_order"):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(
                        run,
                        teacher=_FakeTeacher(),
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ),
                )
            self.assertFalse((root / "iter0").exists())


    def test_single_teacher_committee_student_direct_fps_use_role_native_evidence(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        for policy in ("DIRECT", "FPS"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run = _write_public_run(root, student_model="committee")
                payload = json.loads(run.read_text(encoding="utf-8"))
                payload["sample"]["steps"] = [20]
                payload["screen"].update({
                    "pool": [2],
                    "keep": [1],
                    "fp_pool": 2,
                    "policy": policy,
                })
                payload["fp"]["policy"] = policy
                run.write_text(json.dumps(payload), encoding="utf-8")
                teacher = _PairGuardRecordingTeacher()

                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=teacher,
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ),
                )

                screen = root / "iter0" / "r0" / "screen"
                train = read_extxyz_structures(screen / "train.xyz", allow_empty=True)
                self.assertEqual(len(train), 1)
                self.assertEqual(teacher.label_identity_calls[-1], (str(train[0].info["structure_id"]),))
                with np.load(screen / "representative_features.npz", allow_pickle=False) as evidence:
                    self.assertEqual(evidence["means"].shape, (2, 128))
                    self.assertIn('"name":"readout-input"', str(evidence["descriptor_schema"].item()))
                    self.assertIn('"revision":1', str(evidence["descriptor_schema"].item()))
                with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                    self.assertTrue(np.isfinite(audit["student_force_deviation"]).all())
                self.assertFalse((screen / "student_representative_features.npz").exists())
                self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_single_teacher_committee_student_partial_pass_allows_empty_fp_view(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"].update({
                "pool": [2],
                "keep": [1],
                "teacher_p": {"floor": 0.0, "trust": 0.0, "ceiling": 0.05},
            })
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            self.assertEqual(len(read_extxyz_structures(screen / "train.xyz", allow_empty=True)), 1)
            self.assertEqual(
                read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True), []
            )
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(int(np.count_nonzero(audit["fp_candidate"])), 0)
                self.assertEqual(int(np.count_nonzero(audit["train_selected"])), 1)
            self.assertEqual(
                read_extxyz_structures(root / "iter0" / "fp" / "selected.xyz", allow_empty=True),
                [],
            )

    def test_single_teacher_committee_student_empty_proposal_safe_fills_train_only(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"].update({
                "pool": [2],
                "keep": [1],
                "lo": 0.1,
                "hi": 0.2,
            })
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            self.assertEqual(
                len(read_extxyz_structures(screen / "train.xyz", allow_empty=True)), 1
            )
            self.assertEqual(
                len(read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)), 0
            )
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(int(np.count_nonzero(audit["train_selected"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["normal_teacher_scored"])), 0)
                self.assertEqual(int(np.count_nonzero(audit["safe_fill_attempted"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["fp_candidate"])), 0)

    def test_committee_teacher_single_student_bounded_pool_routes_low_and_middle(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["screen"].update({
                "pool": [2],
                "keep": [2],
                "student_p": {"floor": 0.0, "ceiling": 1.0},
                "policy": "random",
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _RoutingCommitteeTeacher()
            student = _PairGuardRecordingStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=student,
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            train = read_extxyz_structures(screen / "train.xyz", allow_empty=True)
            fp_candidates = read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)
            self.assertEqual(len(student.score_identity_calls[-1]), 3)
            self.assertEqual(len(teacher.label_identity_calls[-2]), 2)
            self.assertEqual(len(teacher.label_identity_calls[-1]), 1)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(fp_candidates), 1)
            self.assertFalse(
                {frame.info["structure_id"] for frame in train}
                & {frame.info["structure_id"] for frame in fp_candidates}
            )
            self.assertTrue((screen / "gmm_router" / "manifest.json").is_file())
            self.assertFalse((root / "iter0" / "teach" / "scan").exists())
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(str(audit["schema"].item()), "dice.lifecycle.mixed-route-audit")
                self.assertEqual(int(audit["schema_revision"].item()), 2)
                self.assertEqual(
                    str(audit["policy"].item()),
                    "teacher-force-deviation_student-reference-tail",
                )
                self.assertTrue(audit["student_p_applicable"].all())
                self.assertEqual(int(np.count_nonzero(audit["student_pool"])), 2)
                self.assertEqual(int(np.count_nonzero(audit["teacher_force_deviation_applicable"])), 3)
                self.assertEqual(int(np.count_nonzero(audit["teacher_low"])), 2)
                self.assertEqual(int(np.count_nonzero(audit["teacher_middle"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["train_selected"])), 2)
                self.assertEqual(int(np.count_nonzero(audit["fp_candidate"])), 1)
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["student_scored"], 3)
            self.assertEqual(summary["counts"]["teacher_evaluated"], 3)
            self.assertEqual(summary["counts"]["teacher_low"], 2)
            self.assertEqual(summary["counts"]["teacher_middle"], 1)

    def test_committee_teacher_single_student_reports_valid_underfill_without_backfill(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["screen"].update({"pool": [3], "keep": [3], "policy": "random"})
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _RoutingCommitteeTeacher()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            self.assertEqual(len(teacher.label_identity_calls[-1]), 3)
            self.assertEqual(len(read_extxyz_structures(screen / "train.xyz", allow_empty=True)), 1)
            self.assertEqual(
                len(read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)), 1
            )
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["teacher_low"], 1)
            self.assertEqual(summary["counts"]["teacher_middle"], 1)
            self.assertEqual(summary["counts"]["teacher_high"], 1)
            self.assertEqual(summary["counts"]["train_selected"], 1)
            self.assertEqual(summary["counts"]["train_underfill"], 2)

    def test_committee_teacher_single_student_pool_minus_one_evaluates_complete_p_band(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["screen"].update({
                "pool": [-1],
                "keep": [2],
                "student_p": {"floor": 0.0, "ceiling": 1.0},
                "policy": "random",
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _RoutingCommitteeTeacher()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                candidate_count = int(np.count_nonzero(audit["student_candidate"]))
                pool_count = int(np.count_nonzero(audit["student_pool"]))
                evaluated_count = int(np.count_nonzero(audit["teacher_evaluated"]))
                self.assertEqual(pool_count, candidate_count)
                self.assertGreaterEqual(evaluated_count, candidate_count)
                self.assertEqual(
                    int(np.count_nonzero(audit["normal_teacher_evaluated"])),
                    candidate_count,
                )
                self.assertEqual(
                    json.loads(str(audit["thresholds"].item()))["quota"]["pool"],
                    -1,
                )

    def test_committee_teacher_single_student_direct_fps_keep_role_native_evidence(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        for policy in ("DIRECT", "FPS"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run = _write_public_run(root, teacher_model="committee", student_model="single")
                payload = json.loads(run.read_text(encoding="utf-8"))
                payload["sample"]["steps"] = [20]
                payload["sample"]["temperatures_k"] = [300, 400, 500]
                payload["screen"].update({"pool": [3], "keep": [1], "policy": policy})
                payload["fp"]["policy"] = policy
                run.write_text(json.dumps(payload), encoding="utf-8")

                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_RoutingCommitteeTeacher(),
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ),
                )

                screen = root / "iter0" / "r0" / "screen"
                with np.load(screen / "student_representative_features.npz", allow_pickle=False) as evidence:
                    self.assertEqual(evidence["means"].shape, (1, 12))
                    self.assertIn('"name":"readout-input"', str(evidence["descriptor_schema"].item()))
                    self.assertIn('"revision":1', str(evidence["descriptor_schema"].item()))
                self.assertFalse((screen / "representative_features.npz").exists())
                with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                    self.assertEqual(len(audit["teacher_force_deviation"]), 3)
                    self.assertTrue(np.isfinite(audit["teacher_force_deviation"]).all())
                self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
                self.assertFalse((root / "iter0" / "teach" / "scan").exists())

    def test_committee_teacher_single_student_empty_train_is_successful_and_audited(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["screen"].update({"pool": [3], "keep": [2], "policy": "random"})
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_NoLowRoutingCommitteeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            self.assertEqual(read_extxyz_structures(screen / "train.xyz", allow_empty=True), [])
            self.assertEqual(
                len(read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)), 2
            )
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(int(np.count_nonzero(audit["teacher_low"])), 0)
                self.assertEqual(int(np.count_nonzero(audit["teacher_middle"])), 2)
                self.assertEqual(int(np.count_nonzero(audit["teacher_high"])), 1)
                self.assertEqual(int(np.count_nonzero(audit["train_selected"])), 0)
                self.assertEqual(int(np.count_nonzero(audit["fp_candidate"])), 2)
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["train_selected"], 0)
            self.assertEqual(summary["counts"]["train_underfill"], 2)

    def test_committee_teacher_single_student_pair_guard_precedes_student_and_teacher_work(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 500]
            payload["screen"].update({
                "pool": [1],
                "keep": [1],
                "pair_guard": {"min": 0.5},
                "policy": "random",
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _RoutingCommitteeTeacher()
            student = _PairGuardRecordingStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=student,
                    sampling=_pair_guard_sampler(root),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            with np.load(screen / "pair_guard.npz", allow_pickle=False) as guard:
                pass_ids = tuple(
                    str(identity)
                    for identity, passed in zip(
                        guard["structure_ids"].tolist(), guard["passed"].tolist(), strict=True
                    )
                    if bool(passed)
                )
            self.assertEqual(student.score_identity_calls[-1], pass_ids)
            self.assertEqual(teacher.label_identity_calls[-1], pass_ids)
            with np.load(screen / "mixed_route_scores.npz", allow_pickle=False) as audit:
                self.assertEqual(tuple(str(item) for item in audit["structure_ids"].tolist()), pass_ids)
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["observations"], 2)
            self.assertEqual(summary["counts"]["guard_pass"], 1)
            self.assertEqual(summary["counts"]["guard_reject"], 1)

    def test_committee_teacher_single_student_all_guard_reject_skips_both_model_roles(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 500]
            payload["screen"].update({
                "pool": [1],
                "keep": [1],
                "pair_guard": {"min": 0.51},
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _RoutingCommitteeTeacher()
            student = _PairGuardRecordingStudent()

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=student,
                    sampling=_pair_guard_sampler(root),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            self.assertEqual(student.reference_calls, [])
            self.assertEqual(student.score_identity_calls, [])
            self.assertEqual(len(teacher.label_identity_calls), 1)
            self.assertEqual(read_extxyz_structures(screen / "train.xyz", allow_empty=True), [])
            self.assertEqual(
                read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True), []
            )
            for name in ("gmm_router", "mixed_route_scores.npz", "screen_summary.json"):
                self.assertFalse((screen / name).exists(), name)

    def test_irrelevant_malformed_applicability_is_ignored_behind_terminal(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["screen"].update({"pool": [2], "keep": [2], "policy": "random"})
            run.write_text(json.dumps(payload), encoding="utf-8")
            student = _PairGuardRecordingStudent()
            bundle = compose_test_backend_roles(run,
                teacher=_RoutingCommitteeTeacher(),
                student=student,
                sampling=_FakeSampler(),
            )

            main([str(run)], backend_composition=bundle)
            audit_path = root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz"
            with np.load(audit_path, allow_pickle=False) as audit:
                corrupted = {name: np.asarray(audit[name]) for name in audit.files}
            corrupted["teacher_force_deviation_applicable"] = np.zeros_like(
                corrupted["teacher_force_deviation_applicable"], dtype=bool
            )
            with audit_path.open("wb") as handle:
                np.savez_compressed(handle, **corrupted)
            score_calls = len(student.score_identity_calls)

            main([str(run)], backend_composition=bundle)

            self.assertEqual(len(student.score_identity_calls), score_calls)
            with np.load(audit_path, allow_pickle=False) as retained:
                self.assertEqual(
                    int(
                        np.count_nonzero(
                            retained["teacher_force_deviation_applicable"]
                        )
                    ),
                    0,
                )

    def test_irrelevant_non_boolean_applicability_is_ignored_behind_terminal(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="single")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"].update({"pool": [1], "keep": [1]})
            run.write_text(json.dumps(payload), encoding="utf-8")
            student = _PairGuardRecordingStudent()
            bundle = compose_test_backend_roles(run,
                teacher=_RoutingCommitteeTeacher(),
                student=student,
                sampling=_FakeSingleModelSampler(),
            )
            main([str(run)], backend_composition=bundle)
            audit_path = root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz"
            with np.load(audit_path, allow_pickle=False) as audit:
                corrupted = {name: np.asarray(audit[name]) for name in audit.files}
            corrupted["student_p_applicable"] = np.ones(
                len(corrupted["student_p_applicable"]), dtype=np.int8
            )
            with audit_path.open("wb") as handle:
                np.savez_compressed(handle, **corrupted)
            score_calls = len(student.score_identity_calls)

            main([str(run)], backend_composition=bundle)

            self.assertEqual(len(student.score_identity_calls), score_calls)
            with np.load(audit_path, allow_pickle=False) as retained:
                self.assertEqual(
                    retained["student_p_applicable"].dtype,
                    np.dtype(np.int8),
                )

    def test_single_teacher_committee_student_rebuilds_a_valid_opposite_policy_audit(self) -> None:
        from workflow_artifacts.lifecycle.mixed_route import (
            load_mixed_route_audit,
            publish_committee_teacher_single_student_audit,
        )
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="single", student_model="committee")
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)

            screen = root / "iter0" / "r0" / "screen"
            audit_path = screen / "mixed_route_scores.npz"
            original = load_mixed_route_audit(audit_path)
            identities = tuple(str(item) for item in original["structure_ids"].tolist())
            systems = tuple(str(item) for item in original["system_ids"].tolist())
            observations = []
            evaluated = []
            for identity, system_id in zip(identities, systems, strict=True):
                observation = Atoms("H", positions=[[0.0, 0.0, 0.0]])
                observation.info.update(
                    structure_id=identity,
                    system_id=system_id,
                    student_p=0.5,
                )
                observations.append(observation)
                labeled = observation.copy()
                labeled.info["force_uncertainty"] = 0.2
                evaluated.append(labeled)
            train_ids = {
                str(frame.info["structure_id"])
                for frame in read_extxyz_structures(screen / "train.xyz", allow_empty=True)
            }
            fp_ids = {
                str(frame.info["structure_id"])
                for frame in read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)
            }
            publish_committee_teacher_single_student_audit(
                audit_path,
                observations=observations,
                student_flags={
                    identity: {
                        "student_floor_discard": False,
                        "student_candidate": True,
                        "student_pool": True,
                    }
                    for identity in identities
                },
                teacher_evaluated=evaluated,
                normal_teacher_evaluated=evaluated,
                train_ids=train_ids,
                normal_train_ids=train_ids,
                fp_ids=fp_ids,
                teacher_lo=0.1,
                teacher_hi=0.5,
                thresholds={"student_p": {"floor": 0.0}, "teacher_force_deviation": {"lo": 0.1, "hi": 0.5}},
                safe_fill_enabled=False,
            )
            score_calls = len(teacher.score_identity_calls)

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertGreater(len(teacher.score_identity_calls), score_calls)
            rebuilt = load_mixed_route_audit(audit_path)
            self.assertEqual(
                str(np.asarray(rebuilt["policy"]).item()),
                "student-force-deviation_teacher-reference-tail",
            )

    def test_single_teacher_committee_student_rebuilds_drifted_audit_thresholds(self) -> None:
        from workflow_artifacts.lifecycle.mixed_route import load_mixed_route_audit
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="single", student_model="committee")
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            audit_path = root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz"
            with np.load(audit_path, allow_pickle=False) as audit:
                corrupted = {name: np.asarray(audit[name]) for name in audit.files}
            corrupted["thresholds"] = np.asarray(
                json.dumps(
                    {
                        "teacher_p": {"floor": 0.0, "trust": 0.1},
                        "student_force_deviation": {"lo": 0.0, "hi": 0.25},
                        "quota": {"fp_pool": 999, "pool": 999, "keep": 999},
                        "selector": "random",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            with audit_path.open("wb") as handle:
                np.savez_compressed(handle, **corrupted)
            score_calls = len(teacher.score_identity_calls)

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertGreater(len(teacher.score_identity_calls), score_calls)
            self.assertEqual(
                load_mixed_route_audit(audit_path)["thresholds"]["quota"],
                {"fp_pool": 1, "pool": 1, "keep": 1},
            )

    def test_single_teacher_committee_student_rebuilds_inconsistent_gate_flags(self) -> None:
        from workflow_artifacts.lifecycle.mixed_route import load_mixed_route_audit
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="single", student_model="committee")
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            audit_path = root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz"
            with np.load(audit_path, allow_pickle=False) as audit:
                corrupted = {name: np.asarray(audit[name]) for name in audit.files}
            corrupted["teacher_trusted"] = np.zeros_like(
                corrupted["teacher_trusted"], dtype=np.bool_
            )
            with audit_path.open("wb") as handle:
                np.savez_compressed(handle, **corrupted)
            score_calls = len(teacher.score_identity_calls)

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertGreater(len(teacher.score_identity_calls), score_calls)
            rebuilt = load_mixed_route_audit(audit_path)
            self.assertTrue(np.any(rebuilt["teacher_trusted"]))

    def test_single_teacher_committee_student_rebuilds_misaligned_system_ids(self) -> None:
        from workflow_artifacts.lifecycle.mixed_route import load_mixed_route_audit
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="single", student_model="committee")
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            audit_path = root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz"
            with np.load(audit_path, allow_pickle=False) as audit:
                corrupted = {name: np.asarray(audit[name]) for name in audit.files}
            corrupted["system_ids"] = np.asarray(
                ["wrong-system"] * len(corrupted["system_ids"]), dtype=np.str_
            )
            with audit_path.open("wb") as handle:
                np.savez_compressed(handle, **corrupted)
            score_calls = len(teacher.score_identity_calls)

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertGreater(len(teacher.score_identity_calls), score_calls)
            rebuilt = load_mixed_route_audit(audit_path)
            self.assertEqual(set(rebuilt["system_ids"].tolist()), {"water"})

    def test_mixed_route_teacher_model_asset_drift_does_not_change_identity(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="single", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            screen_sentinel = root / "iter0" / "r0" / "screen" / "preserve.me"
            screen_sentinel.write_text("screen", encoding="utf-8")
            teacher_model = (
                root / "iter0" / "teach" / "train" / "teacher" / "teacher_model.model"
            )
            teacher_model.write_text("mutated-teacher-model", encoding="utf-8")

            main([str(run)], backend_composition=bundle)

            self.assertTrue(screen_sentinel.is_file())

    def test_mixed_route_student_model_asset_drift_does_not_change_identity(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="single", student_model="committee"
            )
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            sample_sentinel.write_text("sample", encoding="utf-8")
            student_model = (
                root
                / "iter0"
                / "r0"
                / "train"
                / "student_01"
                / "student_model.pt2"
            )
            student_model.write_text("mutated-student-model", encoding="utf-8")

            main([str(run)], backend_composition=bundle)

            self.assertTrue(sample_sentinel.is_file())

    def test_mixed_route_noncanonical_router_manifest_rebuilds_suffix(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="single", student_model="committee"
            )
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            screen_sentinel = root / "iter0" / "r0" / "screen" / "preserve.me"
            screen_sentinel.write_text("screen", encoding="utf-8")
            manifest = (
                root / "iter0" / "teach" / "scan" / "gmm_router" / "manifest.json"
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["diagnostic_note"] = "valid-but-different-router-content"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            main([str(run), "--resume", "teach/scan"], backend_composition=bundle)

            self.assertFalse(screen_sentinel.exists())
            repaired = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertNotIn("diagnostic_note", repaired)

    def test_mixed_dpa4_terminal_drift_rejects_later_resume_before_cleanup(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root,
                teacher_backend="DeepMD",
                teacher_model="single",
                student_model="committee",
            )
            bundle = compose_test_backend_roles(run,
                teacher=_FakeDeepmdTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            scan_sentinel = root / "iter0" / "teach" / "scan" / "preserve.me"
            scan_sentinel.write_text("scan", encoding="utf-8")
            terminal = (
                root
                / "iter0"
                / "teach"
                / "train"
                / "teacher"
                / "teacher_model.pt2"
            )
            terminal.write_text("mutated-terminal", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "required result teach/train for teach/scan is not consumer-runnable",
            ):
                main(
                    [str(run), "--resume", "teach/scan"],
                    backend_composition=bundle,
                )

            self.assertTrue(scan_sentinel.is_file())

    def test_mixed_band_drift_requires_targeted_screen_resume_and_preserves_sample(self) -> None:
        from workflow_artifacts.lifecycle.mixed_route import load_mixed_route_audit
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="single", student_model="committee")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            screen_sentinel = root / "iter0" / "r0" / "screen" / "replace.me"
            downstream_sentinel = root / "iter0" / "r1" / "sample" / "replace.me"
            for sentinel in (sample_sentinel, screen_sentinel, downstream_sentinel):
                sentinel.write_text(sentinel.name, encoding="utf-8")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["lo"] = 0.02
            run.write_text(json.dumps(payload), encoding="utf-8")

            main([str(run)], backend_composition=bundle)
            self.assertTrue(screen_sentinel.is_file())

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            self.assertTrue(sample_sentinel.is_file())
            self.assertFalse(screen_sentinel.exists())
            self.assertFalse(downstream_sentinel.exists())
            audit = load_mixed_route_audit(
                root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz"
            )
            self.assertEqual(
                audit["thresholds"]["student_force_deviation"],
                {"lo": 0.02, "hi": 0.25},
            )

    def test_mixed_screen_score_failure_is_atomic_and_targeted_resume_preserves_sample(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="single", student_model="committee"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["pair_guard"] = {"min": 0.5}
            run.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "injected guarded screen score failure"
            ):
                main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_PairGuardFailingScoreTeacher(),
                        student=_FakeStudent(),
                        sampling=_pair_guard_sampler(root),
                    ),
                )

            sample = root / "iter0" / "r0" / "sample"
            failed_screen = root / "iter0" / "r0" / "screen"
            self.assertTrue(sample.is_dir())
            self.assertTrue((failed_screen / ".attempts").is_dir())
            for name in (
                "train.xyz",
                "fp_candidate.xyz",
                "mixed_route_scores.npz",
                "screen_summary.json",
            ):
                self.assertFalse((failed_screen / name).exists(), name)
            self.assertFalse((root / "iter0" / "r0" / ".screen.staging").exists())
            self.assertFalse(
                (
                    root
                    / ".dice_state"
                    / "markers"
                    / "iter0"
                    / "r0__screen.json"
                ).exists()
            )
            sentinel = sample / "preserve.me"
            sentinel.write_text("sample", encoding="utf-8")

            main(
                [str(run), "--resume", "r0/screen"],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_pair_guard_sampler(root),
                ),
            )

            self.assertTrue(sentinel.is_file())
            self.assertTrue(
                (root / "iter0" / "r0" / "screen" / "mixed_route_scores.npz").is_file()
            )

    def test_mixed_screen_summary_is_diagnostic_only_for_completion(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="single", student_model="committee"
            )
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            summary = root / "iter0" / "r0" / "screen" / "screen_summary.json"
            score_calls = list(teacher.score_identity_calls)
            summary.unlink()

            main([str(run)], backend_composition=bundle)

            self.assertFalse(summary.exists())
            self.assertEqual(teacher.score_identity_calls, score_calls)

    def test_later_iter_mixed_teacher_relabels_prior_student_membership_without_regating(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.lifecycle.runner import _prior_iter_student_train_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="single"
            )
            initial_bundle = compose_test_backend_roles(run,
                teacher=_RoutingCommitteeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=initial_bundle)

            prior_sources = _prior_iter_student_train_sources(root / "iter0")
            expected_ids = [
                str(frame.info["structure_id"])
                for source in prior_sources
                for frame in read_extxyz_structures(source, allow_empty=False)
            ]

            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["iter"] = 1
            payload["student_train_bootstrap"] = []
            payload["fp_data"] = ["data/teacher.xyz"]
            run.write_text(json.dumps(payload), encoding="utf-8")
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_HighDeviationRoutingCommitteeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            relabeled = read_extxyz_structures(
                root / "iter1" / "teach" / "screen" / "train.xyz",
                allow_empty=False,
            )
            self.assertEqual(
                [str(frame.info["structure_id"]) for frame in relabeled], expected_ids
            )
            self.assertTrue(
                all(float(frame.info["force_uncertainty"]) == 0.6 for frame in relabeled)
            )
            self.assertTrue(
                all(
                    float(frame.info["force_uncertainty"])
                    >= float(payload["teacher"]["lo"])
                    for frame in relabeled
                )
            )

    def test_guarded_gmm_reordered_canonical_output_fails_completion_and_rebuilds(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"].update({
                "pair_guard": {"min": 0.49},
                "pool": [2, 2],
                "keep": [2, 2],
            })
            run.write_text(json.dumps(payload), encoding="utf-8")
            teacher = _PairGuardRecordingTeacher()
            bundle = compose_test_backend_roles(run,
                teacher=teacher,
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            output = root / "iter0" / "r0" / "screen" / "train.xyz"
            frames = read_extxyz_structures(output, allow_empty=False)
            self.assertEqual(len(frames), 2)
            expected_order = tuple(str(frame.info["structure_id"]) for frame in frames)
            write_extxyz_structures(output, list(reversed(frames)), allow_empty=False)
            score_call_count = len(teacher.score_identity_calls)

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)

            rebuilt = read_extxyz_structures(output, allow_empty=False)
            self.assertEqual(tuple(str(frame.info["structure_id"]) for frame in rebuilt), expected_order)
            self.assertGreater(len(teacher.score_identity_calls), score_call_count)

    def test_single_gmm_threshold_drift_requires_and_honors_targeted_screen_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            screen_sentinel = root / "iter0" / "r0" / "screen" / "replace.me"
            sample_sentinel.write_text("sample", encoding="utf-8")
            screen_sentinel.write_text("screen", encoding="utf-8")

            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["teacher_p"] = {"trust": 0.2}
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=bundle)
            self.assertTrue(screen_sentinel.is_file())

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)
            self.assertTrue(sample_sentinel.is_file())
            self.assertFalse(screen_sentinel.exists())
            summary = json.loads(
                (root / "iter0" / "r0" / "screen" / "screen_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["thresholds"]["teacher"], {"floor": 0.0, "trust": 0.2, "ceiling": None})
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_safe_fill_toggle_drift_requires_and_honors_targeted_screen_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            screen_sentinel = root / "iter0" / "r0" / "screen" / "replace.me"
            sample_sentinel.write_text("sample", encoding="utf-8")
            screen_sentinel.write_text("screen", encoding="utf-8")

            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"]["fill_underfilled_from_safe_region"] = False
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=bundle)
            self.assertTrue(screen_sentinel.is_file())

            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)
            self.assertTrue(sample_sentinel.is_file())
            self.assertFalse(screen_sentinel.exists())
            summary = json.loads(
                (root / "iter0" / "r0" / "screen" / "screen_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["fill_underfilled_from_safe_region"])
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_single_gmm_direct_fps_use_only_bounded_role_native_frame_means(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        for policy in ("DIRECT", "FPS"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run = _write_public_run(root)
                payload = json.loads(run.read_text(encoding="utf-8"))
                payload["screen"]["policy"] = policy
                payload["screen"]["pair_guard"] = {"min": 0.49}
                payload["fp"]["policy"] = policy
                run.write_text(json.dumps(payload), encoding="utf-8")
                main([str(run)], backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
                ))
                screen = root / "iter0" / "r0" / "screen"
                with np.load(screen / "representative_features.npz", allow_pickle=False) as teacher:
                    self.assertEqual(teacher["means"].shape[1], 128)
                    self.assertLessEqual(teacher["means"].shape[0], 1)
                with np.load(screen / "student_representative_features.npz", allow_pickle=False) as student:
                    self.assertEqual(student["means"].shape[1], 12)
                    self.assertLessEqual(student["means"].shape[0], 1)

                sidecar = screen / "representative_features.npz"
                with np.load(sidecar, allow_pickle=False) as payload:
                    corrupted = {name: np.asarray(payload[name]) for name in payload.files}
                corrupted["means"] = corrupted["means"].copy()
                corrupted["means"][0, 0] = np.nan
                with sidecar.open("wb") as handle:
                    np.savez_compressed(handle, **corrupted)
                main([str(run)], backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
                ))
                with np.load(screen / "representative_features.npz", allow_pickle=False) as teacher:
                    self.assertEqual(str(teacher["schema"].item()), "dice.lifecycle.frame-mean-representatives")
                    self.assertEqual(int(teacher["schema_revision"].item()), 2)
                    self.assertIn('"name":"readout-input"', str(teacher["descriptor_schema"].item()))
                    self.assertIn('"revision":1', str(teacher["descriptor_schema"].item()))
                    self.assertFalse(np.isfinite(teacher["means"]).all())

    def test_direct_revision_two_small_pool_workflow_matches_independent_terminal_reference(self) -> None:
        import warnings

        from sklearn.cluster import Birch
        from sklearn.decomposition import PCA
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.preprocessing import StandardScaler

        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["sample"]["temperatures_k"] = [300, 500]
            payload["screen"].update(
                {
                    "pool": [2],
                    "keep": [2],
                    "fp_pool": 2,
                    "policy": "DIRECT",
                }
            )
            payload["fp"].update({"keep": 2, "policy": "DIRECT"})
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            sidecar = root / "iter0" / "r0" / "screen" / "representative_features.npz"
            with np.load(sidecar, allow_pickle=False) as evidence:
                identities = [str(item) for item in evidence["ids"].tolist()]
                vectors = np.asarray(evidence["means"], dtype=float)
            self.assertGreaterEqual(len(identities), 2)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = np.divide(
                vectors,
                norms,
                out=np.zeros_like(vectors),
                where=norms > 0,
            )
            standardized = StandardScaler().fit_transform(normalized)
            reduced = standardized
            if len(standardized) >= 2:
                pca = PCA().fit(standardized)
                retained = pca.explained_variance_ > 1.0
                if np.any(retained):
                    reduced = (
                        pca.transform(standardized)[:, retained]
                        * pca.explained_variance_ratio_[retained]
                    )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = Birch(
                    threshold=0.5,
                    branching_factor=50,
                    n_clusters=min(2, len(identities)),
                ).fit_predict(reduced)
            expected_indices = []
            for label in sorted(set(int(item) for item in labels)):
                indices = np.flatnonzero(labels == label)
                centroid = np.mean(reduced[indices], axis=0)
                expected_indices.append(
                    int(
                        indices[
                            np.argmin(
                                np.linalg.norm(
                                    reduced[indices] - centroid,
                                    axis=1,
                                )
                            )
                        ]
                    )
                )
            expected_membership = {
                identities[index] for index in expected_indices
            }
            candidates = read_extxyz_structures(
                root / "iter0" / "r0" / "screen" / "fp_candidate.xyz",
                allow_empty=False,
            )
            candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
            candidates.sort(
                key=lambda frame: (
                    -float(
                        frame.info.get(
                            "acquisition",
                            frame.info.get(
                                "force_uncertainty",
                                frame.info.get(
                                    "teacher_p",
                                    frame.info.get("p", 0.0),
                                ),
                            ),
                        )
                    ),
                    str(frame.info["structure_id"]),
                )
            )
            expected = [
                str(frame.info["structure_id"])
                for frame in candidates
                if str(frame.info["structure_id"]) in expected_membership
            ]
            actual = [
                str(frame.info["structure_id"])
                for frame in read_extxyz_structures(
                    root / "iter0" / "fp" / "selected.xyz",
                    allow_empty=False,
                )
            ]

            self.assertEqual(actual, expected)

    def test_single_gmm_complete_p_bands_keep_full_pools_in_memory_and_publish_only_bounded_evidence(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["screen"].update({
                "fp_pool": -1,
                "pool": [-1, -1],
                "keep": [1, 1],
                "policy": "FPS",
                "teacher_p": {"ceiling": 1.0},
                "student_p": {"ceiling": 1.0},
            })
            payload["sample"]["temperatures_k"] = [300, 400, 500]
            payload["fp"] = {"keep": 1, "policy": "FPS"}
            run.write_text(json.dumps(payload), encoding="utf-8")

            main([str(run)], backend_composition=compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            ))

            for round_index in range(2):
                screen = root / "iter0" / f"r{round_index}" / "screen"
                candidates = read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=True)
                train = read_extxyz_structures(screen / "train.xyz", allow_empty=True)
                with np.load(screen / "gmm_scores.npz", allow_pickle=False) as scores:
                    self.assertEqual(len(candidates), int(np.count_nonzero(scores["teacher_fp_candidate"])))
                    self.assertGreater(int(np.count_nonzero(scores["student_p_pool"])), len(train))
                with np.load(screen / "representative_features.npz", allow_pickle=False) as teacher:
                    self.assertEqual(teacher["means"].dtype, np.float32)
                    self.assertEqual(teacher["means"].shape, (len(candidates), 128))
                with np.load(screen / "student_representative_features.npz", allow_pickle=False) as student:
                    self.assertEqual(student["means"].dtype, np.float32)
                    self.assertEqual(student["means"].shape, (len(train), 12))
                    self.assertLessEqual(len(train), 1)

            selected = read_extxyz_structures(root / "iter0" / "fp" / "selected.xyz", allow_empty=True)
            self.assertLessEqual(len(selected), 1)

    def test_supported_single_gmm_route_publishes_student_router_without_descriptor_cache(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["student"]["cli"] = "tf"
            payload["screen"]["policy"] = "random"
            payload["sample"]["temperatures_k"] = [300]
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            ))
            screen = root / "iter0" / "r0" / "screen"
            train = read_extxyz_structures(screen / "train.xyz", allow_empty=False)
            fp_candidates = read_extxyz_structures(screen / "fp_candidate.xyz", allow_empty=False)
            self.assertTrue(0.0 < float(train[0].info["student_p"]) <= 1.0)
            self.assertTrue(
                {frame.info["structure_id"] for frame in train}
                & {frame.info["structure_id"] for frame in fp_candidates}
            )
            self.assertTrue((screen / "gmm_router" / "manifest.json").is_file())
            self.assertTrue((screen / "screen_summary.json").is_file())
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["policy_identity"],
                {"name": "gmm-reference-tail", "revision": 4},
            )
            self.assertEqual(summary["thresholds"]["teacher"], {"floor": 0.0, "trust": 0.1, "ceiling": None})
            self.assertEqual(summary["thresholds"]["student"], {"floor": 0.0, "ceiling": None})
            self.assertEqual(set(summary["p_bins"]["teacher"]), {"0-0.1", "0.1-1"})
            with np.load(screen / "gmm_scores.npz", allow_pickle=False) as audit:
                self.assertFalse(any("descriptor" in name or "nll" in name.lower() for name in audit.files))
                self.assertNotIn("cutoff_discard", audit.files)
                self.assertNotIn("teacher_extreme", audit.files)
            self.assertFalse((screen / "representative_features.npz").exists())
            self.assertFalse((screen / ".screen_work").exists())
            self.assertFalse(any("descriptor_cache" in path.name for path in screen.rglob("*")))

    def test_public_four_backend_single_model_matrix_uses_one_backend_neutral_screen(self) -> None:
        from workflow_artifacts.lifecycle.representative_features import load_frame_mean_representatives
        from workflow_core.structures.formats import read_extxyz_structures
        from workflow_core.lifecycle.entrypoint import main
        from workflow_policies.gmm_reference_tail import validate_gmm_router

        combinations = (
            ("MACE", "DeepMD", "mace", "deepmd-tf", 12),
            ("MACE", "NEP", "mace", "nep", 9),
            ("DeepMD", "DeepMD", "deepmd-pt", "deepmd-tf", 12),
            ("DeepMD", "NEP", "deepmd-pt", "nep", 9),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index, (teacher_backend, student_backend, teacher_schema, student_schema, width) in enumerate(combinations):
                case = root / f"case-{index}"
                case.mkdir()
                run = _write_public_run(
                    case,
                    teacher_backend=teacher_backend,
                    student_backend=student_backend,
                )
                payload = json.loads(run.read_text(encoding="utf-8"))
                payload["sample"]["temperatures_k"] = [300, 500]
                payload["screen"].update({
                    "pool": [2, 2],
                    "keep": [1, 1],
                    "policy": "FPS",
                    "teacher_p": {"floor": 0.0, "trust": 0.0},
                })
                run.write_text(json.dumps(payload), encoding="utf-8")
                bundle = compose_test_backend_roles(
                    run,
                    teacher=(
                        _FakeTeacher()
                        if teacher_backend == "MACE"
                        else _FakeDeepmdTeacher()
                    ),
                    student=(
                        _FakeStudent()
                        if student_backend == "DeepMD"
                        else _FakeNepStudent()
                    ),
                    sampling=(
                        _FakeSampler()
                        if student_backend == "DeepMD"
                        else _FakeGpumdSampler()
                    ),
                )

                main([str(run)], backend_composition=bundle)

                teacher_router = validate_gmm_router(case / "iter0" / "teach" / "scan" / "gmm_router")
                screen = case / "iter0" / "r0" / "screen"
                student_router = validate_gmm_router(screen / "gmm_router")
                self.assertEqual(teacher_router["descriptor_schema"]["backend"], teacher_schema)
                self.assertEqual(student_router["descriptor_schema"]["backend"], student_schema)
                self.assertEqual(student_router["descriptor_schema"]["width"], width)
                train_ids = tuple(
                    str(frame.info["structure_id"])
                    for frame in read_extxyz_structures(screen / "train.xyz", allow_empty=True)
                )
                if train_ids:
                    load_frame_mean_representatives(
                        screen / "student_representative_features.npz",
                        expected_identities=train_ids,
                        expected_descriptor_schema=student_router["descriptor_schema"],
                    )
                self.assertTrue((screen / "gmm_scores.npz").is_file())
                self.assertFalse((screen / ".screen_work").exists())
                self.assertFalse(any("descriptor_cache" in path.name for path in screen.rglob("*")))

    def test_public_single_dpa4_teacher_route_preserves_dynamic_readout_schema_end_to_end(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_backend="DeepMD", teacher_model="single")
            (root / "recipes" / "teacher.json").write_text(
                json.dumps({"model": {"type": "dpa4", "descriptor": {"type": "dpa4"}}}),
                encoding="utf-8",
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["temperatures_k"] = [300, 500]
            payload["screen"]["pool"] = [2, 2]
            payload["screen"]["teacher_p"] = {"floor": 0.0, "trust": 1.0}
            payload["fp"]["policy"] = "FPS"
            run.write_text(json.dumps(payload), encoding="utf-8")

            teacher = _SingleLookupDeepmdTeacher()
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            train_root = root / "iter0" / "teach" / "train"
            train = train_root / "teacher"
            self.assertTrue((train / "teacher_model.pt2").is_file())
            self.assertFalse((train / "teacher_descriptor_checkpoint.pt").exists())
            self.assertFalse((train_root / "teacher_model_assets.json").exists())

            teacher_router = json.loads(
                (root / "iter0" / "teach" / "scan" / "gmm_router" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(teacher_router["descriptor_schema"]["backend"], "deepmd-pt")
            self.assertEqual(teacher_router["descriptor_schema"]["model_family"], "dpa4")
            self.assertEqual(teacher_router["descriptor_schema"]["width"], 17)

            screen = root / "iter0" / "r0" / "screen"
            with np.load(screen / "representative_features.npz", allow_pickle=False) as representatives:
                self.assertEqual(representatives["means"].dtype, np.float32)
                self.assertEqual(representatives["means"].shape[1], 17)
            summary = json.loads((screen / "screen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["provenance"]["operation_profiles"]["teacher_gmm_score"],
                "fake-dpa4-gmm",
            )
            self.assertEqual(
                summary["provenance"]["operation_profiles"]["teacher_label"],
                "fake-dpa4-label",
            )
            self.assertEqual(
                summary["provenance"]["teacher"]["descriptor_schema"],
                teacher_router["descriptor_schema"],
            )
            self.assertFalse(any(path.name == "descriptor_cache" for path in root.rglob("*")))
            self.assertFalse(any(path.name in {"descriptors.npy", "descriptors.npz"} for path in root.rglob("*")))
            self.assertEqual(teacher.reference_calls, 1)
            # Two rounds perform scientific GMM scoring.  FP visualization must
            # use the dedicated structural-vector capability, never another
            # scientific score transport that drops per-element vectors.
            self.assertEqual(teacher.stream_score_calls, 2)
            self.assertFalse(any(path.name == "candidates.xyz" for path in root.rglob("*")))

    def test_public_cli_recipe_change_does_not_block_explicit_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir); run = _write_public_run(root, teacher_model="committee", student_model="committee"); environment = _write_native_fake_environment(root)
            main([str(run), "--environment", str(environment)])
            sample = root / "iter0" / "r1" / "sample" / "tasks" / "task_000000"
            (root / "recipes" / "teacher.yaml").write_text("max_num_epochs: 2\n", encoding="utf-8")
            main([str(run), "--environment", str(environment), "--resume", "r1/sample"])
            self.assertTrue(sample.is_dir())

    def test_public_cli_environment_source_change_does_not_block_explicit_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir); run = _write_public_run(root, teacher_model="committee", student_model="committee"); environment = _write_native_fake_environment(root)
            main([str(run), "--environment", str(environment)])
            sample = root / "iter0" / "r1" / "sample" / "tasks" / "task_000000"
            activation = root / "activation.sh"; activation.write_text("#!/bin/sh\n", encoding="utf-8")
            profile = json.loads(environment.read_text(encoding="utf-8")); profile["software_profiles"]["mace"]["environment"]["source_list"] = [str(activation)]
            environment.write_text(json.dumps(profile), encoding="utf-8")
            main([str(run), "--environment", str(environment), "--resume", "r1/sample"])
            self.assertTrue(sample.is_dir())

    def test_public_single_teacher_gmm_score_can_finish_with_nonempty_fp_terminal_selection(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir); run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8")); payload["sample"]["temperatures_k"] = [300, 500]; payload["screen"]["pool"] = [2, 2]; payload["screen"]["keep"] = [2, 2]
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=compose_test_backend_roles(run, teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler()))
            self.assertEqual(len(read_extxyz_structures(root / "iter0" / "r0" / "screen" / "fp_candidate.xyz", allow_empty=True)), 1)
            self.assertEqual(len(read_extxyz_structures(root / "iter0" / "fp" / "selected.xyz", allow_empty=True)), 1)

    def test_terminal_fp_uses_final_rn_candidate_by_default(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["fp"]["keep"] = 2
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            final_candidates = read_extxyz_structures(
                root / "iter0" / "r1" / "screen" / "fp_candidate.xyz", allow_empty=True
            )
            selected = read_extxyz_structures(root / "iter0" / "fp" / "selected.xyz", allow_empty=True)

            self.assertEqual(
                [frame.info["structure_id"] for frame in selected],
                [frame.info["structure_id"] for frame in final_candidates],
            )

    def test_terminal_element_policies_publish_replay_and_resume_fp_only(self) -> None:
        from workflow_artifacts.lifecycle.element_direct_fp import (
            load_element_direct_fp_evidence,
        )
        from workflow_core.lifecycle.entrypoint import main

        for policy in ("ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run = _write_public_run(root)
                payload = json.loads(run.read_text(encoding="utf-8"))
                payload["visualize"] = False
                payload["fp"].update(policy=policy, keep=1, all_rn=True)
                run.write_text(json.dumps(payload), encoding="utf-8")
                teacher = _ElementFakeTeacher()
                bundle = compose_test_backend_roles(
                    run,
                    teacher=teacher,
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                )

                main([str(run)], backend_composition=bundle)
                evidence_path = root / "iter0" / "fp" / "element_direct_fp.npz"
                evidence = load_element_direct_fp_evidence(evidence_path)
                self.assertEqual(str(evidence["policy"].item()), policy)
                self.assertEqual(tuple(evidence["candidate_rounds"].tolist()), (0, 1))
                self.assertGreater(teacher.structure_vector_calls, 0)
                screen_sentinel = root / "iter0" / "r1" / "screen" / "preserve.me"
                screen_sentinel.write_text("screen", encoding="utf-8")
                with np.load(evidence_path, allow_pickle=False) as archive:
                    corrupted = {
                        name: np.array(archive[name]) for name in archive.files
                    }
                corrupted["selected"] = np.logical_not(corrupted["selected"])
                np.savez_compressed(evidence_path, **corrupted)

                main(
                    [str(run), "--resume", "fp"],
                    backend_composition=bundle,
                )

                self.assertTrue(screen_sentinel.is_file())
                repaired = load_element_direct_fp_evidence(evidence_path)
                selected_ids = set(str(item) for item in repaired["selected_ids"].tolist())
                mask_ids = {
                    str(identity)
                    for identity, chosen in zip(
                        repaired["structure_ids"], repaired["selected"], strict=True
                    )
                    if bool(chosen)
                }
                self.assertEqual(mask_ids, selected_ids)

    def test_single_teacher_terminal_element_evidence_errors_fail_without_fallback(
        self,
    ) -> None:
        from workflow_core.lifecycle.entrypoint import main

        class InvalidElementTeacher(_ElementFakeTeacher):
            def __init__(self, failure: str):
                super().__init__()
                self.failure = failure

            def structure_vectors(self, **kwargs):
                schema, identities, rows, by_element = super().structure_vectors(
                    **kwargs
                )
                broken = {
                    element: (tuple(element_ids), np.array(values, copy=True))
                    for element, (element_ids, values) in by_element.items()
                }
                if self.failure == "nonfinite":
                    first = next(iter(broken))
                    broken[first][1][0, 0] = np.nan
                elif self.failure == "misordered":
                    first = next(iter(broken))
                    element_ids, values = broken[first]
                    changed_ids = tuple(reversed(element_ids))
                    if changed_ids == element_ids:
                        changed_ids = (f"misordered:{element_ids[0]}",)
                    broken[first] = (changed_ids, values)
                else:
                    broken.clear()
                return schema, identities, rows, broken

        for failure in ("missing", "misordered", "nonfinite"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run = _write_public_run(root)
                payload = json.loads(run.read_text(encoding="utf-8"))
                payload["visualize"] = False
                payload["fp"].update(policy="ELEMENT_DIRECT", keep=1)
                run.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises((TypeError, ValueError)):
                    main(
                        [str(run)],
                        backend_composition=compose_test_backend_roles(
                            run,
                            teacher=InvalidElementTeacher(failure),
                            student=_FakeStudent(),
                            sampling=_FakeSampler(),
                        ),
                    )

                fp_root = root / "iter0" / "fp"
                self.assertFalse((fp_root / "policy_resolution.json").exists())
                self.assertFalse(
                    (
                        root
                        / ".dice_state"
                        / "markers"
                        / "iter0"
                        / "fp.json"
                    ).exists()
                )

    def test_committee_teacher_terminal_element_direct_is_exact_random_fallback(self) -> None:
        from workflow_artifacts.lifecycle.policy_resolution import (
            load_policy_resolution,
        )
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        memberships = {}
        for policy in ("random", "ELEMENT_DIRECT"):
            root = tmp_path = Path(tempfile.mkdtemp())
            self.addCleanup(lambda path=root: __import__("shutil").rmtree(path))
            run = _write_public_run(root, teacher_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["visualize"] = False
            payload["screen"]["pool"] = [2, 2]
            payload["fp"].update(policy=policy, keep=1, all_rn=True)
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(
                run,
                teacher=_CommitteeFPTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )

            main([str(run)], backend_composition=bundle)
            memberships[policy] = [
                str(frame.info["structure_id"])
                for frame in read_extxyz_structures(
                    root / "iter0" / "fp" / "selected.xyz", allow_empty=True
                )
            ]
            if policy == "ELEMENT_DIRECT":
                fp_root = root / "iter0" / "fp"
                evidence = load_policy_resolution(
                    fp_root / "policy_resolution.json"
                )
                self.assertEqual(evidence["coordinate"], "fp.policy")
                self.assertEqual(
                    evidence["rows"][0]["selected_ids"], memberships[policy]
                )
                summary = json.loads(
                    (fp_root / "fp_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    summary["selector_resolution"]["fallback_reason"],
                    "committee_descriptor_unavailable",
                )
                self.assertFalse((fp_root / "element_direct_fp.npz").exists())
                screen_sentinel = root / "iter0" / "r1" / "screen" / "preserve.me"
                screen_sentinel.write_text("screen", encoding="utf-8")
                evidence["rows"][0]["selected_ids"] = []
                (fp_root / "policy_resolution.json").write_text(
                    json.dumps(evidence), encoding="utf-8"
                )
                main(
                    [str(run), "--resume", "fp"],
                    backend_composition=bundle,
                )
                self.assertTrue(screen_sentinel.is_file())
                repaired = load_policy_resolution(
                    fp_root / "policy_resolution.json"
                )
                self.assertEqual(
                    repaired["rows"][0]["selected_ids"], memberships[policy]
                )
            else:
                self.assertFalse(
                    (root / "iter0" / "fp" / "policy_resolution.json").exists()
                )

        # The two public runs have different path-derived entropy roots. Exact
        # same-entropy equivalence is covered at the pure terminal-policy seam.
        self.assertTrue(memberships["random"])
        self.assertTrue(memberships["ELEMENT_DIRECT"])

    def test_committee_student_screen_element_direct_recovers_at_screen_only(self) -> None:
        from workflow_artifacts.lifecycle.policy_resolution import (
            load_policy_resolution,
        )
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        element_context = None
        for policy in ("random", "ELEMENT_DIRECT"):
            root = Path(tempfile.mkdtemp())
            self.addCleanup(lambda path=root: __import__("shutil").rmtree(path))
            run = _write_public_run(root, student_model="committee")
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["visualize"] = False
            payload["screen"]["policy"] = policy
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )

            main([str(run)], backend_composition=bundle)
            screen = root / "iter0" / "r1" / "screen"
            membership = [
                str(frame.info["structure_id"])
                for frame in read_extxyz_structures(
                    screen / "train.xyz", allow_empty=True
                )
            ]
            if policy == "ELEMENT_DIRECT":
                evidence = load_policy_resolution(
                    screen / "policy_resolution.json"
                )
                self.assertEqual(evidence["coordinate"], "screen.policy")
                self.assertEqual(evidence["rows"][0]["selected_ids"], membership)
                self.assertFalse((screen / "element_direct_screen.npz").exists())
                sample_sentinel = root / "iter0" / "r1" / "sample" / "preserve.me"
                sample_sentinel.write_text("sample", encoding="utf-8")
                evidence["rows"][0]["warning_emitted"] = False
                (screen / "policy_resolution.json").write_text(
                    json.dumps(evidence), encoding="utf-8"
                )
                main(
                    [str(run), "--resume", "r1/screen"],
                    backend_composition=bundle,
                )
                self.assertTrue(sample_sentinel.is_file())
                self.assertTrue(
                    load_policy_resolution(
                        screen / "policy_resolution.json"
                    )["rows"][0]["warning_emitted"]
                )
                element_context = (root, run, bundle)

        self.assertIsNotNone(element_context)

    def test_committee_teacher_single_student_element_direct_screen_completes(self) -> None:
        from workflow_artifacts.lifecycle.element_direct import (
            load_element_direct_screen_evidence,
        )
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root, teacher_model="committee", student_model="single"
            )
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["visualize"] = False
            payload["screen"]["policy"] = "ELEMENT_DIRECT"
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_RoutingCommitteeTeacher(),
                    student=_ElementFakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            screen = root / "iter0" / "r0" / "screen"
            evidence = load_element_direct_screen_evidence(
                screen / "element_direct_screen.npz"
            )
            self.assertEqual(
                str(evidence["route_kind"].item()),
                "committee-teacher/single-student",
            )
            self.assertFalse((screen / "policy_resolution.json").exists())
            self.assertFalse((root / ".dice_state" / "markers").exists())

    def test_single_single_element_joint_and_element_selector_keep_structure_composition(
        self,
    ) -> None:
        from workflow_artifacts.lifecycle.element_direct import (
            load_element_direct_screen_evidence,
        )
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["visualize"] = False
            payload["screen"].update(
                prescreen="ELEMENT_JOINT_DIRECT",
                policy="ELEMENT_DIRECT",
                teacher_p={"floor": 0.0, "trust": 0.0},
                fill_underfilled_from_safe_region=False,
            )
            payload["screen"].pop("student_p", None)
            run.write_text(json.dumps(payload), encoding="utf-8")

            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(
                    run,
                    teacher=_ElementFakeTeacher(),
                    student=_ElementFakeStudent(),
                    sampling=_FakeSampler(),
                ),
            )

            evidence = load_element_direct_screen_evidence(
                root / "iter0" / "r0" / "screen" / "element_direct_screen.npz"
            )
            self.assertEqual(
                str(evidence["prescreen"].item()), "ELEMENT_JOINT_DIRECT"
            )
            self.assertEqual(
                str(evidence["screen_policy"].item()), "ELEMENT_DIRECT"
            )

    def test_optional_fp_label_runs_after_fp_through_calculator_neutral_service(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["fp"]["label"] = {
                "calculator": "CP2K",
                "recipe": "recipes/cp2k.json",
            }
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "cp2k.json").write_text(
                json.dumps({"template": "cp2k.inp"}), encoding="utf-8"
            )
            (root / "recipes" / "cp2k.inp").write_text(
                "&GLOBAL\n  RUN_TYPE ENERGY_FORCE\n&END GLOBAL\n",
                encoding="utf-8",
            )
            launch = main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                    first_principles=_FakeFirstPrinciplesLabelService(),
                ),
            )

            selected = read_extxyz_structures(
                root / "iter0" / "fp" / "selected.xyz", allow_empty=True
            )
            labeled = read_extxyz_structures(
                root / "iter0" / "fp" / "label" / "labeled.xyz",
                allow_empty=True,
            )
            marker_absent = not (root / ".dice_state" / "markers").exists()

        self.assertEqual(launch.plan.blocks[-1].block_id, "fp/label")
        self.assertEqual(
            [frame.info["structure_id"] for frame in labeled],
            [frame.info["structure_id"] for frame in selected],
        )
        self.assertTrue(marker_absent)

    def test_fp_label_artifact_validator_rebuilds_changed_selected_content(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["fp"]["label"] = {
                "calculator": "CP2K",
                "recipe": "recipes/cp2k.json",
            }
            run.write_text(json.dumps(payload), encoding="utf-8")
            (root / "recipes" / "cp2k.json").write_text(
                json.dumps({"template": "cp2k.inp"}), encoding="utf-8"
            )
            (root / "recipes" / "cp2k.inp").write_text(
                "&GLOBAL\n  RUN_TYPE ENERGY_FORCE\n&END GLOBAL\n",
                encoding="utf-8",
            )
            fp_service = _FakeFirstPrinciplesLabelService()
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
                first_principles=fp_service,
            )
            main([str(run)], backend_composition=bundle)
            main([str(run)], backend_composition=bundle)
            self.assertEqual(fp_service.calls, 1)

            selected_path = root / "iter0" / "fp" / "selected.xyz"
            selected = read_extxyz_structures(selected_path, allow_empty=False)
            selected[0].positions[0, 0] += 0.01
            write_extxyz_structures(selected_path, selected)

            main([str(run)], backend_composition=bundle)

        self.assertEqual(fp_service.calls, 2)

    def test_terminal_fp_all_rn_opt_in_rebuilds_only_fp(self) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import read_extxyz_structures

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["fp"]["keep"] = 2
            run.write_text(json.dumps(payload), encoding="utf-8")
            bundle = compose_test_backend_roles(run,
                teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)
            r0_candidates = read_extxyz_structures(
                root / "iter0" / "r0" / "screen" / "fp_candidate.xyz", allow_empty=True
            )
            r1_candidates = read_extxyz_structures(
                root / "iter0" / "r1" / "screen" / "fp_candidate.xyz", allow_empty=True
            )
            screen_sentinel = root / "iter0" / "r1" / "screen" / "preserve.me"
            screen_sentinel.write_text("screen", encoding="utf-8")

            payload["fp"]["all_rn"] = True
            run.write_text(json.dumps(payload), encoding="utf-8")
            main([str(run)], backend_composition=bundle)
            self.assertTrue(screen_sentinel.is_file())

            main([str(run), "--resume", "fp"], backend_composition=bundle)

            selected = read_extxyz_structures(root / "iter0" / "fp" / "selected.xyz", allow_empty=True)
            self.assertTrue(screen_sentinel.is_file())
            self.assertCountEqual(
                [frame.info["structure_id"] for frame in selected],
                [frame.info["structure_id"] for frame in [*r0_candidates, *r1_candidates]],
            )

    def test_public_entrypoint_runs_complete_native_factory_nep_gpumd_fake_command_lifecycle(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_backend="NEP", student_model="committee")
            environment = _write_native_fake_environment(root, student_backend="NEP")
            main([str(run), "--environment", str(environment)])
            main([str(run), "--environment", str(environment), "--resume", "r1/sample"])
            self.assertTrue((root / "iter0" / "r1" / "sample" / "tasks" / "task_000000" / "gpumd_sample.stdout.log").is_file())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_public_fake_bundle_runs_deepmd_committee_teacher_branch(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(
                root,
                teacher_backend="DeepMD",
                teacher_model="committee",
                student_model="committee",
            )
            main(
                [str(run)],
                backend_composition=compose_test_backend_roles(run, teacher=_FakeDeepmdTeacher(), student=_FakeStudent(), sampling=_FakeSampler()),
            )
            handle = root / "iter0" / "teach" / "train" / "teacher_02" / "teacher_model.pt2"
            self.assertTrue(handle.is_file())
            self.assertFalse(handle.is_symlink())
            self.assertFalse((handle.parent / "teacher_model").exists())
            for member_id in ("student_01", "student_02"):
                student_handle = root / "iter0" / "r0" / "train" / member_id / "student_model.pt2"
                self.assertTrue(student_handle.is_file())
                self.assertFalse(student_handle.is_symlink())
                self.assertFalse((student_handle.parent / "student_model").exists())
            self.assertFalse((root / "iter0" / "teach" / "scan").exists())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_public_fake_bundle_runs_committee_nep_gpumd_matrix_branch(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_backend="NEP", student_model="committee")
            bundle = compose_test_backend_roles(run, teacher=_FakeTeacher(), student=_FakeNepStudent(), sampling=_FakeGpumdSampler())
            main([str(run)], backend_composition=bundle)
            self.assertTrue((root / "iter0" / "teach" / "train" / "teacher_02" / "teacher_model.model").is_file())
            for member_id in ("student_01", "student_02"):
                handle = root / "iter0" / "r0" / "train" / member_id / "student_model.txt"
                self.assertTrue(handle.is_file())
                self.assertFalse(handle.is_symlink())
                self.assertFalse((handle.parent / "student_model").exists())
            self.assertFalse((root / "iter0" / "teach" / "scan").exists())
            self.assertTrue((root / "iter0" / "r1" / "sample" / "tasks" / "task_000000" / "run.in").is_file())
            sample_sentinel = root / "iter0" / "r0" / "sample" / "preserve.me"
            sample_sentinel.write_text("sample", encoding="utf-8")
            main([str(run), "--resume", "r0/screen"], backend_composition=bundle)
            self.assertTrue(sample_sentinel.is_file())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_student_train_recovery_rejects_dataset_identity_and_order_drift_for_both_backends(
        self,
    ) -> None:
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.structures.formats import (
            read_extxyz_structures,
            write_extxyz_structures,
        )

        class _CountingDeepMDStudent(_FakeStudent):
            def __init__(self) -> None:
                self.train_calls = 0

            def train(self, *args, **kwargs):
                del kwargs
                self.train_calls += 1
                return super().train(*args)

        class _CountingNepStudent(_FakeNepStudent):
            def __init__(self) -> None:
                self.train_calls = 0

            def train(self, *args, **kwargs):
                del kwargs
                self.train_calls += 1
                return super().train(*args)

        cases = (
            ("DeepMD", _CountingDeepMDStudent, _FakeSampler),
            ("NEP", _CountingNepStudent, _FakeGpumdSampler),
        )
        for backend, student_type, sampler_type in cases:
            for drift in ("identity", "order"):
                with self.subTest(backend=backend, drift=drift), tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    run = _write_public_run(
                        root,
                        teacher_model="committee",
                        student_backend=backend,
                        student_model="committee",
                    )
                    payload = json.loads(run.read_text(encoding="utf-8"))
                    payload["sample"]["steps"] = [20]
                    payload["screen"]["pool"] = [1]
                    payload["screen"]["keep"] = [1]
                    run.write_text(json.dumps(payload), encoding="utf-8")
                    student = student_type()
                    bundle = compose_test_backend_roles(
                        run,
                        teacher=_FakeTeacher(),
                        student=student,
                        sampling=sampler_type(),
                    )
                    main([str(run)], backend_composition=bundle)
                    self.assertEqual(student.train_calls, 2)

                    train_root = root / "iter0" / "r0" / "train"
                    dataset = train_root / "dataset" / "train.xyz"
                    frames = read_extxyz_structures(dataset, allow_empty=False)
                    canonical_ids = tuple(
                        str(frame.info["structure_id"]) for frame in frames
                    )
                    if drift == "identity":
                        frames[0].info["structure_id"] = "audit-corrupt-identity"
                    else:
                        frames.reverse()
                    write_extxyz_structures(dataset, frames, allow_empty=False)
                    prefix_sentinel = (
                        root / "iter0" / "teach" / "screen" / "preserve.me"
                    )
                    prefix_sentinel.write_text("prefix", encoding="utf-8")
                    suffix_sentinel = (
                        root / "iter0" / "r0" / "sample" / "remove.me"
                    )
                    suffix_sentinel.write_text("suffix", encoding="utf-8")

                    main([str(run)], backend_composition=bundle)

                    self.assertEqual(student.train_calls, 2)
                    repaired = read_extxyz_structures(dataset, allow_empty=False)
                    self.assertEqual(
                        tuple(str(frame.info["structure_id"]) for frame in repaired),
                        (
                            ("audit-corrupt-identity", *canonical_ids[1:])
                            if drift == "identity"
                            else tuple(reversed(canonical_ids))
                        ),
                    )
                    self.assertTrue(prefix_sentinel.is_file())
                    self.assertTrue(suffix_sentinel.exists())

    def test_public_nonbaseline_mixed_matrix_branches_use_shape_native_sample_evidence(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        cases = (
            ("dpa4-single_deepmd-tf-committee", "DeepMD", "single", "DeepMD", "committee", "pt"),
            ("mace-single_nep-committee", "MACE", "single", "NEP", "committee", None),
            ("deepmd-tf-committee_nep-single", "DeepMD", "committee", "NEP", "single", "tf"),
            ("deepmd-pt-committee_deepmd-tf-single", "DeepMD", "committee", "DeepMD", "single", "pt"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, teacher_backend, teacher_model, student_backend, student_model, teacher_cli in cases:
                with self.subTest(case=name):
                    case = root / name
                    case.mkdir()
                    run = _write_public_run(
                        case,
                        teacher_backend=teacher_backend,
                        teacher_model=teacher_model,
                        student_backend=student_backend,
                        student_model=student_model,
                    )
                    payload = json.loads(run.read_text(encoding="utf-8"))
                    if teacher_backend == "DeepMD":
                        payload["teacher"]["cli"] = teacher_cli
                    if student_backend == "DeepMD" and student_model == "committee":
                        payload["student"]["cli"] = "tf"
                        (case / "recipes" / "student.json").write_text(
                            json.dumps({"model": {"descriptor": {"type": "committee-native"}}}),
                            encoding="utf-8",
                        )
                    if teacher_backend == "DeepMD" and teacher_model == "committee":
                        (case / "recipes" / "teacher.json").write_text(
                            json.dumps({"model": {"descriptor": {"type": "committee-native"}}}),
                            encoding="utf-8",
                        )
                    run.write_text(json.dumps(payload), encoding="utf-8")
                    single_deepmd_student = student_backend == "DeepMD" and student_model == "single"
                    bundle = compose_test_backend_roles(
                        run,
                        teacher=(
                            _FakeTeacher()
                            if teacher_backend == "MACE"
                            else _FakeDeepmdTeacher(cli=teacher_cli or "pt")
                        ),
                        student=(
                            _FakeStudent()
                            if student_backend == "DeepMD"
                            else _FakeNepStudent()
                        ),
                        sampling=(
                            _FakeSingleModelSampler()
                            if single_deepmd_student
                            else (
                                _FakeSampler()
                                if student_backend == "DeepMD"
                                else _FakeGpumdSampler()
                            )
                        ),
                    )

                    main([str(run)], backend_composition=bundle)

                    sample_task = case / "iter0" / "r0" / "sample" / "tasks" / "task_000000"
                    self.assertTrue((case / "iter0" / "r0" / "screen" / "mixed_route_scores.npz").is_file())
                    self.assertEqual((sample_task / "model_devi.out").exists(), student_model == "committee" and student_backend == "DeepMD")
                    self.assertFalse(any(path.name == "representative_features.npz" for path in sample_task.rglob("*")))
                    self.assertTrue((case / "iter0" / "fp" / "selected.xyz").is_file())

    def test_public_entrypoint_runs_complete_native_factory_fake_command_lifecycle_and_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root, teacher_model="committee", student_model="committee")
            environment = _write_native_fake_environment(root)
            main([str(run), "--environment", str(environment)])
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
            self.assertFalse((root / ".dice_state" / "markers").exists())
            status = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            self.assertIn("teach/train [SUCCESS] backend=MACE", status)
            self.assertIn("r0/train [SUCCESS] backend=DeepMD", status)
            self.assertIn("r0/screen [SUCCESS] observations=", status)
            self.assertIn("fp [SUCCESS] candidates=", status)
            history = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            self.assertIn("r0/screen [RUNNING] phase=finalizing", history)
            self.assertIn("r0/screen [RUNNING] phase=validating", history)
            self.assertIn("r0/screen [RUNNING] phase=publishing", history)
            self.assertFalse((root / "runs").exists())
            self.assertFalse((root / "student_pipeline_template").exists())
            self.assertFalse(any(path.name in {"manifest.json", "cursors.json", "live_pool"} for path in root.rglob("*")))
            sample_log = root / "iter0" / "r1" / "sample" / "tasks" / "task_000000" / "lammps_sample.stdout.log"
            first_log_mtime = sample_log.stat().st_mtime_ns
            main([str(run), "--environment", str(environment)])
            self.assertEqual(sample_log.stat().st_mtime_ns, first_log_mtime)
            (root / "iter0" / "r0" / "screen" / "train.xyz").unlink()
            main(
                [
                    str(run),
                    "--environment",
                    str(environment),
                    "--resume",
                    "r0/screen",
                ]
            )
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())
            self.assertGreater(sample_log.stat().st_mtime_ns, first_log_mtime)
            main([str(run), "--environment", str(environment), "--resume", "r1/sample"])
            self.assertTrue(sample_log.is_file())

    def test_public_entrypoint_uses_environment_preflight_before_factory_injection(self) -> None:
        from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
        from workflow_core.lifecycle.composition import BackendComposition
        from workflow_core.lifecycle.entrypoint import main
        from workflow_core.lifecycle.sampling_coordinator import (
            LifecycleSamplingCoordinator,
        )
        from workflow_core.lifecycle.student_facade import LifecycleStudentFacade

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            environment = root / "environment.json"
            serial_lammps_environment = (
                Path(__file__).resolve().parent / "fixtures" / "lammps_serial.env.sh"
            )
            environment.write_text(json.dumps({
                "schema": "dice.environment-profile", "schema_revision": 4,
                "software_profiles": {"fake": {"environment": {"source_list": [str(serial_lammps_environment)]}, "executables": {"mace_run_train": "python", "dp": "python", "lmp": "python"}}},
                "bindings": {role: "fake" for role in (
                    "teacher.mace.train", "teacher.mace.label",
                    "teacher.mace.gmm_reference", "teacher.mace.gmm_score",
                    "student.deepmd.tf.train", "student.deepmd.tf.gmm_reference",
                    "student.deepmd.tf.gmm_score", "sampler.lammps.tf",
                )},
            }), encoding="utf-8")
            received = []
            selected_student = _bound_student_role(root, _FakeStudent())
            student = LifecycleStudentFacade(delegate=selected_student)
            composition = BackendComposition(
                teacher=compose_test_backend_roles(
                    run, teacher=_FakeTeacher()
                ).teacher,
                student=student,
                sampling=LifecycleSamplingCoordinator(
                    LammpsSamplingNativeOps(
                        executor=_FakeSampler(),
                        type_map=("H",),
                        recipe=root / "recipes" / "sample.in",
                    ),
                    student.sampling_models,
                ),
                screen=compose_test_backend_roles(
                    run,
                    teacher=_FakeTeacher(),
                    student=_FakeStudent(),
                    sampling=_FakeSampler(),
                ).screen,
            )
            launch = main(
                [str(run), "--environment", str(environment)],
                executor_factory=lambda contexts: (
                    received.append(contexts),
                    composition,
                )[1],
            )

            self.assertEqual(launch.preflight.operation_roles[0], "teacher.mace.train")
            self.assertEqual(set(received[0]), set(launch.preflight.operation_roles))
            self.assertEqual(received[0]["teacher.mace.train"].role, "teacher.mace.train")

    def test_public_entrypoint_executes_an_iter_with_explicit_fake_services(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            terminal = StringIO()
            with redirect_stdout(terminal):
                launch = main(
                    [str(run)],
                    backend_composition=compose_test_backend_roles(run,
                        teacher=_FakeTeacher(),
                        student=_FakeStudent(),
                        sampling=_FakeSampler(),
                    ),
                )

            self.assertEqual(launch.plan.blocks[-1].block_id, "fp")
            self.assertTrue((root / "iter0" / "teach" / "train" / "teacher" / "teacher_model.model").is_file())
            student = root / "iter0" / "r0" / "train" / "student" / "student_model.pb"
            self.assertTrue(student.is_file())
            self.assertFalse(student.is_symlink())
            self.assertFalse((student.parent / "student_model").exists())
            self.assertTrue((root / "iter0" / "teach" / "scan" / "gmm_router" / "manifest.json").is_file())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "gmm_router" / "manifest.json").is_file())
            self.assertTrue((root / "iter0" / "r0" / "screen" / "train.xyz").is_file())
            self.assertTrue((root / "iter0" / "r1" / "screen" / "train.xyz").is_file())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())
            self.assertFalse((root / ".dice_state" / "markers").exists())
            status = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            for block in (
                "teach/train", "teach/screen", "teach/scan",
                "r0/train", "r0/sample", "r0/screen",
                "r1/train", "r1/sample", "r1/screen", "fp",
            ):
                self.assertIn(f"{block} [SUCCESS]", status)
            history = (root / "iter0" / "status.log").read_text(encoding="utf-8")
            self.assertEqual(terminal.getvalue(), history)

    def test_scheduled_worker_executes_exactly_one_public_block(self) -> None:
        from unittest.mock import patch

        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            composition = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )

            with patch.dict(os.environ, {"DICE_SCHEDULED_WORKER": "1"}):
                main(
                    [str(run), "--scheduled-worker", "teach/train"],
                    backend_composition=composition,
                )

            self.assertTrue((root / "iter0" / "teach" / "train").is_dir())
            self.assertFalse((root / ".dice_state" / "markers").exists())
            self.assertFalse((root / "iter0" / "teach" / "screen").exists())

    def test_scheduled_worker_rejects_missing_recursion_sentinel_before_execution(self) -> None:
        from unittest.mock import patch

        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            composition = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DICE_SCHEDULED_WORKER", None)
                with self.assertRaisesRegex(ValueError, "scheduled-worker sentinel"):
                    main(
                        [str(run), "--scheduled-worker", "teach/train"],
                        backend_composition=composition,
                    )

            self.assertFalse((root / "iter0").exists())

    def test_targeted_fp_resume_ignores_removed_prefix_external_inputs(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            initial = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=initial)

            resumed = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            prefix_model = (
                root
                / "iter0"
                / "r1"
                / "train"
                / "student"
                / "student_model.pb"
            )
            prefix_mtime = prefix_model.stat().st_mtime_ns
            for path in (
                root / "data" / "teacher.xyz",
                root / "data" / "student.xyz",
                root / "data" / "seed.xyz",
                root / "recipes" / "teacher.yaml",
                root / "recipes" / "student.json",
                root / "recipes" / "sample.in",
            ):
                path.unlink()

            completed_selection = (
                root / "iter0" / "fp" / "selected.xyz"
            ).read_bytes()
            main([str(run)])
            self.assertEqual(
                (root / "iter0" / "fp" / "selected.xyz").read_bytes(),
                completed_selection,
            )

            main(
                [str(run), "--resume", "fp"],
                backend_composition=resumed,
            )

            self.assertEqual(prefix_model.stat().st_mtime_ns, prefix_mtime)
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_explicit_round_truncation_archives_higher_round_before_fp_resume(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            bundle = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=bundle)

            payload = json.loads(run.read_text(encoding="utf-8"))
            payload["sample"]["steps"] = [20]
            payload["screen"]["pool"] = [1]
            payload["screen"]["keep"] = [1]
            run.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "round shrink|higher-round"):
                main(
                    [str(run), "--resume", "fp"],
                    backend_composition=bundle,
                )
            self.assertTrue((root / "iter0" / "r1").is_dir())

            main(
                [
                    str(run),
                    "--resume", "fp",
                    "--truncate-after-round", "0",
                ],
                backend_composition=bundle,
            )

            archive = (
                root
                / ".dice_state"
                / "round_truncations"
                / "iter0"
                / "after-r0"
            )
            manifest = json.loads(
                (archive / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse((root / "iter0" / "r1").exists())
            self.assertTrue((archive / "evidence" / "iter0" / "r1").is_dir())
            self.assertTrue((root / "iter0" / "fp" / "selected.xyz").is_file())

    def test_targeted_sample_feasibility_fails_before_suffix_cleanup(self) -> None:
        from workflow_core.lifecycle.entrypoint import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = _write_public_run(root)
            initial = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            main([str(run)], backend_composition=initial)
            resumed = compose_test_backend_roles(
                run,
                teacher=_FakeTeacher(),
                student=_FakeStudent(),
                sampling=_FakeSampler(),
            )
            sample_evidence = (
                root
                / "iter0"
                / "r1"
                / "sample"
                / "tasks"
                / "task_000000"
                / "task_context.json"
            )
            before = sample_evidence.read_bytes()
            (root / "recipes" / "sample.in").unlink()

            with self.assertRaisesRegex(ValueError, r"r1/sample.*sampler\.recipe"):
                main(
                    [str(run), "--resume", "r1/sample"],
                    backend_composition=resumed,
                )

            self.assertEqual(sample_evidence.read_bytes(), before)
            self.assertTrue(sample_evidence.is_file())
            self.assertFalse((root / ".dice_state" / "markers").exists())

    def test_executes_fixed_blocks_then_reuses_complete_prefix_and_targeted_resume(self) -> None:
        from workflow_core.lifecycle.runner import execute_lifecycle
        from workflow_core.lifecycle.launch import LifecycleLaunch, LifecyclePlan, LifecycleBlock
        from workflow_core.lifecycle.config import RunConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            root=Path(tmpdir); raw={"iter":0}; config=RunConfig(path=root/"run.json",root=root,raw=raw)
            blocks=("teach/train","r0/train","fp"); launch=LifecycleLaunch(config=config,run_root=root,entropy_root=str(config.path),plan=LifecyclePlan(tuple(LifecycleBlock(name,root/"iter0"/name) for name in blocks)),environment_path=None,preflight=None,resume_target=None)
            seen=[]
            handlers={name: (lambda path,name=name: (seen.append(name),(path/"done").write_text(name,encoding="utf-8"))) for name in blocks}
            checks={name: (lambda path: (path/"done").is_file()) for name in blocks}
            execute_lifecycle(launch,handlers=handlers,artifact_checks=checks)
            self.assertEqual(seen,list(blocks))
            execute_lifecycle(launch,handlers=handlers,artifact_checks=checks)
            self.assertEqual(seen,list(blocks))
            resumed=LifecycleLaunch(config=config,run_root=root,entropy_root=str(config.path),plan=launch.plan,environment_path=None,preflight=None,resume_target="r0/train")
            execute_lifecycle(resumed,handlers=handlers,artifact_checks=checks)
            self.assertEqual(seen,["teach/train","r0/train","fp","r0/train","fp"])
