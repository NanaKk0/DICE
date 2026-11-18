"""CPU-only test doubles copied from the upstream lifecycle regression fixture.

Source: tests/test_lifecycle_executor.py, snapshot 2026-11-01.
These models/trajectories are synthetic and have no physical meaning.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


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
        seed = Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
        sampled = Atoms("H", positions=[[0.1, 0, 0]], cell=[4, 4, 4], pbc=True)
        return [(seed, 0), (sampled, 20)]

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
