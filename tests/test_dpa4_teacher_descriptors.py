from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms


class _FittingNet:
    def __init__(
        self,
        width: int,
        *,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
    ) -> None:
        self.neuron = [width]
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.dim_case_embd = dim_case_embd
        self.mixed_types = True
        self.case_indices: list[int] = []
        projection = SimpleNamespace(
            w=np.ones((width, 1), dtype=np.float64),
            b=np.zeros(1, dtype=np.float64),
        )
        self.nets = SimpleNamespace(
            ndim=0,
            _networks=[SimpleNamespace(layers=[projection])],
        )

    def set_case_embd(self, case_index: int) -> None:
        self.case_indices.append(int(case_index))


class _DeepPot:
    def __init__(
        self,
        width: int,
        *,
        descriptor_width: int = 32,
        force_shift: float = 0.0,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        feature_prediction_shift: float = 0.0,
        descriptor_type: str = "dpa4",
        descriptor_output_width: int | None = None,
        rcut: float = 6.0,
        sel: tuple[int, ...] = (256,),
    ) -> None:
        self.width = width
        self.descriptor_width = descriptor_width
        self.force_shift = force_shift
        self.feature_prediction_shift = feature_prediction_shift
        self.feature_calls = 0
        self.calls = []
        self.descriptor_type = descriptor_type
        self.descriptor_output_width = (
            descriptor_width
            if descriptor_output_width is None
            else descriptor_output_width
        )
        self.rcut = float(rcut)
        self.sel = tuple(int(value) for value in sel)
        fitting = _FittingNet(
            width,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
        )
        atomic_model = SimpleNamespace(
            fitting_net=fitting,
            descriptor=SimpleNamespace(
                get_dim_out=lambda: self.descriptor_output_width
            ),
        )
        model = SimpleNamespace(
            atomic_model=atomic_model,
            set_case_embd=fitting.set_case_embd,
            get_dp_atomic_model=lambda: atomic_model,
            get_sel=lambda: list(self.sel),
        )
        self.deep_eval = SimpleNamespace(_dpmodel=model)

    def get_type_map(self):
        return ["H", "O"]

    def get_rcut(self):
        return self.rcut

    def get_model_def_script(self):
        return {
            "descriptor": {"type": self.descriptor_type, "channels": self.descriptor_width},
            "fitting_net": {"type": "dpa4_ener", "neuron": [0]},
        }

    def eval_descriptor(self, *_args, **_kwargs):
        raise AssertionError("DPA4 readout-input export must never call eval_descriptor")

    def eval_fitting_last_layer(
        self,
        coords,
        cells,
        atom_types,
        *,
        fparam=None,
        aparam=None,
        mixed_type=False,
        dtype="fp32",
    ):
        self.feature_calls += 1
        self.calls.append((
            np.asarray(atom_types).copy(),
            None if fparam is None else np.asarray(fparam).copy(),
            None if aparam is None else np.asarray(aparam).copy(),
            bool(mixed_type),
            np.asarray(coords).dtype,
            None if cells is None else np.asarray(cells).dtype,
        ))
        nframes, natoms = np.asarray(coords).shape[:2]
        base = np.asarray(coords)[:, :, 0:1] + np.asarray(atom_types)[None, :, None]
        return np.repeat(base, self.width, axis=2).reshape(nframes, natoms, self.width)

    def eval(self, coords, cells, atom_types, *, fparam=None, aparam=None, mixed_type=False):
        nframes, natoms = np.asarray(coords).shape[:2]
        energy = np.sum(np.asarray(coords), axis=(1, 2), keepdims=True)
        forces = np.full(
            (nframes, natoms, 3),
            self.force_shift
            + (self.feature_prediction_shift if self.feature_calls else 0.0),
            dtype=np.float64,
        )
        return energy, forces, np.zeros((nframes, 9)), np.zeros((nframes, natoms, 9))


def _frame(symbols: str, offset: float, identity: str) -> Atoms:
    atoms = Atoms(
        symbols,
        positions=np.arange(len(symbols) * 3, dtype=float).reshape(-1, 3) * 0.1 + offset,
        cell=[6, 6, 6],
        pbc=True,
    )
    atoms.info["structure_id"] = identity
    return atoms


@pytest.mark.parametrize("family", ["dpa4", "dpa4c"])
@pytest.mark.parametrize("canonical", [False, True])
def test_zbl_readout_uses_full_model_embedding_and_unique_learned_projection(family, canonical) -> None:
    from workflow_backends.teacher.deepmd.descriptors import DeepMDPTReadoutInputExporter

    terminal = _DeepPot(5, descriptor_type=family)
    learned_definition = terminal.get_model_def_script()
    learned_atomic = terminal.deep_eval._dpmodel.get_dp_atomic_model()
    # The analytical term may precede the learned term. Do not choose models[0].
    terminal.deep_eval._dpmodel = SimpleNamespace(
        get_dp_atomic_model=lambda: None,
        atomic_model=SimpleNamespace(models=[SimpleNamespace(potential="zbl"), learned_atomic]),
    )
    definition = ({"type": "linear_ener", "weights": "sum", "models": [
        {"type": "inner_potential", "mode": "zbl"}, learned_definition,
    ]} if canonical else {**learned_definition, "bridging_method": "zbl"})
    terminal.get_model_def_script = lambda: definition
    calls = []

    def full_embedding(coords, cells, atom_types, **kwargs):
        calls.append(True)
        shape = (*np.asarray(coords).shape[:2], 5)
        return np.zeros((*shape[:2], 2)), np.full(shape, 7, dtype=np.float32), np.zeros((1, 5))

    terminal.eval_embedding = full_embedding
    exporter = DeepMDPTReadoutInputExporter(
        "zbl.pt2", terminal_model=terminal, model_family=family,
    )
    chunk = exporter.export_frames([(0, _frame("OH", 0.0, "zbl-frame"))])

    np.testing.assert_array_equal(chunk.descriptors, np.full((2, 5), 7, dtype=np.float32))
    assert calls == [True]
    assert terminal.feature_calls == 0
    assert exporter.metadata.model_family == family


def test_zbl_readout_rejects_ambiguous_learned_atomic_models() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DeepMDPTReadoutInputExporter

    terminal = _DeepPot(5, descriptor_type="dpa4c")
    definition = terminal.get_model_def_script()
    terminal.get_model_def_script = lambda: {**definition, "bridging_method": "zbl"}
    atomic = terminal.deep_eval._dpmodel.get_dp_atomic_model()
    terminal.deep_eval._dpmodel = SimpleNamespace(
        get_dp_atomic_model=lambda: None,
        atomic_model=SimpleNamespace(models=[atomic, atomic]),
    )
    with pytest.raises(ValueError, match="resolved fitting/output metadata"):
        DeepMDPTReadoutInputExporter("ambiguous.pt2", terminal_model=terminal, model_family="dpa4c")


def test_analytical_zbl_support_does_not_enable_legacy_tabulated_readout() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DeepMDPTReadoutInputExporter

    terminal = _DeepPot(5, descriptor_type="dpa4c")
    definition = terminal.get_model_def_script()
    terminal.get_model_def_script = lambda: {**definition, "use_srtab": "legacy.table"}
    atomic = terminal.deep_eval._dpmodel.get_dp_atomic_model()
    terminal.deep_eval._dpmodel = SimpleNamespace(
        get_dp_atomic_model=lambda: None,
        atomic_model=SimpleNamespace(models=[atomic, SimpleNamespace(table="legacy.table")]),
    )
    with pytest.raises(ValueError, match="resolved fitting/output metadata"):
        DeepMDPTReadoutInputExporter("legacy.pt2", terminal_model=terminal, model_family="dpa4c")
    assert terminal.feature_calls == 0


def test_dpa4_stream_uses_observed_final_hidden_width_and_preserves_atom_order() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )
    from workflow_backends.descriptor_stream import validated_descriptor_chunks

    terminal = _DeepPot(11, descriptor_width=32)
    frames = [_frame("OH", 0.0, "frame-a"), _frame("HOH", 0.2, "frame-b")]
    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=terminal,
    )

    header, chunks = dpa4_readout_input_descriptor_stream(
        frames, exporter=exporter, chunk_frames=1
    )
    rows = tuple(validated_descriptor_chunks(header, chunks))

    assert header.schema["name"] == "readout-input"
    assert header.schema["revision"] == 1
    assert header.schema["backend"] == "deepmd-pt"
    assert header.schema["model_family"] == "dpa4"
    assert header.schema["width"] == 11
    assert header.schema["components"][0]["resolved_fitting_neuron"] == [11]
    assert header.schema["components"][0]["upstream_descriptor_width"] == 32
    assert [chunk.frame_ids for chunk in rows] == [("frame-a",), ("frame-b",)]
    assert [chunk.descriptors.shape for chunk in rows] == [(2, 11), (3, 11)]
    assert rows[0].type_indices.tolist() == [1, 0]
    assert rows[1].type_indices.tolist() == [0, 1, 0]


def test_dpa4_readout_retains_exact_nonzero_projection_rows_only() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    source = _DeepPot(6)
    projection = source.deep_eval._dpmodel.get_dp_atomic_model().fitting_net.nets._networks[0].layers[-1]
    projection.w[:, 0] = np.asarray([1.0, 0.0, 1.0e-15, 0.0, -2.0, 0.0])
    exporter = DPA4ReadoutInputExporter(Path("readout.pt"), terminal_model=source)

    header, chunks = dpa4_readout_input_descriptor_stream(
        [_frame("HO", 0.0, "projection")], exporter=exporter
    )
    row = tuple(chunks)[0]

    assert exporter.metadata.resolved_width == 6
    assert exporter.metadata.connected_hidden_indices == (0, 2, 4)
    assert header.schema["width"] == 3
    assert header.schema["components"][0]["retained_hidden_indices"] == [0, 2, 4]
    assert row.descriptors.shape == (2, 3)


@pytest.mark.parametrize("family", ["dpa4", "dpa4c"])
def test_pt_readout_uses_atomic_embedding_and_keeps_connected_atom_rows(family: str) -> None:
    from workflow_backends.teacher.deepmd.descriptors import DeepMDPTReadoutInputExporter

    class EmbeddingPotential(_DeepPot):
        def eval_embedding(self, coords, cells, atom_types, **kwargs):
            assert kwargs["dtype"] == "fp32"
            nframes, natoms = np.asarray(coords).shape[:2]
            atomic = np.arange(nframes * natoms * self.width, dtype=np.float32).reshape(
                nframes, natoms, self.width
            )
            return np.full((nframes, natoms, 2), -100.0), atomic, atomic.sum(axis=1)

        def eval_fitting_last_layer(self, *args, **kwargs):
            raise AssertionError("modern graph readout must use eval_embedding")

    potential = EmbeddingPotential(4, descriptor_type=family)
    projection = potential.deep_eval._dpmodel.get_dp_atomic_model().fitting_net.nets._networks[0].layers[-1]
    projection.w[:, 0] = [1.0, 0.0, 1.0e-15, -2.0]
    exporter = DeepMDPTReadoutInputExporter(
        Path("model.pt2"), terminal_model=potential, model_family=family
    )

    chunk = exporter.export_frames([(0, _frame("OH", 0.0, "embedding"))])

    np.testing.assert_array_equal(chunk.descriptors, [[0, 2, 3], [4, 6, 7]])
    assert chunk.type_indices.tolist() == [1, 0]
    assert chunk.frame_ids == ("embedding",)
    assert exporter.schema()["width"] == 3


def test_dpa4_readout_preflight_never_calls_terminal_energy_or_force_path() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    source = _DeepPot(5)
    source.eval = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("readout export must not evaluate terminal energy or forces")
    )
    exporter = DPA4ReadoutInputExporter(Path("readout.pt"), terminal_model=source)

    header, chunks = dpa4_readout_input_descriptor_stream(
        [_frame("HO", 0.0, "no-parity")], exporter=exporter
    )

    assert header.schema["width"] == 5
    assert tuple(chunks)[0].descriptors.shape == (2, 5)


def test_pt_readout_keeps_legacy_models_without_embedding_support() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    class LegacyPotential(_DeepPot):
        def eval_embedding(self, *args, **kwargs):
            raise NotImplementedError("archive predates forward_embedding")

    potential = LegacyPotential(3)
    exporter = DPA4ReadoutInputExporter(Path("legacy.pt"), terminal_model=potential)
    chunk = exporter.export_frames([(0, _frame("OH", 0.0, "legacy"))])

    assert chunk.descriptors.shape == (2, 3)
    assert potential.feature_calls == 1
    assert np.isfinite(chunk.descriptors).all()


@pytest.mark.parametrize("descriptor_type", ["dpa4", "DPA4", "SeZM", "sezm"])
def test_official_deepmd_dpa4_aliases_share_one_canonical_readout_family(
    descriptor_type: str,
) -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    exporter = DPA4ReadoutInputExporter(
        Path("readout.pt"),
        terminal_model=_DeepPot(5, descriptor_type=descriptor_type),
    )

    assert exporter.metadata.model_family == "dpa4"
    assert exporter.schema()["model_family"] == "dpa4"


def test_dpa4_readout_rejects_active_vacuum_contribution_removal() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    source = _DeepPot(5)
    source.deep_eval._dpmodel.get_dp_atomic_model().fitting_net.remove_vaccum_contribution = [True]

    with pytest.raises(ValueError, match=r"atom_ener.*remove_vaccum_contribution"):
        DPA4ReadoutInputExporter(Path("readout.pt"), terminal_model=source)


def test_dpa4_score_stream_does_not_repeat_descriptor_preflight() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    frames = [
        _frame("HO", float(index) * 0.1, f"frame-{index}")
        for index in range(4)
    ]
    observed: list[tuple[str, ...]] = []
    for chunk_frames in (1, 4):
        exporter = DPA4ReadoutInputExporter(
            Path("terminal.pt2"), terminal_model=_DeepPot(8)
        )
        exporter.preflight = lambda bounded: observed.append(
            tuple(str(frame.info["structure_id"]) for frame in bounded)
        )
        _header, chunks = dpa4_readout_input_descriptor_stream(
            frames,
            exporter=exporter,
            chunk_frames=chunk_frames,
        )
        tuple(chunks)

    assert observed == []


def test_dpa4_native_load_enables_deterministic_runtime_and_auto_neighbor_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
    )

    terminal = tmp_path / "terminal.pt2"
    terminal.write_bytes(b"terminal")
    calls: list[tuple[str, dict[str, object]]] = []
    infer_module = ModuleType("deepmd.infer")

    def deep_pot(path: str, **kwargs: object) -> _DeepPot:
        calls.append((path, kwargs))
        return _DeepPot(8)

    infer_module.DeepPot = deep_pot
    deepmd_module = ModuleType("deepmd")
    deepmd_module.infer = infer_module
    monkeypatch.setitem(sys.modules, "deepmd", deepmd_module)
    monkeypatch.setitem(sys.modules, "deepmd.infer", infer_module)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    deterministic_calls: list[bool] = []
    torch_module = ModuleType("torch")
    torch_module.use_deterministic_algorithms = deterministic_calls.append
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    DPA4ReadoutInputExporter(terminal, device="cuda")

    assert __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert deterministic_calls == [True]
    assert calls == [(
        str(terminal.resolve()),
        {
            "device": "cuda",
            "auto_batch_size": False,
            "neighbor_graph_method": "auto",
        },
    )]


def test_dpa4c_native_load_requests_auto_neighbor_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    terminal = tmp_path / "student.pt2"
    terminal.write_bytes(b"terminal")
    calls = []
    infer_module = ModuleType("deepmd.infer")

    def deep_pot(path, **kwargs):
        calls.append((path, kwargs))
        return _DeepPot(8, descriptor_type="dpa4c", descriptor_output_width=208)

    infer_module.DeepPot = deep_pot
    deepmd_module = ModuleType("deepmd")
    deepmd_module.infer = infer_module
    monkeypatch.setitem(sys.modules, "deepmd", deepmd_module)
    monkeypatch.setitem(sys.modules, "deepmd.infer", infer_module)
    deterministic_calls: list[bool] = []
    torch_module = ModuleType("torch")
    torch_module.use_deterministic_algorithms = deterministic_calls.append
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    DPA4ReadoutInputExporter(terminal, device="cuda", model_family="dpa4c")
    assert deterministic_calls == [True]
    assert calls[0][1]["neighbor_graph_method"] == "auto"
    assert calls[0][1]["auto_batch_size"] is False


def test_dpa4_atom_budget_controls_total_atoms_per_public_forward() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    terminal = _DeepPot(13)
    frames = [_frame("HO", index * 0.1, f"batch-{index}") for index in range(5)]
    exporter = DPA4ReadoutInputExporter(Path("terminal.pt2"), terminal_model=terminal)
    header, chunks = dpa4_readout_input_descriptor_stream(
        frames, exporter=exporter, batch_size=4, chunk_frames=1
    )
    rows = tuple(chunks)
    assert terminal.feature_calls == 3
    assert [len(row.frame_ids) for row in rows] == [2, 2, 1]
    assert header.schema["width"] == 13


def test_dpa4_readout_source_rejects_symlink(tmp_path: Path) -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
    )

    target = tmp_path / "target.pt"
    target.write_bytes(b"checkpoint")
    source = tmp_path / "teacher_readout_checkpoint.pt"
    source.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        DPA4ReadoutInputExporter(source)


def test_dpa4c_stream_does_not_run_an_ase_neighbor_capacity_guard() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    frame = _frame("H", 0.0, "over-capacity")
    frame.set_cell([4.0, 4.0, 4.0], scale_atoms=False)
    terminal = _DeepPot(
        8,
        descriptor_type="dpa4c",
        descriptor_output_width=208,
        sel=(1,),
    )
    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=terminal,
    )

    header, chunks = dpa4_readout_input_descriptor_stream([frame], exporter=exporter)

    assert len(tuple(chunks)) == 1
    assert terminal.feature_calls == 1
    assert "dense_neighbor_guard" not in header.schema["components"][0]


def test_dpa4_stream_remaps_native_superset_type_map_to_active_route() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )
    from workflow_backends.descriptor_stream import validated_descriptor_chunks

    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=_DeepPot(7),
    )
    frames = [_frame("O", 0.0, "oxygen-a"), _frame("OO", 0.2, "oxygen-b")]

    header, chunks = dpa4_readout_input_descriptor_stream(
        frames,
        exporter=exporter,
        type_map=("O",),
        chunk_frames=1,
    )
    rows = tuple(validated_descriptor_chunks(header, chunks))

    assert header.type_map == ("O",)
    assert [chunk.type_indices.tolist() for chunk in rows] == [[0], [0, 0]]
    assert header.schema["type_map"] == ["H", "O"]


def test_dpa4c_stream_maps_native_inputs_and_active_membership_by_symbol() -> None:
    from workflow_backends.descriptor_stream import validated_descriptor_chunks
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    terminal = _DeepPot(
        10,
        descriptor_width=64,
        descriptor_type="dpa4c",
        descriptor_output_width=208,
    )
    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"), terminal_model=terminal
    )
    header, chunks = dpa4_readout_input_descriptor_stream(
        [_frame("OH", 0.0, "frame-a")],
        exporter=exporter,
        type_map=("O", "H"),
        chunk_frames=1,
    )
    rows = tuple(validated_descriptor_chunks(header, chunks))

    assert header.type_map == ("O", "H")
    assert header.schema["model_family"] == "dpa4c"
    assert header.schema["type_map"] == ["H", "O"]
    assert header.schema["components"][0]["upstream_descriptor_width"] == 208
    assert terminal.calls[0][0].tolist() == [1, 0]
    assert rows[0].type_indices.tolist() == [0, 1]


def test_dpa4c_readout_builds_float32_native_inputs_and_conditioning() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    terminal = _DeepPot(
        10,
        descriptor_type="dpa4c",
        descriptor_output_width=208,
        numb_fparam=2,
        numb_aparam=1,
    )
    frame = _frame("OH", 0.0, "dtype")
    frame.info["deepmd_fparam"] = np.asarray([1.0, 2.0], dtype=np.float64)
    frame.arrays["deepmd_aparam"] = np.asarray([[3.0], [4.0]], dtype=np.float64)
    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"), terminal_model=terminal, model_family="dpa4c"
    )

    _, chunks = dpa4_readout_input_descriptor_stream([frame], exporter=exporter)
    tuple(chunks)

    atom_types, fparam, aparam, _, coords_dtype, cells_dtype = terminal.calls[0]
    assert atom_types.dtype == np.int32
    assert fparam.dtype == np.float32
    assert aparam.dtype == np.float32
    assert coords_dtype == np.float32
    assert cells_dtype == np.float32


def test_dpa4c_exporter_rejects_ambiguous_or_multioutput_projection() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    ambiguous = _DeepPot(8, descriptor_type="dpa4c", descriptor_output_width=208)
    fitting = ambiguous.deep_eval._dpmodel.get_dp_atomic_model().fitting_net
    fitting.nets._networks.append(fitting.nets._networks[0])
    with pytest.raises(ValueError, match="lacks resolved fitting/output metadata"):
        DPA4ReadoutInputExporter(
            Path("terminal.pt2"), terminal_model=ambiguous
        )

    multioutput = _DeepPot(8, descriptor_type="dpa4c", descriptor_output_width=208)
    projection = (
        multioutput.deep_eval._dpmodel.get_dp_atomic_model()
        .fitting_net.nets._networks[0].layers[-1]
    )
    projection.w = np.zeros((8, 2))
    with pytest.raises(ValueError, match="hidden_width x 1"):
        DPA4ReadoutInputExporter(
            Path("terminal.pt2"), terminal_model=multioutput
        )


def test_dpa4_exporter_omits_hidden_dimension_disconnected_from_energy() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    terminal = _DeepPot(8, descriptor_type="dpa4c", descriptor_output_width=208)
    projection = (
        terminal.deep_eval._dpmodel.get_dp_atomic_model()
        .fitting_net.nets._networks[0].layers[-1]
    )
    projection.w[3, 0] = 0.0

    exporter = DPA4ReadoutInputExporter(Path("terminal.pt2"), terminal_model=terminal)
    assert exporter.metadata.connected_hidden_indices == (0, 1, 2, 4, 5, 6, 7)


def test_dpa4c_stream_fails_closed_when_terminal_rejects_nonscalar_source() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    class _Unsupported(_DeepPot):
        def eval_fitting_last_layer(self, *args, **kwargs):
            raise NotImplementedError("equivariant fitting input")

    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=_Unsupported(
            8, descriptor_type="dpa4c", descriptor_output_width=208
        ),
    )
    with pytest.raises(ValueError, match="does not expose scalar fitting-last"):
        dpa4_readout_input_descriptor_stream(
            [_frame("HO", 0.0, "unsupported")], exporter=exporter
        )


def test_dpa4_stream_rejects_active_element_absent_from_native_model() -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )

    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=_DeepPot(7),
    )
    with pytest.raises(ValueError, match="active type_map.*native"):
        dpa4_readout_input_descriptor_stream(
            [_frame("O", 0.0, "oxygen")],
            exporter=exporter,
            type_map=("O", "Pt"),
        )


def test_dpa4_auto_width_uses_resolved_projection_not_recipe_neuron_zero() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=_DeepPot(96, descriptor_width=32),
    )
    assert exporter.metadata.resolved_width == 96
    assert exporter.metadata.configured_neuron == (0,)
    assert exporter.metadata.upstream_descriptor_width == 32


def test_dpa4_required_conditioning_has_explicit_structure_mapping() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    terminal = _DeepPot(7, numb_fparam=2, numb_aparam=1, dim_case_embd=3)
    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=terminal,
    )
    frame = _frame("HO", 0.0, "conditioned")
    with pytest.raises(ValueError, match="deepmd_fparam"):
        exporter.export_frames(((0, frame),))

    frame.info["deepmd_fparam"] = [1.0, 2.0]
    frame.arrays["deepmd_aparam"] = np.asarray([[3.0], [4.0]])
    frame.info["deepmd_case_index"] = 2
    chunk = exporter.export_frames(((0, frame),))
    assert chunk.descriptors.shape == (2, 7)
    assert terminal.calls[0][1].shape == (1, 2)
    assert terminal.calls[0][2].shape == (1, 2, 1)
    fitting = terminal.deep_eval._dpmodel.get_dp_atomic_model().fitting_net
    assert fitting.case_indices == [2]


def test_dpa4_preflight_does_not_compare_terminal_state_around_export() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    frame = _frame("HO", 0.0, "parity")
    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=_DeepPot(9, feature_prediction_shift=0.01),
    )
    exporter.preflight((frame,))


def test_dpa4_preflight_accepts_measured_terminal_force_roundoff() -> None:
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter

    exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"),
        terminal_model=_DeepPot(9, feature_prediction_shift=2.1e-4),
    )
    exporter.preflight((_frame("HO", 0.0, "measured-roundoff"),))


def test_dpa4_adapter_resolves_manifest_readout_source_for_gmm_operations(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.adapter import DeepMDTeacherAdapter
    from workflow_core.lifecycle.data import DataService, LabelContract
    from workflow_core.lifecycle.model_handles import (
        ModelArtifact,
        publish_model_handles,
    )
    from workflow_artifacts.lifecycle.deepmd_model_assets import (
        publish_deepmd_model_bundle,
    )
    from workflow_policies.gmm_reference_tail import GMMReferenceFitResult, GMMScoreResult

    stage = tmp_path / "iter0" / "teach" / "train"
    member = stage / "teacher"
    member.mkdir(parents=True)
    terminal = member / "native.pt2"
    terminal.write_bytes(b"terminal")
    checkpoint = member / "teacher_readout_checkpoint.pt"
    checkpoint.write_bytes(b"selected-readout-checkpoint")
    published = publish_model_handles(
        (ModelArtifact("teacher", terminal),),
        expected_member_ids=("teacher",),
        stage_root=stage,
        role="teacher",
    )[0].path
    (member / "deepmd_model_metadata.json").write_text(
        json.dumps({"model_family": "dpa4", "type_map": ["H", "O"]}),
        encoding="utf-8",
    )
    publish_deepmd_model_bundle(
        member_root=member,
        member_id="teacher",
        member_order=1,
        model_family="dpa4",
        native_type_map=("H", "O"),
        inference_terminal_path=published,
        readout_source_path=checkpoint,
        checkpoint_family="ema",
        selected_checkpoint_argument="model_ema.ckpt.pt",
        freeze_command_profile={"role": "teacher.deepmd.pt.train", "cli": "pt"},
    )

    class Executor:
        def __init__(self) -> None:
            self.fit = None
            self.stream = None

        def fit_gmm_router(self, model, **kwargs):
            self.fit = (model, kwargs)
            return GMMReferenceFitResult(Path(kwargs["router_dir"]), "gmm-reference-tail@4", {})

        def score_gmm_router_stream(self, model, **kwargs):
            self.stream = (model, kwargs)
            ids = tuple(kwargs["frame_ids"])
            return GMMScoreResult(
                np.asarray(ids, dtype=np.str_),
                np.ones(len(ids)),
                np.zeros(len(ids)),
                None,
                descriptor_schema={"name": "readout-input", "width": 1},
            )

    data = DataService(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        teacher_type_map=("H", "O"),
        student_type_map=("H", "O"),
        teacher_contract=LabelContract(),
        student_contract=LabelContract(),
    )
    executor = Executor()
    adapter = DeepMDTeacherAdapter(
        executor=executor,
        run_json_path=data.run_json_path,
        iter_index=data.iter_index,
        cli="pt",
        work_root=stage,
    )
    reference = tmp_path / "reference.xyz"
    reference.write_text("reference", encoding="utf-8")
    frames = (_frame("H", 0.0, "a"), _frame("O", 0.1, "b"))

    from workflow_backends.teacher.readout import (
        TeacherCandidateScoreRequest,
        TeacherReferenceFitRequest,
    )

    adapter.fit_reference(TeacherReferenceFitRequest(
        model=published,
        reference_path=reference,
        router_dir=tmp_path / "router",
        work_dir=tmp_path / "fit-work",
        type_map=("H", "O"),
    ))
    adapter.score_candidates(TeacherCandidateScoreRequest(
        model=published,
        reference_path=reference,
        frame_chunks=(frames,),
        ordered_identities=("a", "b"),
        router_dir=tmp_path / "router",
        work_dir=tmp_path / "stream-work",
        emit_frame_means=False,
    ))

    assert executor.fit[0] == checkpoint.resolve()
    assert "descriptor_checkpoint" not in executor.fit[1]
    assert "checkpoint_family" not in executor.fit[1]
    assert "descriptor_checkpoint" not in executor.stream[1]
    assert executor.stream[0] == checkpoint.resolve()
    assert not (stage / "teacher_model_assets.json").exists()


def test_dpa4c_adapter_keeps_terminal_as_the_gmm_source(tmp_path: Path) -> None:
    from workflow_backends.teacher.deepmd.adapter import DeepMDTeacherAdapter
    from workflow_backends.teacher.readout import TeacherReferenceFitRequest
    from workflow_core.lifecycle.data import DataService, LabelContract
    from workflow_core.lifecycle.model_handles import ModelArtifact, publish_model_handles

    stage = tmp_path / "iter0" / "teach" / "train"
    member = stage / "teacher"
    member.mkdir(parents=True)
    terminal = member / "native.pt2"
    terminal.write_bytes(b"terminal")
    published = publish_model_handles(
        (ModelArtifact("teacher", terminal),),
        expected_member_ids=("teacher",),
        stage_root=stage,
        role="teacher",
    )[0].path
    (member / "deepmd_model_metadata.json").write_text(
        json.dumps({"model_family": "dpa4c", "type_map": ["H", "O"]}),
        encoding="utf-8",
    )

    class Executor:
        def fit_gmm_router(self, model, **kwargs):
            self.model = model
            return GMMReferenceFitResult(Path(kwargs["router_dir"]), "gmm-reference-tail@4", {})

    from workflow_policies.gmm_reference_tail import GMMReferenceFitResult

    adapter = DeepMDTeacherAdapter(
        executor=(executor := Executor()),
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        cli="pt",
        work_root=stage,
    )
    reference = tmp_path / "reference.xyz"
    reference.write_text("reference", encoding="utf-8")

    adapter.fit_reference(TeacherReferenceFitRequest(
        model=published,
        reference_path=reference,
        router_dir=tmp_path / "router",
        work_dir=tmp_path / "fit-work",
        type_map=("H", "O"),
    ))

    assert executor.model == published.resolve()


def test_dpa4_chunk_worker_loads_one_terminal_without_duplicate_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from workflow_artifacts.gmm_worker_evidence import gmm_score_chunks_payload
    from workflow_backends.teacher.deepmd import descriptor_worker
    from workflow_backends.teacher.deepmd.descriptors import DPA4ReadoutInputExporter
    from workflow_core.structures.formats import write_extxyz_structures
    from workflow_policies.gmm_reference_tail import GMMScoreResult

    chunk_a = write_extxyz_structures(
        tmp_path / "chunk-a.xyz", [_frame("H", 0.0, "frame-a")]
    )
    chunk_b = write_extxyz_structures(
        tmp_path / "chunk-b.xyz", [_frame("O", 0.1, "frame-b")]
    )
    manifest = tmp_path / "chunks.json"
    manifest.write_text(
        json.dumps(
            gmm_score_chunks_payload(
                frame_count=2,
                chunks=(
                    {"path": str(chunk_a), "frame_ids": ["frame-a"]},
                    {"path": str(chunk_b), "frame_ids": ["frame-b"]},
                ),
            )
        ),
        encoding="utf-8",
    )
    counts = {"loads": 0, "preflights": 0, "scores": 0}

    def exporter_factory(*args, **kwargs):
        counts["loads"] += 1
        exporter = DPA4ReadoutInputExporter(
            Path(args[0]), terminal_model=_DeepPot(5)
        )
        original_preflight = exporter.preflight

        def measured_preflight(frames):
            counts["preflights"] += 1
            return original_preflight(frames)

        exporter.preflight = measured_preflight
        return exporter

    def score(request):
        counts["scores"] += 1
        chunks = tuple(request.chunks)
        assert [chunk.frame_ids for chunk in chunks] == [("frame-a",), ("frame-b",)]
        return GMMScoreResult(
            np.asarray(["frame-a", "frame-b"], dtype=np.str_),
            np.asarray([0.2, 0.8]),
            np.asarray([1.0, 2.0]),
            None,
            request.header.schema,
        )

    monkeypatch.setattr(descriptor_worker, "DPA4ReadoutInputExporter", exporter_factory)
    monkeypatch.setattr(descriptor_worker, "_router_type_map", lambda _path: ("H", "O"))
    monkeypatch.setattr(descriptor_worker, "score_gmm_router_stream", score)
    assert descriptor_worker.main([
        "--operation", "score",
        "--checkpoint", str(tmp_path / "teacher_readout_checkpoint.pt"),
        "--chunk-manifest", str(manifest),
        "--router-dir", str(tmp_path / "router"),
        "--model-digest", "terminal-digest",
        "--reference-digest", "reference-digest",
        "--chunk-frames", "1",
    ]) == 0

    assert counts == {"loads": 1, "preflights": 0, "scores": 1}
    payload = json.loads(capsys.readouterr().out.split("=", 1)[1])
    assert payload["frame_ids"] == ["frame-a", "frame-b"]


def test_dpa4_reference_and_score_share_schema_and_dynamic_frame_means(tmp_path: Path) -> None:
    from workflow_backends.teacher.deepmd.descriptors import (
        DPA4ReadoutInputExporter,
        dpa4_readout_input_descriptor_stream,
    )
    from workflow_policies.gmm_reference_tail import (
        GMMReferenceStreamFitRequest,
        GMMStreamScoreRequest,
        fit_gmm_router_stream,
        score_gmm_router_stream,
    )

    reference = []
    for symbol in ("H", "O"):
        for index in range(12):
            reference.append(_frame(symbol, index * 0.07, f"ref-{symbol}-{index}"))
    reference_exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"), terminal_model=_DeepPot(13),
    )
    header, chunks = dpa4_readout_input_descriptor_stream(
        reference, exporter=reference_exporter, chunk_frames=5
    )
    fit = fit_gmm_router_stream(GMMReferenceStreamFitRequest(
        router_dir=tmp_path / "router",
        header=header,
        chunks=chunks,
        model_digest="terminal-digest",
        reference_digest="reference-digest",
    ))

    candidates = [_frame("HO", 0.13, "candidate-a"), _frame("OH", 0.31, "candidate-b")]
    score_exporter = DPA4ReadoutInputExporter(
        Path("terminal.pt2"), terminal_model=_DeepPot(13),
    )
    score_header, score_chunks = dpa4_readout_input_descriptor_stream(
        candidates, exporter=score_exporter, chunk_frames=1
    )
    result = score_gmm_router_stream(GMMStreamScoreRequest(
        router_dir=fit.router_dir,
        header=score_header,
        chunks=score_chunks,
        model_digest="terminal-digest",
        reference_digest="reference-digest",
        emit_frame_means=True,
    ))

    assert result.frame_ids.tolist() == ["candidate-a", "candidate-b"]
    assert result.frame_means.shape == (2, 13)
    assert result.descriptor_schema == fit.descriptor_schema == header.schema
    assert not any(path.name.startswith("descriptor") for path in tmp_path.rglob("*"))
