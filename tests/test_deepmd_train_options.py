from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_run(root: Path, payload: dict) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "recipes").mkdir()
    for name in ("teacher.xyz", "student.xyz", "seed.xyz"):
        (root / "data" / name).write_text("placeholder\n", encoding="utf-8")
    (root / "recipes" / "teacher.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "dpa4",
                    "descriptor": {"type": "dpa4", "sel": 12},
                },
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "recipes" / "student.json").write_text(
        json.dumps({"model": {"descriptor": {"type": "se_e2_a"}}, "training": {}}),
        encoding="utf-8",
    )
    (root / "recipes" / "teacher.yaml").write_text("model: MACE\n", encoding="utf-8")
    (root / "recipes" / "student.in").write_text("version 4\n", encoding="utf-8")
    run = root / "run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run


def _payload(*, teacher_backend: str = "DeepMD", student_backend: str = "DeepMD") -> dict:
    teacher = {
        "backend": teacher_backend,
        "model": "single",
        "type_map": ["H"],
        "recipe": "recipes/teacher.json" if teacher_backend == "DeepMD" else "recipes/teacher.yaml",
        "policy": "from_scratch",
        "valid_frac": 0.0,
    }
    if teacher_backend == "DeepMD":
        teacher["cli"] = "pt"
    student = {
        "backend": student_backend,
        "model": "single",
        "type_map": ["H"],
        "recipe": "recipes/student.json" if student_backend == "DeepMD" else "recipes/student.in",
        "policy": "from_scratch",
        "valid_frac": 0.0,
    }
    if student_backend == "DeepMD":
        student["cli"] = "tf"
    return {
        "iter": 0,
        "teacher_train_bootstrap": ["data/teacher.xyz"],
        "student_train_bootstrap": ["data/student.xyz"],
        "fp_data": [],
        "teacher": teacher,
        "student": student,
        "sampler": {
            "backend": "LAMMPS" if student_backend == "DeepMD" else "GPUMD",
            "recipe": "recipes/student.json" if student_backend == "DeepMD" else "recipes/student.in",
        },
        "sample": {
            "systems": {"water": {"seed": "data/seed.xyz"}},
            "steps": [10],
            "ensemble": "nvt",
            "temperatures_k": [300],
        },
        "screen": {
            "pool": [1],
            "keep": [1],
            "fp_pool": 1,
            "policy": "FPS",
        },
        "fp": {"keep": 1, "policy": "random"},
    }


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, None),
        ({}, None),
        ({"skip_neighbor_stat": False}, None),
        ({"skip_neighbor_stat": True}, {"skip_neighbor_stat": True}),
    ],
)
def test_deepmd_teacher_train_options_have_closed_false_to_omission_normalization(
    tmp_path: Path, configured: dict[str, bool] | None, expected: dict[str, bool] | None
) -> None:
    from workflow_core.lifecycle.config import load_run_config

    payload = _payload()
    if configured is not None:
        payload["teacher"]["train_options"] = configured

    config = load_run_config(_write_run(tmp_path, payload))

    assert config.raw["teacher"].get("train_options") == expected


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        ({"unknown": True}, r"teacher\.train_options contains unsupported field\(s\): unknown"),
        ({"skip_neighbor_stat": "true"}, r"teacher\.train_options\.skip_neighbor_stat must be a boolean"),
        ([], r"teacher\.train_options must be an object"),
    ],
)
def test_deepmd_teacher_train_options_reject_malformed_values(
    tmp_path: Path, configured: object, message: str
) -> None:
    from workflow_core.lifecycle.config import load_run_config

    payload = _payload()
    payload["teacher"]["train_options"] = configured

    with pytest.raises(ValueError, match=message):
        load_run_config(_write_run(tmp_path, payload))


def test_non_deepmd_teacher_rejects_train_options_with_backend_specific_error(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.config import load_run_config

    payload = _payload(teacher_backend="MACE")
    payload["teacher"]["train_options"] = {"skip_neighbor_stat": True}

    with pytest.raises(ValueError, match=r"teacher\.train_options is valid only for DeepMD"):
        load_run_config(_write_run(tmp_path, payload))


def test_native_teacher_train_appends_skip_flag_once_and_keeps_freeze_unchanged(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.native import NativeDeepMDTeacherExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> commands.txt\n"
        "if [ \"$2\" = train ]; then printf checkpoint > model.ckpt.index; fi\n"
        "if [ \"$2\" = freeze ]; then printf model > native.pb; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps({"model": {"type_map": ["H"], "descriptor": {"sel": 12}}, "training": {}}),
        encoding="utf-8",
    )
    context = OperationContext(
        role="teacher.deepmd.tf.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    execution = NativeDeepMDTeacherExecutor(train_context=context).train(
        input_json,
        tmp_path,
        "tf",
        train_options={"skip_neighbor_stat": True},
    )

    commands = (tmp_path / "commands.txt").read_text(encoding="utf-8").splitlines()
    assert commands == [
        f"--tf train {input_json.resolve()} --skip-neighbor-stat",
        "--tf freeze -c model.ckpt -o native.pb",
    ]
    assert commands[0].count("--skip-neighbor-stat") == 1
    assert "--skip-neighbor-stat" not in commands[1]
    assert execution.evidence["train_command"] == (
        "dp",
        "--tf",
        "train",
        str(input_json.resolve()),
        "--skip-neighbor-stat",
    )


def test_native_teacher_pt_foundation_keeps_finetune_before_skip_flag(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.native import NativeDeepMDTeacherExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> commands.txt\n"
        "if [ \"$2\" = train ]; then printf checkpoint > model.ckpt.pt; fi\n"
        "if [ \"$2\" = freeze ]; then printf model > native.pt2; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps({"model": {"type_map": ["H"], "descriptor": {"sel": 12}}, "training": {}}),
        encoding="utf-8",
    )
    foundation = tmp_path / "foundation.pt2"
    foundation.write_text("foundation", encoding="utf-8")
    context = OperationContext(
        role="teacher.deepmd.pt.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    NativeDeepMDTeacherExecutor(train_context=context).train(
        input_json,
        tmp_path,
        "pt",
        foundation_model=foundation,
        train_options={"skip_neighbor_stat": True},
    )

    assert (tmp_path / "commands.txt").read_text(encoding="utf-8").splitlines() == [
        f"--pt train {input_json.resolve()} --finetune {foundation.resolve()} --skip-neighbor-stat",
        "--pt freeze -c model.ckpt.pt -o native",
    ]


def test_native_teacher_skip_rejects_every_automatic_sel_before_dp_without_mutation(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.native import NativeDeepMDTeacherExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\nprintf launched > launched\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    rendered = {
        "model": {
            "type_map": ["H"],
            "descriptor": {"sel": "auto"},
            "fitting_net": {"layers": [{"sel": "auto:factor"}]},
        },
        "training": {},
    }
    input_json.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    before = input_json.read_bytes()
    context = OperationContext(
        role="teacher.deepmd.pt.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    with pytest.raises(
        ValueError,
        match=r"model\.descriptor\.sel='auto'.*model\.fitting_net\.layers\[0\]\.sel='auto:factor'",
    ):
        NativeDeepMDTeacherExecutor(train_context=context).train(
            input_json,
            tmp_path,
            "pt",
            train_options={"skip_neighbor_stat": True},
        )

    assert input_json.read_bytes() == before
    assert not (tmp_path / "launched").exists()


def test_typed_teacher_adapter_passes_normalized_train_options_to_every_member(
    tmp_path: Path,
) -> None:
    from workflow_backends.teacher.deepmd.adapter import DeepMDTeacherAdapter
    from workflow_backends.teacher.training import TeacherTrainRequest
    from workflow_core.lifecycle.data import DataService, LabelContract
    from workflow_runtime.lifecycle.environment import OperationContext

    class RecordingExecutor:
        requires_native_dataset = False

        def __init__(self) -> None:
            self.options: list[object] = []

        def train(self, input_json, member_dir, cli, *, train_options=None, **kwargs):
            del kwargs
            self.options.append(train_options)
            terminal = Path(member_dir) / "native.pb"
            terminal.write_text("model", encoding="utf-8")
            return type(
                "Result",
                (),
                {
                    "terminal_model_path": terminal,
                    "evidence": {
                        "train_command": ("dp", f"--{cli}", "train", str(input_json)),
                    },
                },
            )()

    recipe = tmp_path / "teacher.json"
    recipe.write_text(json.dumps({"model": {}, "training": {}}), encoding="utf-8")
    train = tmp_path / "train.xyz"
    train.write_text("train", encoding="utf-8")
    data = DataService(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        teacher_type_map=("H",),
        student_type_map=("H",),
        teacher_contract=LabelContract(),
        student_contract=LabelContract(),
    )
    executor = RecordingExecutor()
    adapter = DeepMDTeacherAdapter(
        executor=executor,
        run_json_path=data.run_json_path,
        iter_index=data.iter_index,
        cli="tf",
        work_root=tmp_path / "teach",
    )
    request = TeacherTrainRequest(
        iter_index=0,
        recipe_path=recipe,
        train_data_path=train,
        validation_data_path=train,
        validation_mode="self",
        model_shape="committee",
        member_ids=("teacher_01", "teacher_02"),
        type_map=("H",),
        policy="from_scratch",
        foundation_asset=None,
        train_options={"skip_neighbor_stat": True},
    )

    result = adapter.train(request)

    assert executor.options == [
        {"skip_neighbor_stat": True},
        {"skip_neighbor_stat": True},
    ]
    assert all(
        artifact.evidence["train_options"] == {"skip_neighbor_stat": True}
        for artifact in result.member_artifacts
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, None),
        ({}, None),
        ({"skip_neighbor_stat": False}, None),
        ({"skip_neighbor_stat": True}, {"skip_neighbor_stat": True}),
    ],
)
def test_deepmd_student_train_options_share_false_to_omission_normalization(
    tmp_path: Path, configured: dict[str, bool] | None, expected: dict[str, bool] | None
) -> None:
    from workflow_core.lifecycle.config import load_run_config

    payload = _payload()
    if configured is not None:
        payload["student"]["train_options"] = configured

    config = load_run_config(_write_run(tmp_path, payload))

    assert config.raw["student"].get("train_options") == expected


def test_non_deepmd_student_rejects_train_options_with_backend_specific_error(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.config import load_run_config

    payload = _payload(student_backend="NEP")
    payload["student"]["train_options"] = {"skip_neighbor_stat": True}

    with pytest.raises(ValueError, match=r"student\.train_options is valid only for DeepMD"):
        load_run_config(_write_run(tmp_path, payload))


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        ({"unknown": True}, r"student\.train_options contains unsupported field\(s\): unknown"),
        ({"skip_neighbor_stat": "true"}, r"student\.train_options\.skip_neighbor_stat must be a boolean"),
    ],
)
def test_deepmd_student_train_options_reject_malformed_values(
    tmp_path: Path, configured: object, message: str
) -> None:
    from workflow_core.lifecycle.config import load_run_config

    payload = _payload()
    payload["student"]["train_options"] = configured

    with pytest.raises(ValueError, match=message):
        load_run_config(_write_run(tmp_path, payload))


def test_native_student_tf_train_appends_skip_flag_once_and_not_to_freeze(
    tmp_path: Path,
) -> None:
    from workflow_backends.student.deepmd.native import NativeDeepMDStudentExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> commands.txt\n"
        "if [ \"$2\" = freeze ]; then printf model > student_native.pb; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps({"model": {"type_map": ["H"], "descriptor": {"sel": 12}}, "training": {}}),
        encoding="utf-8",
    )
    context = OperationContext(
        role="student.deepmd.tf.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    NativeDeepMDStudentExecutor(train_context=context).train(
        input_json,
        tmp_path,
        "tf",
        recipe_kind="standard",
        train_options={"skip_neighbor_stat": True},
    )

    commands = (tmp_path / "commands.txt").read_text(encoding="utf-8").splitlines()
    assert commands == [
        f"--tf train {input_json.resolve()} --skip-neighbor-stat",
        "--tf freeze -o student_native.pb",
    ]
    assert commands[0].count("--skip-neighbor-stat") == 1
    assert "--skip-neighbor-stat" not in commands[1]


@pytest.mark.parametrize(
    ("cli", "recipe_kind", "with_foundation"),
    [
        ("pt", "standard", False),
        ("pt", "standard", True),
        ("pt", "dpa4c", False),
        ("pt", "dpa4c", True),
    ],
)
def test_native_student_pt_and_dpa4c_train_keep_selector_and_finetune_order(
    tmp_path: Path, cli: str, recipe_kind: str, with_foundation: bool
) -> None:
    from workflow_backends.student.deepmd.native import NativeDeepMDStudentExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> commands.txt\n"
        "if [ \"$2\" = freeze ]; then printf model > student_native.pt2; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps({"model": {"type_map": ["H"], "descriptor": {"sel": 12}}, "training": {}}),
        encoding="utf-8",
    )
    foundation = None
    if with_foundation:
        foundation = tmp_path / "foundation.pt2"
        foundation.write_text("foundation", encoding="utf-8")
    context = OperationContext(
        role="student.deepmd.pt.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    NativeDeepMDStudentExecutor(train_context=context).train(
        input_json,
        tmp_path,
        cli,
        recipe_kind=recipe_kind,
        foundation_model=foundation,
        train_options={"skip_neighbor_stat": True},
    )

    selector = "--pt-expt" if recipe_kind == "dpa4c" else "--pt"
    foundation_part = "" if foundation is None else f" --finetune {foundation.resolve()}"
    freeze_extra = " --lower-kind graph" if recipe_kind == "dpa4c" else ""
    assert (tmp_path / "commands.txt").read_text(encoding="utf-8").splitlines() == [
        f"{selector} train {input_json.resolve()}{foundation_part} --skip-neighbor-stat",
        f"{selector} freeze -o student_native.pt2{freeze_extra}",
    ]


def test_native_student_dpa4c_no_sel_recipe_is_unchanged_and_gets_train_flag(
    tmp_path: Path,
) -> None:
    from workflow_backends.student.deepmd.native import NativeDeepMDStudentExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\n"
        "printf '%s\n' \"$*\" >> commands.txt\n"
        "if [ \"$2\" = freeze ]; then printf model > student_native.pt2; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            {
                "model": {
                    "type_map": ["O", "H"],
                    "descriptor": {"type": "dpa4c"},
                },
                "training": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    before = input_json.read_bytes()
    context = OperationContext(
        role="student.deepmd.pt.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    NativeDeepMDStudentExecutor(train_context=context).train(
        input_json,
        tmp_path,
        "pt",
        recipe_kind="dpa4c",
        train_options={"skip_neighbor_stat": True},
    )

    assert input_json.read_bytes() == before
    assert (tmp_path / "commands.txt").read_text(encoding="utf-8").splitlines() == [
        f"--pt-expt train {input_json.resolve()} --skip-neighbor-stat",
        "--pt-expt freeze -o student_native.pt2 --lower-kind graph",
    ]


def test_student_adapter_propagates_one_train_options_mapping_to_all_members_and_evidence(
    tmp_path: Path,
) -> None:
    from workflow_backends.student.deepmd.adapter import DeepMDStudentAdapter

    class RecordingExecutor:
        requires_native_dataset = False

        def __init__(self) -> None:
            self.options: list[object] = []

        def train(self, input_json, member_dir, cli, *, train_options=None, **kwargs):
            del kwargs
            self.options.append(train_options)
            terminal = Path(member_dir) / ("native.pb" if cli == "tf" else "native.pt2")
            terminal.write_text("model", encoding="utf-8")
            return terminal

    recipe = tmp_path / "student.json"
    recipe.write_text(
        json.dumps({"model": {"descriptor": {"type": "se_e2_a"}}, "training": {}}),
        encoding="utf-8",
    )
    dataset = tmp_path / "train.xyz"
    dataset.write_text("dataset", encoding="utf-8")
    executor = RecordingExecutor()

    artifacts = DeepMDStudentAdapter(
        run_json_path=tmp_path / "run.json",
        executor=executor,
    ).train(
        recipe=recipe,
        dataset_path=dataset,
        train_dir=tmp_path / "train",
        cli="pt",
        type_map=("H",),
        model="committee",
        members=2,
        policy="from_scratch",
        foundation_model=None,
        train_options={"skip_neighbor_stat": True},
    )

    assert executor.options == [
        {"skip_neighbor_stat": True},
        {"skip_neighbor_stat": True},
    ]
    assert all(
        artifact.evidence["train_options"] == {"skip_neighbor_stat": True}
        for artifact in artifacts
    )


def test_native_student_skip_rejects_automatic_sel_before_dp_without_mutation(
    tmp_path: Path,
) -> None:
    from workflow_backends.student.deepmd.native import NativeDeepMDStudentExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text("#!/bin/sh\nprintf launched > launched\n", encoding="utf-8")
    tool.chmod(0o755)
    input_json = tmp_path / "input.json"
    rendered = {
        "model": {
            "type_map": ["H"],
            "descriptor": {"sel": "auto:factor"},
        },
        "training": {},
    }
    input_json.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    before = input_json.read_bytes()
    context = OperationContext(
        role="student.deepmd.pt.train",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    with pytest.raises(ValueError, match=r"model\.descriptor\.sel='auto:factor'"):
        NativeDeepMDStudentExecutor(train_context=context).train(
            input_json,
            tmp_path,
            "pt",
            recipe_kind="dpa4c",
            train_options={"skip_neighbor_stat": True},
        )

    assert input_json.read_bytes() == before
    assert not (tmp_path / "launched").exists()


def test_native_student_compress_route_never_receives_train_skip_flag(
    tmp_path: Path,
) -> None:
    from workflow_backends.student.deepmd.native import NativeDeepMDStudentExecutor
    from workflow_runtime.lifecycle.environment import OperationContext

    tool = tmp_path / "dp"
    tool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> commands.txt\n"
        "if [ \"$2\" = compress ]; then printf compressed > compressed.pt2; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    source = tmp_path / "student.pt2"
    source.write_text("source", encoding="utf-8")
    training_script = tmp_path / "input.json"
    training_script.write_text("{}", encoding="utf-8")
    context = OperationContext(
        role="student.deepmd.pt.compress",
        profile_id="fake",
        source_scripts=(),
        executables={"dp": str(tool)},
        runtime={},
    )

    NativeDeepMDStudentExecutor(
        train_context=context,
        compress_context=context,
    ).compress(
        source,
        tmp_path / "compressed.pt2",
        "pt",
        recipe_kind="dpa4c",
        training_script=training_script,
        step=0.01,
        extrapolate=5,
        frequency=-1,
    )

    command = (tmp_path / "commands.txt").read_text(encoding="utf-8").strip()
    assert command.startswith("--pt-expt compress ")
    assert "--skip-neighbor-stat" not in command


def test_public_student_facade_reuses_one_option_for_each_round_train(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.config import load_run_config
    from workflow_core.lifecycle.composition import student_backend_binding
    from workflow_core.lifecycle.student_facade import LifecycleStudentFacade
    from workflow_backends.student.deepmd.adapter import DeepMDStudentBackendFactory
    from workflow_backends.student.role import StudentTrainRequest

    class RecordingExecutor:
        requires_native_dataset = False

        def __init__(self) -> None:
            self.options: list[object] = []

        def train(self, input_json, member_dir, cli, *, train_options=None, **kwargs):
            del input_json, kwargs
            self.options.append(train_options)
            terminal = Path(member_dir) / "native.pb"
            terminal.parent.mkdir(parents=True, exist_ok=True)
            terminal.write_text("model", encoding="utf-8")
            return terminal

    payload = _payload()
    payload["student"]["model"] = "committee"
    payload["student"]["members"] = 2
    payload["screen"]["lo"] = 0.0
    payload["screen"]["hi"] = 1.0
    payload["student"]["train_options"] = {"skip_neighbor_stat": True}
    run_path = _write_run(tmp_path, payload)
    config = load_run_config(run_path)
    executor = RecordingExecutor()
    selected_student = DeepMDStudentBackendFactory(executor, cli="tf").bind(
        student_backend_binding(config)
    )
    student = LifecycleStudentFacade(delegate=selected_student)
    for round_index in (0, 1):
        stage = config.root / "iter0" / f"r{round_index}" / "train.stage"
        dataset = stage / "dataset" / "train.xyz"
        dataset.parent.mkdir(parents=True)
        dataset.write_bytes((config.root / "data" / "student.xyz").read_bytes())
        student.train_and_publish(
            StudentTrainRequest(
                dataset_path=dataset,
                train_dir=stage,
                member_ids=("student_01", "student_02"),
            ),
            published_root=config.root / "iter0" / f"r{round_index}" / "train",
        )

    assert executor.options == [
        {"skip_neighbor_stat": True},
        {"skip_neighbor_stat": True},
        {"skip_neighbor_stat": True},
        {"skip_neighbor_stat": True},
    ]
