import json
from pathlib import Path

from ase.io import read

from workflow_backends.deepmd.committee_config import load_committee_train_config, load_dp_train_template
from workflow_runtime.logging.runtime_logging import quiet_known_third_party_noise
from workflow_core.structures.datasets import collect_xyz_files, merge_xyz_files
from workflow_core.labeling.label import run_labeling
from workflow_core.common.io import append_candidate_xyz, read_json
from workflow_core.labeling.export import export_label_results
from workflow_core.mace.backend import print_backend_banner, resolve_mace_backend
from workflow_core.mace.env import ensure_backend_environment, probe_mace_environment
from workflow_core.mace.model_check import infer_mace_backend


def build_label_core_config(label_cfg, teacher_dir):
    core_cfg = dict(label_cfg)
    core_cfg["teacher_dir"] = str(teacher_dir)

    return core_cfg


def build_label_export_config(paths, round_id, element_order):
    return {
        "labeled_output_dir": str(paths.dp_data_dir(round_id)),
        "active_learning_output_dir": str(paths.active_learning_data_dir(round_id)),
        "element_order": element_order
    }


def resolve_label_train_ratio(label_cfg, context):
    valid_fraction_override = context.get("valid_fraction_override")
    if valid_fraction_override is not None:
        valid_fraction = float(valid_fraction_override)
        if not 0.0 < valid_fraction < 1.0:
            raise ValueError(f"valid_fraction_override must be between 0 and 1, got {valid_fraction_override!r}")
        return 1.0 - valid_fraction

    train_committee_config = context.get("train_committee_config_override")
    if train_committee_config:
        cfg = load_committee_train_config(train_committee_config)
        return 1.0 - float(cfg.valid_fraction)

    return float(label_cfg.get("train_ratio", 0.9))


def resolve_label_element_order(label_cfg, context):
    override = context.get("element_order_override")
    if override is not None:
        return [str(item) for item in override]

    train_committee_config = context.get("train_committee_config_override")
    if train_committee_config:
        train_cfg = load_committee_train_config(train_committee_config)
        training_stage = str(context.get("training_stage_override", "bootstrap")).strip().lower()
        selected_train_config = train_cfg.select_train_config(training_stage)
        train_template = load_dp_train_template(selected_train_config)
        type_map = train_template.get("model", {}).get("type_map")
        if isinstance(type_map, list) and type_map:
            return [str(item) for item in type_map]

    legacy_element_order = label_cfg.get("element_order")
    if isinstance(legacy_element_order, list) and legacy_element_order:
        return [str(item) for item in legacy_element_order]

    raise KeyError("Unable to resolve label export element order from context or DP train config")


def export_cfg_preview_label_dir(context, paths, round_id):
    override = context.get("dp_round_dir_override")
    if override is not None:
        return Path(override).resolve()
    return paths.dp_data_dir(round_id)


def export_cfg_preview_al_dir(context, paths, round_id):
    override = context.get("active_learning_round_dir_override")
    if override is not None:
        return Path(override).resolve()
    return paths.active_learning_data_dir(round_id)


def run_label(cfg, paths, round_id, context=None):
    print("\nStarting LABEL step")
    context = context or {}

    config_path = Path(cfg["config"]).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Label config not found: {config_path}")

    label_cfg = read_json(config_path)

    structure_file = Path(context.get("structure_file_override", paths.selected_xyz(round_id))).resolve()

    teacher_dir_override = context.get("teacher_dir_override")
    if teacher_dir_override is not None:
        teacher_dir = Path(teacher_dir_override).resolve()
    else:
        teacher_round = int(context.get("teacher_round_override", round_id))
        teacher_dir = paths.mace_committee_dir(teacher_round)

    include_user_dp_train = bool(context.get("include_user_dp_train", True))
    user_xyz = []
    if include_user_dp_train:
        user_dir = paths.user_dp_train
        user_xyz = collect_xyz_files(user_dir)

    if include_user_dp_train and user_xyz:
        print("\nUser dataset:")
        for file_path in user_xyz:
            print(" ", file_path)
    elif include_user_dp_train:
        print("\nNo user DP-train dataset")

    all_xyz = list(user_xyz)

    if not structure_file.exists():
        print(f"Selecting output not found: {structure_file}")
    else:
        all_xyz.append(structure_file)

    if len(all_xyz) == 0:
        raise RuntimeError(
            "No dataset for committee labeling\n"
            "Please put xyz into:\n"
            "data/dp_subworkflow/rX/dp_rY/label\n"
            "or\n"
            "user_assets/dp_train"
        )

    if not teacher_dir.exists():
        raise FileNotFoundError(
            f"Teacher models not found: {teacher_dir}"
        )

    merged = paths.dp_data_dir(round_id) / "train_merged.xyz"
    merged_override = context.get("merged_xyz_override")
    if merged_override is not None:
        merged = Path(merged_override).resolve()
    merge_xyz_files(all_xyz, merged)

    print("\nMerged dataset:")
    print(merged)

    print("\nLabel paths")
    print("----------------")
    print("input_structure_file:", merged)
    print("teacher_dir:", teacher_dir)
    print("labeled_output_dir:", export_cfg_preview_label_dir(context, paths, round_id))
    print("active_learning_output_dir:", export_cfg_preview_al_dir(context, paths, round_id))

    atoms_list = read(str(merged), ":")

    core_cfg = build_label_core_config(label_cfg, teacher_dir)
    core_cfg["train_ratio"] = resolve_label_train_ratio(label_cfg, context)
    backend = resolve_mace_backend(
        context.get("mace_backend"),
        label_cfg.get("mace_backend"),
        infer_mace_backend(teacher_dir, fallback="standard"),
    )
    env_report = ensure_backend_environment(backend, probe_mace_environment())
    print_backend_banner(env_report, backend)
    core_cfg["mace_backend"] = backend

    with quiet_known_third_party_noise():
        outputs = run_labeling(atoms_list, core_cfg)

    export_cfg = build_label_export_config(
        paths,
        round_id,
        resolve_label_element_order(label_cfg, context)
    )
    dp_round_dir_override = context.get("dp_round_dir_override")
    if dp_round_dir_override is not None:
        export_cfg["labeled_output_dir"] = str(Path(dp_round_dir_override).resolve())

    active_learning_round_dir_override = context.get("active_learning_round_dir_override")
    if active_learning_round_dir_override is not None:
        export_cfg["active_learning_output_dir"] = str(Path(active_learning_round_dir_override).resolve())

    export_label_results(outputs, export_cfg)

    append_active_learning_xyz = context.get("append_active_learning_xyz")
    if append_active_learning_xyz is not None:
        append_candidate_xyz(Path(append_active_learning_xyz).resolve(), list(outputs.dft_structures))

    print("\nLABEL step finished successfully")


