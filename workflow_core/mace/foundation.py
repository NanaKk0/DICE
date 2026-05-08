from pathlib import Path
import json
import os

from workflow_runtime.logging.runtime_logging import ensure_step_log_file
from workflow_core.structures.datasets import collect_xyz_files, merge_xyz_files
from workflow_core.mace.backend import apply_backend_to_train_config, print_backend_banner, resolve_mace_backend
from workflow_core.mace.env import ensure_backend_environment, probe_mace_environment
from workflow_core.mace.model_check import write_mace_model_metadata
from workflow_core.mace.train import run_mace_training
from workflow_core.common.io import read_json


def validate_train_config(train_cfg, config_path: Path) -> None:
    if bool(train_cfg.get("swa")):
        max_num_epochs = int(train_cfg["max_num_epochs"])
        start_swa = train_cfg.get("start_swa")
        if start_swa is not None and int(start_swa) >= max_num_epochs:
            raise ValueError(
                f"Invalid MACE config {config_path}: start_swa={start_swa} must be smaller than "
                f"max_num_epochs={max_num_epochs}"
            )


def run_train_mace(cfg, paths, r):
    print("\nPreparing MACE foundation training dataset")

    workflow_dataset_dir = paths.mace_train_data_dir(r)
    workflow_xyz = collect_xyz_files(workflow_dataset_dir)

    if workflow_xyz:
        print("\nWorkflow dataset:")
        for file_path in workflow_xyz:
            print("  ", file_path)
    else:
        print("\nNo workflow dataset found.")

    user_dataset_dir = paths.user_mace_train
    user_xyz = collect_xyz_files(user_dataset_dir)

    if user_xyz:
        print("\nUser dataset:")
        for file_path in user_xyz:
            print("  ", file_path)
    else:
        print("\nNo user dataset found.")

    all_xyz = workflow_xyz + user_xyz

    if len(all_xyz) == 0:
        raise RuntimeError(
            "No dataset available for MACE foundation training.\n"
            "Please provide dataset in either:\n"
            "  data/mace_train_data/foundation/train/\n"
            "or\n"
            "  user_assets/mace_train/"
        )

    merged_file = paths.mace_foundation_data_dir() / "train_merged.xyz"
    merge_xyz_files(all_xyz, merged_file)

    print("\nMerged dataset written to:")
    print(merged_file)

    config_path = Path(cfg.get("config", "configs/mace/foundation_train.json")).resolve()

    train_cfg = read_json(config_path)
    validate_train_config(train_cfg, config_path)

    backend = resolve_mace_backend(cfg.get("mace_backend"), train_cfg.get("mace_backend"))
    env_report = ensure_backend_environment(backend, probe_mace_environment())
    print_backend_banner(env_report, backend)

    train_cfg["train_file"] = str(merged_file)
    train_cfg["test_file"] = str(merged_file)
    train_cfg["work_dir"] = str(paths.mace_model_dir())
    train_cfg = apply_backend_to_train_config(train_cfg, backend)

    runtime_config = paths.mace_foundation_data_dir() / "train_mace_runtime.json"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)

    with open(runtime_config, "w") as f:
        json.dump(train_cfg, f, indent=2)

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    log_file = ensure_step_log_file(paths, "train_mace", r)

    run_mace_training(train_cfg, env=env, log_file=log_file)
    write_mace_model_metadata(
        paths.mace_model_dir(),
        backend=backend,
        model_role="foundation",
        env_report=env_report,
        extra={"runtime_config": str(runtime_config.resolve())},
    )


