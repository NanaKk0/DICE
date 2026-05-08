from pathlib import Path
import json

from workflow_runtime.logging.runtime_logging import ensure_step_log_dir
from workflow_core.structures.datasets import merge_xyz_files
from workflow_core.mace.backend import apply_backend_to_train_config, print_backend_banner, resolve_mace_backend
from workflow_core.mace.env import ensure_backend_environment, probe_mace_environment
from workflow_core.mace.model_check import (
    ensure_committee_backend_compatible,
    ensure_foundation_backend_compatible,
    write_mace_model_metadata,
)
from workflow_core.mace.finetune import run_committee_training
from workflow_core.common.io import read_json


def _collect_existing_committee_models(work_root: Path, teacher_num: int) -> list[Path]:
    models = []
    for teacher_index in range(1, teacher_num + 1):
        model_path = work_root / f"teacher_{teacher_index:02d}" / f"mace_teacher_{teacher_index:02d}.model"
        if model_path.exists():
            models.append(model_path.resolve())
    return models


def _find_foundation_model(paths) -> Path:
    user_foundation_dir = paths.user_foundation_models
    if user_foundation_dir.exists():
        user_candidates = sorted(user_foundation_dir.rglob("*stagetwo*.model"))
        if user_candidates:
            return user_candidates[0].resolve()

        user_candidates = sorted(user_foundation_dir.rglob("*.model"))
        if user_candidates:
            return user_candidates[0].resolve()

    foundation_dir = paths.mace_model_dir() / "checkpoints"
    ckpt = sorted(foundation_dir.glob("*stagetwo*.model"))
    if ckpt:
        return ckpt[0].resolve()

    ckpt = sorted(foundation_dir.glob("*.model"))
    if ckpt:
        return ckpt[0].resolve()

    legacy_candidates = sorted((paths.mace_model_dir()).glob("r*/checkpoints/*stagetwo*.model"))
    if legacy_candidates:
        return legacy_candidates[0].resolve()

    legacy_candidates = sorted((paths.mace_model_dir()).glob("r*/checkpoints/*.model"))
    if legacy_candidates:
        return legacy_candidates[0].resolve()

    raise RuntimeError(
        "No foundation model found in "
        f"{user_foundation_dir}, {foundation_dir}, or legacy mace_foundation_model/r*/checkpoints directories"
    )


def validate_finetune_config(train_cfg, config_path: Path) -> None:
    if bool(train_cfg.get("swa")):
        max_num_epochs = int(train_cfg["max_num_epochs"])
        start_swa = train_cfg.get("start_swa")
        if start_swa is not None and int(start_swa) >= max_num_epochs:
            raise ValueError(
                f"Invalid MACE finetune config {config_path}: start_swa={start_swa} must be smaller than "
                f"max_num_epochs={max_num_epochs}"
            )


def run_finetune(cfg, paths, r, context=None):
    context = context or {}
    print("\nPreparing MACE finetune dataset")

    config_path = Path(cfg.get("config", "configs/mace/committee_finetune.json")).resolve()

    train_cfg = read_json(config_path)
    validate_finetune_config(train_cfg, config_path)
    training_input = dict(context.get("training_input", {}) or {})
    physical_xyz_files = [Path(path).resolve() for path in training_input.get("physical_xyz_files", [])]
    rgps_xyz_files = [Path(path).resolve() for path in training_input.get("rgps_xyz_files", [])]
    historical_labeled_dft_xyz_files = [
        Path(path).resolve() for path in training_input.get("historical_labeled_dft_xyz_files", [])
    ]
    all_xyz = [Path(path).resolve() for path in training_input.get("all_xyz_files", [])]

    if physical_xyz_files:
        print("\nPhysical dataset:")
        for file_path in physical_xyz_files:
            print(" ", file_path)
    else:
        print("\nNo physical finetune dataset")

    if rgps_xyz_files:
        print("\nBase RGPS dataset:")
        for file_path in rgps_xyz_files:
            print(" ", file_path)
    else:
        print("\nBase RGPS dataset is disabled for this finetune run")

    if historical_labeled_dft_xyz_files:
        print("\nHistorical labeled DFT dataset:")
        for file_path in historical_labeled_dft_xyz_files:
            print(" ", file_path)

    if len(all_xyz) == 0:
        raise RuntimeError(
            "No dataset for mace finetune\n"
            "Please put xyz into:\n"
            "user_assets/mace_train\n"
            "or\n"
            "user_assets/mace_finetune"
        )

    merged = Path(context.get("merged_xyz_override", paths.committee_cycle_finetune_merged_xyz(r))).resolve()
    merge_xyz_files(all_xyz, merged)

    print("\nMerged dataset:")
    print(merged)

    backend = resolve_mace_backend(cfg.get("mace_backend"), context.get("mace_backend"), train_cfg.get("mace_backend"))
    env_report = ensure_backend_environment(backend, probe_mace_environment())
    print_backend_banner(env_report, backend)

    work_root = Path(context.get("work_root_override", paths.mace_committee_dir(r))).resolve()
    train_cfg["train_file"] = str(merged)
    train_cfg["work_root"] = str(work_root)

    foundation_model = _find_foundation_model(paths)
    ensure_foundation_backend_compatible(foundation_model, backend)
    train_cfg["foundation_model"] = str(foundation_model)
    train_cfg = apply_backend_to_train_config(train_cfg, backend)

    runtime = Path(context.get("runtime_json_override", paths.committee_cycle_finetune_runtime_json(r))).resolve()
    runtime.parent.mkdir(parents=True, exist_ok=True)

    with open(runtime, "w") as f:
        json.dump(train_cfg, f, indent=2)

    teacher_num = int(train_cfg["teacher_num"])
    reuse_existing_committee = bool(context.get("reuse_existing_committee", False))
    if reuse_existing_committee:
        existing_models = _collect_existing_committee_models(work_root, teacher_num)
        if len(existing_models) == teacher_num:
            ensure_committee_backend_compatible(work_root, backend, teacher_num=teacher_num)
            print("\nReusing existing MACE committee:")
            for model_path in existing_models:
                print(" ", model_path)
            return

    log_dir = ensure_step_log_dir(paths, r)

    run_committee_training(train_cfg, log_dir=log_dir)
    write_mace_model_metadata(
        work_root,
        backend=backend,
        model_role="committee_root",
        env_report=env_report,
        extra={"teacher_num": teacher_num, "foundation_model": str(foundation_model)},
    )
    for teacher_index in range(1, teacher_num + 1):
        write_mace_model_metadata(
            work_root / f"teacher_{teacher_index:02d}",
            backend=backend,
            model_role="committee_teacher",
            env_report=env_report,
            extra={"teacher_index": teacher_index, "foundation_model": str(foundation_model)},
        )


