import os
import subprocess
import sys
from pathlib import Path

from workflow_runtime.logging.runtime_logging import run_logged_command
from workflow_core.mace.backend import get_model_type_for_backend, normalize_mace_backend

OPTIONAL_TEACHER_ARGS = [
    "num_interactions",
    "num_channels",
    "max_L",
    "correlation",
    "r_max",
    "valid_batch_size",
    "start_swa",
    "scheduler_patience",
    "patience",
    "eval_interval",
    "error_table",
    "restart_latest",
    "plot",
    "plot_frequency",
    "weight_decay",
    "swa_lr",
    "swa_energy_weight",
    "swa_forces_weight",
    "swa_stress_weight",
]

OPTIONAL_TEACHER_FLAG_ARGS = {
    "ema",
    "swa",
    "restart_latest",
}


def _teacher_work_dir(cfg, teacher_index: int) -> Path:
    return Path(cfg["work_root"]) / f"teacher_{teacher_index:02d}"


def _teacher_model_path(cfg, teacher_index: int) -> Path:
    return _teacher_work_dir(cfg, teacher_index) / f"mace_teacher_{teacher_index:02d}.model"


def teacher_training_is_complete(cfg, teacher_index: int) -> bool:
    model_path = _teacher_model_path(cfg, teacher_index)
    return model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0


def build_teacher_command(cfg, teacher_index):
    teacher_name = f"mace_teacher_{teacher_index:02d}"
    backend = normalize_mace_backend(cfg.get("mace_backend", "standard"))

    work_dir = _teacher_work_dir(cfg, teacher_index)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parent
    wrapper_candidates = [
        base_dir / "run_train_wrapper.py",
        base_dir / "mace_run_train_wrapper.py",
    ]
    wrapper = next((path for path in wrapper_candidates if path.exists()), wrapper_candidates[0])
    cmd = [
        sys.executable,
        "-u",
        str(wrapper),
        "--name", teacher_name,
        "--model", get_model_type_for_backend(backend),
        "--foundation_model", cfg["foundation_model"],
        "--work_dir", str(work_dir),
        "--train_file", cfg["train_file"],
        "--valid_fraction", str(cfg["valid_fraction"]),
        "--default_dtype", str(cfg["default_dtype"]),
        "--loss", cfg["loss"],
        "--energy_key", cfg["energy_key"],
        "--forces_key", cfg["forces_key"],
        "--stress_key", cfg["stress_key"],
        "--energy_weight", str(cfg["energy_weight"]),
        "--forces_weight", str(cfg["forces_weight"]),
        "--stress_weight", str(cfg["stress_weight"]),
        "--E0s", cfg["E0s"],
        "--batch_size", str(cfg["batch_size"]),
        "--max_num_epochs", str(cfg["max_num_epochs"]),
        "--lr", str(cfg["lr"]),
        "--device", cfg["device"],
        "--seed", str(teacher_index),
    ]

    if cfg.get("freeze_interactions"):
        cmd.extend(["--freeze", str(cfg["freeze_interactions"])])

    if cfg.get("save_cpu"):
        cmd.append("--save_cpu")

    if cfg.get("ema", True):
        cmd.append("--ema")

    for key in OPTIONAL_TEACHER_ARGS:
        value = cfg.get(key)
        if value is None:
            continue
        if key in OPTIONAL_TEACHER_FLAG_ARGS:
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])

    return cmd


def run_committee_training(cfg, env=None, log_dir=None, quiet=None):
    teacher_num = cfg["teacher_num"]
    skip_completed_teachers = bool(cfg.get("skip_completed_teachers", True))

    if env is None:
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for i in range(1, teacher_num + 1):
        model_path = _teacher_model_path(cfg, i)
        if skip_completed_teachers and teacher_training_is_complete(cfg, i):
            print("=" * 90)
            print(f"Skipping completed MACE committee teacher {i:02d}")
            print(str(model_path))
            print("=" * 90)
            continue

        cmd = build_teacher_command(cfg, i)

        print("=" * 90)
        print(f"Running MACE committee teacher {i:02d}")
        print(" ".join(cmd))
        print("=" * 90)

        if log_dir is None:
            subprocess.run(cmd, check=True, env=env)
            continue

        log_path = Path(log_dir) / f"teacher_{i:02d}.log"
        run_logged_command(cmd, log_file=log_path, env=env, quiet=quiet)


