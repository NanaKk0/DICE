import subprocess
import sys
from pathlib import Path

from workflow_runtime.logging.runtime_logging import run_logged_command
from workflow_core.mace.backend import apply_backend_to_train_config

ARG_MAP = {
    "use_ema": "ema",
    "use_swa": "swa",
    "scheduler_paience": "scheduler_patience",
    "plot_frequence": "plot_frequency"
}

FLAG_ARGS = {
    "ema",
    "swa",
    "restart_latest",
    "save_cpu",
    "keep_checkpoints",
    "save_all_checkpoints",
    "wandb"
}

INTERNAL_ONLY_ARGS = {
    "mace_backend",
}


def map_key(key):
    return ARG_MAP.get(key, key)


def build_mace_command(cfg):
    cfg = apply_backend_to_train_config(cfg, cfg.get("mace_backend", "standard"))
    base_dir = Path(__file__).resolve().parent
    wrapper_candidates = [
        base_dir / "run_train_wrapper.py",
        base_dir / "mace_run_train_wrapper.py",
    ]
    wrapper = next((path for path in wrapper_candidates if path.exists()), wrapper_candidates[0])
    cmd = [sys.executable, "-u", str(wrapper)]

    for key, value in cfg.items():
        key = map_key(key)
        if key in INTERNAL_ONLY_ARGS:
            continue
        arg = f"--{key}"

        if value is None:
            continue

        if key in FLAG_ARGS:
            if value:
                cmd.append(arg)
            continue

        cmd.append(arg)
        cmd.append(str(value))

    return cmd


def run_mace_training(cfg, env=None, log_file=None, quiet=None):
    cmd = build_mace_command(cfg)
    if env is not None:
        env.setdefault("PYTHONUNBUFFERED", "1")

    print("=" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80)

    if log_file is None:
        subprocess.run(cmd, check=True, env=env)
        return

    run_logged_command(cmd, log_file=log_file, env=env, quiet=quiet)


