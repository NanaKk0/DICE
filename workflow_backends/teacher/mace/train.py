import subprocess
from pathlib import Path
from typing import Any

from workflow_runtime.logging.runtime_logging import run_logged_command
import yaml

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
    "backend_name",
    "backend_native",
    "execution",
    "foundation_model_path",
    "freeze_interactions",
    "mace_backend",
    "mace_run_train_executable",
    "model_family",
    "python_executable",
    "rendered_yaml",
    "skip_completed_teachers",
    "source_path",
    "teacher_num",
    "teacher_seeds",
    "training_data",
    "type_map",
    "workflow",
    "work_root",
}


def map_key(key):
    return ARG_MAP.get(key, key)


def write_mace_rendered_yaml(train_cfg: dict[str, Any], path: str | Path) -> Path:
    rendered = _renderable_mace_config(train_cfg)
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    return resolved


def _renderable_mace_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in train_cfg.items():
        mapped = map_key(str(key))
        if mapped.startswith("_") or mapped in INTERNAL_ONLY_ARGS or value is None:
            continue
        rendered[mapped] = value
    return rendered


def build_mace_command(cfg, *, python_executable=None):
    del python_executable
    rendered_yaml = cfg.get("rendered_yaml")
    if not rendered_yaml:
        raise KeyError("MACE training requires rendered_yaml")
    executable = cfg.get("mace_run_train_executable")
    if not executable:
        execution = cfg.get("execution")
        if execution is None:
            raise KeyError("MACE training requires executable 'mace_run_train'")
        executable = execution.required_executable("mace_run_train")
    return [str(executable), "--config", str(rendered_yaml)]


def run_mace_training(cfg, env=None, log_file=None, quiet=None):
    cmd = build_mace_command(cfg)
    execution = cfg.get("execution")
    if execution is not None:
        cmd = execution.shell_command(cmd)
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


