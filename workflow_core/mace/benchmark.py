from __future__ import annotations

import json
from pathlib import Path

from workflow_core.mace.backend import print_backend_banner, resolve_mace_backend
from workflow_core.mace.env import ensure_backend_environment, probe_mace_environment
from workflow_core.mace.model_check import infer_mace_backend
from workflow_core.common.io import read_json


def run_task(cfg, paths, round_id: int) -> None:
    config_path = Path(cfg["config"]).resolve() if cfg.get("config") else None
    benchmark_cfg = {}
    if config_path is not None:
        benchmark_cfg = read_json(config_path)

    model_root = cfg.get("model_root")
    if model_root is None:
        model_root = paths.mace_model_dir()

    backend = resolve_mace_backend(
        cfg.get("mace_backend"),
        benchmark_cfg.get("mace_backend"),
        infer_mace_backend(model_root, fallback="standard"),
    )
    env_report = ensure_backend_environment(backend, probe_mace_environment())
    print_backend_banner(env_report, backend)
    print(f"[INFO] Benchmark backend resolved to {backend} for model root: {Path(model_root).resolve()}")

