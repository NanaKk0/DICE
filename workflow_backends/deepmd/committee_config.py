from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalize_command(value: str | list[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError(f"Invalid command value: {value!r}")


def _normalize_dp_module_command(command: list[str]) -> list[str]:
    if not command:
        return command
    if command[0] == "dp":
        return [sys.executable, "-m", "deepmd", *command[1:]]
    return command


@dataclass(slots=True)
class DPCommitteeTrainConfig:
    committee_size: int
    valid_fraction: float
    split_seed: int
    split_mode: str
    model_seeds: list[int]
    dp_command: list[str]
    freeze_command: list[str]
    reuse_existing_split: bool
    reuse_existing_model: bool
    max_retries_per_member: int
    retry_seed_stride: int
    nan_watchdog_enabled: bool
    nan_terminate_timeout_seconds: float
    nan_watch_lcurve_fallback: bool
    nan_lcurve_poll_interval_seconds: float
    recommend_disp_freq: int | None
    recommend_save_freq_mode: str | None
    bootstrap_train_config: Path
    finetune_train_config: Path
    env: dict[str, str]

    def select_train_config(self, training_stage: str | None = None) -> Path:
        stage = str(training_stage or "bootstrap").strip().lower()
        if stage == "bootstrap":
            return self.bootstrap_train_config
        if stage == "finetune":
            return self.finetune_train_config
        return self.bootstrap_train_config


def load_committee_train_config(config_path: str | Path) -> DPCommitteeTrainConfig:
    config_path = Path(config_path).resolve()

    def _discover_workspace_root(path: Path) -> Path:
        for candidate in [path.parent, *path.parent.parents]:
            if (candidate / "workflow.py").exists() and (candidate / "configs").is_dir():
                return candidate
        return path.parent.parent

    workspace_root = _discover_workspace_root(config_path)

    with open(config_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    nan_watchdog_payload = payload.get("nan_watchdog", {})
    if nan_watchdog_payload is None:
        nan_watchdog_payload = {}

    committee_size = int(payload.get("committee_size", 4))
    if committee_size <= 0:
        raise ValueError("committee_size must be positive")

    valid_fraction = float(payload.get("valid_fraction", 0.1))
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")

    split_seed = int(payload.get("split_seed", 42))
    split_mode = str(payload.get("split_mode", "frame_within_system"))

    raw_model_seeds = payload.get("model_seeds")
    if raw_model_seeds is None:
        model_seeds = [split_seed + 10000 + idx for idx in range(committee_size)]
    else:
        model_seeds = [int(item) for item in raw_model_seeds]
        if len(model_seeds) < committee_size:
            raise ValueError(
                f"model_seeds has {len(model_seeds)} entries, but committee_size={committee_size}"
            )
        model_seeds = model_seeds[:committee_size]

    def _resolve_config_path(value: str | Path | None, default: str | Path) -> Path:
        resolved = Path(value if value is not None else default)
        if not resolved.is_absolute():
            candidate_from_root = (workspace_root / resolved).resolve()
            candidate_from_config = (config_path.parent / resolved).resolve()
            resolved = candidate_from_root if candidate_from_root.exists() else candidate_from_config
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Base DP train config not found: {resolved}")
        return resolved

    bootstrap_train_config = _resolve_config_path(
        payload.get("bootstrap_train_config", Path("configs") / "dp" / "committee_base.json"),
        Path("configs") / "dp" / "committee_base.json",
    )
    finetune_train_config = _resolve_config_path(
        payload.get("finetune_train_config", bootstrap_train_config),
        bootstrap_train_config,
    )
    raw_env = payload.get("env", {}) or {}
    if not isinstance(raw_env, dict):
        raise TypeError("dp_student_committee.env must be a mapping of environment variables")
    normalized_env = {str(key): str(value) for key, value in raw_env.items()}

    return DPCommitteeTrainConfig(
        committee_size=committee_size,
        valid_fraction=valid_fraction,
        split_seed=split_seed,
        split_mode=split_mode,
        model_seeds=model_seeds,
        dp_command=_normalize_dp_module_command(
            _normalize_command(payload.get("dp_command"), ["dp", "train"])
        ),
        freeze_command=_normalize_dp_module_command(
            _normalize_command(
                payload.get("freeze_command"),
                ["dp", "freeze", "-o", "graph.pb"],
            )
        ),
        reuse_existing_split=bool(payload.get("reuse_existing_split", True)),
        reuse_existing_model=bool(payload.get("reuse_existing_model", True)),
        max_retries_per_member=max(0, int(payload.get("max_retries_per_member", 2))),
        retry_seed_stride=max(1, int(payload.get("retry_seed_stride", 1000))),
        nan_watchdog_enabled=bool(nan_watchdog_payload.get("enabled", True)),
        nan_terminate_timeout_seconds=max(
            1.0,
            float(nan_watchdog_payload.get("terminate_timeout_seconds", 10.0)),
        ),
        nan_watch_lcurve_fallback=bool(nan_watchdog_payload.get("watch_lcurve_fallback", True)),
        nan_lcurve_poll_interval_seconds=max(
            0.5,
            float(nan_watchdog_payload.get("lcurve_poll_interval_seconds", 1.5)),
        ),
        recommend_disp_freq=(
            int(nan_watchdog_payload["recommend_disp_freq"])
            if nan_watchdog_payload.get("recommend_disp_freq") is not None
            else 100
        ),
        recommend_save_freq_mode=(
            str(nan_watchdog_payload.get("recommend_save_freq_mode", "same_as_disp"))
            if nan_watchdog_payload.get("recommend_save_freq_mode") is not None
            else "same_as_disp"
        ),
        bootstrap_train_config=bootstrap_train_config,
        finetune_train_config=finetune_train_config,
        env=normalized_env,
    )


def load_dp_train_template(train_config_path: str | Path) -> dict[str, Any]:
    train_config_path = Path(train_config_path).resolve()

    with open(train_config_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if "dp_config" not in payload:
        raise KeyError(f"DP train config missing 'dp_config': {train_config_path}")

    return payload["dp_config"]

