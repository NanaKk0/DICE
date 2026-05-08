from __future__ import annotations

from pathlib import Path


def resolve_config_path(cfg, default_path: str) -> Path:
    return Path(cfg.get("config", default_path)).resolve()

