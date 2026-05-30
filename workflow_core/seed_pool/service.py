from __future__ import annotations

ROLE_ANCHOR = "anchor"


def init_seed_pool(cfg, paths) -> str:
    return _retired_formal_seed_pool_task(cfg, paths)


def _retired_formal_seed_pool_task(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("formal seed-pool task runtime is retired; use pipeline template seed-pool stages")


def collect_seed_candidates_from_md(*args, **kwargs) -> str:
    return _retired_formal_seed_pool_task(*args, **kwargs)


def update_seed_pool(*args, **kwargs) -> str:
    return _retired_formal_seed_pool_task(*args, **kwargs)


def select_md_seeds_from_pool(*args, **kwargs) -> str:
    return _retired_formal_seed_pool_task(*args, **kwargs)


def refresh_seed_pool_after_training(*args, **kwargs) -> str:
    return _retired_formal_seed_pool_task(*args, **kwargs)


def record_seed_usage_after_sampling(*args, **kwargs) -> str:
    return _retired_formal_seed_pool_task(*args, **kwargs)
