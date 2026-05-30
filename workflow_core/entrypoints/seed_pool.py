from __future__ import annotations


def _retired_formal_seed_pool_task(*args, **kwargs) -> None:
    del args, kwargs
    raise RuntimeError(
        "formal seed-pool task entrypoints are retired; use student_pipeline_template "
        "live seed selection and seed_pool_refresh stages"
    )


def run_init_seed_pool_task(*args, **kwargs) -> None:
    _retired_formal_seed_pool_task(*args, **kwargs)


def run_update_seed_pool_task(*args, **kwargs) -> None:
    _retired_formal_seed_pool_task(*args, **kwargs)


SEED_POOL_TASKS: dict[str, object] = {}
