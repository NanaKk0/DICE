from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from workflow_backends.first_principles.cp2k.collect import parse_cp2k_task
from workflow_backends.first_principles.cp2k.defaults import (
    DEFAULT_DFT_STRUCTURE_NAME,
    DEFAULT_DFT_TASK_DIR_PREFIX,
)
from workflow_backends.first_principles.cp2k.runtime import resolve_cp2k_runtime
from workflow_backends.first_principles.cp2k.status import cp2k_output_looks_successful
from workflow_backends.first_principles.cp2k.submit import submit_module


class CP2KFirstPrinciplesBackend:
    """CP2K implementation of the first-principles backend contract."""

    name = "cp2k"
    structure_name = DEFAULT_DFT_STRUCTURE_NAME
    task_dir_prefix = DEFAULT_DFT_TASK_DIR_PREFIX

    def resolve_runtime(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return resolve_cp2k_runtime(cfg)

    def prepare_task(self, *, task_dir: Path, structure_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        del task_dir, structure_path, cfg
        return {"input_rendering": "deferred_to_run_tasks"}

    def run_tasks(self, cfg: dict[str, Any]) -> dict[str, Any]:
        submit_cfg = dict(cfg)
        submit_cfg["input_dir"] = str(Path(cfg["task_root"]).resolve())
        submit_cfg["output_dir"] = str(Path(cfg["output_root"]).resolve())
        return submit_module(submit_cfg)

    def task_is_successful(self, task_dir: Path, cfg: dict[str, Any]) -> bool:
        return cp2k_output_looks_successful(task_dir, cfg)

    def parse_labeled_task(self, task_dir: Path, cfg: dict[str, Any]) -> Atoms:
        return parse_cp2k_task(task_dir, cfg)
