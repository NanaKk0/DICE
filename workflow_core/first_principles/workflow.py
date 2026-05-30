from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_core.first_principles.collect import collect_first_principles_results
from workflow_core.first_principles.prepare import prepare_first_principles_jobs
from workflow_core.first_principles.run import run_first_principles_jobs
from workflow_artifacts.first_principles.handoff import first_principles_handoff_input_path


def load_dft_workflow_config(config_path: str | Path) -> dict[str, Any]:
    return dict(read_json(Path(config_path).resolve()) or {})


load_first_principles_workflow_config = load_dft_workflow_config


def _load_optional_workflow_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path in (None, ""):
        return {}
    return load_first_principles_workflow_config(config_path)


def build_dft_runtime_cfg(base_cfg: dict[str, Any], paths, outer_round: int) -> dict[str, Any]:
    cfg = dict(base_cfg or {})
    root = paths.first_principles_artifacts_dir(outer_round)
    prepare_dir = paths.first_principles_substep_artifacts_dir(outer_round, "prepare")
    run_dir = paths.first_principles_substep_artifacts_dir(outer_round, "run")
    collect_dir = paths.first_principles_substep_artifacts_dir(outer_round, "collect")
    jobid = str(cfg.get("jobid") or f"teacher_round_{int(outer_round):03d}")

    cfg["input_xyz"] = str(first_principles_handoff_input_path(paths, outer_round))
    cfg["task_root"] = str(paths.first_principles_tasks_dir(outer_round))
    cfg["output_root"] = str(prepare_dir)
    cfg["jobid"] = jobid
    cfg.setdefault("task_map_path", str(prepare_dir / "task_map.json"))
    cfg.setdefault("record_path", str(prepare_dir / "record.dft.json"))
    cfg.setdefault("failed_prepare_path", str(prepare_dir / "failed_prepare.xyz"))
    cfg.setdefault("run_output_root", str(run_dir))
    cfg.setdefault("collect_output_root", str(collect_dir))
    cfg.setdefault("labeled_xyz", str(paths.first_principles_labeled_xyz(outer_round)))
    cfg.setdefault("failed_xyz", str(paths.first_principles_failed_xyz(outer_round)))
    cfg.setdefault("first_principles_root", str(root))
    return cfg


build_first_principles_runtime_cfg = build_dft_runtime_cfg


def run_dft_workflow(config_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    del config_path, paths, outer_round, metrics_logger
    raise RuntimeError(
        "top-level step id 'dft' is retired from target-mode; use 'first_principles' "
        "and the canonical handoff input instead"
    )


def run_first_principles_workflow(
    config_path: str | Path | None,
    paths,
    outer_round: int,
    metrics_logger=None,
) -> str:
    del metrics_logger
    base_cfg = _load_optional_workflow_config(config_path)
    cfg = build_first_principles_runtime_cfg(base_cfg, paths, outer_round)
    input_xyz = Path(cfg["input_xyz"]).resolve()
    if not input_xyz.exists() or not input_xyz.is_file():
        raise FileNotFoundError(
            "Missing canonical first_principles handoff input: "
            f"{input_xyz}. Run teacher_dft_candidate_filtering to publish "
            "artifacts/teacher_round_<XXX>/first_principles/prepare/source/"
            "input_for_first_principles.xyz before the top-level first_principles step."
        )

    prepare_result = prepare_first_principles_jobs(cfg)
    run_cfg = {
        **cfg,
        "output_root": str(Path(cfg["run_output_root"]).resolve()),
    }
    run_first_principles_jobs(run_cfg)
    collect_cfg = {
        **cfg,
        "output_root": str(Path(cfg["collect_output_root"]).resolve()),
        "task_map_path": prepare_result.get("task_map_path", cfg.get("task_map_path")),
    }
    collect_first_principles_results(collect_cfg)
    return str(paths.first_principles_artifacts_dir(outer_round).resolve())
