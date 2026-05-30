from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_runtime.artifacts.checks import (
    student_subworkflow_complete,
    teacher_committee_finetune_complete,
)
from workflow_runtime.paths.path_manager import PathManager
from workflow_runtime.plan import PreflightCheck, PreflightCheckResult, PreflightPlan


def _load_json(path: Path) -> dict[str, Any]:
    return read_json(path)


def _resolve_workspace_path(paths: PathManager, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (paths.workspace_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_previous_round_dft_source(paths: PathManager, prev_round: int) -> Path | None:
    external_files = _discover_external_first_principles_xyz_files(paths, prev_round)
    if external_files:
        return external_files[0]
    runtime_labeled = paths.first_principles_labeled_xyz(prev_round)
    if runtime_labeled.exists() and runtime_labeled.is_file() and runtime_labeled.stat().st_size > 0:
        return runtime_labeled.resolve()
    return None


def _discover_external_first_principles_xyz_files(paths: PathManager, teacher_round: int) -> list[Path]:
    round_dir = paths.user_dft / f"r{int(teacher_round)}"
    if not round_dir.exists():
        return []
    preferred = round_dir / "labeled.xyz"
    if preferred.exists() and preferred.is_file() and preferred.stat().st_size > 0:
        return [preferred.resolve()]
    return [
        path.resolve()
        for path in sorted(round_dir.rglob("*.xyz"))
        if path.is_file() and path.stat().st_size > 0 and path.name.lower() != "failed.xyz"
    ]


def _resolve_previous_round_all_dft_candidate_sources(paths: PathManager, prev_round: int) -> list[Path]:
    root = paths.student_subworkflow_artifacts_dir(prev_round)
    return [
        path.resolve()
        for path in sorted(root.glob("student_round_*/stage_*_relabel/active_learning/dft.xyz"))
        if path.is_file() and path.stat().st_size > 0
    ]


def _resolve_preflight_config_context(paths: PathManager, wf: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "committee_cycle_config_path": None,
        "student_subworkflow_config_path": None,
    }
    committee_cycle_cfg_path = _resolve_workspace_path(paths, (wf.get("committee_cycle") or {}).get("config"))
    context["committee_cycle_config_path"] = str(committee_cycle_cfg_path) if committee_cycle_cfg_path else None
    if committee_cycle_cfg_path is None or not committee_cycle_cfg_path.exists():
        return context

    committee_cycle_cfg = _load_json(committee_cycle_cfg_path)
    student_cfg_path = _resolve_workspace_path(
        paths,
        (committee_cycle_cfg.get("student_subworkflow") or {}).get("config"),
    )
    context["student_subworkflow_config_path"] = str(student_cfg_path) if student_cfg_path else None
    return context


def _path_exists_check(path: Path, detail: str) -> PreflightCheckResult:
    return PreflightCheckResult(ok=path.exists(), detail=detail, path=path)


def _previous_final_student_model_path(paths: PathManager, prev_round: int) -> Path | None:
    root = paths.student_subworkflow_artifacts_dir(prev_round)
    patterns = ("*.model", "*.pt", "graph.pb", "nep.txt", "model.ckpt-*.index")
    candidates: list[Path] = []
    for train_dir in sorted(root.glob("student_round_*/stage_*_train")):
        for pattern in patterns:
            candidates.extend(path for path in train_dir.rglob(pattern) if path.is_file() and path.stat().st_size > 0)
    return candidates[-1].resolve() if candidates else None


def build_outer_round_preflight_plan(wf: dict[str, Any], paths: PathManager, round_id: int) -> PreflightPlan | None:
    if int(round_id) <= 0:
        return None

    prev_round = int(round_id) - 1
    config_context = _resolve_preflight_config_context(paths, wf)

    def previous_committee_cycle_artifacts() -> PreflightCheckResult:
        ok = teacher_committee_finetune_complete(
            paths.teacher_committee_finetune_artifacts_dir(prev_round),
        ) and student_subworkflow_complete(paths.student_subworkflow_artifacts_dir(prev_round))
        return PreflightCheckResult(
            ok=ok,
            detail="previous teacher_round committee_cycle canonical artifacts must be complete",
            path=paths.committee_cycle_artifacts_dir(prev_round),
        )

    def previous_student_subworkflow_artifacts() -> PreflightCheckResult:
        root = paths.student_subworkflow_artifacts_dir(prev_round)
        return PreflightCheckResult(
            ok=student_subworkflow_complete(root),
            detail="previous teacher_round student_subworkflow canonical artifacts must be complete",
            path=root,
        )

    def previous_final_model() -> PreflightCheckResult:
        final_model_path = _previous_final_student_model_path(paths, prev_round)
        return PreflightCheckResult(
            ok=final_model_path is not None,
            detail="previous teacher_round final student model artifact must exist",
            path=final_model_path,
        )

    def previous_round_dft_source() -> PreflightCheckResult:
        source = _resolve_previous_round_dft_source(paths, prev_round)
        return PreflightCheckResult(
            ok=source is not None,
            detail="teacher finetune historical first_principles source for previous teacher_round must exist",
            path=source,
        )

    def previous_round_all_dft_candidates() -> PreflightCheckResult:
        sources = _resolve_previous_round_all_dft_candidate_sources(paths, prev_round)
        first_candidate = sources[0] if sources else None
        return PreflightCheckResult(
            ok=bool(sources),
            detail="cross-teacher_round entry sampling source requires previous teacher_round first_principles candidate xyz files",
            path=first_candidate,
            assets={"previous_round_all_dft_candidate_source_count": str(len(sources))},
        )

    return PreflightPlan(
        round_id=int(round_id),
        run_name=paths.runtime_root.name,
        runtime_root=paths.runtime_root,
        pipeline=list(wf.get("pipeline", [])),
        previous_round=prev_round,
        config={
            **config_context,
            "use_cross_outer_entry_sampling": True,
            "cross_teacher_round0_source_mode": "previous_teacher_round_all_first_principles_candidates",
        },
        report_path=paths.workflow_round_preflight_json(round_id),
        checks=[
            PreflightCheck(
                key="previous_committee_cycle_artifacts",
                run=previous_committee_cycle_artifacts,
            ),
            PreflightCheck(key="previous_student_subworkflow_artifacts", run=previous_student_subworkflow_artifacts),
            PreflightCheck(key="previous_final_student_model", run=previous_final_model),
            PreflightCheck(key="previous_round_historical_dft_source", run=previous_round_dft_source),
            PreflightCheck(key="previous_round_all_dft_candidate_sources", run=previous_round_all_dft_candidates),
        ],
    )
