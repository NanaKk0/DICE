from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json, write_json
from workflow_core.dft.screening import rebuild_screened_dft_pool
from workflow_core.labeling.split_stats import summarize_label_split_by_system


def count_extxyz_frames(path: str | Path | None) -> int | None:
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.exists() or not path.is_file():
        return None
    try:
        count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "Properties=" in line:
                    count += 1
        return count
    except Exception:
        return None


def load_status(status_path: Path, outer_round: int, mace_teacher_dir: Path) -> dict[str, Any]:
    if status_path.exists():
        return read_json(status_path)
    return {
        "outer_round": int(outer_round),
        "mace_teacher_dir": str(mace_teacher_dir.resolve()),
        "current_dp_round": 0,
        "finished_rounds": [],
        "stopped": False,
        "stop_reason": None,
        "total_candidate_structures": 0,
        "total_selected_structures": 0,
        "total_labeled_structures": 0,
        "total_screened_dft_structures": 0,
        "total_discarded_structures": 0,
        "total_train_structures_added": 0,
        "final_dp_model_path": None,
    }


def write_status(status_path: Path, payload: dict[str, Any]) -> Path:
    payload["finished_rounds"] = sorted(set(int(item) for item in payload.get("finished_rounds", [])))
    return write_json(status_path, payload)


def base_round_manifest(outer_round: int, dp_round: int) -> dict[str, Any]:
    return {
        "outer_round": int(outer_round),
        "dp_round": int(dp_round),
        "completed": False,
        "steps": {},
    }


def manifest_step(manifest: dict[str, Any], step_key: str) -> dict[str, Any]:
    return manifest.get("steps", {}).get(step_key, {}) or {}


def step_completed(manifest: dict[str, Any], step_key: str) -> bool:
    return bool(manifest_step(manifest, step_key).get("completed"))


def completed_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).resolve()
    if path.exists():
        return path
    return None


def load_completed_sampling_summary_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    step_payload = manifest_step(manifest, "dp_sample_committee")
    if not step_payload.get("completed"):
        return None
    summary_path = completed_path(step_payload.get("sampling_summary_path"))
    if summary_path is None:
        summary_path = paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round) / "sampling_summary.json"
        if not summary_path.exists():
            return None
    try:
        return read_json(summary_path)
    except Exception:
        return None


def load_completed_label_info_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    step_payload = manifest_step(manifest, "label")
    if not step_payload.get("completed"):
        return None

    selected_xyz = completed_path(step_payload.get("selected_xyz"))
    if selected_xyz is None:
        selected_xyz = paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round) / "selected_candidates.xyz"
        if not selected_xyz.exists():
            return None

    label_dir = completed_path(step_payload.get("label_dir"))
    if label_dir is None:
        label_dir = paths.dp_subworkflow_round_label_dir(outer_round, dp_round)
    active_learning_dir = completed_path(step_payload.get("active_learning_dir"))
    if active_learning_dir is None:
        active_learning_dir = paths.dp_subworkflow_round_active_learning_dir(outer_round, dp_round)

    if not label_dir.exists() or not active_learning_dir.exists():
        return None

    if not any(
        candidate.exists()
        for candidate in (
            label_dir / "train.xyz",
            label_dir / "discard.xyz",
            active_learning_dir / "dft.xyz",
        )
    ):
        return None

    return {
        "selected_xyz": str(selected_xyz.resolve()),
        "label_dir": str(label_dir.resolve()),
        "active_learning_dir": str(active_learning_dir.resolve()),
        "metrics_attempt": int(step_payload.get("metrics_attempt", 0) or 0),
        "label_counts": {
            "input": count_extxyz_frames(selected_xyz) or 0,
            "train": count_extxyz_frames(label_dir / "train.xyz") or 0,
            "dft": count_extxyz_frames(active_learning_dir / "dft.xyz") or 0,
            "discard": count_extxyz_frames(label_dir / "discard.xyz") or 0,
        },
        "label_system_stats": summarize_label_split_by_system(
            label_dir / "train.xyz",
            active_learning_dir / "dft.xyz",
            label_dir / "discard.xyz",
        ),
    }


def load_completed_dft_screening_info_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    step_payload = manifest_step(manifest, "dft_screening")
    if not step_payload.get("completed"):
        return None
    summary_path = completed_path(step_payload.get("summary_path"))
    if summary_path is None:
        summary_path = paths.dp_subworkflow_round_dft_screening_summary_json(outer_round, dp_round)
        if not summary_path.exists():
            return None
    try:
        return read_json(summary_path)
    except Exception:
        return None


def load_completed_seed_pool_output_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> Path | None:
    step_payload = manifest_step(manifest, "build_seed_pool")
    if not step_payload.get("completed"):
        return None
    seed_pool_xyz = completed_path(step_payload.get("selected_seed_xyz"))
    if seed_pool_xyz is None:
        seed_pool_xyz = paths.dp_subworkflow_seed_pool_xyz(outer_round, dp_round)
        if not seed_pool_xyz.exists():
            return None
    return seed_pool_xyz.resolve()


def load_completed_seed_update_summary_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    step_payload = manifest_step(manifest, "update_seed_pool")
    if not step_payload.get("completed"):
        return None
    summary_path = completed_path(step_payload.get("summary_path"))
    if summary_path is None:
        summary_path = paths.seed_pool_update_summary_json(outer_round, dp_round)
        if not summary_path.exists():
            return None
    try:
        return read_json(summary_path)
    except Exception:
        return None


def rebuild_status_from_manifests(paths, outer_round: int, cfg: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    rebuilt = dict(status)
    rebuilt["finished_rounds"] = []
    rebuilt["total_candidate_structures"] = 0
    rebuilt["total_selected_structures"] = 0
    rebuilt["total_labeled_structures"] = 0
    rebuilt["total_screened_dft_structures"] = 0
    rebuilt["total_discarded_structures"] = 0
    rebuilt["total_train_structures_added"] = 0
    rebuilt["final_dp_model_path"] = None

    max_observed_round = 0
    for dp_round in range(0, int(cfg.get("max_dp_rounds", 0)) + 1):
        manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round)
        manifest = read_json(manifest_path) if manifest_path.exists() else base_round_manifest(outer_round, dp_round)
        manifest_exists = manifest_path.exists()

        if manifest_exists:
            max_observed_round = max(max_observed_round, dp_round)

        sampling_summary = load_completed_sampling_summary_from_manifest(manifest, paths, outer_round, dp_round)
        if sampling_summary is not None:
            rebuilt["total_candidate_structures"] += int(sampling_summary.get("candidate_count", 0) or 0) + int(
                sampling_summary.get("suspicious_candidate_count", 0) or 0
            )
            rebuilt["total_selected_structures"] += int(sampling_summary.get("selected_count", 0) or 0)

        label_info = load_completed_label_info_from_manifest(manifest, paths, outer_round, dp_round)
        if label_info is not None:
            label_counts = label_info["label_counts"]
            rebuilt["total_labeled_structures"] += int(label_counts.get("dft", 0) or 0)
            rebuilt["total_discarded_structures"] += int(label_counts.get("discard", 0) or 0)
            rebuilt["total_train_structures_added"] += int(label_counts.get("train", 0) or 0)

        screening_info = load_completed_dft_screening_info_from_manifest(manifest, paths, outer_round, dp_round)
        if screening_info is not None:
            rebuilt["total_screened_dft_structures"] += int(screening_info.get("screened_count", 0) or 0)

        train_step = manifest_step(manifest, "train_dp_committee")
        summary_path = completed_path(train_step.get("summary_path"))
        if train_step.get("completed") and summary_path is not None:
            rebuilt["final_dp_model_path"] = str(summary_path.resolve())
            if dp_round == 0 and dp_round not in rebuilt["finished_rounds"]:
                rebuilt["finished_rounds"].append(dp_round)

        if manifest.get("completed") and dp_round not in rebuilt["finished_rounds"]:
            rebuilt["finished_rounds"].append(dp_round)

    rebuilt["current_dp_round"] = int(max_observed_round)
    return rebuilt


def sync_screened_dft_pool_from_manifests(paths, outer_round: int, cfg: dict[str, Any]) -> Path | None:
    policy = dict(cfg.get("dft_screening_policy", {}) or {})
    if not policy.get("enabled", False):
        return None
    selected_xyz_files: list[str] = []
    for dp_round in range(1, int(cfg.get("max_dp_rounds", 0)) + 1):
        manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round)
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        screening_info = load_completed_dft_screening_info_from_manifest(manifest, paths, outer_round, dp_round)
        if screening_info is None:
            continue
        screened_xyz = screening_info.get("screened_xyz")
        if screened_xyz:
            selected_xyz_files.append(str(screened_xyz))
    return rebuild_screened_dft_pool(paths, outer_round, selected_xyz_files)
