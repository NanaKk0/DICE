from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures
from workflow_artifacts.lifecycle.pair_guard import publish_pair_guard
from workflow_policies.selection import mean_descriptor_frames, p_space_sample, select_representatives
from workflow_policies.final_fp import select_final_fp_frames
from workflow_policies.pair_guard import (
    PairGuardDecision,
    PairGuardThresholds,
    evaluate_pair_guard,
)
from workflow_core.lifecycle.sampling_plan import SelectionPlan


def apply_pair_guard(
    observations: Sequence[Atoms],
    *,
    screen_dir: str | Path,
    config: Mapping[str, Any] | None,
) -> tuple[list[Atoms], tuple[PairGuardDecision, ...] | None]:
    """Evaluate configured guard evidence and retain pass frames in memory."""
    frames = list(observations)
    if config is None:
        return frames, None
    thresholds = PairGuardThresholds(
        minimum=float(config["min"]),
        pairs=dict(config.get("pairs", {})),
    )
    decisions = evaluate_pair_guard(frames, thresholds)
    publish_pair_guard(Path(screen_dir) / "pair_guard.npz", decisions, thresholds=thresholds)
    return [frame for frame, decision in zip(frames, decisions, strict=True) if decision.passed], decisions


def reserve_teacher_gmm_fp(
    observations: Sequence[Atoms],
    *,
    fp_pool: int,
    teacher_floor: float,
    teacher_trust: float,
    teacher_ceiling: float | None,
    entropy: str,
) -> tuple[list[Atoms], list[Atoms], dict[str, dict[str, bool]]]:
    """Construct independent teacher-FP and student-eligibility views."""
    if fp_pool == -1 and teacher_ceiling is None:
        raise ValueError("single-model GMM screen.fp_pool=-1 requires a teacher ceiling")
    if fp_pool == 0 or fp_pool < -1:
        raise ValueError("single-model GMM screen.fp_pool must be positive or conditional -1")
    floor, ceiling = _validated_p_candidate_band(
        floor=teacher_floor,
        ceiling=teacher_ceiling,
        role="teacher",
    )
    trust = float(teacher_trust)
    if not math.isfinite(trust) or not 0 <= trust <= 1:
        raise ValueError("single-model GMM teacher trust must be finite in [0, 1]")
    frames = _deduplicate(observations)
    flags = {
        str(frame.info["structure_id"]): {
            "teacher_floor_discard": float(frame.info["teacher_p"]) < floor,
            "teacher_candidate": (
                float(frame.info["teacher_p"]) >= floor
                and (ceiling is None or float(frame.info["teacher_p"]) < ceiling)
            ),
            "teacher_fp_candidate": False,
            "teacher_covered": (
                float(frame.info["teacher_p"]) >= floor
                and float(frame.info["teacher_p"]) > trust
            ),
            "student_eligible": False,
            "teacher_trust_reject": False,
        }
        for frame in frames
    }
    candidate_ids: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame for frame in frames
            if str(frame.info["system_id"]) == system_id
            and float(frame.info["teacher_p"]) >= floor
            and (ceiling is None or float(frame.info["teacher_p"]) < ceiling)
        ]
        if fp_pool == -1:
            chosen = [str(frame.info["structure_id"]) for frame in candidates]
        else:
            chosen = p_space_sample(
                [(str(frame.info["structure_id"]), float(frame.info["teacher_p"])) for frame in candidates],
                keep=min(fp_pool, len(candidates)),
                entropy=f"{entropy}/teacher-p/{system_id}",
                lower_bound=floor,
                upper_bound=ceiling,
            )
        candidate_ids.update(chosen)
    fp_candidates: list[Atoms] = []
    student_eligible: list[Atoms] = []
    for frame in frames:
        identity = str(frame.info["structure_id"])
        probability = float(frame.info["teacher_p"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("teacher p values must be finite in [0, 1]")
        if flags[identity]["teacher_floor_discard"]:
            continue
        if identity in candidate_ids:
            flags[identity]["teacher_fp_candidate"] = True
            fp_candidates.append(frame)
        if probability > trust:
            flags[identity]["student_eligible"] = True
            student_eligible.append(frame)
        else:
            flags[identity]["teacher_trust_reject"] = True
    return fp_candidates, student_eligible, flags


def select_student_gmm_train(
    frames: Sequence[Atoms],
    *,
    pool: int,
    keep: int,
    policy: str,
    entropy: str,
    feature_vectors: Mapping[str, np.ndarray] | None,
    student_floor: float,
    student_ceiling: float | None,
) -> tuple[list[Atoms], dict[str, dict[str, bool]]]:
    """Apply the student floor, conditional p-space pool, then one selector."""
    if pool == -1 and student_ceiling is None:
        raise ValueError("single-model GMM screen.pool=-1 requires a student ceiling")
    if pool == 0 or pool < -1:
        raise ValueError("single-model GMM screen.pool must be positive or conditional -1")
    floor, ceiling = _validated_p_candidate_band(
        floor=student_floor,
        ceiling=student_ceiling,
        role="student",
    )
    probabilities = {str(frame.info["structure_id"]): float(frame.info["student_p"]) for frame in frames}
    if len(probabilities) != len(frames):
        raise ValueError("student p-band identities must be unique")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("student p values must be finite in [0, 1]")
    selected: list[Atoms] = []
    flags = {
        str(frame.info["structure_id"]): {
            "student_floor_discard": probabilities[str(frame.info["structure_id"])] < floor,
            "student_candidate": (
                probabilities[str(frame.info["structure_id"])] >= floor
                and (ceiling is None or probabilities[str(frame.info["structure_id"])] < ceiling)
            ),
            "student_covered": (
                ceiling is not None
                and probabilities[str(frame.info["structure_id"])] >= ceiling
            ),
            "student_pool": False,
            "train_selected": False,
        }
        for frame in frames
    }
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame for frame in frames
            if str(frame.info["system_id"]) == system_id
            and float(frame.info["student_p"]) >= floor
            and (ceiling is None or float(frame.info["student_p"]) < ceiling)
        ]
        if pool == -1:
            pool_ids = [str(frame.info["structure_id"]) for frame in candidates]
        else:
            pool_ids = p_space_sample(
                [(str(frame.info["structure_id"]), float(frame.info["student_p"])) for frame in candidates],
                keep=min(pool, len(candidates)),
                entropy=f"{entropy}/student-p/{system_id}",
                lower_bound=floor,
                upper_bound=ceiling,
            )
        by_id = {str(frame.info["structure_id"]): frame for frame in candidates}
        pooled = [by_id[identity] for identity in sorted(pool_ids)]
        for identity in pool_ids:
            flags[identity]["student_pool"] = True
        identities = [str(frame.info["structure_id"]) for frame in pooled]
        if policy == "random":
            vectors = np.zeros((len(pooled), 1), dtype=float)
        else:
            if feature_vectors is None:
                raise ValueError(f"{policy} student GMM selection requires readout-input frame means")
            try:
                vectors = np.stack([np.asarray(feature_vectors[identity], dtype=float) for identity in identities])
            except KeyError as exc:
                raise ValueError(f"student GMM representative evidence is missing for {exc.args[0]!r}") from exc
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError("student GMM representative frame means must be finite frame x feature rows")
        chosen = set(select_representatives(
            identities, vectors, keep=min(keep, len(pooled)), policy=policy,
            entropy=f"{entropy}/selector/{system_id}",
        ))
        for frame in pooled:
            identity = str(frame.info["structure_id"])
            if identity in chosen:
                flags[identity]["train_selected"] = True
                selected.append(frame)
    return selected, flags


def fill_student_gmm_underfill(
    frames: Sequence[Atoms],
    *,
    normal_selected: Sequence[Atoms],
    flags: dict[str, dict[str, bool]],
    keep: int,
    enabled: bool,
    entropy: str,
) -> list[Atoms]:
    """Uniformly fill each system's normal GMM shortfall from covered frames."""
    selected_ids = {str(frame.info["structure_id"]) for frame in normal_selected}
    for identity, values in flags.items():
        values["safe_fill_eligible"] = bool(enabled and values["student_covered"])
        values["safe_fill_selected"] = False
    fill: list[Atoms] = []
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        system_frames = [
            frame for frame in frames if str(frame.info["system_id"]) == system_id
        ]
        shortfall = max(
            0,
            keep - sum(str(frame.info["structure_id"]) in selected_ids for frame in system_frames),
        )
        candidates = [
            frame for frame in system_frames
            if flags[str(frame.info["structure_id"])]["safe_fill_eligible"]
            and str(frame.info["structure_id"]) not in selected_ids
        ]
        identities = sorted(str(frame.info["structure_id"]) for frame in candidates)
        chosen = set(
            select_representatives(
                identities,
                np.zeros((len(identities), 1), dtype=float),
                keep=min(shortfall, len(identities)),
                policy="random",
                entropy=f"{entropy}/safe-fill/{system_id}",
            )
        )
        for frame in candidates:
            identity = str(frame.info["structure_id"])
            if identity in chosen:
                flags[identity]["safe_fill_selected"] = True
                fill.append(frame)
    return fill


def _validated_p_candidate_band(
    *,
    floor: float,
    ceiling: float | None,
    role: str,
) -> tuple[float, float | None]:
    lower = float(floor)
    if not math.isfinite(lower) or not 0 <= lower < 1:
        raise ValueError(f"single-model GMM {role} floor must be finite in [0, 1)")
    if ceiling is None:
        return lower, None
    upper = float(ceiling)
    if not math.isfinite(upper) or not lower < upper <= 1:
        raise ValueError(f"single-model GMM {role} ceiling must be finite in (floor, 1]")
    return lower, upper


def _score(frame: Atoms) -> float:
    return float(frame.info.get("acquisition", frame.info.get("force_uncertainty", frame.info.get("teacher_p", frame.info.get("p", 0.0)))))


def _deduplicate(frames: Sequence[Atoms]) -> list[Atoms]:
    selected: dict[str, Atoms] = {}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        previous = selected.get(identity)
        if previous is None:
            selected[identity] = frame
        elif exact_structure_hash(previous) != exact_structure_hash(frame):
            raise ValueError("conflicting geometry for duplicate structure_id before screening")
    return list(selected.values())


def _per_system_select(frames: Sequence[Atoms], *, pool: int, keep: int, policy: str, entropy: str, feature_vectors: Mapping[str, np.ndarray] | None = None) -> list[Atoms]:
    result: list[Atoms] = []
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [frame for frame in frames if str(frame.info["system_id"]) == system_id]
        candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
        candidates.sort(key=lambda frame: (-_score(frame), str(frame.info["structure_id"])))
        pooled = candidates if pool < 0 else candidates[:pool]
        identities = [str(frame.info["structure_id"]) for frame in pooled]
        if feature_vectors is None:
            vectors = np.asarray([[_score(frame)] for frame in pooled], dtype=float)
        else:
            try:
                raw_vectors = [np.asarray(feature_vectors[str(frame.info["structure_id"])]) for frame in pooled]
            except KeyError as exc:
                raise ValueError(f"representative descriptor evidence is missing for structure {exc.args[0]!r}") from exc
            vectors = (
                np.stack(raw_vectors).astype(float, copy=False)
                if raw_vectors and all(vector.ndim == 1 for vector in raw_vectors)
                else mean_descriptor_frames(raw_vectors)
            )
        chosen = set(select_representatives(identities, vectors, keep=min(keep, len(pooled)), policy=policy, entropy=f"{entropy}/{system_id}"))
        result.extend(frame for frame in pooled if str(frame.info["structure_id"]) in chosen)
    return result


def publish_screen(
    *,
    observations: list[Atoms],
    screen_dir: str | Path,
    teacher_model: str,
    student_model: str,
    keep: int,
    teacher_band: tuple[float, float] | None,
    label: Callable[[list[Atoms]], list[Atoms]],
    pool: int | None = None,
    policy: str = "FPS",
    entropy: str = "lifecycle/screen",
    student_band: tuple[float, float] | None = None,
    feature_vectors: Mapping[str, np.ndarray] | None = None,
    write_feature_sidecar: bool = True,
) -> tuple[Path, Path]:
    """Apply model-shape routing and atomically publish the two lifecycle screen files."""
    frames = _deduplicate(observations)
    effective_pool = keep if pool is None else pool
    if teacher_model == "single":
        raise ValueError(
            "single-model screens must use gmm-reference-tail revision 4 with gmm_reference/gmm_score roles"
        )
    if teacher_band is None:
        raise ValueError("committee teacher screen requires lo/hi")
    acquisition = frames
    if student_model == "committee":
        if student_band is None:
            raise ValueError("committee student screen requires its lo/hi acquisition band")
        lo, hi = student_band
        acquisition = [frame for frame in acquisition if lo <= float(frame.info.get("force_uncertainty", 0.0)) < hi]
    labeled = label(_per_system_select(acquisition, pool=effective_pool, keep=keep, policy=policy, entropy=entropy, feature_vectors=feature_vectors))
    lo, hi = teacher_band
    train = [frame for frame in labeled if float(frame.info.get("force_uncertainty", 0.0)) < lo]
    candidates = [frame for frame in labeled if lo <= float(frame.info.get("force_uncertainty", 0.0)) < hi]
    root = Path(screen_dir)
    train_path = write_extxyz_structures(root / "train.xyz", train, allow_empty=False)
    candidate_path = write_extxyz_structures(root / "fp_candidate.xyz", candidates, allow_empty=True)
    if feature_vectors is not None and write_feature_sidecar:
        _write_representative_features(root / "representative_features.npz", [*train, *candidates], feature_vectors)
    return train_path, candidate_path


def publish_fp(
    *,
    candidate_paths: list[str | Path],
    output: str | Path,
    keep: int,
    policy: str = "random",
    entropy: str = "lifecycle/fp",
    required_frame_mean_descriptor_schema: Mapping[str, object] | None = None,
    selection_plan: SelectionPlan | None = None,
) -> Path:
    """Apply later-lifecycle candidate precedence and per-system final selection."""
    if required_frame_mean_descriptor_schema is not None and policy not in {"DIRECT", "FPS"}:
        raise ValueError("required frame-mean evidence is valid only for DIRECT/FPS final FP selection")
    merged, candidate_batches = collect_final_fp_candidates(
        candidate_paths, selection_plan=selection_plan
    )
    feature_vectors: dict[str, np.ndarray] = {}
    feature_evidence_present = False
    for path, frames in zip(candidate_paths, candidate_batches, strict=True):
        identities = tuple(str(frame.info["structure_id"]) for frame in frames)
        feature_path = Path(path).parent / "representative_features.npz"
        if required_frame_mean_descriptor_schema is not None and frames:
            from workflow_artifacts.lifecycle.representative_features import load_frame_mean_representatives

            feature_vectors.update(
                load_frame_mean_representatives(
                    feature_path,
                    expected_identities=identities,
                    expected_descriptor_schema=required_frame_mean_descriptor_schema,
                )
            )
            feature_evidence_present = True
        elif required_frame_mean_descriptor_schema is None and feature_path.is_file():
            feature_vectors.update(_read_representative_features(feature_path))
            feature_evidence_present = True
    selected = select_final_fp_frames(
        list(merged.values()),
        keep=keep,
        policy=policy,
        entropy=entropy,
        feature_vectors=feature_vectors if feature_evidence_present else None,
        group_ids=(selection_plan.group_ids if selection_plan is not None else None),
    )
    target = write_extxyz_structures(output, selected, allow_empty=True)
    if selection_plan is not None:
        eligible_counts = {
            group_id: sum(
                str(frame.info["system_id"]) == group_id
                for frame in merged.values()
            )
            for group_id in selection_plan.group_ids
        }
        selected_counts = {
            group_id: sum(
                str(frame.info["system_id"]) == group_id for frame in selected
            )
            for group_id in selection_plan.group_ids
        }
        payload = {
            "schema": "dice.lifecycle.selection-group-fp-outcomes",
            "schema_revision": 1,
            "selection_groups": list(selection_plan.group_ids),
            "role_to_group": [list(item) for item in selection_plan.role_to_group],
            "policy": policy,
            "keep": int(keep),
            "rows": [
                {
                    "selection_group_id": group_id,
                    "eligible_count": eligible_counts[group_id],
                    "selected_count": selected_counts[group_id],
                    "underfill": int(keep) - selected_counts[group_id],
                }
                for group_id in selection_plan.group_ids
            ],
        }
        audit = target.parent / "selection_group_outcomes.json"
        audit.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return target


def collect_final_fp_candidates(
    candidate_paths: Sequence[str | Path],
    *,
    selection_plan: SelectionPlan | None = None,
) -> tuple[dict[str, Atoms], tuple[tuple[Atoms, ...], ...]]:
    """Load candidates with the established later-round identity precedence."""
    merged: dict[str, Atoms] = {}
    batches: list[tuple[Atoms, ...]] = []
    for path in candidate_paths:
        frames = tuple(read_extxyz_structures(path, allow_empty=True))
        if selection_plan is not None:
            role_to_group = dict(selection_plan.role_to_group)
            for frame in frames:
                role_id = frame.info.get("sampling_role_id")
                group_id = frame.info.get("selection_group_id")
                if (
                    role_id not in role_to_group
                    or role_to_group[role_id] != group_id
                    or frame.info.get("system_id") != group_id
                ):
                    raise ValueError(
                        "terminal FP candidate disagrees with the selection plan"
                    )
        for frame in frames:
            merged[str(frame.info["structure_id"])] = frame
        batches.append(frames)
    return merged, tuple(batches)


def _write_representative_features(path: Path, frames: Sequence[Atoms], feature_vectors: Mapping[str, np.ndarray]) -> None:
    unique: dict[str, np.ndarray] = {}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        if identity not in feature_vectors:
            raise ValueError(f"representative descriptor evidence is missing for selected structure {identity!r}")
        unique[identity] = np.asarray(feature_vectors[identity], dtype=np.float32)
    with path.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray(list(unique), dtype=str), **{f"frame_{index:06d}": value for index, value in enumerate(unique.values())})


def _read_representative_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        identities = [str(item) for item in payload["ids"].tolist()]
        if "means" in payload.files:
            means = np.asarray(payload["means"], dtype=np.float32)
            if means.ndim != 2 or means.shape[0] != len(identities) or not np.isfinite(means).all():
                raise ValueError("representative frame-mean sidecar is malformed")
            return {identity: means[index] for index, identity in enumerate(identities)}
        return {identity: np.asarray(payload[f"frame_{index:06d}"], dtype=np.float32) for index, identity in enumerate(identities)}
