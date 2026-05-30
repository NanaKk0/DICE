from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from workflow_backends.teacher import get_teacher_backend
from workflow_backends.teacher.base import TeacherRelabelRequest
from workflow_core.common.io import append_candidate_xyz
from workflow_core.labeling.export import export_label_results
from workflow_core.labeling.grouping import build_grouped_train_val, sort_atoms
from workflow_core.labeling.request import TeacherLabelingRequest
from workflow_core.labeling.results import LabelingOutputs
from workflow_core.labeling.routing import TeacherRoutingPolicy, route_teacher_labeled_structures
from workflow_core.labeling.split_stats import summarize_label_split_by_system
from workflow_core.structures.datasets import collect_xyz_files, merge_xyz_files
from workflow_runtime.logging.runtime_logging import quiet_known_third_party_noise


@dataclass(slots=True)
class TeacherLabelingRunResult:
    outputs: LabelingOutputs
    summary: dict[str, Any]


def _uncertainty_summary(uncertainties) -> dict[str, float | None]:
    return {
        "min": float(np.min(uncertainties)) if len(uncertainties) else None,
        "max": float(np.max(uncertainties)) if len(uncertainties) else None,
        "mean": float(np.mean(uncertainties)) if len(uncertainties) else None,
        "median": float(np.median(uncertainties)) if len(uncertainties) else None,
    }


def _print_uncertainty_histogram(uncertainties) -> None:
    if len(uncertainties) == 0:
        print("\nForce uncertainty statistics")
        print("no labeled structures")
        return
    print("\nForce uncertainty statistics")
    print("min:", uncertainties.min())
    print("max:", uncertainties.max())
    print("mean:", uncertainties.mean())
    print("median:", np.median(uncertainties))

    bins = np.linspace(uncertainties.min(), uncertainties.max(), 10)
    hist, edges = np.histogram(uncertainties, bins=bins)

    print("\nHistogram")

    for i in range(len(hist)):
        print(f"{edges[i]:.4f} - {edges[i+1]:.4f} : {hist[i]}")


def _merge_labeling_inputs(request: TeacherLabelingRequest, paths, round_id: int) -> Path:
    structure_file = Path(request.labeling_input.structure_file).resolve()
    include_user_student_train = bool(request.labeling_input.include_user_student_train)
    user_xyz = []
    if include_user_student_train:
        user_xyz = collect_xyz_files(paths.user_dp_train)

    if include_user_student_train and user_xyz:
        print("\nUser dataset:")
        for file_path in user_xyz:
            print(" ", file_path)
    elif include_user_student_train:
        print("\nNo user student-train dataset")

    all_xyz = list(user_xyz)

    if not structure_file.exists():
        print(f"Selecting output not found: {structure_file}")
    else:
        all_xyz.append(structure_file)

    if len(all_xyz) == 0:
        raise RuntimeError(
            "No dataset for committee labeling\n"
            "Please provide a stage-local input artifact or user_assets/dp_train"
        )

    if request.labeling_input.merged_xyz is None:
        raise ValueError("target-mode labeling requires request.labeling_input.merged_xyz")
    merged = Path(request.labeling_input.merged_xyz).resolve()
    merge_xyz_files(all_xyz, merged)
    return merged


def run_teacher_labeling(
    *,
    request: TeacherLabelingRequest,
    label_cfg: dict[str, Any],
    paths,
    round_id: int,
    element_order: list[str],
    train_ratio: float,
    write_deepmd_dataset: bool,
    routing_policy: TeacherRoutingPolicy,
    teacher_backend=None,
    backend_log_dir: str | Path | None = None,
    context: dict[str, Any] | None = None,
) -> TeacherLabelingRunResult:
    teacher_dir = Path(request.teacher_committee.committee_dir).resolve()
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Teacher models not found: {teacher_dir}")

    merged = _merge_labeling_inputs(request, paths, round_id)

    print("\nMerged dataset:")
    print(merged)

    print("\nLabel paths")
    print("----------------")
    print("input_structure_file:", merged)
    print("teacher_dir:", teacher_dir)
    print("labeled_output_dir:", Path(request.output_targets.labeled_output_dir).resolve())
    print("active_learning_output_dir:", Path(request.output_targets.active_learning_output_dir).resolve())

    _reject_top_level_backend_execution_fields(label_cfg)
    backend_options = dict(label_cfg.get("teacher_backend_options", {}) or {})
    backend_options.setdefault("teacher_backend", request.teacher_committee.backend)
    inherited_teacher_num = getattr(request.teacher_committee, "teacher_num", None)
    device = str(backend_options.get("device", "cpu"))
    default_dtype = str(backend_options.get("default_dtype", "float32"))
    request_context = dict(context or {})
    execution_context = request_context.get("execution_context")

    relabel_request = TeacherRelabelRequest(
        committee_dir=teacher_dir,
        teacher_num=_resolve_relabel_teacher_num(
            inherited_teacher_num,
            backend_options.get("teacher_num"),
            label_cfg.get("teacher_num"),
        ),
        device=device,
        default_dtype=default_dtype,
        backend_options=backend_options,
        backend_log_dir=Path(backend_log_dir).resolve() if backend_log_dir is not None else None,
        execution_context=execution_context if isinstance(execution_context, dict) else None,
        context=request_context,
    )
    backend = teacher_backend or get_teacher_backend(request.teacher_committee.backend)
    raw_atoms_list = list(read(str(merged), ":"))
    preserve_atom_order = getattr(backend, "preserve_relabel_atom_order", lambda request: False)
    if preserve_atom_order(relabel_request):
        atoms_list = raw_atoms_list
    else:
        atoms_list = [sort_atoms(atoms) for atoms in raw_atoms_list]
    print("Frames:", len(atoms_list))
    _validate_compat_teacher_num(backend, teacher_dir, relabel_request.teacher_num)
    with quiet_known_third_party_noise():
        relabel_result = backend.relabel_structures(atoms_list, relabel_request)

    _print_uncertainty_histogram(relabel_result.uncertainties)

    label_core_cfg = dict(label_cfg)
    label_core_cfg["train_ratio"] = float(train_ratio)
    valid_structures, train_structures, dft_structures, discard_structures = route_teacher_labeled_structures(
        relabel_result.labeled_structures,
        routing_policy=routing_policy,
        label_cfg=label_core_cfg,
    )

    print("\nValid structures:", len(valid_structures))
    print("\nDataset split")
    print("Teacher label candidate threshold:", routing_policy.teacher_trust_lo)
    print("Student train:", len(train_structures))
    print("DFT:", len(dft_structures))
    print("Discard:", len(discard_structures))

    outputs = LabelingOutputs(
        labeled_structures=relabel_result.labeled_structures,
        uncertainties=relabel_result.uncertainties,
        valid_structures=valid_structures,
        train_structures=train_structures,
        dft_structures=dft_structures,
        discard_structures=discard_structures,
        grouped_train_val=build_grouped_train_val(train_structures, label_core_cfg),
    )
    export_label_results(
        outputs,
        {
            "labeled_output_dir": str(Path(request.output_targets.labeled_output_dir).resolve()),
            "active_learning_output_dir": str(Path(request.output_targets.active_learning_output_dir).resolve()),
            "element_order": element_order,
            "write_deepmd_dataset": bool(write_deepmd_dataset),
        },
    )

    append_active_learning_xyz = request.output_targets.append_active_learning_xyz
    if append_active_learning_xyz is not None:
        append_candidate_xyz(Path(append_active_learning_xyz).resolve(), list(outputs.dft_structures))

    label_counts = {
        "input": int(len(atoms_list)),
        "train": int(len(train_structures)),
        "dft": int(len(dft_structures)),
        "discard": int(len(discard_structures)),
    }
    labeled_output_dir = Path(request.output_targets.labeled_output_dir).resolve()
    active_learning_output_dir = Path(request.output_targets.active_learning_output_dir).resolve()
    summary = {
        "completed": True,
        "input_xyz": str(merged.resolve()),
        "label_dir": str(labeled_output_dir),
        "active_learning_dir": str(active_learning_output_dir),
        "label_counts": label_counts,
        "label_system_stats": summarize_label_split_by_system(
            labeled_output_dir / "train.xyz",
            active_learning_output_dir / "dft.xyz",
            labeled_output_dir / "discard.xyz",
        ),
        "uncertainty": _uncertainty_summary(relabel_result.uncertainties),
    }
    return TeacherLabelingRunResult(outputs=outputs, summary=summary)


def _reject_top_level_backend_execution_fields(label_cfg: dict[str, Any]) -> None:
    legacy_keys = {"mace_backend", "device", "default_dtype"}
    present = sorted(set(label_cfg) & legacy_keys)
    if present:
        raise ValueError(
            "relabel config rejects top-level backend execution fields; "
            f"use teacher_backend_options instead: {present}"
        )


def _validate_compat_teacher_num(backend, teacher_dir: Path, declared_teacher_num: int | None) -> None:
    if declared_teacher_num is None:
        return
    if not hasattr(backend, "collect_models"):
        return
    discovered = [
        artifact
        for artifact in backend.collect_models(str(teacher_dir))
        if Path(artifact.model_path).is_file() and Path(artifact.model_path).stat().st_size > 0
    ]
    if not discovered:
        return
    if len(discovered) != int(declared_teacher_num):
        raise ValueError(
            "compat relabel teacher_num does not match discovered teacher committee: "
            f"teacher_num={declared_teacher_num}, discovered={len(discovered)}, backend={backend.backend_name}"
        )


def _resolve_relabel_teacher_num(*candidates: Any) -> int | None:
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = int(candidate)
        if normalized > 0:
            return normalized
    return None
