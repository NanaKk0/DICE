from __future__ import annotations

import numpy as np

from workflow_backends.teacher import get_teacher_backend
from workflow_backends.teacher.base import TeacherRelabelRequest
from workflow_core.labeling.grouping import (
    build_grouped_train_val,
    composition_key,
    composition_name,
    group_by_composition,
    sort_atoms,
    split_train_validation,
)
from workflow_core.labeling.results import LabelingOutputs
from workflow_core.labeling.routing import TeacherRoutingPolicy, route_teacher_labeled_structures, structure_is_valid


def print_uncertainty_histogram(uncertainties):
    print("\nForce uncertainty statistics")
    if len(uncertainties) == 0:
        print("no labeled structures")
        return
    print("min:", uncertainties.min())
    print("max:", uncertainties.max())
    print("mean:", uncertainties.mean())
    print("median:", np.median(uncertainties))

    bins = np.linspace(uncertainties.min(), uncertainties.max(), 10)
    hist, edges = np.histogram(uncertainties, bins=bins)

    print("\nHistogram")

    for i in range(len(hist)):
        print(f"{edges[i]:.4f} - {edges[i+1]:.4f} : {hist[i]}")


def split_dataset(atoms_list, cfg):
    _, train_structures, dft_structures, discard_structures = route_teacher_labeled_structures(
        atoms_list,
        routing_policy=TeacherRoutingPolicy(
            teacher_trust_lo=float(cfg.get("candidate_sigma_threshold", 0.10)),
            source="teacher_labeling",
        ),
        label_cfg=cfg,
    )
    print("\nDataset split")
    print("Teacher label candidate threshold:", float(cfg.get("candidate_sigma_threshold", 0.10)))
    print("Student train:", len(train_structures))
    print("DFT:", len(dft_structures))
    print("Discard:", len(discard_structures), "(disabled for ordinary teacher labeling)")
    return train_structures, dft_structures, discard_structures


def run_labeling(atoms_list, cfg):
    atoms_list = [sort_atoms(atoms) for atoms in atoms_list]

    print("Frames:", len(atoms_list))

    backend = get_teacher_backend(cfg.get("teacher_backend", "mace"))
    relabel_result = backend.relabel_structures(
        atoms_list,
        TeacherRelabelRequest(
            committee_dir=cfg["teacher_dir"],
            teacher_num=int(cfg.get("teacher_num", 0) or 0) or None,
            device=str(cfg.get("device", "cpu")),
            default_dtype=str(cfg.get("default_dtype", "float32")),
            backend_options={
                "teacher_backend": cfg.get("teacher_backend"),
                "mace_backend": cfg.get("mace_backend"),
                "teacher_num": cfg.get("teacher_num"),
            },
        ),
    )

    print_uncertainty_histogram(relabel_result.uncertainties)

    valid_structures, train_structures, dft_structures, discard_structures = route_teacher_labeled_structures(
        relabel_result.labeled_structures,
        routing_policy=TeacherRoutingPolicy(
            teacher_trust_lo=float(cfg.get("candidate_sigma_threshold", 0.10)),
            source="teacher_labeling",
        ),
        label_cfg=cfg,
    )

    print("\nValid structures:", len(valid_structures))

    grouped_train_val = build_grouped_train_val(train_structures, cfg)

    return LabelingOutputs(
        labeled_structures=relabel_result.labeled_structures,
        uncertainties=relabel_result.uncertainties,
        valid_structures=valid_structures,
        train_structures=train_structures,
        dft_structures=dft_structures,
        discard_structures=discard_structures,
        grouped_train_val=grouped_train_val,
    )
