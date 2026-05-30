from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase.neighborlist import neighbor_list


@dataclass(slots=True)
class TeacherRoutingPolicy:
    teacher_trust_lo: float
    teacher_trust_hi: float | None = None
    source: str = "teacher_labeling"


def structure_is_valid(atoms, cfg: dict[str, Any]) -> bool:
    max_force = cfg.get("max_force", 50.0)

    if atoms.calc is not None:
        forces = atoms.get_forces()
        if np.max(np.linalg.norm(forces, axis=1)) > max_force:
            return False

    min_distances = cfg.get(
        "min_distances",
        {
            "Pt-Pt": 2.0,
            "Pt-O": 1.3,
            "O-H": 0.6,
        },
    )

    if len(atoms) < 2:
        return True

    i, j, d = neighbor_list("ijd", atoms, 3.0)

    for a, b, dist in zip(i, j, d):
        s1 = atoms[a].symbol
        s2 = atoms[b].symbol
        pair = "-".join(sorted([s1, s2]))

        if pair in min_distances and dist < min_distances[pair]:
            return False

    return True


def route_teacher_labeled_structures(
    labeled_structures: list[Any],
    *,
    routing_policy: TeacherRoutingPolicy,
    label_cfg: dict[str, Any],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    if routing_policy.teacher_trust_hi is not None and float(routing_policy.teacher_trust_hi) <= float(routing_policy.teacher_trust_lo):
        raise ValueError("teacher_trust_hi must be larger than teacher_trust_lo")

    valid_structures = []
    train_structures = []
    dft_structures = []
    discard_structures = []

    for atoms in labeled_structures:
        atoms.info["teacher_relabel_source"] = routing_policy.source
        atoms.info["teacher_trust_lo"] = float(routing_policy.teacher_trust_lo)
        if routing_policy.teacher_trust_hi is not None:
            atoms.info["teacher_trust_hi"] = float(routing_policy.teacher_trust_hi)

        if not structure_is_valid(atoms, label_cfg):
            atoms.info["teacher_relabel_route"] = "teacher_failed_uncertainty_structure"
            atoms.info["label_split"] = "discard"
            atoms.info["discard_reason"] = f"{routing_policy.source}_invalid_structure"
            discard_structures.append(atoms)
            continue

        valid_structures.append(atoms)
        uncertainty = float(atoms.info["force_uncertainty"])
        atoms.info["label_uncertainty_threshold"] = float(routing_policy.teacher_trust_lo)

        if uncertainty < float(routing_policy.teacher_trust_lo):
            atoms.info["teacher_relabel_route"] = "teacher_train_structure"
            atoms.info["label_split"] = "accurate"
            train_structures.append(atoms)
        elif routing_policy.teacher_trust_hi is not None and uncertainty >= float(routing_policy.teacher_trust_hi):
            atoms.info["teacher_relabel_route"] = "teacher_failed_uncertainty_structure"
            atoms.info["label_split"] = "discard"
            atoms.info["discard_reason"] = f"{routing_policy.source}_teacher_uncertainty"
            discard_structures.append(atoms)
        else:
            atoms.info["teacher_relabel_route"] = "teacher_dft_candidate_structure"
            atoms.info["label_split"] = "candidate"
            dft_structures.append(atoms)

    return valid_structures, train_structures, dft_structures, discard_structures
