from __future__ import annotations

import hashlib

import numpy as np


def _stable_random_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) ^ int(base_seed)) % (2**32)


def select_training_candidates_by_system(
    candidate_records: list[dict],
    *,
    max_train_candidates_per_system: int,
    selection_mode: str,
    random_seed: int,
    teacher_round: int,
    student_round: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in candidate_records:
        atoms = record["atoms"]
        system_id = str(atoms.info.get("system_id") or atoms.info.get("seed_system_id") or atoms.get_chemical_formula(mode="hill"))
        grouped.setdefault(system_id, []).append(record)

    selected_records: list[dict] = []
    for system_id in sorted(grouped):
        records = list(grouped[system_id])
        limit = max(0, int(max_train_candidates_per_system))
        if str(selection_mode).strip().lower() == "random_per_system":
            seed = _stable_random_seed("candidate_train", teacher_round, student_round, system_id, base_seed=random_seed)
            rng = np.random.default_rng(seed)
            if len(records) > 1:
                rng.shuffle(records)
            selected_records.extend(records[:limit])
            continue
        ranked = sorted(
            records,
            key=lambda item: (
                -float(item["decision_metric"]),
                str(item.get("source_job_id", "")),
                int(item.get("source_frame_index", 0)),
            ),
        )
        selected_records.extend(ranked[:limit])
    return selected_records
