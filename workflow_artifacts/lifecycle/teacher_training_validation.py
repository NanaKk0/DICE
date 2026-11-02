"""Physical dataset checks for fresh teacher training publication."""

from __future__ import annotations

from pathlib import Path

from workflow_core.structures.formats import read_extxyz_structures


def validate_teacher_training_publication_dataset(
    stage_root: str | Path,
    facts: object,
) -> bool:
    """Match staged train and validation IDs to the path-free expectation."""

    try:
        dataset_root = Path(stage_root) / "dataset"
        train = read_extxyz_structures(
            dataset_root / "train.xyz", allow_empty=False
        )
        train_ids = tuple(
            str(frame.info.get("structure_id", "")) for frame in train
        )
        if train_ids != facts.dataset.train.ordered_structure_ids:
            return False

        valid_path = dataset_root / "valid.xyz"
        if facts.dataset.validation_mode == "self":
            if valid_path.exists():
                return False
            validation_ids = train_ids
        else:
            validation = read_extxyz_structures(valid_path, allow_empty=False)
            validation_ids = tuple(
                str(frame.info.get("structure_id", "")) for frame in validation
            )
        return validation_ids == facts.dataset.validation.ordered_structure_ids
    except (OSError, TypeError, ValueError):
        return False


__all__ = ["validate_teacher_training_publication_dataset"]
