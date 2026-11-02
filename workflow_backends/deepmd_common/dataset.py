from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from workflow_backends.deepmd_common.data_format import has_reference_stress, write_deepmd_dataset
from workflow_core.labeling.grouping import composition_key, composition_name
from workflow_core.structures.formats import read_extxyz_structures


@dataclass(frozen=True, slots=True)
class DeepMDTeacherDatasetPresentation:
    train_systems: tuple[Path, ...]
    validation_systems: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_systems", tuple(Path(path).resolve() for path in self.train_systems))
        object.__setattr__(
            self,
            "validation_systems",
            tuple(Path(path).resolve() for path in self.validation_systems),
        )


def materialize_native_teacher_datasets(
    executor: Any,
    *,
    train_data_path: str | Path,
    validation_data_path: str | Path,
    validation_mode: str,
    work_root: str | Path,
    type_map: Sequence[str],
) -> DeepMDTeacherDatasetPresentation:
    """Present canonical membership at DeepMD's backend-native boundary.

    ``self`` is the lifecycle's zero-holdout mode: all canonical structures
    remain in training and DeepMD receives no validation dataset. A positive
    validation fraction is represented by ``holdout`` and materialized
    separately.
    """
    train_source = Path(train_data_path).resolve()
    validation_source = Path(validation_data_path).resolve()
    if validation_mode not in {"holdout", "self"}:
        raise ValueError("DeepMD lifecycle teacher validation_mode must be holdout or self")
    if validation_mode == "holdout" and train_source == validation_source:
        raise ValueError("DeepMD lifecycle teacher holdout paths must be disjoint")
    if validation_mode == "self" and train_source != validation_source:
        raise ValueError("DeepMD lifecycle teacher self-validation paths must be equal")
    if not bool(getattr(executor, "requires_native_dataset", False)):
        return DeepMDTeacherDatasetPresentation(
            train_systems=(train_source,),
            validation_systems=(
                () if validation_mode == "self" else (validation_source,)
            ),
        )
    native_root = Path(work_root).resolve() / "deepmd_dataset"
    train_systems = _materialize_composition_systems(
        train_source,
        native_root / "train",
        type_map=type_map,
    )
    if validation_mode == "holdout":
        validation_systems = _materialize_composition_systems(
            validation_source,
            native_root / "validation",
            type_map=type_map,
        )
    else:
        validation_systems = ()
    return DeepMDTeacherDatasetPresentation(
        train_systems=train_systems,
        validation_systems=validation_systems,
    )


def _materialize_composition_systems(
    source: Path,
    target_root: Path,
    *,
    type_map: Sequence[str],
) -> tuple[Path, ...]:
    frames = read_extxyz_structures(source, allow_empty=False)
    grouped: dict[tuple[Any, bool], list[Any]] = {}
    for atoms in frames:
        grouped.setdefault((composition_key(atoms), has_reference_stress(atoms)), []).append(atoms)
    systems: list[Path] = []
    names: set[str] = set()
    for (key, has_stress), atoms_group in sorted(grouped.items()):
        name = composition_name(key) + ("" if has_stress else "_energy_force_only")
        if name in names:
            raise ValueError(f"colliding DeepMD composition system name: {name}")
        names.add(name)
        systems.append(
            write_deepmd_dataset(
                atoms_group,
                target_root / name,
                [str(item) for item in type_map],
            )
        )
    return tuple(systems)


def materialize_native_dataset(
    executor: Any,
    dataset_path: str | Path,
    *,
    work_root: str | Path,
    type_map: Sequence[str],
) -> tuple[Path, ...]:
    """Convert canonical lifecycle extxyz only when the selected executor is native DeePMD."""
    source = Path(dataset_path).resolve()
    if not bool(getattr(executor, "requires_native_dataset", False)):
        return (source,)
    target = Path(work_root) / "deepmd_dataset"
    frames = read_extxyz_structures(source, allow_empty=False)
    grouped: dict[tuple[Any, bool], list[Any]] = {}
    for atoms in frames:
        grouped.setdefault((composition_key(atoms), has_reference_stress(atoms)), []).append(atoms)
    if len(grouped) == 1:
        return (write_deepmd_dataset(frames, target, [str(item) for item in type_map]),)
    systems: list[Path] = []
    for (key, has_stress), atoms_group in sorted(grouped.items()):
        name = composition_name(key) + ("" if has_stress else "_energy_force_only")
        systems.append(
            write_deepmd_dataset(
                atoms_group,
                target / name,
                [str(item) for item in type_map],
            )
        )
    return tuple(systems)
