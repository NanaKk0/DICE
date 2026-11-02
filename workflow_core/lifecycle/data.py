from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from ase import Atoms

from workflow_core.structures.canonical_identity import (
    IdentityAudit,
    copy_atoms_preserving_singlepoint_results,
    stamp_external_ingress_identity,
)
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


_TEACHER_SPLIT_ALGORITHM = "teacher-system-stratified"
_TEACHER_SPLIT_ALGORITHM_REVISION = 1
_HISTORICAL_TEACHER_SPLIT_HASH_DOMAIN = "v2_teacher_system_stratified_v1"


@dataclass(frozen=True, slots=True)
class LabelContract:
    require_stress: bool = False
    require_virial: bool = False


@dataclass(frozen=True, slots=True)
class TeacherDatasetMaterialization:
    train_data_path: Path
    validation_data_path: Path
    validation_mode: str
    split_summary_path: Path
    split_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_data_path", Path(self.train_data_path).resolve())
        object.__setattr__(self, "validation_data_path", Path(self.validation_data_path).resolve())
        object.__setattr__(self, "split_summary_path", Path(self.split_summary_path).resolve())
        object.__setattr__(self, "split_summary", MappingProxyType(dict(self.split_summary)))

    def __fspath__(self) -> str:
        """Keep path-oriented compatibility while callers migrate to the typed result."""
        return str(self.train_data_path)


@dataclass(frozen=True, slots=True)
class TrainRequest:
    iter_index: int
    round_index: int
    dataset_path: Path
    model_aliases: tuple[Path, ...]
    operation_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path).resolve())
        object.__setattr__(self, "model_aliases", tuple(Path(path).resolve() for path in self.model_aliases))
        object.__setattr__(self, "operation_context", MappingProxyType(dict(self.operation_context)))


@dataclass(frozen=True, slots=True)
class LabelRequest:
    iter_index: int
    round_index: int | None
    input_path: Path
    teacher_aliases: tuple[Path, ...]
    operation_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path).resolve())
        object.__setattr__(self, "teacher_aliases", tuple(Path(path).resolve() for path in self.teacher_aliases))
        object.__setattr__(self, "operation_context", MappingProxyType(dict(self.operation_context)))


@dataclass(frozen=True, slots=True)
class SamplingRequest:
    iter_index: int
    round_index: int
    seed_path: Path
    model_aliases: tuple[Path, ...]
    operation_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed_path", Path(self.seed_path).resolve())
        object.__setattr__(self, "model_aliases", tuple(Path(path).resolve() for path in self.model_aliases))
        object.__setattr__(self, "operation_context", MappingProxyType(dict(self.operation_context)))


class DataService:
    """Canonical extxyz materialization for the isolated lifecycle iter/round route."""

    def __init__(
        self,
        *,
        run_json_path: str | Path,
        iter_index: int,
        teacher_type_map: tuple[str, ...],
        student_type_map: tuple[str, ...],
        teacher_contract: LabelContract,
        student_contract: LabelContract,
    ) -> None:
        self.run_json_path = Path(run_json_path).resolve()
        self.iter_index = int(iter_index)
        self.teacher_type_map = tuple(teacher_type_map)
        self.student_type_map = tuple(student_type_map)
        self.teacher_contract = teacher_contract
        self.student_contract = student_contract

    def load_student_bootstrap(self, sources: list[str | Path]) -> list[Atoms]:
        frames = self._load_sources(sources)
        self._validate_type_map(frames, self.student_type_map, "student_train_bootstrap")
        result: list[Atoms] = []
        for atoms in frames:
            item = copy_atoms_preserving_singlepoint_results(atoms)
            item.info["structure_id"] = self._structure_id(item)
            item.info["structure_source"] = "external"
            item.info.pop("system_id", None)
            result.append(item)
        return result

    def load_teacher_bootstrap(
        self,
        sources: list[str | Path],
        *,
        audit: IdentityAudit | None = None,
    ) -> list[Atoms]:
        """Load labeled teacher bootstrap frames through canonical external ingress."""
        frames = self._load_sources(sources)
        self._validate_type_map(frames, self.teacher_type_map, "teacher_train_bootstrap")
        self.validate_labels(frames, self.teacher_contract)
        result: list[Atoms] = []
        for atoms in frames:
            had_system_id = bool(str(atoms.info.get("system_id", "") or "").strip())
            item = stamp_external_ingress_identity(atoms)
            item.info["structure_id"] = self._structure_id(item)
            if audit is not None:
                if not had_system_id:
                    audit.record_formula_fallback(
                        system_id=str(item.info["system_id"]),
                        formula=item.get_chemical_formula(mode="hill"),
                    )
                audit.record_generated_structure_id(str(item.info["structure_id"]))
            result.append(item)
        return result

    def load_fp_data(
        self,
        sources: list[str | Path],
        *,
        audit: IdentityAudit | None = None,
    ) -> list[Atoms]:
        """Load later-iter returned labels using the computed geometry as authoritative."""
        if self.iter_index <= 0:
            raise ValueError("fp_data is valid only for a later lifecycle iter")
        return self.load_teacher_bootstrap(sources, audit=audit)

    def load_teacher_training_frames(
        self,
        bootstrap_sources: list[str | Path],
        *,
        fp_sources: list[str | Path] | None = None,
        audit: IdentityAudit | None = None,
    ) -> list[Atoms]:
        """Return the ordered, canonical sequence consumed by lifecycle teacher training."""
        source_frames = [self.load_teacher_bootstrap(bootstrap_sources, audit=audit)]
        if fp_sources:
            source_frames.append(self.load_fp_data(fp_sources, audit=audit))
        return self._merge_teacher_frame_sources(source_frames, audit=audit)

    def teacher_dataset_handle(
        self,
        bootstrap_sources: list[str | Path],
        *,
        dataset_id: str,
        valid_frac: float = 0.0,
        fp_sources: list[str | Path] | None = None,
    ):
        """Return the canonical path-free train/validation membership view."""

        from workflow_core.lifecycle.teacher_capabilities import TeacherDatasetHandle

        if not 0.0 <= valid_frac < 1.0:
            raise ValueError("valid_frac must be in [0, 1)")
        frames = self.load_teacher_training_frames(
            bootstrap_sources,
            fp_sources=fp_sources,
        )
        validation_ids = self._teacher_validation_ids(frames, valid_frac=valid_frac)
        train_ids = tuple(
            str(frame.info["structure_id"])
            for frame in frames
            if str(frame.info["structure_id"]) not in validation_ids
        )
        held_out_ids = tuple(
            str(frame.info["structure_id"])
            for frame in frames
            if str(frame.info["structure_id"]) in validation_ids
        )
        validation_mode = "self" if valid_frac == 0.0 else "holdout"
        return TeacherDatasetHandle(
            dataset_id=dataset_id,
            train_structure_ids=train_ids,
            validation_structure_ids=(
                train_ids if validation_mode == "self" else held_out_ids
            ),
            validation_mode=validation_mode,
        )

    def _merge_teacher_frame_sources(
        self,
        source_frames: list[list[Atoms]],
        *,
        audit: IdentityAudit | None,
    ) -> list[Atoms]:
        merged: dict[str, Atoms] = {}
        duplicate_count = 0
        for frames in source_frames:
            for atoms in frames:
                item = copy_atoms_preserving_singlepoint_results(atoms)
                identity = str(item.info["structure_id"])
                previous = merged.get(identity)
                if previous is not None:
                    self._assert_compatible_teacher_duplicate(previous, item)
                    duplicate_count += 1
                merged[identity] = item
        if audit is not None:
            audit.record_external_dedup_drop(route="teacher_training", count=duplicate_count)
        return list(merged.values())

    def stamp_workflow_frames(self, frames: list[Atoms], *, system_id: str) -> list[Atoms]:
        result: list[Atoms] = []
        for atoms in frames:
            item = copy_atoms_preserving_singlepoint_results(atoms)
            item.info["system_id"] = str(system_id)
            item.info["structure_id"] = self._structure_id(item)
            item.info["structure_source"] = "workflow"
            result.append(item)
        return result

    def materialize_teacher_dataset(
        self,
        sources: list[str | Path],
        target: str | Path,
        *,
        valid_frac: float = 0.0,
        fp_sources: list[str | Path] | None = None,
    ) -> TeacherDatasetMaterialization:
        if not 0.0 <= valid_frac < 1.0:
            raise ValueError("valid_frac must be in [0, 1)")
        audit = IdentityAudit()
        bootstrap_frames = self.load_teacher_bootstrap(sources, audit=audit)
        fp_frames: list[Atoms] = []
        if fp_sources:
            fp_frames = self.load_fp_data(fp_sources, audit=audit)
        frames = self._merge_teacher_frame_sources([bootstrap_frames, fp_frames], audit=audit)
        validation_ids = self._teacher_validation_ids(frames, valid_frac=valid_frac)
        train_frames = [frame for frame in frames if str(frame.info["structure_id"]) not in validation_ids]
        validation_frames = [frame for frame in frames if str(frame.info["structure_id"]) in validation_ids]
        train_path = Path(target).resolve()
        validation_path = train_path.with_name("valid.xyz")
        validation_mode = "self" if valid_frac == 0.0 else "holdout"
        write_extxyz_structures(train_path, train_frames, allow_empty=False)
        if validation_mode == "self":
            validation_path.unlink(missing_ok=True)
            resolved_validation_path = train_path
        else:
            write_extxyz_structures(validation_path, validation_frames, allow_empty=False)
            resolved_validation_path = validation_path
        system_summary: dict[str, dict[str, int | float]] = {}
        for frame in frames:
            system_id = str(frame.info["system_id"])
            row = system_summary.setdefault(
                system_id,
                {
                    "frame_count": 0,
                    "train_count": 0,
                    "held_out_count": 0,
                    "self_evaluation_count": 0,
                    "realized_valid_fraction": 0.0,
                },
            )
            row["frame_count"] += 1
            if str(frame.info["structure_id"]) in validation_ids:
                row["held_out_count"] += 1
            else:
                row["train_count"] += 1
        for row in system_summary.values():
            row["realized_valid_fraction"] = row["held_out_count"] / row["frame_count"]
        summary = {
            "algorithm": _TEACHER_SPLIT_ALGORITHM,
            "algorithm_revision": _TEACHER_SPLIT_ALGORITHM_REVISION,
            "requested_valid_fraction": valid_frac,
            "realized_valid_fraction": len(validation_frames) / len(frames),
            "validation_mode": validation_mode,
            "frame_count": len(frames),
            "merged_frame_count": len(frames),
            "source_counts": {
                "teacher_train_bootstrap": {
                    "path_count": len(sources),
                    "frame_count": len(bootstrap_frames),
                },
                "fp_data": {
                    "path_count": len(fp_sources or []),
                    "frame_count": len(fp_frames),
                },
            },
            "duplicate_drop_count": audit.external_dedup_dropped_counts.get("teacher_training", 0),
            "formula_fallback_count": len(audit.formula_fallbacks),
            "systems": system_summary,
        }
        summary_path = train_path.with_name("split_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return TeacherDatasetMaterialization(
            train_data_path=train_path,
            validation_data_path=resolved_validation_path,
            validation_mode=validation_mode,
            split_summary_path=summary_path,
            split_summary=summary,
        )

    def _teacher_validation_ids(self, frames: list[Atoms], *, valid_frac: float) -> set[str]:
        if valid_frac == 0.0:
            return set()
        by_system: dict[str, list[str]] = {}
        for frame in frames:
            by_system.setdefault(str(frame.info["system_id"]), []).append(str(frame.info["structure_id"]))
        selected: set[str] = set()
        for system_id, identities in by_system.items():
            if len(identities) < 2:
                continue
            quota = min(len(identities) - 1, max(1, math.floor(len(identities) * valid_frac + 0.5)))
            ranked = sorted(
                identities,
                key=lambda identity: (
                    hashlib.sha256(
                            f"{self.run_json_path}\0{_HISTORICAL_TEACHER_SPLIT_HASH_DOMAIN}\0{system_id}\0{identity}".encode(
                            "utf-8"
                        )
                    ).digest(),
                    identity,
                ),
            )
            selected.update(ranked[:quota])
        if not selected:
            raise ValueError("positive valid_frac requires at least one system with two or more teacher frames")
        return selected

    def materialize_student_dataset(
        self,
        sources: list[list[Atoms]],
        target: str | Path,
    ) -> Path:
        merged: dict[str, Atoms] = {}
        for source in sources:
            for atoms in source:
                item = copy_atoms_preserving_singlepoint_results(atoms)
                identity = str(item.info.get("structure_id", ""))
                if not identity.startswith("sha256:"):
                    raise ValueError("student training data requires a canonical structure_id")
                previous = merged.get(identity)
                if previous is not None:
                    self._assert_compatible_duplicate(previous, item)
                merged[identity] = item
        frames = list(merged.values())
        self._validate_type_map(frames, self.student_type_map, "student training data")
        self.validate_labels(frames, self.student_contract)
        return write_extxyz_structures(target, frames, allow_empty=False)

    def merge_student_relabel_sources(
        self,
        sources: list[list[Atoms]],
    ) -> list[Atoms]:
        """Build one exact-geometry union for teacher relabeling.

        Earlier sources own both ordering and metadata when the same geometry
        appears again. Labels are deliberately not validated here because the
        returned population is always passed through the current teacher.
        """

        result: list[Atoms] = []
        identities: dict[str, str] = {}
        geometries: set[str] = set()
        for source in sources:
            for atoms in source:
                item = copy_atoms_preserving_singlepoint_results(atoms)
                identity = str(item.info.get("structure_id", ""))
                if not identity.startswith("sha256:"):
                    raise ValueError(
                        "teacher relabel data requires a canonical structure_id"
                    )
                geometry = exact_structure_hash(item)
                previous_geometry = identities.get(identity)
                if previous_geometry is not None and previous_geometry != geometry:
                    raise ValueError("conflicting geometry for duplicate structure_id")
                identities[identity] = geometry
                if geometry in geometries:
                    continue
                result.append(item)
                geometries.add(geometry)
        self._validate_type_map(result, self.student_type_map, "teacher relabel data")
        return result

    def validation_split(self, frames: list[Atoms], *, valid_frac: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not 0 <= valid_frac < 1:
            raise ValueError("valid_frac must be in [0, 1)")
        train: list[str] = []
        valid: list[str] = []
        for atoms in frames:
            identity = str(atoms.info.get("structure_id", ""))
            if not identity.startswith("sha256:"):
                raise ValueError("validation splitting requires structure_id")
            digest = hashlib.sha256(f"{self.run_json_path}\0{identity}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") / 2**64
            (valid if bucket < valid_frac else train).append(identity)
        return tuple(train), tuple(valid)

    def validate_labels(self, frames: list[Atoms], contract: LabelContract) -> None:
        validate_label_properties(frames, contract)

    def _load_sources(self, sources: list[str | Path]) -> list[Atoms]:
        frames: list[Atoms] = []
        for source in sources:
            frames.extend(read_extxyz_structures(source, allow_empty=False))
        if not frames:
            raise ValueError("lifecycle external input must contain at least one frame")
        return frames

    def _structure_id(self, atoms: Atoms) -> str:
        payload = f"{self.run_json_path}\0{exact_structure_hash(atoms)}".encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _assert_compatible_duplicate(previous: Atoms, current: Atoms) -> None:
        if exact_structure_hash(previous) != exact_structure_hash(current):
            raise ValueError("conflicting geometry for duplicate structure_id")
        for key in ("energy", "forces", "stress", "virial"):
            previous_value = _label_value(previous, key)
            current_value = _label_value(current, key)
            if previous_value is None and current_value is None:
                continue
            if previous_value is None or current_value is None or not np.array_equal(np.asarray(previous_value), np.asarray(current_value)):
                raise ValueError(f"conflicting {key} label for duplicate structure_id")

    def _assert_compatible_teacher_duplicate(self, previous: Atoms, current: Atoms) -> None:
        if exact_structure_hash(previous) != exact_structure_hash(current):
            raise ValueError("conflicting geometry for duplicate structure_id")
        required_labels = ["energy", "forces"]
        if self.teacher_contract.require_stress:
            required_labels.append("stress")
        if self.teacher_contract.require_virial:
            required_labels.append("virial")
        for key in required_labels:
            previous_value = _label_value(previous, key)
            current_value = _label_value(current, key)
            if previous_value is None or current_value is None or not np.array_equal(
                np.asarray(previous_value), np.asarray(current_value)
            ):
                raise ValueError(f"conflicting {key} label for duplicate structure_id")

    @staticmethod
    def _validate_type_map(frames: list[Atoms], type_map: tuple[str, ...], field_name: str) -> None:
        supported = set(type_map)
        for atoms in frames:
            unsupported = sorted(set(atoms.get_chemical_symbols()) - supported)
            if unsupported:
                raise ValueError(f"{field_name} contains unsupported elements: {unsupported}")


def resolve_label_contract(teacher_recipe: str | Path, student_recipe: str | Path) -> tuple[LabelContract, LabelContract]:
    return _contract_from_recipe(teacher_recipe), _contract_from_recipe(student_recipe)


def validate_label_properties(frames: list[Atoms], contract: LabelContract) -> None:
    """Validate canonical required label presence, shape, and finiteness."""

    for index, atoms in enumerate(frames):
        results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
        if "energy" not in results:
            raise ValueError(f"frame {index} is missing required energy label")
        energy = np.asarray(results["energy"])
        if energy.size != 1 or not np.isfinite(energy).all():
            raise ValueError(f"frame {index} has a non-finite or malformed energy label")
        if "forces" not in results:
            raise ValueError(f"frame {index} is missing required forces label")
        forces = np.asarray(results["forces"])
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError(f"frame {index} has non-finite or malformed forces labels")
        if contract.require_stress and "stress" not in results:
            raise ValueError(f"frame {index} is missing required stress label")
        if "stress" in results and not np.isfinite(np.asarray(results["stress"])).all():
            raise ValueError(f"frame {index} has a non-finite stress label")
        virial = results.get("virial", atoms.info.get("virial"))
        if contract.require_virial and virial is None:
            raise ValueError(f"frame {index} is missing required virial label")
        if virial is not None and not np.isfinite(np.asarray(virial)).all():
            raise ValueError(f"frame {index} has a non-finite virial label")
        for field_name in (
            "force_uncertainty",
            "force_std_max",
            "force_std_avg",
            "force_std_p90",
        ):
            if field_name in atoms.info and not math.isfinite(
                float(atoms.info[field_name])
            ):
                raise ValueError(
                    f"frame {index} has non-finite teacher {field_name} evidence"
                )


def _contract_from_recipe(recipe: str | Path) -> LabelContract:
    path = Path(recipe)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml

            payload = yaml.safe_load(text)
        except Exception:
            payload = {}
    words = _recipe_words(payload)
    return LabelContract(require_stress="stress" in words, require_virial="virial" in words)


def _recipe_words(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(*(_recipe_words(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_recipe_words(item) for item in value)) if value else set()
    if isinstance(value, str):
        return {value.lower()}
    return set()


def _label_value(atoms: Atoms, key: str) -> Any:
    results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
    return results.get(key, atoms.info.get(key))


def publish_model_alias(native_model: str | Path, alias: str | Path) -> Path:
    """Compatibility wrapper for the artifact-owned model alias publisher."""

    from workflow_core.lifecycle.model_handles import (
        publish_model_alias as publish_physical_model_alias,
    )

    return publish_physical_model_alias(native_model, alias)
