"""Bounded read-only validation for committed teacher lifecycle artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from workflow_core.structures.canonical_identity import (
    validate_canonical_identity,
    validate_structure_source,
)
from workflow_core.structures.formats import read_extxyz_structures


_TEACHER_MEMBER = re.compile(r"^teacher(?:_[0-9]{2})?$")
_STRUCTURE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_teacher_training_artifacts(
    root: str | Path,
    facts: object,
) -> bool:
    """Prove train completion without consulting current external inputs."""

    try:
        block = _regular_directory(root, label="teacher train block")
        train = _validated_labeled_population(
            block / "dataset" / "train.xyz",
            type_map=facts.type_map,
            allow_empty=False,
        )
        train_ids = _ordered_ids(train)
        valid_path = block / "dataset" / "valid.xyz"
        if facts.validation_mode == "self":
            if valid_path.exists() or valid_path.is_symlink():
                return False
        else:
            validation = _validated_labeled_population(
                valid_path,
                type_map=facts.type_map,
                allow_empty=False,
            )
            valid_ids = _ordered_ids(validation)
            if set(train_ids).intersection(valid_ids):
                return False
        _validate_teacher_models(block, facts)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def validate_teacher_screen_artifacts(
    root: str | Path,
    facts: object,
) -> bool:
    """Validate the teacher publication; reports and logs are ignored."""

    try:
        block = _regular_directory(root, label="teacher screen block")
        train = _validated_labeled_population(
            block / "train.xyz",
            type_map=facts.type_map,
            allow_empty=False,
            require_system_id=False,
        )
        candidate = block / "fp_candidate.xyz"
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != 0
            or read_extxyz_structures(candidate, allow_empty=True)
        ):
            return False
        if facts.iter_index == 0 and facts.committee:
            lo = float(facts.teacher_lo)
            for frame in train:
                if "force_uncertainty" not in frame.info:
                    return False
                value = float(frame.info["force_uncertainty"])
                if not math.isfinite(value) or not value < lo:
                    return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def validate_teacher_scan_artifacts(
    root: str | Path,
    facts: object,
) -> bool:
    """Validate one strict router and its configured present-element order."""

    from workflow_policies.gmm_reference_tail import validate_gmm_router

    try:
        block = _regular_directory(root, label="teacher scan block")
        manifest = validate_gmm_router(block / "gmm_router")
        router_map = tuple(str(item) for item in manifest["type_map"])
        configured = facts.type_map
        if not router_map or any(item not in configured for item in router_map):
            return False
        expected_order = tuple(item for item in configured if item in set(router_map))
        if router_map != expected_order:
            return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _validated_labeled_population(
    path: Path,
    *,
    type_map: tuple[str, ...],
    allow_empty: bool,
    require_system_id: bool = True,
):
    if path.is_symlink():
        raise ValueError(f"canonical teacher extxyz must be regular: {path}")
    frames = read_extxyz_structures(path, allow_empty=allow_empty)
    supported = set(type_map)
    identities: list[str] = []
    for frame in frames:
        if require_system_id:
            identity = validate_canonical_identity(frame)["structure_id"]
        else:
            identity = str(frame.info.get("structure_id", "")).strip()
            if not _STRUCTURE_ID.fullmatch(identity):
                raise ValueError("canonical teacher extxyz has invalid structure identity")
            validate_structure_source(frame.info.get("structure_source"))
        identities.append(identity)
        if not set(frame.get_chemical_symbols()).issubset(supported):
            raise ValueError("canonical teacher extxyz contains unsupported elements")
        if (
            not np.isfinite(np.asarray(frame.positions, dtype=float)).all()
            or not np.isfinite(np.asarray(frame.cell.array, dtype=float)).all()
        ):
            raise ValueError("canonical teacher extxyz contains non-finite geometry")
    if len(set(identities)) != len(identities):
        raise ValueError("canonical teacher extxyz identities must be ordered unique")
    _validate_required_labels(frames)
    return frames


def _ordered_ids(frames) -> tuple[str, ...]:
    return tuple(str(frame.info["structure_id"]) for frame in frames)


def _validate_teacher_models(
    root: Path,
    facts: object,
) -> None:
    actual_members = tuple(
        path.name
        for path in root.iterdir()
        if path.is_dir() and _TEACHER_MEMBER.fullmatch(path.name)
    )
    if set(actual_members) != set(facts.member_ids):
        raise ValueError("teacher model member inventory is not exact")
    expected_suffix = _teacher_suffix(facts)
    handles: list[Path] = []
    for member_id in facts.member_ids:
        member = _regular_directory(root / member_id, label="teacher model member")
        old = member / "teacher_model"
        candidates = tuple(
            path for path in member.iterdir() if path.name.startswith("teacher_model.")
        )
        if old.exists() or old.is_symlink() or len(candidates) != 1:
            raise ValueError("teacher member requires exactly one native-suffix handle")
        handle = candidates[0]
        if (
            handle.name != f"teacher_model{expected_suffix}"
            or handle.is_symlink()
            or not handle.is_file()
            or handle.stat().st_size <= 0
        ):
            raise ValueError("teacher canonical model handle is invalid")
        handles.append(handle)
    if facts.backend == "DeepMD":
        _validate_deepmd_models(tuple(handles), facts)


def _teacher_suffix(facts: object) -> str:
    if facts.backend == "MACE":
        return ".model"
    if facts.backend == "DeepMD":
        return ".pt2" if facts.cli == "pt" else ".pb"
    raise ValueError(f"unsupported teacher backend for recovery: {facts.backend}")


def _validate_deepmd_models(
    handles: tuple[Path, ...],
    facts: object,
) -> None:
    metadata = tuple(_deepmd_metadata(handle, facts) for handle in handles)
    families = tuple(item["model_family"] for item in metadata)
    if len(set(families)) != 1:
        raise ValueError("DeepMD teacher members changed model family")
    if facts.cli != "pt":
        return
    family = families[0]
    if family == "dpa4":
        from workflow_artifacts.lifecycle.deepmd_model_assets import (
            load_deepmd_model_bundles,
        )

        bundles = load_deepmd_model_bundles(
            stage_root=handles[0].parent.parent,
            expected_member_ids=facts.member_ids,
            expected_native_type_maps=tuple(
                tuple(item["type_map"]) for item in metadata
            ),
        )
        if tuple(bundle.inference_terminal for bundle in bundles) != handles:
            raise ValueError("DeepMD DPA4 bundle terminal inventory changed")
    elif family == "dpa4c":
        for handle in handles:
            member = handle.parent
            for name in ("deepmd_model_assets.json", "teacher_readout_checkpoint.pt"):
                path = member / name
                if path.exists() or path.is_symlink():
                    raise ValueError("DeepMD DPA4C recovery is terminal-only")


def _deepmd_metadata(
    handle: Path,
    facts: object,
) -> dict[str, object]:
    payload = json.loads(
        (handle.parent / "deepmd_model_metadata.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("DeepMD teacher metadata must be an object")
    type_map = tuple(payload.get("type_map", ()))
    family = payload.get("model_family")
    if (
        not type_map
        or len(set(type_map)) != len(type_map)
        or any(not isinstance(item, str) or not item for item in type_map)
        or any(item not in type_map for item in facts.type_map)
        or family not in {"standard", "dpa4", "dpa4c"}
        or payload.get("native_model_suffix") != handle.suffix
    ):
        raise ValueError("DeepMD teacher static metadata disagrees with settings")
    if facts.cli == "pt":
        digest = payload.get("terminal_model_sha256")
        if digest != _sha256(handle):
            raise ValueError("DeepMD teacher terminal digest metadata drifted")
    return {"model_family": family, "type_map": type_map}


def _regular_directory(path: str | Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return raw.resolve()


def _validate_required_labels(frames) -> None:
    for index, frame in enumerate(frames):
        results = getattr(getattr(frame, "calc", None), "results", {}) or {}
        energy = np.asarray(results.get("energy"))
        forces = np.asarray(results.get("forces"))
        if energy.size != 1 or not np.isfinite(energy).all():
            raise ValueError(f"teacher frame {index} has invalid energy")
        if forces.shape != (len(frame), 3) or not np.isfinite(forces).all():
            raise ValueError(f"teacher frame {index} has invalid forces")
        for name in (
            "stress",
            "virial",
            "force_uncertainty",
            "force_std_max",
            "force_std_avg",
            "force_std_p90",
        ):
            value = results.get(name, frame.info.get(name))
            if value is not None and not np.isfinite(np.asarray(value)).all():
                raise ValueError(f"teacher frame {index} has non-finite {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "validate_teacher_scan_artifacts",
    "validate_teacher_screen_artifacts",
    "validate_teacher_training_artifacts",
]
