"""Bounded read-only validation for committed student train artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from workflow_core.structures.canonical_identity import validate_structure_source
from workflow_core.structures.formats import read_extxyz_structures


_STUDENT_MEMBER = re.compile(r"^student(?:_[0-9]{2})?$")
_STRUCTURE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_student_training_artifacts(root: str | Path, facts: object) -> bool:
    try:
        block = _regular_directory(root)
        frames = read_extxyz_structures(
            block / "dataset" / "train.xyz", allow_empty=False
        )
        supported = set(facts.type_map)
        identities: list[str] = []
        for frame in frames:
            identity = str(frame.info.get("structure_id", "")).strip()
            if not _STRUCTURE_ID.fullmatch(identity):
                return False
            validate_structure_source(frame.info.get("structure_source"))
            identities.append(identity)
            if not set(frame.get_chemical_symbols()).issubset(supported):
                return False
            if (
                not np.isfinite(np.asarray(frame.positions)).all()
                or not np.isfinite(np.asarray(frame.cell.array)).all()
            ):
                return False
            results = getattr(getattr(frame, "calc", None), "results", {}) or {}
            energy = np.asarray(results.get("energy"))
            forces = np.asarray(results.get("forces"))
            if (
                energy.size != 1
                or not np.isfinite(energy).all()
                or forces.shape != (len(frame), 3)
                or not np.isfinite(forces).all()
            ):
                return False
        if len(set(identities)) != len(identities):
            return False
        _validate_models(block, facts)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _validate_models(root: Path, facts: object) -> None:
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and _STUDENT_MEMBER.fullmatch(path.name)
    }
    if actual != set(facts.member_ids):
        raise ValueError("student train member inventory is not exact")
    suffix = (
        ".txt"
        if facts.backend == "NEP"
        else ".pb" if facts.cli == "tf" else ".pt2"
    )
    for member_id in facts.member_ids:
        member = _regular_directory(root / member_id)
        old = member / "student_model"
        candidates = tuple(
            path for path in member.iterdir() if path.name.startswith("student_model.")
        )
        if old.exists() or old.is_symlink() or len(candidates) != 1:
            raise ValueError("student member requires one native-suffix handle")
        handle = candidates[0]
        if (
            handle.name != f"student_model{suffix}"
            or handle.is_symlink()
            or not handle.is_file()
            or handle.stat().st_size <= 0
        ):
            raise ValueError("student model handle is invalid")
        if facts.backend == "DeepMD":
            _validate_deepmd_input(member / "input.json", facts.type_map)
            if any(
                (member / name).exists() or (member / name).is_symlink()
                for name in (
                    "deepmd_model_assets.json",
                    "student_readout_checkpoint.pt",
                )
            ):
                raise ValueError("student DeepMD train is terminal-only")


def _validate_deepmd_input(path: Path, active_type_map: tuple[str, ...]) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("DeepMD student recovery requires retained input.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeepMD retained input must be an object")
    model = payload.get("model", {})
    native = model.get("type_map") if isinstance(model, dict) else None
    if (
        not isinstance(native, list)
        or not native
        or any(not isinstance(item, str) or not item for item in native)
        or len(set(native)) != len(native)
        or any(item not in native for item in active_type_map)
    ):
        raise ValueError("DeepMD retained type map disagrees with student settings")


def _regular_directory(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("student recovery requires a regular directory")
    return raw.resolve()


__all__ = ["validate_student_training_artifacts"]
