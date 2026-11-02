from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from workflow_policies.gmm_reference_tail import GMMScoreResult


SCHEMA = "dice.compact-gmm-score"
VERSION = 1


def compact_score_view_from_stdout(stdout, *, ordered_identities, model_digest, reference_digest):
    from workflow_core.lifecycle.compact_scores import CompactScoreView

    prefix = "DICE_GMM_SCORE_RESULT="
    lines = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("compact GMM worker must return exactly one receipt")
    receipt = json.loads(lines[0])
    if not isinstance(receipt, dict):
        raise ValueError("compact GMM worker receipt must be an object")
    return CompactScoreView(receipt, ordered_identities=ordered_identities,
                            expected_model_digest=model_digest, expected_reference_digest=reference_digest)


def write_compact_score_store(
    root: str | Path,
    scored: GMMScoreResult,
    *,
    model_digest: str,
    reference_digest: str,
    runtime_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write ordinal arrays atomically and return a population-invariant receipt."""
    target = Path(root).resolve()
    staging = target.with_name(f".{target.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    count = int(len(scored.frame_p))
    if count <= 0 or scored.frame_statistics.shape[0] != count:
        raise ValueError("compact GMM result arrays have inconsistent frame coverage")
    arrays: dict[str, np.ndarray] = {
        "probabilities": np.asarray(scored.frame_p, dtype=np.float64),
        "statistics": np.asarray(scored.frame_statistics, dtype=np.float64),
    }
    if scored.frame_means is not None:
        means = np.asarray(scored.frame_means, dtype=np.float32)
        if means.ndim != 2 or means.shape[0] != count:
            raise ValueError("compact GMM frame means have invalid ordinal coverage")
        arrays["means"] = means
    elements = {}
    positions = {str(identity): index for index, identity in enumerate(scored.frame_ids)}
    for index, (element, (identities, rows)) in enumerate((scored.element_means or {}).items()):
        means_key, presence_key = f"element_{index}_means", f"element_{index}_presence"
        width = int((scored.descriptor_schema or {})["width"])
        dense = np.zeros((count, width), dtype=np.float32)
        presence = np.zeros(count, dtype=np.bool_)
        ids, values = tuple(identities), np.asarray(rows, dtype=np.float32)
        ordinals = tuple(positions.get(identity, -1) for identity in ids)
        if not isinstance(element, str) or not element or not ids or -1 in ordinals or ordinals != tuple(sorted(set(ordinals))) or values.shape != (len(ids), width) or not np.isfinite(values).all():
            raise ValueError("compact GMM element means changed order, coverage or width")
        ordinals = [positions[identity] for identity in ids]
        dense[ordinals], presence[ordinals] = values, True
        arrays[means_key], arrays[presence_key] = dense, presence
        elements[element] = {"means": means_key, "presence": presence_key}
    evidence: dict[str, object] = {}
    for name, array in arrays.items():
        path = staging / f"{name}.npy"
        with path.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        evidence[name] = {
            "file": path.name,
            "dtype": np.dtype(array.dtype).str,
            "shape": list(array.shape),
            "sha256": _digest(path),
        }
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "root": str(target),
        "frame_count": count,
        "ordinal_start": 0,
        "ordinal_stop": count,
        "model_digest": model_digest,
        "reference_digest": reference_digest,
        "descriptor_schema": dict(scored.descriptor_schema or {}),
        "arrays": evidence,
        "runtime_diagnostics": dict(runtime_diagnostics or {}),
    }
    if scored.element_means is not None:
        receipt["elements"] = elements
    (staging / "receipt.json").write_text(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    return receipt


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["SCHEMA", "VERSION", "write_compact_score_store"]
