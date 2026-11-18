"""Atomic publication of the shared sampling seed-selection contract."""
import json
import os
from pathlib import Path

from workflow_backends.sampling import SamplingSeedSelection, seed_selection_manifest_payload


def publish_seed_selection_manifest(
    path: Path,
    *,
    iter_index: int,
    round_index: int,
    selections: tuple[SamplingSeedSelection, ...],
) -> None:
    payload = seed_selection_manifest_payload(
        iter_index=iter_index,
        round_index=round_index,
        selections=selections,
    )
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing seed-frame selection manifest disagrees with resolved sampling")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("stale seed-frame selection manifest temporary path")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
