from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.uncertainty import UncertaintyRecord, route_uncertainty
from workflow_core.structures.formats import active_extxyz_to_canonical_structures


def parse_active_out(
    active_out_path: str | Path,
    *,
    task_id: str,
    temperature: float | None,
    trust_lo: float,
    trust_hi: float,
    structure_dir: str | Path | None = None,
) -> list[UncertaintyRecord]:
    active_out_path = Path(active_out_path).resolve()
    if not active_out_path.exists():
        return []
    records: list[UncertaintyRecord] = []
    with open(active_out_path, "r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            step = int(float(parts[0]))
            force_std_max = float(parts[1])
            route = route_uncertainty(force_std_max, trust_lo=trust_lo, trust_hi=trust_hi)
            structure_path = None
            if structure_dir is not None and route != "accurate":
                structure_path = str(Path(structure_dir).resolve() / f"{task_id}_{index:06d}.extxyz")
            records.append(
                UncertaintyRecord(
                    backend="nep",
                    sampling_backend="gpumd",
                    source="active.out",
                    task_id=task_id,
                    frame_id=f"{task_id}:{index}",
                    step=step,
                    temperature=temperature,
                    force_std_max=force_std_max,
                    route=route,
                    structure_path=structure_path,
                    metadata={},
                )
            )
    return records


def pair_active_records_with_structures(
    records: list[UncertaintyRecord],
    active_xyz_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
):
    routed_records = [record for record in records if record.route != "accurate"]
    frames = active_extxyz_to_canonical_structures(
        active_xyz_path,
        force_std_values=[record.force_std_max for record in routed_records],
        routes=[record.route for record in routed_records],
        metadata=metadata,
    )
    paired: list[tuple[UncertaintyRecord, Any]] = []
    for record, atoms in zip(routed_records, frames):
        paired.append((record, atoms))
    return paired
