from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any


def _sortable_metric(value: Any) -> float:
    if value is None:
        return float("inf")
    try:
        numeric = float(value)
    except Exception:
        return float("inf")
    if not math.isfinite(numeric):
        return float("inf")
    return numeric


def choose_best_driver_index(members: list[dict[str, Any]]) -> int:
    selection_ready = [
        member for member in members
        if member.get("status") == "success"
        and isinstance(member.get("selection_metrics"), dict)
        and member["selection_metrics"].get("physical_force_rmse") is not None
        and math.isfinite(float(member["selection_metrics"].get("physical_force_rmse")))
    ]
    if selection_ready:
        best = min(
            selection_ready,
            key=lambda item: (
                _sortable_metric(item["selection_metrics"].get("physical_force_rmse")),
                _sortable_metric(item["selection_metrics"].get("physical_energy_per_atom_rmse")),
                _sortable_metric(item["selection_metrics"].get("aux_force_rmse")),
                _sortable_metric(item.get("final_val_rmse_f")),
                _sortable_metric(item.get("final_val_rmse_e")),
                int(item["model_index"]),
            ),
        )
        return int(best["model_index"])

    successful = [
        member for member in members
        if member.get("status") == "success"
        and member.get("final_val_rmse_f") is not None
        and member.get("final_val_rmse_e") is not None
        and math.isfinite(float(member.get("final_val_rmse_f")))
        and math.isfinite(float(member.get("final_val_rmse_e")))
    ]
    if not successful:
        successful = [
            member for member in members
            if member.get("status") == "success"
            and member.get("final_train_rmse_f") is not None
            and member.get("final_train_rmse_e") is not None
            and math.isfinite(float(member.get("final_train_rmse_f")))
            and math.isfinite(float(member.get("final_train_rmse_e")))
        ]
    if not successful:
        raise RuntimeError("No successful committee members available for driver selection")

    best = min(
        successful,
        key=lambda item: (
            _sortable_metric(item.get("final_val_rmse_f")),
            _sortable_metric(item.get("final_val_rmse_e")),
            _sortable_metric(item.get("final_train_rmse_f")),
            _sortable_metric(item.get("final_train_rmse_e")),
            int(item["model_index"]),
        ),
    )
    return int(best["model_index"])


def write_committee_summary(
    summary_path: str | Path,
    committee_size: int,
    work_root: str | Path,
    split_meta_path: str | Path,
    train_systems: list[str],
    valid_systems: list[str],
    members: list[dict[str, Any]],
) -> Path:
    summary_path = Path(summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    best_driver_index = choose_best_driver_index(members)
    best_member = next(member for member in members if int(member["model_index"]) == best_driver_index)
    normalized_members = []
    for member in members:
        normalized = dict(member)
        model_path = str(Path(str(normalized.get("model_path") or normalized["graph_pb"])).resolve())
        normalized["model_path"] = model_path
        normalized.setdefault("graph_pb", model_path)
        normalized_members.append(normalized)
    best_driver_member = next(member for member in normalized_members if int(member["model_index"]) == best_driver_index)
    best_driver_graph = str(Path(best_driver_member["model_path"]).resolve())

    payload = {
        "committee_size": int(committee_size),
        "work_root": str(Path(work_root).resolve()),
        "split_meta_path": str(Path(split_meta_path).resolve()),
        "train_systems": [str(Path(path).resolve()) for path in train_systems],
        "valid_systems": [str(Path(path).resolve()) for path in valid_systems],
        "members": normalized_members,
        "best_driver_index": int(best_driver_index),
        "best_driver_graph": best_driver_graph,
        "driver_selection_mode": "physical_eval_selection" if isinstance(best_member.get("selection_metrics"), dict) else "legacy_val_rmse",
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    driver_copy_path = summary_path.parent / Path(best_driver_graph).name
    shutil.copy2(best_driver_graph, driver_copy_path)
    return summary_path
