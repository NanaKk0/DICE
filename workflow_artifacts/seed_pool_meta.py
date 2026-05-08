from __future__ import annotations

from collections import Counter

from workflow_core.common.io import write_json


def update_seed_pool_meta(paths, seed_cfg: dict, seed_pool_atoms: list, update_info: dict) -> str:
    source_round_counter = Counter()
    for atoms in seed_pool_atoms:
        source_round = atoms.info.get("source_round")
        if source_round is None:
            source_round = "initial"
        source_round_counter[str(source_round)] += 1

    payload = {
        "pool_size": len(seed_pool_atoms),
        "max_pool_size_per_system": int(seed_cfg["max_pool_size_per_system"]),
        "source_round_counts": dict(sorted(source_round_counter.items())),
        **update_info,
    }
    return write_json(paths.seed_pool_meta_json(), payload)
