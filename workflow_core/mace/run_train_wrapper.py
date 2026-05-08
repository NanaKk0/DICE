from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow_core.mace.atomic_number_patch import apply_atomic_number_table_patch
from workflow_core.mace.noise import suppress_known_mace_noise


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    apply_atomic_number_table_patch()
    with suppress_known_mace_noise():
        from mace.cli.run_train import main as mace_main
        mace_main()


if __name__ == "__main__":
    main()

