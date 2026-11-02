from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

from workflow_app.entrypoint import main as app_main


def main(argv: Sequence[str] | None = None) -> None:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        block = arguments[arguments.index("--scheduled-worker") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("scheduled worker requires one public block coordinate") from exc
    app_main(arguments)
    Path("scheduled-worker-receipt.json").write_text(
        json.dumps(
            {
                "schema": "dice.scheduled-worker-receipt",
                "schema_revision": 1,
                "block": block,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

