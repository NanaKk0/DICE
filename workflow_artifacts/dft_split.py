"""Import-safe compatibility CLI for splitting legacy DFT extxyz inputs.

The canonical implementation lives in `workflow_artifacts.first_principles.split`.
This module keeps the old import path and script behavior without doing work at
import time.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from workflow_artifacts.first_principles.split import split_dft_xyz


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m workflow_artifacts.dft_split <input.xyz> <out_dir>", file=sys.stderr)
        return 2
    result = split_dft_xyz(args[0], args[1])
    print("Total structures:", result["structure_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
