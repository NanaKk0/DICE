from __future__ import annotations

import sys

from workflow_runtime.launch import (
    build_runtime_root,
    initialize_runtime_layout,
    launch_workflow,
    load_yaml,
)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise RuntimeError("Usage: python workflow.py workflow.yaml")
    launch_workflow(args[0])


if __name__ == "__main__":
    main()
