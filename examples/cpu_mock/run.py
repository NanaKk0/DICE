"""Exercise the public lifecycle using synthetic, CPU-only backend doubles."""
from __future__ import annotations

import argparse
from pathlib import Path

from backend_roles import compose_test_backend_roles
from fixtures import _FakeSampler, _FakeStudent, _FakeTeacher, _write_public_run
from workflow_app.entrypoint import main


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--reuse", action="store_true", help="reuse existing demo inputs/results")
    parser.add_argument("--resume", metavar="BLOCK", help="force an exact demo block and suffix")
    args = parser.parse_args()
    root = args.run_directory.resolve()
    config = root / "run.json"
    marker = root / ".dice_mock_example"
    if args.reuse or args.resume:
        if not config.is_file() or not marker.is_file():
            parser.error("reuse/resume requires a directory created by this mock example")
    else:
        if root.exists() and any(root.iterdir()):
            parser.error("choose a new or empty directory; existing files are not overwritten")
        root.mkdir(parents=True, exist_ok=True)
        config = _write_public_run(root)
        marker.write_text("DICE synthetic CPU demonstration; not scientific data.\n", encoding="utf-8")
    argv = [str(config)]
    if args.resume:
        argv.extend(["--resume", args.resume])
    main(argv, backend_composition=compose_test_backend_roles(
        config, teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
    ))
    selected = root / "iter0/fp/selected.xyz"
    if not selected.is_file():
        raise RuntimeError("mock lifecycle did not publish the terminal artifact")
    print(f"Synthetic demonstration complete: {selected}")


if __name__ == "__main__":
    run()
