from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from workflow_backends.execution_context import BackendExecution
from workflow_backends.teacher.base import TeacherRelabelRequest, TeacherRelabelResult
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


def run_mace_relabel_worker(
    atoms_list: list[Any],
    request: TeacherRelabelRequest,
    execution: BackendExecution,
    core_cfg: dict[str, Any],
) -> TeacherRelabelResult:
    backend_log_dir = Path(request.backend_log_dir).resolve() if request.backend_log_dir is not None else None
    if backend_log_dir is None:
        with tempfile.TemporaryDirectory(prefix="mace_relabel_worker_") as tmpdir:
            return _run_worker_in_dir(Path(tmpdir), atoms_list, execution, dict(core_cfg))

    backend_log_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="mace_relabel_worker_", dir=str(backend_log_dir)))
    return _run_worker_in_dir(work_dir, atoms_list, execution, dict(core_cfg))


def _run_worker_in_dir(
    work_dir: Path,
    atoms_list: list[Any],
    execution: BackendExecution,
    core_cfg: dict[str, Any],
) -> TeacherRelabelResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    input_xyz = work_dir / "input.xyz"
    output_xyz = work_dir / "labeled.xyz"
    uncertainties_json = work_dir / "uncertainties.json"
    config_json = work_dir / "worker_config.json"
    command_json = work_dir / "worker_command.json"
    stdout_log = work_dir / "stdout.log"
    stderr_log = work_dir / "stderr.log"

    write_extxyz_structures(input_xyz, atoms_list, allow_empty=True)
    config_json.write_text(json.dumps(_jsonable(core_cfg), indent=2, sort_keys=True), encoding="utf-8")

    command = execution.shell_command(
        [
            sys.executable,
            "-m",
            "workflow_backends.teacher.mace.relabel_worker",
            "--input",
            str(input_xyz),
            "--output",
            str(output_xyz),
            "--uncertainties",
            str(uncertainties_json),
            "--config",
            str(config_json),
        ]
    )
    command_json.write_text(
        json.dumps(
            {
                "command": command,
                "execution_context": execution.redacted_diagnostics(),
                "work_dir": str(work_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        command,
        env=execution.merge_env(os.environ),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "MACE relabel worker failed "
            f"(returncode={completed.returncode}, command_log={command_json}, stderr_log={stderr_log})"
        )

    labeled = read_extxyz_structures(output_xyz, allow_empty=True)
    uncertainties = np.asarray(json.loads(uncertainties_json.read_text(encoding="utf-8")), dtype=float)
    if len(labeled) != len(uncertainties):
        raise RuntimeError(
            "MACE relabel worker returned mismatched outputs: "
            f"labeled_structures={len(labeled)}, uncertainties={len(uncertainties)}"
        )
    return TeacherRelabelResult(labeled_structures=labeled, uncertainties=uncertainties)


def worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MACE teacher relabel in the selected Python runtime.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncertainties", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    atoms_list = read_extxyz_structures(args.input, allow_empty=True)
    if atoms_list:
        from workflow_backends.teacher.mace import _committee_label, _load_mace_committee

        result = _committee_label(atoms_list, _load_mace_committee(config))
        labeled = result.labeled_structures
        uncertainties = np.asarray(result.uncertainties, dtype=float)
    else:
        labeled = []
        uncertainties = np.asarray([], dtype=float)

    write_extxyz_structures(args.output, labeled, allow_empty=True)
    Path(args.uncertainties).write_text(json.dumps([float(value) for value in uncertainties]), encoding="utf-8")
    return 0


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))


if __name__ == "__main__":
    raise SystemExit(worker_main())
