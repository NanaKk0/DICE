from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from workflow_backends.execution_context import BackendExecution


def run_deepmd_evaluation_worker(
    *,
    frames_path: str | Path,
    committee_graphs: list[str],
    execution: BackendExecution,
    work_root: str | Path | None = None,
) -> dict[str, Any]:
    python = execution.python_executable() or "python"

    if work_root is None:
        with tempfile.TemporaryDirectory(prefix="deepmd_eval_worker_") as tmpdir:
            return _run_worker_in_dir(Path(tmpdir), frames_path, committee_graphs, execution, str(python))

    root = Path(work_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="deepmd_eval_worker_", dir=str(root)))
    return _run_worker_in_dir(work_dir, frames_path, committee_graphs, execution, str(python))


def _run_worker_in_dir(
    work_dir: Path,
    frames_path: str | Path,
    committee_graphs: list[str],
    execution: BackendExecution,
    python: str,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    config_json = work_dir / "worker_config.json"
    output_json = work_dir / "evaluation_result.json"
    command_json = work_dir / "worker_command.json"
    stdout_log = work_dir / "stdout.log"
    stderr_log = work_dir / "stderr.log"

    config_json.write_text(
        json.dumps(
            {
                "frames_path": str(Path(frames_path).resolve()),
                "committee_graphs": [str(Path(path).resolve()) for path in committee_graphs],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    command = execution.shell_command([
        python,
        "-m",
        "workflow_backends.student.deepmd.evaluation_worker",
        "--config",
        str(config_json),
        "--output",
        str(output_json),
    ])
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
            "DeepMD committee evaluation worker failed "
            f"(returncode={completed.returncode}, command_log={command_json}, stderr_log={stderr_log})"
        )
    return json.loads(output_json.read_text(encoding="utf-8"))


def worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DeepMD committee evaluation in the selected Python runtime.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from workflow_backends.student.deepmd.committee_infer import evaluate_frames_with_committee
    from workflow_backends.student.deepmd.uncertainty import compute_frame_uncertainty

    infer_result = evaluate_frames_with_committee(config["frames_path"], list(config["committee_graphs"]))
    npz_path = Path(infer_result["npz_path"]).resolve()
    try:
        import numpy as np

        payload = np.load(npz_path)
        frame_metrics = [
            compute_frame_uncertainty(payload["forces"][frame_index])
            for frame_index in range(payload["forces"].shape[0])
        ]
    except Exception:
        frame_metrics = []
    output = {**infer_result, "frame_metrics": frame_metrics}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
