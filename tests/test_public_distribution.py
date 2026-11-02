from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_installed_public_cli_help():
    result = subprocess.run(["dice", "--help"], text=True, capture_output=True, check=True)
    assert "run.json" in result.stdout
    assert "--probe" in result.stdout
    assert "--resume" in result.stdout


def test_cpu_mock_lifecycle_and_exact_suffix_recovery(tmp_path):
    script = ROOT / "examples/cpu_mock/run.py"
    run_root = tmp_path / "demo"
    for extra in ([], ["--reuse"], ["--resume", "r1/sample"]):
        subprocess.run([sys.executable, str(script), str(run_root), *extra], check=True)
    assert (run_root / "iter0/fp/selected.xyz").is_file()
    status = (run_root / "iter0/status.log").read_text()
    for block in ("teach/train", "teach/screen", "teach/scan", "r0/train", "r0/sample", "r0/screen", "r1/train", "r1/sample", "r1/screen", "fp"):
        assert f"{block} [SUCCESS]" in status


def test_documentation_build_is_local_and_dependency_free(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tools/build_docs.py"), "--output", str(tmp_path / "site")], check=True)
    assert (tmp_path / "site/index.html").is_file()
    assert (tmp_path / "site/docs/recovery.html").is_file()


def test_profile_templates_use_official_loaders():
    from workflow_runtime.lifecycle.environment import load_environment_profile
    from workflow_runtime.execution.profile import load_execution_profile
    environment = load_environment_profile(ROOT / "examples/profiles/environment.json")
    execution = load_execution_profile(ROOT / "examples/profiles/execution.json")
    assert environment.bindings["sampler.lammps.tf"] == "lammps-tf"
    assert execution.resource_classes["cpu"].gpus_per_node == 0


