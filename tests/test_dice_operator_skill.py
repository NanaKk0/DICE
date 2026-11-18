"""Exercise the distributed skill outside its source and agent installation paths."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SKILL = PUBLIC_ROOT / "skills" / "dice-operator"


def code_root():
    spec = importlib.util.find_spec("workflow_core")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent


def run_inspector(script, run, cwd, *, source=None, import_root=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in ("DICE_ENVIRONMENT", "DICE_EXECUTION", "PYTHONPATH"):
        env.pop(name, None)
    if import_root is not None:
        env["PYTHONPATH"] = str(import_root)
    command = [sys.executable, str(script), str(run)]
    if source is not None:
        command += ["--repo-root", str(source)]
    return subprocess.run(command, cwd=cwd, env=env, text=True,
                          capture_output=True, timeout=30)


def test_explicit_source_without_internal_agent_files(tmp_path):
    # A public source checkout has the DICE packages; AGENTS.md is optional.
    source = tmp_path / "DICE source"
    source.mkdir()
    for name in ("workflow_app", "workflow_artifacts", "workflow_backends",
                 "workflow_core", "workflow_policies", "workflow_runtime"):
        shutil.copytree(code_root() / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    run_root = tmp_path / "research run"
    run_root.mkdir()
    fixture = PUBLIC_ROOT / "examples/dpa4_dpa4c/run.json"
    run = run_root / "run.json"
    shutil.copyfile(fixture, run)
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    result = run_inspector(SKILL / "scripts/inspect_dice_operation.py", run,
                           unrelated, source=source)
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert output["ok"] is True
    assert output["code_source"] == "explicit-source"
    assert Path(output["repository_root"]) == source
    assert Path(output["run_root"]) == run_root
    assert not (source / "AGENTS.md").exists()
    assert {p.name for p in run_root.iterdir()} == {"run.json"}


@pytest.mark.parametrize("agent_directory", [".agents", ".claude"])
def test_relocated_skill_uses_installed_python_package(tmp_path, agent_directory):
    installed = tmp_path / "agent home" / agent_directory / "skills" / "dice-operator"
    shutil.copytree(SKILL, installed)
    library = tmp_path / "python library"
    library.mkdir()
    for name in ("workflow_app", "workflow_artifacts", "workflow_backends",
                 "workflow_core", "workflow_policies", "workflow_runtime"):
        shutil.copytree(code_root() / name, library / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    run_root = tmp_path / "another run"
    run_root.mkdir()
    run = run_root / "run_iter1.json"
    shutil.copyfile(PUBLIC_ROOT / "examples/dpa4_dpa4c/run_iter1.json", run)
    result = run_inspector(installed / "scripts/inspect_dice_operation.py", run,
                           tmp_path, import_root=library)
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert output["code_source"] == "python-environment"
    assert Path(output["repository_root"]) == library
    assert Path(output["run_root"]) == run_root
    assert output["iter"] == 1
    assert {p.name for p in run_root.iterdir()} == {"run_iter1.json"}


def test_invalid_explicit_source_is_not_silently_replaced(tmp_path):
    run = tmp_path / "run.json"
    shutil.copyfile(PUBLIC_ROOT / "examples/dpa4_dpa4c/run.json", run)
    result = run_inspector(SKILL / "scripts/inspect_dice_operation.py", run,
                           tmp_path, source=tmp_path, import_root=code_root())
    assert result.returncode == 2
    output = json.loads(result.stdout)
    assert output["ok"] is False
    assert "--repo-root is not a DICE checkout" in output["error"]
    assert not (tmp_path / "iter0").exists()


def test_missing_dice_has_actionable_error(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in ("DICE_ENVIRONMENT", "DICE_EXECUTION", "PYTHONPATH"):
        env.pop(name, None)
    # -S models a Python interpreter with no site-installed DICE or .pth hooks.
    result = subprocess.run(
        [sys.executable, "-S", str(SKILL / "scripts/inspect_dice_operation.py"),
         str(tmp_path / "run.json")], cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 2
    output = json.loads(result.stdout)
    assert "DICE is not importable" in output["error"]
    assert "--repo-root" in output["error"]
    assert not list(tmp_path.iterdir())
