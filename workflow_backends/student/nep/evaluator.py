from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from workflow_backends.committee_evaluation import canonical_committee_evaluation_result
from workflow_backends.execution_context import BackendExecution
from workflow_backends.common.model_files import link_model_file
from workflow_backends.student.nep.data_writer import read_extxyz_frames, write_prediction_extxyz
from workflow_backends.student.nep.env_check import resolve_nep_command
from workflow_backends.student.nep.input_writer import write_nep_prediction_input
from workflow_backends.student.nep.model_registry import collect_nep_model_paths
from workflow_core.common.io import read_json, write_json
from workflow_runtime.environment import student_backend_role
from workflow_runtime.environment.execution_context import coerce_execution_context


def load_nep_models(summary_path: str | Path) -> list[str]:
    payload = read_json(summary_path)
    return collect_nep_model_paths(payload)


def _nep_config_from_summary(summary_path: str | Path) -> dict[str, Any]:
    payload = read_json(summary_path)
    for key in ("nep_config", "backend_cfg"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    foundation = payload.get("foundation_bootstrap")
    if isinstance(foundation, dict):
        cfg: dict[str, Any] = {}
        for key in ("type_map",):
            if foundation.get(key) is not None:
                cfg[key] = foundation[key]
        arch = foundation.get("arch")
        if isinstance(arch, dict):
            cfg.update(dict(arch))
        return cfg
    return {}


def _parse_member_nep_input(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload: dict[str, Any] = {}
    if not source.is_file():
        return payload
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        keyword = parts[0].lower()
        values = parts[1:]
        if keyword == "type" and values:
            try:
                type_count = int(values[0])
            except ValueError:
                continue
            payload["type_map"] = values[1:1 + type_count]
        elif keyword == "zbl" and values:
            payload["zbl"] = float(values[0])
        elif keyword == "version" and values:
            payload["version"] = int(values[0])
        elif keyword == "cutoff" and len(values) >= 2:
            payload["cutoff"] = {"radial": float(values[0]), "angular": float(values[1])}
        elif keyword == "n_max" and len(values) >= 2:
            payload["n_max"] = {"radial": int(values[0]), "angular": int(values[1])}
        elif keyword == "basis_size" and len(values) >= 2:
            payload["basis_size"] = {"radial": int(values[0]), "angular": int(values[1])}
        elif keyword == "l_max" and values:
            payload["l_max"] = " ".join(values)
        elif keyword == "neuron" and values:
            payload["neuron"] = int(values[0])
    return payload


def _member_prediction_configs(summary_path: str | Path, base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    payload = read_json(summary_path)
    configs: list[dict[str, Any]] = []
    for member in list(payload.get("members", []) or []):
        if not isinstance(member, dict):
            continue
        member_cfg = dict(base_cfg)
        member_nep_in = member.get("nep_in")
        if member_nep_in:
            member_cfg.update(_parse_member_nep_input(member_nep_in))
        configs.append(member_cfg)
    return configs


def _frame_atom_counts(dataset_path: str | Path) -> list[int]:
    return [int(len(atoms)) for atoms in read_extxyz_frames(dataset_path)]


def _parse_force_prediction_rows(force_path: str | Path) -> list[list[float]]:
    force_path = Path(force_path).resolve()
    if not force_path.exists():
        raise FileNotFoundError(f"NEP force prediction file not found: {force_path}")
    rows: list[list[float]] = []
    for line in force_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            raise RuntimeError(f"Invalid NEP force prediction row in {force_path.name}: {line!r}")
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return rows


def _parse_energy_prediction_rows(energy_path: str | Path) -> list[float]:
    energy_path = Path(energy_path).resolve()
    if not energy_path.exists():
        return []
    values: list[float] = []
    for line in energy_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        values.append(float(parts[0]))
    return values


def _reshape_force_rows(rows: list[list[float]], atom_counts: list[int]) -> list[list[list[float]]]:
    expected = sum(int(count) for count in atom_counts)
    if len(rows) != expected:
        raise RuntimeError(f"NEP force prediction row count mismatch: expected {expected}, got {len(rows)}")
    frames: list[list[list[float]]] = []
    cursor = 0
    for atom_count in atom_counts:
        next_cursor = cursor + int(atom_count)
        frames.append(rows[cursor:next_cursor])
        cursor = next_cursor
    return frames


def _compute_frame_uncertainty(forces_by_model: list[list[list[float]]]) -> dict[str, Any]:
    n_models = len(forces_by_model)
    if n_models <= 0:
        return {
            "force_std_max": 0.0,
            "force_std_avg": 0.0,
            "force_std_p90": 0.0,
            "per_atom_force_std": [],
        }
    per_atom_force_std: list[float] = []
    for atom_index in range(len(forces_by_model[0])):
        means = [0.0, 0.0, 0.0]
        for model_index in range(n_models):
            for axis in range(3):
                means[axis] += float(forces_by_model[model_index][atom_index][axis])
        means = [value / n_models for value in means]
        sq_norm = 0.0
        for model_index in range(n_models):
            dx = float(forces_by_model[model_index][atom_index][0]) - means[0]
            dy = float(forces_by_model[model_index][atom_index][1]) - means[1]
            dz = float(forces_by_model[model_index][atom_index][2]) - means[2]
            sq_norm += dx * dx + dy * dy + dz * dz
        per_atom_force_std.append(math.sqrt(sq_norm / n_models))
    sorted_std = sorted(per_atom_force_std)
    if not sorted_std:
        p90 = 0.0
    else:
        p90_index = min(len(sorted_std) - 1, max(0, int(math.ceil(0.9 * len(sorted_std))) - 1))
        p90 = float(sorted_std[p90_index])
    return {
        "force_std_max": float(max(per_atom_force_std)) if per_atom_force_std else 0.0,
        "force_std_avg": float(sum(per_atom_force_std) / len(per_atom_force_std)) if per_atom_force_std else 0.0,
        "force_std_p90": p90,
        "per_atom_force_std": [float(value) for value in per_atom_force_std],
    }


def _run_nep_prediction_for_model(
    dataset_path: str | Path,
    model_path: str | Path,
    nep_cfg: dict[str, Any],
    *,
    work_root: str | Path,
    command: list[str],
    member_index: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    member_dir = Path(work_root).resolve() / f"predict_{member_index:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)
    dataset_target = member_dir / "train.xyz"
    write_prediction_extxyz(dataset_path, dataset_target)
    link_model_file(model_path, member_dir / "nep.txt")
    write_nep_prediction_input(member_dir / "nep.in", nep_cfg)
    stdout_log = member_dir / "nep_predict.stdout.log"
    with open(stdout_log, "w", encoding="utf-8") as handle:
        proc = subprocess.run(
            list(command),
            cwd=member_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    if int(proc.returncode) != 0:
        raise RuntimeError(f"NEP prediction failed for {Path(model_path).name}; see {stdout_log.resolve()}")
    return {
        "member_dir": str(member_dir.resolve()),
        "stdout_log": str(stdout_log.resolve()),
        "force_train_out": str((member_dir / "force_train.out").resolve()),
        "energy_train_out": str((member_dir / "energy_train.out").resolve()),
    }


def evaluate_nep_committee(
    dataset_path: str | Path,
    summary_path: str | Path,
    *,
    nep_cfg: dict[str, Any] | None = None,
    work_root: str | Path | None = None,
    env: dict[str, str] | None = None,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    summary_path = Path(summary_path).resolve()
    request_nep_cfg = dict(nep_cfg or {})
    summary_nep_cfg = _nep_config_from_summary(summary_path)
    request_nep_cfg.update(summary_nep_cfg)
    nep_cfg = request_nep_cfg
    models = load_nep_models(summary_path)
    member_prediction_cfgs = _member_prediction_configs(summary_path, nep_cfg)
    if len(models) < 2:
        raise RuntimeError("At least two NEP committee members are required for committee evaluation")
    if work_root is None:
        work_root = summary_path.parent / "_committee_eval" / dataset_path.stem
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    atom_counts = _frame_atom_counts(dataset_path)
    execution = BackendExecution(
        coerce_execution_context(execution_context),
        role=student_backend_role("nep", "evaluate"),
    )
    if execution.configured:
        command = execution.shell_command([str(execution.required_executable("nep"))])
    else:
        command = resolve_nep_command(nep_cfg)
    model_outputs: list[dict[str, Any]] = []
    frame_forces_by_model: list[list[list[list[float]]]] = []
    frame_energies_by_model: list[list[float]] = []

    for member_index, model_path in enumerate(models):
        member_nep_cfg = (
            member_prediction_cfgs[member_index]
            if member_index < len(member_prediction_cfgs)
            else nep_cfg
        )
        output = _run_nep_prediction_for_model(
            dataset_path,
            model_path,
            member_nep_cfg,
            work_root=work_root,
            command=command,
            member_index=member_index,
            env=env,
        )
        model_outputs.append(output)
        frame_forces_by_model.append(
            _reshape_force_rows(_parse_force_prediction_rows(output["force_train_out"]), atom_counts)
        )
        frame_energies_by_model.append(_parse_energy_prediction_rows(output["energy_train_out"]))

    frame_metrics: list[dict[str, Any]] = []
    for frame_index, atom_count in enumerate(atom_counts):
        forces_for_frame = [model_frames[frame_index] for model_frames in frame_forces_by_model]
        metrics = _compute_frame_uncertainty(forces_for_frame)
        predicted_energies = [
            model_energies[frame_index]
            for model_energies in frame_energies_by_model
            if frame_index < len(model_energies)
        ]
        if predicted_energies:
            metrics["energy_pred_mean"] = float(sum(predicted_energies) / len(predicted_energies))
        metrics["frame_index"] = int(frame_index)
        metrics["natoms"] = int(atom_count)
        frame_metrics.append(metrics)

    has_energy_predictions = any(bool(values) for values in frame_energies_by_model)
    result = canonical_committee_evaluation_result(
        backend="nep",
        committee_size=len(models),
        evaluation_mode="prediction",
        capabilities={
            "force_deviation": True,
            "committee_mean_labels": False,
            "mean_energy": has_energy_predictions,
        },
        frame_metrics=frame_metrics,
        dataset_path=str(dataset_path.resolve()),
        summary_path=str(summary_path.resolve()),
        models=models,
        work_root=str(work_root.resolve()),
        model_outputs=model_outputs,
    )
    write_json(work_root / "committee_eval_summary.json", result)
    return result


__all__ = [
    "evaluate_nep_committee",
    "load_nep_models",
    "_compute_frame_uncertainty",
    "_parse_force_prediction_rows",
    "_reshape_force_rows",
]
