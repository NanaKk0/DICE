"""
dft_submit_executor.py

Prepare CP2K inputs from structure.xyz files inside task directories and submit them
in grouped batches. This module acts as the DFT run executor for the workflow.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from workflow_backends.first_principles.cp2k.status import cp2k_output_looks_successful
from workflow_backends.first_principles.cp2k.defaults import (
    DEFAULT_DFT_COORDFILE_NAME,
    DEFAULT_DFT_INPUT_NAME,
    DEFAULT_DFT_OUTPUT_NAME,
    DEFAULT_DFT_STRUCTURE_NAME,
)

TASK_DIR_PATTERN = re.compile(r"^(?:task\.)?(\d+)$")
SET_LINE_PATTERN = re.compile(
    r"^(\s*@SET\s+)(COORDFILE|CELL_A|CELL_B|CELL_C|ANGLE_A|ANGLE_B|ANGLE_C|KPOINTS_A|KPOINTS_B|KPOINTS_C)(\s+)(.*?)(\s*)$",
    re.MULTILINE,
)
PLACEHOLDER_WORDS = {
    "coordfile",
    "cell_a",
    "cell_b",
    "cell_c",
    "angle_a",
    "angle_b",
    "angle_c",
    "kpoints_a",
    "kpoints_b",
    "kpoints_c",
    "kpoints_block",
    "basis_set_file",
    "potential_file",
}
CP2K_FILE_DIRECTIVES = (
    "BASIS_SET_FILE_NAME",
    "POTENTIAL_FILE_NAME",
    "PARAMETER_FILE_NAME",
)


def read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_cell_and_angles(struct_path: Path):
    atoms = read(str(struct_path), index=0)
    try:
        a_len, b_len, c_len, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    except Exception:
        cell = atoms.get_cell()
        a_len = float(np.linalg.norm(cell[0]))
        b_len = float(np.linalg.norm(cell[1]))
        c_len = float(np.linalg.norm(cell[2]))

        def angle_between(u, v):
            cu = np.linalg.norm(u)
            cv = np.linalg.norm(v)
            if cu * cv == 0:
                return 90.0
            cosv = float(np.dot(u, v) / (cu * cv))
            cosv = max(-1.0, min(1.0, cosv))
            return math.degrees(math.acos(cosv))

        alpha = angle_between(cell[1], cell[2])
        beta = angle_between(cell[0], cell[2])
        gamma = angle_between(cell[0], cell[1])
    return float(a_len), float(b_len), float(c_len), float(alpha), float(beta), float(gamma)


def compute_reciprocal_lengths(struct_path: Path) -> tuple[float, float, float]:
    atoms = read(str(struct_path), index=0)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"Invalid cell shape from {struct_path}: {cell.shape}")

    a1, a2, a3 = cell[0], cell[1], cell[2]
    volume = float(np.dot(a1, np.cross(a2, a3)))
    if abs(volume) < 1e-12:
        raise ValueError(f"Cell volume is zero or too small in {struct_path}")

    b1 = 2.0 * math.pi * np.cross(a2, a3) / volume
    b2 = 2.0 * math.pi * np.cross(a3, a1) / volume
    b3 = 2.0 * math.pi * np.cross(a1, a2) / volume
    return float(np.linalg.norm(b1)), float(np.linalg.norm(b2)), float(np.linalg.norm(b3))


def write_cp2k_coordfile(struct_path: Path, coord_path: Path) -> Path:
    atoms = read(str(struct_path), index=0)
    lines = []
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        x, y, z = position
        lines.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f}")
    write_text(coord_path, "\n".join(lines) + "\n")
    return coord_path


def _template_uses_xyz_coordfile(template_text: str) -> bool:
    upper = template_text.upper()
    return "COORD_FILE_FORMAT XYZ" in upper and "COORD_FILE_NAME" in upper


def _extract_template_static_file_refs(template_text: str) -> list[str]:
    refs: list[str] = []
    for directive in CP2K_FILE_DIRECTIVES:
        pattern = re.compile(rf"^\s*{directive}\s+(\S+)", re.MULTILINE | re.IGNORECASE)
        for match in pattern.finditer(template_text):
            candidate = str(match.group(1)).strip()
            if not candidate or candidate in refs:
                continue
            if candidate in PLACEHOLDER_WORDS:
                continue
            if candidate.startswith("${") and candidate.endswith("}"):
                continue
            refs.append(candidate)
    return refs


def _resolve_template_static_file(
    filename: str,
    cfg: dict[str, Any],
    basis_set_file: str | None,
    potential_file: str | None,
) -> Path | None:
    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    lookup_paths: list[Path] = []
    if basis_set_file:
        basis_path = Path(basis_set_file).resolve()
        if basis_path.name == filename:
            lookup_paths.append(basis_path)
    if potential_file:
        potential_path = Path(potential_file).resolve()
        if potential_path.name == filename:
            lookup_paths.append(potential_path)

    cp2k_data_dir = cfg.get("cp2k_data_dir")
    if cp2k_data_dir:
        lookup_paths.append(Path(str(cp2k_data_dir)).resolve() / filename)

    for path in lookup_paths:
        if path.exists():
            return path.resolve()
    return None


def _stage_template_static_files(
    task_dir: Path,
    template_text: str,
    cfg: dict[str, Any],
    basis_set_file: str | None,
    potential_file: str | None,
) -> None:
    for filename in _extract_template_static_file_refs(template_text):
        source = _resolve_template_static_file(filename, cfg, basis_set_file, potential_file)
        if source is None:
            raise RuntimeError(
                "Template references static CP2K file "
                f"{filename!r}, but the workflow could not resolve it from cp2k_data_dir "
                "or configured runtime files."
            )
        destination = task_dir / filename
        if destination.resolve() == source:
            continue
        shutil.copy2(source, destination)


def _stage_coord_input(
    struct_path: Path,
    task_dir: Path,
    structure_name: str,
    coordfile_name: str,
    template_text: str,
) -> str:
    if _template_uses_xyz_coordfile(template_text):
        if coordfile_name == structure_name:
            return structure_name
        destination = task_dir / coordfile_name
        shutil.copy2(struct_path, destination)
        return coordfile_name

    write_cp2k_coordfile(struct_path, task_dir / coordfile_name)
    return coordfile_name


def _resolve_coordfile_name(template_text: str, structure_name: str, coordfile_name: str) -> str:
    if _template_uses_xyz_coordfile(template_text):
        return structure_name
    return coordfile_name


def _validate_kpoint_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    validated = {
        "kpoint_mode": str(cfg.get("kpoint_mode", "slab_xy")).lower(),
        "kspacing": float(cfg.get("kspacing", 0.20)),
        "kgrid": cfg.get("kgrid"),
        "vacuum_axis": str(cfg.get("vacuum_axis", "c")).lower(),
        "kpoint_symmetry": bool(cfg.get("kpoint_symmetry", False)),
        "kpoint_parallel_group_size": cfg.get("kpoint_parallel_group_size"),
        "gamma_real_wavefunctions": bool(cfg.get("gamma_real_wavefunctions", True)),
        "cp2k_output_name": DEFAULT_DFT_OUTPUT_NAME,
    }
    allowed_modes = {"gamma", "slab_xy", "bulk_3d", "manual", "auto"}
    if validated["kpoint_mode"] not in allowed_modes:
        raise ValueError(f"Unsupported kpoint_mode={validated['kpoint_mode']!r}")

    allowed_axes = {"a", "b", "c", "auto"}
    if validated["vacuum_axis"] not in allowed_axes:
        raise ValueError(f"Unsupported vacuum_axis={validated['vacuum_axis']!r}")

    if validated["kpoint_mode"] != "gamma":
        if validated["kspacing"] <= 0:
            raise ValueError("kspacing must be > 0 for non-gamma k-point modes")
        if validated["kspacing"] >= 5.0:
            print(
                "[DFT] Warning: kspacing is very large "
                f"({validated['kspacing']}); k-grid may collapse to 1x1x1."
            )

    if validated["kpoint_mode"] == "manual":
        kgrid = validated.get("kgrid")
        if not isinstance(kgrid, list) or len(kgrid) != 3:
            raise ValueError("manual kpoint_mode requires kgrid=[ka, kb, kc]")
        normalized_kgrid = []
        for item in kgrid:
            kval = int(item)
            if kval < 1:
                raise ValueError("manual kgrid values must all be >= 1")
            normalized_kgrid.append(kval)
        validated["kgrid"] = normalized_kgrid

    if validated["kpoint_parallel_group_size"] is not None:
        value = int(validated["kpoint_parallel_group_size"])
        if value < 1:
            raise ValueError("kpoint_parallel_group_size must be >= 1 when provided")
        validated["kpoint_parallel_group_size"] = value

    return validated


def _k_from_reciprocal_length(reciprocal_length: float, kspacing: float) -> int:
    if reciprocal_length <= 0:
        return 1
    return int(math.ceil(max(reciprocal_length / kspacing, 1.0)))


def _infer_vacuum_axis(a_len: float, b_len: float, c_len: float, cfg: dict[str, Any]) -> str | None:
    vacuum_axis = str(cfg.get("vacuum_axis", "c")).lower()
    if vacuum_axis in {"a", "b", "c"}:
        return vacuum_axis

    lengths = {"a": float(a_len), "b": float(b_len), "c": float(c_len)}
    min_len = min(lengths.values())
    max_axis, max_len = max(lengths.items(), key=lambda item: item[1])
    if min_len <= 0:
        return max_axis
    if max_len / min_len < 1.5:
        return None
    return max_axis


def _compute_kgrid(
    a_len: float,
    b_len: float,
    c_len: float,
    reciprocal_lengths: tuple[float, float, float],
    cfg: dict[str, Any],
) -> tuple[int, int, int]:
    mode = str(cfg["kpoint_mode"]).lower()
    if mode == "gamma":
        return 1, 1, 1
    if mode == "manual":
        k_a, k_b, k_c = cfg["kgrid"]
        return int(k_a), int(k_b), int(k_c)
    g_a, g_b, g_c = reciprocal_lengths
    if mode == "slab_xy":
        return _k_from_reciprocal_length(g_a, cfg["kspacing"]), _k_from_reciprocal_length(g_b, cfg["kspacing"]), 1
    if mode == "bulk_3d":
        return (
            _k_from_reciprocal_length(g_a, cfg["kspacing"]),
            _k_from_reciprocal_length(g_b, cfg["kspacing"]),
            _k_from_reciprocal_length(g_c, cfg["kspacing"]),
        )

    if mode == "auto":
        vacuum_axis = _infer_vacuum_axis(a_len, b_len, c_len, cfg)
        if vacuum_axis is None:
            return (
                _k_from_reciprocal_length(g_a, cfg["kspacing"]),
                _k_from_reciprocal_length(g_b, cfg["kspacing"]),
                _k_from_reciprocal_length(g_c, cfg["kspacing"]),
            )
        axis_lengths = {"a": g_a, "b": g_b, "c": g_c}
        result = {}
        for axis, reciprocal_length in axis_lengths.items():
            result[axis] = 1 if axis == vacuum_axis else _k_from_reciprocal_length(reciprocal_length, cfg["kspacing"])
        return int(result["a"]), int(result["b"]), int(result["c"])

    raise ValueError(f"Unsupported kpoint_mode={mode!r}")


def _build_kpoints_block(k_a: int, k_b: int, k_c: int, cfg: dict[str, Any]) -> str:
    mode = str(cfg["kpoint_mode"]).lower()
    lines = ["    &KPOINTS"]
    if mode == "gamma":
        lines.append("      SCHEME GAMMA")
        if bool(cfg.get("gamma_real_wavefunctions", True)):
            lines.append("      WAVEFUNCTIONS REAL")
    else:
        lines.append(f"      SCHEME MONKHORST-PACK {k_a} {k_b} {k_c}")

    if bool(cfg.get("kpoint_symmetry", False)):
        lines.append("      SYMMETRY")
    parallel_group_size = cfg.get("kpoint_parallel_group_size")
    if parallel_group_size is not None:
        lines.append(f"      PARALLEL_GROUP_SIZE {int(parallel_group_size)}")
    lines.append("    &END KPOINTS")
    return "\n".join(lines)


def fill_template_set_lines(template_text: str, values: dict[str, Any]) -> str:
    key_map = {
        "COORDFILE": "coordfile",
        "CELL_A": "cell_a",
        "CELL_B": "cell_b",
        "CELL_C": "cell_c",
        "ANGLE_A": "angle_a",
        "ANGLE_B": "angle_b",
        "ANGLE_C": "angle_c",
        "KPOINTS_A": "kpoints_a",
        "KPOINTS_B": "kpoints_b",
        "KPOINTS_C": "kpoints_c",
    }

    def repl(match: re.Match[str]) -> str:
        prefix, set_key, sep, _oldval, suffix = match.groups()
        value_key = key_map.get(set_key)
        if value_key is None or value_key not in values:
            return match.group(0)
        return f"{prefix}{set_key}{sep}{values[value_key]}{suffix}"

    return SET_LINE_PATTERN.sub(repl, template_text)


def iter_task_dirs(jobdir: Path) -> list[tuple[int, Path]]:
    tasks: list[tuple[int, Path]] = []
    for child in sorted(jobdir.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        match = TASK_DIR_PATTERN.match(child.name)
        if not match:
            continue
        tasks.append((int(match.group(1)), child))
    return tasks


def prepare_inputs_for_job(
    jobdir: Path,
    template_text: str,
    modified_name: str,
    structure_name: str,
    coordfile_name: str,
    cfg: dict[str, Any],
) -> list[tuple[int, Path, Path]]:
    tasks = []
    k_cfg = _validate_kpoint_cfg(cfg)
    cp2k_data_dir = cfg.get("cp2k_data_dir")
    basis_set_file = cfg.get("basis_set_file")
    potential_file = cfg.get("potential_file")
    if basis_set_file is None and cp2k_data_dir:
        basis_set_file = str((Path(str(cp2k_data_dir)) / "BASIS_MOLOPT").as_posix())
    if potential_file is None and cp2k_data_dir:
        potential_file = str((Path(str(cp2k_data_dir)) / "POTENTIAL").as_posix())
    resolved_coordfile_name = _resolve_coordfile_name(template_text, structure_name, coordfile_name)
    for idx, child in iter_task_dirs(jobdir):
        struct_path = child / structure_name
        if not struct_path.exists():
            continue
        a_len, b_len, c_len, alpha, beta, gamma = compute_cell_and_angles(struct_path)
        reciprocal_lengths = compute_reciprocal_lengths(struct_path)
        k_a, k_b, k_c = _compute_kgrid(a_len, b_len, c_len, reciprocal_lengths, k_cfg)
        kpoints_block = _build_kpoints_block(k_a, k_b, k_c, k_cfg)
        coord_token = _stage_coord_input(struct_path, child, structure_name, resolved_coordfile_name, template_text)
        _stage_template_static_files(child, template_text, cfg, basis_set_file, potential_file)
        placeholder_map = {
            "cell_a": f"{a_len:.10f}",
            "cell_b": f"{b_len:.10f}",
            "cell_c": f"{c_len:.10f}",
            "angle_a": f"{alpha:.8f}",
            "angle_b": f"{beta:.8f}",
            "angle_c": f"{gamma:.8f}",
            "coordfile": coord_token,
            "kpoints_a": str(k_a),
            "kpoints_b": str(k_b),
            "kpoints_c": str(k_c),
            "kpoints_block": kpoints_block,
        }
        if basis_set_file:
            placeholder_map["basis_set_file"] = str(basis_set_file)
        if potential_file:
            placeholder_map["potential_file"] = str(potential_file)
        out_path = child / modified_name
        rendered = fill_template_set_lines(template_text, placeholder_map)
        if basis_set_file:
            rendered = rendered.replace("basis_set_file", str(basis_set_file))
        if potential_file:
            rendered = rendered.replace("potential_file", str(potential_file))
        rendered = rendered.replace("kpoints_block", kpoints_block)
        write_text(out_path, rendered)
        tasks.append((idx, child, out_path))
    return tasks


def _task_status_path(task_dir: Path) -> Path:
    return task_dir / "task_status.json"


def _load_task_status(task_dir: Path) -> dict[str, Any] | None:
    path = _task_status_path(task_dir)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def task_is_finished(task_dir: Path, cfg: dict[str, Any]) -> bool:
    status = _load_task_status(task_dir)
    if status and status.get("status") == "success":
        return True
    return cp2k_output_looks_successful(task_dir, cfg)


def write_task_status(task_dir: Path, payload: dict[str, Any]) -> None:
    write_json(_task_status_path(task_dir), payload)


def _normalize_shell_prelude_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def run_group(
    tasks_group: list[tuple[int, Path, Path]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    command_template = str(cfg["command"])
    source_list = _normalize_source_list(cfg.get("source_list"))
    shell_prelude_commands = _normalize_shell_prelude_commands(cfg.get("shell_prelude_commands"))
    modified_name = DEFAULT_DFT_INPUT_NAME
    log_level = str(cfg.get("log_level", "medium")).lower()
    nproc_single = int(cfg["nproc_single"])
    use_mpirun = bool(cfg.get("use_mpirun", True))
    dry_run = bool(cfg.get("dry_run", False))

    procs = []
    results = []
    for idx, subdir, _input_path in tasks_group:
        quoted_input = shlex.quote(modified_name)
        cmd_body = command_template.replace("{input}", quoted_input).replace("{nproc_single}", str(nproc_single))
        cmd_strip = cmd_body.strip()
        if use_mpirun and nproc_single > 1 and not cmd_strip.lower().startswith("mpirun"):
            full_cmd = f"mpirun -np {nproc_single} {cmd_body}"
        else:
            full_cmd = cmd_body
        shell_parts: list[str] = []
        if source_list:
            shell_parts.extend([f". {shlex.quote(item)}" for item in source_list])
        shell_parts.extend(shell_prelude_commands)
        shell_parts.append(full_cmd)
        shell_cmd = " && ".join(part for part in shell_parts if part)

        runlog = subdir / "run.log"
        start_time = time.time()
        if dry_run:
            result = {
                "idx": idx,
                "subdir": str(subdir),
                "runlog": str(runlog),
                "returncode": 0,
                "status": "prepared",
                "start_time": start_time,
                "end_time": start_time,
                "elapsed_s": 0.0,
                "command_executed": shell_cmd,
            }
            write_task_status(
                subdir,
                {
                    "idx": idx,
                    "status": "prepared",
                    "returncode": 0,
                    "start_time": start_time,
                    "end_time": start_time,
                    "elapsed_s": 0.0,
                    "command_executed": shell_cmd,
                    "runlog": str(runlog),
                },
            )
            results.append(result)
            continue

        f = open(str(runlog), "wb")
        if os.name == "nt":
            proc = subprocess.Popen(
                shell_cmd,
                shell=True,
                cwd=str(subdir),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        else:
            proc = subprocess.Popen(
                ["bash", "-lc", shell_cmd],
                cwd=str(subdir),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        procs.append((idx, subdir, proc, f, runlog, start_time, shell_cmd))

    for idx, subdir, proc, fhandle, runlog, start_time, shell_cmd in procs:
        ret = proc.wait()
        end_time = time.time()
        fhandle.close()
        status = "success" if ret == 0 else "failure"
        result = {
            "idx": idx,
            "subdir": str(subdir),
            "runlog": str(runlog),
            "returncode": int(ret),
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_s": end_time - start_time,
            "command_executed": shell_cmd,
        }
        write_task_status(
            subdir,
            {
                "idx": idx,
                "status": status,
                "returncode": int(ret),
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_s": end_time - start_time,
                "command_executed": shell_cmd,
                "runlog": str(runlog),
            },
        )
        results.append(result)

    return results


def _normalize_source_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def submit_module(cfg: dict[str, Any]) -> dict[str, Any]:
    jobid = str(cfg["jobid"])
    template_path = Path(cfg["template_path"]).resolve()
    input_dir = Path(cfg.get("input_dir", ".")).resolve()
    output_dir = Path(cfg.get("output_dir", ".")).resolve()
    modified_name = DEFAULT_DFT_INPUT_NAME
    nproc_total = int(cfg.get("nproc_total", 64))
    nproc_single = int(cfg.get("nproc_single", 4))
    group_size = cfg.get("group_size")
    task_size = cfg.get("task_size")
    if group_size is not None:
        effective_group_size = int(group_size)
    elif task_size is not None:
        effective_group_size = int(task_size)
    else:
        effective_group_size = max(1, nproc_total // nproc_single)

    structure_name = DEFAULT_DFT_STRUCTURE_NAME
    coordfile_name = DEFAULT_DFT_COORDFILE_NAME
    source_list = _normalize_source_list(cfg.get("source_list"))
    command = cfg.get("command")
    if not command:
        raise ValueError("Please provide 'command' in JSON")

    skip_finished = bool(cfg.get("skip_finished", True))
    retry_failed = bool(cfg.get("retry_failed", False))
    log_level = str(cfg.get("log_level", "medium")).lower()
    dry_run = bool(cfg.get("dry_run", False))

    jobdir = input_dir / jobid
    if not jobdir.exists():
        raise ValueError(f"Job directory not found: {jobdir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    template_text = read_template(template_path)
    tasks = prepare_inputs_for_job(jobdir, template_text, modified_name, structure_name, coordfile_name, cfg)
    if not tasks:
        raise RuntimeError("No tasks prepared")

    log_path = output_dir / cfg.get("log_path", "submit.log")
    submit_summary_path = output_dir / "submit_summary.json"
    start_all = time.time()

    runnable: list[tuple[int, Path, Path]] = []
    skipped_results: list[dict[str, Any]] = []
    for idx, subdir, input_path in tasks:
        status = _load_task_status(subdir)
        if skip_finished and task_is_finished(subdir, cfg):
            skipped_results.append(
                {
                    "idx": idx,
                    "subdir": str(subdir),
                    "runlog": str(subdir / "run.log"),
                    "returncode": 0,
                    "status": "skipped",
                    "command_executed": None,
                }
            )
            continue
        if status and status.get("status") == "failure" and not retry_failed:
            skipped_results.append(
                {
                    "idx": idx,
                    "subdir": str(subdir),
                    "runlog": str(subdir / "run.log"),
                    "returncode": int(status.get("returncode", 1)),
                    "status": "skipped_failed",
                    "command_executed": status.get("command_executed"),
                }
            )
            continue
        runnable.append((idx, subdir, input_path))

    groups = [runnable[i : i + effective_group_size] for i in range(0, len(runnable), effective_group_size)]

    log_lines = [f"Submit run for jobid={jobid} tasks={len(tasks)} start={time.ctime(start_all)}"]
    overall_results: list[dict[str, Any]] = list(skipped_results)
    if log_level in ("medium", "high"):
        print(log_lines[-1])

    total_tasks = len(tasks)
    completed_so_far = len(skipped_results)
    success_so_far = sum(1 for item in skipped_results if item["status"] == "skipped")
    failure_so_far = sum(1 for item in skipped_results if item["status"] == "skipped_failed")
    skipped_so_far = len(skipped_results)
    if skipped_results and log_level in ("medium", "high"):
        print(
            "[DFT] Initial skipped tasks: "
            f"{skipped_so_far}/{total_tasks} | "
            f"success_like={success_so_far} failed_skip={failure_so_far}"
        )

    for group_index, group in enumerate(groups, start=1):
        if log_level in ("medium", "high"):
            print(f"Submitting group {group_index}/{len(groups)} tasks {[g[0] for g in group]}")
        results = run_group(group, cfg)
        overall_results.extend(results)
        succ = sum(1 for item in results if item["status"] == "success")
        fail = sum(1 for item in results if item["status"] == "failure")
        log_lines.append(
            f"Group {group_index}: succ {succ}/{len(results)} fail {fail} tasks={[r['idx'] for r in results]}"
        )
        if log_level in ("medium", "high"):
            for item in results:
                completed_so_far += 1
                if item["status"] == "success":
                    success_so_far += 1
                elif item["status"] == "failure":
                    failure_so_far += 1
                task_name = Path(item["subdir"]).name
                elapsed_s = float(item.get("elapsed_s", 0.0))
                print(
                    "[DFT] Progress "
                    f"{completed_so_far}/{total_tasks} | "
                    f"success={success_so_far} fail={failure_so_far} skipped={skipped_so_far} | "
                    f"{task_name} -> {item['status']} ({elapsed_s:.1f}s)"
                )

    end_all = time.time()
    n_success = sum(1 for item in overall_results if item["status"] == "success")
    n_failure = sum(1 for item in overall_results if item["status"] == "failure")
    n_skipped = sum(1 for item in overall_results if item["status"] in {"skipped", "skipped_failed"})
    n_submitted = len(runnable)

    log_lines.append(
        f"Submit finished: total={len(tasks)} submitted={n_submitted} skipped={n_skipped} success={n_success} fail={n_failure} walltime_s={(end_all-start_all):.2f}"
    )
    write_text(log_path, "\n".join(log_lines))

    summary = {
        "jobid": jobid,
        "n_total": len(tasks),
        "n_submitted": n_submitted,
        "n_skipped": n_skipped,
        "n_success": n_success,
        "n_failure": n_failure,
        "groups": [[task[0] for task in group] for group in groups],
        "tasks": overall_results,
        "log_path": str(log_path.resolve()),
        "submit_summary_json": str(submit_summary_path.resolve()),
        "dry_run": dry_run,
    }
    write_json(submit_summary_path, summary)
    print(f"Log written to {log_path}")
    print(f"Done. success={n_success} fail={n_failure} total={len(tasks)}")
    return summary

