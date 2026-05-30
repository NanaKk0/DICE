from __future__ import annotations

import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import Atoms

from workflow_backends.execution_context import BackendExecution, normalize_backend_request_context
from workflow_backends.teacher.base import TeacherModelArtifact, TeacherRelabelRequest, TeacherRelabelResult
from workflow_backends.teacher.committee_labeling import label_with_committee
from workflow_backends.teacher.deepmd.family import get_deepmd_family_adapter
from workflow_core.structures.formats import atoms_to_lammps_data
from workflow_runtime.environment import teacher_backend_role


BAR_TO_EV_PER_A3 = 6.241509074e-7


@dataclass(slots=True)
class LammpsRelabelTaskSpec:
    model_index: int
    model_path: Path
    teacher_dir: Path
    group_index: int
    frame_indices: list[int]
    atoms_list: list[Atoms]
    task_dir: Path
    conf_path: Path
    frames_path: Path
    input_path: Path
    log_path: Path
    thermo_path: Path
    forces_path: Path
    model_name: str
    type_map: list[str]
    lammps_cfg: dict[str, Any]
    lammps_command: list[str]
    execution: BackendExecution


@dataclass(slots=True)
class LammpsRelabelTaskResult:
    task_dir: Path
    returncode: int
    log_path: Path
    thermo_path: Path
    forces_path: Path
    payloads: list[dict[str, Any]]
    stdout: str = ""
    stderr: str = ""


class _SequenceEvaluator:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self._cursor = 0

    def evaluate(self, atoms: Any) -> dict[str, Any]:
        del atoms
        if self._cursor >= len(self._payloads):
            raise IndexError("committee evaluator exhausted payloads")
        payload = dict(self._payloads[self._cursor])
        self._cursor += 1
        return payload


def _reject_retired_lammps_execution_fields(lammps_cfg: dict[str, Any]) -> None:
    retired = {"command", "env", "plugin_path"}
    present = sorted(set(lammps_cfg) & retired)
    if present:
        raise ValueError(
            "DeepMD/DPA4 LAMMPS relabel no longer accepts lammps command/env/plugin_path fields; "
            f"move them to environment_profile executables/source scripts: {present}"
        )


def relabel_structures_with_python(
    atoms_list: list[Any],
    request: TeacherRelabelRequest,
    *,
    model_family: str,
    type_map: list[str],
) -> TeacherRelabelResult:
    teacher_dir = Path(request.committee_dir).resolve()
    models = _collect_teacher_models(teacher_dir, model_family=model_family)
    if request.teacher_num is not None and int(request.teacher_num) != len(models):
        raise RuntimeError(
            f"DeePMD teacher_num mismatch: requested {request.teacher_num}, discovered {len(models)} models"
        )
    if not models:
        raise RuntimeError(f"No DeePMD teacher models found in {teacher_dir}")

    options = dict(request.backend_options or {})
    batch_size = options.get("batch_size")
    batch_size = None if batch_size is None else int(batch_size)
    from workflow_backends.teacher.deepmd import build_deepmd_evaluator

    evaluators = [
        build_deepmd_evaluator(
            artifact.model_path,
            device=request.device,
            batch_size=batch_size,
            model_family=model_family,
            type_map=type_map,
        )
        for artifact in models
    ]
    return label_with_committee(atoms_list, evaluators)


def relabel_structures_with_lammps(
    atoms_list: list[Any],
    request: TeacherRelabelRequest,
    *,
    model_family: str,
    type_map: list[str],
    runner=None,
) -> TeacherRelabelResult:
    teacher_dir = Path(request.committee_dir).resolve()
    models = _collect_teacher_models(teacher_dir, model_family=model_family)
    if request.teacher_num is not None and int(request.teacher_num) != len(models):
        raise RuntimeError(
            f"DeePMD teacher_num mismatch: requested {request.teacher_num}, discovered {len(models)} models"
        )
    if not models:
        raise RuntimeError(f"No DeePMD teacher models found in {teacher_dir}")

    options = dict(request.backend_options or {})
    lammps_cfg = dict(options.get("lammps", {}) or {})
    _reject_retired_lammps_execution_fields(lammps_cfg)
    max_parallel_tasks = max(1, int(lammps_cfg.get("max_parallel_tasks", 1) or 1))
    backend_log_dir = Path(request.backend_log_dir).resolve() if request.backend_log_dir is not None else None
    payloads_by_model: list[list[dict[str, Any]]] = []
    execution = normalize_backend_request_context(request, role=teacher_backend_role("deepmd", "relabel")).execution
    lammps_command = execution.shell_command([execution.required_executable("lmp"), "-in", "input.lammps"])

    with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
        futures = [
            executor.submit(
                _evaluate_committee_member,
                atoms_list,
                artifact=artifact,
                model_index=model_index,
                model_family=model_family,
                type_map=type_map,
                lammps_cfg=lammps_cfg,
                lammps_command=lammps_command,
                execution=execution,
                backend_log_dir=backend_log_dir,
                max_parallel_tasks=max_parallel_tasks,
                runner=runner,
            )
            for model_index, artifact in enumerate(models)
        ]
        for future in futures:
            payloads_by_model.append(future.result())

    evaluators = [_SequenceEvaluator(payloads) for payloads in payloads_by_model]
    return label_with_committee(atoms_list, evaluators)


def _evaluate_committee_member(
    atoms_list: list[Any],
    *,
    artifact: TeacherModelArtifact,
    model_index: int,
    model_family: str,
    type_map: list[str],
    lammps_cfg: dict[str, Any],
    lammps_command: list[str],
    execution: BackendExecution,
    backend_log_dir: Path | None,
    max_parallel_tasks: int,
    runner=None,
) -> list[dict[str, Any]]:
    grouped = group_rerun_compatible_structures(atoms_list)
    per_frame_payloads: list[dict[str, Any] | None] = [None] * len(atoms_list)
    base_dir = (
        backend_log_dir / "deepmd_lammps_relabel" / f"teacher_{model_index + 1:02d}"
        if backend_log_dir is not None
        else None
    )
    for group_index, group in enumerate(grouped):
        group_payloads = _evaluate_group_or_fallback(
            group,
            artifact=artifact,
            model_index=model_index,
            group_index=group_index,
            model_family=model_family,
            type_map=type_map,
            lammps_cfg=lammps_cfg,
            lammps_command=lammps_command,
            execution=execution,
            backend_log_dir=base_dir,
            runner=runner,
            max_parallel_tasks=max_parallel_tasks,
        )
        for frame_index, payload in group_payloads:
            per_frame_payloads[frame_index] = payload
    if any(payload is None for payload in per_frame_payloads):
        raise RuntimeError("DPA4 LAMMPS relabel lost frame alignment while assembling committee payloads")
    return [dict(payload) for payload in per_frame_payloads if payload is not None]


def _evaluate_group_or_fallback(
    group: list[tuple[int, Any]],
    *,
    artifact: TeacherModelArtifact,
    model_index: int,
    group_index: int,
    model_family: str,
    type_map: list[str],
    lammps_cfg: dict[str, Any],
    lammps_command: list[str],
    execution: BackendExecution,
    backend_log_dir: Path | None,
    runner=None,
    max_parallel_tasks: int,
) -> list[tuple[int, dict[str, Any]]]:
    group_dir = None if backend_log_dir is None else backend_log_dir / f"group_{group_index:06d}"
    if group_dir is not None:
        group_dir.mkdir(parents=True, exist_ok=True)
    try:
        task_result = run_lammps_relabel_task(
            _build_task_spec(
                group,
                artifact=artifact,
                model_index=model_index,
                group_index=group_index,
                model_family=model_family,
                type_map=type_map,
                lammps_cfg=lammps_cfg,
                lammps_command=lammps_command,
                execution=execution,
                backend_log_dir=group_dir,
            ),
            lammps_cfg=lammps_cfg,
            runner=runner,
        )
        return _payloads_from_task_result(task_result, group)
    except Exception:
        fallback: list[tuple[int, dict[str, Any]]] = []
        for fallback_index, (frame_index, atoms) in enumerate(group):
            single_dir = None if group_dir is None else group_dir / f"frame_{fallback_index:06d}"
            if single_dir is not None:
                single_dir.mkdir(parents=True, exist_ok=True)
            task_result = run_lammps_relabel_task(
                _build_task_spec(
                    [(frame_index, atoms)],
                    artifact=artifact,
                    model_index=model_index,
                    group_index=group_index,
                    model_family=model_family,
                    type_map=type_map,
                    lammps_cfg=lammps_cfg,
                    lammps_command=lammps_command,
                    execution=execution,
                    backend_log_dir=single_dir,
                ),
                lammps_cfg=lammps_cfg,
                runner=runner,
            )
            payloads = _payloads_from_task_result(task_result, [(frame_index, atoms)])
            if len(payloads) != 1:
                raise RuntimeError("single-frame fallback returned an unexpected payload count")
            fallback.extend(payloads)
        return fallback


def _payloads_from_task_result(
    task_result: LammpsRelabelTaskResult,
    group: list[tuple[int, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    payloads = []
    if int(task_result.returncode) != 0:
        raise RuntimeError(f"LAMMPS relabel task failed with returncode={task_result.returncode}")
    parsed = parse_lammps_relabel_outputs(
        task_result.log_path,
        task_result.forces_path,
        expected_frame_count=len(group),
        natoms=len(group[0][1]),
    )
    if len(parsed) != len(group):
        raise RuntimeError(
            f"LAMMPS relabel parser returned {len(parsed)} payloads for {len(group)} expected frames"
        )
    for (frame_index, _), payload in zip(group, parsed):
        payloads.append((frame_index, payload))
    return payloads


def _build_task_spec(
    group: list[tuple[int, Any]],
    *,
    artifact: TeacherModelArtifact,
    model_index: int,
    group_index: int,
    model_family: str,
    type_map: list[str],
    lammps_cfg: dict[str, Any],
    lammps_command: list[str],
    execution: BackendExecution,
    backend_log_dir: Path | None,
) -> LammpsRelabelTaskSpec:
    frames = [atoms for _, atoms in group]
    if not frames:
        raise ValueError("Cannot build a LAMMPS relabel task for an empty group")
    group_dir = backend_log_dir if backend_log_dir is not None else Path(tempfile.mkdtemp(prefix="dpa4_lammps_relabel_")).resolve()
    group_dir.mkdir(parents=True, exist_ok=True)
    conf_path = _write_conf_lammps_data(frames[0], group_dir / "conf.lmp", type_map=type_map)
    frames_path = _write_lammpstrj_frames(group_dir / "frames.lammpstrj", frames, type_map=type_map)
    input_path = _write_lammps_relabel_input(
        group_dir / "input.lammps",
        conf_path=conf_path,
        frames_path=frames_path,
        model_path=artifact.model_path,
        type_map=type_map,
        lammps_cfg=lammps_cfg,
    )
    return LammpsRelabelTaskSpec(
        model_index=model_index,
        model_path=Path(artifact.model_path).resolve(),
        teacher_dir=Path(artifact.metadata.get("member_dir", artifact.model_path)).resolve(),
        group_index=group_index,
        frame_indices=[frame_index for frame_index, _ in group],
        atoms_list=frames,
        task_dir=group_dir,
        conf_path=conf_path,
        frames_path=frames_path,
        input_path=input_path,
        log_path=group_dir / "log.lammps",
        thermo_path=group_dir / "thermo.out",
        forces_path=group_dir / "forces.dump",
        model_name=Path(artifact.model_path).name,
        type_map=list(type_map),
        lammps_cfg=dict(lammps_cfg),
        lammps_command=list(lammps_command),
        execution=execution,
    )


def run_lammps_relabel_task(
    task: LammpsRelabelTaskSpec,
    *,
    lammps_cfg: dict[str, Any],
    runner=None,
) -> LammpsRelabelTaskResult:
    if runner is not None:
        return runner(task, lammps_cfg=lammps_cfg)

    completed = subprocess.run(
        task.lammps_command,
        cwd=task.task_dir,
        text=True,
        capture_output=True,
        env=task.execution.merge_env(),
        check=False,
    )
    task.log_path.write_text(str(completed.stdout or "") + str(completed.stderr or ""), encoding="utf-8")
    return LammpsRelabelTaskResult(
        task_dir=task.task_dir,
        returncode=int(completed.returncode),
        log_path=task.log_path,
        thermo_path=task.thermo_path,
        forces_path=task.forces_path,
        payloads=[],
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
    )


def parse_lammps_relabel_outputs(
    log_path: str | Path,
    forces_path: str | Path,
    *,
    expected_frame_count: int,
    natoms: int,
) -> list[dict[str, Any]]:
    thermo_rows = _parse_thermo_rows(Path(log_path), expected_frame_count=expected_frame_count)
    force_frames = _parse_force_dump_frames(Path(forces_path), expected_frame_count=expected_frame_count, natoms=natoms)
    if len(thermo_rows) != expected_frame_count:
        raise RuntimeError(
            f"LAMMPS thermo output has {len(thermo_rows)} frames, expected {expected_frame_count}"
        )
    if len(force_frames) != expected_frame_count:
        raise RuntimeError(
            f"LAMMPS force dump has {len(force_frames)} frames, expected {expected_frame_count}"
        )

    payloads: list[dict[str, Any]] = []
    for thermo, forces in zip(thermo_rows, force_frames):
        payloads.append(
            {
                "energy": thermo["energy"],
                "volume": thermo["volume"],
                "stress": pressure_to_ase_stress_matrix(thermo["pressure"]),
                "forces": forces,
            }
        )
    return payloads


def pressure_to_ase_stress6(pressure: Iterable[float]) -> np.ndarray:
    pxx, pyy, pzz, pxy, pxz, pyz = [float(item) for item in pressure]
    return -BAR_TO_EV_PER_A3 * np.array([pxx, pyy, pzz, pyz, pxz, pxy], dtype=float)


def pressure_to_ase_stress_matrix(pressure: Iterable[float]) -> np.ndarray:
    xx, yy, zz, yz, xz, xy = pressure_to_ase_stress6(pressure).tolist()
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def group_rerun_compatible_structures(atoms_list: list[Any]) -> list[list[tuple[int, Any]]]:
    groups: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
    for index, atoms in enumerate(atoms_list):
        groups.setdefault(_group_key(atoms), []).append((index, atoms))
    return list(groups.values())


def _group_key(atoms: Any) -> tuple[Any, ...]:
    symbols = tuple(str(item) for item in atoms.get_chemical_symbols())
    return (len(atoms), symbols, _box_style(atoms))


def _box_style(atoms: Any) -> str:
    cell = np.asarray(getattr(atoms.get_cell(), "array", np.zeros((3, 3))), dtype=float)
    if not np.any(np.abs(cell)):
        return "none"
    if np.allclose(cell[np.triu_indices(3, 1)], 0.0):
        return "orthogonal"
    return "triclinic"


def _collect_teacher_models(teacher_dir: Path, *, model_family: str) -> list[TeacherModelArtifact]:
    adapter = get_deepmd_family_adapter(model_family)
    artifacts: list[TeacherModelArtifact] = []
    for member_dir in sorted(path for path in teacher_dir.glob("teacher_*") if path.is_dir()):
        member_text = member_dir.name.rsplit("_", 1)[-1]
        try:
            teacher_index = int(member_text)
        except ValueError:
            continue
        model_path = member_dir / adapter.primary_model_name(teacher_index)
        if not model_path.exists() or not model_path.is_file() or model_path.stat().st_size <= 0:
            continue
        artifacts.append(
            TeacherModelArtifact(
                backend="deepmd",
                member_index=teacher_index - 1,
                model_path=str(model_path.resolve()),
                metadata={"member_dir": str(member_dir.resolve()), "model_family": model_family},
            )
        )
    return artifacts


def _write_lammpstrj_frames(path: Path, atoms_list: list[Atoms], *, type_map: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    type_index = {symbol: idx + 1 for idx, symbol in enumerate(type_map)}
    with open(path, "w", encoding="utf-8") as handle:
        for step, atoms in enumerate(atoms_list):
            cell = np.asarray(atoms.get_cell().array, dtype=float)
            if cell.shape != (3, 3):
                cell = np.eye(3) * float(math.sqrt(max(float(atoms.get_volume() or 0.0), 1.0)))
            xlo, xhi = 0.0, float(cell[0, 0])
            ylo, yhi = 0.0, float(cell[1, 1])
            zlo, zhi = 0.0, float(cell[2, 2])
            xy, xz, yz = float(cell[1, 0]), float(cell[2, 0]), float(cell[2, 1])
            handle.write("ITEM: TIMESTEP\n")
            handle.write(f"{step}\n")
            handle.write("ITEM: NUMBER OF ATOMS\n")
            handle.write(f"{len(atoms)}\n")
            if np.allclose([xy, xz, yz], 0.0):
                handle.write("ITEM: BOX BOUNDS pp pp pp\n")
                handle.write(f"{xlo:.16f} {xhi:.16f}\n")
                handle.write(f"{ylo:.16f} {yhi:.16f}\n")
                handle.write(f"{zlo:.16f} {zhi:.16f}\n")
            else:
                handle.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                handle.write(f"{xlo:.16f} {xhi:.16f} {xy:.16f}\n")
                handle.write(f"{ylo:.16f} {yhi:.16f} {xz:.16f}\n")
                handle.write(f"{zlo:.16f} {zhi:.16f} {yz:.16f}\n")
            handle.write("ITEM: ATOMS id type x y z\n")
            for atom_id, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_positions()), start=1):
                handle.write(
                    f"{atom_id:d} {type_index[str(symbol)]:d} "
                    f"{float(pos[0]):.16f} {float(pos[1]):.16f} {float(pos[2]):.16f}\n"
                )
    return path.resolve()


def _write_lammps_relabel_input(
    path: Path,
    *,
    conf_path: Path,
    frames_path: Path,
    model_path: Path,
    type_map: list[str],
    lammps_cfg: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.extend(
        [
            "units           metal",
            "boundary        p p p",
            "atom_style      atomic",
            "atom_modify     map yes",
            f"log             {path.parent / 'log.lammps'}",
            "",
            f"read_data       {conf_path.name}",
            "",
        ]
    )
    for atom_type, symbol in enumerate(type_map, start=1):
        lines.append(f"mass            {atom_type} {_atomic_mass(symbol):.6f}")
    lines.extend(
        [
            "",
            f"pair_style      deepmd {Path(model_path).resolve()}",
            f"pair_coeff      * * {' '.join(type_map)}",
            "",
            "neighbor        2.0 bin",
            "neigh_modify    delay 0 every 1 check yes",
            "",
            "thermo          1",
            "thermo_style    custom step pe vol pxx pyy pzz pxy pxz pyz",
            "thermo_modify   flush yes",
            "",
            "dump            1 all custom 1 forces.dump id type x y z fx fy fz",
            "dump_modify     1 sort id",
            "",
            f"rerun           {frames_path.name} dump x y z box yes",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.resolve()


def _atomic_mass(symbol: str) -> float:
    from ase.data import atomic_masses, atomic_numbers

    number = atomic_numbers[str(symbol)]
    return float(atomic_masses[number])


def _write_conf_lammps_data(atoms: Any, path: Path, *, type_map: list[str]) -> Path:
    try:
        return atoms_to_lammps_data(atoms, path, type_map=type_map)
    except Exception:
        return _write_minimal_lammps_data(atoms, path, type_map=type_map)


def _write_minimal_lammps_data(atoms: Any, path: Path, *, type_map: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = [str(item) for item in atoms.get_chemical_symbols()]
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(getattr(atoms.get_cell(), "array", np.zeros((3, 3))), dtype=float)
    if cell.shape != (3, 3) or not np.any(np.abs(cell)):
        span = np.ptp(positions, axis=0) if len(positions) else np.zeros(3)
        padding = np.where(span > 0.0, span * 0.5, 5.0)
        cell = np.diag(np.maximum(span + 2.0 * padding, 10.0))
    xhi = float(abs(cell[0, 0]))
    yhi = float(abs(cell[1, 1]))
    zhi = float(abs(cell[2, 2]))
    lines = [
        "LAMMPS data file written by DPA4 teacher relabel",
        "",
        f"{len(symbols)} atoms",
        f"{len(type_map)} atom types",
        "",
        f"0.0 {xhi:.16f} xlo xhi",
        f"0.0 {yhi:.16f} ylo yhi",
        f"0.0 {zhi:.16f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for atom_type, symbol in enumerate(type_map, start=1):
        lines.append(f"{atom_type:d} {_atomic_mass(symbol):.16f}")
    lines.extend(["", "Atoms # atomic", ""])
    type_index = {str(symbol): idx + 1 for idx, symbol in enumerate(type_map)}
    for atom_id, (symbol, pos) in enumerate(zip(symbols, positions), start=1):
        lines.append(
            f"{atom_id:d} {type_index[str(symbol)]:d} "
            f"{float(pos[0]):.16f} {float(pos[1]):.16f} {float(pos[2]):.16f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.resolve()


def _parse_thermo_rows(path: Path, *, expected_frame_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"LAMMPS thermo log not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 9:
                continue
            try:
                step = int(float(parts[0]))
                numeric = [float(item) for item in parts[1:]]
            except ValueError:
                continue
            if not all(math.isfinite(item) for item in numeric):
                raise RuntimeError(f"Non-finite thermo value in {path}: {stripped}")
            rows.append(
                {
                    "step": step,
                    "energy": float(numeric[0]),
                    "volume": float(numeric[1]),
                    "pressure": tuple(float(item) for item in numeric[2:8]),
                }
            )
    if len(rows) != expected_frame_count:
        raise RuntimeError(
            f"LAMMPS thermo parser expected {expected_frame_count} frame rows but found {len(rows)}"
        )
    return rows


def _parse_force_dump_frames(path: Path, *, expected_frame_count: int, natoms: int) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"LAMMPS force dump not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    frames: list[np.ndarray] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line != "ITEM: TIMESTEP":
            index += 1
            continue
        step = int(lines[index + 1].strip())
        if not lines[index + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError(f"Unexpected force dump format near timestep {step}")
        frame_natoms = int(lines[index + 3].strip())
        if frame_natoms != natoms:
            raise RuntimeError(f"LAMMPS force dump atom count mismatch: {frame_natoms} != {natoms}")
        if not lines[index + 4].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError(f"Unexpected force dump format near timestep {step}")
        header = lines[index + 8].strip().split()
        if header[:2] != ["ITEM:", "ATOMS"]:
            raise RuntimeError(f"Unexpected force dump atom header near timestep {step}")
        columns = header[2:]
        required = ["id", "type", "x", "y", "z", "fx", "fy", "fz"]
        if columns[: len(required)] != required:
            raise RuntimeError(f"LAMMPS force dump columns mismatch: {columns!r}")
        rows = lines[index + 9 : index + 9 + natoms]
        frame = np.zeros((natoms, 3), dtype=float)
        seen: set[int] = set()
        for row in rows:
            fields = row.split()
            atom_id = int(fields[0])
            if atom_id < 1 or atom_id > natoms:
                raise RuntimeError(f"Invalid atom id {atom_id} in {path}")
            if atom_id in seen:
                raise RuntimeError(f"Duplicate atom id {atom_id} in {path}")
            seen.add(atom_id)
            force = np.asarray([float(fields[5]), float(fields[6]), float(fields[7])], dtype=float)
            if not np.all(np.isfinite(force)):
                raise RuntimeError(f"Non-finite force value in {path}: {row.strip()}")
            frame[atom_id - 1] = force
        if len(seen) != natoms:
            raise RuntimeError(f"Missing atom ids in force dump {path}")
        frames.append(frame)
        index = index + 9 + natoms
    if len(frames) != expected_frame_count:
        raise RuntimeError(
            f"LAMMPS force dump parser expected {expected_frame_count} frame rows but found {len(frames)}"
        )
    return frames
