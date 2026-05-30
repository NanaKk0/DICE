from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ase.data import atomic_masses, atomic_numbers

LAMMPS_MODEL_DEVI_BLOCK_BEGIN = "# DICE_BEGIN_MODEL_DEVI"
LAMMPS_MODEL_DEVI_BLOCK_END = "# DICE_END_MODEL_DEVI"


def rotate_models(models: list[str | Path], task_index: int, enabled: bool) -> list[Path]:
    resolved = [Path(item).resolve() for item in models]
    if not resolved or not enabled:
        return resolved
    shift = int(task_index) % len(resolved)
    return resolved[shift:] + resolved[:shift]


def build_lammps_env(lammps_cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    device_policy = str(lammps_cfg.get("device_policy", "cpu")).strip().lower()
    # For CPU parallel sampling, explicitly hide GPUs from the DeePMD/TensorFlow
    # runtime so multiple LAMMPS tasks do not contend for one device and OOM.
    if device_policy == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    extra_lib_dirs: list[str] = []
    if conda_prefix:
        extra_lib_dirs.append(str((Path(conda_prefix) / "lib").resolve()))
        extra_lib_dirs.extend(
            str(item.resolve())
            for item in sorted((Path(conda_prefix) / "lib").glob("python*/site-packages/deepmd/lib"))
        )
        extra_lib_dirs.extend(
            str(item.resolve())
            for item in sorted((Path(conda_prefix) / "lib").glob("python*/site-packages/tensorflow"))
        )
        extra_lib_dirs.extend(
            str(item.resolve())
            for item in sorted((Path(conda_prefix) / "lib").glob("python*/site-packages/torch/lib"))
        )
    current_ld = env.get("LD_LIBRARY_PATH", "")
    merged_ld = [item for item in extra_lib_dirs if item]
    if current_ld:
        merged_ld.append(current_ld)
    if merged_ld:
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(merged_ld))
    return env


def find_lammps_command(lammps_cfg: dict[str, Any]) -> str:
    command = lammps_cfg.get("command")
    if command in (None, ""):
        raise KeyError("LAMMPS sampling requires executable 'lmp' from the script-first environment profile")
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        return list(command)
    return [str(command)]


def _mass_line(atom_type: int, symbol: str) -> str:
    number = atomic_numbers[str(symbol)]
    mass = atomic_masses[number]
    return f"mass            {atom_type} {mass:.6f}"


def write_lammps_deepmd_input(
    path: str | Path,
    *,
    model_paths: list[str | Path],
    type_map: list[str],
    nsteps: int,
    timestep_fs: float,
    temperature: float,
    ensemble: str,
    thermo_freq: int,
    dump_freq: int,
    model_devi_freq: int,
    atomic_model_devi: bool,
    neighbor_bin: float,
    thermostat_damp_fs: float,
    seed: int,
    traj_filename: str = "traj.lammpstrj",
    log_filename: str = "log.lammps",
    template_path: str | Path | None = None,
    temperature_override_declared: bool = False,
    steps_override_declared: bool = False,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    timestep_ps = float(timestep_fs) / 1000.0
    tdamp_ps = float(thermostat_damp_fs) / 1000.0
    ensemble_norm = str(ensemble).strip().lower()
    if ensemble_norm not in {"nve", "nvt", "langevin"}:
        raise ValueError(f"Unsupported LAMMPS ensemble: {ensemble}")

    model_tokens = " ".join(Path(item).name for item in model_paths)
    pair_style = f"pair_style      deepmd {model_tokens} out_file model_devi.out out_freq {int(model_devi_freq)}"
    if atomic_model_devi:
        pair_style += " atomic"

    if template_path is not None:
        managed_lines = [
            "read_data       conf.lmp",
            *[_mass_line(atom_type, symbol) for atom_type, symbol in enumerate(type_map, start=1)],
            pair_style,
            f"pair_coeff      * * {' '.join(type_map)}",
            f"dump            1 all custom {int(dump_freq)} {traj_filename} id type x y z fx fy fz",
            "dump_modify     1 sort id",
        ]
        return render_lammps_model_devi_template(
            template_path,
            output,
            managed_lines=managed_lines,
            temperature=float(temperature),
            steps=int(nsteps),
            seed=int(seed),
            thermostat_damp_ps=float(tdamp_ps),
            temperature_override_declared=bool(temperature_override_declared),
            steps_override_declared=bool(steps_override_declared),
        )

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        f"log             {log_filename}",
        "",
        "read_data       conf.lmp",
        "",
    ]
    for atom_type, symbol in enumerate(type_map, start=1):
        lines.append(_mass_line(atom_type, symbol))
    lines.extend(
        [
            "",
            pair_style,
            f"pair_coeff      * * {' '.join(type_map)}",
            "",
            f"neighbor        {float(neighbor_bin):.6f} bin",
            "neigh_modify    delay 0 every 1 check yes",
            "",
            f"velocity        all create {float(temperature):.6f} {int(seed)} mom yes rot yes dist gaussian",
        ]
    )

    if ensemble_norm == "nve":
        lines.append("fix             1 all nve")
    elif ensemble_norm == "nvt":
        lines.append(
            f"fix             1 all nvt temp {float(temperature):.6f} {float(temperature):.6f} {tdamp_ps:.6f}"
        )
    else:
        lines.append("fix             1 all nve")
        lines.append(
            f"fix             2 all langevin {float(temperature):.6f} {float(temperature):.6f} {tdamp_ps:.6f} {int(seed)}"
        )

    lines.extend(
        [
            "",
            f"timestep        {timestep_ps:.6f}",
            "",
            f"thermo          {int(thermo_freq)}",
            "thermo_style    custom step temp pe ke etotal press",
            "",
            f"dump            1 all custom {int(dump_freq)} {traj_filename} id type x y z fx fy fz",
            "dump_modify     1 sort id",
            "",
            f"run             {int(nsteps)}",
            "",
        ]
    )

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def render_lammps_model_devi_template(
    template_path: str | Path,
    output_path: str | Path,
    *,
    managed_lines: list[str],
    temperature: float | None = None,
    steps: int | None = None,
    seed: int | None = None,
    thermostat_damp_ps: float | None = None,
    temperature_override_declared: bool = False,
    steps_override_declared: bool = False,
) -> Path:
    template = Path(template_path).resolve()
    if not template.is_file():
        raise FileNotFoundError(f"LAMMPS input template not found: {template}")
    text = template.read_text(encoding="utf-8")
    begin_count = text.count(LAMMPS_MODEL_DEVI_BLOCK_BEGIN)
    end_count = text.count(LAMMPS_MODEL_DEVI_BLOCK_END)
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            "LAMMPS input template requires exactly one DICE-managed model-deviation block "
            f"delimited by {LAMMPS_MODEL_DEVI_BLOCK_BEGIN!r} and {LAMMPS_MODEL_DEVI_BLOCK_END!r}: {template}"
        )
    before, remainder = text.split(LAMMPS_MODEL_DEVI_BLOCK_BEGIN, 1)
    _, after = remainder.split(LAMMPS_MODEL_DEVI_BLOCK_END, 1)
    rendered = (
        before.rstrip()
        + "\n"
        + LAMMPS_MODEL_DEVI_BLOCK_BEGIN
        + "\n"
        + "\n".join(managed_lines)
        + "\n"
        + LAMMPS_MODEL_DEVI_BLOCK_END
        + after
    )
    rendered = patch_lammps_sampling_scalars(
        rendered,
        temperature=temperature,
        steps=steps,
        seed=seed,
        thermostat_damp_ps=thermostat_damp_ps,
        temperature_override_declared=temperature_override_declared,
        steps_override_declared=steps_override_declared,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered if rendered.endswith("\n") else rendered + "\n", encoding="utf-8")
    return output.resolve()


def patch_lammps_sampling_scalars(
    text: str,
    *,
    temperature: float | None,
    steps: int | None,
    seed: int | None,
    thermostat_damp_ps: float | None,
    temperature_override_declared: bool,
    steps_override_declared: bool,
) -> str:
    if not temperature_override_declared and not steps_override_declared:
        return text

    lines = text.splitlines()
    if temperature_override_declared:
        if temperature is None:
            raise ValueError("LAMMPS sampling temperature override requires a task temperature")
        lines = _patch_lammps_temperature_lines(
            lines,
            temperature=float(temperature),
            seed=int(seed or 20260319),
            thermostat_damp_ps=float(thermostat_damp_ps if thermostat_damp_ps is not None else 0.1),
        )
    if steps_override_declared:
        if steps is None:
            raise ValueError("LAMMPS sampling steps override requires task steps")
        lines = _patch_lammps_run_line(lines, steps=int(steps))
    return "\n".join(lines)


def _patch_lammps_run_line(lines: list[str], *, steps: int) -> list[str]:
    run_indices = [
        index
        for index, line in enumerate(lines)
        if _lammps_code_tokens(line) and _lammps_code_tokens(line)[0].lower() == "run"
    ]
    if len(run_indices) > 1:
        raise ValueError("ambiguous LAMMPS sampling steps override: template contains multiple run commands")
    if not run_indices:
        return [*lines, f"run {int(steps)}"]

    index = run_indices[0]
    tokens = _lammps_code_tokens(lines[index])
    if len(tokens) < 2 or not _looks_numeric(tokens[1]):
        raise ValueError(f"unsupported LAMMPS sampling steps override form: {lines[index].strip()}")
    tokens[1] = str(int(steps))
    updated = list(lines)
    updated[index] = _replace_lammps_code_tokens(lines[index], tokens)
    return updated


def _patch_lammps_temperature_lines(
    lines: list[str],
    *,
    temperature: float,
    seed: int,
    thermostat_damp_ps: float,
) -> list[str]:
    velocity_indices = []
    nvt_indices = []
    unsupported_temperature_lines = []
    for index, line in enumerate(lines):
        tokens = _lammps_code_tokens(line)
        lower = [item.lower() for item in tokens]
        if len(lower) >= 4 and lower[0] == "velocity" and lower[2] == "create":
            velocity_indices.append(index)
            if not _looks_numeric(tokens[3]):
                unsupported_temperature_lines.append(line)
        elif lower[:1] == ["fix"] and len(lower) >= 4 and lower[3] == "langevin":
            unsupported_temperature_lines.append(line)
        elif "temp" in lower and lower[:1] == ["fix"]:
            if len(lower) >= 6 and lower[3] == "nvt" and lower[4] == "temp":
                nvt_indices.append(index)
                if not _looks_numeric(tokens[5]) or len(tokens) < 7 or not _looks_numeric(tokens[6]):
                    unsupported_temperature_lines.append(line)
            else:
                unsupported_temperature_lines.append(line)

    if len(velocity_indices) > 1:
        raise ValueError("ambiguous LAMMPS sampling temperature override: template contains multiple velocity create commands")
    if len(nvt_indices) > 1:
        raise ValueError("ambiguous LAMMPS sampling temperature override: template contains multiple fix nvt temp commands")
    if unsupported_temperature_lines:
        examples = "; ".join(item.strip() for item in unsupported_temperature_lines[:3])
        raise ValueError(f"unsupported LAMMPS sampling temperature override form: {examples}")

    updated = list(lines)
    if velocity_indices:
        index = velocity_indices[0]
        tokens = _lammps_code_tokens(updated[index])
        tokens[3] = _format_lammps_number(temperature)
        if len(tokens) >= 5 and _looks_numeric(tokens[4]):
            tokens[4] = str(int(seed))
        updated[index] = _replace_lammps_code_tokens(updated[index], tokens)
    else:
        updated = _insert_before_first_lammps_run(
            updated,
            f"velocity all create {_format_lammps_number(temperature)} {int(seed)} mom yes rot yes dist gaussian",
        )

    if nvt_indices:
        index = nvt_indices[0]
        tokens = _lammps_code_tokens(updated[index])
        tokens[5] = _format_lammps_number(temperature)
        tokens[6] = _format_lammps_number(temperature)
        updated[index] = _replace_lammps_code_tokens(updated[index], tokens)
    else:
        updated = _insert_before_first_lammps_run(
            updated,
            f"fix 1 all nvt temp {_format_lammps_number(temperature)} {_format_lammps_number(temperature)} {_format_lammps_number(thermostat_damp_ps)}",
        )
    return updated


def _insert_before_first_lammps_run(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        tokens = _lammps_code_tokens(line)
        if tokens and tokens[0].lower() == "run":
            return [*lines[:index], new_line, *lines[index:]]
    return [*lines, new_line]


def _lammps_code_tokens(line: str) -> list[str]:
    code = line.split("#", 1)[0].strip()
    if not code:
        return []
    return code.split()


def _replace_lammps_code_tokens(original_line: str, tokens: list[str]) -> str:
    indent = original_line[: len(original_line) - len(original_line.lstrip())]
    suffix = ""
    if "#" in original_line:
        suffix = " #" + original_line.split("#", 1)[1]
    return indent + " ".join(tokens) + suffix


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _format_lammps_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")
