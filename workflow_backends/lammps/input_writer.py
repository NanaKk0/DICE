from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase.data import atomic_masses, atomic_numbers


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
    command = str(lammps_cfg.get("command", "auto")).strip()
    if command and command.lower() != "auto":
        candidate = Path(command)
        if candidate.exists():
            return str(candidate.resolve())
        resolved = shutil.which(command)
        if resolved:
            return resolved
        raise FileNotFoundError(f"LAMMPS command not found: {command}")

    resolved = shutil.which("lmp")
    if resolved:
        return resolved
    raise FileNotFoundError("Unable to find LAMMPS executable 'lmp' in PATH")


def _lammps_supports_deepmd(lammps_cmd: str, env: dict[str, str]) -> bool:
    try:
        proc = subprocess.run(
            [lammps_cmd, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    text = str(proc.stdout or "").lower()
    return "deepmd" in text


def find_deepmd_plugin(lammps_cfg: dict[str, Any], *, lammps_cmd: str, env: dict[str, str]) -> Path | None:
    plugin_path = str(lammps_cfg.get("plugin_path", "auto")).strip()
    if plugin_path and plugin_path.lower() != "auto":
        resolved = Path(plugin_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"DeePMD LAMMPS plugin not found: {resolved}")
        return resolved

    if _lammps_supports_deepmd(lammps_cmd, env):
        return None

    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates = [
            Path(conda_prefix) / "lib" / "libdeepmd_lmp.so",
            Path(conda_prefix) / "lib" / "deepmd_lmp" / "libdeepmd_lmp.so",
            Path(conda_prefix) / "lib" / "deepmd_lmp" / "dpplugin.so",
        ]
        candidates.extend(
            sorted((Path(conda_prefix) / "lib").glob("python*/site-packages/deepmd/lib/libdeepmd_lmpplugin.so"))
        )
        candidates.extend(sorted((Path(conda_prefix) / "lib").glob("deepmd_lmp/*.so")))
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

    raise FileNotFoundError(
        "Unable to locate libdeepmd_lmp.so automatically. Set sampling_seed_policy.lammps.plugin_path explicitly."
    )


def _mass_line(atom_type: int, symbol: str) -> str:
    number = atomic_numbers[str(symbol)]
    mass = atomic_masses[number]
    return f"mass            {atom_type} {mass:.6f}"


def write_lammps_deepmd_input(
    path: str | Path,
    *,
    model_paths: list[str | Path],
    type_map: list[str],
    plugin_path: str | Path | None,
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

    lines = []
    if plugin_path is not None:
        lines.append(f"plugin load {Path(plugin_path).resolve()}")
        lines.append("")

    lines.extend(
        [
            "units           metal",
            "atom_style      atomic",
            "boundary        p p p",
            f"log             {log_filename}",
            "",
            "read_data       conf.lmp",
            "",
        ]
    )
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
