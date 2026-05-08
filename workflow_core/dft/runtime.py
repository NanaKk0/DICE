from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from workflow_core.dft.defaults import DEFAULT_DFT_OUTPUT_NAME


CP2K_EXEC_CANDIDATES = [
    "cp2k.psmp",
    "cp2k.popt",
    "cp2k.ssmp",
    "cp2k.sopt",
    "cp2k",
]


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_explicit_exec(cp2k_exec: str | None) -> str | None:
    if not cp2k_exec:
        return None
    explicit = Path(cp2k_exec)
    if explicit.is_absolute() and explicit.exists():
        return str(explicit.resolve())
    found = shutil.which(cp2k_exec)
    if found:
        return str(Path(found).resolve())
    if explicit.exists():
        return str(explicit.resolve())
    raise RuntimeError(
        f"Configured cp2k_exec was not found: {cp2k_exec}. "
        "You can unset it to allow auto-detection, or point it to a valid CP2K executable."
    )


def _auto_detect_cp2k_exec() -> str | None:
    for candidate in CP2K_EXEC_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return str(Path(found).resolve())
    return None


def _infer_cp2k_data_dir_from_exec(cp2k_exec: str | None) -> str | None:
    if not cp2k_exec:
        return None
    exec_path = Path(cp2k_exec).resolve()
    prefixes = []
    try:
        prefixes.append(exec_path.parent.parent)
    except Exception:
        pass
    for prefix in prefixes:
        candidate = prefix / "share" / "cp2k" / "data"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _run_cp2k_version(cp2k_exec: str | None) -> str:
    if not cp2k_exec:
        return ""
    try:
        proc = subprocess.run(
            [cp2k_exec, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return ""
    return "\n".join(part for part in [proc.stdout, proc.stderr] if part)


def _extract_data_dir_from_version_output(version_text: str) -> str | None:
    for raw_line in version_text.splitlines():
        line = raw_line.strip().replace("\x00", "")
        lower = line.lower()
        if "data directory path" in lower:
            parts = line.split("path", 1)
            if len(parts) == 2:
                candidate = parts[1].strip(" :\t")
                if candidate:
                    path = Path(candidate)
                    if path.exists():
                        return str(path.resolve())
    return None


def _normalize_existing_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.exists():
        return str(path.resolve())
    return None


def _infer_file_from_data_dir(data_dir: str | None, filename: str) -> str | None:
    if not data_dir:
        return None
    candidate = Path(data_dir) / filename
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _default_use_mpirun(cp2k_exec: str) -> bool:
    name = Path(cp2k_exec).name.lower()
    return name in {"cp2k.psmp", "cp2k.popt"}


def _infer_conda_env_from_exec(cp2k_exec: str | None) -> dict[str, str] | None:
    if not cp2k_exec:
        return None
    exec_path = Path(cp2k_exec).resolve()
    parts = exec_path.parts
    try:
        envs_idx = parts.index("envs")
    except ValueError:
        envs_idx = -1
    if envs_idx >= 0 and envs_idx + 1 < len(parts):
        env_prefix = Path(*parts[: envs_idx + 2])
        env_name = parts[envs_idx + 1]
        conda_root = env_prefix.parent.parent
    else:
        if exec_path.parent.name != "bin":
            return None
        env_prefix = exec_path.parent.parent
        env_name = env_prefix.name
        conda_root = env_prefix.parent
    conda_sh = conda_root / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        return None
    return {
        "env_prefix": str(env_prefix.resolve()),
        "env_name": env_name,
        "conda_sh": str(conda_sh.resolve()),
    }


def resolve_dft_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(cfg)
    labelling_style = str(runtime.get("labelling_style", "cp2k")).lower()
    if labelling_style != "cp2k":
        return runtime

    explicit_exec = _as_optional_str(runtime.get("cp2k_exec")) or _as_optional_str(os.environ.get("CP2K_EXEC"))
    cp2k_exec = _resolve_explicit_exec(explicit_exec) if explicit_exec else _auto_detect_cp2k_exec()
    if not cp2k_exec:
        raise RuntimeError(
            "Failed to auto-detect a CP2K executable. Tried: "
            f"{', '.join(CP2K_EXEC_CANDIDATES)}. "
            "You can optionally set cp2k_exec in the DFT config or export CP2K_EXEC."
        )

    version_text = _run_cp2k_version(cp2k_exec)

    cp2k_data_dir = (
        _normalize_existing_path(_as_optional_str(runtime.get("cp2k_data_dir")))
        or _normalize_existing_path(_as_optional_str(os.environ.get("CP2K_DATA_DIR")))
        or _normalize_existing_path(_infer_cp2k_data_dir_from_exec(cp2k_exec))
        or _normalize_existing_path(_extract_data_dir_from_version_output(version_text))
    )

    basis_set_file = (
        _normalize_existing_path(_as_optional_str(runtime.get("basis_set_file")))
        or _normalize_existing_path(_as_optional_str(os.environ.get("CP2K_BASIS_FILE")))
        or _infer_file_from_data_dir(cp2k_data_dir, "BASIS_MOLOPT")
    )
    potential_file = (
        _normalize_existing_path(_as_optional_str(runtime.get("potential_file")))
        or _normalize_existing_path(_as_optional_str(os.environ.get("CP2K_POTENTIAL_FILE")))
        or _infer_file_from_data_dir(cp2k_data_dir, "POTENTIAL")
    )

    if cp2k_data_dir is None:
        raise RuntimeError(
            f"Resolved CP2K executable at {cp2k_exec}, but failed to determine cp2k_data_dir. "
            "You can optionally set cp2k_data_dir or export CP2K_DATA_DIR."
        )
    if basis_set_file is None:
        raise RuntimeError(
            f"Resolved cp2k_data_dir at {cp2k_data_dir}, but BASIS_MOLOPT was not found. "
            "You can optionally set basis_set_file or export CP2K_BASIS_FILE."
        )
    if potential_file is None:
        raise RuntimeError(
            f"Resolved cp2k_data_dir at {cp2k_data_dir}, but POTENTIAL was not found. "
            "You can optionally set potential_file or export CP2K_POTENTIAL_FILE."
        )

    force_use_mpirun = runtime.get("force_use_mpirun")
    if force_use_mpirun is None:
        use_mpirun = _default_use_mpirun(cp2k_exec)
    else:
        use_mpirun = bool(force_use_mpirun)

    output_name = DEFAULT_DFT_OUTPUT_NAME
    current_conda_prefix = _as_optional_str(os.environ.get("CONDA_PREFIX"))
    inferred_conda = _infer_conda_env_from_exec(cp2k_exec)
    shell_prelude_commands: list[str] = []
    auto_activate_cp2k_env = False
    if inferred_conda:
        target_prefix = inferred_conda["env_prefix"]
        if current_conda_prefix != target_prefix:
            shell_prelude_commands = [
                f"source {inferred_conda['conda_sh']}",
                f"conda activate {inferred_conda['env_name']}",
            ]
            auto_activate_cp2k_env = True

    runtime["cp2k_exec"] = cp2k_exec
    runtime["cp2k_data_dir"] = cp2k_data_dir
    runtime["basis_set_file"] = basis_set_file
    runtime["potential_file"] = potential_file
    runtime["use_mpirun"] = use_mpirun
    runtime["cp2k_output_name"] = DEFAULT_DFT_OUTPUT_NAME
    if not _as_optional_str(runtime.get("command")):
        runtime["command"] = f"{cp2k_exec} -i {{input}} -o {output_name}"
    runtime.setdefault("source_list", [])
    runtime["shell_prelude_commands"] = shell_prelude_commands
    runtime["dft_runtime_resolution"] = {
        "cp2k_exec": cp2k_exec,
        "use_mpirun": use_mpirun,
        "cp2k_data_dir": cp2k_data_dir,
        "basis_set_file": basis_set_file,
        "potential_file": potential_file,
        "version_probe_available": bool(version_text.strip()),
        "current_conda_prefix": current_conda_prefix,
        "cp2k_conda_env_prefix": inferred_conda["env_prefix"] if inferred_conda else None,
        "cp2k_conda_env_name": inferred_conda["env_name"] if inferred_conda else None,
        "auto_activate_cp2k_env": auto_activate_cp2k_env,
    }
    return runtime

