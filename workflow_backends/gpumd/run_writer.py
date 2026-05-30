from __future__ import annotations

from pathlib import Path

GPUMD_ACTIVE_BLOCK_BEGIN = "# DICE_BEGIN_GPUMD_ACTIVE_LEARNING"
GPUMD_ACTIVE_BLOCK_END = "# DICE_END_GPUMD_ACTIVE_LEARNING"


def write_gpumd_run_input(
    output_path: str | Path,
    *,
    model_paths: list[str],
    temperature: float,
    steps: int,
    timestep_fs: float,
    active_interval: int,
    active_threshold: float,
    dump_interval: int,
    ensemble: str = "nvt",
    thermostat_damp: float = 100.0,
    save_forces: bool = True,
    save_velocities: bool = True,
    template_path: str | Path | None = None,
    temperature_override_declared: bool = False,
    steps_override_declared: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    potential_lines = [f"potential    {Path(item).name}" for item in model_paths]
    if not potential_lines:
        raise ValueError("GPUMD run.in requires at least one NEP potential")
    active_force_flag = 1 if save_forces else 0
    active_velocity_flag = 1 if save_velocities else 0
    active_uncertainty_flag = 1
    managed_lines = [
        *potential_lines,
        f"dump_thermo  {max(1, int(dump_interval))}",
        f"dump_position {max(1, int(dump_interval))}",
        f"active       {max(1, int(active_interval))} {active_velocity_flag} {active_force_flag} {active_uncertainty_flag} {float(active_threshold)}",
    ]
    if template_path is not None:
        return render_gpumd_run_template(
            template_path,
            output_path,
            managed_lines=managed_lines,
            temperature=float(temperature),
            steps=int(steps),
            temperature_override_declared=bool(temperature_override_declared),
            steps_override_declared=bool(steps_override_declared),
        )

    ensemble_name = str(ensemble).strip().lower()
    if ensemble_name == "nvt":
        ensemble_line = f"ensemble     nvt_ber {float(temperature)} {float(temperature)} {float(thermostat_damp)}"
    else:
        ensemble_line = f"ensemble     {ensemble_name}"
    lines = [
        *potential_lines,
        f"time_step    {float(timestep_fs)}",
        f"velocity     {float(temperature)}",
        ensemble_line,
        managed_lines[-1],
        f"run          {max(1, int(steps))}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.resolve()


def render_gpumd_run_template(
    template_path: str | Path,
    output_path: str | Path,
    *,
    managed_lines: list[str],
    temperature: float | None = None,
    steps: int | None = None,
    temperature_override_declared: bool = False,
    steps_override_declared: bool = False,
) -> Path:
    template = Path(template_path).resolve()
    if not template.is_file():
        raise FileNotFoundError(f"GPUMD run.in template not found: {template}")
    text = template.read_text(encoding="utf-8")
    begin_count = text.count(GPUMD_ACTIVE_BLOCK_BEGIN)
    end_count = text.count(GPUMD_ACTIVE_BLOCK_END)
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            "GPUMD run.in template requires exactly one DICE-managed active-learning block "
            f"delimited by {GPUMD_ACTIVE_BLOCK_BEGIN!r} and {GPUMD_ACTIVE_BLOCK_END!r}: {template}"
        )
    before, remainder = text.split(GPUMD_ACTIVE_BLOCK_BEGIN, 1)
    _, after = remainder.split(GPUMD_ACTIVE_BLOCK_END, 1)
    rendered = (
        before.rstrip()
        + "\n"
        + GPUMD_ACTIVE_BLOCK_BEGIN
        + "\n"
        + "\n".join(managed_lines)
        + "\n"
        + GPUMD_ACTIVE_BLOCK_END
        + after
    )
    rendered = patch_gpumd_sampling_scalars(
        rendered,
        temperature=temperature,
        steps=steps,
        temperature_override_declared=temperature_override_declared,
        steps_override_declared=steps_override_declared,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered if rendered.endswith("\n") else rendered + "\n", encoding="utf-8")
    return output.resolve()


def patch_gpumd_sampling_scalars(
    text: str,
    *,
    temperature: float | None,
    steps: int | None,
    temperature_override_declared: bool,
    steps_override_declared: bool,
) -> str:
    if not temperature_override_declared and not steps_override_declared:
        return text

    lines = text.splitlines()
    if temperature_override_declared:
        if temperature is None:
            raise ValueError("GPUMD sampling temperature override requires a task temperature")
        lines = _patch_gpumd_temperature_lines(lines, temperature=float(temperature))
    if steps_override_declared:
        if steps is None:
            raise ValueError("GPUMD sampling steps override requires task steps")
        lines = _patch_gpumd_run_line(lines, steps=int(steps))
    return "\n".join(lines)


def _patch_gpumd_run_line(lines: list[str], *, steps: int) -> list[str]:
    run_indices = [
        index
        for index, line in enumerate(lines)
        if _gpumd_code_tokens(line) and _gpumd_code_tokens(line)[0].lower() == "run"
    ]
    if len(run_indices) > 1:
        raise ValueError("ambiguous GPUMD sampling steps override: template contains multiple run commands")
    if not run_indices:
        return [*lines, f"run {max(1, int(steps))}"]
    index = run_indices[0]
    tokens = _gpumd_code_tokens(lines[index])
    if len(tokens) < 2 or not _looks_numeric(tokens[1]):
        raise ValueError(f"unsupported GPUMD sampling steps override form: {lines[index].strip()}")
    tokens[1] = str(max(1, int(steps)))
    updated = list(lines)
    updated[index] = _replace_gpumd_code_tokens(lines[index], tokens)
    return updated


def _patch_gpumd_temperature_lines(lines: list[str], *, temperature: float) -> list[str]:
    velocity_indices = []
    nvt_indices = []
    unsupported_temperature_lines = []
    for index, line in enumerate(lines):
        tokens = _gpumd_code_tokens(line)
        lower = [item.lower() for item in tokens]
        if len(lower) >= 1 and lower[0] == "velocity":
            velocity_indices.append(index)
            if len(tokens) != 2 or not _looks_numeric(tokens[1]):
                unsupported_temperature_lines.append(line)
        elif len(lower) >= 1 and lower[0] == "ensemble":
            if len(lower) >= 4 and lower[1] == "nvt_ber":
                nvt_indices.append(index)
                if not _looks_numeric(tokens[2]) or not _looks_numeric(tokens[3]):
                    unsupported_temperature_lines.append(line)
            elif any(_looks_numeric(item) for item in tokens[1:]):
                unsupported_temperature_lines.append(line)

    if len(velocity_indices) > 1:
        raise ValueError("ambiguous GPUMD sampling temperature override: template contains multiple velocity commands")
    if len(nvt_indices) > 1:
        raise ValueError("ambiguous GPUMD sampling temperature override: template contains multiple ensemble nvt_ber commands")
    if unsupported_temperature_lines:
        examples = "; ".join(item.strip() for item in unsupported_temperature_lines[:3])
        raise ValueError(f"unsupported GPUMD sampling temperature override form: {examples}")

    updated = list(lines)
    if velocity_indices:
        index = velocity_indices[0]
        tokens = _gpumd_code_tokens(updated[index])
        tokens[1] = _format_gpumd_number(temperature)
        updated[index] = _replace_gpumd_code_tokens(updated[index], tokens)
    else:
        updated = _insert_before_first_gpumd_run(updated, f"velocity {_format_gpumd_number(temperature)}")

    if nvt_indices:
        index = nvt_indices[0]
        tokens = _gpumd_code_tokens(updated[index])
        tokens[2] = _format_gpumd_number(temperature)
        tokens[3] = _format_gpumd_number(temperature)
        updated[index] = _replace_gpumd_code_tokens(updated[index], tokens)
    else:
        updated = _insert_before_first_gpumd_run(
            updated,
            f"ensemble nvt_ber {_format_gpumd_number(temperature)} {_format_gpumd_number(temperature)} 100",
        )
    return updated


def _insert_before_first_gpumd_run(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        tokens = _gpumd_code_tokens(line)
        if tokens and tokens[0].lower() == "run":
            return [*lines[:index], new_line, *lines[index:]]
    return [*lines, new_line]


def _gpumd_code_tokens(line: str) -> list[str]:
    code = line.split("#", 1)[0].strip()
    if not code:
        return []
    return code.split()


def _replace_gpumd_code_tokens(original_line: str, tokens: list[str]) -> str:
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


def _format_gpumd_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")
