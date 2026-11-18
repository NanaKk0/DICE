"""Backend recipe feasibility shared by suffix preflight and task preparation."""
from pathlib import Path
import re
from workflow_backends.lammps.protocol import assemble_lammps_protocol


def validate_sampling_recipe( recipe: Path, *, kind: str | None = None, ensemble: str | None = None, text: str | None = None) -> None:
    compiled = text is not None
    from workflow_backends.lammps.program_protocol import prepare_sampling_protocol
    text = text if text is not None else assemble_lammps_protocol(prepare_sampling_protocol(recipe.read_text(encoding="utf-8"),kind))
    if kind == 'md_mc':
        from workflow_backends.lammps.identity_swap import validate_recipe
        validate_recipe(text, ensemble=ensemble)
        return
    if "DICE_BEGIN" in text or "DICE_END" in text:
        raise ValueError("lifecycle LAMMPS source recipe must not contain managed markers")
    includes = re.findall(r"(?m)^\s*include\s+dice_task_setup\.lammps\s*$", text)
    if len(includes) != 1: raise ValueError("lifecycle LAMMPS recipe requires exactly one include dice_task_setup.lammps")
    capability = re.findall(
        r"(?m)^\s*#\s*DICE_CAPABILITY:\s*nvt_cell_program\s*$", text
    )
    if kind == "nvt_cell_program":
        cadence = re.findall(
            r"(?m)^\s*variable\s+dice_trajectory_every_steps\s+equal\s+([1-9]\d*)\s*$",
            text,
        )
        if len(capability) != 1 or len(cadence) != 1:
            raise ValueError(
                "lifecycle LAMMPS nvt_cell_program recipe requires its explicit capability and cadence"
            )
        if re.search(
            r"(?mi)^\s*(?:run|clear|reset_timestep|read_data|read_restart|restart|write_restart)\b",
            text,
        ) or re.search(
            r"(?mi)^\s*fix\s+\S+\s+\S+\s+(?:nvt|npt|deform)\b", text
        ):
            raise ValueError(
                "lifecycle LAMMPS nvt_cell_program recipe may not own managed dynamics"
            )
        return
    if capability:
        raise ValueError(
            "lifecycle LAMMPS cell-program capability is valid only for nvt_cell_program"
        )
    if kind not in {"nvt_program", "npt_program"}:
        pattern = r'(?m)^\s*run\s+(?:[1-9]\d*|\$\{dice_run_steps\})\s*$' if compiled else r'(?m)^\s*run\s+\$\{dice_run_steps\}\s*$'
        runs = re.findall(pattern, text)
        if len(runs) != 1: raise ValueError("lifecycle LAMMPS recipe requires exactly one main run ${dice_run_steps}")
        return
    cadence = re.findall(
        r"(?m)^\s*variable\s+dice_trajectory_every_steps\s+equal\s+([1-9]\d*)\s*$",
        text,
    )
    ensemble=kind.removesuffix('_program')
    fix_pattern=rf'(?m)^\s*fix\s+(\S+)\s+\S+\s+{ensemble}\s+temp\s+\$\{{dice_tstart\}}\s+\$\{{dice_tstop\}}\s+[^\n]+$'
    fix_match=re.search(fix_pattern,text)
    unfix_pattern=rf'(?m)^\s*unfix\s+{re.escape(fix_match.group(1)) if fix_match else "dice_missing"}\s*$'
    required = (
        r"(?m)^\s*variable\s+dice_tstart\s+file\s+dice_temperature_start\.dat\s*$",
        r"(?m)^\s*variable\s+dice_tstop\s+file\s+dice_temperature_stop\.dat\s*$",
        r"(?m)^\s*variable\s+dice_segment_steps\s+file\s+dice_temperature_steps\.dat\s*$",
        r"(?m)^\s*label\s+dice_segment_loop\s*$",
        fix_pattern,
        r"(?m)^\s*run\s+\$\{dice_segment_steps\}\s*$",
        unfix_pattern,
        r"(?m)^\s*next\s+dice_tstart\s+dice_tstop\s+dice_segment_steps\s*$",
        r"(?m)^\s*jump\s+SELF\s+dice_segment_loop\s*$",
    )
    matches = [re.findall(pattern, text) for pattern in required]
    if len(cadence) != 1 or any(len(items) != 1 for items in matches):
        raise ValueError(f"lifecycle LAMMPS {kind} recipe requires the canonical schedule loop")
    include_at = text.index(re.search(r"(?m)^\s*include\s+dice_task_setup\.lammps\s*$", text).group())
    cadence_at = text.index(re.search(r"(?m)^\s*variable\s+dice_trajectory_every_steps", text).group())
    positions = [re.search(pattern, text).start() for pattern in required]
    if cadence_at >= include_at or positions != sorted(positions):
        raise ValueError(f"lifecycle LAMMPS {kind} recipe loop order is invalid")
    loop_at = positions[3]
    if re.search(r"(?mi)^\s*(?:clear|reset_timestep|read_data|read_restart)\b", text):
        raise ValueError(f"lifecycle LAMMPS {kind} recipe may not restart task state")
    velocities = tuple(re.finditer(r"(?mi)^\s*velocity\b", text))
    if len(velocities) > 1 or any(match.start() >= loop_at for match in velocities):
        raise ValueError(f"lifecycle LAMMPS {kind} recipe velocity creation is allowed only once before the loop")
