from __future__ import annotations

from typing import Literal


LammpsExecutionMode = Literal["ordinary", "kokkos"]

ORDINARY_COMMAND = ("lmp", "-in", "input.lammps")
KOKKOS_COMMAND = (
    "lmp",
    "-k",
    "on",
    "g",
    "1",
    "-sf",
    "kk",
    "-in",
    "input.lammps",
)


def select_lammps_execution_mode(
    *,
    sampler_role: str,
    deepmd_recipe_kind: str,
    model_count: int,
) -> LammpsExecutionMode:
    """Select the closed backend-native route from resolved workflow facts."""

    if sampler_role not in {"sampler.lammps.tf", "sampler.lammps.pt"}:
        raise ValueError("LAMMPS execution mode requires a typed sampler role")
    if deepmd_recipe_kind not in {"standard", "dpa4c"}:
        raise ValueError("LAMMPS execution mode requires a known DeepMD recipe kind")
    if isinstance(model_count, bool) or model_count < 1:
        raise ValueError("LAMMPS execution mode requires a positive model count")
    if deepmd_recipe_kind == "dpa4c" and sampler_role != "sampler.lammps.pt":
        raise ValueError("DPA4C LAMMPS execution requires the PT sampler ABI")
    if deepmd_recipe_kind == "dpa4c" and model_count == 1:
        return "kokkos"
    return "ordinary"


def command_for_mode(mode: LammpsExecutionMode) -> tuple[str, ...]:
    if mode == "kokkos":
        return KOKKOS_COMMAND
    if mode == "ordinary":
        return ORDINARY_COMMAND
    raise ValueError("unsupported LAMMPS execution mode")


__all__ = [
    "KOKKOS_COMMAND",
    "LammpsExecutionMode",
    "ORDINARY_COMMAND",
    "command_for_mode",
    "select_lammps_execution_mode",
]
