"""Assemble a native sampling protocol with DICE-owned task setup."""
from __future__ import annotations

import re
import shlex


_SETUP = "include dice_task_setup.lammps"
_EXPLICIT_SETUP = re.compile(r"(?m)^\s*include\s+dice_task_setup\.lammps\s*$")
_PRE_BOX = frozenset({
    "units", "dimension", "boundary", "atom_style", "atom_modify",
    "newton", "processors", "comm_style", "package", "suffix", "plugin",
})
_PREAMBLE = _PRE_BOX | {"variable", "log", "echo"}
_MANAGED_COMMANDS = frozenset({
    "read_data", "read_restart", "create_box", "create_atoms", "replicate",
    "pair_style", "pair_coeff", "mass", "clear", "reset_timestep",
})
_MANAGED_VARIABLES = frozenset({
    "dice_run_steps", "dice_temperature", "dice_pressure", "dice_seed",
})


def command_spans(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Read logical command boundaries, preserving the author's source lines."""
    commands: list[tuple[int, int, list[str]]] = []
    tokens: list[str] = []
    start = 0
    continued = False
    for index, line in enumerate(lines):
        try:
            parts = shlex.split(line, comments=True)
            quoted_parts = shlex.split(line, comments=True, posix=False)
        except ValueError as exc:
            raise ValueError(f"automatic LAMMPS protocol: invalid quoting on line {index + 1}") from exc
        if not parts:
            continue
        if not continued:
            start = index
        continued = quoted_parts[-1] == "&"
        tokens.extend(parts[:-1] if continued else parts)
        if not continued:
            commands.append((start, index + 1, tokens))
            tokens = []
    if continued:
        raise ValueError("automatic LAMMPS protocol: unfinished command continuation")
    return commands


def _commands(lines: list[str]) -> list[tuple[int, list[str]]]:
    return [(start, tokens) for start, _end, tokens in command_spans(lines)]


def assemble_lammps_protocol(text: str, *, validate_existing: bool = False) -> str:
    """Insert task setup between initialization and the physical protocol.

    Existing explicit setup recipes retain their exact bytes and validation.
    Automatic recipes keep all physical commands and their order; ambiguous
    initialization or competing task setup is rejected before native execution.
    """
    explicit = bool(_EXPLICIT_SETUP.search(text))
    if explicit and not validate_existing:
        return text
    lines = text.splitlines(keepends=True)
    commands = _commands(lines)
    insertion = len(lines)
    in_protocol = False
    for index, tokens in commands:
        command = tokens[0]
        if command in _MANAGED_COMMANDS:
            raise ValueError(f"automatic LAMMPS protocol: {command} is supplied by DICE")
        if tokens == ["include", "dice_task_setup.lammps"] and explicit:
            in_protocol = True
            continue
        if command == "include" and tokens != ["include", "dice_md_mc.lammps"]:
            raise ValueError("automatic LAMMPS protocol: supply the protocol in one file")
        if command == "variable" and len(tokens) > 1 and tokens[1] in _MANAGED_VARIABLES:
            raise ValueError(f"automatic LAMMPS protocol: {tokens[1]} is supplied by DICE")
        if command == "dump" and (
            (len(tokens) > 1 and tokens[1] == "1") or "traj.lammpstrj" in tokens
        ):
            raise ValueError("automatic LAMMPS protocol: trajectory dump 1 is supplied by DICE")
        if command in {"dump_modify", "undump"} and len(tokens) > 1 and tokens[1] == "1":
            raise ValueError("automatic LAMMPS protocol: trajectory dump 1 is supplied by DICE")
        if in_protocol and command in _PRE_BOX:
            raise ValueError(f"automatic LAMMPS protocol: place {command} before protocol commands")
        if not in_protocol and (
            command not in _PREAMBLE
            or any("$" in token for token in tokens)
        ):
            insertion = index
            in_protocol = True
    if explicit:
        if sum(tokens == ["include", "dice_task_setup.lammps"] for _, tokens in commands) != 1:
            raise ValueError("automatic LAMMPS protocol: requires one task setup include")
        return text
    prefix = "".join(lines[:insertion])
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return prefix + _SETUP + "\n" + "".join(lines[insertion:])
