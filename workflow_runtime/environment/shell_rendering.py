from __future__ import annotations

from pathlib import Path
import shlex
from typing import Iterable


def render_strict_mode() -> str:
    return "set -euo pipefail"


def render_source_activation(source_list: Iterable[str], *, include_strict: bool = True) -> str:
    lines = [render_strict_mode()] if include_strict else []
    for source_path in source_list:
        lines.extend(
            [
                "set +u",
                f"source {shlex.quote(str(source_path))}",
                "set -u",
            ]
        )
    return "\n".join(lines)


def render_inline_shell_command(command: list[str] | tuple[str, ...], *, source_list: Iterable[str]) -> str:
    lines = [render_source_activation(source_list)]
    lines.append("exec " + shlex.join([str(item) for item in command]))
    return "\n".join(lines)


def render_local_task_shell_script(
    *,
    command: str,
    workdir: str | Path,
    source_list: Iterable[str],
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        render_strict_mode(),
        f"cd {shlex.quote(str(workdir))}",
    ]
    activation = render_source_activation(source_list, include_strict=False)
    if activation:
        lines.append(activation)
    if str(command).strip():
        lines.append(str(command))
    return "\n".join(lines) + "\n"
