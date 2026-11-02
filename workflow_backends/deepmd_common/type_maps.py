from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from workflow_backends.execution import run_operation, structured_result_stdout
from workflow_runtime.lifecycle.environment import OperationContext

FOUNDATION_TYPE_MAP_RESULT_PREFIX = "DICE_DEEPMD_FOUNDATION_TYPE_MAP="


def resolve_training_type_maps(
    *,
    active_type_map: Sequence[str],
    native_recipe: Mapping[str, Any],
    foundation_model: str | Path | None,
    executor: object,
    work_dir: str | Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve explicit active/native symbol maps before native training."""

    active = validated_type_map(active_type_map, role="active workflow")
    if foundation_model is None:
        return active, active
    foundation = validated_foundation_model(foundation_model)
    resolver = getattr(executor, "foundation_type_map", None)
    if not callable(resolver):
        raise TypeError(
            "DeepMD foundation finetuning requires a native foundation type-map resolver"
        )
    native = validated_type_map(
        resolver(foundation, work_dir=Path(work_dir).resolve()),
        role="foundation native",
    )
    missing = tuple(symbol for symbol in active if symbol not in native)
    if missing:
        raise ValueError(
            "DeepMD active workflow type_map contains symbols absent from the "
            f"foundation native map: {missing}"
        )
    model = native_recipe.get("model", {})
    if not isinstance(model, Mapping):
        raise TypeError("DeepMD foundation recipe model must be an object")
    declared = model.get("type_map")
    if declared is not None:
        recipe_map = validated_type_map(declared, role="foundation recipe")
        if recipe_map != native:
            raise ValueError(
                "DeepMD foundation recipe type_map must exactly preserve the "
                f"foundation native map {native}; received {recipe_map}"
            )
    return active, native


def resolve_foundation_type_map_operation(
    context: OperationContext,
    model: str | Path,
    *,
    work_dir: str | Path,
) -> tuple[str, ...]:
    """Read immutable model metadata through the selected DeepMD train profile."""

    foundation = validated_foundation_model(model)
    operation = run_operation(
        context,
        "python",
        (
            "-m",
            "workflow_backends.deepmd_common.foundation_type_map_worker",
            "--model",
            str(foundation),
        ),
        cwd=Path(work_dir) / ".foundation_type_map",
        diagnostic_name="deepmd_foundation_type_map",
        structured_result_prefix=FOUNDATION_TYPE_MAP_RESULT_PREFIX,
    )
    stdout = structured_result_stdout(
        operation, prefix=FOUNDATION_TYPE_MAP_RESULT_PREFIX
    )
    position = stdout.rfind(FOUNDATION_TYPE_MAP_RESULT_PREFIX)
    if position < 0:
        raise ValueError("DeepMD foundation type-map worker returned no result")
    try:
        payload = json.loads(
            stdout[position + len(FOUNDATION_TYPE_MAP_RESULT_PREFIX) :].splitlines()[0]
        )
    except (json.JSONDecodeError, IndexError) as exc:
        raise ValueError(
            "DeepMD foundation type-map worker returned malformed JSON"
        ) from exc
    return validated_type_map(payload, role="foundation native")


def validated_foundation_model(model: str | Path) -> Path:
    raw = Path(model)
    resolved = raw.resolve()
    if (
        raw.is_symlink()
        or resolved.suffix not in {".pt", ".pt2"}
        or not resolved.is_file()
        or resolved.stat().st_size <= 0
    ):
        raise ValueError(
            "DeepMD foundation model must be a non-empty regular .pt or .pt2 file"
        )
    return resolved


def validated_type_map(values: Sequence[str], *, role: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"DeepMD {role} type_map must be a symbol sequence")
    result = tuple(values)
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"DeepMD {role} type_map must contain unique symbols")
    return result


__all__ = [
    "FOUNDATION_TYPE_MAP_RESULT_PREFIX",
    "resolve_foundation_type_map_operation",
    "resolve_training_type_maps",
    "validated_foundation_model",
    "validated_type_map",
]
