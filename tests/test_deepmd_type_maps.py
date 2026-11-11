from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from workflow_backends.deepmd_common.type_maps import (
    FOUNDATION_TYPE_MAP_RESULT_PREFIX,
    resolve_foundation_type_map_operation,
    resolve_training_type_maps,
)
from workflow_runtime.lifecycle.environment import OperationContext


class _Resolver:
    def __init__(self, native_type_map: tuple[str, ...]) -> None:
        self.native_type_map = native_type_map
        self.calls: list[tuple[Path, Path]] = []

    def foundation_type_map(self, model: Path, *, work_dir: Path) -> tuple[str, ...]:
        self.calls.append((Path(model), Path(work_dir)))
        return self.native_type_map


def _foundation(tmp_path: Path) -> Path:
    path = tmp_path / "foundation.pt2"
    path.write_bytes(b"foundation")
    return path


def test_foundation_native_map_accepts_arbitrary_active_subset_order(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    resolver = _Resolver(("H", "O", "C", "Pt"))

    active, native = resolve_training_type_maps(
        active_type_map=("Pt", "H", "C"),
        native_recipe={"model": {}},
        foundation_model=foundation,
        executor=resolver,
        work_dir=tmp_path / "train",
    )

    assert active == ("Pt", "H", "C")
    assert native == ("H", "O", "C", "Pt")
    assert resolver.calls == [(foundation.resolve(), (tmp_path / "train").resolve())]


@pytest.mark.parametrize(
    "active,native,recipe_map,match",
    [
        (("H", "H"), ("H", "O"), None, "unique"),
        (("Pt",), ("H", "O"), None, "absent"),
        (("H",), ("H", "O"), ("O", "H"), "exactly preserve"),
        (("H",), ("H", "O"), ("H",), "exactly preserve"),
        (("H",), ("H", "O"), ("H", "O", "C"), "exactly preserve"),
    ],
)
def test_foundation_map_rejects_duplicate_missing_or_recipe_transformation(
    tmp_path: Path,
    active: tuple[str, ...],
    native: tuple[str, ...],
    recipe_map: tuple[str, ...] | None,
    match: str,
) -> None:
    foundation = _foundation(tmp_path)
    resolver = _Resolver(native)
    recipe = {"model": {}}
    if recipe_map is not None:
        recipe["model"]["type_map"] = list(recipe_map)

    with pytest.raises(ValueError, match=match):
        resolve_training_type_maps(
            active_type_map=active,
            native_recipe=recipe,
            foundation_model=foundation,
            executor=resolver,
            work_dir=tmp_path / "train",
        )


def test_scratch_map_comes_from_active_chemistry_without_foundation_query(
    tmp_path: Path,
) -> None:
    resolver = _Resolver(("unused",))
    active, native = resolve_training_type_maps(
        active_type_map=("O", "H"),
        native_recipe={"model": {"type_map": ["ignored", "recipe"]}},
        foundation_model=None,
        executor=resolver,
        work_dir=tmp_path,
    )
    assert active == native == ("O", "H")
    assert resolver.calls == []


def test_foundation_map_operation_uses_selected_profile_and_exact_asset(
    tmp_path: Path,
) -> None:
    foundation = _foundation(tmp_path)
    context = OperationContext(
        role="student.deepmd.pt.train",
        profile_id="dpa4c",
        source_scripts=(),
        executables={"python": "python"},
        runtime={},
    )
    calls = []

    def run(operation_context, logical_executable, arguments, **kwargs):
        calls.append((operation_context, logical_executable, tuple(arguments), kwargs))
        return SimpleNamespace(
            stdout=FOUNDATION_TYPE_MAP_RESULT_PREFIX
            + json.dumps(["H", "O", "C"])
            + "\n",
            structured_result=None,
        )

    with patch("workflow_backends.deepmd_common.type_maps.run_operation", run):
        resolved = resolve_foundation_type_map_operation(
            context, foundation, work_dir=tmp_path / "train"
        )

    assert resolved == ("H", "O", "C")
    assert calls[0][0] is context
    assert calls[0][1] == "python"
    assert calls[0][2][-1] == str(foundation.resolve())
    assert calls[0][3]["structured_result_prefix"] == FOUNDATION_TYPE_MAP_RESULT_PREFIX
