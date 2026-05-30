from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_backends.teacher.base import TeacherBackend, TeacherModelArtifact, TeacherRelabelRequest, TeacherRelabelResult
from workflow_backends.teacher.committee_labeling import label_with_committee
from workflow_backends.teacher.deepmd.config import load_deepmd_committee_finetune_config
from workflow_backends.teacher.deepmd.evaluator import build_deepmd_evaluator
from workflow_backends.teacher.deepmd.family import get_deepmd_family_adapter
from workflow_backends.teacher.deepmd.foundation import run_deepmd_foundation_train
from workflow_backends.teacher.deepmd.finetune import run_deepmd_committee_finetune


class DeepMDTeacherBackend(TeacherBackend):
    backend_name = "deepmd"
    aliases = ("deepmd-kit", "dp")
    strict_model_discovery_completion = True

    def collect_models(self, model_root: str) -> list[TeacherModelArtifact]:
        root = Path(model_root).resolve()
        metadata = _read_root_metadata(root)
        model_family = str(metadata.get("model_family", "dpa4"))
        adapter = get_deepmd_family_adapter(model_family)
        artifacts: list[TeacherModelArtifact] = []
        for member_dir in sorted(root.glob("teacher_*")):
            if not member_dir.is_dir():
                continue
            member_text = member_dir.name.rsplit("_", 1)[-1]
            try:
                teacher_index = int(member_text)
            except ValueError:
                continue
            model_path = member_dir / adapter.primary_model_name(teacher_index)
            if not model_path.exists() or not model_path.is_file() or model_path.stat().st_size <= 0:
                continue
            artifacts.append(
                TeacherModelArtifact(
                    backend=self.backend_name,
                    member_index=teacher_index - 1,
                    model_path=str(model_path.resolve()),
                    metadata={
                        "member_dir": str(member_dir.resolve()),
                        "model_family": model_family,
                    },
                )
            )
        return artifacts

    def relabel_structures(self, atoms_list: list[Any], request: TeacherRelabelRequest) -> TeacherRelabelResult:
        teacher_dir = Path(request.committee_dir).resolve()
        metadata = _read_root_metadata(teacher_dir)
        model_family = str(metadata.get("model_family", "dpa4"))
        type_map = [str(item) for item in metadata.get("type_map", []) or []]
        options = dict(request.backend_options or {})
        relabel_engine = _resolve_relabel_engine(model_family, options.get("relabel_engine"))
        options["relabel_engine"] = relabel_engine
        request.backend_options = options
        element_order = options.get("element_order")
        if element_order is not None and type_map and [str(item) for item in element_order] != type_map:
            raise ValueError(
                f"DeePMD teacher relabel requires element_order to match type_map in first version: "
                f"element_order={element_order!r}, type_map={type_map!r}"
            )

        if relabel_engine == "lammps":
            return relabel_structures_with_lammps(atoms_list, request)

        return relabel_structures_with_python(atoms_list, request)

    def preserve_relabel_atom_order(self, request: TeacherRelabelRequest) -> bool:
        options = dict(request.backend_options or {})
        model_family = _read_model_family(request.committee_dir)
        return _resolve_relabel_engine(model_family, options.get("relabel_engine")) == "lammps"

    def load_committee_finetune_config(self, config_path: str | Path) -> dict[str, Any]:
        return load_deepmd_committee_finetune_config(config_path)

    def finetune_committee(self, request, *, runner=None) -> dict[str, Any]:
        return run_deepmd_committee_finetune(request, runner=runner)

    def train_foundation(self, request, *, runner=None) -> dict[str, Any]:
        if runner is None:
            runner = request.context.get("foundation_runner")
        return run_deepmd_foundation_train(request, runner=runner)


def _read_root_metadata(root: Path) -> dict[str, Any]:
    metadata_path = root / "deepmd_teacher_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        import json

        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_model_family(committee_dir: str | Path) -> str:
    root = Path(committee_dir).resolve()
    metadata = _read_root_metadata(root)
    return str(metadata.get("model_family", "dpa4"))


def _resolve_relabel_engine(model_family: str, relabel_engine: Any) -> str:
    normalized = str(relabel_engine or "").strip().lower()
    if normalized:
        if normalized not in {"lammps", "python"}:
            raise ValueError(f"Unsupported DeePMD relabel_engine={relabel_engine!r}. Expected 'lammps' or 'python'")
        return normalized
    if str(model_family).strip().lower() == "dpa4":
        return "lammps"
    return "python"


def relabel_structures_with_python(
    atoms_list: list[Any],
    request: TeacherRelabelRequest,
    *,
    model_family: str | None = None,
    type_map: list[str] | None = None,
) -> TeacherRelabelResult:
    from workflow_backends.teacher.deepmd.lammps_relabel import relabel_structures_with_python as impl

    if model_family is None or type_map is None:
        model_family, type_map = _resolve_model_identity(request.committee_dir)
    return impl(atoms_list, request, model_family=model_family, type_map=type_map)


def relabel_structures_with_lammps(
    atoms_list: list[Any],
    request: TeacherRelabelRequest,
    *,
    model_family: str | None = None,
    type_map: list[str] | None = None,
    runner=None,
) -> TeacherRelabelResult:
    from workflow_backends.teacher.deepmd.lammps_relabel import relabel_structures_with_lammps as impl

    if model_family is None or type_map is None:
        model_family, type_map = _resolve_model_identity(request.committee_dir)
    return impl(atoms_list, request, model_family=model_family, type_map=type_map, runner=runner)


def _resolve_model_identity(committee_dir: str | Path) -> tuple[str, list[str]]:
    metadata = _read_root_metadata(Path(committee_dir).resolve())
    model_family = str(metadata.get("model_family", "dpa4"))
    type_map = [str(item) for item in metadata.get("type_map", []) or []]
    return model_family, type_map


__all__ = [
    "DeepMDTeacherBackend",
    "build_deepmd_evaluator",
    "load_deepmd_committee_finetune_config",
    "run_deepmd_foundation_train",
]
