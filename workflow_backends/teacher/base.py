from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TeacherModelArtifact:
    backend: str
    member_index: int
    model_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


PathLike = str | Path


@dataclass(slots=True)
class TeacherRelabelRequest:
    committee_dir: PathLike
    teacher_num: int | None = None
    device: str = "cpu"
    default_dtype: str = "float32"
    backend_options: dict[str, Any] = field(default_factory=dict)
    backend_log_dir: PathLike | None = None
    execution_context: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TeacherRelabelResult:
    labeled_structures: list[Any]
    uncertainties: Any


@dataclass(slots=True)
class TeacherCommitteeFinetuneRequest:
    train_cfg: dict[str, Any]
    config_path: Path
    paths: Any
    teacher_round: int
    merged_xyz: Path
    work_root: Path
    runtime_json: Path
    backend_log_dir: Path
    reuse_existing_committee: bool = False
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TeacherFoundationTrainRequest:
    cfg: dict[str, Any]
    paths: Any
    teacher_round: int
    artifact_root: Path
    foundation_model_root: Path
    runtime_json: Path
    backend_log_dir: Path
    context: dict[str, Any] = field(default_factory=dict)


class TeacherBackend(ABC):
    backend_name = ""
    aliases: tuple[str, ...] = ()
    strict_model_discovery_completion = False

    def matches_name(self, name: str) -> bool:
        normalized = str(name or "").strip().lower()
        return normalized == self.backend_name or normalized in self.aliases

    @abstractmethod
    def collect_models(self, model_root: str) -> list[TeacherModelArtifact]:
        raise NotImplementedError

    @abstractmethod
    def relabel_structures(self, atoms_list: list[Any], request: TeacherRelabelRequest) -> TeacherRelabelResult:
        raise NotImplementedError

    def preserve_relabel_atom_order(self, request: TeacherRelabelRequest) -> bool:
        del request
        return False

    def load_committee_finetune_config(self, config_path: str | Path) -> dict[str, Any]:
        del config_path
        raise NotImplementedError(f"{self.backend_name} does not support teacher committee finetune config loading")

    def finetune_committee(self, request: TeacherCommitteeFinetuneRequest) -> dict[str, Any] | None:
        del request
        raise NotImplementedError(f"{self.backend_name} does not support teacher committee finetune execution")

    def train_foundation(self, request: TeacherFoundationTrainRequest) -> dict[str, Any] | None:
        del request
        raise NotImplementedError(f"{self.backend_name} does not support teacher foundation training")
