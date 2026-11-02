from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol

import yaml

from workflow_core.model_families import deepmd_recipe_model_family

from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter
from workflow_artifacts.lifecycle.model_publication import (
    ModelArtifact,
    PublishedModelHandle,
)
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherTrainingCapability,
    TeacherTrainingIntent,
    TeacherTrainingOutcome,
)


# ``self`` is the stable zero-holdout lifecycle spelling.  It guarantees that
# every canonical frame remains in training; backend adapters decide whether
# their native runtime needs no validation input or a small monitor dataset.
TeacherValidationMode = Literal["holdout", "self"]
TeacherModelShape = Literal["single", "committee"]
TeacherTrainPolicy = Literal["from_scratch", "foundation_finetune"]


@dataclass(frozen=True, slots=True)
class TeacherFoundationAsset:
    kind: str
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())


@dataclass(frozen=True, slots=True)
class TeacherTrainResult:
    member_artifacts: tuple[ModelArtifact, ...]
    backend_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_artifacts", tuple(self.member_artifacts))
        object.__setattr__(self, "backend_evidence", MappingProxyType(dict(self.backend_evidence)))


@dataclass(frozen=True, slots=True)
class TeacherTrainCapability:
    backend: str
    cli: str | None
    model_shapes: frozenset[str]
    policies: frozenset[str]
    foundation_asset_kinds: frozenset[str]
    output_model_kinds: frozenset[str]
    required_operation_roles: tuple[str, ...]
    recipe_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_shapes", frozenset(self.model_shapes))
        object.__setattr__(self, "policies", frozenset(self.policies))
        object.__setattr__(self, "foundation_asset_kinds", frozenset(self.foundation_asset_kinds))
        object.__setattr__(self, "output_model_kinds", frozenset(self.output_model_kinds))
        object.__setattr__(self, "required_operation_roles", tuple(self.required_operation_roles))
        object.__setattr__(self, "recipe_conditions", tuple(self.recipe_conditions))


@dataclass(frozen=True, slots=True)
class TeacherTrainRequest:
    """Resolved lifecycle ``teach/train`` intent passed to a teacher backend."""

    iter_index: int
    recipe_path: Path
    train_data_path: Path
    validation_data_path: Path
    validation_mode: TeacherValidationMode
    model_shape: TeacherModelShape
    member_ids: tuple[str, ...]
    type_map: tuple[str, ...]
    policy: TeacherTrainPolicy
    foundation_asset: TeacherFoundationAsset | None
    train_dir: Path | None = None
    train_options: Mapping[str, bool] = field(default_factory=dict)
    progress_reporter_factory: Callable[[str], OperationProgressReporter] | None = None

    def __post_init__(self) -> None:
        recipe_path = Path(self.recipe_path).resolve()
        train_data_path = Path(self.train_data_path).resolve()
        validation_data_path = Path(self.validation_data_path).resolve()
        if self.train_dir is None:
            train_dir = train_data_path.parent
            if train_dir.name == "dataset":
                train_dir = train_dir.parent
        else:
            train_dir = Path(self.train_dir).resolve()
        member_ids = tuple(str(member_id) for member_id in self.member_ids)
        valid_shape = (
            self.model_shape == "single" and len(member_ids) == 1
        ) or (
            self.model_shape == "committee" and len(member_ids) >= 2
        )
        if not valid_shape or any(not member_id for member_id in member_ids) or len(set(member_ids)) != len(member_ids):
            raise ValueError("lifecycle teacher model shape requires ordered unique member identities")
        if self.validation_mode == "holdout" and train_data_path == validation_data_path:
            raise ValueError("lifecycle teacher holdout train and validation paths must be disjoint")
        if self.validation_mode == "self" and train_data_path != validation_data_path:
            raise ValueError("lifecycle teacher self validation requires equal train and validation paths")
        if self.policy == "foundation_finetune" and self.foundation_asset is None:
            raise ValueError("lifecycle teacher foundation_finetune requires a foundation asset")
        if self.policy == "from_scratch" and self.foundation_asset is not None:
            raise ValueError("lifecycle teacher from_scratch must not carry a foundation asset")
        options = dict(self.train_options)
        if set(options) - {"skip_neighbor_stat"}:
            raise ValueError("lifecycle teacher train_options contains unsupported fields")
        if any(not isinstance(value, bool) for value in options.values()):
            raise ValueError("lifecycle teacher train_options values must be booleans")
        object.__setattr__(self, "recipe_path", recipe_path)
        object.__setattr__(self, "train_data_path", train_data_path)
        object.__setattr__(self, "validation_data_path", validation_data_path)
        object.__setattr__(self, "train_dir", train_dir)
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "type_map", tuple(self.type_map))
        object.__setattr__(self, "train_options", MappingProxyType(options))


@dataclass(frozen=True, slots=True)
class TeacherTrainPublicationRequest:
    train: TeacherTrainResult
    handles: tuple[PublishedModelHandle, ...]
    stage_root: Path
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        handles = tuple(self.handles)
        member_ids = tuple(str(item) for item in self.member_ids)
        returned = tuple(item.member_id for item in self.train.member_artifacts)
        published = tuple(item.member_id for item in handles)
        if returned != member_ids or published != member_ids:
            raise ValueError(
                "teacher publication requires ordered train artifacts and handles"
            )
        object.__setattr__(self, "handles", handles)
        object.__setattr__(self, "stage_root", Path(self.stage_root).resolve())
        object.__setattr__(self, "member_ids", member_ids)


class TeacherTrainingOps(Protocol):
    def train(self, request: TeacherTrainRequest) -> TeacherTrainResult: ...

    def finalize_published_training(
        self, request: TeacherTrainPublicationRequest
    ) -> None: ...


def resolve_teacher_train_capability(
    *,
    backend: str,
    cli: str | None,
    recipe_path: str | Path,
    model_shape: str,
    policy: str,
    foundation_asset_kind: str | None,
) -> TeacherTrainCapability:
    """Resolve backend-owned native recipe truth for one selected lifecycle route."""

    capability = _resolve_teacher_train_capability(
        backend=backend,
        cli=cli,
        recipe_path=recipe_path,
        model_shape=model_shape,
        policy=policy,
    )
    _validate_capability_selection(capability, model_shape, policy, foundation_asset_kind)
    return capability


def preflight_teacher_train_capability(
    *,
    backend: str,
    cli: str | None,
    recipe_path: str | Path,
    model_shape: str,
    policy: str,
    foundation_asset_present: bool,
) -> TeacherTrainCapability:
    """Validate launch-time intent without resolving lifecycle artifact aliases."""

    capability = _resolve_teacher_train_capability(
        backend=backend,
        cli=cli,
        recipe_path=recipe_path,
        model_shape=model_shape,
        policy=policy,
    )
    foundation_asset_kind = None
    if foundation_asset_present:
        if len(capability.foundation_asset_kinds) != 1:
            raise ValueError(
                f"lifecycle teacher capability backend {capability.backend!r} does not declare one "
                "unambiguous foundation asset kind"
            )
        foundation_asset_kind = next(iter(capability.foundation_asset_kinds))
    _validate_capability_selection(capability, model_shape, policy, foundation_asset_kind)
    return capability


def _resolve_teacher_train_capability(
    *,
    backend: str,
    cli: str | None,
    recipe_path: str | Path,
    model_shape: str,
    policy: str,
) -> TeacherTrainCapability:
    normalized_backend = str(backend).strip().lower()
    resolver = _CAPABILITY_RESOLVERS.get(normalized_backend)
    if resolver is None:
        raise ValueError(f"unsupported lifecycle teacher capability backend {backend!r}")
    return resolver(Path(recipe_path).resolve(), cli, model_shape, policy)


def _resolve_mace_capability(
    resolved_recipe: Path,
    cli: str | None,
    model_shape: str,
    policy: str,
) -> TeacherTrainCapability:
    if cli is not None:
        raise ValueError(
            f"lifecycle teacher capability rejected backend='mace' cli={cli!r} policy={policy!r}: "
            "MACE uses its native mace_run_train CLI"
        )
    try:
        recipe = yaml.safe_load(resolved_recipe.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"lifecycle teacher capability rejected backend='mace' cli='mace_run_train' policy={policy!r}: "
            "recipe must be YAML"
        ) from exc
    if not isinstance(recipe, dict):
        raise ValueError(
            f"lifecycle teacher capability rejected backend='mace' cli='mace_run_train' policy={policy!r}: "
            "recipe must be a YAML object"
        )
    native_model = recipe.get("model")
    if native_model is not None and (not isinstance(native_model, str) or not native_model.strip()):
        raise ValueError(
            f"lifecycle teacher capability rejected backend='mace' cli='mace_run_train' policy={policy!r}: "
            "native recipe model selector must be a non-empty string when present"
        )
    selector_condition = (
        "native model selector is omitted"
        if native_model is None
        else f"native model selector is {native_model!r}"
    )
    roles = ["teacher.mace.train", "teacher.mace.label"]
    if model_shape == "single":
        roles.extend(("teacher.mace.gmm_reference", "teacher.mace.gmm_score"))
    return TeacherTrainCapability(
        backend="mace",
        cli="mace_run_train",
        model_shapes=frozenset({"single", "committee"}),
        policies=frozenset({"from_scratch", "foundation_finetune"}),
        foundation_asset_kinds=frozenset({"mace_model"}),
        output_model_kinds=frozenset({"mace_model"}),
        required_operation_roles=tuple(roles),
        recipe_conditions=("recipe is a YAML object", selector_condition),
    )


def _resolve_deepmd_capability(
    resolved_recipe: Path,
    cli: str | None,
    model_shape: str,
    policy: str,
) -> TeacherTrainCapability:
    if cli not in {"tf", "pt"}:
        raise ValueError(
            f"lifecycle teacher capability rejected backend='deepmd' cli={cli!r} policy={policy!r}: "
            "selected native CLI must be 'tf' or 'pt'"
        )
    try:
        recipe = json.loads(resolved_recipe.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"lifecycle teacher capability rejected backend='deepmd' cli={cli!r} policy={policy!r}: "
            "recipe must be native DeepMD JSON"
        ) from exc
    if not isinstance(recipe, dict):
        raise ValueError(
            f"lifecycle teacher capability rejected backend='deepmd' cli={cli!r} policy={policy!r}: "
            "recipe must be a JSON object"
        )
    native_model = recipe.get("model", {})
    if not isinstance(native_model, dict):
        raise ValueError(
            f"lifecycle teacher capability rejected backend='deepmd' cli={cli!r} policy={policy!r}: "
            "native recipe model must be an object"
        )
    dpa4_selected = deepmd_recipe_model_family(recipe) == "dpa4"
    if dpa4_selected and cli != "pt":
        raise ValueError(
            f"lifecycle teacher capability rejected backend='deepmd' cli={cli!r} policy={policy!r}: "
            "native recipe selects DPA4, which requires the DeepMD PT CLI"
        )
    output_kind = "deepmd_pt2" if cli == "pt" else "deepmd_pb"
    recipe_condition = (
        "native recipe selects DPA4"
        if dpa4_selected
        else f"native recipe is compatible with DeepMD {cli.upper()}"
    )
    roles = [
        f"teacher.deepmd.{cli}.train",
        f"teacher.deepmd.{cli}.label",
    ]
    if dpa4_selected and cli == "pt" and model_shape == "single":
        roles.extend((
            "teacher.deepmd.pt.gmm_reference",
            "teacher.deepmd.pt.gmm_score",
        ))
    return TeacherTrainCapability(
        backend="deepmd",
        cli=cli,
        model_shapes=frozenset({"single", "committee"}),
        policies=frozenset({"from_scratch", "foundation_finetune"}),
        foundation_asset_kinds=frozenset({output_kind}),
        output_model_kinds=frozenset({output_kind}),
        required_operation_roles=tuple(roles),
        recipe_conditions=("recipe is a native DeepMD JSON object", recipe_condition),
    )


_CAPABILITY_RESOLVERS = {
    "mace": _resolve_mace_capability,
    "deepmd": _resolve_deepmd_capability,
}


def _validate_capability_selection(
    capability: TeacherTrainCapability,
    model_shape: str,
    policy: str,
    foundation_asset_kind: str | None,
) -> None:
    rejected: list[str] = []
    if model_shape not in capability.model_shapes:
        rejected.append(f"model shape {model_shape!r} is unsupported")
    if policy not in capability.policies:
        rejected.append(f"policy {policy!r} is unsupported")
    if policy == "foundation_finetune" and foundation_asset_kind is None:
        rejected.append("foundation_finetune requires a foundation asset")
    if policy == "from_scratch" and foundation_asset_kind is not None:
        rejected.append("from_scratch must not carry a foundation asset")
    if foundation_asset_kind is not None and foundation_asset_kind not in capability.foundation_asset_kinds:
        rejected.append(f"foundation asset kind {foundation_asset_kind!r} is unsupported")
    if rejected:
        conditions = "; ".join(capability.recipe_conditions)
        raise ValueError(
            f"lifecycle teacher capability rejected backend={capability.backend!r} cli={capability.cli!r} "
            f"policy={policy!r}: {'; '.join(rejected)}; recipe conditions: {conditions}"
        )
