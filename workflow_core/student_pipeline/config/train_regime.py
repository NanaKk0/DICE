from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TRAIN_RECIPE_BRANCH_SCRATCH_ENTRY = "scratch_entry"
TRAIN_RECIPE_BRANCH_FOUNDATION_ENTRY = "foundation_entry"
TRAIN_RECIPE_BRANCH_CONTINUATION = "continuation"

DATASET_MODE_INCREMENTAL = "incremental"
DATASET_MODE_CUMULATIVE = "cumulative"

INITIALIZATION_MODE_SCRATCH = "scratch"
INITIALIZATION_MODE_FOUNDATION = "foundation"
INITIALIZATION_MODE_PREVIOUS_COMMITTEE = "previous_committee"


@dataclass(frozen=True, slots=True)
class TrainRegimeConfig:
    use_foundation_model: bool
    continue_training: bool


@dataclass(frozen=True, slots=True)
class ResolvedTrainDecision:
    train_recipe_ref: str
    effective_train_recipe_branch: str
    dataset_mode: str
    initialization_mode: str
    is_entry_train: bool
    uses_foundation_model: bool
    requires_previous_committee: bool


def normalize_train_regime(payload: dict[str, Any]) -> TrainRegimeConfig:
    raw = dict(payload or {})
    return TrainRegimeConfig(
        use_foundation_model=_require_bool(raw["use_foundation_model"], "train_regime.use_foundation_model"),
        continue_training=_require_bool(raw["continue_training"], "train_regime.continue_training"),
    )


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def resolve_train_decision(
    *,
    train_regime: dict[str, Any],
    is_entry_train: bool,
    train_recipe_ref: str,
) -> ResolvedTrainDecision:
    regime = normalize_train_regime(train_regime)
    dataset_mode = (
        DATASET_MODE_INCREMENTAL if regime.continue_training else DATASET_MODE_CUMULATIVE
    )

    if bool(is_entry_train):
        if regime.use_foundation_model:
            return ResolvedTrainDecision(
                train_recipe_ref=str(train_recipe_ref),
                effective_train_recipe_branch=TRAIN_RECIPE_BRANCH_FOUNDATION_ENTRY,
                dataset_mode=dataset_mode,
                initialization_mode=INITIALIZATION_MODE_FOUNDATION,
                is_entry_train=True,
                uses_foundation_model=True,
                requires_previous_committee=False,
            )
        return ResolvedTrainDecision(
            train_recipe_ref=str(train_recipe_ref),
            effective_train_recipe_branch=TRAIN_RECIPE_BRANCH_SCRATCH_ENTRY,
            dataset_mode=dataset_mode,
            initialization_mode=INITIALIZATION_MODE_SCRATCH,
            is_entry_train=True,
            uses_foundation_model=False,
            requires_previous_committee=False,
        )

    if regime.continue_training:
        return ResolvedTrainDecision(
            train_recipe_ref=str(train_recipe_ref),
            effective_train_recipe_branch=TRAIN_RECIPE_BRANCH_CONTINUATION,
            dataset_mode=dataset_mode,
            initialization_mode=INITIALIZATION_MODE_PREVIOUS_COMMITTEE,
            is_entry_train=False,
            uses_foundation_model=False,
            requires_previous_committee=True,
        )

    if regime.use_foundation_model:
        return ResolvedTrainDecision(
            train_recipe_ref=str(train_recipe_ref),
            effective_train_recipe_branch=TRAIN_RECIPE_BRANCH_FOUNDATION_ENTRY,
            dataset_mode=dataset_mode,
            initialization_mode=INITIALIZATION_MODE_FOUNDATION,
            is_entry_train=False,
            uses_foundation_model=True,
            requires_previous_committee=False,
        )

    return ResolvedTrainDecision(
        train_recipe_ref=str(train_recipe_ref),
        effective_train_recipe_branch=TRAIN_RECIPE_BRANCH_SCRATCH_ENTRY,
        dataset_mode=dataset_mode,
        initialization_mode=INITIALIZATION_MODE_SCRATCH,
        is_entry_train=False,
        uses_foundation_model=False,
        requires_previous_committee=False,
    )
