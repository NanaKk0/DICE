from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from workflow_core.student_pipeline.config.stage_config import canonical_stage_type
from workflow_core.student_pipeline.artifacts.stage_contracts import student_stage_contract_descriptor
from workflow_runtime.artifacts.checks import xyz_file_complete
from workflow_core.student_pipeline.orchestration.runtime_node_contracts import student_stage_runtime_node_contract


@dataclass(frozen=True, slots=True)
class SuccessfulStudentStage:
    teacher_round: int
    student_round: int
    stage_run_index: int
    stage_type: str
    stage_dir: Path
    marker_path: Path
    stage_record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainCommitteeArtifact:
    stage: SuccessfulStudentStage
    model_root: Path
    committee_dir: Path
    committee_summary_path: Path


@dataclass(frozen=True, slots=True)
class SeedPoolSource:
    path: Path
    source_type: str
    stage: SuccessfulStudentStage | None = None


@dataclass(frozen=True, slots=True)
class RelabelOutputs:
    stage: SuccessfulStudentStage
    train_xyz: Path
    dft_xyz: Path


@dataclass(frozen=True, slots=True)
class SamplingOutputs:
    stage: SuccessfulStudentStage
    candidate_xyz: Path
    failed_xyz: Path
    seed_pool_after_sampling_xyz: Path


class SuccessfulStageArtifactResolver:
    """Resolve successful student stage artifacts using marker plus artifact checks."""

    def __init__(self, runtime_state: Mapping[str, Any]) -> None:
        self._runtime_state = dict(runtime_state or {})
        self._paths = self._runtime_state.get("paths")
        if self._paths is None:
            raise ValueError("student pipeline runtime_state is missing paths")

    def successful_stages_before(
        self,
        stage: Mapping[str, Any],
        *,
        stage_type: str | None = None,
        teacher_round: int | None = None,
    ) -> tuple[SuccessfulStudentStage, ...]:
        stage_filter = canonical_stage_type(stage_type) if stage_type is not None else None
        result: list[SuccessfulStudentStage] = []
        for prior in self._stages_before(stage):
            if stage_filter is not None and canonical_stage_type(str(prior["stage"])) != stage_filter:
                continue
            if teacher_round is not None and int(prior["teacher_round"]) != int(teacher_round):
                continue
            successful = self._successful_stage(prior)
            if successful is not None:
                result.append(successful)
        return tuple(result)

    def latest_train_committee_before(self, stage: Mapping[str, Any]) -> TrainCommitteeArtifact | None:
        trains = self.successful_stages_before(stage, stage_type="train")
        if trains:
            successful = trains[-1]
        else:
            previous_teacher_round = int(stage["teacher_round"]) - 1
            if previous_teacher_round < 0:
                return None
            successful = self._latest_successful_train_stage_for_teacher_round(previous_teacher_round)
            if successful is None:
                return None
        descriptor = student_stage_contract_descriptor("train")
        committee_dir = descriptor.published_output_path("student_committee", successful.stage_dir)
        model_root = committee_dir if committee_dir.exists() else successful.stage_dir
        return TrainCommitteeArtifact(
            stage=successful,
            model_root=model_root.resolve(),
            committee_dir=committee_dir,
            committee_summary_path=descriptor.published_output_path("student_committee_summary", successful.stage_dir),
        )

    def _latest_successful_train_stage_for_teacher_round(self, teacher_round: int) -> SuccessfulStudentStage | None:
        train_stages = [
            successful
            for successful in (
                self._successful_stage(stage)
                for stage in self._scan_stage_dirs(int(teacher_round))
                if canonical_stage_type(str(stage["stage"])) == "train"
            )
            if successful is not None
        ]
        return sorted(train_stages, key=lambda stage: (stage.student_round, stage.stage_run_index))[-1] if train_stages else None

    def current_effective_seed_pool_before(self, stage: Mapping[str, Any]) -> SeedPoolSource:
        for prior in reversed(self.successful_stages_before(stage)):
            if prior.stage_type == "sampling":
                return SeedPoolSource(
                    path=_successor_path(prior),
                    source_type="sampling",
                    stage=prior,
                )
            if prior.stage_type == "seed_pool_refresh":
                return SeedPoolSource(
                    path=_successor_path(prior),
                    source_type="seed_pool_refresh",
                    stage=prior,
                )

        initial = Path(self._paths.student_subworkflow_initial_seed_pool_xyz(int(stage["teacher_round"]))).resolve()
        if not xyz_file_complete(initial):
            raise ValueError(f"initial seed-pool snapshot is missing or invalid: {initial}")
        return SeedPoolSource(path=initial, source_type="initial_seed_pool", stage=None)

    def previous_teacher_seed_successor(self, teacher_round: int) -> SeedPoolSource | None:
        candidates: list[tuple[tuple[int, int], SeedPoolSource]] = []
        for stage in self._scan_stage_dirs(int(teacher_round)):
            stage_type = canonical_stage_type(str(stage["stage"]))
            if stage_type not in {"sampling", "seed_pool_refresh"}:
                continue
            successful = self._successful_stage(stage)
            if successful is None:
                continue
            if successful.stage_type == "sampling":
                source_type = "sampling"
            else:
                source_type = "seed_pool_refresh"
            candidates.append(
                (
                    (successful.student_round, successful.stage_run_index),
                    SeedPoolSource(path=_successor_path(successful), source_type=source_type, stage=successful),
                )
            )
        if candidates:
            return sorted(candidates, key=lambda item: item[0])[-1][1]

        initial = Path(self._paths.student_subworkflow_initial_seed_pool_xyz(int(teacher_round))).resolve()
        if xyz_file_complete(initial):
            return SeedPoolSource(path=initial, source_type="initial_seed_pool", stage=None)
        return None

    def relabel_outputs_before(
        self,
        stage: Mapping[str, Any],
        *,
        teacher_round: int | None = None,
    ) -> tuple[RelabelOutputs, ...]:
        return tuple(
            RelabelOutputs(
                stage=successful,
                train_xyz=student_stage_contract_descriptor("relabel").published_output_path(
                    "student_training_labeled",
                    successful.stage_dir,
                ),
                dft_xyz=student_stage_contract_descriptor("relabel").published_output_path(
                    "teacher_dft_candidate",
                    successful.stage_dir,
                ),
            )
            for successful in self.successful_stages_before(
                stage,
                stage_type="relabel",
                teacher_round=teacher_round,
            )
        )

    def sampling_outputs_after_boundary(
        self,
        stage: Mapping[str, Any],
        *,
        boundary_stage_type: str,
    ) -> tuple[SamplingOutputs, ...]:
        boundary_type = canonical_stage_type(boundary_stage_type)
        before = self._stages_before(stage)
        boundary_index = -1
        for index, prior in enumerate(before):
            if canonical_stage_type(str(prior["stage"])) != boundary_type:
                continue
            if self._successful_stage(prior) is not None:
                boundary_index = index

        result: list[SamplingOutputs] = []
        for prior in before[boundary_index + 1 :]:
            if canonical_stage_type(str(prior["stage"])) != "sampling":
                continue
            successful = self._successful_stage(prior)
            if successful is None:
                continue
            result.append(
                SamplingOutputs(
                    stage=successful,
                    candidate_xyz=student_stage_contract_descriptor("sampling").published_output_path(
                        "student_candidate",
                        successful.stage_dir,
                    ),
                    failed_xyz=student_stage_contract_descriptor("sampling").published_output_path(
                        "student_failed",
                        successful.stage_dir,
                    ),
                    seed_pool_after_sampling_xyz=student_stage_contract_descriptor("sampling").published_output_path(
                        "seed_pool_successor",
                        successful.stage_dir,
                    ),
                )
            )
        return tuple(result)

    def teacher_round_dft_candidate_sources_before(
        self,
        stage: Mapping[str, Any],
    ) -> tuple[Path, ...]:
        return tuple(
            output.dft_xyz
            for output in self.relabel_outputs_before(stage, teacher_round=int(stage["teacher_round"]))
        )

    def teacher_round_dft_candidate_sources(self, teacher_round: int) -> tuple[Path, ...]:
        stage_dirs = tuple(
            successful
            for successful in (self._successful_stage(stage) for stage in self._scan_stage_dirs(int(teacher_round)))
            if successful is not None and successful.stage_type == "relabel"
        )
        descriptor = student_stage_contract_descriptor("relabel")
        return tuple(descriptor.published_output_path("teacher_dft_candidate", stage.stage_dir) for stage in stage_dirs)

    def _successful_stage(self, stage: Mapping[str, Any]) -> SuccessfulStudentStage | None:
        stage_record = dict(stage)
        stage_record["stage"] = canonical_stage_type(str(stage_record["stage"]))
        contract = student_stage_runtime_node_contract(self._runtime_state, stage_record)
        if not contract.marker_path.exists():
            return None
        try:
            if not contract.artifact_check():
                return None
        except Exception:
            return None
        return SuccessfulStudentStage(
            teacher_round=int(stage_record["teacher_round"]),
            student_round=int(stage_record["student_round"]),
            stage_run_index=int(stage_record["stage_run_index"]),
            stage_type=str(stage_record["stage"]),
            stage_dir=Path(str(stage_record["stage_dir"])).resolve(),
            marker_path=contract.marker_path,
            stage_record=stage_record,
        )

    def _stages_before(self, stage: Mapping[str, Any]) -> list[dict[str, Any]]:
        current_key = _stage_sort_key(stage)
        return [item for item in self._ordered_runtime_stages() if _stage_sort_key(item) < current_key]

    def _ordered_runtime_stages(self) -> list[dict[str, Any]]:
        stages: list[dict[str, Any]] = []
        for invocation in list(self._runtime_state.get("invocations", []) or []):
            stages.extend(dict(stage) for stage in list(invocation.get("stages", []) or []))
        return sorted(stages, key=_stage_sort_key)

    def _scan_stage_dirs(self, teacher_round: int) -> tuple[dict[str, Any], ...]:
        subworkflow_root = Path(self._paths.student_subworkflow_artifacts_dir(teacher_round)).resolve()
        if not subworkflow_root.exists():
            return ()
        stages: list[dict[str, Any]] = []
        for stage_dir in subworkflow_root.glob("student_round_*/stage_*_*"):
            if not stage_dir.is_dir():
                continue
            parsed = _parse_stage_dir(stage_dir)
            if parsed is None:
                continue
            student_round, stage_run_index, stage_type = parsed
            stages.append(
                {
                    "teacher_round": int(teacher_round),
                    "student_round": int(student_round),
                    "stage_run_index": int(stage_run_index),
                    "stage": canonical_stage_type(stage_type),
                    "stage_dir": str(stage_dir.resolve()),
                }
            )
        return tuple(sorted(stages, key=_stage_sort_key))


def _stage_sort_key(stage: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(stage["teacher_round"]), int(stage["student_round"]), int(stage["stage_run_index"])


def _parse_stage_dir(stage_dir: Path) -> tuple[int, int, str] | None:
    round_match = re.fullmatch(r"student_round_(\d+)", stage_dir.parent.name)
    stage_match = re.fullmatch(r"stage_(\d+)_(.+)", stage_dir.name)
    if not round_match or not stage_match:
        return None
    return int(round_match.group(1)), int(stage_match.group(1)), stage_match.group(2)


def _successor_path(stage: SuccessfulStudentStage) -> Path:
    descriptor = student_stage_contract_descriptor(stage.stage_type)
    for role in descriptor.successor_roles:
        successor = descriptor.successor_path(role, stage.stage_dir)
        if successor is not None:
            return successor
    raise ValueError(f"student pipeline stage has no successor artifact role: {stage.stage_type!r}")
