from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Mapping

from workflow_backends.first_principles.protocols import (
    FPLabelBatchResult,
    FPLabelRequest,
    FPParseRequest,
    FPPrepareRequest,
    FPPreparedTask,
    FirstPrinciplesCalculatorAdapter,
    fp_task_id,
)
from workflow_core.lifecycle.fp_capabilities import (
    FirstPrinciplesLabelArtifactFacts,
    FirstPrinciplesLabelFact,
    FirstPrinciplesLabelIntent,
    FirstPrinciplesLabelOutcome,
    FirstPrinciplesSelection,
    require_fp_label_artifacts,
    validate_fp_label_outcome,
)
from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.artifact_handles import StructureBatchHandle
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)
from workflow_artifacts.lifecycle.fp_task_result import (
    read_validated_fp_result,
    validate_fp_label_result,
    write_validated_fp_result,
)
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.local_fp_executor import (
    FPLocalTaskCommand,
    LocalFPExecutor,
)
from workflow_runtime.lifecycle.artifact_validation import BoundArtifactValidator


class FPTaskExecutionError(RuntimeError):
    pass


class LocalFirstPrinciplesLabelService:
    """Calculator-neutral prepare/run/parse façade for one local CPU pool."""

    def __init__(
        self,
        *,
        adapter: FirstPrinciplesCalculatorAdapter,
        operation_context: OperationContext,
    ) -> None:
        if adapter.operation_role != operation_context.role:
            raise ValueError(
                "first-principles adapter role does not match operation context"
            )
        self.adapter = adapter
        self.operation_context = operation_context
        self.calculator = adapter.calculator
        self.operation_role = adapter.operation_role

    def bind_capability(
        self, request: FPLabelRequest
    ) -> BoundFirstPrinciplesLabelCapability:
        """Bind runtime locations outside the logic-owned semantic request."""

        return BoundFirstPrinciplesLabelCapability(self, request)

    def label(self, request: FPLabelRequest) -> FPLabelBatchResult:
        self._validate_request(request)
        controller_root = Path(request.controller_root)
        recovery_task_root = controller_root / "tasks"
        recovery_task_root.mkdir(parents=True, exist_ok=True)
        prepared = self.adapter.prepare(
            FPPrepareRequest(
                selected_path=request.selected_path,
                task_root=recovery_task_root,
                recipe_path=request.recipe_path,
                operation_context=self.operation_context,
            )
        )
        self._validate_prepared(request, prepared)
        self._write_or_validate_plan(request, prepared)
        selected = read_extxyz_structures(request.selected_path, allow_empty=True)
        results_by_task = {
            task.task_id: cached
            for task, source in zip(prepared, selected, strict=True)
            if (cached := read_validated_fp_result(task, source=source)) is not None
        }
        incomplete = tuple(
            task for task in prepared if task.task_id not in results_by_task
        )
        commands = tuple(
            FPLocalTaskCommand(
                task_id=task.task_id,
                workspace=task.task_path,
                input_digest=task.input_digest,
                logical_executable=task.logical_executable,
                arguments=task.arguments,
            )
            for task in incomplete
        )
        completed = LocalFPExecutor(self.operation_context).run(commands)
        failed = []
        for task, terminal in zip(incomplete, completed, strict=True):
            if terminal.returncode != 0:
                failed.append(terminal)
                continue
            result = self.adapter.parse(
                FPParseRequest(prepared_task=task, completed_task=terminal)
            )
            if result.required_labels != task.required_labels:
                raise ValueError(
                    f"first-principles parsed label contract mismatch for {task.task_id}"
                )
            validate_fp_label_result(selected[task.selected_index], result)
            write_validated_fp_result(task, result)
            results_by_task[task.task_id] = result
        if failed:
            detail = ", ".join(
                f"{item.task_id}=exit-{item.returncode}" for item in failed
            )
            raise FPTaskExecutionError(
                f"local first-principles task failure; no labeled artifact published: {detail}"
            )
        results = tuple(results_by_task[task.task_id] for task in prepared)
        self._copy_publication_tasks(prepared, Path(request.task_root))
        summary_evidence = {
            "calculator": self.calculator,
            "task_count": len(prepared),
            "fp_workers": self.operation_context.runtime["fp_workers"],
            "cpu_threads": self.operation_context.runtime["cpu_threads"],
        }
        convergence = {
            result.task_id: result.parser_evidence.get("scf_converged")
            for result in results
            if isinstance(result.parser_evidence.get("scf_converged"), bool)
        }
        if convergence:
            unconverged = [
                task_id for task_id, converged in convergence.items() if not converged
            ]
            summary_evidence.update(
                {
                    "scf_converged": not unconverged,
                    "scf_unconverged_tasks": unconverged,
                }
            )
        return FPLabelBatchResult(
            results=results,
            summary_evidence=summary_evidence,
        )

    def _validate_request(self, request: FPLabelRequest) -> None:
        if (
            request.calculator != self.calculator
            or request.operation_role != self.operation_role
            or request.operation_context not in (None, self.operation_context)
            or not request.operation_id
        ):
            raise ValueError("local first-principles label request does not match service")

    def _validate_prepared(
        self, request: FPLabelRequest, prepared: tuple[FPPreparedTask, ...]
    ) -> None:
        selected = read_extxyz_structures(request.selected_path, allow_empty=True)
        if len(prepared) != len(selected):
            raise ValueError("prepared FP tasks do not exactly cover selected structures")
        recovery_root = (Path(request.controller_root) / "tasks").resolve()
        for index, (source, task) in enumerate(zip(selected, prepared, strict=True)):
            try:
                task_root = Path(task.task_path).resolve().parent
            except OSError as exc:
                raise ValueError("prepared FP task path is invalid") from exc
            if (
                task.task_id != fp_task_id(index)
                or task.selected_index != index
                or task.structure_id != str(source.info.get("structure_id", ""))
                or task.system_id != str(source.info.get("system_id", ""))
                or task_root != recovery_root
                or not Path(task.task_path).is_dir()
                or not task.input_digest
                or task.logical_executable != self.adapter.logical_executable
            ):
                raise ValueError(f"invalid prepared FP task inventory at index {index}")

    def _write_or_validate_plan(
        self, request: FPLabelRequest, prepared: tuple[FPPreparedTask, ...]
    ) -> None:
        path = Path(request.controller_root) / "plan.json"
        semantic = {
            "schema": "dice.fp-label-controller-plan",
            "schema_revision": 1,
            "operation_id": request.operation_id,
            "calculator": self.calculator,
            "operation_role": self.operation_role,
            "task_inputs": [
                {"task_id": task.task_id, "input_digest": task.input_digest}
                for task in prepared
            ],
        }
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid durable FP controller plan") from exc
            if existing != semantic:
                raise ValueError("durable FP controller plan does not match immutable inputs")
            return
        staged = path.with_suffix(".tmp")
        staged.write_text(json.dumps(semantic, sort_keys=True), encoding="utf-8")
        staged.replace(path)

    @staticmethod
    def _copy_publication_tasks(
        prepared: tuple[FPPreparedTask, ...], publication_root: Path
    ) -> None:
        if publication_root.exists():
            raise ValueError("FP publication task root must be new staging")
        publication_root.mkdir(parents=True)
        for task in prepared:
            shutil.copytree(task.task_path, publication_root / task.task_id)


class BoundFirstPrinciplesLabelCapability:
    """Translate one bound runtime request to the path-free lifecycle port."""

    def __init__(self, service: Any, request: FPLabelRequest) -> None:
        self.service = service
        self.request = request

    def label(
        self, intent: FirstPrinciplesLabelIntent
    ) -> FirstPrinciplesLabelOutcome:
        selected = read_extxyz_structures(
            self.request.selected_path, allow_empty=True
        )
        expected_selection = FirstPrinciplesSelection(
            StructureBatchHandle(
                intent.selection.batch.batch_id,
                tuple(
                    str(source.info.get("structure_id", ""))
                    for source in selected
                ),
            ),
            tuple(str(source.info.get("system_id", "")) for source in selected),
        )
        if (
            intent.selection != expected_selection
            or intent.calculator != self.request.calculator
            or intent.operation_role != self.request.operation_role
            or intent.operation_id != self.request.operation_id
        ):
            raise ValueError(
                "first-principles semantic intent does not match bound runtime request"
            )
        batch = self.service.label(self.request)
        labels = []
        for result in batch.results:
            convergence = result.parser_evidence.get("scf_converged")
            labels.append(
                FirstPrinciplesLabelFact(
                    task_id=result.task_id,
                    selected_index=result.selected_index,
                    structure_id=result.structure_id,
                    system_id=result.system_id,
                    input_digest=result.input_digest,
                    required_labels=result.required_labels,
                    scf_converged=(
                        convergence if isinstance(convergence, bool) else None
                    ),
                )
            )
        return FirstPrinciplesLabelOutcome(tuple(labels))


@dataclass(slots=True)
class BoundFirstPrinciplesLabelPublication:
    """Bind one FP label handler to runtime paths and E publication validation."""

    service: Any
    request: FPLabelRequest
    expected_artifacts: ExpectedArtifactSemantics[
        FirstPrinciplesLabelArtifactFacts
    ]
    artifact_validator: BoundArtifactValidator[FirstPrinciplesLabelArtifactFacts]

    def __post_init__(self) -> None:
        if self.artifact_validator.expectation != self.expected_artifacts:
            raise ValueError(
                "FP label publication must reuse the lifecycle-bound artifact expectation"
            )

    def execute(
        self,
        intent: FirstPrinciplesLabelIntent,
        *,
        stage_root: str | Path,
    ) -> tuple[FirstPrinciplesLabelOutcome, Mapping[str, Any]]:
        require_fp_label_artifacts(self.expected_artifacts, intent=intent)
        root = Path(stage_root).resolve()
        if Path(self.request.task_root).resolve() != root / "tasks":
            raise ValueError("FP label task publication is not bound to the staging root")
        selected = read_extxyz_structures(self.request.selected_path, allow_empty=True)
        expected_selection = FirstPrinciplesSelection(
            StructureBatchHandle(
                intent.selection.batch.batch_id,
                tuple(str(frame.info.get("structure_id", "")) for frame in selected),
            ),
            tuple(str(frame.info.get("system_id", "")) for frame in selected),
        )
        if intent.selection != expected_selection:
            raise ValueError(
                "FP label selected artifact changed canonical handle or system order"
            )

        batch = self.service.label(self.request)
        results_by_index = {}
        for result in batch.results:
            if result.selected_index in results_by_index:
                raise ValueError(
                    "duplicate first-principles result for selected index "
                    f"{result.selected_index}"
                )
            results_by_index[result.selected_index] = result
        if set(results_by_index) != set(range(len(selected))):
            raise ValueError(
                "first-principles results do not exactly cover selected structures"
            )
        ordered_results = tuple(
            results_by_index[index] for index in range(len(selected))
        )
        labeled = []
        for index, (source, result) in enumerate(
            zip(selected, ordered_results, strict=True)
        ):
            expected = intent.selection.structures[index]
            if (
                result.task_id != fp_task_id(index)
                or result.selected_index != index
                or result.structure_id != expected.structure_id
                or result.system_id != expected.system_id
            ):
                raise ValueError(
                    "first-principles result changed canonical task/structure/system order"
                )
            validate_fp_label_result(source, result)
            item = result.labeled_atoms
            item.info["structure_id"] = expected.structure_id
            item.info["system_id"] = expected.system_id
            labeled.append(item)
        write_extxyz_structures(root / "labeled.xyz", labeled, allow_empty=True)
        labels = []
        for result in ordered_results:
            convergence = result.parser_evidence.get("scf_converged")
            labels.append(
                FirstPrinciplesLabelFact(
                    task_id=result.task_id,
                    selected_index=result.selected_index,
                    structure_id=result.structure_id,
                    system_id=result.system_id,
                    input_digest=result.input_digest,
                    required_labels=result.required_labels,
                    scf_converged=(
                        convergence if isinstance(convergence, bool) else None
                    ),
                )
            )
        outcome = FirstPrinciplesLabelOutcome(tuple(labels))
        accepted = validate_fp_label_outcome(intent, outcome)
        if not self.artifact_validator(root):
            raise ValueError("FP label publication failed bound artifact validation")
        return accepted, dict(batch.summary_evidence)


__all__ = [
    "BoundFirstPrinciplesLabelPublication",
    "BoundFirstPrinciplesLabelCapability",
    "FPTaskExecutionError",
    "LocalFirstPrinciplesLabelService",
]
