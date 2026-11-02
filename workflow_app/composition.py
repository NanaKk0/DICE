from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Protocol

from workflow_backends.sampling import (
    SamplingBackendBinding,
    SamplingBackendFactory,
    SamplingRecipeSource,
)
from workflow_backends.student.role import (
    StudentBackendBinding,
    StudentBackendFactory,
)
from workflow_backends.teacher.role import (
    TeacherBackendBinding,
    TeacherBackendFactory,
)
from workflow_core.lifecycle.launch import LifecycleLaunch
from workflow_core.lifecycle.state_model import PublicBlockId, PublicLifecyclePlan
from workflow_core.lifecycle.use_cases import (
    LifecycleRuntimeCapability,
    PublicBlockOutcome,
)
from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
from workflow_core.lifecycle.screen_module import LifecycleScreenModule
from workflow_core.lifecycle.student_facade import LifecycleStudentFacade
from workflow_core.lifecycle.teacher_facade import LifecycleTeacherFacade
from workflow_app.screen_io import LifecycleScreenIOAdapter


def compose_screen_io_adapter(
    *, config: Any, teacher: Any, student: Any, gmm_score_chunk_size: int
) -> LifecycleScreenIOAdapter:
    """Construct physical screening DTO factories at the sole app boundary."""
    from workflow_backends.student.readout import StudentReadoutRequest
    from workflow_backends.teacher.label_result import TeacherLabelRequest
    from workflow_backends.teacher.readout import TeacherReadoutScoreRequest
    from workflow_artifacts.lifecycle.gmm_io import validate_gmm_router
    from workflow_artifacts.lifecycle.screen_publication import (
        ScreenArtifactPublicationCapability,
    )
    from workflow_runtime.lifecycle.cpu_pool import CPUWorkPool

    return LifecycleScreenIOAdapter.bind(
        config=config,
        teacher=teacher,
        student=student,
        gmm_score_chunk_size=gmm_score_chunk_size,
        student_readout_request_factory=StudentReadoutRequest,
        teacher_score_request_factory=TeacherReadoutScoreRequest,
        teacher_label_request_factory=TeacherLabelRequest,
        router_validator=validate_gmm_router,
        artifacts=ScreenArtifactPublicationCapability(),
        cpu_pool_factory=CPUWorkPool,
    )
from workflow_runtime.lifecycle.environment import OperationContext, operation_contexts
from workflow_runtime.lifecycle.state import LifecycleState
from workflow_runtime.lifecycle.state_adapter import ManagedStateLifecycleRuntime
from workflow_runtime.lifecycle.state_adapter import LifecycleStateBindingController


class SelectedBackendAdapters(Protocol):
    """Backend-owned values selected from exact operation contexts."""

    teacher: TeacherBackendFactory
    student: StudentBackendFactory
    sampling: SamplingBackendFactory
    first_principles: Any | None


class LifecycleRuntimeFactory(Protocol):
    def __call__(
        self,
        *,
        state: LifecycleState | None,
        state_controller: LifecycleStateBindingController | None,
        handlers: Mapping[str, Callable[[Path], object]],
        reporter_factory: Callable[[PublicBlockId], object],
        phase_callback: Callable[[PublicBlockId, str], None],
        failure_callback: Callable[[PublicBlockId, Path, BaseException], None],
        success_callback: Callable[[PublicBlockOutcome], None],
    ) -> LifecycleRuntimeCapability: ...


def _validate_gmm_router_at_app_boundary(path: str | Path):
    from workflow_artifacts.lifecycle.gmm_io import validate_gmm_router

    return validate_gmm_router(path)


def _validate_screen_at_app_boundary(request):
    from workflow_artifacts.lifecycle.screen_validation import validate_screen_artifacts
    from workflow_runtime.lifecycle.cpu_pool import CPUWorkPool

    with CPUWorkPool() as pool:
        return validate_screen_artifacts(replace(request, ordered_map=pool.map))


def _publish_joint_direct_fp_at_app_boundary(path: str | Path, **kwargs):
    from workflow_artifacts.lifecycle.joint_direct_fp import (
        publish_joint_direct_fp_evidence,
    )

    return publish_joint_direct_fp_evidence(path, **kwargs)


def _publish_element_direct_fp_at_app_boundary(path: str | Path, **kwargs):
    from workflow_artifacts.lifecycle.element_direct_fp import (
        publish_element_direct_fp_evidence,
    )

    return publish_element_direct_fp_evidence(path, **kwargs)


@dataclass(frozen=True, slots=True)
class BackendComposition:
    """The one selected set of roles and capabilities passed to lifecycle core."""

    validate_screen_artifacts = staticmethod(_validate_screen_at_app_boundary)

    teacher: LifecycleTeacherFacade
    student: LifecycleStudentFacade
    sampling: LifecycleSamplingCoordinator
    screen: LifecycleScreenModule
    first_principles: Any | None = None
    observation_chunk_size: int = 64
    gmm_score_chunk_size: int = 500
    lifecycle_runtime_factory: LifecycleRuntimeFactory = ManagedStateLifecycleRuntime
    gmm_router_validator: Callable[..., Any] = _validate_gmm_router_at_app_boundary
    joint_direct_fp_publisher: Callable[..., Any] = (
        _publish_joint_direct_fp_at_app_boundary
    )
    element_direct_fp_publisher: Callable[..., Any] = (
        _publish_element_direct_fp_at_app_boundary
    )

    @staticmethod
    def build_lifecycle_state_controller(
        *,
        root: str | Path,
        iter_index: int,
        plan: PublicLifecyclePlan,
        binding_specs_factory: Callable[
            [tuple[PublicBlockId, ...]],
            tuple[tuple[PublicBlockId, object], ...],
        ],
    ) -> LifecycleStateBindingController:
        """Assemble runtime-owned state bindings without exposing paths to core."""

        from workflow_runtime.lifecycle.state_adapter import (
            LifecycleStateBindingProvider,
            bind_lifecycle_runtime_block,
        )

        def bindings(completed_prefix: tuple[PublicBlockId, ...]):
            return tuple(
                bind_lifecycle_runtime_block(
                    block=block,
                    artifact_validator=validator,
                )
                for block, validator in binding_specs_factory(
                    completed_prefix
                )
            )

        return LifecycleStateBindingController(
            root=root,
            iter_index=iter_index,
            plan=plan,
            provider=LifecycleStateBindingProvider(plan, bindings),
        )

    @staticmethod
    def adapt_compatibility_lifecycle_state(
        *,
        root: str | Path,
        iter_index: int,
        binding_specs: tuple[tuple[PublicBlockId, object], ...],
    ) -> LifecycleState:
        """Keep the injected non-public seam behind the runtime adapter."""

        from workflow_runtime.lifecycle.state_adapter import (
            adapt_lifecycle_state,
            bind_lifecycle_runtime_block,
        )

        return adapt_lifecycle_state(
            root=root,
            iter_index=iter_index,
            bindings=tuple(
                bind_lifecycle_runtime_block(
                    block=block,
                    artifact_validator=validator,
                )
                for block, validator in binding_specs
            ),
        )

    def bind_teacher_training_artifact_validator(self, expectation: Any):
        """Bind D03 dataset/model facts to the shared runtime validator."""

        from workflow_core.lifecycle.teacher_capabilities import (
            TeacherTrainingArtifactFacts,
        )
        from workflow_artifacts.lifecycle.teacher_training_validation import (
            validate_teacher_training_publication_dataset,
        )
        from workflow_runtime.lifecycle.artifact_validation import (
            BoundArtifactValidator,
        )

        def validate(root: Path, facts: TeacherTrainingArtifactFacts) -> bool:
            if not validate_teacher_training_publication_dataset(root, facts):
                return False
            return self.teacher.validate_published_training(
                stage_root=root,
                member_ids=tuple(model.member_id for model in facts.models),
            )

        return BoundArtifactValidator(expectation, validate)

    @staticmethod
    def bind_teacher_training_recovery_validator(facts: Any):
        """Bind path-free teacher settings to bounded canonical recovery."""

        from workflow_core.lifecycle.teacher_capabilities import (
            TeacherTrainingRecoveryFacts,
        )
        from workflow_artifacts.lifecycle.teacher_recovery import (
            validate_teacher_training_artifacts,
        )

        if not isinstance(facts, TeacherTrainingRecoveryFacts):
            raise TypeError("teacher train recovery requires logic-owned facts")
        return lambda root: validate_teacher_training_artifacts(root, facts)

    @staticmethod
    def bind_teacher_screen_recovery_validator(facts: Any):
        """Bind path-free teacher-screen settings to canonical recovery."""

        from workflow_core.lifecycle.teacher_capabilities import (
            TeacherScreenRecoveryFacts,
        )
        from workflow_artifacts.lifecycle.teacher_recovery import (
            validate_teacher_screen_artifacts,
        )

        if not isinstance(facts, TeacherScreenRecoveryFacts):
            raise TypeError("teacher screen recovery requires logic-owned facts")
        return lambda root: validate_teacher_screen_artifacts(root, facts)

    @staticmethod
    def bind_teacher_scan_recovery_validator(facts: Any):
        """Bind the configured teacher element order to strict router recovery."""

        from workflow_core.lifecycle.teacher_capabilities import TeacherScanRecoveryFacts
        from workflow_artifacts.lifecycle.teacher_recovery import (
            validate_teacher_scan_artifacts,
        )

        if not isinstance(facts, TeacherScanRecoveryFacts):
            raise TypeError("teacher scan recovery requires logic-owned facts")
        return lambda root: validate_teacher_scan_artifacts(root, facts)

    @staticmethod
    def bind_student_training_recovery_validator(facts: Any):
        """Bind path-free student settings to canonical train recovery."""

        from workflow_core.lifecycle.student_capabilities import (
            StudentTrainingRecoveryFacts,
        )
        from workflow_artifacts.lifecycle.student_recovery import (
            validate_student_training_artifacts,
        )

        if not isinstance(facts, StudentTrainingRecoveryFacts):
            raise TypeError("student train recovery requires logic-owned facts")
        return lambda root: validate_student_training_artifacts(root, facts)

    @staticmethod
    def bind_sample_recovery_validator(facts: Any):
        """Bind a normalized sample plan to backend-owned committed evidence."""

        from workflow_core.lifecycle.sampling import SampleRecoveryFacts
        from workflow_backends.sample_recovery import validate_sample_artifacts

        if not isinstance(facts, SampleRecoveryFacts):
            raise TypeError("sample recovery requires logic-owned facts")
        return lambda root: validate_sample_artifacts(root, facts)

    @staticmethod
    def bind_final_fp_recovery_validator(
        facts: Any,
        *,
        candidate_paths: tuple[Path, ...],
        teacher_router: Path | None,
        teacher_training_path: Path | None = None,
    ):
        """Bind terminal settings to committed screen/router publications."""

        from workflow_core.lifecycle.fp_capabilities import FinalFPRecoveryFacts
        from workflow_artifacts.lifecycle.fp_recovery import (
            validate_final_fp_artifacts,
        )

        if not isinstance(facts, FinalFPRecoveryFacts):
            raise TypeError("terminal FP recovery requires logic-owned facts")

        def validate(root: Path) -> bool:
            return validate_final_fp_artifacts(
                fp_root=root,
                candidate_paths=candidate_paths,
                facts=facts,
                teacher_router_path=teacher_router,
                teacher_training_path=teacher_training_path,
            )

        return validate

    def execute_teacher_training_expectation(
        self,
        expectation: Any,
        *,
        intent: Any,
        physical_request: Any,
        stage_root: str | Path,
    ) -> Any:
        """Execute one canonical teacher train intent through D03 bindings."""

        from workflow_artifacts.lifecycle.model_publication import (
            ResolvedModelArtifact,
        )
        from workflow_backends.teacher.handle_binding import (
            BoundTeacherTrainingCapability,
        )
        from workflow_core.lifecycle.teacher_facade import TeacherTrainingUseCase

        foundation = None
        if intent.foundation_model is not None:
            asset = physical_request.foundation_asset
            if asset is None:
                raise ValueError(
                    "teacher foundation semantic handle requires a physical asset"
                )
            foundation = ResolvedModelArtifact(
                intent.foundation_model,
                asset.path,
                _sha256_file(asset.path),
            )
        validator = self.bind_teacher_training_artifact_validator(expectation)
        capability = BoundTeacherTrainingCapability(
            delegate=self.teacher.delegate,
            semantic_intent=intent,
            physical_request=physical_request,
            stage_root=Path(stage_root),
            expected_artifacts=expectation,
            artifact_validator=validator,
            foundation_artifact=foundation,
        )
        return TeacherTrainingUseCase(capability).execute(intent, expectation)

    @staticmethod
    def bind_fp_label_artifact_validator(
        expectation: Any,
        *,
        selected_path: str | Path,
        calculator: str,
    ):
        """Bind D03 FP facts to the existing schema-specific validator."""

        from workflow_artifacts.lifecycle.fp_label_validation import (
            validate_fp_label_artifacts,
        )
        from workflow_core.lifecycle.fp_capabilities import (
            FirstPrinciplesLabelArtifactFacts,
        )
        from workflow_runtime.lifecycle.artifact_validation import (
            BoundArtifactValidator,
        )

        selected = Path(selected_path).resolve()
        normalized_calculator = str(calculator).upper()

        def validate(root: Path, facts: FirstPrinciplesLabelArtifactFacts) -> bool:
            return validate_fp_label_artifacts(
                label_root=root,
                selected_path=selected,
                calculator=normalized_calculator,
                facts=facts,
            )

        return BoundArtifactValidator(expectation, validate)

    def execute_fp_label_expectation(
        self,
        expectation: Any,
        *,
        intent: Any,
        physical_request: Any,
        stage_root: str | Path,
    ) -> Any:
        """Execute one canonical FP label intent through D03 bindings."""

        if self.first_principles is None:
            raise ValueError(
                "lifecycle fp/label execution requires an explicit first-principles service"
            )
        from workflow_runtime.lifecycle.fp_service import (
            BoundFirstPrinciplesLabelPublication,
        )

        validator = self.bind_fp_label_artifact_validator(
            expectation,
            selected_path=physical_request.selected_path,
            calculator=physical_request.calculator,
        )
        return BoundFirstPrinciplesLabelPublication(
            self.first_principles,
            physical_request,
            expectation,
            validator,
        ).execute(intent, stage_root=stage_root)


ExecutorFactory = Callable[[Mapping[str, OperationContext]], object]


def compose_backend_roles(
    config: Any,
    selected: SelectedBackendAdapters,
) -> BackendComposition:
    """Bind selected backend adapters to route-resolved role inputs."""

    selected_teacher = selected.teacher.bind(_teacher_binding(config))
    from workflow_core.lifecycle.data import LabelContract

    teacher_contract = LabelContract()
    teacher = LifecycleTeacherFacade(
        delegate=selected_teacher,
        label_contract=teacher_contract,
    )
    selected_student = selected.student.bind(_student_binding(config))
    student = LifecycleStudentFacade(delegate=selected_student)
    screen = LifecycleScreenModule.bind(
        config=config,
        io=compose_screen_io_adapter(
            config=config,
            teacher=teacher,
            student=student,
            gmm_score_chunk_size=500,
        ),
        observation_chunk_size=64,
        gmm_score_chunk_size=500,
    )
    return BackendComposition(
        teacher=teacher,
        student=student,
        sampling=LifecycleSamplingCoordinator(
            selected.sampling.bind(_sampling_binding(config)),
            student.sampling_models,
        ),
        screen=screen,
        first_principles=selected.first_principles,
    )


def compose_observation_backend_roles(launch: LifecycleLaunch) -> BackendComposition:
    """Select the read-only provider set for a prepared probe invocation."""

    if launch.preflight is None:
        raise ValueError("lifecycle probe requires --environment or DICE_ENVIRONMENT")
    from workflow_backends.factory import build_observation_backend_adapters

    selected = build_observation_backend_adapters(
        launch.config,
        _operation_contexts(launch),
    )
    return compose_backend_roles(launch.config, selected)


def compose_executor_factory_roles(
    launch: LifecycleLaunch,
    executor_factory: ExecutorFactory,
) -> BackendComposition:
    """Bind a compatibility factory result at the application composition root."""

    if launch.preflight is None:
        raise ValueError(
            "lifecycle executor factory requires --environment or DICE_ENVIRONMENT"
        )
    selected = executor_factory(_operation_contexts(launch))
    return _bind_selected_backends(launch, selected)


def compose_selected_backend_roles(launch: LifecycleLaunch) -> BackendComposition:
    """Select and bind the concrete provider set for one execute invocation."""

    if launch.preflight is None:
        raise ValueError("lifecycle execution requires a prepared environment preflight")
    from workflow_backends.factory import build_selected_backend_adapters
    from workflow_core.lifecycle.launch import required_operation_roles

    selected = build_selected_backend_adapters(
        _operation_contexts(launch),
        dormant_roles=required_operation_roles(launch.config),
    )
    return compose_backend_roles(launch.config, selected)


def _operation_contexts(launch: LifecycleLaunch) -> dict[str, OperationContext]:
    if launch.preflight is None:
        raise ValueError("lifecycle operation contexts require environment preflight")
    allocations = None
    if launch.execution_profile is not None:
        from workflow_runtime.execution.allocation import (
            allocations_by_operation_role,
        )

        allocations = allocations_by_operation_role(
            launch.execution_profile,
            launch.execution_plans,
        )
    return operation_contexts(launch.preflight, allocations=allocations)


def is_slurm_lifecycle_dispatch(launch: LifecycleLaunch, block_id: str) -> bool:
    """Classify one prepared public block without constructing a scheduler."""

    from workflow_runtime.execution.lifecycle_slurm import is_slurm_lifecycle_block

    return is_slurm_lifecycle_block(launch, block_id)


def dispatch_slurm_lifecycle_block(
    launch: LifecycleLaunch,
    *,
    block_id: str,
    commands: object | None = None,
    backend_composition: BackendComposition | None = None,
):
    """Bind the durable Slurm provider to ordinary lifecycle completion authority."""

    from workflow_core.lifecycle.runner import observe_public_lifecycle_block
    from workflow_runtime.execution.lifecycle_slurm import (
        execute_slurm_lifecycle_block,
    )

    def canonical_validator(_collected: Path) -> bool:
        observation = observe_public_lifecycle_block(
            launch,
            block_id,
            backend_composition=backend_composition,
        )
        return observation.state == "succeeded"

    return execute_slurm_lifecycle_block(
        launch,
        block_id=block_id,
        canonical_validator=canonical_validator,
        commands=commands,
    )


def bind_lifecycle_screen(
    config: Any,
    composition: BackendComposition,
) -> BackendComposition:
    """Bind the invocation's validated route/config to the existing role facades."""

    screen = LifecycleScreenModule.bind(
        config=config,
        io=compose_screen_io_adapter(
            config=config,
            teacher=composition.teacher,
            student=composition.student,
            gmm_score_chunk_size=composition.gmm_score_chunk_size,
        ),
        observation_chunk_size=composition.observation_chunk_size,
        gmm_score_chunk_size=composition.gmm_score_chunk_size,
    )
    return BackendComposition(
        teacher=composition.teacher,
        student=composition.student,
        sampling=composition.sampling,
        screen=screen,
        first_principles=composition.first_principles,
        observation_chunk_size=composition.observation_chunk_size,
        gmm_score_chunk_size=composition.gmm_score_chunk_size,
        lifecycle_runtime_factory=composition.lifecycle_runtime_factory,
        gmm_router_validator=composition.gmm_router_validator,
        joint_direct_fp_publisher=composition.joint_direct_fp_publisher,
    )


def _bind_selected_backends(
    launch: LifecycleLaunch,
    selected: object,
) -> BackendComposition:
    if isinstance(selected, BackendComposition):
        return selected
    from workflow_backends.factory import SelectedBackendAdapters

    if not isinstance(selected, SelectedBackendAdapters):
        raise TypeError(
            "lifecycle executor factory must return selected backend adapters or a backend composition"
        )
    return compose_backend_roles(launch.config, selected)


def _sampling_binding(config: Any) -> SamplingBackendBinding:
    sampler = config.sampler
    sample = config.raw["sample"]
    if "condition_groups" in sample:
        recipes = tuple(
            SamplingRecipeSource(
                path=config.root / (group['md']['recipe'] if group.get('kind') == 'gcmc_md' else group['recipe']),
                condition_group_id=str(group_id),
            )
            for group_id, group in sample["condition_groups"].items()
            if "recipe" in group or (group.get('kind') == 'gcmc_md' and 'md' in group)
        )
    elif "recipe" in sampler:
        recipes = (SamplingRecipeSource(path=config.root / sampler["recipe"]),)
    else:
        recipes = tuple(
            SamplingRecipeSource(
                path=config.root / recipe,
                ensemble=str(ensemble),
            )
            for ensemble, recipe in sampler["recipes"].items()
        )
    systems = sample.get("systems", {})
    return SamplingBackendBinding(
        recipes=recipes,
        type_map=tuple(config.student["type_map"]),
        system_inputs={
            str(system_id): Path(config.root / system["plumed"])
            for system_id, system in systems.items()
            if "plumed" in system
        },
        widom_group_ids=tuple(k for k, g in sample.get("condition_groups", {}).items() if g.get("kind") == "widom"),
        guest_inputs={
            str(group_id): config.root / (group['guest'] if group.get('kind') == 'widom' else group['mc']['guest'])
            for group_id, group in sample.get('condition_groups', {}).items()
            if group.get('kind') in {'gcmc_md', 'widom'}
        },
    )


def sampling_backend_binding(config: Any) -> SamplingBackendBinding:
    """Return route-resolved inputs for a selected sampling role factory."""

    return _sampling_binding(config)


def preflight_sampling_protocols(launch: LifecycleLaunch, *, first_block: str, single_block: bool = False) -> None:
    """Selected native recipe feasibility, before any lifecycle mutation."""
    config = launch.config
    plan = config.sampling_plan
    if config.sampler['backend'] != 'LAMMPS' or not hasattr(plan, 'tasks_for_round'):
        return
    blocks = tuple(block.block_id for block in launch.plan.blocks)
    selected = (first_block,) if single_block else blocks[blocks.index(first_block):]
    groups = {task.condition_group_id for block in selected if block.endswith('/sample')
              for task in plan.tasks_for_round(int(block.split('/')[0][1:])) if task.kind == 'md_mc'}
    if not groups:
        return
    from workflow_backends.lammps.identity_swap import validate_recipe
    for group_id in groups:
        declaration = config.raw['sample']['condition_groups'][group_id]
        validate_recipe((config.root/declaration['recipe']).read_text(), ensemble=declaration['ensemble'])
    if config.student['model'] == 'committee' and launch.preflight is not None:
        role = f"sampler.lammps.{config.student['cli']}"
        if 'python' not in launch.preflight.evidence[role]['executables']:
            raise ValueError('committee md_mc requires the sampler Python executable for formal-frame evaluation')


def student_backend_binding(config: Any) -> StudentBackendBinding:
    """Return the one validated student binding without selecting an implementation."""

    return _student_binding(config)


def teacher_backend_binding(config: Any) -> TeacherBackendBinding:
    """Return the validated teacher binding without selecting an implementation."""

    return _teacher_binding(config)


def _teacher_binding(config: Any) -> TeacherBackendBinding:
    teacher = config.teacher
    member_ids = _teacher_member_ids(teacher)
    return TeacherBackendBinding(
        run_json_path=config.path,
        iter_index=int(config.iter_index),
        recipe_path=config.root / teacher["recipe"],
        type_map=tuple(teacher["type_map"]),
        model_shape=str(teacher["model"]),
        member_ids=member_ids,
        policy=str(teacher["policy"]),
        foundation_model=_optional_path(
            config.root,
            teacher.get("foundation_model"),
        ),
        train_options=teacher.get("train_options", {}),
        requires_readout=bool(
            getattr(
                getattr(config, "route", None),
                "teacher_requires_reference_tail",
                teacher["model"] == "single",
            )
        ),
    )


def _teacher_member_ids(teacher: Mapping[str, object]) -> tuple[str, ...]:
    members = int(teacher.get("members", 1))
    return tuple(
        "teacher" if members == 1 else f"teacher_{index:02d}"
        for index in range(1, members + 1)
    )


def _student_binding(config: Any) -> StudentBackendBinding:
    student = config.student
    return StudentBackendBinding(
        run_json_path=config.path,
        recipe_path=config.root / student["recipe"],
        type_map=tuple(student["type_map"]),
        model_shape=str(student["model"]),
        policy=str(student["policy"]),
        foundation_model=_optional_path(config.root, student.get("foundation_model")),
        foundation_restart=_optional_path(
            config.root,
            student.get("foundation_restart"),
        ),
        train_options=student.get("train_options", {}),
        compress_sampling_models=bool(student.get("compress", False)),
        foundation_model_id=(
            None
            if student.get("foundation_model") is None
            else f"student-foundation:{student['foundation_model']}"
        ),
    )


def _optional_path(root: Path, value: object) -> Path | None:
    return None if value is None else root / str(value)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BackendComposition",
    "SelectedBackendAdapters",
    "bind_lifecycle_screen",
    "compose_backend_roles",
    "compose_executor_factory_roles",
    "compose_observation_backend_roles",
    "compose_selected_backend_roles",
    "dispatch_slurm_lifecycle_block",
    "is_slurm_lifecycle_dispatch",
    "sampling_backend_binding",
    "student_backend_binding",
    "teacher_backend_binding",
]
