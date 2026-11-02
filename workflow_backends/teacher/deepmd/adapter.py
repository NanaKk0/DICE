from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ase import Atoms

from workflow_backends.common.committee_pool import execute_committee_members
from workflow_backends.deepmd_common.training_pool import DEEPMD_TRAINING_POOL
from workflow_backends.deepmd_common.dataset import (
    materialize_native_teacher_datasets,
)
from workflow_backends.deepmd_common.type_maps import resolve_training_type_maps
from workflow_backends.deepmd_common.model_recipe import synchronize_model_random_seeds
from workflow_core.model_families import deepmd_recipe_model_family
from workflow_artifacts.lifecycle.deepmd_model_assets import (
    DPA4_MODEL_FAMILY,
    DPA4_READOUT_SOURCE_NAME,
    DeepMDModelBundleInput,
    file_sha256,
    load_deepmd_model_bundle,
)
from workflow_backends.teacher.training import (
    TeacherTrainPublicationRequest,
    TeacherTrainRequest,
    TeacherTrainResult,
)
from workflow_backends.teacher.readout import (
    TeacherCandidateScoreRequest,
    TeacherCandidateScoreResult,
    TeacherReadoutOps,
    TeacherReferenceFitRequest,
    TeacherStructureVectors,
    partitioned_teacher_structure_vectors as _teacher_structure_vectors,
)
from workflow_backends.teacher.label_result import (
    TeacherNativeLabelRequest,
    TeacherNativeLabelResult,
)
from workflow_backends.teacher.role import (
    TeacherBackendBinding,
    TeacherHistoricalModelRequest,
    TeacherHistoricalModelResult,
    TeacherPublishedTrainingValidationRequest,
)
from workflow_backends.teacher.deepmd.native import DeepMDNativeTrainResult
from workflow_artifacts.lifecycle.model_publication import (
    ModelArtifact,
    resolve_model_handles,
)
from workflow_policies.gmm_reference_tail import GMMReferenceFitResult, GMMScoreResult
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter


class DeepMDTeacherExecutor(Protocol):
    def train(
        self,
        input_json: Path,
        member_dir: Path,
        cli: str,
        *,
        foundation_model: Path | None = None,
        train_options: Mapping[str, bool] | None = None,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path | DeepMDNativeTrainResult: ...

    def label(
        self,
        models: tuple[Path, ...],
        frames: list[Atoms],
        cli: str,
        work_dir: Path,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        trusted_below: float | None = None,
    ) -> list[Atoms]: ...

    def fit_gmm_router(self, model: Path, **kwargs) -> GMMReferenceFitResult: ...

    def score_gmm_router(self, model: Path, **kwargs) -> GMMScoreResult: ...

    def score_gmm_router_chunks(self, model: Path, **kwargs) -> GMMScoreResult: ...

    def score_gmm_router_stream(self, model: Path, **kwargs): ...

    def foundation_type_map(self, model: Path, *, work_dir: Path) -> tuple[str, ...]: ...

@dataclass(slots=True)
class DeepMDTeacherAdapter:
    executor: DeepMDTeacherExecutor
    run_json_path: Path = Path("run.json")
    iter_index: int = 0
    cli: str | None = None
    work_root: Path | None = None

    def __post_init__(self) -> None:
        self.run_json_path = Path(self.run_json_path).resolve()
        self.iter_index = int(self.iter_index)

    @property
    def readout_ops(self) -> TeacherReadoutOps:
        return self

    def fit_reference(
        self, request: TeacherReferenceFitRequest
    ) -> GMMReferenceFitResult:
        readout_source = self._single_dpa4_gmm_readout_source((request.model,))
        return self.executor.fit_gmm_router(
            readout_source,
            reference_path=request.reference_path,
            router_dir=request.router_dir,
            work_dir=request.work_dir,
            type_map=request.type_map,
            model_digest=self._gmm_model_digest(request.model),
            reference_digest=file_sha256(request.reference_path),
            progress_reporter=request.progress_reporter,
        )

    def score_candidates(
        self, request: TeacherCandidateScoreRequest
    ) -> TeacherCandidateScoreResult:
        readout_source = self._single_dpa4_gmm_readout_source((request.model,))
        native = self.executor.score_gmm_router_stream(
            readout_source,
            frame_chunks=request.frame_chunks,
            frame_ids=request.ordered_identities,
            router_dir=request.router_dir,
            work_dir=request.work_dir,
            model_digest=self._gmm_model_digest(request.model),
            reference_digest=file_sha256(request.reference_path),
            emit_frame_means=request.emit_frame_means,
            progress_reporter=request.progress_reporter,
            progress_systems=request.progress_systems,
        )
        if isinstance(native, GMMScoreResult):
            identities = tuple(str(item) for item in native.frame_ids.tolist())
            compact_view = None
        else:
            identities = tuple(str(item) for item in native.identities)
            compact_view = native
        return TeacherCandidateScoreResult(
            ordered_identities=identities,
            frame_p=native.frame_p,
            frame_statistics=native.frame_statistics,
            frame_means=native.frame_means,
            descriptor_schema=native.descriptor_schema,
            compact_view=compact_view,
        )

    def read_structure_vectors(
        self,
        *,
        model: Path,
        candidate_frames: Sequence[Atoms],
        background_frames: Sequence[Atoms],
        type_map: tuple[str, ...],
        work_dir: Path,
    ) -> TeacherStructureVectors:
        readout_source = self._single_dpa4_gmm_readout_source((model,))
        return _teacher_structure_vectors(
            self.executor,
            model=readout_source,
            candidate_frames=candidate_frames,
            background_frames=background_frames,
            type_map=type_map,
            work_dir=work_dir,
        )

    def finalize_published_training(
        self, request: TeacherTrainPublicationRequest
    ) -> None:
        artifacts = request.train.member_artifacts
        if self.cli != "pt":
            if any(artifact.backend_asset is not None for artifact in artifacts):
                raise ValueError(
                    "DeepMD-TF teacher training cannot publish PT model assets"
                )
            return
        from workflow_backends.teacher.deepmd.native import (
            publish_terminal_digest_metadata,
        )

        publish_terminal_digest_metadata(request.handles)
        from workflow_artifacts.lifecycle.deepmd_model_assets import (
            publish_deepmd_model_bundle,
        )

        bundle_inputs = tuple(artifact.backend_asset for artifact in artifacts)
        if any(item is not None for item in bundle_inputs) and not all(
            isinstance(item, DeepMDModelBundleInput) for item in bundle_inputs
        ):
            raise ValueError(
                "DeepMD ordinary-DPA4 train must return one typed model bundle input per member"
            )
        for order, (artifact, handle, bundle_input) in enumerate(
            zip(artifacts, request.handles, bundle_inputs, strict=True), start=1
        ):
            if bundle_input is None:
                continue
            publish_deepmd_model_bundle(
                member_root=handle.path.parent,
                member_id=artifact.member_id,
                member_order=order,
                model_family=bundle_input.model_family,
                native_type_map=bundle_input.native_type_map,
                inference_terminal_path=handle.path,
                readout_source_path=bundle_input.readout_source_path,
                checkpoint_family=bundle_input.checkpoint_family,
                selected_checkpoint_argument=bundle_input.selected_checkpoint_argument,
                freeze_command_profile=bundle_input.freeze_command_profile,
            )

    def train(self, request: TeacherTrainRequest) -> TeacherTrainResult:
        if self.cli not in {"tf", "pt"}:
            raise ValueError("typed DeepMD lifecycle teacher adapter requires cli='tf' or cli='pt'")
        work_root = request.train_dir
        native_recipe = self._load_recipe(request.recipe_path)
        foundation_model = (
            None if request.foundation_asset is None else request.foundation_asset.path
        )
        active_type_map, native_type_map = resolve_training_type_maps(
            active_type_map=request.type_map,
            native_recipe=native_recipe,
            foundation_model=foundation_model,
            executor=self.executor,
            work_dir=work_root,
        )
        datasets = materialize_native_teacher_datasets(
            self.executor,
            train_data_path=request.train_data_path,
            validation_data_path=request.validation_data_path,
            validation_mode=request.validation_mode,
            work_root=work_root,
            type_map=active_type_map,
        )
        def execute_member(member: tuple[int, str]) -> ModelArtifact:
            member_index, member_id = member
            member_dir = work_root / member_id
            rendered = self._render(
                native_recipe,
                train_dataset_paths=datasets.train_systems,
                validation_dataset_paths=datasets.validation_systems,
                type_map=native_type_map,
                seed=self._seed(member_index),
            )
            input_json = member_dir / "input.json"
            member_dir.mkdir(parents=True, exist_ok=True)
            input_json.write_text(json.dumps(rendered, indent=2, sort_keys=True), encoding="utf-8")
            reporter = (
                request.progress_reporter_factory(member_id)
                if request.progress_reporter_factory is not None
                else None
            )
            execution = self._execute_train(
                input_json,
                member_dir,
                self.cli,
                foundation_model=(
                    foundation_model
                ),
                train_options=(
                    {"skip_neighbor_stat": True}
                    if request.train_options.get("skip_neighbor_stat") is True
                    else None
                ),
                progress_reporter=reporter,
                member_index=member_index,
                total_members=len(request.member_ids),
            )
            terminal_model = _validated_terminal_model(execution)
            execution_evidence = dict(getattr(execution, "evidence", {}) or {})
            evidence = {
                "output_kind": "deepmd_pt2" if self.cli == "pt" else "deepmd_pb",
                "input_json": str(input_json.resolve()),
                "validation_mode": request.validation_mode,
                "policy": request.policy,
                "active_type_map": active_type_map,
                "native_type_map": native_type_map,
            }
            if request.train_options:
                evidence["train_options"] = {"skip_neighbor_stat": True}
            evidence.update(execution_evidence)
            backend_asset = self._prepare_dpa4_bundle_input(
                execution,
                rendered=rendered,
                member_dir=member_dir,
            )
            return ModelArtifact(
                member_id=member_id,
                terminal_model_path=terminal_model,
                evidence=evidence,
                backend_asset=backend_asset,
            )
        members = tuple(enumerate(request.member_ids, start=1))
        artifacts = execute_committee_members(
            members,
            context=getattr(self.executor, "train_context", None),
            capability=DEEPMD_TRAINING_POOL,
            diagnostic_root=work_root,
            task_id=lambda member: member[1],
            execute=execute_member,
        )
        return TeacherTrainResult(
            member_artifacts=artifacts,
            backend_evidence={
                "backend": "deepmd",
                "cli": self.cli,
                "train_systems": tuple(str(path) for path in datasets.train_systems),
                "validation_systems": tuple(str(path) for path in datasets.validation_systems),
                "validation_mode": request.validation_mode,
                "active_type_map": active_type_map,
                "native_type_map": native_type_map,
            },
        )

    def _prepare_dpa4_bundle_input(
        self,
        execution: object,
        *,
        rendered: dict,
        member_dir: Path,
    ) -> DeepMDModelBundleInput | None:
        """Materialize the exact ordinary-DPA4 freeze source for publication."""

        if self.cli != "pt" or _rendered_model_family(rendered) != DPA4_MODEL_FAMILY:
            return None
        selected = getattr(execution, "freeze_source_checkpoint_path", None)
        selected_argument = getattr(execution, "selected_checkpoint_argument", None)
        family = getattr(execution, "checkpoint_family", None)
        profile = dict(getattr(execution, "freeze_command_profile", {}) or {})
        if selected is None:
            from workflow_backends.teacher.deepmd.native import _selected_checkpoint

            selected_argument, selected = _selected_checkpoint(
                rendered, member_dir=member_dir, cli=self.cli
            )
        selected = Path(selected)
        if selected.is_symlink() or selected.suffix != ".pt" or not selected.is_file() or selected.stat().st_size <= 0:
            raise ValueError(
                "ordinary DPA4 training must return the exact non-empty .pt checkpoint passed to freeze"
            )
        try:
            selected.resolve().relative_to(member_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "ordinary DPA4 freeze-source checkpoint must be inside the staged member directory"
            ) from exc
        if family is None:
            from workflow_backends.teacher.deepmd.native import _checkpoint_family

            family = _checkpoint_family(rendered)
        if selected_argument is None:
            selected_argument = str(selected.relative_to(member_dir.resolve()))
        if not profile:
            profile = {
                "role": "teacher.deepmd.pt.train",
                "cli": "pt",
                "freeze": "dp --pt freeze",
            }
        stable = member_dir / DPA4_READOUT_SOURCE_NAME
        from workflow_backends.common.model_files import link_model_file

        link_model_file(selected, stable)
        return DeepMDModelBundleInput(
            model_family=DPA4_MODEL_FAMILY,
            native_type_map=tuple(str(item) for item in rendered["model"]["type_map"]),
            readout_source_path=stable,
            checkpoint_family=str(family),
            selected_checkpoint_argument=str(selected_argument),
            freeze_command_profile=profile,
        )

    def _execute_train(
        self,
        input_json: Path,
        member_dir: Path,
        cli: str,
        *,
        foundation_model: Path | None,
        train_options: Mapping[str, bool] | None,
        progress_reporter: OperationProgressReporter | None,
        member_index: int,
        total_members: int,
    ) -> object:
        return self.executor.train(
            input_json,
            member_dir,
            cli,
            foundation_model=foundation_model,
            train_options=train_options,
            progress_reporter=progress_reporter,
            member_index=member_index,
            total_members=total_members,
        )

    def label(
        self, request: TeacherNativeLabelRequest
    ) -> TeacherNativeLabelResult:
        if self.cli not in {"tf", "pt"}:
            raise ValueError("DeepMD lifecycle teacher cli must be tf or pt")
        labeled = self.executor.label(
            tuple(Path(model).absolute() for model in request.models),
            list(request.frames),
            self.cli,
            request.work_dir,
            progress_reporter=request.progress_reporter,
            trusted_below=request.trusted_below,
        )
        return TeacherNativeLabelResult(tuple(labeled))

    def _gmm_model_digest(self, model: str | Path) -> str:
        """Return the GMM identity digest for the resolved backend role pair."""
        terminal = Path(model).resolve()
        readout_source = self._single_dpa4_gmm_readout_source((terminal,))
        if readout_source == terminal:
            return file_sha256(terminal)
        bundle = load_deepmd_model_bundle(member_root=terminal.parent)
        return hashlib.sha256(
            (
                "deepmd-dpa4-gmm-model@1\0"
                f"terminal:{bundle.inference_terminal_sha256}\0"
                f"readout:{bundle.readout_source_sha256}\0"
                f"receipt:{bundle.manifest_sha256}"
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _single_dpa4_gmm_readout_source(models: Sequence[str | Path]) -> Path:
        if len(models) != 1:
            raise ValueError("DPA4 lifecycle GMM operation requires exactly one model")
        terminal = Path(models[0])
        if terminal.is_symlink() or not terminal.is_file() or terminal.stat().st_size <= 0:
            raise ValueError("DPA4 lifecycle GMM terminal must be a non-empty regular file")
        if terminal.suffix != ".pt2":
            raise ValueError("DPA4 lifecycle GMM terminal must use the native .pt2 suffix")
        terminal = terminal.resolve()
        member_id = terminal.parent.name
        train_root = terminal.parent.parent
        handles = resolve_model_handles(
            stage_root=train_root,
            role="teacher",
            expected_member_ids=(member_id,),
        )
        if handles[0].path != terminal:
            raise ValueError("DPA4 lifecycle GMM model is not the declared canonical terminal")
        metadata_path = terminal.parent / "deepmd_model_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "DeepMD lifecycle GMM requires model-family metadata beside the canonical terminal"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError("DeepMD lifecycle GMM model-family metadata must be an object")
        family = metadata.get("model_family")
        native_type_map = metadata.get("type_map")
        if (
            not isinstance(native_type_map, list)
            or not native_type_map
            or any(not isinstance(item, str) or not item for item in native_type_map)
            or len(set(native_type_map)) != len(native_type_map)
        ):
            raise ValueError("DeepMD lifecycle GMM model-family metadata has an invalid type_map")
        if family == "dpa4":
            bundle = load_deepmd_model_bundle(
                member_root=terminal.parent,
                expected_member_id=member_id,
                expected_native_type_map=tuple(native_type_map),
            )
            if bundle.inference_terminal != terminal:
                raise ValueError(
                    "DeepMD lifecycle GMM bundle terminal is not the declared canonical terminal"
                )
            return bundle.readout_source
        if family == "dpa4c":
            return terminal
        raise ValueError(
            "DeepMD lifecycle GMM supports ordinary DPA4 bundles and terminal-only DPA4C models"
        )

    @staticmethod
    def _load_recipe(source: Path) -> dict:
        try:
            native_recipe = json.loads(Path(source).resolve().read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("DeepMD lifecycle recipe must be JSON") from exc
        if not isinstance(native_recipe, dict):
            raise ValueError("DeepMD lifecycle recipe must be a JSON object")
        training = native_recipe.get("training")
        if isinstance(training, dict) and "foundation_model" in training:
            raise ValueError(
                "DeepMD lifecycle recipe training.foundation_model conflicts with the resolved policy; "
                "foundation initialization must come only from TeacherTrainRequest"
            )
        return native_recipe

    def _render(
        self,
        native: dict,
        *,
        train_dataset_paths: Sequence[Path],
        validation_dataset_paths: Sequence[Path],
        type_map: tuple[str, ...],
        seed: int,
    ) -> dict:
        rendered = copy.deepcopy(native)
        model = rendered.setdefault("model", {})
        if not isinstance(model, dict):
            raise ValueError("DeepMD lifecycle recipe model must be an object")
        model["type_map"] = list(type_map)
        training = rendered.setdefault("training", {})
        if not isinstance(training, dict):
            raise ValueError("DeepMD lifecycle recipe training must be an object")
        training_data = training.setdefault("training_data", {})
        if not isinstance(training_data, dict):
            raise ValueError("DeepMD lifecycle recipe training data must be an object")
        training_data["systems"] = [str(path) for path in train_dataset_paths]
        if validation_dataset_paths:
            validation_data = training.setdefault("validation_data", {})
            if not isinstance(validation_data, dict):
                raise ValueError(
                    "DeepMD lifecycle recipe validation data must be an object"
                )
            validation_data["systems"] = [
                str(path) for path in validation_dataset_paths
            ]
        else:
            training.pop("validation_data", None)
        synchronize_model_random_seeds(rendered, seed)
        return rendered

    def _seed(self, member: int) -> int:
        entropy = f"{self.run_json_path}\0{self.iter_index}\0teach/train\0deepmd\0{member}"
        return int.from_bytes(hashlib.sha256(entropy.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True, slots=True)
class DeepMDTeacherBackendFactory:
    """Select one DeepMD CLI family before binding route-stable inputs."""

    executor: DeepMDTeacherExecutor
    cli: str

    def __post_init__(self) -> None:
        if self.cli not in {"tf", "pt"}:
            raise ValueError("DeepMD teacher role factory requires tf or pt CLI")

    def bind(self, binding: TeacherBackendBinding) -> DeepMDTeacherBackend:
        from workflow_backends.teacher.role import (
            TeacherScreenOperationProfiles,
            screen_profile_id,
        )

        return DeepMDTeacherBackend(
            adapter=DeepMDTeacherAdapter(
                executor=self.executor,
                run_json_path=binding.run_json_path,
                iter_index=binding.iter_index,
                cli=self.cli,
            ),
            binding=binding,
            cli=self.cli,
            screen_operation_profiles=TeacherScreenOperationProfiles(
                gmm_score=screen_profile_id(
                    getattr(self.executor, "gmm_score_context", None)
                ),
                label=screen_profile_id(getattr(self.executor, "label_context", None)),
            ),
        )


@dataclass(slots=True)
class DeepMDTeacherBackend:
    adapter: DeepMDTeacherAdapter
    binding: TeacherBackendBinding
    cli: str
    screen_operation_profiles: object

    @property
    def readout_ops(self) -> TeacherReadoutOps:
        return self.adapter

    @property
    def gmm_reference_context(self):
        return getattr(self.adapter.executor, "gmm_reference_context", None)

    @property
    def gmm_score_context(self):
        return getattr(self.adapter.executor, "gmm_score_context", None)

    def train(self, request: TeacherTrainRequest) -> TeacherTrainResult:
        return self.adapter.train(request)

    def finalize_published_training(
        self, request: TeacherTrainPublicationRequest
    ) -> None:
        self.adapter.finalize_published_training(request)

    def label(
        self, request: TeacherNativeLabelRequest
    ) -> TeacherNativeLabelResult:
        return self.adapter.label(request)

    def validate_published_training(
        self, request: TeacherPublishedTrainingValidationRequest
    ) -> bool:
        if request.member_ids != self.binding.member_ids:
            return False
        expected_suffix = ".pt2" if self.cli == "pt" else ".pb"
        if any(
            handle.path.suffix != expected_suffix
            or handle.path.parent.name != handle.member_id
            for handle in request.handles
        ):
            return False
        if self.cli != "pt":
            return True
        from workflow_backends.teacher.deepmd.native import (
            validate_terminal_digest_metadata,
        )

        if not validate_terminal_digest_metadata(
            request.handles, active_type_map=self.binding.type_map
        ):
            return False
        recipe = self.adapter._load_recipe(self.binding.recipe_path)
        expected_family = _rendered_model_family(recipe)
        metadata = tuple(
            _deepmd_member_metadata(handle.path.parent) for handle in request.handles
        )
        families = tuple(item.get("model_family") for item in metadata)
        if any(family != expected_family for family in families):
            return False
        if len(set(families)) != 1:
            return False
        if expected_family == DPA4_MODEL_FAMILY:
            from workflow_artifacts.lifecycle.deepmd_model_assets import (
                load_deepmd_model_bundles,
            )

            load_deepmd_model_bundles(
                stage_root=request.stage_root,
                expected_member_ids=request.member_ids,
                expected_native_type_maps=tuple(
                    tuple(item["type_map"]) for item in metadata
                ),
            )
        return True

    def recover_historical_model(
        self, request: TeacherHistoricalModelRequest
    ) -> TeacherHistoricalModelResult:
        from workflow_backends.teacher.deepmd.historical_model_recovery import (
            resolve_legacy_deepmd_teacher_model,
        )

        return TeacherHistoricalModelResult(
            request.member_id,
            resolve_legacy_deepmd_teacher_model(
                request.member_id, request.legacy_handle
            ),
        )


def _validated_terminal_model(execution: object) -> Path:
    terminal = Path(getattr(execution, "terminal_model_path", execution))
    if terminal.is_symlink() or not terminal.is_file() or terminal.stat().st_size <= 0:
        raise ValueError(f"DeepMD lifecycle teacher requires a non-empty regular terminal model: {terminal}")
    return terminal.resolve()


def _rendered_model_family(rendered: dict) -> str:
    return deepmd_recipe_model_family(rendered)


def _deepmd_member_metadata(member_root: Path) -> dict:
    path = Path(member_root) / "deepmd_model_metadata.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"DeepMD published training metadata is invalid: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"DeepMD published training metadata must be an object: {path}")
    type_map = payload.get("type_map")
    if (
        not isinstance(type_map, list)
        or not type_map
        or any(not isinstance(item, str) or not item for item in type_map)
        or len(set(type_map)) != len(type_map)
    ):
        raise ValueError(f"DeepMD published training metadata has invalid type_map: {path}")
    return payload
