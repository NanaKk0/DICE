from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any

from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.student.nep.evaluator import evaluate_nep_committee
from workflow_backends.student.nep.model_registry import collect_nep_model_paths
from workflow_backends.student.nep.trainer import load_nep_committee_summary, prepare_nep_committee_training, run_nep_committee_training
from workflow_backends.student.config import load_nep_training_config
from workflow_backends.student.base import (
    CommitteeModelArtifact,
    StudentBackend,
    StudentCommitteeSummary,
    StudentContinuationContract,
    StudentExportContract,
    StudentEvaluationRequest,
    StudentFoundationBootstrapSpec,
    StudentSamplingRequest,
    StudentSamplingSummary,
    StudentTrainRequest,
)
from workflow_backends.student.foundation import validate_foundation_paths, validate_type_map_coverage
from workflow_core.common.uncertainty import UncertaintyRecord
from workflow_runtime.environment import student_backend_role


class NEPStudentBackend(StudentBackend):
    backend_name = "nep"
    aliases = ()

    def _resolve_train_recipe_ref(self, cfg: dict[str, Any]) -> str | None:
        for template in dict(cfg.get("resolved_student_pipeline_templates", {}) or {}).values():
            for stage in list(template.get("stages", []) or []):
                if str(stage.get("stage")) != "train":
                    continue
                stage_cfg = dict(stage.get("stage_config", {}) or {})
                candidate = stage_cfg.get("train_recipe_ref")
                if candidate not in (None, ""):
                    return str(candidate)
        return None

    def _training_cfg_for_stage(self, cfg: dict[str, Any], training_stage: str | None) -> dict[str, Any]:
        config_path = self._resolve_train_recipe_ref(cfg)
        if config_path:
            train_cfg = load_nep_training_config(config_path, training_stage=training_stage or "bootstrap")
        else:
            train_cfg = dict(cfg.get("nep", cfg) or {})
        identity_cfg = dict(cfg.get("student_model", {}) or {})
        for key in ("type_map", "committee_size"):
            if identity_cfg.get(key) is not None:
                train_cfg[key] = identity_cfg[key]
        return train_cfg

    def export_contract(self, cfg: dict[str, Any]) -> StudentExportContract:
        del cfg
        return StudentExportContract(writes_deepmd_dataset=False)

    def continuation_contract(self, cfg: dict[str, Any]) -> StudentContinuationContract:
        del cfg
        return StudentContinuationContract(
            supports_bootstrap_from_summary=True,
            supports_finetune_from_previous_round=True,
            requires_restart_assets=True,
            required_member_assets=("model_path", "restart_path"),
        )

    def resolve_train_ratio(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> float:
        if config_path is not None:
            nep_cfg = load_nep_training_config(config_path, training_stage=training_stage or "bootstrap")
        else:
            nep_cfg = self._training_cfg_for_stage(cfg, training_stage or "bootstrap")
        split = float(nep_cfg.get("train_test_split", 0.95))
        if not 0.0 < split <= 1.0:
            raise ValueError(f"nep.train_test_split must be in (0, 1], got {split!r}")
        return split

    def resolve_element_order(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> list[str]:
        identity_type_map = dict(cfg.get("student_model", {}) or {}).get("type_map")
        if isinstance(identity_type_map, list) and identity_type_map:
            return [str(item) for item in identity_type_map]
        if config_path is not None:
            nep_cfg = load_nep_training_config(config_path, training_stage=training_stage or "bootstrap")
        else:
            nep_cfg = self._training_cfg_for_stage(cfg, training_stage or "bootstrap")
        type_map = nep_cfg.get("type_map")
        if isinstance(type_map, list) and type_map:
            return [str(item) for item in type_map]
        raise KeyError("Unable to resolve NEP type_map from config")

    def load_committee_summary(self, summary_path: str | Path) -> StudentCommitteeSummary:
        resolved_summary_path = Path(summary_path).resolve()
        payload = load_nep_committee_summary(resolved_summary_path)
        model_paths = collect_nep_model_paths(payload)
        restart_paths = [str(Path(item).resolve()) for item in payload.get("restarts", []) or []]
        artifacts: list[CommitteeModelArtifact] = []
        continuation_assets_complete = True
        for index, model_path in enumerate(model_paths):
            restart_path = restart_paths[index] if index < len(restart_paths) else None
            metadata = {"restart_path": restart_path}
            artifact = CommitteeModelArtifact(
                backend="nep",
                member_index=index,
                model_path=str(Path(model_path).resolve()),
                metadata=metadata,
            )
            if not Path(artifact.model_path).resolve().exists():
                continuation_assets_complete = False
            if restart_path is None or not Path(restart_path).resolve().exists():
                continuation_assets_complete = False
            artifacts.append(artifact)
        return StudentCommitteeSummary(
            backend="nep",
            summary_path=str(resolved_summary_path),
            committee_root=str(resolved_summary_path.parent.resolve()),
            member_count=len(model_paths),
            usable_member_count=len(artifacts),
            members=artifacts,
            supports_continuation=len(artifacts) >= 2,
            continuation_assets_complete=continuation_assets_complete and len(artifacts) >= 2,
            metadata={"raw_summary": payload, "restarts": restart_paths},
        )

    def _training_root(self, paths, round_id: int, dp_round_id: int | None) -> Path:
        raise RuntimeError("NEP target-mode callers must pass explicit training work_root")

    def _sampling_root(self, paths, round_id: int, student_round_id: int) -> Path:
        raise RuntimeError("NEP target-mode callers must pass explicit sampling work_root")

    def committee_work_root(self, paths, outer_round: int, student_round: int, *, create: bool = True) -> Path:
        raise RuntimeError("NEP target-mode callers must pass explicit stage work_root")

    def committee_summary_path(self, paths, outer_round: int, student_round: int) -> Path:
        raise RuntimeError("NEP target-mode callers must pass explicit summary_path")

    def sampling_work_root(self, paths, outer_round: int, student_round: int, *, create: bool = True) -> Path:
        raise RuntimeError("NEP target-mode callers must pass explicit sampling work_root")

    def normalize_sampling_summary(
        self,
        paths,
        outer_round: int,
        student_round: int,
        summary_path: str | Path,
        summary_payload: dict[str, Any],
    ) -> StudentSamplingSummary:
        del paths, outer_round, student_round
        payload = dict(summary_payload or {})
        summary_path = Path(summary_path).resolve()
        payload.setdefault("selected_count", int(payload.get("candidate_count", 0) or 0))
        payload.setdefault("effective_sample_count", int(payload.get("total_frames", 0) or 0))
        payload.setdefault("high_unc_count", int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0))
        effective_sample_count = int(payload.get("effective_sample_count", 0) or 0)
        if "high_unc_ratio" not in payload:
            payload["high_unc_ratio"] = float(payload["high_unc_count"] / effective_sample_count) if effective_sample_count else 0.0
        payload.setdefault("failed_uncertainty_count", int(payload.get("failed_count", 0) or 0))
        payload.setdefault("sampling_backend", "gpumd")
        return StudentSamplingSummary(
            backend=self.backend_name,
            sampling_backend=str(payload.get("sampling_backend", "gpumd")),
            summary_path=str(summary_path.resolve()),
            work_root=str(summary_path.parent.resolve()),
            selected_xyz=str(Path(str(payload.get("selected_xyz") or payload.get("candidate_xyz") or "")).resolve()),
            candidate_xyz=str(Path(str(payload.get("candidate_xyz") or "")).resolve()),
            failed_xyz=str(Path(str(payload.get("failed_xyz") or "")).resolve()),
            uncertainty_records_jsonl=(
                str(Path(str(payload.get("uncertainty_records_jsonl"))).resolve())
                if payload.get("uncertainty_records_jsonl")
                else None
            ),
            effective_sample_count=int(payload.get("effective_sample_count", 0) or 0),
            selected_count=int(payload.get("selected_count", 0) or 0),
            candidate_count=int(payload.get("candidate_count", 0) or 0),
            failed_count=int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0),
            high_unc_count=int(payload.get("high_unc_count", 0) or 0),
            high_unc_ratio=float(payload.get("high_unc_ratio", 0.0) or 0.0),
            payload=payload,
        )

    def build_train_request(
        self,
        cfg: dict[str, Any],
        paths,
        outer_round: int,
        student_round: int,
        *,
        work_root: str | Path,
        summary_path: str | Path,
        dataset_roots: list[str],
        dataset_path: str | Path | None,
        previous_model_root: str | Path | None,
        continue_training: bool,
        training_stage: str,
        extra_context: dict[str, Any] | None = None,
    ) -> StudentTrainRequest:
        del dataset_roots
        if dataset_path is None:
            raise ValueError("NEP training requires dataset_path")
        nep_cfg = self._training_cfg_for_stage(cfg, training_stage)
        work_root = Path(work_root).resolve()
        summary_path = Path(summary_path).resolve()
        context = {
            "dataset_path": str(Path(dataset_path).resolve()),
            "work_root": str(work_root),
            "summary_path": str(summary_path),
            "previous_model_root": str(Path(previous_model_root).resolve()) if previous_model_root is not None else None,
            "continue_training_override": bool(continue_training),
            "training_stage": str(training_stage),
            "log_dir": str(work_root.parent.resolve()),
            "log_prefix": f"student_subworkflow_r{outer_round}_student{student_round}_train",
        }
        context.update(dict(extra_context or {}))
        return StudentTrainRequest(
            backend_name=self.backend_name,
            outer_round=int(outer_round),
            student_round=int(student_round),
            work_root=str(work_root),
            summary_path=str(summary_path),
            backend_cfg=nep_cfg,
            dataset_path=str(Path(dataset_path).resolve()),
            previous_model_root=str(Path(previous_model_root).resolve()) if previous_model_root is not None else None,
            continue_training=bool(continue_training),
            training_stage=str(training_stage),
            context=context,
        )

    def train_committee(self, paths, request: StudentTrainRequest) -> str:
        normalized = normalize_backend_request_context(request, role=student_backend_role(self.backend_name, "train"))
        context = normalized.context
        dataset_path = context.get("dataset_path") or request.dataset_path
        if not dataset_path:
            raise ValueError("NEP training requires dataset_path in request")
        work_root = Path(context.get("work_root", request.work_root)).resolve()
        summary_path = Path(context.get("summary_path", request.summary_path)).resolve()
        env = context.get("env")
        nep_cfg = dict(request.backend_cfg or {})
        execution = normalized.execution
        nep_executable = execution.required_executable("nep")
        if nep_executable:
            nep_command = execution.shell_command([nep_executable])
            nep_cfg["command"] = shlex.join(nep_command)
        foundation_bootstrap = dict(context.get("foundation_bootstrap") or {})
        if foundation_bootstrap:
            nep_cfg["fine_tune_model_path"] = str(Path(str(foundation_bootstrap["model_path"])).resolve())
            nep_cfg["fine_tune_restart_path"] = str(Path(str(foundation_bootstrap["restart_path"])).resolve())
            arch_policy = str(foundation_bootstrap.get("arch_policy", "override_from_foundation")).strip().lower()
            foundation_arch = dict(foundation_bootstrap.get("arch", {}) or {})
            if arch_policy == "override_from_foundation" and foundation_arch:
                nep_cfg["arch"] = foundation_arch
            nep_cfg["foundation_bootstrap"] = {
                "model_path": nep_cfg["fine_tune_model_path"],
                "restart_path": nep_cfg["fine_tune_restart_path"],
                "type_map": list(foundation_bootstrap.get("type_map", []) or []),
                "arch_policy": arch_policy,
                "arch": foundation_arch,
            }
        return run_nep_committee_training(
            dataset_path,
            nep_cfg,
            work_root=work_root,
            summary_path=summary_path,
            random_seed=int(context.get("random_seed", 42)),
            env=env,
            previous_model_root=context.get("previous_model_root"),
            continue_training=bool(context.get("continue_training_override", False)),
            training_stage=str(context.get("training_stage", "bootstrap")),
            resolved_train_decision=dict(context.get("resolved_train_decision", {}) or {}),
            log_dir=context.get("backend_log_dir", context.get("log_dir")),
            log_prefix=str(context.get("log_prefix", "nep_train")),
        )

    def reset_interrupted_training(self, request: StudentTrainRequest) -> None:
        summary_path = Path(request.summary_path).resolve()
        if summary_path.exists():
            summary_path.unlink()

    def validate_foundation_bootstrap(
        self,
        cfg: dict[str, Any],
        spec: StudentFoundationBootstrapSpec,
        *,
        element_order: list[str],
    ) -> None:
        validate_foundation_paths(spec.model_path, spec.restart_path)
        if not spec.type_map:
            raise ValueError("NEP foundation asset requires student_foundation_asset.nep.type_map")
        validate_type_map_coverage(spec.type_map, element_order, label="NEP")
        arch = dict(spec.backend_options.get("arch", {}) or {})
        arch_policy = str(spec.backend_options.get("arch_policy", "override_from_foundation")).strip().lower()
        if arch_policy == "require_exact" and arch:
            current = self._training_cfg_for_stage(cfg, "bootstrap")
            expected = self._flatten_foundation_arch(arch)
            mismatches = []
            for key, expected_value in expected.items():
                current_value = current.get(key)
                if current_value != expected_value:
                    mismatches.append(f"{key}: current={current_value!r}, foundation={expected_value!r}")
            if mismatches:
                raise ValueError("NEP foundation architecture mismatch: " + "; ".join(mismatches))

    @staticmethod
    def _flatten_foundation_arch(arch: dict[str, Any]) -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        for key in ("version", "zbl", "cutoff", "n_max", "basis_size", "l_max", "neuron"):
            if key in arch:
                value = arch[key]
                if key == "l_max" and isinstance(value, list):
                    value = " ".join(str(int(item)) for item in value)
                if key == "zbl" and isinstance(value, dict):
                    value = None if not bool(value.get("enabled", True)) else float(value.get("outer_cutoff", value.get("cutoff", 0.0)))
                flattened[key] = value
        return flattened

    def prepare_foundation_bootstrap_context(
        self,
        cfg: dict[str, Any],
        spec: StudentFoundationBootstrapSpec,
    ) -> dict[str, Any]:
        payload = super().prepare_foundation_bootstrap_context(cfg, spec)
        payload["arch_policy"] = str(spec.backend_options.get("arch_policy", "override_from_foundation"))
        payload["arch"] = dict(spec.backend_options.get("arch", {}) or {})
        return payload

    def sample_with_committee(
        self,
        paths,
        request: StudentSamplingRequest,
    ) -> str:
        from workflow_backends.sampling.md.gpumd import GPUMDSamplingBackend

        return GPUMDSamplingBackend().sample_with_committee(paths, request)

    def build_evaluation_request(
        self,
        cfg: dict[str, Any],
        *,
        dataset_path: str | Path,
        committee_summary_path: str | Path,
        work_root: str | Path | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> StudentEvaluationRequest:
        context = {
            "committee_summary_path": str(Path(committee_summary_path).resolve()),
        }
        if work_root is not None:
            context["work_root"] = str(Path(work_root).resolve())
        context.update(dict(extra_context or {}))
        execution_context = context.get("execution_context")
        return StudentEvaluationRequest(
            backend_name=self.backend_name,
            dataset_path=str(Path(dataset_path).resolve()),
            committee_summary_path=str(Path(committee_summary_path).resolve()),
            backend_cfg=dict(cfg.get("nep", cfg) or {}),
            work_root=str(Path(work_root).resolve()) if work_root is not None else None,
            execution_context=execution_context if isinstance(execution_context, dict) else None,
            context=context,
        )

    def evaluate_committee(self, paths, request: StudentEvaluationRequest) -> dict[str, Any]:
        del paths
        context = dict(request.context or {})
        summary_path = context.get("committee_summary_path") or request.committee_summary_path
        if not summary_path:
            raise ValueError("NEP evaluation requires committee_summary_path in request")
        nep_cfg = dict(request.backend_cfg or {})
        return evaluate_nep_committee(
            request.dataset_path,
            summary_path,
            nep_cfg=nep_cfg,
            work_root=context.get("work_root"),
            env=context.get("env"),
            execution_context=request.execution_context,
        )

    def parse_uncertainty_outputs(self, task_dir: str, cfg: dict[str, Any]) -> list[UncertaintyRecord]:
        from workflow_backends.sampling.md.gpumd import parse_active_out

        uncertainty_cfg = dict(cfg.get("uncertainty", {}) or {})
        gpumd_cfg = dict(cfg.get("gpumd", {}) or {})
        task_root = Path(task_dir).resolve()
        task_meta = {}
        if (task_root / "task_meta.json").exists():
            from workflow_core.common.io import read_json

            task_meta = read_json(task_root / "task_meta.json")
        return parse_active_out(
            task_root / "active.out",
            task_id=str(task_meta.get("task_id") or task_root.name),
            temperature=float(task_meta["temperature"]) if "temperature" in task_meta else None,
            trust_lo=float(uncertainty_cfg.get("trust_lo", gpumd_cfg.get("active_threshold", 0.10))),
            trust_hi=float(uncertainty_cfg.get("trust_hi", 0.40)),
        )

    def collect_models(self, model_root: str) -> list[CommitteeModelArtifact]:
        payload = load_nep_committee_summary(model_root)
        artifacts: list[CommitteeModelArtifact] = []
        for index, model_path in enumerate(collect_nep_model_paths(payload)):
            artifacts.append(
                CommitteeModelArtifact(
                    backend="nep",
                    member_index=index,
                    model_path=model_path,
                    metadata={
                        "restart_path": (
                            payload.get("restarts", [None] * len(collect_nep_model_paths(payload)))[index]
                            if index < len(payload.get("restarts", []) or [])
                            else None
                        )
                    },
                )
            )
        return artifacts

    def prepare_training(self, dataset_path: str | Path, cfg: dict[str, Any], *, work_root: str | Path, random_seed: int = 42) -> dict[str, Any]:
        return prepare_nep_committee_training(dataset_path, cfg, work_root=work_root, random_seed=random_seed)

    def prepare_sampling(
        self,
        *,
        seed_xyz_path: str | Path,
        model_paths: list[str],
        gpumd_cfg: dict[str, Any],
        uncertainty_cfg: dict[str, Any],
        work_root: str | Path,
        outer_round: int,
        student_round: int,
    ) -> dict[str, Any]:
        from workflow_backends.sampling.md.gpumd import prepare_gpumd_sampling_tasks

        return prepare_gpumd_sampling_tasks(
            seed_xyz_path=seed_xyz_path,
            model_paths=model_paths,
            gpumd_cfg=gpumd_cfg,
            uncertainty_cfg=uncertainty_cfg,
            work_root=work_root,
            outer_round=outer_round,
            student_round=student_round,
        )
