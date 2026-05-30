from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from workflow_backends.committee_evaluation import canonical_committee_evaluation_result
from workflow_backends.student.deepmd.committee_config import load_committee_train_config, load_dp_train_template
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
from workflow_core.common.io import read_json, write_candidate_xyz, write_json
from workflow_core.common.uncertainty import UncertaintyRecord, route_uncertainty
from workflow_runtime.environment import student_backend_role


def _resolve_train_committee_config_path(cfg: dict[str, Any]) -> str:
    for template in dict(cfg.get("resolved_student_pipeline_templates", {}) or {}).values():
        for stage in list(template.get("stages", []) or []):
            if str(stage.get("stage")) != "train":
                continue
            stage_cfg = dict(stage.get("stage_config", {}) or {})
            candidate = stage_cfg.get("train_recipe_ref")
            if candidate not in (None, ""):
                return str(candidate)
    raise KeyError("DeepMD train request requires train_recipe_ref from train stage config")


def _evaluation_contract_metadata(context: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    mode = str(context.get("evaluation_mode", "committee_mean")).strip().lower()
    if mode in {"model_devi", "force_deviation", "force_deviation_only"}:
        return "force_deviation", {
            "force_deviation": True,
            "committee_mean_labels": False,
            "mean_energy": False,
            "mean_forces": False,
        }
    return "committee_mean", {
        "force_deviation": True,
        "committee_mean_labels": True,
        "mean_energy": True,
        "mean_forces": True,
    }


class DeepMDStudentBackend(StudentBackend):
    backend_name = "deepmd"
    aliases = ("dp",)

    def export_contract(self, cfg: dict[str, Any]) -> StudentExportContract:
        del cfg
        return StudentExportContract(writes_deepmd_dataset=True)

    def continuation_contract(self, cfg: dict[str, Any]) -> StudentContinuationContract:
        del cfg
        return StudentContinuationContract(
            supports_bootstrap_from_summary=True,
            supports_finetune_from_previous_round=True,
            requires_restart_assets=False,
            required_member_assets=("model_path",),
        )

    def resolve_train_ratio(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> float:
        del training_stage
        del config_path
        valid_fraction = float(cfg.get("valid_fraction", 0.1))
        if not 0.0 < valid_fraction < 1.0:
            raise ValueError(f"deepmd valid_fraction must be in (0, 1), got {valid_fraction!r}")
        return 1.0 - valid_fraction

    def resolve_element_order(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> list[str]:
        student_model = dict(cfg.get("student_model", {}) or {})
        identity_type_map = student_model.get("type_map")
        if isinstance(identity_type_map, list) and identity_type_map:
            return [str(item) for item in identity_type_map]
        if config_path is None:
            raise ValueError("DeepMD element order resolution requires config_path")
        train_cfg = load_committee_train_config(config_path)
        selected_train_config = train_cfg.select_train_config(training_stage or "bootstrap")
        template = load_dp_train_template(selected_train_config)
        type_map = template.get("model", {}).get("type_map")
        if isinstance(type_map, list) and type_map:
            return [str(item) for item in type_map]
        raise KeyError(f"Unable to resolve DeepMD type_map from {selected_train_config}")

    def load_committee_summary(self, summary_path: str | Path) -> StudentCommitteeSummary:
        summary_path = Path(summary_path).resolve()
        payload = read_json(summary_path) if summary_path.is_file() else read_json(summary_path / "summary.json")
        members = self.collect_models(str(summary_path))
        member_count = len(payload.get("members", []) or [])
        return StudentCommitteeSummary(
            backend="deepmd",
            summary_path=str(summary_path if summary_path.is_file() else (summary_path / "summary.json").resolve()),
            committee_root=str((summary_path.parent if summary_path.is_file() else summary_path).resolve()),
            member_count=int(member_count),
            usable_member_count=len(members),
            members=members,
            supports_continuation=len(members) >= 2,
            continuation_assets_complete=len(members) >= 2,
            metadata={"raw_summary": payload},
        )

    def committee_work_root(self, paths, outer_round: int, student_round: int, *, create: bool = True) -> Path:
        raise RuntimeError("DeepMD target-mode callers must pass explicit stage work_root")

    def committee_summary_path(self, paths, outer_round: int, student_round: int) -> Path:
        return (self.committee_work_root(paths, outer_round, student_round, create=False) / "summary.json").resolve()

    def sampling_work_root(self, paths, outer_round: int, student_round: int, *, create: bool = True) -> Path:
        raise RuntimeError("DeepMD target-mode callers must pass explicit sampling work_root")

    def normalize_sampling_summary(
        self,
        paths,
        outer_round: int,
        student_round: int,
        summary_path: str | Path,
        summary_payload: dict[str, Any],
    ) -> StudentSamplingSummary:
        payload = dict(summary_payload or {})
        summary_path = Path(summary_path).resolve()
        work_dir = summary_path.parent
        payload.setdefault("sampling_summary_path", str(summary_path.resolve()))
        payload.setdefault("selected_count", int(payload.get("candidate_count", 0) or 0))
        payload.setdefault("effective_sample_count", int(payload.get("total_frames", 0) or 0))
        payload.setdefault("high_unc_count", int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0))
        effective_sample_count = int(payload.get("effective_sample_count", 0) or 0)
        if "high_unc_ratio" not in payload:
            payload["high_unc_ratio"] = float(payload["high_unc_count"] / effective_sample_count) if effective_sample_count else 0.0
        payload.setdefault("failed_uncertainty_count", int(payload.get("failed_count", 0) or 0))
        payload.setdefault("sampling_backend", "lammps")

        selected_source = Path(str(payload.get("selected_xyz") or payload.get("candidate_xyz") or "")).resolve()
        failed_source = Path(str(payload.get("failed_xyz") or "")).resolve() if payload.get("failed_xyz") else None
        uncertainty_source = Path(str(payload.get("uncertainty_records_jsonl") or "")).resolve() if payload.get("uncertainty_records_jsonl") else None
        candidate_by_system_source = Path(str(payload.get("candidate_by_system_dir") or "")).resolve() if payload.get("candidate_by_system_dir") else None
        failed_by_system_source = Path(str(payload.get("failed_by_system_dir") or "")).resolve() if payload.get("failed_by_system_dir") else None

        candidate_xyz = selected_source if selected_source.exists() else work_dir / "candidate.xyz"
        failed_xyz = failed_source if failed_source is not None and failed_source.exists() else work_dir / "failed.xyz"
        uncertainty_jsonl = uncertainty_source if uncertainty_source is not None and uncertainty_source.exists() else work_dir / "uncertainty_records.jsonl"
        if not candidate_xyz.exists():
            write_candidate_xyz(candidate_xyz, [])
        if not failed_xyz.exists():
            write_candidate_xyz(failed_xyz, [])
        if candidate_by_system_source is not None and candidate_by_system_source.exists():
            target = work_dir / "candidate_by_system"
            if candidate_by_system_source != target:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(candidate_by_system_source, target)
        if failed_by_system_source is not None and failed_by_system_source.exists():
            target = work_dir / "failed_by_system"
            if failed_by_system_source != target:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(failed_by_system_source, target)
        payload["selected_xyz"] = str(candidate_xyz.resolve())
        payload["candidate_xyz"] = str(candidate_xyz.resolve())
        payload["failed_xyz"] = str(failed_xyz.resolve())
        payload["sampling_summary_path"] = str(summary_path.resolve())
        return StudentSamplingSummary(
            backend=self.backend_name,
            sampling_backend="lammps",
            summary_path=str(summary_path.resolve()),
            work_root=str(work_dir.resolve()),
            selected_xyz=str(candidate_xyz.resolve()),
            candidate_xyz=str(candidate_xyz.resolve()),
            failed_xyz=str(failed_xyz.resolve()),
            uncertainty_records_jsonl=str(uncertainty_jsonl.resolve()) if uncertainty_jsonl.exists() else None,
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
        del paths, dataset_path
        work_root = Path(work_root).resolve()
        summary_path = Path(summary_path).resolve()
        context = dict(extra_context or {})
        if isinstance(cfg.get("deepmd"), dict) and cfg.get("deepmd"):
            context.setdefault("student_identity", dict(cfg.get("deepmd", {}) or {}))
        train_committee_config = _resolve_train_committee_config_path(cfg)
        return StudentTrainRequest(
            backend_name=self.backend_name,
            outer_round=int(outer_round),
            student_round=int(student_round),
            work_root=str(work_root),
            summary_path=str(summary_path),
            backend_cfg=train_committee_config,
            dataset_roots=list(dataset_roots),
            previous_model_root=str(Path(previous_model_root).resolve()) if previous_model_root is not None else None,
            continue_training=bool(continue_training),
            training_stage=str(training_stage),
            context=context,
        )

    def train_committee(self, paths, request: StudentTrainRequest) -> str:
        from workflow_backends.student.deepmd.training import run_deepmd_train_committee

        return run_deepmd_train_committee(paths, request)

    def reset_interrupted_training(self, request: StudentTrainRequest) -> None:
        work_root = Path(request.work_root).resolve()
        summary_path = Path(request.summary_path).resolve()
        if summary_path.exists():
            summary_path.unlink()
        if not work_root.exists():
            return
        for member_dir in sorted(work_root.glob("model_*")):
            if not member_dir.is_dir():
                continue
            result_file = member_dir / "train_result.json"
            lcurve_out = member_dir / "lcurve.out"
            keep_completed = False
            if result_file.exists() and lcurve_out.exists():
                try:
                    payload = read_json(result_file)
                    model_path = Path(str(payload.get("graph_pb") or member_dir / "graph.pb"))
                    keep_completed = payload.get("status") == "success" and model_path.exists()
                except Exception:
                    keep_completed = False
            if not keep_completed:
                shutil.rmtree(member_dir)

    def validate_foundation_bootstrap(
        self,
        cfg: dict[str, Any],
        spec: StudentFoundationBootstrapSpec,
        *,
        element_order: list[str],
    ) -> None:
        del cfg
        validate_foundation_paths(spec.model_path)
        if not spec.type_map:
            raise ValueError("DeepMD foundation asset requires student_foundation_asset.deepmd.type_map")
        validate_type_map_coverage(spec.type_map, element_order, label="DeepMD")
        if bool(spec.backend_options.get("multitask", False)) and not spec.backend_options.get("model_branch"):
            raise ValueError("DeepMD multitask foundation fine-tune requires model_branch")

    def prepare_foundation_bootstrap_context(
        self,
        cfg: dict[str, Any],
        spec: StudentFoundationBootstrapSpec,
    ) -> dict[str, Any]:
        payload = super().prepare_foundation_bootstrap_context(cfg, spec)
        payload["use_pretrain_script"] = bool(spec.backend_options.get("use_pretrain_script", False))
        payload["model_branch"] = spec.backend_options.get("model_branch")
        payload["multitask"] = bool(spec.backend_options.get("multitask", False))
        return payload

    def sample_with_committee(
        self,
        paths,
        request: StudentSamplingRequest,
    ) -> str:
        from workflow_backends.student.deepmd.sampling import run_deepmd_sample_committee

        return run_deepmd_sample_committee(paths, request)

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
            "committee_summary_override": str(Path(committee_summary_path).resolve()),
        }
        if work_root is not None:
            context["work_root"] = str(Path(work_root).resolve())
        context.update(dict(extra_context or {}))
        execution_context = context.get("execution_context")
        return StudentEvaluationRequest(
            backend_name=self.backend_name,
            dataset_path=str(Path(dataset_path).resolve()),
            committee_summary_path=str(Path(committee_summary_path).resolve()),
            backend_cfg=_resolve_train_committee_config_path(cfg),
            work_root=str(Path(work_root).resolve()) if work_root is not None else None,
            execution_context=execution_context if isinstance(execution_context, dict) else None,
            context=context,
        )

    def evaluate_committee(self, paths, request: StudentEvaluationRequest) -> dict[str, Any]:
        del paths
        from workflow_backends.execution_context import resolve_backend_execution
        from workflow_backends.student.deepmd.sampling import load_deepmd_committee_pool

        context = dict(request.context or {})
        summary_path = context.get("committee_summary_override") or request.backend_cfg
        _, committee_pool = load_deepmd_committee_pool(summary_path)
        committee_graphs = [str(item["graph_pb"]) for item in committee_pool]
        execution = resolve_backend_execution(request, role=student_backend_role(self.backend_name, "evaluate"))
        evaluation_mode, capabilities = _evaluation_contract_metadata(context)
        if execution.configured:
            from workflow_backends.student.deepmd.evaluation_worker import run_deepmd_evaluation_worker

            worker_result = run_deepmd_evaluation_worker(
                frames_path=request.dataset_path,
                committee_graphs=committee_graphs,
                execution=execution,
                work_root=request.work_root or context.get("work_root"),
            )
            worker_extra = dict(worker_result)
            frame_metrics = list(worker_extra.pop("frame_metrics", []) or [])
            return canonical_committee_evaluation_result(
                backend="deepmd",
                committee_size=len(committee_graphs),
                evaluation_mode=evaluation_mode,
                capabilities=capabilities,
                frame_metrics=frame_metrics,
                **worker_extra,
            )

        from workflow_backends.student.deepmd.committee_infer import evaluate_frames_with_committee
        from workflow_backends.student.deepmd.uncertainty import compute_frame_uncertainty

        infer_result = evaluate_frames_with_committee(request.dataset_path, committee_graphs)
        npz_path = Path(infer_result["npz_path"]).resolve()
        try:
            import numpy as np

            payload = np.load(npz_path)
            frame_metrics = [
                compute_frame_uncertainty(payload["forces"][frame_index])
                for frame_index in range(payload["forces"].shape[0])
            ]
        except Exception:
            frame_metrics = []
        infer_extra = dict(infer_result)
        infer_extra.pop("frame_metrics", None)
        return canonical_committee_evaluation_result(
            backend="deepmd",
            committee_size=len(committee_graphs),
            evaluation_mode=evaluation_mode,
            capabilities=capabilities,
            frame_metrics=frame_metrics,
            **infer_extra,
        )

    def parse_uncertainty_outputs(self, task_dir: str, cfg: dict[str, Any]) -> list[UncertaintyRecord]:
        from workflow_backends.lammps.model_devi_parser import parse_model_devi_out

        task_root = Path(task_dir).resolve()
        task_meta = read_json(task_root / "task_meta.json") if (task_root / "task_meta.json").exists() else {}
        job_payload = read_json(task_root / "job.json") if (task_root / "job.json").exists() else {}
        lammps_cfg = dict(cfg.get("lammps", {}) or {})
        uncertainty_cfg = dict(cfg.get("uncertainty", {}) or {})
        trust_lo = float(lammps_cfg.get("f_trust_lo", uncertainty_cfg.get("level_f_lo", 0.10)))
        trust_hi = float(lammps_cfg.get("f_trust_hi", uncertainty_cfg.get("level_f_hi", 0.40)))
        rows = parse_model_devi_out(
            task_root / "model_devi.out",
            natoms=None,
            atomic_enabled=bool(lammps_cfg.get("atomic_model_devi", False)),
        )
        records: list[UncertaintyRecord] = []
        for index, row in enumerate(rows):
            force_std_max = float(row["max_devi_f"])
            route = route_uncertainty(force_std_max, trust_lo=trust_lo, trust_hi=trust_hi)
            records.append(
                UncertaintyRecord(
                    backend="deepmd",
                    sampling_backend="lammps",
                    source="model_devi.out",
                    task_id=str(task_meta.get("task_id") or job_payload.get("job_id") or task_root.name),
                    frame_id=f"{task_root.name}:{index}",
                    step=int(row["step"]),
                    temperature=float(job_payload["temperature"]) if "temperature" in job_payload else None,
                    force_std_max=force_std_max,
                    route=route,
                    structure_path=None,
                    metadata={
                        "avg_devi_f": float(row.get("avg_devi_f", 0.0)),
                        "driver_model_index": job_payload.get("driver_model_index"),
                    },
                )
            )
        return records

    def collect_models(self, model_root: str) -> list[CommitteeModelArtifact]:
        summary_path = Path(model_root).resolve()
        payload = read_json(summary_path) if summary_path.is_file() else read_json(summary_path / "summary.json")
        artifacts: list[CommitteeModelArtifact] = []
        for member in payload.get("members", []):
            if member.get("status") != "success":
                continue
            model_path = Path(str(member.get("model_path") or member.get("graph_pb"))).resolve()
            artifacts.append(
                CommitteeModelArtifact(
                    backend="deepmd",
                    member_index=int(member["model_index"]),
                    model_path=str(model_path),
                    metadata={"lcurve_out": member.get("lcurve_out")},
                )
            )
        return artifacts
