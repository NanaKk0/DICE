from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.lammps.model_devi_driver import run_lammps_model_devi_sampling
from workflow_core.common.io import read_json
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures
from workflow_runtime.environment import sampling_backend_role


def run_lammps_model_devi_sampling(*args: Any, **kwargs: Any):
    from workflow_backends.lammps.model_devi_driver import run_lammps_model_devi_sampling as _impl

    return _impl(*args, **kwargs)


def __getattr__(name: str):
    if name == "LammpsTaskSampler":
        from workflow_backends.lammps.task_sampler import LammpsTaskSampler

        return LammpsTaskSampler
    raise AttributeError(name)


class LAMMPSSamplingBackend:
    backend_name = "lammps"

    def sample_with_committee(self, paths, request) -> str:
        if str(request.student_backend or "").strip().lower() != "deepmd":
            raise ValueError(
                f"sampling_backend='lammps' requires student_backend='deepmd', got {request.student_backend!r}"
            )
        backend_cfg = dict(request.backend_cfg or {})
        if str(backend_cfg.get("backend", "lammps")).strip().lower() != "lammps":
            raise ValueError(
                f"sampling_backend='lammps' received incompatible request backend={backend_cfg.get('backend')!r}"
            )
        normalized = normalize_backend_request_context(request, role=sampling_backend_role(self.backend_name, "sample"))
        context = normalized.context
        execution = normalized.execution
        lmp_executable = execution.required_executable("lmp")
        if lmp_executable:
            lammps_cfg = dict(backend_cfg.get("lammps", {}) or {})
            lammps_cfg["command"] = execution.shell_command([lmp_executable])
            backend_cfg["lammps"] = lammps_cfg
        summary_extra = {
            "teacher_round": int(request.teacher_round),
            "student_round": int(request.student_round),
            **dict(request.summary_extra or {}),
        }
        initial_structure_paths = self._materialize_initial_structure_paths(request.seed_xyz_path, request.work_root)
        context.update(
            {
                "work_root_override": str(Path(request.work_root).resolve()),
                "seed_pool_override": str(Path(request.seed_xyz_path).resolve()),
                "reference_xyz_files_override": list(request.reference_xyz_files or []),
                "candidate_train_selection": dict(backend_cfg.get("candidate_train_selection", {}) or {}),
                "summary_extra": summary_extra,
            }
        )
        committee_pool = self._committee_pool_from_paths(request.committee_model_paths)
        summary = {
            "backend": request.student_backend,
            "committee_size": len(committee_pool),
            "usable_committee_size": len(committee_pool),
            "members": committee_pool,
        }
        return run_lammps_model_devi_sampling(
            initial_structure_paths=initial_structure_paths,
            committee_pool=committee_pool,
            cfg=backend_cfg,
            work_root=str(Path(request.work_root).resolve()),
            summary=summary,
            context=context,
        )

    def normalize_sampling_summary(self, paths, request, summary_path: str | Path, summary_payload: dict[str, Any]):
        del paths, request
        from workflow_backends.sampling.summary import normalize_lammps_sampling_summary

        return normalize_lammps_sampling_summary(
            summary_path=summary_path,
            summary_payload=dict(summary_payload or {}),
        )

    def load_committee_pool(self, summary_path: str | Path):
        payload = read_json(summary_path)
        committee_pool = [
            {
                "model_index": index,
                "graph_pb": str(Path(member["graph_pb"]).resolve()),
                "status": str(member.get("status", "success")),
            }
            for index, member in enumerate(list(payload.get("members", []) or []))
            if member.get("graph_pb")
        ]
        if len(committee_pool) < 2:
            raise RuntimeError(
                f"Usable DP committee size is {len(committee_pool)}; at least 2 successful graph.pb files are required"
            )
        return payload, committee_pool

    def read_summary(self, summary_path: str | Path) -> dict[str, Any]:
        return read_json(summary_path)

    @staticmethod
    def _committee_pool_from_paths(model_paths: list[str]) -> list[dict[str, Any]]:
        committee_pool = [
            {"model_index": index, "graph_pb": str(Path(model_path).resolve())}
            for index, model_path in enumerate(list(model_paths or []))
            if str(model_path).strip()
        ]
        if len(committee_pool) < 2:
            raise RuntimeError(
                f"Usable DP committee size is {len(committee_pool)}; at least 2 successful graph.pb files are required"
            )
        return committee_pool

    @staticmethod
    def _materialize_initial_structure_paths(seed_xyz_path: str | Path, work_root: str | Path) -> list[str]:
        work_root = Path(work_root).resolve()
        initial_dir = (work_root / "initial_structures").resolve()
        initial_dir.mkdir(parents=True, exist_ok=True)
        frames = read_extxyz_structures(seed_xyz_path, allow_empty=False)
        output_paths: list[str] = []
        for index, atoms in enumerate(frames):
            path = (initial_dir / f"seed_{index:06d}.xyz").resolve()
            write_extxyz_structures(path, [atoms], allow_empty=False)
            output_paths.append(str(path))
        if not output_paths:
            raise RuntimeError("No initial structures prepared for committee sampling")
        return output_paths


__all__ = ["LAMMPSSamplingBackend", "LammpsTaskSampler", "run_lammps_model_devi_sampling"]
