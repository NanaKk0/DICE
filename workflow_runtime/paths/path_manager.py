from pathlib import Path


class RuntimeRoots:

    def __init__(self, artifacts: Path, runtime: Path, logs: Path) -> None:
        self.artifacts = Path(artifacts).resolve()
        self.runtime = Path(runtime).resolve()
        self.logs = Path(logs).resolve()


class DomainPaths:

    def __init__(self, paths: "PathManager") -> None:
        self._paths = paths

    def teacher_round_artifacts_dir(self, teacher_round):
        return self._paths.teacher_round_artifacts_dir(teacher_round)

    def committee_cycle_artifacts_dir(self, teacher_round):
        return self._paths.committee_cycle_artifacts_dir(teacher_round)

    def student_subworkflow_artifacts_dir(self, teacher_round):
        return self._paths.student_subworkflow_artifacts_dir(teacher_round)

    def student_round_artifacts_dir(self, teacher_round, student_round):
        return self._paths.student_round_artifacts_dir(teacher_round, student_round)

    def student_stage_artifacts_dir(self, teacher_round, student_round, stage_run_index, stage_type):
        return self._paths.student_stage_artifacts_dir(teacher_round, student_round, stage_run_index, stage_type)

    def student_stage_backend_logs_dir(self, teacher_round, student_round, stage_run_index, stage_type):
        return self._paths.student_stage_backend_logs_dir(teacher_round, student_round, stage_run_index, stage_type)

    def first_principles_artifacts_dir(self, teacher_round):
        return self._paths.first_principles_artifacts_dir(teacher_round)

    def first_principles_substep_artifacts_dir(self, teacher_round, substep):
        return self._paths.first_principles_substep_artifacts_dir(teacher_round, substep)

    def first_principles_input_xyz(self, teacher_round):
        return self._paths.first_principles_input_xyz(teacher_round)

    def first_principles_tasks_dir(self, teacher_round):
        return self._paths.first_principles_tasks_dir(teacher_round)

    def first_principles_run_dir(self, teacher_round):
        return self._paths.first_principles_run_dir(teacher_round)

    def first_principles_collect_dir(self, teacher_round):
        return self._paths.first_principles_collect_dir(teacher_round)

    def first_principles_labeled_xyz(self, teacher_round):
        return self._paths.first_principles_labeled_xyz(teacher_round)

    def first_principles_failed_xyz(self, teacher_round):
        return self._paths.first_principles_failed_xyz(teacher_round)


class PathManager:

    def __init__(self, workspace_root=".", runtime_root=None):

        self.workspace_root = Path(workspace_root).resolve()
        if runtime_root is None:
            runtime_root = self.workspace_root / "runs" / "default"
        self.runtime_root = Path(runtime_root).resolve()

        # Run root plus the target-mode canonical run roots.
        self.root = self.runtime_root
        self.artifacts = self.runtime_root / "artifacts"
        self.runtime = self.runtime_root / "runtime"
        self.logs = self.runtime_root / "logs"
        self.runtime_roots = RuntimeRoots(self.artifacts, self.runtime, self.logs)
        self.domain_paths = DomainPaths(self)
        self.user_assets = self.workspace_root / "user_assets"
        self.mydataset = self.user_assets
        self.configs = self.workspace_root / "configs"
        self.runs = self.workspace_root / "runs"

        self.user_mace_train = self.user_assets / "mace_train"
        self.user_mace_finetune = self.user_assets / "mace_finetune"
        self.user_teacher_finetune = self.user_assets / "teacher_finetune"
        self.user_student_initial = self.user_assets / "student_initial"
        self.user_dp_train = self.user_assets / "dp_train"
        self.user_seed_pool = self.user_assets / "seed_pool"
        self.user_foundation_models = self.user_assets / "foundation_models"
        self.user_dft = self.user_assets / "dft"

    @staticmethod
    def teacher_round_key(teacher_round):

        return f"teacher_round_{int(teacher_round):03d}"

    def teacher_round_artifacts_dir(self, teacher_round):

        return self.artifacts / self.teacher_round_key(teacher_round)

    def teacher_round_runtime_dir(self, teacher_round):

        return self.runtime / self.teacher_round_key(teacher_round)

    def teacher_round_logs_dir(self, teacher_round):

        return self.logs / self.teacher_round_key(teacher_round)

    def workflow_round_marker_dir(self, outer_round):

        return self.teacher_round_runtime_dir(outer_round) / "markers"

    def workflow_round_preflight_json(self, outer_round):

        return self.teacher_round_runtime_dir(outer_round) / "preflight" / "preflight_report.json"

    def workflow_round_snapshot_dir(self, outer_round):

        return self.teacher_round_runtime_dir(outer_round) / "snapshot"

    @staticmethod
    def student_round_key(student_round):

        return f"student_round_{int(student_round):03d}"

    @staticmethod
    def stage_key(stage_run_index, stage_type):

        return f"stage_{int(stage_run_index):02d}_{str(stage_type).strip()}"

    def semantic_artifacts_dir(self, teacher_round, *parts):

        path = self.teacher_round_artifacts_dir(teacher_round)
        for part in parts:
            path = path / str(part)
        return path

    def semantic_logs_dir(self, teacher_round, *parts):

        path = self.teacher_round_logs_dir(teacher_round)
        for part in parts:
            path = path / str(part)
        return path

    def semantic_stdout_log(self, teacher_round, *parts):

        return self.semantic_logs_dir(teacher_round, *parts) / "stdout.log"

    def semantic_stderr_log(self, teacher_round, *parts):

        return self.semantic_logs_dir(teacher_round, *parts) / "stderr.log"

    def teacher_foundation_train_artifacts_dir(self, teacher_round):

        return self.semantic_artifacts_dir(teacher_round, "teacher_foundation_train")

    def committee_cycle_artifacts_dir(self, teacher_round):

        return self.semantic_artifacts_dir(teacher_round, "committee_cycle")

    def teacher_committee_finetune_artifacts_dir(self, teacher_round):

        return self.semantic_artifacts_dir(teacher_round, "committee_cycle", "teacher_committee_finetune")

    def teacher_committee_finetune_backend_logs_dir(self, teacher_round):

        return self.semantic_logs_dir(teacher_round, "committee_cycle", "teacher_committee_finetune") / "backend_logs"

    def student_subworkflow_artifacts_dir(self, teacher_round):

        return self.semantic_artifacts_dir(teacher_round, "committee_cycle", "student_subworkflow")

    def student_subworkflow_source_dir(self, teacher_round):

        return self.student_subworkflow_artifacts_dir(teacher_round) / "source"

    def student_subworkflow_initial_seed_pool_xyz(self, teacher_round):

        return self.student_subworkflow_source_dir(teacher_round) / "initial_seed_pool.xyz"

    def student_round_artifacts_dir(self, teacher_round, student_round):

        return self.student_subworkflow_artifacts_dir(teacher_round) / self.student_round_key(student_round)

    def student_stage_artifacts_dir(self, teacher_round, student_round, stage_run_index, stage_type):

        return self.student_round_artifacts_dir(teacher_round, student_round) / self.stage_key(stage_run_index, stage_type)

    def student_stage_backend_logs_dir(self, teacher_round, student_round, stage_run_index, stage_type):

        return (
            self.semantic_logs_dir(
                teacher_round,
                "committee_cycle",
                "student_subworkflow",
                self.student_round_key(student_round),
                self.stage_key(stage_run_index, stage_type),
            )
            / "backend_logs"
        )

    def student_stage_marker_path(self, teacher_round, student_round, stage_run_index, stage_type, suffix):

        return (
            self.teacher_round_runtime_dir(teacher_round)
            / "markers"
            / "committee_cycle"
            / "student_subworkflow"
            / self.student_round_key(student_round)
            / f"{self.stage_key(stage_run_index, stage_type)}.{suffix}"
        )

    def first_principles_artifacts_dir(self, teacher_round):

        return self.semantic_artifacts_dir(teacher_round, "first_principles")

    def first_principles_substep_artifacts_dir(self, teacher_round, substep):

        return self.first_principles_artifacts_dir(teacher_round) / str(substep)

    def first_principles_input_xyz(self, teacher_round):

        return self.first_principles_substep_artifacts_dir(teacher_round, "prepare") / "source" / "input_for_first_principles.xyz"

    def first_principles_tasks_dir(self, teacher_round):

        return self.first_principles_substep_artifacts_dir(teacher_round, "prepare") / "tasks"

    def first_principles_run_dir(self, teacher_round):

        return self.first_principles_substep_artifacts_dir(teacher_round, "run")

    def first_principles_collect_dir(self, teacher_round):

        return self.first_principles_substep_artifacts_dir(teacher_round, "collect")

    def first_principles_labeled_xyz(self, teacher_round):

        return self.first_principles_collect_dir(teacher_round) / "labeled.xyz"

    def first_principles_failed_xyz(self, teacher_round):

        return self.first_principles_collect_dir(teacher_round) / "failed.xyz"

    def user_initial_seed_pool_dir(self):

        return self.user_seed_pool
