from pathlib import Path


class PathManager:

    def __init__(self, workspace_root=".", runtime_root=None):

        self.workspace_root = Path(workspace_root).resolve()
        if runtime_root is None:
            runtime_root = self.workspace_root / "runs" / "default"
        self.runtime_root = Path(runtime_root).resolve()

        # Backward-compatible alias for runtime-scoped generated files.
        self.root = self.runtime_root

        self.data = self.runtime_root / "data"
        self.results = self.runtime_root / "results"
        self.logs = self.runtime_root / "logs"
        self.run_meta = self.runtime_root / "run_meta"
        self.metrics = self.runtime_root / "metrics"
        self.plots = self.runtime_root / "plots"
        self.user_assets = self.workspace_root / "user_assets"
        self.mydataset = self.user_assets
        self.configs = self.workspace_root / "configs"
        self.runs = self.workspace_root / "runs"

        self.dp_train = self.data / "dp_train_data"
        self.dp_subworkflow = self.data / "dp_subworkflow"
        self.sampling = self.data / "sampling"
        self.active_learning = self.data / "active_learning"
        self.mace_train = self.data / "mace_train_data"
        self.cycle = self.data / "committee_cycle"
        self.seed_pool = self.data / "seed_pool"

        self.dp_committee = self.results / "dp_committee_model"
        self.dp_subworkflow_results = self.results / "dp_subworkflow"
        self.mace_committee = self.results / "mace_committee_model"
        self.mace_model = self.results / "mace_foundation_model"
        self.cycle_results = self.results / "committee_cycle"
        self.seed_pool_results = self.results / "seed_pool"
        self.dft_submit = self.data / "dft_submit"
        self.dft_submit_results = self.results / "dft_submit"

        self.user_mace_train = self.user_assets / "mace_train"
        self.user_mace_finetune = self.user_assets / "mace_finetune"
        self.user_dp_train = self.user_assets / "dp_train"
        self.user_seed_pool = self.user_assets / "seed_pool"
        self.user_foundation_models = self.user_assets / "foundation_models"
        self.user_dft = self.user_assets / "dft"

    def runtime_data_dir(self):

        return self.data

    def runtime_results_dir(self):

        return self.results

    def runtime_logs_dir(self):

        return self.logs

    def runtime_meta_dir(self):

        return self.run_meta

    def runtime_metrics_dir(self):

        return self.metrics

    def runtime_plots_dir(self):

        return self.plots

    def run_manifest_json(self):

        return self.runtime_meta_dir() / "run_manifest.json"

    def run_summary_json(self):

        return self.runtime_meta_dir() / "run_summary.json"

    def metrics_run_summary_json(self):

        return self.runtime_metrics_dir() / "run_summary.json"

    def metrics_run_summary_csv(self):

        return self.runtime_metrics_dir() / "run_summary.csv"

    def metrics_round_metrics_csv(self):

        return self.runtime_metrics_dir() / "round_metrics.csv"

    def metrics_step_metrics_csv(self):

        return self.runtime_metrics_dir() / "step_metrics.csv"

    def metrics_training_metrics_csv(self):

        return self.runtime_metrics_dir() / "training_metrics.csv"

    def metrics_sampling_metrics_csv(self):

        return self.runtime_metrics_dir() / "sampling_metrics.csv"

    def metrics_labeling_metrics_csv(self):

        return self.runtime_metrics_dir() / "labeling_metrics.csv"

    def metrics_resource_metrics_csv(self):

        return self.runtime_metrics_dir() / "resource_metrics.csv"

    def metrics_event_log_jsonl(self):

        return self.runtime_metrics_dir() / "event_log.jsonl"

    def metrics_config_snapshot_json(self):

        return self.runtime_metrics_dir() / "config_snapshot.json"

    def distribution_plots_dir(self):

        return self.runtime_plots_dir() / "distribution"

    def outer_round_distribution_dir(self):

        return self.distribution_plots_dir() / "outer_rounds"

    def dp_round_distribution_dir(self):

        return self.distribution_plots_dir() / "dp_rounds"

    def outer_round_dft_distribution_png(self, outer_round):

        return self.outer_round_distribution_dir() / f"r{outer_round}_dft_increment_vs_base.png"

    def outer_round_dft_distribution_json(self, outer_round):

        return self.outer_round_distribution_dir() / f"r{outer_round}_dft_increment_vs_base.json"

    def outer_round_dft_distribution_csv(self, outer_round):

        return self.outer_round_distribution_dir() / f"r{outer_round}_dft_increment_vs_base.csv"

    def dp_round_distribution_png(self, outer_round, dp_round):

        return self.dp_round_distribution_dir() / f"r{outer_round}_dp_r{dp_round}_increment_vs_prev.png"

    def dp_round_distribution_json(self, outer_round, dp_round):

        return self.dp_round_distribution_dir() / f"r{outer_round}_dp_r{dp_round}_increment_vs_prev.json"

    def dp_round_distribution_csv(self, outer_round, dp_round):

        return self.dp_round_distribution_dir() / f"r{outer_round}_dp_r{dp_round}_increment_vs_prev.csv"

    def workflow_snapshot_yaml(self):

        return self.runtime_meta_dir() / "workflow_snapshot.yaml"

    def workflow_round_meta_dir(self, outer_round):

        return self.runtime_meta_dir() / "rounds" / f"r{outer_round}"

    def workflow_round_state_json(self, outer_round):

        return self.workflow_round_meta_dir(outer_round) / "workflow_state.json"

    def workflow_round_marker_dir(self, outer_round):

        return self.workflow_round_meta_dir(outer_round) / "markers"

    def workflow_round_preflight_json(self, outer_round):

        return self.workflow_round_meta_dir(outer_round) / "preflight_report.json"

    def mace_foundation_data_dir(self):

        return self.mace_train / "foundation"

    def dp_data_dir(self, r):

        return self.dp_train / f"r{r}"

    def mace_model_dir(self, r=None):

        return self.mace_model

    def mace_committee_dir(self, r):

        return self.mace_committee / f"r{r}"

    def dp_train_data_dir(self, r):

        return self.dp_data_dir(r) / "train"

    def dp_val_data_dir(self, r):

        return self.dp_data_dir(r) / "val"

    def mace_train_data_dir(self, r):

        return self.mace_foundation_data_dir() / "train"

    def active_learning_data_dir(self, r):

        return self.active_learning / f"r{r}"

    def sampling_dir(self, r):

        return self.sampling / f"r{r}"

    def sampling_seed_pool_xyz(self, r):

        return self.sampling_dir(r) / "seed_pool.xyz"

    def manual_seed_dir(self, r):

        return self.sampling_dir(r) / "manual_seed"

    def sampling_traj_pattern(self, r):

        return str(self.sampling_dir(r) / "T_*" / "*.xyz")

    def init_xyz(self, r):

        return self.active_learning_data_dir(r) / "dft.xyz"

    def selected_xyz(self, r):

        return self.dp_data_dir(r) / "fps_selected.xyz"

    def dp_committee_round_dir(self, r):

        return self.dp_committee / f"r{r}"

    def dp_committee_member_model_dir(self, r, model_index):

        return self.dp_committee_round_dir(r) / f"model_{model_index:03d}"

    def dp_committee_summary_json(self, r):

        return self.dp_committee_round_dir(r) / "summary.json"

    def dp_sampling_committee_dir(self, r):

        return self.sampling_dir(r) / "committee"

    def dp_subworkflow_dir(self, outer_round):

        return self.dp_subworkflow / f"r{outer_round}"

    def dp_subworkflow_round_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_dir(outer_round) / f"dp_r{dp_round}"

    def dp_subworkflow_round_label_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_round_dir(outer_round, dp_round) / "label"

    def dp_subworkflow_round_active_learning_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_round_dir(outer_round, dp_round) / "active_learning"

    def dp_subworkflow_round_dft_screening_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_results_round_dir(outer_round, dp_round) / "dft_screening"

    def dp_subworkflow_round_dft_screening_summary_json(self, outer_round, dp_round):

        return self.dp_subworkflow_round_dft_screening_dir(outer_round, dp_round) / "summary.json"

    def dp_subworkflow_round_dft_screened_xyz(self, outer_round, dp_round):

        return self.dp_subworkflow_round_active_learning_dir(outer_round, dp_round) / "dft_screened.xyz"

    def dp_subworkflow_round_seed_pool_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_round_dir(outer_round, dp_round) / "seed_pool"

    def dp_subworkflow_seed_pool_xyz(self, outer_round, dp_round):

        return self.dp_subworkflow_round_seed_pool_dir(outer_round, dp_round) / "seed_pool.xyz"

    def dp_subworkflow_round_train_xyz(self, outer_round, dp_round):

        return self.dp_subworkflow_round_label_dir(outer_round, dp_round) / "train.xyz"

    def dp_subworkflow_mace_al_pool_xyz(self, outer_round):

        return self.dp_subworkflow_dir(outer_round) / "mace_al_pool.xyz"

    def dp_subworkflow_screened_dft_pool_xyz(self, outer_round):

        return self.dp_subworkflow_dir(outer_round) / "screened_dft_pool.xyz"

    def dp_subworkflow_dft_screening_reference_xyz(self, outer_round):

        return self.dp_subworkflow_dir(outer_round) / "dft_screening_reference" / "physical_merged.xyz"

    def dp_subworkflow_status_json(self, outer_round):

        return self.dp_subworkflow_results / f"r{outer_round}" / "status.json"

    def dp_subworkflow_results_dir(self, outer_round):

        return self.dp_subworkflow_results / f"r{outer_round}"

    def dp_subworkflow_results_round_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_results_dir(outer_round) / f"dp_r{dp_round}"

    def dp_subworkflow_round_manifest_json(self, outer_round, dp_round):

        return self.dp_subworkflow_results_round_dir(outer_round, dp_round) / "manifest.json"

    def dp_subworkflow_round_summary_json(self, outer_round, dp_round):

        return self.dp_subworkflow_results_round_dir(outer_round, dp_round) / "round_summary.json"

    def dp_subworkflow_round_sampling_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_results_round_dir(outer_round, dp_round) / "sampling"

    def dp_subworkflow_round_entry_summary_json(self, outer_round):

        return self.dp_subworkflow_results_round_dir(outer_round, 0) / "cross_outer_entry_summary.json"

    def dp_subworkflow_round_entry_merged_xyz(self, outer_round):

        return self.dp_subworkflow_round_dir(outer_round, 0) / "cross_outer_entry_merged.xyz"

    def dp_subworkflow_round_committee_dir(self, outer_round, dp_round):

        return self.dp_subworkflow_results_round_dir(outer_round, dp_round) / "dp_committee"

    def dp_subworkflow_round_arch_meta_json(self, outer_round, dp_round):

        return self.dp_subworkflow_round_committee_dir(outer_round, dp_round) / "arch_meta.json"

    def committee_cycle_dir(self, outer_round):

        return self.cycle / f"r{outer_round}"

    def committee_cycle_results_dir(self, outer_round):

        return self.cycle_results / f"r{outer_round}"

    def committee_cycle_status_json(self, outer_round):

        return self.committee_cycle_results_dir(outer_round) / "status.json"

    def committee_cycle_finetune_merged_xyz(self, outer_round):

        return self.committee_cycle_dir(outer_round) / "finetune_input" / "train_merged.xyz"

    def committee_cycle_dp_bootstrap_dir(self, outer_round):

        return self.committee_cycle_dir(outer_round) / "dp_bootstrap_input"

    def committee_cycle_dp_bootstrap_merged_xyz(self, outer_round):

        return self.committee_cycle_dp_bootstrap_dir(outer_round) / "train_merged.xyz"

    def committee_cycle_dp_bootstrap_summary_json(self, outer_round):

        return self.committee_cycle_dp_bootstrap_dir(outer_round) / "summary.json"

    def committee_cycle_finetune_runtime_json(self, outer_round):

        return self.committee_cycle_dir(outer_round) / "finetune_runtime.json"

    def seed_pool_root_dir(self):

        return self.seed_pool

    def seed_pool_systems_dir(self):

        return self.seed_pool_root_dir() / "systems"

    def seed_pool_xyz(self):

        return self.seed_pool_root_dir() / "seed_pool.xyz"

    def seed_pool_meta_json(self):

        return self.seed_pool_root_dir() / "seed_pool_meta.json"

    def seed_candidate_pool_xyz(self):

        return self.seed_pool_root_dir() / "seed_candidate_pool.xyz"

    def seed_candidate_pool_systems_dir(self):

        return self.seed_pool_root_dir() / "candidate_systems"

    def seed_pool_archive_dir(self):

        return self.seed_pool_root_dir() / "archive"

    def seed_pool_archive_candidate_xyz(self, outer_round, dp_round):

        return self.seed_pool_archive_dir() / f"r{outer_round}_dp{dp_round}_candidate.xyz"

    def seed_pool_archive_candidate_summary_json(self, outer_round, dp_round):

        return self.seed_pool_archive_dir() / f"r{outer_round}_dp{dp_round}_candidate_summary.json"

    def seed_pool_selected_dir(self):

        return self.seed_pool_root_dir() / "selected"

    def seed_pool_selected_xyz(self, outer_round, dp_round):

        return self.seed_pool_selected_dir() / f"r{outer_round}_dp{dp_round}_seeds.xyz"

    def seed_pool_selected_systems_dir(self, outer_round, dp_round):

        return self.seed_pool_selected_dir() / f"r{outer_round}_dp{dp_round}_systems"

    def seed_pool_results_dir(self):

        return self.seed_pool_results

    def seed_pool_update_summary_json(self, outer_round, dp_round):

        return self.seed_pool_results_dir() / f"r{outer_round}_dp{dp_round}_update_summary.json"

    def seed_pool_final_snapshot_xyz(self, outer_round):

        return self.seed_pool_results_dir() / f"r{outer_round}_final_seed_pool.xyz"

    def seed_pool_final_snapshot_meta_json(self, outer_round):

        return self.seed_pool_results_dir() / f"r{outer_round}_final_seed_pool_meta.json"

    def user_initial_seed_pool_dir(self):

        return self.user_seed_pool

    def user_external_dft_root_dir(self):

        return self.user_dft

    def user_external_dft_round_dir(self, outer_round):

        return self.user_external_dft_root_dir() / f"r{outer_round}"

    def user_external_dft_labeled_xyz(self, outer_round):

        return self.user_external_dft_round_dir(outer_round) / "labeled.xyz"

    def user_external_dft_failed_xyz(self, outer_round):

        return self.user_external_dft_round_dir(outer_round) / "failed.xyz"

    def user_external_dft_summary_json(self, outer_round):

        return self.user_external_dft_round_dir(outer_round) / "summary.json"

    def user_external_dft_manifest_json(self, outer_round):

        return self.user_external_dft_round_dir(outer_round) / "import_manifest.json"

    def dft_submit_data_dir(self, outer_round):

        return self.dft_submit / f"r{outer_round}"

    def dft_submit_source_dir(self, outer_round):

        return self.dft_submit_data_dir(outer_round) / "source"

    def dft_submit_source_xyz(self, outer_round):

        return self.dft_submit_source_dir(outer_round) / "input_for_dft.xyz"

    def dft_submit_source_meta_json(self, outer_round):

        return self.dft_submit_source_dir(outer_round) / "input_meta.json"

    def dft_submit_job_root(self, outer_round):

        return self.dft_submit_data_dir(outer_round) / "cp2k_jobs"

    def dft_submit_results_dir(self, outer_round):

        return self.dft_submit_results / f"r{outer_round}"

    def dft_submit_task_map_json(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "task_map.json"

    def dft_submit_record_json(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "record.dft.json"

    def dft_submit_status_json(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "status.json"

    def dft_submit_manifest_json(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "manifest.json"

    def dft_submit_log_path(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "submit.log"

    def dft_submit_submit_summary_json(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "submit_summary.json"

    def dft_submit_labeled_xyz(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "labeled.xyz"

    def dft_submit_failed_xyz(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "failed.xyz"

    def dft_submit_collect_summary_json(self, outer_round):

        return self.dft_submit_results_dir(outer_round) / "summary.json"
