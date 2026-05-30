from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from workflow_backends.execution_context import normalize_backend_request_context, resolve_backend_execution
from workflow_backends.student.deepmd.committee_config import load_committee_train_config, load_dp_train_template
from workflow_backends.student.deepmd.commands import build_dp_train_command
from workflow_artifacts.dp_committee_summary import write_committee_summary
from workflow_backends.student.deepmd.dataset_split import collect_existing_split, make_deepmd_train_valid_split
from workflow_runtime.logging.runtime_logging import run_logged_command
from workflow_backends.student.deepmd.training_watchdog import run_dp_train_with_watchdog, summarize_watchdog_attempts
from workflow_core.common.io import read_json, write_json
from workflow_runtime.environment import student_backend_role

ARCH_META_VERSION = 1
ARCH_META_FIELDS = (
    ("type_map", ("model", "type_map")),
    ("descriptor_type", ("model", "descriptor", "type")),
    ("rcut", ("model", "descriptor", "rcut")),
    ("rcut_smth", ("model", "descriptor", "rcut_smth")),
    ("descriptor_neuron", ("model", "descriptor", "neuron")),
    ("axis_neuron", ("model", "descriptor", "axis_neuron")),
    ("descriptor_resnet_dt", ("model", "descriptor", "resnet_dt")),
    ("fitting_neuron", ("model", "fitting_net", "neuron")),
    ("fitting_resnet_dt", ("model", "fitting_net", "resnet_dt")),
)


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(" -> ".join(path))
        current = current[key]
    return current


def _normalize_sel_value(value: Any) -> list[int]:
    if not isinstance(value, list) or not value or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Descriptor sel must be an explicit non-empty numeric list, got: {value!r}")
    return [int(item) for item in value]


def _extract_arch_meta_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    resolved_sel = _normalize_sel_value(_nested_get(payload, ("model", "descriptor", "sel")))
    arch_meta = {
        "resolved_sel": resolved_sel,
        "model_version": ARCH_META_VERSION,
    }
    for field_name, path in ARCH_META_FIELDS:
        arch_meta[field_name] = copy.deepcopy(_nested_get(payload, path))
    return arch_meta


def _try_extract_arch_meta_from_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path).resolve()
    try:
        payload = read_json(path)
    except Exception:
        return None
    try:
        arch_meta = _extract_arch_meta_from_payload(payload)
    except Exception:
        return None
    arch_meta["source_json"] = str(path)
    return arch_meta


def load_arch_meta(arch_meta_path: str | Path) -> dict[str, Any]:
    arch_meta_path = Path(arch_meta_path).resolve()
    payload = read_json(arch_meta_path)
    payload["resolved_sel"] = _normalize_sel_value(payload.get("resolved_sel"))
    return payload


def write_arch_meta(arch_meta_path: str | Path, arch_meta: dict[str, Any]) -> Path:
    arch_meta_path = Path(arch_meta_path).resolve()
    arch_meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(arch_meta)
    payload["resolved_sel"] = _normalize_sel_value(payload.get("resolved_sel"))
    payload.setdefault("model_version", ARCH_META_VERSION)
    with open(arch_meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return arch_meta_path


def apply_locked_architecture(payload: dict[str, Any], arch_meta: dict[str, Any]) -> dict[str, Any]:
    payload["model"]["descriptor"]["sel"] = _normalize_sel_value(arch_meta["resolved_sel"])
    return payload


def assert_architecture_compatible(payload: dict[str, Any], arch_meta: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for field_name, path in ARCH_META_FIELDS:
        expected = arch_meta.get(field_name)
        current = copy.deepcopy(_nested_get(payload, path))
        if current != expected:
            mismatches.append(f"{field_name}: current={current!r}, locked={expected!r}")

    current_sel = _nested_get(payload, ("model", "descriptor", "sel"))
    if isinstance(current_sel, list):
        normalized_current_sel = _normalize_sel_value(current_sel)
        if normalized_current_sel != _normalize_sel_value(arch_meta["resolved_sel"]):
            mismatches.append(
                f"resolved_sel: current={normalized_current_sel!r}, locked={_normalize_sel_value(arch_meta['resolved_sel'])!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Current DeePMD architecture is incompatible with the locked round0 architecture. "
            "This run cannot continue with --init-model. Mismatches: " + "; ".join(mismatches)
        )


def _harvest_arch_meta(model_dir: str | Path, rendered_input_path: str | Path) -> dict[str, Any] | None:
    rendered_input_path = Path(rendered_input_path).resolve()
    model_dir = Path(model_dir).resolve()

    arch_meta = _try_extract_arch_meta_from_json(rendered_input_path)
    if arch_meta is not None:
        arch_meta["source"] = "rendered_input"
        return arch_meta

    for candidate in sorted(model_dir.rglob("*.json")):
        if candidate.resolve() == rendered_input_path:
            continue
        arch_meta = _try_extract_arch_meta_from_json(candidate)
        if arch_meta is not None:
            arch_meta["source"] = "model_dir_json_scan"
            return arch_meta
    return None


def collect_systems_from_dataset_root(dataset_root: str | Path) -> tuple[list[str], list[str]]:
    dataset_root = Path(dataset_root).resolve()
    train_systems, valid_systems = collect_existing_split(dataset_root)

    if train_systems and valid_systems:
        return train_systems, valid_systems

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "val"

    if train_dir.exists():
        train_systems = [str(path.resolve()) for path in sorted(train_dir.iterdir()) if path.is_dir()]
    if valid_dir.exists():
        valid_systems = [str(path.resolve()) for path in sorted(valid_dir.iterdir()) if path.is_dir()]

    return train_systems, valid_systems


def collect_round_dp_systems(paths, round_id: int) -> tuple[list[str], list[str]]:
    del paths, round_id
    raise RuntimeError("DeepMD target-mode training requires explicit dataset_roots")


def collect_history_dp_systems(paths, round_id: int) -> tuple[list[str], list[str]]:
    train_systems: list[str] = []
    valid_systems: list[str] = []

    for history_round in range(round_id + 1):
        round_train, round_valid = collect_round_dp_systems(paths, history_round)
        train_systems.extend(round_train)
        valid_systems.extend(round_valid)

    return train_systems, valid_systems


def find_latest_checkpoint(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir).resolve()
    candidates = {}

    for path in model_dir.glob("model.ckpt-*"):
        suffix = path.name.split("model.ckpt-", 1)[-1]
        step = suffix.split(".", 1)[0]
        if step.isdigit():
            candidates[int(step)] = model_dir / f"model.ckpt-{step}"

    if not candidates:
        raise RuntimeError(f"No checkpoint found in {model_dir}")

    return candidates[max(candidates)]


def ensure_train_valid_systems(paths, round_id: int, cfg, context: dict[str, Any] | None = None) -> tuple[list[str], list[str], Path]:
    context = context or {}
    dataset_roots = context.get("dataset_roots")
    if dataset_roots:
        train_systems: list[str] = []
        valid_systems: list[str] = []
        for dataset_root in dataset_roots:
            round_train, round_valid = collect_systems_from_dataset_root(dataset_root)
            train_systems.extend(round_train)
            valid_systems.extend(round_valid)

        split_meta_path = Path(context.get("split_meta_path", Path(context["work_root"]) / "split_meta.json")).resolve()
        split_meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_roots": [str(Path(path).resolve()) for path in dataset_roots],
            "split_root": str(split_meta_path.parent.resolve()),
            "valid_fraction": cfg.valid_fraction,
            "split_seed": cfg.split_seed,
            "mode": "existing_split_dataset_roots",
            "train_systems": train_systems,
            "valid_systems": valid_systems,
        }
        with open(split_meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return train_systems, valid_systems, split_meta_path

    del paths, round_id, cfg
    raise RuntimeError("DeepMD target-mode training requires explicit dataset_roots")


def render_committee_member_input(
    train_config_path: str | Path,
    output_input_path: str | Path,
    train_systems: list[str],
    valid_systems: list[str],
    model_seed: int,
    model_index: int,
    type_map: list[str] | None = None,
    recommend_disp_freq: int | None = None,
    recommend_save_freq_mode: str | None = None,
    locked_arch_meta: dict[str, Any] | None = None,
) -> Path:
    train_template = load_dp_train_template(train_config_path, type_map=type_map)
    payload = copy.deepcopy(train_template)

    if locked_arch_meta is not None:
        assert_architecture_compatible(payload, locked_arch_meta)
        apply_locked_architecture(payload, locked_arch_meta)

    payload["training"]["training_data"]["systems"] = [str(Path(path).resolve()) for path in train_systems]
    payload["training"]["validation_data"]["systems"] = [str(Path(path).resolve()) for path in valid_systems]
    payload["model"]["descriptor"]["seed"] = int(model_seed)
    payload["model"]["fitting_net"]["seed"] = int(model_seed) + 1000
    payload["training"]["seed"] = int(model_seed) + 2000
    training_cfg = payload.setdefault("training", {})
    if recommend_disp_freq is not None:
        training_cfg["disp_freq"] = int(recommend_disp_freq)
        current_save_freq = training_cfg.get("save_freq")
        mode = (recommend_save_freq_mode or "same_as_disp").strip().lower()
        if mode == "same_as_disp":
            training_cfg["save_freq"] = int(recommend_disp_freq)
        elif mode == "double_disp":
            training_cfg["save_freq"] = int(recommend_disp_freq) * 2
        elif current_save_freq is None:
            training_cfg["save_freq"] = int(recommend_disp_freq)

    output_input_path = Path(output_input_path).resolve()
    output_input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_input_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_input_path


def parse_lcurve_last_metrics(lcurve_path: str | Path) -> dict[str, float | None]:
    lcurve_path = Path(lcurve_path).resolve()
    if not lcurve_path.exists():
        raise FileNotFoundError(f"lcurve.out not found: {lcurve_path}")

    with open(lcurve_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]

    if len(raw_lines) < 2:
        raise RuntimeError(f"lcurve.out does not contain enough rows: {lcurve_path}")

    header_line = raw_lines[0].lstrip("#").strip()
    data_lines = [line for line in raw_lines[1:] if not line.startswith("#")]
    if not header_line or not data_lines:
        raise RuntimeError(f"Unable to parse lcurve.out header/data rows: {lcurve_path}")

    header = header_line.split()
    values = data_lines[-1].split()
    if len(values) != len(header):
        raise RuntimeError(f"lcurve.out header/value mismatch in {lcurve_path}")

    table = dict(zip(header, values))

    def pick(*names: str) -> float | None:
        for name in names:
            if name in table:
                return float(table[name])
        return None

    return {
        "final_train_rmse_e": pick("rmse_e_trn", "rmse_e_train", "rmse_e"),
        "final_train_rmse_f": pick("rmse_f_trn", "rmse_f_train", "rmse_f"),
        "final_val_rmse_e": pick("rmse_e_val", "rmse_e_tst", "rmse_e_valid"),
        "final_val_rmse_f": pick("rmse_f_val", "rmse_f_tst", "rmse_f_valid"),
    }


def metrics_are_finite(metrics: dict[str, float | None]) -> bool:
    required_values = [
        metrics.get("final_train_rmse_e"),
        metrics.get("final_train_rmse_f"),
    ]
    optional_values = [
        metrics.get("final_val_rmse_e"),
        metrics.get("final_val_rmse_f"),
    ]
    return all(value is not None and math.isfinite(float(value)) for value in required_values) and all(
        value is None or math.isfinite(float(value)) for value in optional_values
    )


def can_reuse_member_result(result_file: Path, model_dir: Path) -> bool:
    if not result_file.exists():
        return False
    cached = read_json(result_file)
    graph_pb = Path(str(cached.get("graph_pb") or model_dir / "graph.pb"))
    lcurve_out = model_dir / "lcurve.out"
    if not graph_pb.exists() or not lcurve_out.exists():
        return False

    if cached.get("status") != "success":
        return False

    cached_metrics = {
        "final_train_rmse_e": cached.get("final_train_rmse_e"),
        "final_train_rmse_f": cached.get("final_train_rmse_f"),
        "final_val_rmse_e": cached.get("final_val_rmse_e"),
        "final_val_rmse_f": cached.get("final_val_rmse_f"),
    }
    return metrics_are_finite(cached_metrics)


def prepare_model_dir_for_training(model_dir: Path) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)


def _resolve_member_log_path(
    paths,
    round_id: int,
    *,
    backend_log_dir: str | Path | None,
    step_name: str,
    filename: str,
) -> Path:
    if backend_log_dir not in (None, ""):
        log_dir = Path(backend_log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / filename
    raise ValueError(
        "student backend member logs require an explicit backend_log_dir; "
        f"refusing legacy semantic log fallback for {step_name!r} in teacher_round={round_id}"
    )


def _command_with_execution_executable(command: list[str], executable: str | None) -> list[str]:
    if not executable:
        return list(command)
    if len(command) >= 3 and command[1:3] == ["-m", "deepmd"]:
        tail = command[3:]
    else:
        tail = command[1:]
    return [str(executable), *tail]


def _fold_shell_exec_args(command: list[str]) -> list[str]:
    command_items = [str(item) for item in command]
    if len(command_items) <= 3:
        return command_items
    if command_items[0] == "bash" and command_items[1] == "-lc":
        import shlex

        lines = str(command_items[2]).splitlines()
        for index in range(len(lines) - 1, -1, -1):
            stripped = lines[index].lstrip()
            if stripped.startswith("exec "):
                indent = lines[index][: len(lines[index]) - len(stripped)]
                lines[index] = f"{indent}{stripped} {shlex.join(command_items[3:])}"
                return ["bash", "-lc", "\n".join(lines)]
    return command_items


def _freeze_output_stem(freeze_command: list[str]) -> str:
    command = list(freeze_command)
    for index, item in enumerate(command):
        if item == "-o" and index + 1 < len(command):
            return str(command[index + 1])
        if item.startswith("--output="):
            return item.split("=", 1)[1]
    return "graph.pb"


def resolve_deepmd_freeze_output(model_dir: str | Path, freeze_command: list[str]) -> Path:
    model_dir = Path(model_dir).resolve()
    output = Path(_freeze_output_stem(freeze_command))
    if output.is_absolute():
        candidates = [output]
    else:
        raw = model_dir / output
        candidates = [raw]
        suffix = raw.suffix
        if suffix == ".pb":
            candidates.extend([raw.with_suffix(".pth"), raw.with_suffix(".pt")])
        elif suffix == "":
            candidates.extend([Path(str(raw) + ext) for ext in (".pb", ".pth", ".pt", ".pt2")])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    names = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"DeepMD freeze output not found after freeze; checked: {names}")


def train_one_committee_member(
    model_dir: str | Path,
    dp_command: list[str],
    freeze_command: list[str],
    train_log: str | Path,
    freeze_log: str | Path,
    watch_log: str | Path,
    attempt_index: int,
    attempt_seed: int,
    nan_watchdog_enabled: bool = True,
    nan_terminate_timeout_seconds: float = 10.0,
    nan_watch_lcurve_fallback: bool = True,
    nan_lcurve_poll_interval_seconds: float = 1.5,
    init_model: str | Path | None = None,
    finetune_model: str | Path | None = None,
    finetune_use_pretrain_script: bool = False,
    finetune_model_branch: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    input_json = model_dir / "input.json"

    train_cmd = build_dp_train_command(
        dp_command,
        input_json.name,
        init_model=init_model,
        finetune_model=finetune_model,
        finetune_use_pretrain_script=finetune_use_pretrain_script,
        finetune_model_branch=finetune_model_branch,
    )
    train_cmd = _fold_shell_exec_args(train_cmd)

    if nan_watchdog_enabled:
        watch_result = run_dp_train_with_watchdog(
            command=train_cmd,
            workdir=model_dir,
            log_file=train_log,
            watch_log_file=watch_log,
            input_json=input_json,
            attempt=attempt_index + 1,
            seed=attempt_seed,
            env=env,
            terminate_timeout_seconds=nan_terminate_timeout_seconds,
            watch_lcurve_fallback=nan_watch_lcurve_fallback,
            lcurve_poll_interval_seconds=nan_lcurve_poll_interval_seconds,
            quiet=True,
        )
        if watch_result["status"] != "success":
            raise RuntimeError(
                json.dumps(
                    {
                        "kind": "dp_train_watchdog_failure",
                        "status": watch_result["status"],
                        "attempt": attempt_index + 1,
                        "seed": attempt_seed,
                        "return_code": watch_result["return_code"],
                        "nan_detected": watch_result["nan_detected"],
                        "matched_line": watch_result.get("matched_line"),
                        "watch_log_file": watch_result["watch_log_file"],
                    },
                    ensure_ascii=False,
                )
            )
    else:
        run_logged_command(train_cmd, log_file=train_log, cwd=model_dir, env=env)
        watch_result = {
            "status": "success",
            "attempt": attempt_index + 1,
            "return_code": 0,
            "nan_detected": False,
            "watch_log_file": str(Path(watch_log).resolve()),
        }
    run_logged_command(list(freeze_command), log_file=freeze_log, cwd=model_dir, env=env)

    graph_pb = resolve_deepmd_freeze_output(model_dir, freeze_command)
    lcurve_out = model_dir / "lcurve.out"
    if not lcurve_out.exists():
        raise RuntimeError(f"lcurve.out not found after training: {lcurve_out}")

    metrics = parse_lcurve_last_metrics(lcurve_out)
    if not metrics_are_finite(metrics):
        raise RuntimeError(
            json.dumps(
                {
                    "kind": "dp_train_watchdog_failure",
                    "status": "nan_detected",
                    "attempt": attempt_index + 1,
                    "seed": attempt_seed,
                    "return_code": 0,
                    "nan_detected": True,
                    "matched_line": f"non-finite metrics in {lcurve_out}",
                    "watch_log_file": str(Path(watch_log).resolve()),
                },
                ensure_ascii=False,
            )
        )

    result = {
        "status": "success",
        "model_path": str(graph_pb.resolve()),
        "graph_pb": str(graph_pb.resolve()),
        "lcurve_out": str(lcurve_out.resolve()),
        "watch_log_file": str(Path(watch_log).resolve()),
        "watchdog_result": watch_result,
        **metrics,
    }

    result_path = model_dir / "train_result.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    return result


def _parse_watchdog_error(exc: Exception) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(exc))
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("kind") == "dp_train_watchdog_failure":
        return payload
    return None


def run_dp_train_committee(cfg_path: str | Path, paths, round_id: int, context: dict[str, Any] | None = None) -> str:
    context = context or {}
    config = load_committee_train_config(cfg_path, identity=dict(context.get("student_identity", {}) or {}))
    continue_training = bool(context.get("continue_training_override", False))
    resolved_decision = dict(context.get("resolved_train_decision", {}) or {})
    training_stage = str(
        context.get("training_stage", "finetune" if continue_training else "bootstrap")
    ).strip().lower()
    foundation_bootstrap = dict(context.get("foundation_bootstrap") or {})
    foundation_finetune_enabled = bool(foundation_bootstrap)
    foundation_finetune_model = Path(str(foundation_bootstrap["model_path"])).resolve() if foundation_finetune_enabled else None
    foundation_use_pretrain_script = bool(foundation_bootstrap.get("use_pretrain_script", False))
    foundation_model_branch = foundation_bootstrap.get("model_branch")
    selected_train_config = config.select_train_config(
        resolved_decision.get("effective_train_recipe_branch") or training_stage
    )
    effective_train_recipe_branch = str(
        resolved_decision.get("effective_train_recipe_branch")
        or training_stage
    )
    if context.get("work_root") in (None, ""):
        raise ValueError("DeepMD target-mode training requires an explicit work_root")
    work_root = Path(context["work_root"]).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    arch_meta_path = Path(context.get("arch_meta_path", work_root / "arch_meta.json")).resolve()
    backend_execution = context.get("backend_execution")
    if backend_execution is None:
        backend_execution = resolve_backend_execution(
            _RequestExecutionProxy(execution_context=context.get("execution_context")),
            role=student_backend_role("deepmd", "train"),
        )
    base_train_env = dict(os.environ)
    base_train_env.update(config.env)
    train_env = backend_execution.merge_env(
        base_train_env,
        context.get("env_override", {}),
    )
    dp_executable = backend_execution.required_executable("dp")
    dp_command = backend_execution.shell_command(_command_with_execution_executable(config.dp_command, dp_executable))
    freeze_command = backend_execution.shell_command(_command_with_execution_executable(config.freeze_command, dp_executable))
    backend_log_dir = context.get("backend_log_dir")

    locked_arch_meta = None
    locked_arch_meta_path = None

    train_systems, valid_systems, split_meta_path = ensure_train_valid_systems(paths, round_id, config, context=context)

    members: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for model_index, model_seed in enumerate(config.model_seeds[:config.committee_size]):
        model_dir = (work_root / f"model_{model_index:03d}").resolve()
        input_json_path = (model_dir / "input.json").resolve()

        member_record: dict[str, Any] = {
            "model_index": model_index,
            "seed": model_seed,
            "attempted_seeds": [],
            "model_dir": str(model_dir),
            "input_json": str(input_json_path),
            "graph_pb": str((model_dir / "graph.pb").resolve()),
            "lcurve_out": str((model_dir / "lcurve.out").resolve()),
            "status": "pending",
            "attempts": 0,
            "nan_attempts": 0,
            "watch_log_file": None,
            "watchdog_attempts": [],
            "final_train_rmse_e": None,
            "final_train_rmse_f": None,
            "final_val_rmse_e": None,
            "final_val_rmse_f": None,
        }

        result_file = model_dir / "train_result.json"
        if config.reuse_existing_model and can_reuse_member_result(result_file, model_dir):
            cached = read_json(result_file)
            member_record.update(cached)
            members.append(member_record)
            print(f"[INFO] Reusing existing DP committee member {model_index:03d}")
            continue

        log_round_id = int(context.get("log_round_id", round_id))
        log_prefix = str(context.get("log_prefix", ""))

        init_model = None
        finetune_model = None
        finetune_use_pretrain_script = False
        finetune_model_branch = None
        previous_model_root = context.get("previous_model_root")
        if foundation_finetune_enabled:
            finetune_model = foundation_finetune_model
            finetune_use_pretrain_script = foundation_use_pretrain_script
            finetune_model_branch = None if foundation_model_branch is None else str(foundation_model_branch)
        elif continue_training and previous_model_root is not None:
            prev_member_dir = Path(previous_model_root).resolve() / f"model_{model_index:03d}"
            if prev_member_dir.exists():
                init_model = find_latest_checkpoint(prev_member_dir)
            candidate_arch_meta_path = Path(previous_model_root).resolve() / "arch_meta.json"
            if locked_arch_meta is None and candidate_arch_meta_path.exists():
                locked_arch_meta = load_arch_meta(candidate_arch_meta_path)
                locked_arch_meta_path = candidate_arch_meta_path.resolve()
        elif continue_training and round_id > 0:
            raise RuntimeError("Continuing DeePMD training requires an explicit previous_model_root")

        if continue_training and locked_arch_meta is None:
            raise RuntimeError(
                "Continuing DeePMD training requires a locked architecture file, but none was found. "
                f"Expected previous arch_meta.json near: {Path(previous_model_root).resolve() if previous_model_root is not None else 'previous round directory'}"
            )

        for attempt_index in range(config.max_retries_per_member + 1):
            attempt_seed = int(model_seed) + attempt_index * int(config.retry_seed_stride)
            member_record["seed"] = attempt_seed
            member_record["attempted_seeds"].append(attempt_seed)
            member_record["attempts"] = attempt_index + 1

            if model_dir.exists():
                print(f"[INFO] Resetting DP committee member directory before retraining: {model_dir}")
            prepare_model_dir_for_training(model_dir)

            input_json = render_committee_member_input(
                train_config_path=selected_train_config,
                output_input_path=input_json_path,
                train_systems=train_systems,
                valid_systems=valid_systems,
                model_seed=attempt_seed,
                model_index=model_index,
                type_map=config.type_map,
                recommend_disp_freq=config.recommend_disp_freq,
                recommend_save_freq_mode=config.recommend_save_freq_mode,
                locked_arch_meta=locked_arch_meta,
            )
            member_record["input_json"] = str(input_json.resolve())
            if foundation_finetune_enabled:
                member_record["initialization_mode"] = "foundation_finetune"
                member_record["foundation_model_path"] = str(foundation_finetune_model)
                member_record["foundation_use_pretrain_script"] = bool(foundation_use_pretrain_script)
                member_record["foundation_model_branch"] = finetune_model_branch

            print(
                f"[INFO] Training DP committee member {model_index:03d} in {model_dir} "
                f"(attempt {attempt_index + 1}/{config.max_retries_per_member + 1}, seed={attempt_seed})"
            )

            log_suffix = "" if attempt_index == 0 else f"_retry{attempt_index}"
            train_log = _resolve_member_log_path(
                paths,
                log_round_id,
                backend_log_dir=backend_log_dir,
                step_name="train_dp_committee",
                filename=f"{log_prefix}dp_member_{model_index:03d}{log_suffix}_train.log",
            )
            freeze_log = _resolve_member_log_path(
                paths,
                log_round_id,
                backend_log_dir=backend_log_dir,
                step_name="freeze_dp_committee",
                filename=f"{log_prefix}dp_member_{model_index:03d}{log_suffix}_freeze.log",
            )
            watch_log = _resolve_member_log_path(
                paths,
                log_round_id,
                backend_log_dir=backend_log_dir,
                step_name="train_dp_committee_watchdog",
                filename=f"{log_prefix}dp_member_{model_index:03d}_train_watch.log",
            )
            member_record["watch_log_file"] = str(Path(watch_log).resolve())
            if attempt_index == 0:
                Path(watch_log).write_text("", encoding="utf-8")

            try:
                result = train_one_committee_member(
                    model_dir=model_dir,
                    dp_command=dp_command,
                    freeze_command=freeze_command,
                    train_log=train_log,
                    freeze_log=freeze_log,
                    watch_log=watch_log,
                    attempt_index=attempt_index,
                    attempt_seed=attempt_seed,
                    nan_watchdog_enabled=config.nan_watchdog_enabled,
                    nan_terminate_timeout_seconds=config.nan_terminate_timeout_seconds,
                    nan_watch_lcurve_fallback=config.nan_watch_lcurve_fallback,
                    nan_lcurve_poll_interval_seconds=config.nan_lcurve_poll_interval_seconds,
                    init_model=init_model,
                    finetune_model=finetune_model,
                    finetune_use_pretrain_script=finetune_use_pretrain_script,
                    finetune_model_branch=finetune_model_branch,
                    env=train_env,
                )
                member_record.update(result)
                watch_result = result.get("watchdog_result") or {
                    "status": "success",
                    "return_code": 0,
                    "nan_detected": False,
                    "attempt": attempt_index + 1,
                    "seed": attempt_seed,
                }
                member_record["watchdog_attempts"].append(
                    {
                        "status": watch_result.get("status", "success"),
                        "return_code": watch_result.get("return_code", 0),
                        "nan_detected": bool(watch_result.get("nan_detected", False)),
                        "attempt": watch_result.get("attempt", attempt_index + 1),
                        "seed": attempt_seed,
                    }
                )
                member_record["train_watch_summary"] = summarize_watchdog_attempts(
                    attempts=member_record["watchdog_attempts"],
                    workdir=model_dir,
                    log_file=watch_log,
                )
                break
            except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                last_error = exc
                member_record["status"] = "failed"
                member_record["error"] = str(exc)
                member_record["watch_log_file"] = str(watch_log)
                error_payload = _parse_watchdog_error(exc)
                if isinstance(error_payload, dict):
                    member_record["last_failure"] = error_payload
                    member_record["watchdog_attempts"].append(
                        {
                            "status": error_payload.get("status", "failed"),
                            "return_code": error_payload.get("return_code", -1),
                            "nan_detected": bool(error_payload.get("nan_detected")),
                            "attempt": error_payload.get("attempt", attempt_index + 1),
                            "seed": error_payload.get("seed", attempt_seed),
                        }
                    )
                    if error_payload.get("nan_detected"):
                        member_record["nan_attempts"] = int(member_record.get("nan_attempts", 0)) + 1
                member_record["train_watch_summary"] = summarize_watchdog_attempts(
                    attempts=member_record["watchdog_attempts"]
                    or [{"status": "failed", "return_code": -1, "nan_detected": False}],
                    workdir=model_dir,
                    log_file=watch_log,
                )
                with open(result_file, "w", encoding="utf-8") as handle:
                    json.dump(member_record, handle, indent=2, ensure_ascii=False)
                if attempt_index >= config.max_retries_per_member:
                    break

        members.append(member_record)

    successful_count = sum(1 for member in members if member.get("status") == "success")
    if successful_count == 0:
        raise RuntimeError("DP committee training produced no successful members") from last_error

    summary_path = Path(context.get("summary_path", work_root / "summary.json")).resolve()
    write_committee_summary(
        summary_path=summary_path,
        committee_size=config.committee_size,
        work_root=work_root,
        split_meta_path=split_meta_path,
        train_systems=train_systems,
        valid_systems=valid_systems,
        members=members,
    )

    arch_meta_payload = None
    if training_stage == "finetune" and locked_arch_meta is not None:
        arch_meta_payload = copy.deepcopy(locked_arch_meta)
        if locked_arch_meta_path is not None:
            arch_meta_payload["inherited_from"] = str(locked_arch_meta_path)
    else:
        for member in members:
            if member.get("status") != "success":
                continue
            candidate_meta = _harvest_arch_meta(member["model_dir"], member["input_json"])
            if candidate_meta is not None:
                arch_meta_payload = candidate_meta
                break

    if arch_meta_payload is not None:
        if resolved_decision:
            arch_meta_payload["effective_train_recipe_branch"] = effective_train_recipe_branch
            if resolved_decision.get("initialization_mode") is not None:
                arch_meta_payload["initialization_mode"] = str(resolved_decision["initialization_mode"])
        else:
            arch_meta_payload["training_stage"] = training_stage
        write_arch_meta(arch_meta_path, arch_meta_payload)

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["continue_training"] = continue_training
        if resolved_decision:
            payload.pop("training_stage", None)
            payload["selected_train_recipe_path"] = str(selected_train_config.resolve())
            payload["effective_train_recipe_branch"] = effective_train_recipe_branch
            payload["dataset_mode"] = str(resolved_decision.get("dataset_mode", ""))
            payload["initialization_mode"] = str(resolved_decision.get("initialization_mode", ""))
            payload["train_recipe_branches"] = {
                "scratch_entry": str(config.scratch_entry_train_config.resolve()),
                "foundation_entry": str(config.foundation_entry_train_config.resolve()),
                "continuation": str(config.continuation_train_config.resolve()),
            }
        else:
            payload["training_stage"] = training_stage
            payload["selected_train_config"] = str(selected_train_config.resolve())
            payload["bootstrap_train_config"] = str(config.scratch_entry_train_config.resolve())
            payload["finetune_train_config"] = str(config.continuation_train_config.resolve())
            payload["initialization_mode"] = "foundation_finetune" if foundation_finetune_enabled else (
                "incremental_finetune" if continue_training else "bootstrap"
            )
        if foundation_finetune_enabled:
            payload["foundation_bootstrap"] = {
                "model_path": str(foundation_finetune_model),
                "type_map": list(foundation_bootstrap.get("type_map", []) or []),
                "use_pretrain_script": bool(foundation_use_pretrain_script),
                "model_branch": foundation_model_branch,
            }
        payload["arch_meta_path"] = str(arch_meta_path.resolve()) if arch_meta_path.exists() else None
        payload["env"] = dict(config.env)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    print(f"[INFO] DP committee summary written to {summary_path.resolve()}")
    return str(summary_path.resolve())


def run_deepmd_train_committee(paths, request) -> str:
    normalized = normalize_backend_request_context(request, role=student_backend_role("deepmd", "train"))
    context = normalized.context
    context["backend_execution"] = normalized.execution
    work_root = Path(request.work_root).resolve()
    context.update(
        {
            "dataset_roots": list(request.dataset_roots or []),
            "work_root": str(work_root),
            "summary_path": str(Path(request.summary_path).resolve()),
            "arch_meta_path": str(Path(context.get("arch_meta_path", work_root / "arch_meta.json")).resolve()),
            "split_meta_path": str(Path(context.get("split_meta_path", work_root / "split_meta.json")).resolve()),
            "log_round_id": int(context.get("log_round_id", request.outer_round)),
            "log_prefix": str(
                context.get(
                    "log_prefix",
                    f"student_subworkflow_r{request.outer_round}_student{request.student_round}_",
                )
            ),
            "previous_model_root": (
                str(Path(request.previous_model_root).resolve())
                if request.previous_model_root is not None
                else None
            ),
            "continue_training_override": bool(request.continue_training),
            "training_stage": str(request.training_stage),
        }
    )
    return run_dp_train_committee(request.backend_cfg, paths, request.outer_round, context=context)


class _RequestExecutionProxy:
    def __init__(self, *, execution_context):
        self.execution_context = execution_context
        self.context = {}
