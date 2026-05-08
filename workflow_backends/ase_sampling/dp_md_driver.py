from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from deepmd.calculator import DP

from workflow_backends.deepmd.online_monitor import CommitteeTrajectoryMonitor, NoveltyTracker, get_consistent_energy_forces
from workflow_policies.sampling_jobs import build_sampling_jobs
from workflow_core.feature_extractor import build_feature_extractor
from workflow_core.common.io import append_jsonl, load_atoms_list


def _load_reference_frames(reference_xyz_files: list[str | Path]) -> list:
    frames = []
    for path in reference_xyz_files:
        try:
            frames.extend(load_atoms_list(path))
        except Exception as exc:
            print(f"[WARN] Failed to read committee reference xyz {path}: {exc}")
    return frames


def build_novelty_context(
    initial_structure_paths: list[str],
    reference_xyz_files: list[str | Path],
    novelty_cfg: dict[str, Any],
) -> dict[str, Any]:
    seed_atoms = [read(str(Path(path).resolve()), index=0) for path in initial_structure_paths]
    reference_frames = _load_reference_frames(reference_xyz_files)

    extractor_cfg = {
        "feature_type": novelty_cfg.get("feature_type", "geometric"),
    }
    extractor = build_feature_extractor(extractor_cfg)
    fit_atoms = seed_atoms + reference_frames
    extractor.fit(fit_atoms if fit_atoms else seed_atoms)

    ref_features = [extractor.transform(atoms) for atoms in reference_frames]
    from workflow_backends.deepmd.online_monitor import estimate_novelty_threshold

    novelty_threshold = estimate_novelty_threshold(ref_features, novelty_cfg)
    return {
        "feature_extractor": extractor,
        "reference_features": ref_features,
        "novelty_threshold": novelty_threshold,
        "n_reference_frames": len(reference_frames),
    }


def run_single_md_job(
    job_cfg: dict[str, Any],
    work_root: str | Path,
    md_cfg: dict[str, Any],
    hard_stop_cfg: dict[str, Any],
    save_policy_cfg: dict[str, Any],
    early_stop_cfg: dict[str, Any],
    novelty_context: dict[str, Any],
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    job_dir = work_root / job_cfg["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)

    initial_structure_path = Path(job_cfg["initial_structure_path"]).resolve()
    atoms = read(str(initial_structure_path), index=0)
    atoms.calc = DP(str(Path(job_cfg["driver_graph_path"]).resolve()))

    rng = np.random.default_rng(int(job_cfg["job_seed"]))
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=float(job_cfg["temperature"]),
        rng=rng,
        force_temp=bool(md_cfg.get("force_temp", True)),
    )
    Stationary(atoms)
    if bool(md_cfg.get("zero_rotation", False)):
        ZeroRotation(atoms)

    friction = float(md_cfg.get("friction", 0.05)) / units.fs
    dyn = Langevin(
        atoms,
        timestep=float(job_cfg["timestep_fs"]) * units.fs,
        temperature_K=float(job_cfg["temperature"]),
        friction=friction,
    )

    trajectory_xyz = job_dir / "trajectory.xyz"
    saved_frames_xyz = job_dir / "saved_frames.xyz"
    md_log_jsonl = job_dir / "md_log.jsonl"
    online_frames_jsonl = job_dir / "online_frames.jsonl"
    driver_info_json = job_dir / "driver_info.json"

    trajectory_xyz.write_text("", encoding="utf-8")
    saved_frames_xyz.write_text("", encoding="utf-8")
    md_log_jsonl.write_text("", encoding="utf-8")
    online_frames_jsonl.write_text("", encoding="utf-8")

    with open(driver_info_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "job_id": job_cfg["job_id"],
                "driver_model_index": int(job_cfg["driver_model_index"]),
                "driver_graph_path": str(Path(job_cfg["driver_graph_path"]).resolve()),
                "initial_structure_path": str(initial_structure_path),
                "temperature": float(job_cfg["temperature"]),
                "steps": int(job_cfg["steps"]),
                "timestep_fs": float(job_cfg["timestep_fs"]),
                "eval_every": int(md_cfg.get("eval_every", md_cfg.get("dump_every", 20))),
                "seed": int(job_cfg["job_seed"]),
                "novelty_threshold": float(novelty_context["novelty_threshold"]),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    feature_extractor = novelty_context["feature_extractor"]
    novelty_tracker = NoveltyTracker(ref_features=list(novelty_context["reference_features"]))
    monitor = CommitteeTrajectoryMonitor(
        hard_stop_cfg=hard_stop_cfg,
        save_policy_cfg=save_policy_cfg,
        early_stop_cfg=early_stop_cfg,
        novelty_threshold=float(novelty_context["novelty_threshold"]),
    )

    saved_count = 0
    status = "finished"
    stop_reason = "max_steps_reached"
    eval_every = int(md_cfg.get("eval_every", md_cfg.get("dump_every", 20)))
    dynamic_reference = bool(save_policy_cfg.get("dynamic_reference", True))

    for step in range(1, int(job_cfg["steps"]) + 1):
        try:
            dyn.run(1)
            energy, forces, calc_meta = get_consistent_energy_forces(atoms)
        except Exception as exc:
            status = "failed"
            stop_reason = repr(exc)
            append_jsonl(
                md_log_jsonl,
                {
                    "event": "error",
                    "step": int(step),
                    "job_id": job_cfg["job_id"],
                    "error": repr(exc),
                },
            )
            break

        max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(atoms) else 0.0
        if step % int(md_cfg.get("log_every", 10)) == 0 or step == 1:
            append_jsonl(
                md_log_jsonl,
                {
                    "event": "md_step",
                    "job_id": job_cfg["job_id"],
                    "step": int(step),
                    "time_fs": float(step * float(job_cfg["timestep_fs"])),
                    "temperature": float(atoms.get_temperature()),
                    "potential_energy": float(energy),
                    "max_force": max_force,
                    "selected_energy_key": calc_meta.get("selected_energy_key"),
                },
            )

        hard_stop, hard_reason, hard_metrics = monitor.check_hard_stop(atoms)
        if hard_stop:
            status = "hard_stop"
            stop_reason = str(hard_reason)
            append_jsonl(
                md_log_jsonl,
                {
                    "event": "hard_stop",
                    "job_id": job_cfg["job_id"],
                    "step": int(step),
                    "reason": stop_reason,
                    **hard_metrics,
                },
            )
            break

        if step % eval_every != 0:
            continue

        feat = feature_extractor.transform(atoms)
        novelty = novelty_tracker.nearest_distance(feat)
        save, reasons, eval_metrics = monitor.evaluate_online_save(
            atoms=atoms,
            feat=feat,
            novelty=novelty,
            step=step,
        )

        append_jsonl(
            online_frames_jsonl,
            {
                "job_id": job_cfg["job_id"],
                "step": int(step),
                "frame_index": int(saved_count if save else -1),
                "time_fs": float(step * float(job_cfg["timestep_fs"])),
                "novelty_score": float(eval_metrics["novelty_score"]),
                "rmsd_to_last_saved": float(eval_metrics["rmsd_to_last_saved"]),
                "rmsd_to_prev_eval": float(eval_metrics["rmsd_to_prev_eval"]),
                "geometry_changed": bool(eval_metrics["geometry_changed"]),
                "save": bool(save),
                "save_reason": ",".join(reasons),
            },
        )

        if save:
            snapshot = atoms.copy()
            snapshot.info["source_job_id"] = job_cfg["job_id"]
            snapshot.info["source_frame_index"] = saved_count
            snapshot.info["driver_model_index"] = int(job_cfg["driver_model_index"])
            snapshot.info["md_step"] = int(step)
            snapshot.info["save_reason"] = ",".join(reasons)
            snapshot.info["novelty_score"] = float(eval_metrics["novelty_score"])
            snapshot.info["rmsd_to_last_saved"] = float(eval_metrics["rmsd_to_last_saved"])
            snapshot.info["rmsd_to_prev_eval"] = float(eval_metrics["rmsd_to_prev_eval"])

            write(
                trajectory_xyz,
                snapshot,
                format="extxyz",
                append=trajectory_xyz.stat().st_size > 0,
            )
            write(
                saved_frames_xyz,
                snapshot,
                format="extxyz",
                append=saved_frames_xyz.stat().st_size > 0,
            )
            saved_count += 1
            if dynamic_reference:
                novelty_tracker.add_reference(feat)

        should_stop, early_reason, early_debug = monitor.should_stop(step)
        if should_stop:
            stop_reason = str(early_reason)
            append_jsonl(
                md_log_jsonl,
                {
                    "event": "stop_decision",
                    "job_id": job_cfg["job_id"],
                    "step": int(step),
                    "reason": stop_reason,
                    **early_debug,
                },
            )
            break

    return {
        "job_id": job_cfg["job_id"],
        "job_dir": str(job_dir.resolve()),
        "trajectory_xyz": str(trajectory_xyz.resolve()),
        "saved_frames_xyz": str(saved_frames_xyz.resolve()),
        "online_frames_jsonl": str(online_frames_jsonl.resolve()),
        "md_log_jsonl": str(md_log_jsonl.resolve()),
        "driver_info_json": str(driver_info_json.resolve()),
        "status": status,
        "stop_reason": stop_reason,
        "n_saved_frames": int(saved_count),
    }


