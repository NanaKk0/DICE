from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path

from workflow_app.composition import (
    BackendComposition,
    compose_executor_factory_roles,
    compose_selected_backend_roles,
    preflight_sampling_protocols,
)
from workflow_app.lifecycle_controller import (
    execute_public_lifecycle_suffix,
    suffix_uses_public_slurm,
)
from workflow_core.lifecycle.launch import (
    LifecycleLaunch,
    lifecycle_preflight_evidence,
    prepare_execution_block,
    prepare_execution_suffix,
    prepare_lifecycle_launch,
)
from workflow_core.lifecycle.runner import (
    archive_public_lifecycle_higher_round_evidence,
    execute_public_lifecycle,
    execute_public_lifecycle_block,
    probe_public_lifecycle_resume,
    reconcile_public_lifecycle_read_only,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_composition: BackendComposition | None = None,
    executor_factory: Callable[[Mapping[str, object]], object] | None = None,
    slurm_commands: object | None = None,
) -> object:
    """Run the sole public DICE lifecycle application."""

    parser = argparse.ArgumentParser(
        prog="dice",
        description="Run one DICE iter/round lifecycle.",
    )
    parser.add_argument("run_json", metavar="run.json")
    parser.add_argument("--environment", dest="environment_path")
    parser.add_argument("--execution", dest="execution_path")
    parser.add_argument(
        "--resume",
        metavar="BLOCK",
        help="rebuild one exact public block and its remaining suffix in the current iter",
    )
    parser.add_argument(
        "--truncate-after-round",
        type=int,
        metavar="N",
        help=(
            "archive higher-round evidence after configured round N before an "
            "explicit --resume fp"
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate selected operation environments without executing a block",
    )
    parser.add_argument(
        "--probe",
        metavar="BLOCK",
        help="check exact-resume feasibility without mutation",
    )
    parser.add_argument(
        "--scheduled-worker",
        metavar="BLOCK",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    worker_sentinel = os.environ.get("DICE_SCHEDULED_WORKER")
    if args.scheduled_worker is not None and worker_sentinel != "1":
        raise ValueError("scheduled-worker sentinel is missing or invalid")
    if args.scheduled_worker is None and worker_sentinel == "1":
        raise ValueError("scheduled-worker sentinel requires one public block coordinate")
    if args.truncate_after_round is not None and (
        args.resume != "fp" or args.preflight
    ):
        parser.error("--truncate-after-round requires an executing --resume fp")
    if args.probe is not None and (
        args.preflight
        or args.resume is not None
        or args.truncate_after_round is not None
    ):
        parser.error("--probe is mutually exclusive with --preflight and --resume")
    if args.scheduled_worker is not None and (
        args.preflight
        or args.probe is not None
        or args.resume is not None
        or args.truncate_after_round is not None
    ):
        parser.error(
            "--scheduled-worker is mutually exclusive with --preflight, --probe, and --resume"
        )
    if Path(args.run_json).suffix.lower() != ".json":
        parser.error("the lifecycle route requires a run.json file")
    if backend_composition is not None and executor_factory is not None:
        raise ValueError(
            "lifecycle execution accepts either a backend composition or an executor factory, not both"
        )
    launch = prepare_lifecycle_launch(
        args.run_json,
        environment_path=args.environment_path,
        execution_path=args.execution_path,
        preflight=False,
        resume=(
            args.scheduled_worker
            if args.scheduled_worker is not None
            else args.resume
        ),
    )
    if args.scheduled_worker is not None and launch.execution_path is not None:
        raise ValueError("scheduled worker refuses nested execution-profile submission")
    if args.preflight:
        launch = prepare_execution_suffix(
            launch,
            first_block=launch.plan.blocks[0].block_id,
            require_environment=True,
        )
        preflight_sampling_protocols(launch, first_block=launch.plan.blocks[0].block_id)
        print(
            json.dumps(
                lifecycle_preflight_evidence(launch),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return launch
    if args.probe is not None:
        observation = probe_public_lifecycle_resume(launch, args.probe)
        if observation.status == "runnable":
            prepared = prepare_execution_suffix(
                launch,
                first_block=args.probe,
                require_environment=launch.environment_path is not None,
            )
            preflight_sampling_protocols(prepared, first_block=args.probe)
        print(
            json.dumps(
                observation.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return observation

    if args.scheduled_worker is not None:
        first_block = args.scheduled_worker
    else:
        first_block = reconcile_public_lifecycle_read_only(launch)
    if first_block is None:
        return launch
    if args.scheduled_worker is None:
        launch = prepare_execution_suffix(
            launch,
            first_block=first_block,
            require_environment=(backend_composition is None),
        )
        preflight_sampling_protocols(launch, first_block=first_block)
        if args.truncate_after_round is not None:
            archive_public_lifecycle_higher_round_evidence(
                launch,
                after_round=args.truncate_after_round,
            )
    else:
        launch = prepare_execution_block(
            launch,
            block_id=args.scheduled_worker,
            require_environment=(backend_composition is None),
        )
        preflight_sampling_protocols(launch, first_block=first_block, single_block=True)

    composition = backend_composition

    def require_composition() -> BackendComposition:
        nonlocal composition
        if composition is None and executor_factory is not None:
            composition = compose_executor_factory_roles(launch, executor_factory)
        if composition is None and launch.environment_path is not None:
            if launch.preflight is None:
                raise ValueError(
                    "prepared lifecycle execution requires environment preflight"
                )
            composition = compose_selected_backend_roles(launch)
        if composition is None:
            raise AssertionError("unreachable lifecycle execution route")
        return composition

    if (
        args.scheduled_worker is None
        and suffix_uses_public_slurm(launch, first_block=first_block)
    ):
        execute_public_lifecycle_suffix(
            launch,
            first_block=first_block,
            backend_composition=backend_composition,
            composition_factory=require_composition,
            slurm_commands=slurm_commands,
        )
        return launch

    composition = require_composition()
    if args.scheduled_worker is None:
        execute_public_lifecycle(
            launch,
            backend_composition=composition,
        )
    else:
        execute_public_lifecycle_block(
            launch,
            block_id=args.scheduled_worker,
            backend_composition=composition,
        )
    return launch


def console_main() -> None:
    """Run the public CLI while preserving ``main``'s programmatic result."""

    main()


__all__ = ["console_main", "main"]
