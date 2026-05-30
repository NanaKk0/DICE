# DICE

`DICE (Distillation-driven Co-Evolution)` is a machine-learning force-field
workflow for teacher-guided, student-driven active learning. The current
workflow centers on teacher committee training, student active-learning
sampling, and external first-principles handoff.

This README is a human-facing project homepage. It is intentionally brief; the
durable workflow language, contracts, backend execution details, and agent
operating rules live in the documents below.

## Current Shape

The current top-level workflow concepts are:

- `teacher_foundation_train`
- `committee_cycle`
- `first_principles`

`committee_cycle` is the core mainline. It contains teacher committee finetune
and the student subworkflow. First-principles calculation is currently more of
an external handoff path and usually depends on user-managed result backfill.

## Documentation

- Agent operating rules: [`AGENTS.md`](./AGENTS.md)
- Workflow quick entrypoint: [`docs/agent/WORKFLOW_OVERVIEW.md`](./docs/agent/WORKFLOW_OVERVIEW.md)
- Engineering contracts: [`docs/agent/contracts/README.md`](./docs/agent/contracts/README.md)
- Backend execution references: [`docs/agent/backends/README.md`](./docs/agent/backends/README.md)
- Architecture decisions: [`docs/agent/adr/`](./docs/agent/adr/)
- Task and PRD tracking rules: [`docs/agent/task-tracker.md`](./docs/agent/task-tracker.md)
- Environment requirements: [`docs/user/ENVIRONMENT_REQUIREMENTS.md`](./docs/user/ENVIRONMENT_REQUIREMENTS.md)

## Repository Layout

- `configs/`: workflow configuration examples and shipped configs
- `workflow_core/`: workflow orchestration and domain services
- `workflow_runtime/`: runtime paths, logs, markers, resume, and reconcile infrastructure
- `workflow_backends/`: backend-owned numerical execution
- `workflow_policies/`: selection, routing, and screening policies
- `workflow_artifacts/`: structured artifact and report helpers
- `docs/`: durable documentation modules
- `docs/agent/tasks/`: PRDs, issues, plans, and task-local working notes
