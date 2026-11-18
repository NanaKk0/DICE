---
name: dice-operator
description: Prepare, check, run, and resume DICE iter/round workflows using run.json and environment/execution profiles. Use for DICE configuration, sampling, screening, progress, and recovery; source development and standalone simulation jobs use their own workflows.
---

# Operate DICE

Help the user operate the public `dice run.json` workflow. Follow the current
workspace instructions and preserve the user's scientific and resource choices.

## Locate the operation

- Identify the requested task and `RUN_ROOT`, the directory containing the
  selected run configuration. Its inputs, recipes, logs and `iterN/` outputs
  belong there. A DICE source checkout and this skill can live elsewhere.
- Use the Python environment containing DICE and its matching `dice` command.
  Use user-supplied environment/execution profiles for backend activation and
  placement. Ask for a missing location only when the task depends on it.
- Resolve bundled references and scripts relative to this `SKILL.md`.
  This package uses ordinary file access and a terminal; it requires no other
  skills, agent-specific tools, hooks or service credentials.

## Read for the task

| Task | Read |
| --- | --- |
| Create/edit a run, profiles, model recipes or the next iter | [Configuration](references/configuration.md) |
| Native sampling, explicit overrides, batch conditions or advanced modes | [Sampling](references/sampling.md) |
| GMM, committees, DIRECT or selection quotas | [Screening](references/screening.md) |
| Preflight, run, observe, probe, resume or change round count | [Lifecycle](references/lifecycle.md) |

Read only the relevant references. For exact fields and backend support, use
the matching DICE version's documentation index (`docs/index.md` in a checkout),
configuration reference and official loaders. The public repository is
<https://github.com/NanaKk0/DICE>. Keep a reference example's particular model,
thresholds and step budgets scoped to that example.

## Prepare and validate

1. Read the user's configuration or named reference. Put scientific settings
   in `run.json`, software bindings in an environment profile, resources in an
   execution profile, and native algorithms in their backend recipe.
2. Write a complete valid candidate when creating a configuration. Preserve
   declared relative paths and actual input files. For a new iter, create a
   separate configuration and keep the previous iter's file.
3. Use bundled `scripts/inspect_dice_operation.py` for read-only structural
   inspection. Resolve its absolute path and invoke it with DICE's Python,
   the run configuration, and any selected `--environment` / `--execution`
   profiles. It uses the DICE importable in that Python environment;
   `--repo-root PATH` selects an explicit checkout. `--resume BLOCK` lists
   that suffix and `--pretty` formats the JSON.
4. Repair structural errors before handing off the candidate. Inspect the
   normalized settings, route, blocks and roles. The helper's `not_checked`
   list identifies checks left to public preflight; it never runs engines.
5. Explain material choices and their source: user instruction, named
   reference, official default or agent proposal. Reuse existing authorization.
   Ask only about new unresolved proposals affecting science, inputs, cost or
   placement, after preparing the reviewable candidate and structural checks.

## Execute and report

- A request to create or explain a configuration authorizes that task. A request
  to run authorizes preflight and execution once material choices are settled.
  Use public `--preflight` before ordinary execution or an exact rebuild.
- Preserve an explicitly selected `--resume BLOCK`. If an earlier dependency
  blocks it, report the recommended producer; obtain authorization before
  expanding the rebuild scope. Use DICE's continuation/recovery commands rather
  than manually removing lifecycle state or results.
- Inspect logs for progress and the published result for completion.
  `--probe BLOCK` checks rebuild feasibility. Report task failures, pending work
  and usable published results distinctly; plots are diagnostic.
- Keep responses proportional: answer a field question directly; for changes or
  execution, report the files/important settings, checks, outcome and next step.
  Include pending questions only when there is an unresolved decision.

## Boundary

This skill prepares configurations and operates DICE. Source changes, Git,
backend installation, standalone DeePMD/LAMMPS jobs and external DFT services
follow their own requested workflow. When an operation fails, retain the exact
command, block and relevant error/log path for diagnosis. Inspect only the
requested run and relevant documentation; large trajectories are read through
DICE's existing result interfaces.
