# DICE Operator skill

Prepare configurations, check inputs, run DICE iterations and recover selected
stages with a local coding agent. The package follows the
[Agent Skills format](https://agentskills.io/specification) used by Codex,
Claude Code and other compatible agents.

## Install with your agent

Tell your agent:

> Install the `dice-operator` skill from
> `https://github.com/NanaKk0/DICE/tree/main/skills/dice-operator` for my current
> agent. Keep the complete folder, including references and scripts, and confirm
> that the skill is discoverable.

From a downloaded checkout or archive, give the agent the local
`skills/dice-operator` folder instead. A complete copy is sufficient; the
package uses standard skill metadata and ordinary terminal/file access.

| Agent | User installation | Project installation |
| --- | --- | --- |
| Codex | `~/.agents/skills/dice-operator/` | `.agents/skills/dice-operator/` |
| Claude Code | `~/.claude/skills/dice-operator/` | `.claude/skills/dice-operator/` |

Use the discovery directory configured by your agent if it differs. Existing
Codex installations may use a configured `CODEX_HOME` skills directory.
Keep one active installation for a given scope. See the official
[Codex](https://developers.openai.com/codex/skills) and
[Claude Code](https://code.claude.com/docs/en/skills) installation guidance.

## Use

For example:

> Use dice-operator to inspect this `run.json` and its environment profile.
> Explain any missing inputs before running calculations.

> Use dice-operator to prepare the next iter with my returned DFT labels.
> Keep the previous configuration and inherit its execution settings.

Explicit invocation is `$dice-operator` in Codex and `/dice-operator` in
Claude Code. The same instructions also support automatic selection.

Install DICE in the control Python environment using its installation guide,
then prepare the backend environments needed by your route. The bundled
inspection helper uses the DICE visible to the chosen Python interpreter.
An explicit `--repo-root PATH` selects a checkout. The skill and run directory
can be moved independently of that installation.

The helper's `--help` lists its arguments. It checks configuration structure
through DICE's loaders; public `dice --preflight` checks environment, recipe
and execution feasibility. DICE documentation and the full water example are
available from the [repository documentation index](https://github.com/NanaKk0/DICE/blob/main/docs/index.md).

Python code uses the [MIT licence](LICENSE). Skill instructions and documentation
use [CC BY 4.0](LICENSE-DATA.md), following DICE's licensing terms.
