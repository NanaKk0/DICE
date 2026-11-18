# DICE

DICE (Distillation-driven Co-Evolution) combines active learning and knowledge distillation for interatomic potential development. A student explores new configurations, and DICE evaluates the reliability of the teacher's labels. Reliable teacher labels expand student training data; configurations requiring DFT are selected for reference calculations, whose labels update the teacher in the next iter.

## Get started

| Read | Purpose |
| --- | --- |
| [Introduction to DICE](docs/getting-started/introduction.md) | Workflow and preparation checklist |
| [Installation](docs/getting-started/installation.md) | Install the control package and check the CLI |
| [Quickstart](docs/getting-started/quickstart.md) | Run the water example through the first iter, DFT-label return and the next iter |

Use Linux and Python 3.10 or newer. From the source checkout:

```sh
git clone https://github.com/NanaKk0/DICE.git
cd DICE
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
dice --help
```

Follow the [Quickstart](docs/getting-started/quickstart.md) directly in the supplied example directory. See [Backend environments](docs/guides/backends.md) and [Compute resources](docs/guides/resources.md) for installation-specific settings.

## Documentation and support

- [Documentation index](docs/index.md): guides, concepts, references and development.
- [Inspect results and resume](docs/guides/results-and-recovery.md): outputs, continuation and rebuilds.
- [Troubleshooting](docs/guides/troubleshooting.md): configuration, data, environment and task errors.
- [CLI reference](docs/reference/cli.md): preflight, execution, recovery and round changes.
- [DICE Operator skill](skills/dice-operator/README.md): install reusable guidance for Codex, Claude Code and compatible agents.
- [GitHub Issues](https://github.com/NanaKk0/DICE/issues): problems and suggestions.

## Development and licensing

See [Contributing](docs/development/contributing.md) for development installation, tests and contributions, and [CHANGELOG](CHANGELOG.md) for changes.

Code is MIT; author-owned documentation, figures and data use CC BY 4.0. Third-party materials retain their licences. See [Licensing and attribution](LICENSING.md).
