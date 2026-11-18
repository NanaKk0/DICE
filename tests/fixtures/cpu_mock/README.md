# CPU lifecycle test fixture

This developer fixture runs DICE with synthetic teacher, student and sampler
services. It exercises lifecycle coordination, ordinary continuation and
stage rebuilding. See [Development](../../../docs/development/contributing.md) for testing.

From the installed checkout:

```sh
python tests/fixtures/cpu_mock/run.py demo-cpu-lifecycle
python tests/fixtures/cpu_mock/run.py demo-cpu-lifecycle --reuse
python tests/fixtures/cpu_mock/run.py demo-cpu-lifecycle --resume r1/sample
```

Use a new directory for the first command. Inspect its `run.json`,
`iter0/status.log`, `iter0/dice.log` and round outputs to follow the workflow.
`fixtures.py` supplies the inputs and `backend_roles.py` supplies the synthetic
services.

For a workflow with native engines, use the
[water example](../../../examples/dpa4_dpa4c/README.md).
