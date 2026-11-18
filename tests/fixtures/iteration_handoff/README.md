# Iteration-handoff test fixture

This developer fixture exercises reference-label ingestion and student-data
carry-over from iter0 to iter1 using synthetic backend services.

From the installed checkout:

```sh
python tests/fixtures/iteration_handoff/check.py demo-fp-handoff
```

Use a new directory. The script runs both iterations and checks that returned
geometry and labels enter teacher training, prior student geometries carry
over, and required labels remain readable. It uses the services from
[cpu_mock](../cpu_mock/README.md).

The retained configurations and `handoff_check.json` describe the check.
Testing instructions are in [Development](../../../docs/development/contributing.md).
For reference calculations and the next scientific iteration, follow the
[water example](../../../examples/dpa4_dpa4c/README.md).
