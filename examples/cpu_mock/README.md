# CPU-only lifecycle demonstration

Run from the distribution root after installation:

```sh
python examples/cpu_mock/run.py /tmp/dice-mock-demo
python examples/cpu_mock/run.py /tmp/dice-mock-demo --reuse
python examples/cpu_mock/run.py /tmp/dice-mock-demo --resume r1/sample
```

Use a new or empty run directory for the first invocation. The example
generates a complete `run.json`, bootstrap extxyz and recipe-shaped fixtures.
All paths inside `run.json` are relative to that file. It exercises two
rounds, canonical dataset publication, GMM routing, sampling replay, screen
selection, terminal publication and block-level recovery.

`fixtures.py` copies only the relevant definitions from the official
`tests/test_lifecycle_executor.py` fixture, and `backend_roles.py` copies its
role-composition helper. They inject synthetic services into the real application.

Scientific values are inherited from that fixture: iteration 0, H-only
synthetic structures, single MACE-shaped teacher, single DeepMD-TF-shaped
student, zero validation fraction, two 20-step LAMMPS-shaped rounds at
300/500 K, random screening with pool/keep 1, and FP keep 1. The fixture's
legacy system name `water` identifies the H-only test population.
Omitted settings use the official DICE loader defaults.
Visualization is default-on, non-authoritative and may skip tiny populations.

The fake teacher writes text model handles and zero energy/force labels;
the fake student writes text handles and fabricated descriptors; the fake
sampler writes a short synthetic trajectory. Temperatures and step budgets
are fixture metadata. A valid terminal selection can be empty.

For real calculations, replace the fixture with your validated labeled data,
native recipes and installed engines; write the matching environment profile;
then use `dice run.json --environment environment.json` in a separate scientific
run directory. Keep the mock demonstration in its own directory.
