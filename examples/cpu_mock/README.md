# CPU lifecycle example

This example runs the actual application lifecycle using synthetic teacher,
student and sampler services. All models, labels and trajectories are fabricated;
no native engine is required. Follow the [CPU quickstart](../../docs/quickstart.md)
for a guided explanation.

From the installed checkout, with the control environment activated:

```sh
python examples/cpu_mock/run.py /tmp/dice-mock-demo
python examples/cpu_mock/run.py /tmp/dice-mock-demo --reuse
python examples/cpu_mock/run.py /tmp/dice-mock-demo --resume r1/sample
```

Use a new or empty directory for the first command. The script writes a complete
`run.json`, geometry/label fixtures and recipe-shaped inputs. The second command
reuses completed outputs; the third deliberately rebuilds the requested suffix.
The generated directory has a marker so reuse cannot silently adopt another run.

The fixture uses an H-only population named `water`, a single MACE-shaped
teacher, a single DeepMD-TF-shaped student, two 20-step LAMMPS-shaped rounds
at 300/500 K, zero validation fraction and small random-selection quotas.
These are synthetic demonstration choices. `fixtures.py` and `backend_roles.py`
provide the services used by the example; the installed application coordinates them.

Success prints the terminal `iter0/fp/selected.xyz` path. A valid selection may
be empty. Inspect `run.json`, `iter0/status.log` and the round directories to
follow the execution. Optional plots may skip tiny populations.

For real calculations, prepare [your own inputs and native environments](../../docs/own-system.md).
