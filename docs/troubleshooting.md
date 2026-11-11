# Troubleshoot a run

First identify the failed public block in `iterN/status.log`, then inspect its
backend logs. Keep the run directory, source commit, native versions and the
exact command together when diagnosing a problem. Use [recovery](recovery.md)
for rebuilds rather than deleting state files.

| Symptom | Check | Next action |
| --- | --- | --- |
| `dice` absent or imports from another checkout | Active Python, `which dice`, installed module path | Activate/reinstall the intended control package |
| Input/recipe path rejected | Path relative to run.json, traversal/globs, missing regular file | Correct the declared run-local path; do not move outputs into the source tree |
| Profile role missing | Selected teacher/student shape and profile bindings | Use the matching PT/TF/NEP profile and bind all selected roles |
| Native executable or shared library missing | Activation script, PATH, native stdout/stderr, matching libraries | Repair the native environment; do not infer it from control CLI success |
| Foundation element-map mismatch | Full foundation symbol order versus active chemistry and recipe child maps | Preserve native map; correct active coverage separately |
| New `.pt2` cannot load in older LAMMPS | Exporter, C++ plugin and model schema/build versions | Use matching runtime or supported re-export; changing the suffix cannot convert a model |
| DPA4 compression requested | Actual model family | Use supported DPA4 freeze; compression is a DPA4C student capability |
| Kokkos fails with one DPA4C student | `deepmd/kk`, Kokkos/CUDA build, model/runtime compatibility | Qualify the selected Kokkos route; failure is not an ordinary fallback |
| Labels lack energy/forces/stress | ASE results, tensor units and native recipe contract | Correct the external export and retain geometry/force correspondence |
| No new student data | Student candidate counts, teacher trust rejection, reference size, quotas | Identify the gate or shortage before adjusting scientific thresholds |
| FP selection empty | Active groups, candidate bands, selected-round scope, eligible counts | Treat a valid empty endpoint as an outcome; review exploration/coverage |
| GMM p values too coarse | Number of calibration reference frames | Compare thresholds with finite tail-rank resolution; do not interpret p as accuracy probability |
| Ordinary run skips an edited training recipe | Existing valid terminal result and its dependency closure | Inspect a precise probe and deliberately rebuild the intended suffix |
| MD unstable at very close contacts | Short-distance coverage, timestep, E/F, export parity, complete-model ZBL transition | Test the physical regime separately; workflow completion cannot certify collision accuracy |
| Optional plots absent | Population size, diagnostic warnings and available readout | Check scientific outputs first; plots are not completion authority |

## Report a reproducible issue

Provide a minimal configuration, failed block, redacted environment/execution
profiles, exact software/native versions, model family and relevant log excerpt.
State whether it reproduces with the CPU quickstart or only a native engine.
Report in the [issue tracker](https://github.com/NanaKk0/DICE/issues).
Share model/data identifiers or small releasable fixtures rather than credentials
or private research assets. See [capabilities](support.md) for qualification scope.
