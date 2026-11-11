# DPA4/DPA4C presets, ZBL and readout

This update supports the newer native DPA4/DPA4C recipe and embedding
interfaces. The public workflow remains `dice run.json`; native model settings
belong in the teacher/student recipe. See [native versions and CLI selectors](native-models.md).

## Native presets

A native model may declare `model.preset`, for example
`"dpa4-nano-v20260911"` or `"dpa4c-nano-v20260911"`. DICE recognizes the
family before selecting the operation roles and native backend. DeePMD owns
preset validation and architecture expansion. DICE supplies workflow-owned
datasets, native type maps and distinct member seeds; minimal preset recipes
also receive descriptor and fitting seeds without expanding the architecture
inside DICE.

## Analytical ZBL

The following is a model fragment for a compatible native recipe, not a
complete training input:

```json
{
  "model": {
    "preset": "dpa4c-nano-v20260911",
    "bridging_method": "zbl"
  }
}
```

Use the corresponding DPA4 preset for the teacher. DICE also recognizes the
canonical `linear_ener` composition of exactly one learned model and one
`inner_potential` with `weights: "sum"`. It routes member seeds and model
metadata to the learned component. General linear ensembles and legacy
tabulated `use_srtab` readout do not gain support through this change.

Fine-tune after enabling ZBL and preserve it when continuing training. Let
DeePMD own window parameters, recipe validation and short-distance data
filtering. Check interactions with `training_data.min_pair_dist` against the
selected native version. DICE labels with the full model and does not subtract
ZBL; do not add another ZBL contribution in the LAMMPS input.

## Readout and neighbour graphs

DICE uses the full model's official `eval_embedding` second return value,
`atomic_feature`: the per-atom hidden input to the final energy projection.
It retains dimensions connected to that projection and preserves atom order
and the active/native element mapping. Inspecting the learned child supplies
metadata only; it does not bypass the full composite forward or append
analytical-potential features.

Older models use `eval_fitting_last_layer` when `eval_embedding` is absent or
explicitly raises `NotImplementedError`. Other embedding errors propagate.
Neighbour graphs use `auto`; newer ordinary DPA4 no longer accepts the
previous explicit `dense` selector. Maintainer profiling of the newer plain
and ZBL readout paths observed graph execution without entering the legacy
padded-neighbour preparation path. This does not certify performance for
every model or system size.

## Native qualification and limits

Maintainer GPU checks used Nano OMat24 and a separate Nano OMol25
charge/spin compatibility test on the newer build listed in
[native-models.md](native-models.md). These checks are distinct from the
bundled CPU regression suite; weights, trajectories and native run logs are
not redistributed in this software update.

| Check | Observed scope |
| --- | --- |
| ZBL mechanism | Six YSZ element pairs, 39 distances per pair, both families in FP64; analytic inner-region forces, energy/force derivatives and window continuity passed |
| Fine-tuning | Plain OMat24/OMol25 foundation to ZBL, then continuation from a ZBL checkpoint, for DPA4 and DPA4C |
| Deployment | DPA4 freeze; DPA4C freeze/compress; normal-frame comparisons and ordinary/Kokkos LAMMPS |
| Readout | Full-model atomic features, atom order, element mapping and single/batch calls; DPA4 short-distance exceptions below |
| DICE lifecycle | Teacher-only/student-only ZBL short flows and both-ZBL YSZ iter0 with r0/r1, GMM/DIRECT, compressed Kokkos sampling, terminal validation and ordinary continuation |

The both-ZBL YSZ run used an OMat24 DPA4 teacher fine-tuned for 2500 steps
and DPA4C students fine-tuned for 5000 steps in each round. Eight NPT
trajectories each reached 5000 steps at 1 fs (40 ps in aggregate); the workflow
produced 2000 observations and four final FP selections. Neither round added
student training frames: r1 still used the original 15 bootstrap frames.
No new DFT calculation or DFT-return ingestion was exercised.

### DPA4 short-distance numerical consistency

Normal held-out YSZ configurations passed the export comparisons. The formal
teacher's maximum checkpoint-to-frozen force difference on those frames was
about `6.10e-5 eV/angstrom`. The tested DPA4C student checkpoints, frozen and
compressed models passed normal and dimer comparisons.

DPA4 retained a strict short-distance failure. At an isolated O-O separation
of `0.6678 angstrom` in the ZBL transition window, the formal teacher's
along-bond force on the second atom was approximately `-158219.97445` before
freezing and `-158210.21664 eV/angstrom` after freezing. The difference was
`9.7578125 eV/angstrom` (about `0.00617%`), above the test's component-wise
`atol=0.01`, `rtol=2e-5` criterion. This measures agreement between model
representations, not error against DFT. Both outputs at that point were
strongly attractive, so enabling ZBL alone does not establish physically
reliable behaviour throughout the full model's transition region.

A separate short-step OMat24 DPA4 model showed up to `0.03369` difference
between single-frame and batched atomic-feature components on short dimers;
normal frames passed. This is a feature difference, not a force unit. Changed
GMM/DIRECT decisions were not demonstrated. Precision controls support
execution/precision sensitivity, but the upstream root cause remains open.
The 15 bootstrap frames contained no bridging-transition contacts, so the
completed workflow does not qualify short-range collision accuracy.

These scans and representation/batch comparisons were standalone acceptance
tests. DICE does not automatically repeat them or block a workflow on their
tolerances. Normal runtime checks validate operation success, artifacts,
output shapes and finite values; a finite but physically poor prediction can
pass those checks. Qualify the intended contact distances before production
use involving collisions or strong compression.

### Charge/spin conditioning

OMol25's `add_chg_spin_ebd` supplies system charge and spin-multiplicity
conditioning, not per-atom magnetic-spin dynamics. Separate Python inference
checks covered `[0,1]`, `[0,3]` and `[1,2]`, including DPA4C compression
comparisons. LAMMPS tests used model-default conditioning; Kokkos does not
support runtime charge/spin overrides in the qualified build. No native
magnetic-spin dynamics qualification is claimed.
