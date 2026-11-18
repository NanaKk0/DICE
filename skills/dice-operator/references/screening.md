# Screening

Inspect the normalized route and the matching configuration reference before
changing thresholds or quotas. GMM and committee evidence use different
parameters. Preserve the requested method and each descriptor owner's model.

## Selection responsibilities

- Screening is student-first. A single student's `screen.prescreen` can be
  `GMM`, `JOINT_DIRECT` or `ELEMENT_JOINT_DIRECT`. Joint prescreens replace the
  student GMM prescreen, require positive `screen.pool`, and reject
  `screen.student_p` and safe fill.
- Joint student prescreening uses the current student training set as
  non-selectable background. When `pool >= N`, that prescreen can bypass
  descriptor inference and retain the surviving group.
- `screen.policy` selects the training addition after prescreening. Candidate
  selectors include `DIRECT`, `ELEMENT_DIRECT`, `FPS` and `random`.
- Teacher reliability separates usable teacher labels for student training from
  DFT candidates. Keep its bands and the candidate pool explicit; use the
  current guide's GMM probability/committee-deviation definitions.
- Terminal `fp.policy` selects DFT work. A single teacher can use
  `JOINT_DIRECT` or `ELEMENT_JOINT_DIRECT` with its training set as background;
  this joint selection still runs when `fp.keep >= N` and can underfill.

Joint methods require the corresponding single-model descriptor owner.
Candidate-only `ELEMENT_DIRECT` with a known committee owner resolves to
`random`; report requested/effective policy and
`committee_descriptor_unavailable`. Corrupt or missing single-model evidence
is an error, rather than a reason to switch selection methods.

Element-aware methods use per-element means, equal present-element weight and
presence-aware distances. They allow missing elements and retain supercell
replication invariance. Preserve the fixed representation; arbitrary weights or
committee descriptor aggregation are outside this interface.

## Groups, data and diagnostics

Quotas are resolved per active selection group. `fp.all_rn: false` selects from
the terminal round; `true` combines candidate rounds before terminal selection.
Inspect normalized task/group ownership when changing rounds or seed expansion.

Treat values in a tutorial as example inputs. Choose thresholds, pool sizes and
training/DFT quotas from the user's instruction or named reference, and discuss
new material proposals before execution. A screening split depends on the run;
report observed populations only when the user asks to inspect its results.

Selection plots reuse available scientific descriptors and are diagnostic.
Omitting `visualize` enables them; `false` disables them. Readout-free bypasses,
small populations and unavailable caches can skip plots. Their presence does
not decide block completion; plotting does not justify extra scientific
inference or a historical sample rebuild.
