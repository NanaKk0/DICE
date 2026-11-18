# Configuration screening

DICE evaluates sampled configurations and selects teacher-labelled student-training additions and candidates for DFT reference calculations. Representative selection controls the retained population and coverage.

This page focuses on a single teacher/student. See [Workflow](workflow.md) for data destinations.

## Evaluation evidence

Each role uses evidence appropriate to its model shape:

| Model shape | Evidence | Score interpretation |
| --- | --- | --- |
| Single model | Atomic descriptors and a Gaussian mixture model (GMM) | Smaller `p` means less support relative to the model's current training data |
| Committee | Differences in members' force predictions | Larger deviation means greater disagreement, in eV/Å |

Descriptors represent atomic chemical environments. DICE builds reference distributions from training descriptors and evaluates sampled configurations against them.

The teacher reference is its current iter's training data; the student reference is its current round's data. A configuration can be familiar to the teacher and unfamiliar to the student.

## GMM scores

1. **Fit reference distributions.** Extract training atomic descriptors and fit GMMs by element.
2. **Evaluate atomic environments.** Calculate atomic negative log likelihood and calibrate against reference atoms of the same element to obtain empirical atomic tail scores.
3. **Aggregate local deviation.** Transform atomic scores with `-log(atom_p)` and average the highest-deviation 5% of atoms in a frame, rounding up to at least one atom.
4. **Score the frame.** Compare its aggregate with the reference-frame aggregates to obtain `p`.

`p` is the smoothed empirical fraction of reference frames whose deviation reaches or exceeds the candidate's. It lies in `(0, 1]`; smaller values indicate a rarer configuration relative to the reference.

With `N` reference frames, the score floor is `1/(N+1)`. For example, 99 reference frames give a floor of 0.01. Interpret threshold resolution with the reference population in mind.

After a model/data update, evaluation uses a reference matching the current model and training data.

## Screening order

For a single teacher with student GMM prescreening:

1. Read and deduplicate observations, applying configured pair-distance guards.
2. Form a student candidate pool from the score band and pool capacity.
3. Evaluate the pool with the teacher, applying teacher-label trust and DFT-candidate conditions separately.
4. Select and publish teacher-labelled training additions and retain round DFT candidates.
5. After the student rounds, aggregate the configured candidate rounds and select the final DFT configurations.

Optional safe fill can supplement training additions from the student's covered region when the requested training quota is underfilled. Supplementary configurations still pass teacher-trust checks.

A single student can instead prescreen with `JOINT_DIRECT`: sampled configurations and the current student training data share a representation space for candidate selection, followed by teacher assessment.

## Overlap between the two routes

Teacher assessment has two independently configured conditions:

- **Teacher-label trust:** whether the prediction can augment student training.
- **DFT candidacy:** whether the configuration enters the reference-calculation candidate band.

| Teacher-trusted | DFT candidate | Eligible selection |
| --- | --- | --- |
| Yes | Yes | Both routes |
| Yes | No | Student-training additions |
| No | Yes | DFT candidates |
| No | No | Neither route |

When bands overlap, the same geometry can receive a teacher label for student training and later a DFT label for teacher updating. Pool limits and representative selection determine retained membership.

Committees apply force-deviation thresholds. Mixed single/committee routes retain each role's own evidence. See [run.json reference](../reference/run-json.md) for legal combinations and boundaries.

## Representative selection

Selection retains a bounded set from eligible, potentially similar configurations:

| Method | Idea |
| --- | --- |
| `random` | Random selection from eligible candidates |
| `FPS` | Repeatedly choose configurations far from the selected set in representation space |
| `DIRECT` | Transform, reduce and cluster representations, choose cluster representatives and fill remaining capacity by distance selection |
| `ELEMENT_DIRECT` | Use element-organized representations with balanced weight among present elements |
| `JOINT_DIRECT`, `ELEMENT_JOINT_DIRECT` | Include the descriptor owner's current training data as background and choose candidate additions by joint coverage |

For repeated local fluctuations, DIRECT organizes selection using representation similarity; FPS progressively extends coverage by distance. Applicability and model requirements are in [Screening fields](../reference/run-json.md#screening-fields-and-terminal-selection).

## Quotas and groups

Four main capacities govern the flow: student prescreen pool, per-round training additions, per-round DFT candidate pool and terminal DFT selection. They apply per selection group.

Seeds and conditions sharing a group share its quota. For example, two temperatures in one group with a four-frame training quota jointly supply up to four additions.

Eligible populations, deduplication and capacities determine the actual selection. A group aggregates its tasks before applying pool and selection limits.

See [Screening fields](../reference/run-json.md#screening-fields-and-terminal-selection), [sampling groups](../guides/sampling.md#7-connect-sampling-to-screening) and [Read screening reports](../guides/results-and-recovery.md#2-read-screening-reports).
