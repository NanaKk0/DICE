# Choose screening and representative selection

Screening answers two different questions: which teacher-labelled structures
can train the student, and which structures deserve new FP labels? Representative
selection then controls redundancy and cost within eligible populations.
Use the exact [field reference](configuration.md) when setting bands and quotas.

## Follow a candidate

A round starts with observations from its sampling tasks. The student route
proposes structures, after which the teacher evaluates its eligibility rules.
The workflow can publish trusted student-training additions and a separate
teacher FP-candidate reserve. The terminal `fp` block selects a limited number
of candidates from the configured rounds.

A structure's eligibility is not an accuracy certificate. Teacher predictions
are still model labels; a new FP calculation supplies the independent reference.

## Single-model support scores

For supported single models, DICE fits a GMM reference using that model's
**current train membership**. Atomic negative log likelihood (NLL) gives an
outlyingness statistic. The frame statistic averages the largest
`ceil(0.05 × atom_count)` atomic NLL values. Its empirical reference-tail rank is:

```text
p(query) = (number of reference statistics >= query statistic + 1)
           / (number of reference statistics + 1)
```

A smaller p means more surprising relative to that reference. It is not a
calibrated probability of correct forces. Finite reference size limits the
resolution; the minimum attainable empirical p is `1/(N+1)`. Model/reference
changes require the matching workflow-owned fitted reference.

## Teacher and student bands

| Setting | Decision for a single-model GMM route |
| --- | --- |
| Teacher floor | Below it, reject from both teacher views |
| Teacher trust | Student labels require `p > trust` as well as the floor rule |
| Teacher ceiling | FP reserve uses `[floor, ceiling)`; omission includes p=1 |
| Student floor/ceiling | Candidate proposals use the configured band; a ceiling creates a covered region above it |

For teacher floor 0, trust 0.1 and ceiling 0.3:

| Teacher p | Student-label trust | In FP-reservation band |
| --- | --- | --- |
| 0.05 | No | Yes |
| 0.10 | No: trust equality is rejected | Yes |
| 0.20 | Yes | Yes |
| 0.30 | Yes | No: ceiling equality is excluded |

These are eligibility decisions before representative quotas. The two teacher
views can overlap. For a student ceiling of 0.3, candidates are below 0.3 and
the region at or above 0.3 is covered/safe. Safe filling can fill a proposal
shortfall from that covered region after other gates; it cannot resurrect
teacher-rejected structures or below-floor frames. With no student ceiling
there is no such covered region.

## Committees

Committees use force-deviation evidence in **eV/Å**, not p values. Student
candidate deviations lie in `[lo, hi)`; below-lo structures may supply safe
filling, while high deviations are rejected. Teacher trust and FP reservation
use their corresponding deviation rules. Train independent members; correlated
members can understate uncertainty. Model-shape and backend constraints are
listed in [capabilities](support.md).

## Quotas and grouping

| Quota | Capacity it controls, per active selection group |
| --- | --- |
| `screen.pool` | Student proposal pool |
| `screen.keep` | Student training representatives |
| `screen.fp_pool` | Single-teacher FP-candidate reserve |
| `fp.keep` | Final representatives handed to FP calculations |

These are upper bounds on eligible frames. Multiple seed roles sharing a
`selection_group` share its quota, even when many sampling conditions execute.
`fp.all_rn: false` uses the final round; true merges round candidate pools,
with later duplicate-frame precedence, before one terminal group selection.

## Select representatives

| Policy | Use and boundary |
| --- | --- |
| `random` | Random representatives after route eligibility |
| `FPS` | Farthest-point representatives in the model-derived structure representation |
| `DIRECT` | Representation-based coverage selection on eligible candidates |
| `ELEMENT_DIRECT` | Candidate-only element-balanced structure representation |
| `JOINT_DIRECT` / `ELEMENT_JOINT_DIRECT` | Joint selection with the descriptor owner's exact current training set as non-selectable background |

Joint student prescreens replace GMM, require positive pool capacities, forbid
student p-bands and safe fill. Joint terminal selectors require a single teacher.
Background-selected capacity is not refilled with candidates. ELEMENT policies
give chemically present elements equal weight; absent elements remain absent.

A known committee owner has a documented random fallback only for candidate-only
`ELEMENT_DIRECT`, recorded in `policy_resolution.json`. It is not a fallback
for corrupt or missing single-model descriptors, or for joint element selection.
See [configuration](configuration.md) for all applicability and boundary rules.

Inspect actual screened counts, trust rejection, quotas and fitted-reference
sizes in `dice.log`. Empty outputs can be legitimate; first check which gate
or shortage caused them before changing thresholds.
