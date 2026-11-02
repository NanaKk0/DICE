from __future__ import annotations

import unittest

from ase import Atoms


def _frame(
    identity: str,
    system: str,
    symbols: str,
    positions: list[list[float]],
    *,
    cell: list[float] | None = None,
    pbc: bool | tuple[bool, bool, bool] = False,
) -> Atoms:
    frame = Atoms(symbols, positions=positions, cell=cell, pbc=pbc)
    frame.info.update({"structure_id": identity, "system_id": system})
    return frame


class LifecyclePairGuardPolicyTests(unittest.TestCase):
    def test_global_and_pair_rules_use_strict_rejection_boundaries(self) -> None:
        from workflow_policies.pair_guard import PairGuardThresholds, evaluate_pair_guard

        frames = [
            _frame("global-reject", "water", "HH", [[0, 0, 0], [0.49, 0, 0]]),
            _frame("global-equal", "water", "HH", [[0, 0, 0], [0.5, 0, 0]]),
            _frame("pair-reject", "water", "HO", [[0, 0, 0], [0.84, 0, 0]]),
            _frame("pair-equal", "water", "HO", [[0, 0, 0], [0.85, 0, 0]]),
            _frame("unconfigured", "water", "HPt", [[0, 0, 0], [0.6, 0, 0]]),
        ]

        thresholds = PairGuardThresholds(minimum=0.5, pairs={"O-H": 0.85})
        decisions = evaluate_pair_guard(frames, thresholds)

        self.assertEqual([item.structure_id for item in decisions], [frame.info["structure_id"] for frame in frames])
        self.assertEqual([item.passed for item in decisions], [False, True, False, True, True])
        self.assertEqual(decisions[0].reject_rule, "min")
        self.assertEqual(decisions[2].reject_rule, "H-O")
        self.assertAlmostEqual(decisions[0].reject_distance, 0.49)
        self.assertAlmostEqual(decisions[2].reject_distance, 0.84)
        self.assertEqual(dict(thresholds.pairs), {"H-O": 0.85})

        global_only = evaluate_pair_guard(
            [_frame("global-only", "water", "HO", [[0, 0, 0], [0.49, 0, 0]])],
            PairGuardThresholds(minimum=0.5, pairs={}),
        )[0]
        self.assertEqual((global_only.passed, global_only.reject_rule), (False, "min"))
        equal_threshold = PairGuardThresholds(minimum=0.5, pairs={"H-O": 0.5})
        self.assertEqual(dict(equal_threshold.pairs), {"H-O": 0.5})
        equal_rule_frames = [
            _frame(
                "equal-rule-orthorhombic",
                "water",
                "HO",
                [[0, 0, 0], [0.49, 0, 0]],
                cell=[10, 10, 10],
                pbc=True,
            ),
            _frame(
                "equal-rule-triclinic",
                "water",
                "HO",
                [[0, 0, 0], [0.49, 0, 0]],
                cell=[[10, 0, 0], [1, 10, 0], [0, 0, 10]],
                pbc=True,
            ),
        ]
        auto_equal_rules = evaluate_pair_guard(equal_rule_frames, equal_threshold)
        ase_equal_rules = evaluate_pair_guard(
            equal_rule_frames, equal_threshold, algorithm="ase"
        )
        self.assertEqual(
            [item.reject_rule for item in auto_equal_rules],
            [item.reject_rule for item in ase_equal_rules],
        )
        with self.assertRaisesRegex(ValueError, "at least the minimum"):
            PairGuardThresholds(minimum=0.5, pairs={"H-O": 0.49})
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            PairGuardThresholds(minimum=0.5, pairs={"H-O": 0.85, "O-H": 0.9})

    def test_periodic_fast_path_and_ase_fallback_have_identical_decisions(self) -> None:
        from workflow_policies.pair_guard import PairGuardThresholds, evaluate_pair_guard

        orthorhombic = _frame(
            "periodic",
            "water",
            "HO",
            [[0.1, 0, 0], [9.3, 0, 0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        thresholds = PairGuardThresholds(minimum=0.5, pairs={"H-O": 0.85})

        comparison_frames = [
            orthorhombic,
            _frame("global", "water", "HH", [[0.1, 0, 0], [0.4, 0, 0]], cell=[10, 10, 10], pbc=True),
            _frame("equal", "water", "HO", [[0.1, 0, 0], [0.95, 0, 0]], cell=[10, 10, 10], pbc=True),
            _frame("pass", "mixed", "HPt", [[0.1, 0, 0], [0.7, 0, 0]], cell=[10, 10, 10], pbc=True),
        ]
        fast_decisions = evaluate_pair_guard(comparison_frames, thresholds, algorithm="ckdtree")
        fallback_decisions = evaluate_pair_guard(comparison_frames, thresholds, algorithm="ase")
        fast = fast_decisions[0]
        fallback = fallback_decisions[0]

        self.assertEqual(evaluate_pair_guard([orthorhombic], thresholds)[0].evaluation_path, "ckdtree")
        self.assertEqual(fast.evaluation_path, "ckdtree")
        self.assertEqual(fallback.evaluation_path, "ase")
        self.assertEqual((fast.passed, fast.reject_rule, fast.reject_pair), (fallback.passed, fallback.reject_rule, fallback.reject_pair))
        self.assertAlmostEqual(fast.reject_distance, fallback.reject_distance)
        for fast_item, fallback_item in zip(fast_decisions, fallback_decisions, strict=True):
            self.assertEqual(
                (fast_item.structure_id, fast_item.passed, fast_item.reject_rule, fast_item.reject_pair),
                (fallback_item.structure_id, fallback_item.passed, fallback_item.reject_rule, fallback_item.reject_pair),
            )
            if not fast_item.passed:
                self.assertAlmostEqual(fast_item.reject_distance, fallback_item.reject_distance)

        triclinic = orthorhombic.copy()
        triclinic.info.update(orthorhombic.info)
        triclinic.set_cell([[10.0, 0, 0], [1.0, 10.0, 0], [0, 0, 10.0]])
        mixed_pbc = orthorhombic.copy()
        mixed_pbc.info.update(orthorhombic.info)
        mixed_pbc.set_pbc((True, False, True))
        triclinic_fast = evaluate_pair_guard([triclinic], thresholds)[0]
        triclinic_reference = evaluate_pair_guard(
            [triclinic], thresholds, algorithm="ase"
        )[0]
        self.assertEqual(triclinic_fast.evaluation_path, "periodic")
        self.assertEqual(
            (
                triclinic_fast.passed,
                triclinic_fast.reject_rule,
                triclinic_fast.reject_pair,
                triclinic_fast.reject_atom_indices,
            ),
            (
                triclinic_reference.passed,
                triclinic_reference.reject_rule,
                triclinic_reference.reject_pair,
                triclinic_reference.reject_atom_indices,
            ),
        )
        self.assertAlmostEqual(
            triclinic_fast.reject_distance,
            triclinic_reference.reject_distance,
        )
        self.assertEqual(evaluate_pair_guard([mixed_pbc], thresholds)[0].evaluation_path, "ase")
        tiny_triclinic = _frame(
            "tiny-triclinic",
            "water",
            "HO",
            [[0.01, 0, 0], [0.12, 0, 0]],
            cell=[[0.2, 0, 0], [0.05, 0.2, 0], [0, 0, 0.2]],
            pbc=True,
        )
        self.assertEqual(
            evaluate_pair_guard([tiny_triclinic], thresholds)[0].evaluation_path,
            "ase",
        )

    def test_worst_violation_ties_use_pair_then_atom_order(self) -> None:
        from workflow_policies.pair_guard import PairGuardThresholds, evaluate_pair_guard

        tied = _frame(
            "tied",
            "mixed",
            "HOHOOO",
            [[0, 0, 0], [0.5, 0, 0], [5, 0, 0], [5.5, 0, 0], [10, 0, 0], [11, 0, 0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        thresholds = PairGuardThresholds(minimum=0.25, pairs={"O-H": 1.0, "O-O": 1.5})

        fast = evaluate_pair_guard([tied], thresholds, algorithm="ckdtree")[0]
        fallback = evaluate_pair_guard([tied], thresholds, algorithm="ase")[0]

        self.assertEqual((fast.reject_rule, fast.reject_pair, fast.reject_atom_indices), ("H-O", "H-O", (0, 1)))
        self.assertEqual((fallback.reject_rule, fallback.reject_pair, fallback.reject_atom_indices), ("H-O", "H-O", (0, 1)))
        self.assertAlmostEqual(fast.reject_distance, 0.5)
        self.assertAlmostEqual(fallback.reject_distance, 0.5)

        triclinic = tied.copy()
        triclinic.info.update(tied.info)
        triclinic.set_cell([[20.0, 0, 0], [1.0, 20.0, 0], [0, 0, 20.0]])
        periodic = evaluate_pair_guard([triclinic], thresholds)[0]
        triclinic_fallback = evaluate_pair_guard(
            [triclinic], thresholds, algorithm="ase"
        )[0]
        self.assertEqual(periodic.evaluation_path, "periodic")
        self.assertEqual(
            (
                periodic.reject_rule,
                periodic.reject_pair,
                periodic.reject_atom_indices,
            ),
            (
                triclinic_fallback.reject_rule,
                triclinic_fallback.reject_pair,
                triclinic_fallback.reject_atom_indices,
            ),
        )
        self.assertAlmostEqual(
            periodic.reject_distance,
            triclinic_fallback.reject_distance,
        )


if __name__ == "__main__":
    unittest.main()
