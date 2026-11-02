from __future__ import annotations

import unittest
from unittest import mock

import numpy as np


class LifecyclePoliciesTests(unittest.TestCase):
    def test_element_embedding_is_replication_invariant_and_type_ordered(self) -> None:
        from workflow_policies.selection import prepare_element_frame_vectors

        embedding = prepare_element_frame_vectors(
            [
                {
                    "Al": np.array([[2.0, 4.0], [4.0, 4.0]]),
                    "O": np.array([[0.0, 1.0], [0.0, 3.0]]),
                },
                {
                    "O": np.array(
                        [[0.0, 3.0], [0.0, 1.0], [0.0, 3.0], [0.0, 1.0]]
                    ),
                    "Al": np.array(
                        [[4.0, 4.0], [2.0, 4.0], [4.0, 4.0], [2.0, 4.0]]
                    ),
                },
            ],
            presence=[("O", "Al"), ("Al", "O")],
            type_map=("Al", "O"),
        )

        self.assertEqual(embedding.active_elements, ("Al", "O"))
        self.assertEqual(embedding.inactive_elements, ())
        np.testing.assert_allclose(embedding.vectors[0], embedding.vectors[1])
        np.testing.assert_allclose(
            embedding.element_means["Al"],
            np.array([[3.0, 4.0], [3.0, 4.0]]),
        )
        np.testing.assert_allclose(
            embedding.element_means["O"],
            np.array([[0.0, 2.0], [0.0, 2.0]]),
        )

    def test_element_direct_selects_exact_quota_without_whole_frame_preprocessing(self) -> None:
        import workflow_policies.selection as selection

        rows = [
            {"Al": np.array([[1.0, 0.0]]), "O": np.array([[0.0, 1.0]])},
            {"Al": np.array([[0.9, 0.1]]), "O": np.array([[0.1, 0.9]])},
            {"Al": np.array([[0.0, 1.0]]), "O": np.array([[1.0, 0.0]])},
        ]
        with mock.patch.object(
            selection,
            "_preprocess_frame_vectors",
            side_effect=AssertionError("whole-frame preprocessing is forbidden"),
        ):
            result = selection.select_element_representatives(
                ["first", "second", "third"],
                rows,
                presence=[("Al", "O")] * 3,
                type_map=("Al", "O"),
                keep=2,
            )

        self.assertEqual(len(result.selected_identities), 2)
        self.assertEqual(len(set(result.selected_identities)), 2)
        self.assertTrue(set(result.selected_identities).issubset({"first", "second", "third"}))
        self.assertEqual(result.audit["selected_candidate_count"], 2)
        self.assertEqual(result.audit["background_count"], 0)

    def test_element_joint_direct_uses_nonselectable_background_and_exact_candidate_quota(self) -> None:
        import workflow_policies.selection as selection

        candidates = [
            {"Al": np.array([[1.0, 0.0]])},
            {"Al": np.array([[0.8, 0.2]])},
            {"Al": np.array([[0.0, 1.0]])},
        ]
        background = [{"Al": np.array([[0.9, 0.1]])}]
        with mock.patch.object(
            selection,
            "_preprocess_frame_vectors",
            side_effect=AssertionError("whole-frame preprocessing is forbidden"),
        ):
            result = selection.select_element_joint_representatives(
                ["shared", "candidate-b", "candidate-c"],
                candidates,
                candidate_presence=[("Al",)] * 3,
                background_identities=["shared"],
                background_element_rows=background,
                background_presence=[("Al",)],
                type_map=("Al",),
                keep=2,
            )

        self.assertEqual(len(result.selected_identities), 2)
        self.assertTrue(
            set(result.selected_identities).issubset(
                {"shared", "candidate-b", "candidate-c"}
            )
        )
        self.assertEqual(result.audit["candidate_count"], 3)
        self.assertEqual(result.audit["background_count"], 1)
        self.assertEqual(result.audit["selected_candidate_count"], 2)
        self.assertEqual(result.audit["underfill"], 0)

    def test_element_direct_policies_have_distinct_fixed_identities(self) -> None:
        from workflow_policies.identities import selector_identity
        from workflow_policies.selection import POLICY_EVIDENCE

        self.assertEqual(
            selector_identity("ELEMENT_DIRECT").as_dict(),
            {"name": "element-direct", "revision": 1},
        )
        self.assertEqual(
            selector_identity("ELEMENT_JOINT_DIRECT").as_dict(),
            {"name": "element-joint-direct", "revision": 1},
        )
        self.assertEqual(
            POLICY_EVIDENCE["ELEMENT_DIRECT"]["constants"]["population"],
            "ordered-candidates-only",
        )
        self.assertEqual(
            POLICY_EVIDENCE["ELEMENT_DIRECT"]["constants"]["fallback"],
            "committee-descriptor-owner-to-random",
        )
        self.assertEqual(
            POLICY_EVIDENCE["ELEMENT_JOINT_DIRECT"]["constants"]["population"],
            "ordered-candidates-then-training-background",
        )
        self.assertEqual(
            selector_identity("DIRECT").as_dict(),
            {"name": "DIRECT", "revision": 3},
        )
        self.assertEqual(
            selector_identity("JOINT_DIRECT").as_dict(),
            {"name": "joint-direct", "revision": 2},
        )

    def test_element_embedding_records_absence_and_presence_coverage(self) -> None:
        from workflow_policies.selection import select_element_representatives

        result = select_element_representatives(
            ["al-only", "mixed", "o-only"],
            [
                {"Al": np.zeros((1, 2))},
                {"Al": np.zeros((1, 2)), "O": np.zeros((3, 2))},
                {"O": np.zeros((2, 2))},
            ],
            presence=[("Al",), ("O", "Al"), ("O",)],
            type_map=("Al", "O", "H"),
            keep=2,
        )

        embedding = result.embedding
        self.assertEqual(embedding.active_elements, ("Al", "O"))
        self.assertEqual(embedding.inactive_elements, ("H",))
        self.assertEqual(result.audit["presence_counts"], {"Al": 2, "O": 2, "H": 0})
        self.assertEqual(embedding.vectors.shape, (3, 6))
        self.assertAlmostEqual(float(embedding.vectors[0, 2]), 1.0)
        self.assertAlmostEqual(float(embedding.vectors[0, 5]), 0.0)
        self.assertAlmostEqual(float(embedding.vectors[1, 2]), 1.0 / np.sqrt(2.0))
        self.assertAlmostEqual(float(embedding.vectors[1, 5]), 1.0 / np.sqrt(2.0))
        self.assertAlmostEqual(float(embedding.vectors[2, 2]), 0.0)
        self.assertAlmostEqual(float(embedding.vectors[2, 5]), 1.0)

    def test_element_joint_background_changes_membership_without_becoming_selectable(self) -> None:
        from workflow_policies.selection import (
            select_element_joint_representatives,
            select_element_representatives,
        )

        angles = [0.0, 0.2, 0.5, 1.0, 1.3]
        rows = [
            {"Al": np.array([[np.cos(angle), np.sin(angle)]])}
            for angle in angles
        ]
        identities = [f"candidate-{index}" for index in range(len(rows))]
        direct = select_element_representatives(
            identities,
            rows,
            presence=[("Al",)] * len(rows),
            type_map=("Al",),
            keep=2,
        )
        joint = select_element_joint_representatives(
            identities,
            rows,
            candidate_presence=[("Al",)] * len(rows),
            background_identities=["background"],
            background_element_rows=[
                {"Al": np.array([[np.cos(0.3), np.sin(0.3)]])}
            ],
            background_presence=[("Al",)],
            type_map=("Al",),
            keep=2,
        )

        self.assertNotEqual(joint.selected_identities, direct.selected_identities)
        self.assertNotIn("background", joint.selected_identities)
        self.assertEqual(len(joint.selected_identities), 2)

    def test_element_direct_rejects_incomplete_or_malformed_evidence(self) -> None:
        from workflow_policies.selection import (
            prepare_element_frame_vectors,
            select_element_representatives,
        )

        with self.assertRaisesRegex(ValueError, "chemically present"):
            prepare_element_frame_vectors(
                [{"Al": np.ones((1, 2))}],
                presence=[("Al", "O")],
                type_map=("Al", "O"),
            )
        with self.assertRaisesRegex(ValueError, "share one feature width"):
            prepare_element_frame_vectors(
                [{"Al": np.ones((1, 2)), "O": np.ones((1, 3))}],
                presence=[("Al", "O")],
                type_map=("Al", "O"),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_element_frame_vectors(
                [{"Al": np.array([[np.nan, 0.0]])}],
                presence=[("Al",)],
                type_map=("Al",),
            )
        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            select_element_representatives(
                ["same", "same"],
                [{"Al": np.ones((1, 2))}] * 2,
                presence=[("Al",)] * 2,
                type_map=("Al",),
                keep=1,
            )

    def test_element_embedding_allows_one_row_and_partition_local_presence(self) -> None:
        from workflow_policies.selection import select_element_joint_representatives

        result = select_element_joint_representatives(
            ["candidate"],
            [{"Li": np.array([[1.0, 0.0]])}],
            candidate_presence=[("Li",)],
            background_identities=["background"],
            background_element_rows=[{"Al": np.array([[0.0, 1.0]])}],
            background_presence=[("Al",)],
            type_map=("Al", "Li", "H"),
            keep=1,
        )

        self.assertEqual(result.selected_identities, ("candidate",))
        self.assertEqual(result.embedding.active_elements, ("Al", "Li"))
        self.assertEqual(result.embedding.inactive_elements, ("H",))
        self.assertEqual(
            result.audit["presence_counts"],
            {"Al": 1, "Li": 1, "H": 0},
        )
        self.assertTrue(np.isfinite(result.embedding.vectors).all())

    def test_joint_direct_clusters_candidates_with_background_but_returns_only_candidates(self) -> None:
        from workflow_policies.selection import (
            select_joint_representatives_with_audit,
        )

        selected, audit = select_joint_representatives_with_audit(
            ["candidate-x-low", "candidate-x-high", "candidate-y"],
            np.array([[1.0, -0.2], [1.0, 0.2], [0.0, 1.0]]),
            ["background-x"],
            np.array([[1.0, 0.0]]),
            keep=2,
        )

        self.assertEqual(selected, ["candidate-x-high", "candidate-y"])
        self.assertEqual(
            audit,
            {
                "realized_clusters": 2,
                "selected_background_count": 1,
                "selected_candidate_count": 2,
                "quota_completion_count": 1,
                "underfill": 0,
            },
        )

    def test_joint_direct_has_a_distinct_revision_two_policy_identity(self) -> None:
        from workflow_policies.identities import selector_identity
        from workflow_policies.selection import POLICY_EVIDENCE

        self.assertEqual(
            selector_identity("JOINT_DIRECT").as_dict(),
            {"name": "joint-direct", "revision": 2},
        )
        self.assertEqual(
            POLICY_EVIDENCE["JOINT_DIRECT"]["constants"]["fallback"],
            None,
        )

    def test_joint_direct_validates_candidate_and_background_partitions(self) -> None:
        from workflow_policies.selection import select_joint_representatives

        self.assertEqual(
            select_joint_representatives(
                [],
                np.empty((0, 2)),
                ["shared"],
                np.array([[1.0, 0.0]]),
                keep=1,
            ),
            [],
        )
        self.assertEqual(
            select_joint_representatives(
                ["shared"],
                np.array([[0.0, 1.0]]),
                ["shared"],
                np.array([[1.0, 0.0]]),
                keep=1,
            ),
            ["shared"],
        )
        invalid_cases = (
            (["a"], np.ones((1, 2)), [], np.empty((0, 2)), 1, "background"),
            (["a", "a"], np.ones((2, 2)), ["b"], np.ones((1, 2)), 1, "unique"),
            (["a"], np.ones((1, 2)), ["b"], np.ones((1, 3)), 1, "feature width"),
            (["a"], np.array([[np.nan, 0.0]]), ["b"], np.ones((1, 2)), 1, "finite"),
            (["a"], np.ones((1, 2)), ["b"], np.ones((1, 2)), 0, "positive"),
        )
        for candidate_ids, candidate_vectors, background_ids, background_vectors, keep, expected in invalid_cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                select_joint_representatives(
                    candidate_ids,
                    candidate_vectors,
                    background_ids,
                    background_vectors,
                    keep=keep,
                )

    def test_joint_direct_preprocesses_once_and_fills_a_collapsed_population(self) -> None:
        import workflow_policies.selection as selection

        candidate_ids = [f"candidate-{index:03d}" for index in range(250)]
        background_ids = [f"background-{index:03d}" for index in range(50)]
        original = selection._preprocess_frame_vectors
        calls: list[tuple[int, int]] = []

        def recording_preprocess(vectors):
            calls.append(tuple(np.asarray(vectors).shape))
            return original(vectors)

        with mock.patch.object(
            selection, "_preprocess_frame_vectors", recording_preprocess
        ):
            selected, audit = selection.select_joint_representatives_with_audit(
                candidate_ids,
                np.ones((250, 4)),
                background_ids,
                np.ones((50, 4)),
                keep=200,
            )

        self.assertEqual(calls, [(300, 4)])
        self.assertEqual(selected, candidate_ids[:200])
        self.assertEqual(audit["realized_clusters"], 1)
        self.assertEqual(audit["quota_completion_count"], 199)
        self.assertEqual(audit["selected_candidate_count"], 200)
        self.assertEqual(audit["underfill"], 0)
    def test_p_space_sampling_shares_tied_mass_and_is_stable_without_replacement(self) -> None:
        from workflow_policies.selection import p_space_information_masses, p_space_sample

        records = [("a", 0.1), ("b", 0.1), ("c", 0.8), ("d", 0.9)]
        first = p_space_sample(records, keep=3, entropy="run/r0/water")
        second = p_space_sample(records, keep=3, entropy="run/r0/water")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)
        self.assertTrue(set(first).issubset({"a", "b", "c", "d"}))
        masses = p_space_information_masses([0.1, 0.1, 0.8, 0.9])
        self.assertAlmostEqual(masses[0], masses[1])
        self.assertGreater(masses[0], 0.0)
        self.assertAlmostEqual(float(np.sum(p_space_information_masses([0.5, 0.5]))), 1.0)

    def test_gmm_p_space_conditions_mass_on_fixed_point_zero_one_floor(self) -> None:
        from workflow_policies.selection import p_space_information_masses, p_space_sample

        values = [0.01, 0.01, 0.2, 0.8]
        masses = p_space_information_masses(values, lower_bound=0.01)
        self.assertAlmostEqual(float(np.sum(masses)), 1.0)
        self.assertAlmostEqual(float(masses[0]), float(masses[1]))
        with self.assertRaisesRegex(ValueError, "conditional bounds"):
            p_space_information_masses([0.009], lower_bound=0.01)
        first = p_space_sample(
            [(str(index), value) for index, value in enumerate(values)],
            keep=3, entropy="gmm/water", lower_bound=0.01,
        )
        second = p_space_sample(
            list(reversed([(str(index), value) for index, value in enumerate(values)])),
            keep=3, entropy="gmm/water", lower_bound=0.01,
        )
        self.assertEqual(set(first), set(second))

    def test_p_space_mass_conditions_on_an_explicit_upper_bound(self) -> None:
        from workflow_policies.selection import p_space_information_masses

        values = [0.1, 0.2]
        masses = p_space_information_masses(values, lower_bound=0.01, upper_bound=0.3)

        def g_cdf(value: float) -> float:
            return value * (1.0 - np.log(value))

        midpoint = 0.15
        normalizer = g_cdf(0.3) - g_cdf(0.01)
        self.assertAlmostEqual(float(np.sum(masses)), 1.0)
        self.assertAlmostEqual(float(masses[0]), (g_cdf(midpoint) - g_cdf(0.01)) / normalizer)
        self.assertAlmostEqual(float(masses[1]), (g_cdf(0.3) - g_cdf(midpoint)) / normalizer)

    def test_p_space_sample_accepts_upper_bound_and_is_identity_order_deterministic(self) -> None:
        from workflow_policies.selection import p_space_sample

        records = [("z", 0.25), ("a", 0.01), ("m", 0.1), ("b", 0.1)]
        selected = p_space_sample(
            records,
            keep=3,
            entropy="iter0/r0/water",
            lower_bound=0.01,
            upper_bound=0.3,
        )
        reordered = p_space_sample(
            list(reversed(records)),
            keep=3,
            entropy="iter0/r0/water",
            lower_bound=0.01,
            upper_bound=0.3,
        )
        self.assertEqual(selected, reordered)
        self.assertEqual(len(selected), len(set(selected)))

    def test_p_space_explicit_bounds_include_floor_and_exclude_upper_equality(self) -> None:
        from workflow_policies.selection import p_space_information_masses

        masses = p_space_information_masses([0.1, 0.2], lower_bound=0.1, upper_bound=0.3)
        self.assertAlmostEqual(float(np.sum(masses)), 1.0)
        with self.assertRaisesRegex(ValueError, "conditional bounds"):
            p_space_information_masses([0.1, 0.3], lower_bound=0.1, upper_bound=0.3)

    def test_p_space_sample_validates_bounds_even_for_empty_and_full_quota_paths(self) -> None:
        from workflow_policies.selection import p_space_sample

        invalid_bounds = (
            {"lower_bound": -0.1, "upper_bound": 0.5},
            {"lower_bound": 0.5, "upper_bound": 0.5},
            {"lower_bound": 0.5, "upper_bound": 1.1},
            {"lower_bound": float("nan"), "upper_bound": 1.0},
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "p-space bounds"):
                p_space_sample([], keep=0, entropy="empty", **bounds)
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "p-space bounds"):
                p_space_sample([("a", 0.6)], keep=1, entropy="full", **bounds)

    def test_p_space_mass_rejects_invalid_bounds(self) -> None:
        from workflow_policies.selection import p_space_information_masses

        for lower, upper in ((-0.1, 0.5), (0.5, 0.5), (0.5, 1.1), (0.1, float("inf"))):
            with self.subTest(lower=lower, upper=upper), self.assertRaisesRegex(ValueError, "p-space bounds"):
                p_space_information_masses([], lower_bound=lower, upper_bound=upper)

    def test_p_space_omitted_upper_bound_preserves_legacy_full_interval_results(self) -> None:
        from workflow_policies.selection import p_space_information_masses, p_space_sample

        records = [("0", 0.01), ("1", 0.01), ("2", 0.2), ("3", 0.8)]
        masses = p_space_information_masses([value for _, value in records], lower_bound=0.01)
        np.testing.assert_allclose(
            masses,
            np.array([0.15127776, 0.15127776, 0.53490760, 0.16253688]),
            rtol=1e-7,
            atol=1e-8,
        )
        self.assertEqual(
            p_space_sample(records, keep=3, entropy="gmm/water", lower_bound=0.01),
            ["3", "2", "0"],
        )

    def test_p_space_empty_and_full_quota_paths_are_validated_and_ordered(self) -> None:
        from workflow_policies.selection import p_space_information_masses, p_space_sample

        self.assertEqual(p_space_information_masses([], lower_bound=0.1, upper_bound=0.4).shape, (0,))
        self.assertEqual(
            p_space_sample([], keep=0, entropy="empty", lower_bound=0.1, upper_bound=0.4),
            [],
        )
        records = [("z", 0.3), ("a", 0.1)]
        self.assertEqual(
            p_space_sample(records, keep=2, entropy="full", lower_bound=0.1, upper_bound=0.4),
            ["z", "a"],
        )
        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            p_space_sample(
                [("a", 0.1), ("a", 0.2)],
                keep=2,
                entropy="duplicate",
                lower_bound=0.1,
                upper_bound=0.4,
            )

    def test_configured_upper_one_excludes_one_while_omission_accepts_it(self) -> None:
        from workflow_policies.selection import p_space_information_masses

        self.assertAlmostEqual(float(p_space_information_masses([1.0])[0]), 1.0)
        with self.assertRaisesRegex(ValueError, "conditional bounds"):
            p_space_information_masses([1.0], upper_bound=1.0)

    def test_descriptor_reduction_uses_committee_mean_and_kaiser_fallback(self) -> None:
        from workflow_policies.selection import reduce_descriptors

        committee = [
            np.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]),
            np.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]),
        ]
        reduced = reduce_descriptors(committee)
        self.assertEqual(reduced.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(reduced)))
        fallback = reduce_descriptors(np.ones((3, 2, 2)))
        self.assertEqual(fallback.shape, (3, 2))

    def test_descriptor_frame_reduction_accepts_ragged_atom_counts(self) -> None:
        from workflow_policies.selection import reduce_descriptor_frames

        reduced = reduce_descriptor_frames([np.ones((1, 64)), np.full((3, 64), 2.0)])
        self.assertEqual(reduced.shape, (2, 64))

    def test_direct_revision_3_matches_independent_frame_vector_reference(self) -> None:
        import warnings

        from sklearn.cluster import Birch
        from sklearn.decomposition import PCA
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.preprocessing import StandardScaler

        from workflow_policies.selection import select_representatives

        identities = ["x1", "x2", "x3", "y1", "y2", "y3"]
        vectors = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
            ]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms > 0,
        )
        standardized = StandardScaler().fit_transform(normalized)
        pca = PCA().fit(standardized)
        retained = pca.explained_variance_ > 1.0
        reduced = (
            pca.transform(standardized)[:, retained]
            * pca.explained_variance_ratio_[retained]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = Birch(
                threshold=0.5,
                branching_factor=50,
                n_clusters=3,
            ).fit_predict(reduced)
        expected_indices = []
        for label in sorted(set(int(item) for item in labels)):
            indices = np.flatnonzero(labels == label)
            centroid = np.mean(reduced[indices], axis=0)
            expected_indices.append(
                int(
                    indices[
                        np.argmin(
                            np.linalg.norm(reduced[indices] - centroid, axis=1)
                        )
                    ]
                )
            )
        while len(expected_indices) < 3:
            nearest = np.min(
                np.linalg.norm(
                    reduced[:, None, :] - reduced[expected_indices][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            nearest[expected_indices] = -np.inf
            expected_indices.append(int(np.argmax(nearest)))

        selected = select_representatives(
            identities,
            vectors,
            keep=3,
            policy="DIRECT",
            entropy="direct-revision-3/reference",
        )

        self.assertEqual(selected, [identities[index] for index in expected_indices])
        self.assertEqual(len(selected), 3)

    def test_direct_keep_is_an_exact_quota_for_identical_frames(self) -> None:
        from workflow_policies.selection import select_representatives

        selected = select_representatives(
            ["first", "second", "third"],
            np.ones((3, 4), dtype=float),
            keep=3,
            policy="DIRECT",
            entropy="direct-revision-3/identical",
        )

        self.assertEqual(selected, ["first", "second", "third"])

    def test_direct_fails_closed_when_l2_norm_is_non_finite(self) -> None:
        from workflow_policies.selection import select_representatives

        huge = np.finfo(float).max / 2.0
        with self.assertRaisesRegex(ValueError, "non-finite"):
            select_representatives(
                ["a", "b"],
                np.array([[huge, huge], [huge, -huge]]),
                keep=1,
                policy="DIRECT",
                entropy="direct-revision-3/non-finite-intermediate",
            )

    def test_direct_selects_the_only_frame(self) -> None:
        from workflow_policies.selection import select_representatives

        self.assertEqual(
            select_representatives(
                ["only"],
                np.array([[3.0, 4.0]]),
                keep=1,
                policy="DIRECT",
                entropy="direct-revision-3/one-frame",
            ),
            ["only"],
        )

    def test_selectors_are_deterministic_and_fill_the_exact_quota(self) -> None:
        from workflow_policies.selection import select_representatives

        ids = ["a", "b", "c", "d", "e"]
        vectors = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        for policy in ("DIRECT", "FPS", "random"):
            with self.subTest(policy=policy):
                first = select_representatives(ids, vectors, keep=3, policy=policy, entropy="iter0/fp/water")
                second = select_representatives(ids, vectors, keep=3, policy=policy, entropy="iter0/fp/water")
                self.assertEqual(first, second)
                self.assertEqual(len(first), 3)
                self.assertEqual(len(set(first)), len(first))

    def test_direct_fills_a_collapsed_birch_population_to_the_exact_quota(self) -> None:
        from workflow_policies.selection import select_representatives

        identities = [f"frame-{index:03d}" for index in range(250)]
        selected = select_representatives(
            identities,
            np.ones((250, 4)),
            keep=200,
            policy="DIRECT",
            entropy="direct-revision-3/collapsed-birch",
        )

        self.assertEqual(len(selected), 200)
        self.assertEqual(selected, identities[:200])

    def test_policy_evidence_is_fixed_and_runtime_tuning_does_not_change_selection(self) -> None:
        from workflow_policies.identities import selector_identity
        from workflow_policies.selection import POLICY_EVIDENCE, select_representatives

        self.assertEqual(POLICY_EVIDENCE["gmm-reference-tail"]["revision"], 4)
        self.assertEqual(POLICY_EVIDENCE["pair-guard"]["revision"], 1)
        self.assertEqual(selector_identity("DIRECT").as_dict(), {"name": "DIRECT", "revision": 3})
        self.assertEqual(
            POLICY_EVIDENCE["DIRECT"],
            {
                "revision": 3,
                "constants": {
                    "frame_pooling": "mean-if-atom-resolved",
                    "normalization": "per-frame-l2",
                    "standardization": "population-standard-deviation",
                    "zero_variance_divisor": 1.0,
                    "pca_retention": "eigenvalue > 1.0",
                    "pca_weight": "explained-variance-ratio",
                    "clustering": "BIRCH",
                    "threshold": 0.5,
                    "branching_factor": 50,
                    "target_clusters": "min(keep, population)",
                    "per_cluster_keep": 1,
                    "representative": "centroid-nearest-canonical-first-tie",
                    "quota": "exact-min-keep-candidate-population",
                    "completion": "greedy-farthest-from-selected",
                    "underfill": "zero-against-achievable-quota",
                    "fallback": None,
                },
            },
        )
        self.assertEqual(
            POLICY_EVIDENCE["FPS"],
            {"revision": 1, "constants": {"start": "pool_medoid", "strategy": "greedy_farthest"}},
        )
        self.assertEqual(
            POLICY_EVIDENCE["random"],
            {"revision": 1, "constants": {"strategy": "stable_uniform_without_replacement"}},
        )
        self.assertEqual(POLICY_EVIDENCE["DIRECT"]["constants"]["threshold"], 0.5)
        ids = ["a", "b", "c"]
        vectors = np.array([[0.0], [1.0], [2.0]])
        self.assertEqual(
            select_representatives(ids, vectors, keep=2, policy="FPS", entropy="same"),
            select_representatives(ids, vectors, keep=2, policy="FPS", entropy="same", runtime_tuning={"descriptor_chunk_frames": 1}),
        )

    def test_element_direct_fallback_is_resolved_only_from_known_model_shape(self) -> None:
        from workflow_policies.resolution import resolve_element_direct_policy

        single = resolve_element_direct_policy(
            "ELEMENT_DIRECT", descriptor_model_shape="single"
        )
        committee = resolve_element_direct_policy(
            "ELEMENT_DIRECT", descriptor_model_shape="committee"
        )
        ordinary = resolve_element_direct_policy(
            "DIRECT", descriptor_model_shape="committee"
        )

        self.assertEqual(
            (single.requested_policy, single.effective_policy, single.fallback_reason),
            ("ELEMENT_DIRECT", "ELEMENT_DIRECT", None),
        )
        self.assertEqual(
            (
                committee.requested_policy,
                committee.effective_policy,
                committee.fallback_reason,
            ),
            (
                "ELEMENT_DIRECT",
                "random",
                "committee_descriptor_unavailable",
            ),
        )
        self.assertEqual(
            (ordinary.effective_policy, ordinary.fallback_reason),
            ("DIRECT", None),
        )
        with self.assertRaisesRegex(ValueError, "model shape"):
            resolve_element_direct_policy(
                "ELEMENT_DIRECT", descriptor_model_shape="unknown"
            )
