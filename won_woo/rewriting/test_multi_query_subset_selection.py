"""Run: python -m unittest discover -s rewriting -p test_multi_query_subset_selection.py"""

import unittest

import numpy as np

from multi_query_subset_selection import select_subset


class MultiQuerySubsetSelectionTest(unittest.TestCase):
    def select(self, original, candidates, **kwargs):
        return select_subset(
            original, candidates, 0,
            ["candidate_" + str(i) for i in range(len(candidates))],
            ["positive"] + ["negative_" + str(i) for i in range(len(original) - 1)],
            original_query_id="original", **kwargs,
        )

    def test_individually_harmful_candidates_can_form_helpful_pair(self):
        result = self.select([0, 0, 0], [[0, 2, -5], [0, -5, 2]])
        self.assertEqual(result["selected_candidate_indices"], [0, 1])
        self.assertEqual(result["selected_query_ids"], ["original", "candidate_0", "candidate_1"])
        self.assertEqual([row["margin"] for row in result["evaluations"][1:3]], [-1, -1])
        self.assertAlmostEqual(result["selected"]["margin"], 1)
        self.assertEqual(result["evaluated_pair_count"], 1)
        self.assertTrue(all(gain > 0 for gain in result["selected"]["duplicate_margin_gains"]))

    def test_pair_without_distinct_candidate_gain_is_rejected(self):
        original, candidates = [0, 4, 0], [[0, -4, 1], [0, -2, 0]]
        result = self.select(original, candidates)
        self.assertEqual(result["selected_candidate_indices"], [0])
        pair = result["evaluations"][-1]
        self.assertTrue(all(gain > 0 for gain in pair["removal_margin_gains"]))
        self.assertFalse(pair["eligible"])
        self.assertLess(pair["duplicate_margin_gains"][1], 0)
        result_without_control = self.select(original, candidates, require_duplicate_gain=False)
        self.assertEqual(result_without_control["selected_candidate_indices"], [0, 1])

    def test_mean_scores_precedes_maximum_negative(self):
        result = self.select([2, 1, 0], [[2, 0, 3]])
        single = result["evaluations"][1]
        self.assertAlmostEqual(single["margin"], 0.5)
        self.assertEqual(single["hardest_negative_video_id"], "negative_1")
        self.assertEqual(result["selected_candidate_count"], 0)

    def test_empty_and_objective_tie_choose_original(self):
        empty = self.select([1, 0], [])
        self.assertEqual(empty["selected_query_ids"], ["original"])
        self.assertEqual(empty["evaluated_pair_count"], 0)
        tied = self.select([1, 0], [[2, 0]], lambda_penalty=0.5)
        self.assertEqual(tied["selected_candidate_count"], 0)
        nearly_tied = self.select([1, 0], [[1 + 1e-7, 0]])
        self.assertEqual(nearly_tied["selected_candidate_count"], 0)

    def test_duplicate_pair_prefers_smaller_query_set(self):
        result = self.select([1, 0], [[2, 0], [2, 0]])
        self.assertEqual(result["selected_candidate_count"], 1)
        self.assertEqual(result["evaluations"][-1]["complementary_margin_gain"], 0)

    def test_gallery_reordering_keeps_choice_and_hardest_negative_id(self):
        original = np.array([0, 0, -0.3])
        candidates = np.array([[0, 2, -5], [0, -5, 2]])
        kwargs = dict(query_ids=["A", "B"], original_query_id="original")
        first = select_subset(original, candidates, 0, video_ids=["P", "N1", "N2"], **kwargs)
        permutation = [2, 0, 1]
        reordered = select_subset(
            original[permutation], candidates[:, permutation], 1,
            video_ids=["N2", "P", "N1"], **kwargs,
        )
        self.assertEqual(first["selected_candidate_ids"], reordered["selected_candidate_ids"])
        self.assertAlmostEqual(first["selected"]["margin"], reordered["selected"]["margin"])
        self.assertEqual(first["selected"]["hardest_negative_video_id"],
                         reordered["selected"]["hardest_negative_video_id"])

    def test_invalid_gallery_scores_are_not_silently_selected(self):
        for original, candidates in [([0, np.nan], [[0, 1]]), ([0, 1], [[0, np.inf]])]:
            with self.assertRaises(ValueError):
                self.select(original, candidates)
        with self.assertRaises(ValueError):
            select_subset([1, 0], [[2, 0]], 0, ["A"], ["P", "P"])


if __name__ == "__main__":
    unittest.main()
