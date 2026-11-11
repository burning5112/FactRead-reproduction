import math
import unittest
from factread.evaluation import ranking_metrics, value_attribution


class MetricsTest(unittest.TestCase):
    def test_shared_first_place(self):
        m = ranking_metrics([0, 0, -1, -2, -3, -4, -5, -6], 0)
        self.assertEqual(m["acc_at1"], 0.5)
        self.assertAlmostEqual(m["mrr"], 1 / 1.5)

    def test_tie_below_first_place(self):
        m = ranking_metrics([1, 0, 0, -1, -2, -3, -4, -5], 1)
        self.assertEqual(m["acc_at1"], 0)
        self.assertAlmostEqual(m["mrr"], 1 / 2.5)

    def test_eight_way_tie(self):
        m = ranking_metrics([0] * 8, 3)
        self.assertEqual(m["acc_at1"], 1 / 8)
        self.assertAlmostEqual(m["mrr"], 1 / 4.5)

    def test_support_effect_is_signed_and_strict(self):
        self.assertEqual(value_attribution([-3, 1, 2], 0), 0)
        self.assertEqual(value_attribution([2, -2, 1], 0), 0)
        self.assertEqual(value_attribution([3, -2, 1], 0), 1)

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([math.nan] + [0] * 7, 0)
        with self.assertRaises(ValueError):
            value_attribution([1, math.inf], 0)


if __name__ == "__main__":
    unittest.main()
