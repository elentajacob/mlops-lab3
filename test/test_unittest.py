"""unittest suite: TestCase classes, subTest for table-driven cases, assertRaises checks."""

import unittest

from src import ml_metrics


class TestDescriptiveStatistics(unittest.TestCase):
    def test_mean(self):
        cases = [([1, 2, 3, 4], 2.5), ([5], 5.0), ([-2, 2], 0.0), ((10, 20, 30), 20.0)]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertAlmostEqual(ml_metrics.mean(values), expected)

    def test_population_and_sample_variance(self):
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(ml_metrics.variance(values), 4.0)
        self.assertAlmostEqual(ml_metrics.variance(values, sample=True), 32 / 7)

    def test_std(self):
        self.assertAlmostEqual(ml_metrics.std([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)

    def test_z_scores(self):
        z = ml_metrics.z_scores([10, 20, 30, 40, 50])
        self.assertAlmostEqual(ml_metrics.mean(z), 0.0)
        self.assertAlmostEqual(ml_metrics.std(z), 1.0)
        self.assertEqual(ml_metrics.z_scores([3, 3]), [0.0, 0.0])

    def test_min_max_scale(self):
        self.assertEqual(ml_metrics.min_max_scale([0, 5, 10]), [0.0, 0.5, 1.0])
        self.assertEqual(ml_metrics.min_max_scale([2, 2]), [0.0, 0.0])


class TestInputValidation(unittest.TestCase):
    def test_empty_input(self):
        for func in (ml_metrics.mean, ml_metrics.variance, ml_metrics.std):
            with self.subTest(func=func.__name__):
                with self.assertRaises(ValueError):
                    func([])

    def test_non_numeric_input(self):
        for bad in ("abc", [1, "x"], [None], [True]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    ml_metrics.mean(bad)

    def test_sample_variance_single_value(self):
        with self.assertRaises(ValueError):
            ml_metrics.variance([1], sample=True)

    def test_bad_feature_range(self):
        with self.assertRaises(ValueError):
            ml_metrics.min_max_scale([1, 2], feature_range=(5, 5))

    def test_invalid_labels(self):
        cases = [([], []), ([1, 0], [1]), ([1, 3], [1, 0])]
        for y_true, y_pred in cases:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                with self.assertRaises(ValueError):
                    ml_metrics.accuracy(y_true, y_pred)


class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        # TP=4, FP=1, TN=3, FN=2
        self.y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        self.y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]

    def test_confusion_matrix(self):
        self.assertEqual(
            ml_metrics.confusion_matrix(self.y_true, self.y_pred),
            {"tp": 4, "fp": 1, "tn": 3, "fn": 2},
        )

    def test_scores(self):
        self.assertAlmostEqual(ml_metrics.accuracy(self.y_true, self.y_pred), 0.7)
        self.assertAlmostEqual(ml_metrics.precision(self.y_true, self.y_pred), 0.8)
        self.assertAlmostEqual(ml_metrics.recall(self.y_true, self.y_pred), 4 / 6)

    def test_f1_between_precision_and_recall(self):
        f1 = ml_metrics.f1_score(self.y_true, self.y_pred)
        self.assertGreaterEqual(f1, min(4 / 6, 0.8))
        self.assertLessEqual(f1, max(4 / 6, 0.8))

    def test_zero_division(self):
        self.assertEqual(ml_metrics.precision([0, 1], [0, 0]), 0.0)
        self.assertEqual(ml_metrics.recall([0, 0], [0, 1]), 0.0)
        self.assertEqual(ml_metrics.f1_score([1, 1], [0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
