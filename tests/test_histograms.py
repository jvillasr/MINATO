import unittest

import numpy as np

from minato.binary_population.histograms import (
    DEFAULT_DRV_BINS,
    complete_nonnegative_bins,
    histogram_nonnegative,
)


class CompleteHistogramBinTests(unittest.TestCase):
    def test_default_bins_include_low_and_high_tails(self):
        self.assertEqual(DEFAULT_DRV_BINS[0], 0.0)
        self.assertTrue(np.isposinf(DEFAULT_DRV_BINS[-1]))

        values = np.array([0.0, 0.5, 2.0, 1000.0, 1500.0])
        counts, _ = np.histogram(values, bins=DEFAULT_DRV_BINS)
        self.assertEqual(int(counts.sum()), len(values))

    def test_custom_interior_edges_are_completed(self):
        bins = complete_nonnegative_bins([2.5, 10.0, 1000.0])
        np.testing.assert_array_equal(bins, [0.0, 2.5, 10.0, 1000.0, np.inf])

    def test_invalid_edges_are_rejected(self):
        with self.assertRaises(ValueError):
            complete_nonnegative_bins([0.0, 10.0, 10.0])
        with self.assertRaises(ValueError):
            complete_nonnegative_bins([0.0, np.nan, 10.0])
        with self.assertRaises(ValueError):
            complete_nonnegative_bins([-1.0, 10.0])

    def test_exact_edges_and_tail_values_conserve_rows(self):
        values = np.array([0.0, 2.5, 10.0, 1000.0, 1500.0])
        counts, bins = histogram_nonnegative(
            values,
            [2.5, 10.0, 1000.0],
            name="test dRV_max",
        )

        np.testing.assert_array_equal(bins, [0.0, 2.5, 10.0, 1000.0, np.inf])
        np.testing.assert_array_equal(counts, [1, 1, 1, 2])
        self.assertEqual(int(counts.sum()), values.size)

    def test_invalid_values_are_rejected_instead_of_dropped(self):
        for invalid in (-1.0, np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    histogram_nonnegative([0.0, invalid], [0.0, 10.0])


if __name__ == "__main__":
    unittest.main()
