import unittest

import numpy as np

from minato.binary_population.orbits import solve_kepler


class KeplerSolverTests(unittest.TestCase):
    def test_circular_orbit_returns_mean_anomaly(self):
        mean_anomaly = np.linspace(-8.0 * np.pi, 8.0 * np.pi, 41)
        np.testing.assert_allclose(solve_kepler(mean_anomaly, 0.0), mean_anomaly)

    def test_random_elliptical_orbits_satisfy_equation(self):
        rng = np.random.default_rng(20260712)
        mean_anomaly = rng.uniform(-100.0 * np.pi, 100.0 * np.pi, 10_000)
        eccentricity = rng.uniform(0.0, 0.95, mean_anomaly.size)

        eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)

        np.testing.assert_allclose(residual, mean_anomaly, atol=5e-12, rtol=0.0)

    def test_invalid_eccentricity_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_kepler(0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
