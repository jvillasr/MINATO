import unittest

import numpy as np

from minato.binary_population import BinaryPopulation
from minato.binary_population.orbits import orbital_semi_amplitudes_kms, solve_kepler


class OrbitalSemiAmplitudeTests(unittest.TestCase):
    def test_representative_mass_period_q_eccentricity_grid_matches_si_truth(self):
        primary_mass = np.array([5.0, 10.0, 20.0, 40.0])
        mass_ratio = np.array([0.2, 0.5, 0.8, 1.0])
        secondary_mass = primary_mass * mass_ratio
        period_days = np.array([2.0, 10.0, 100.0, 1000.0])
        inclination = np.radians([30.0, 60.0, 90.0, 120.0])
        eccentricity = np.array([0.0, 0.2, 0.6, 0.9])

        k1, k2 = orbital_semi_amplitudes_kms(
            primary_mass,
            secondary_mass,
            period_days,
            inclination,
            eccentricity,
        )

        gravitational_constant_si = 6.67430e-11
        solar_mass_kg = 1.98847e30
        expected_k1 = (
            (2.0 * np.pi * gravitational_constant_si / (period_days * 86400.0))
            ** (1.0 / 3.0)
            * (secondary_mass * solar_mass_kg)
            * np.sin(inclination)
            / ((primary_mass + secondary_mass) * solar_mass_kg) ** (2.0 / 3.0)
            / np.sqrt(1.0 - eccentricity**2)
            / 1000.0
        )

        np.testing.assert_allclose(k1, expected_k1, rtol=5e-4)
        np.testing.assert_allclose(k1 / k2, mass_ratio, rtol=1e-14)

    def test_eccentric_amplitude_matches_independent_si_calculation(self):
        primary_mass = 10.0
        secondary_mass = 5.0
        period_days = 10.0
        inclination = np.radians(60.0)
        eccentricity = 0.6

        k1, k2 = orbital_semi_amplitudes_kms(
            primary_mass,
            secondary_mass,
            period_days,
            inclination,
            eccentricity,
        )

        gravitational_constant_si = 6.67430e-11
        solar_mass_kg = 1.98847e30
        period_seconds = period_days * 86400.0
        expected_k1 = (
            (2.0 * np.pi * gravitational_constant_si / period_seconds) ** (1.0 / 3.0)
            * (secondary_mass * solar_mass_kg)
            * np.sin(inclination)
            / ((primary_mass + secondary_mass) * solar_mass_kg) ** (2.0 / 3.0)
            / np.sqrt(1.0 - eccentricity**2)
            / 1000.0
        )
        np.testing.assert_allclose(k1, expected_k1, rtol=5e-4)
        np.testing.assert_allclose(k1 / k2, secondary_mass / primary_mass, rtol=1e-14)

    def test_eccentric_factor_and_circular_regression(self):
        inputs = (10.0, 5.0, 10.0, np.radians(60.0))
        circular_k1, _ = orbital_semi_amplitudes_kms(*inputs, 0.0)
        eccentric_k1, _ = orbital_semi_amplitudes_kms(*inputs, 0.6)
        np.testing.assert_allclose(eccentric_k1 / circular_k1, 1.25, rtol=1e-14)

    def test_rv_curve_peak_to_peak_is_twice_semi_amplitude(self):
        population = BinaryPopulation()
        period_days = 10.0
        k1, k2 = orbital_semi_amplitudes_kms(
            10.0,
            5.0,
            period_days,
            np.radians(60.0),
            0.6,
        )
        velocities = population.rvcurve(
            np.array([0.0, period_days / 2.0]),
            period_days,
            0.0,
            0.6,
            0.0,
            13.0,
            k1,
            k2,
        )
        np.testing.assert_allclose(np.ptp(velocities), 2.0 * k1, rtol=1e-12)

    def test_epoch_velocities_match_independent_kepler_formula_across_phases(self):
        population = BinaryPopulation()
        period_days = 37.0
        eccentricity = 0.7
        omega_deg = 43.0
        gamma = 17.0
        epochs = period_days * np.array([0.0, 0.03, 0.17, 0.41, 0.76, 1.25])
        k1, k2 = orbital_semi_amplitudes_kms(
            20.0,
            8.0,
            period_days,
            np.radians(67.0),
            eccentricity,
        )

        actual_primary, actual_secondary = population.rvcurve(
            epochs,
            period_days,
            0.0,
            eccentricity,
            omega_deg,
            gamma,
            k1,
            k2,
            SB2=True,
        )
        mean_anomaly = 2.0 * np.pi * epochs / period_days
        eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
        true_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
            np.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
        )
        omega = np.radians(omega_deg)
        phase_term = (
            np.cos(true_anomaly + omega)
            + eccentricity * np.cos(omega)
        )

        np.testing.assert_allclose(actual_primary, gamma + k1 * phase_term)
        np.testing.assert_allclose(actual_secondary, gamma - k2 * phase_term)

    def test_invalid_orbital_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            orbital_semi_amplitudes_kms(10.0, 5.0, 0.0, 1.0, 0.2)
        with self.assertRaises(ValueError):
            orbital_semi_amplitudes_kms(10.0, 5.0, 10.0, 1.0, 1.0)


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
