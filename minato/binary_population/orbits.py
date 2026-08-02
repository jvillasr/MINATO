"""Numerical helpers for binary orbits."""

import numpy as np
from scipy.optimize import newton


G_KM3_S2_MSUN = 1.3271244e11
SECONDS_PER_DAY = 86400.0


def orbital_semi_amplitudes_kms(
    primary_mass_msun,
    secondary_mass_msun,
    period_days,
    inclination_rad,
    eccentricity,
):
    """Return primary and secondary Keplerian RV semi-amplitudes in km/s.

    Inputs follow NumPy broadcasting rules. The returned amplitudes include
    the standard eccentric-orbit factor ``1 / sqrt(1 - e**2)`` required by
    ``v = gamma + K * (cos(theta + omega) + e*cos(omega))``.
    """
    primary_mass, secondary_mass, period, inclination, eccentricity = np.broadcast_arrays(
        np.asarray(primary_mass_msun, dtype=float),
        np.asarray(secondary_mass_msun, dtype=float),
        np.asarray(period_days, dtype=float),
        np.asarray(inclination_rad, dtype=float),
        np.asarray(eccentricity, dtype=float),
    )
    for name, values in (
        ("primary_mass_msun", primary_mass),
        ("secondary_mass_msun", secondary_mass),
        ("period_days", period),
        ("inclination_rad", inclination),
        ("eccentricity", eccentricity),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(primary_mass <= 0.0) or np.any(secondary_mass <= 0.0):
        raise ValueError("component masses must be positive")
    if np.any(period <= 0.0):
        raise ValueError("period_days must be positive")
    if np.any((eccentricity < 0.0) | (eccentricity >= 1.0)):
        raise ValueError("eccentricity must lie in [0, 1)")

    period_sec = period * SECONDS_PER_DAY
    total_mass_factor = np.power(primary_mass + secondary_mass, 2.0 / 3.0)
    orbital_factor = (
        np.power(2.0 * np.pi * G_KM3_S2_MSUN, 1.0 / 3.0)
        * np.power(period_sec, -1.0 / 3.0)
        * np.power(1.0 - np.square(eccentricity), -0.5)
    )
    sin_inclination = np.sin(inclination)
    k1 = orbital_factor * secondary_mass * sin_inclination / total_mass_factor
    k2 = orbital_factor * primary_mass * sin_inclination / total_mass_factor
    return k1, k2


def solve_kepler(mean_anomaly, eccentricity, *, tolerance=1e-12, maxiter=20):
    """Solve ``E - e sin(E) = M`` for elliptical orbits.

    Mean anomalies are reduced to ``[-pi, pi)`` for stable vectorised Halley
    iterations and restored to their original cycle afterwards. Inputs follow
    NumPy broadcasting rules and eccentricities must lie in ``[0, 1)``.
    """
    mean_anomaly, eccentricity = np.broadcast_arrays(
        np.asarray(mean_anomaly, dtype=float),
        np.asarray(eccentricity, dtype=float),
    )
    if mean_anomaly.size == 0:
        return mean_anomaly.copy()
    if not np.all(np.isfinite(mean_anomaly)):
        raise ValueError("mean_anomaly must contain only finite values")
    if not np.all(np.isfinite(eccentricity)):
        raise ValueError("eccentricity must contain only finite values")
    if np.any((eccentricity < 0.0) | (eccentricity >= 1.0)):
        raise ValueError("eccentricity must lie in [0, 1)")

    reduced_mean = (mean_anomaly + np.pi) % (2.0 * np.pi) - np.pi
    cycle_offset = mean_anomaly - reduced_mean
    initial = reduced_mean + 0.85 * eccentricity * np.sign(np.sin(reduced_mean))

    reduced_solution = newton(
        lambda value: value - eccentricity * np.sin(value) - reduced_mean,
        initial,
        fprime=lambda value: 1.0 - eccentricity * np.cos(value),
        fprime2=lambda value: eccentricity * np.sin(value),
        tol=float(tolerance),
        maxiter=int(maxiter),
    )
    return reduced_solution + cycle_offset
