"""Numerical helpers for binary orbits."""

import numpy as np
from scipy.optimize import newton


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
