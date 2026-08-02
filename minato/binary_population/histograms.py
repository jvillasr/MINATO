"""Shared histogram definitions for non-negative RV summary statistics."""

import numpy as np


DEFAULT_DRV_INTERIOR_EDGES = np.logspace(0.4, 3.0, 30)


def complete_nonnegative_bins(edges=None):
    """Return strictly increasing bins covering every finite value >= 0.

    The historical MINATO defaults contained only the logarithmic interior
    edges, silently omitting values below 10**0.4 and above 1000 km/s.
    Missing zero and positive-infinity edges are added explicitly.
    """
    values = DEFAULT_DRV_INTERIOR_EDGES if edges is None else edges
    bins = np.asarray(values, dtype=float)
    if bins.ndim != 1 or bins.size < 1:
        raise ValueError("histogram edges must be a one-dimensional non-empty sequence")
    if np.any(np.isnan(bins)):
        raise ValueError("histogram edges cannot contain NaN")
    if np.any(np.isneginf(bins)):
        raise ValueError("non-negative histogram edges cannot contain -inf")
    if bins[0] < 0.0:
        raise ValueError("non-negative histogram edges cannot start below zero")
    if bins[0] > 0.0:
        bins = np.concatenate(([0.0], bins))
    if not np.isposinf(bins[-1]):
        bins = np.concatenate((bins, [np.inf]))
    if not np.all(np.isfinite(bins[:-1])):
        raise ValueError("only the final histogram edge may be infinite")
    if not np.all(np.diff(bins) > 0.0):
        raise ValueError("histogram edges must be strictly increasing")
    return bins


DEFAULT_DRV_BINS = complete_nonnegative_bins()


def validate_nonnegative_finite(values, *, name="values"):
    """Return a float array after requiring finite, non-negative values."""
    array = np.asarray(values, dtype=float)
    invalid = ~np.isfinite(array) | (array < 0.0)
    if np.any(invalid):
        raise ValueError(
            f"{name} must contain only finite non-negative values; "
            f"found {int(np.count_nonzero(invalid))} invalid value(s)"
        )
    return array


def histogram_nonnegative(values, bins=None, *, name="values"):
    """Histogram finite non-negative values and require exact row conservation."""
    array = validate_nonnegative_finite(values, name=name)
    completed_bins = complete_nonnegative_bins(bins)
    counts, _ = np.histogram(array, bins=completed_bins)
    if int(counts.sum()) != int(array.size):
        raise RuntimeError(
            f"{name} histogram assigned {int(counts.sum())} of "
            f"{int(array.size)} values"
        )
    return counts, completed_bins
