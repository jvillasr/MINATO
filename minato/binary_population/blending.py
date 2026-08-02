"""Generic hooks for per-epoch empirical RV blending biases."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def blending_metadata(blending_kernel, blending_flux_fraction=None) -> dict[str, object]:
    """Return lightweight provenance for an optional blending-bias sampler."""

    if blending_kernel is None:
        return {"enabled": False}

    metadata = getattr(blending_kernel, "metadata", None)
    if callable(metadata):
        metadata = metadata()
    if metadata is None:
        metadata = getattr(blending_kernel, "provenance", None)
        if callable(metadata):
            metadata = metadata()
    if metadata is None:
        metadata = {}

    result = dict(metadata)
    result.setdefault("enabled", True)
    result.setdefault("kernel_type", type(blending_kernel).__name__)
    if hasattr(blending_kernel, "sample_bias_signed"):
        result.setdefault("sampling_protocol", "signed_velocity_separation")
    else:
        result.setdefault("sampling_protocol", "legacy_absolute_velocity_separation")
    if blending_flux_fraction is None:
        result.setdefault("flux_fraction_source", None)
    elif callable(blending_flux_fraction):
        result.setdefault(
            "flux_fraction_source",
            getattr(blending_flux_fraction, "__name__", type(blending_flux_fraction).__name__),
        )
    else:
        result.setdefault("flux_fraction_source", "constant")
    return result


def resolve_blending_flux_fraction(
    blending_flux_fraction,
    *,
    intrinsic_arrays: dict[str, np.ndarray],
    system_index: int,
    n_epochs: int,
):
    """
    Resolve the secondary light fraction passed to a blending-bias sampler.

    ``blending_flux_fraction`` may be ``None``, a scalar, or a callable. Callable
    providers are called with keyword arguments so project-specific code can
    compute an effective secondary flux fraction from the intrinsic binary
    arrays without MINATO knowing that study-specific model.
    """

    if blending_flux_fraction is None:
        return None
    if callable(blending_flux_fraction):
        return blending_flux_fraction(
            intrinsic_arrays=intrinsic_arrays,
            system_index=int(system_index),
            n_epochs=int(n_epochs),
        )

    value = np.asarray(blending_flux_fraction, dtype=float)
    if value.shape == ():
        return float(value)
    if value.shape == (int(n_epochs),):
        return value
    raise ValueError(
        "blending_flux_fraction must be None, a scalar, an epoch-length array, "
        "or a callable accepting intrinsic_arrays, system_index, and n_epochs."
    )


def sample_blending_bias(
    blending_kernel,
    *,
    velocity_separation: np.ndarray | None = None,
    abs_velocity_separation: np.ndarray | None = None,
    secondary_flux_fraction,
    blend_unit: np.ndarray | None,
) -> np.ndarray:
    """
    Sample per-epoch RV blending bias from a user-supplied kernel.

    The kernel data/model is intentionally not part of MINATO. The supplied
    Signed-aware kernels should provide
    ``sample_bias_signed(delta_v, f_secondary, u)``. Existing kernels that
    provide ``sample_bias(abs_delta_v, f_secondary, u)`` or are directly
    callable with that legacy signature remain supported.
    """

    if velocity_separation is not None and abs_velocity_separation is not None:
        raise ValueError(
            "Provide velocity_separation or abs_velocity_separation, not both."
        )
    if velocity_separation is None and abs_velocity_separation is None:
        raise ValueError(
            "velocity_separation or abs_velocity_separation is required."
        )

    signed_separation_available = velocity_separation is not None
    if signed_separation_available:
        separation = np.asarray(velocity_separation, dtype=float)
    else:
        separation = np.asarray(abs_velocity_separation, dtype=float)
    if blending_kernel is None:
        return np.zeros_like(separation, dtype=float)
    if blend_unit is None:
        raise ValueError(
            "A blending_kernel was supplied, but the binary random bank has no "
            "blend_unit draws. Rebuild the bank with build_mixture_crn_banks."
        )

    blend_unit = np.asarray(blend_unit, dtype=float)
    if blend_unit.shape != separation.shape:
        raise ValueError("blend_unit must have the same shape as velocity_separation.")

    sampler: Callable
    if hasattr(blending_kernel, "sample_bias_signed"):
        if not signed_separation_available:
            raise ValueError(
                "A signed blending kernel requires velocity_separation; "
                "absolute separation alone is insufficient."
            )
        sampler = blending_kernel.sample_bias_signed
        sampler_separation = separation
    elif hasattr(blending_kernel, "sample_bias"):
        sampler = blending_kernel.sample_bias
        sampler_separation = np.abs(separation)
    elif callable(blending_kernel):
        sampler = blending_kernel
        sampler_separation = np.abs(separation)
    else:
        raise TypeError(
            "blending_kernel must provide sample_bias_signed(delta_v, "
            "f_secondary, u), sample_bias(abs_delta_v, f_secondary, u), or "
            "be callable with the legacy absolute-separation signature."
        )

    bias = np.asarray(
        sampler(sampler_separation, secondary_flux_fraction, blend_unit),
        dtype=float,
    )
    if bias.shape == ():
        bias = np.full_like(separation, float(bias), dtype=float)
    if bias.shape != separation.shape:
        raise ValueError("blending_kernel returned a bias array with the wrong shape.")
    if not np.all(np.isfinite(bias)):
        raise ValueError("blending_kernel returned non-finite RV bias values.")
    return bias
