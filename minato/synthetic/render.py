"""Rendering functions for single-star and binary synthetic spectra."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .models import AtmosphereGrid, BinarySystem, ObservationModel, Spectrum, Star
from .physics import (
    add_noise,
    apply_instrumental_broadening,
    apply_rotational_broadening,
    doppler_shift,
    make_log_wavelength_grid,
    resample_spectrum,
)


def _coerce_spectrum(raw: Spectrum | tuple[np.ndarray, np.ndarray]) -> Spectrum:
    if isinstance(raw, Spectrum):
        return raw
    if isinstance(raw, tuple) and len(raw) >= 2:
        return Spectrum(raw[0], raw[1])
    raise TypeError("atmosphere grid must return Spectrum or (wavelength, flux)")


def _fetch_atmosphere(
    atmosphere_grid: AtmosphereGrid | Callable[[Star], Spectrum | tuple[np.ndarray, np.ndarray]],
    star: Star,
) -> Spectrum:
    if hasattr(atmosphere_grid, "get_spectrum"):
        raw = atmosphere_grid.get_spectrum(star)  # type: ignore[union-attr]
    elif callable(atmosphere_grid):
        raw = atmosphere_grid(star)
    else:
        raise TypeError("atmosphere_grid must define get_spectrum(star) or be callable")
    return _coerce_spectrum(raw)


def _limits_for_spectra(
    spectra: list[Spectrum],
    observation: ObservationModel,
) -> tuple[float, float]:
    lower = max(float(spec.wavelength[0]) for spec in spectra)
    upper = min(float(spec.wavelength[-1]) for spec in spectra)
    if observation.wavelength_min is not None:
        lower = max(lower, float(observation.wavelength_min))
    if observation.wavelength_max is not None:
        upper = min(upper, float(observation.wavelength_max))
    if lower >= upper:
        raise ValueError("requested wavelength range does not overlap the atmosphere spectra")
    return lower, upper


def _output_grid(
    spectra: list[Spectrum],
    observation: ObservationModel,
) -> np.ndarray:
    lower, upper = _limits_for_spectra(spectra, observation)
    if observation.velocity_step is None:
        base = spectra[0].wavelength
        mask = (base >= lower) & (base <= upper)
        grid = base[mask]
        if grid.size < 2:
            raise ValueError("output wavelength range contains fewer than two samples")
        return grid.copy()
    return make_log_wavelength_grid(lower, upper, observation.velocity_step)


def _render_component(
    star: Star,
    spectrum: Spectrum,
    observation: ObservationModel,
    wavelength: np.ndarray,
) -> np.ndarray:
    flux = resample_spectrum(spectrum.wavelength, spectrum.flux, wavelength)

    if observation.velocity_step is not None:
        flux = apply_rotational_broadening(
            wavelength,
            flux,
            star.vsini,
            limb_darkening=observation.limb_darkening,
        )
        flux = apply_instrumental_broadening(wavelength, flux, observation.resolving_power)
    elif star.vsini != 0 or observation.resolving_power is not None:
        raise ValueError(
            "rotational and instrumental broadening require ObservationModel.velocity_step"
        )

    if star.rv != 0:
        flux = doppler_shift(wavelength, flux, star.rv)
    return flux


def render_single_star(
    star: Star,
    atmosphere_grid: AtmosphereGrid | Callable[[Star], Spectrum | tuple[np.ndarray, np.ndarray]],
    observation: ObservationModel | None = None,
) -> Spectrum:
    """Render one stellar spectrum into memory."""

    observation = observation or ObservationModel()
    base_spectrum = _fetch_atmosphere(atmosphere_grid, star)
    wavelength = _output_grid([base_spectrum], observation)
    flux = _render_component(star, base_spectrum, observation, wavelength)
    if star.flux_scale is not None:
        flux = flux * float(star.flux_scale)
    flux, error = add_noise(flux, observation.snr, seed=observation.seed)
    metadata: dict[str, Any] = {
        "kind": "single_star",
        "star": star.label,
        "teff": float(star.teff),
        "logg": float(star.logg),
        "radius": float(star.radius),
        "rv": float(star.rv),
        "vsini": float(star.vsini),
        "resolving_power": observation.resolving_power,
        "snr": observation.snr,
    }
    return Spectrum(wavelength, flux, error=error, metadata=metadata)


def render_binary(
    system: BinarySystem,
    atmosphere_grid: AtmosphereGrid | Callable[[Star], Spectrum | tuple[np.ndarray, np.ndarray]],
    observation: ObservationModel | None = None,
    *,
    normalise: bool = True,
) -> Spectrum:
    """
    Render a binary composite.

    By default component light weights are normalised before summing, which is
    useful for continuum-normalised atmosphere spectra. Set ``normalise=False``
    to retain absolute component scaling.
    """

    observation = observation or ObservationModel()
    if system.secondary is None:
        return render_single_star(system.primary, atmosphere_grid, observation)

    primary_spectrum = _fetch_atmosphere(atmosphere_grid, system.primary)
    secondary_spectrum = _fetch_atmosphere(atmosphere_grid, system.secondary)
    wavelength = _output_grid([primary_spectrum, secondary_spectrum], observation)
    primary_flux = _render_component(system.primary, primary_spectrum, observation, wavelength)
    secondary_flux = _render_component(system.secondary, secondary_spectrum, observation, wavelength)

    primary_weight = system.primary.light_weight
    secondary_weight = system.secondary.light_weight
    if normalise:
        total = primary_weight + secondary_weight
        if total <= 0:
            raise ValueError("component light weights must sum to a positive value")
        primary_weight /= total
        secondary_weight /= total

    composite = primary_weight * primary_flux + secondary_weight * secondary_flux
    composite, error = add_noise(composite, observation.snr, seed=observation.seed)
    metadata: dict[str, Any] = {
        "kind": "binary",
        "primary_weight": float(primary_weight),
        "secondary_weight": float(secondary_weight),
        "normalised_weights": bool(normalise),
        "primary_rv": float(system.primary.rv),
        "secondary_rv": float(system.secondary.rv),
        "resolving_power": observation.resolving_power,
        "snr": observation.snr,
    }
    metadata.update(system.metadata)
    return Spectrum(wavelength, composite, error=error, metadata=metadata)
