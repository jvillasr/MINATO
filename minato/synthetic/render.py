"""Rendering functions for single-star and binary synthetic spectra."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .models import (
    AtmosphereGrid,
    BinarySystem,
    ObservationModel,
    RenderedAtmosphereGrid,
    Spectrum,
    Star,
)
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


def render_atmosphere_grid(
    atmosphere_grid: AtmosphereGrid
    | Callable[[Star], Spectrum | tuple[np.ndarray, np.ndarray]],
    atmosphere_nodes: Iterable[tuple[float, float]],
    vsini_values: Iterable[float],
    observation: ObservationModel | None = None,
    *,
    exact_nodes: bool = True,
    validate_normalised: bool = True,
) -> RenderedAtmosphereGrid:
    """
    Render an exact atmosphere grid over temperature, gravity, and rotation.

    Every output model shares one wavelength grid and is rendered through the
    same resampling and broadening path as :func:`render_single_star`. The
    returned grid remains in memory and contains no noise or radial-velocity
    shift, making it suitable for model fitting.

    Parameters
    ----------
    atmosphere_grid
        Backend defining ``get_spectrum(star)`` or a callable returning a
        :class:`Spectrum`.
    atmosphere_nodes
        Requested ``(teff_K, logg)`` nodes.
    vsini_values
        Projected rotational velocities in km/s.
    observation
        Wavelength sampling, instrumental resolution, and limb darkening.
        ``snr`` must be ``None`` because fitting models must remain noiseless.
    exact_nodes
        Require backend metadata to confirm that every requested atmosphere
        node exists exactly rather than accepting a nearest neighbour.
    validate_normalised
        Require finite source fluxes with a median between 0.5 and 1.5. This
        catches calibrated or logarithmic spectra supplied accidentally to a
        continuum-normalised fitting workflow. Disable only when another flux
        convention is intentional.

    Returns
    -------
    RenderedAtmosphereGrid
        In-memory spectra keyed by ``(teff, logg, vsini)``.
    """

    observation = observation or ObservationModel()
    if observation.snr is not None:
        raise ValueError("rendered atmosphere grids must not include noise")

    nodes = sorted({(float(teff), float(logg)) for teff, logg in atmosphere_nodes})
    rotations = sorted({float(vsini) for vsini in vsini_values})
    if not nodes:
        raise ValueError("atmosphere_nodes must contain at least one node")
    if not rotations:
        raise ValueError("vsini_values must contain at least one value")
    if any(not np.isfinite(value) or value < 0 for value in rotations):
        raise ValueError("vsini_values must be finite and non-negative")

    source_spectra: dict[tuple[float, float], Spectrum] = {}
    for teff, logg in nodes:
        spectrum = _fetch_atmosphere(atmosphere_grid, Star(teff=teff, logg=logg))
        if validate_normalised:
            if not np.all(np.isfinite(spectrum.flux)):
                raise ValueError(
                    f"atmosphere node Teff={teff:g} K, logg={logg:g} "
                    "contains non-finite fluxes"
                )
            median_flux = float(np.median(spectrum.flux))
            if not 0.5 <= median_flux <= 1.5:
                source_path = spectrum.metadata.get("source_path", "unknown source")
                raise ValueError(
                    "Atmosphere spectrum does not look continuum-normalised.\n"
                    f"Node: Teff={teff:g} K, logg={logg:g}; "
                    f"median flux={median_flux:g} (expected near 1).\n"
                    f"Source: {source_path}\n"
                    "Use normalised spectra, or set validate_normalised=False "
                    "if this flux scale is intentional."
                )
        if exact_nodes:
            selected = spectrum.metadata.get("atmosphere_node", {})
            selected_teff = selected.get("teff")
            selected_logg = selected.get("logg")
            if selected_teff is None or selected_logg is None:
                raise ValueError(
                    "exact node validation requires atmosphere_node metadata"
                )
            if not np.isclose(float(selected_teff), teff) or not np.isclose(
                float(selected_logg), logg
            ):
                raise LookupError(
                    f"requested atmosphere node Teff={teff:g} K, logg={logg:g} "
                    f"is unavailable; nearest node is Teff={float(selected_teff):g} K, "
                    f"logg={float(selected_logg):g}"
                )
        source_spectra[(teff, logg)] = spectrum

    wavelength = _output_grid(list(source_spectra.values()), observation)
    rendered: dict[tuple[float, float, float], Spectrum] = {}
    for (teff, logg), source in source_spectra.items():
        for vsini in rotations:
            star = Star(teff=teff, logg=logg, vsini=vsini)
            flux = _render_component(star, source, observation, wavelength)
            metadata = dict(source.metadata)
            metadata.update(
                {
                    "kind": "rendered_atmosphere_model",
                    "teff": teff,
                    "logg": logg,
                    "vsini": vsini,
                    "resolving_power": observation.resolving_power,
                    "velocity_step": observation.velocity_step,
                }
            )
            rendered[(teff, logg, vsini)] = Spectrum(
                wavelength,
                flux,
                metadata=metadata,
            )

    return RenderedAtmosphereGrid(
        rendered,
        metadata={
            "resolving_power": observation.resolving_power,
            "velocity_step": observation.velocity_step,
            "wavelength_min": float(wavelength[0]),
            "wavelength_max": float(wavelength[-1]),
            "limb_darkening": observation.limb_darkening,
            "exact_nodes": bool(exact_nodes),
            "validated_normalised": bool(validate_normalised),
        },
    )


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
