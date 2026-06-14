"""Small spectral operations used by the synthetic renderer."""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT_KMS = 299_792.458


def make_log_wavelength_grid(
    wavelength_min: float,
    wavelength_max: float,
    velocity_step: float,
) -> np.ndarray:
    """Return a wavelength grid with constant spacing in velocity."""

    if wavelength_min <= 0 or wavelength_max <= 0:
        raise ValueError("wavelength limits must be positive")
    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min must be smaller than wavelength_max")
    if velocity_step <= 0:
        raise ValueError("velocity_step must be positive")

    log_min = np.log(float(wavelength_min))
    log_max = np.log(float(wavelength_max))
    n_pix = int(np.floor((log_max - log_min) * SPEED_OF_LIGHT_KMS / velocity_step)) + 1
    if n_pix < 2:
        raise ValueError("wavelength range is too narrow for the requested velocity_step")
    return np.exp(log_min + np.arange(n_pix) * velocity_step / SPEED_OF_LIGHT_KMS)


def resample_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    target_wavelength: np.ndarray,
) -> np.ndarray:
    """Interpolate a spectrum onto ``target_wavelength`` using edge fill."""

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    target_wavelength = np.asarray(target_wavelength, dtype=float)
    return np.interp(target_wavelength, wavelength, flux, left=flux[0], right=flux[-1])


def doppler_shift(
    wavelength: np.ndarray,
    flux: np.ndarray,
    rv: float | np.ndarray,
) -> np.ndarray:
    """Apply a radial-velocity shift; positive RV moves features redward."""

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    velocities = np.atleast_1d(np.asarray(rv, dtype=float))
    shifted = np.empty((velocities.size, wavelength.size), dtype=float)

    for index, velocity in enumerate(velocities):
        rest_wavelength = wavelength / (1.0 + velocity / SPEED_OF_LIGHT_KMS)
        shifted[index] = np.interp(rest_wavelength, wavelength, flux, left=flux[0], right=flux[-1])

    if np.ndim(rv) == 0:
        return shifted[0]
    return shifted


def _velocity_step_from_log_grid(wavelength: np.ndarray) -> float:
    log_wavelength = np.log(np.asarray(wavelength, dtype=float))
    dlog = np.diff(log_wavelength)
    if not np.allclose(dlog, dlog[0], rtol=1e-5, atol=0):
        raise ValueError("wavelength must be sampled on a constant log-wavelength grid")
    return float(dlog[0] * SPEED_OF_LIGHT_KMS)


def rotational_broadening_kernel(
    vsini: float,
    velocity_step: float,
    limb_darkening: float = 0.6,
) -> np.ndarray:
    """Return a normalised linear-limb-darkening rotational kernel."""

    if vsini < 0:
        raise ValueError("vsini must be non-negative")
    if velocity_step <= 0:
        raise ValueError("velocity_step must be positive")
    if not 0 <= limb_darkening <= 1:
        raise ValueError("limb_darkening must be between 0 and 1")
    if vsini == 0:
        return np.array([1.0])

    half_width = int(np.ceil(vsini / velocity_step))
    if half_width == 0:
        return np.array([1.0])
    velocity = np.arange(-half_width, half_width + 1, dtype=float) * velocity_step
    x = velocity / float(vsini)
    kernel = np.zeros_like(x)
    inside = np.abs(x) <= 1.0
    x_inside = x[inside]
    kernel[inside] = (
        2.0 * (1.0 - limb_darkening) * np.sqrt(np.clip(1.0 - x_inside**2, 0.0, None))
        + 0.5 * np.pi * limb_darkening * (1.0 - x_inside**2)
    )
    total = kernel.sum()
    if total <= 0:
        return np.array([1.0])
    return kernel / total


def _convolve_with_edge_padding(flux: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 1:
        return np.asarray(flux, dtype=float).copy()
    half_width = kernel.size // 2
    padded = np.pad(np.asarray(flux, dtype=float), half_width, mode="edge")
    return np.convolve(padded, kernel, mode="same")[half_width:-half_width]


def apply_rotational_broadening(
    wavelength: np.ndarray,
    flux: np.ndarray,
    vsini: float,
    limb_darkening: float = 0.6,
) -> np.ndarray:
    """Convolve a log-wavelength spectrum with a rotational kernel."""

    if vsini == 0:
        return np.asarray(flux, dtype=float).copy()
    velocity_step = _velocity_step_from_log_grid(wavelength)
    kernel = rotational_broadening_kernel(vsini, velocity_step, limb_darkening)
    return _convolve_with_edge_padding(flux, kernel)


def apply_instrumental_broadening(
    wavelength: np.ndarray,
    flux: np.ndarray,
    resolving_power: float | None,
) -> np.ndarray:
    """Convolve a log-wavelength spectrum to a constant resolving power."""

    if resolving_power is None:
        return np.asarray(flux, dtype=float).copy()
    if resolving_power <= 0:
        raise ValueError("resolving_power must be positive")

    velocity_step = _velocity_step_from_log_grid(wavelength)
    fwhm_velocity = SPEED_OF_LIGHT_KMS / float(resolving_power)
    sigma_pixels = fwhm_velocity / (2.355 * velocity_step)
    half_width = int(np.ceil(4.0 * sigma_pixels))
    if half_width == 0:
        return np.asarray(flux, dtype=float).copy()
    pixel = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (pixel / sigma_pixels) ** 2)
    kernel /= kernel.sum()
    return _convolve_with_edge_padding(flux, kernel)


def add_noise(
    flux: np.ndarray,
    snr: float | None,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Add reproducible Gaussian noise and return ``(noisy_flux, error)``."""

    flux = np.asarray(flux, dtype=float)
    if snr is None:
        return flux.copy(), None
    if snr <= 0:
        raise ValueError("snr must be positive")
    if rng is None:
        rng = np.random.default_rng(seed)

    continuum_scale = float(np.nanmedian(np.abs(flux)))
    if not np.isfinite(continuum_scale) or continuum_scale <= 0:
        continuum_scale = 1.0
    sigma = continuum_scale / float(snr)
    error = np.full_like(flux, sigma, dtype=float)
    return flux + rng.normal(0.0, sigma, size=flux.shape), error
