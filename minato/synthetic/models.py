"""Data containers and backend interfaces for synthetic spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass
class Spectrum:
    """One in-memory spectrum on a one-dimensional wavelength grid."""

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.wavelength.ndim != 1 or self.flux.ndim != 1:
            raise ValueError("wavelength and flux must be one-dimensional arrays")
        if self.wavelength.size != self.flux.size:
            raise ValueError("wavelength and flux must have the same length")
        if self.wavelength.size < 2:
            raise ValueError("a spectrum needs at least two wavelength samples")
        if not np.all(np.diff(self.wavelength) > 0):
            raise ValueError("wavelength must be strictly increasing")
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=float)
            if self.error.shape != self.flux.shape:
                raise ValueError("error must have the same shape as flux")
        self.metadata = dict(self.metadata)

    def with_flux(
        self,
        flux: np.ndarray,
        *,
        error: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Spectrum":
        """Return a copy on the same wavelength grid with replacement flux."""

        next_metadata = dict(self.metadata)
        if metadata:
            next_metadata.update(metadata)
        return Spectrum(self.wavelength.copy(), flux, error=error, metadata=next_metadata)


@dataclass
class Star:
    """Physical parameters needed to render one stellar component."""

    teff: float
    logg: float
    radius: float = 1.0
    rv: float = 0.0
    vsini: float = 0.0
    flux_scale: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.teff <= 0:
            raise ValueError("teff must be positive")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.vsini < 0:
            raise ValueError("vsini must be non-negative")
        if self.flux_scale is not None and self.flux_scale < 0:
            raise ValueError("flux_scale must be non-negative")
        self.metadata = dict(self.metadata)

    @classmethod
    def from_mass(
        cls,
        mass: float,
        log_age: float,
        isochrone_bank: "IsochroneProvider",
        *,
        rv: float = 0.0,
        vsini: float = 0.0,
        flux_scale: float | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Star":
        """Create a star by interpolating mass and age through an isochrone bank."""

        params = isochrone_bank.interpolate(mass=mass, log_age=log_age)
        next_metadata = {"mass": float(mass), "log_age": float(log_age)}
        if metadata:
            next_metadata.update(metadata)
        return cls(
            teff=params.teff,
            logg=params.logg,
            radius=params.radius,
            rv=rv,
            vsini=vsini,
            flux_scale=flux_scale,
            label=label,
            metadata=next_metadata,
        )

    @property
    def light_weight(self) -> float:
        """Relative light weight used when combining normalised component spectra."""

        if self.flux_scale is not None:
            return float(self.flux_scale)
        return float(self.radius) ** 2


@dataclass
class BinarySystem:
    """A binary system rendered from one or two stellar components."""

    primary: Star
    secondary: Star | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata)

    @classmethod
    def from_masses(
        cls,
        m1: float,
        q: float,
        log_age: float,
        isochrone_bank: "IsochroneProvider",
        *,
        rv1: float = 0.0,
        rv2: float = 0.0,
        vsini1: float = 0.0,
        vsini2: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BinarySystem":
        """Create a binary from primary mass, mass ratio, and a shared age."""

        if m1 <= 0:
            raise ValueError("m1 must be positive")
        if q <= 0:
            raise ValueError("q must be positive")
        m2 = float(m1) * float(q)
        primary = Star.from_mass(
            m1,
            log_age,
            isochrone_bank,
            rv=rv1,
            vsini=vsini1,
            label="primary",
        )
        secondary = Star.from_mass(
            m2,
            log_age,
            isochrone_bank,
            rv=rv2,
            vsini=vsini2,
            label="secondary",
        )
        next_metadata = {"m1": float(m1), "q": float(q), "m2": m2, "log_age": float(log_age)}
        if metadata:
            next_metadata.update(metadata)
        return cls(primary=primary, secondary=secondary, metadata=next_metadata)


@dataclass
class ObservationModel:
    """Instrument and noise settings used during rendering."""

    resolving_power: float | None = None
    snr: float | None = None
    wavelength_min: float | None = None
    wavelength_max: float | None = None
    velocity_step: float | None = 5.0
    limb_darkening: float = 0.6
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.resolving_power is not None and self.resolving_power <= 0:
            raise ValueError("resolving_power must be positive")
        if self.snr is not None and self.snr <= 0:
            raise ValueError("snr must be positive")
        if self.velocity_step is not None and self.velocity_step <= 0:
            raise ValueError("velocity_step must be positive")
        if not 0 <= self.limb_darkening <= 1:
            raise ValueError("limb_darkening must be between 0 and 1")
        if (
            self.wavelength_min is not None
            and self.wavelength_max is not None
            and self.wavelength_min >= self.wavelength_max
        ):
            raise ValueError("wavelength_min must be smaller than wavelength_max")


@dataclass(frozen=True)
class IsochronePoint:
    """Interpolated stellar parameters from an isochrone bank."""

    teff: float
    logg: float
    radius: float


@runtime_checkable
class IsochroneProvider(Protocol):
    """Minimal protocol required by ``Star.from_mass``."""

    def interpolate(self, mass: float, log_age: float) -> IsochronePoint:
        """Return stellar parameters for one mass and log-age point."""


@runtime_checkable
class AtmosphereGrid(Protocol):
    """Minimal protocol for atmosphere-grid backends."""

    def get_spectrum(self, star: Star) -> Spectrum | tuple[np.ndarray, np.ndarray]:
        """Return a model spectrum for ``star``."""
