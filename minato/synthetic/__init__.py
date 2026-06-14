"""Generic synthetic-spectrum rendering tools for MINATO."""

from .age_sampling import (
    IsochroneAgeCandidate,
    IsochroneAgeSampler,
    IsochroneAgeSamplingError,
    LoggSkewWeight,
    StellarConstraints,
)
from .grids import AtmosphereGridNode, FallbackAtmosphereGrid, TextAtmosphereGrid
from .isochrones import IsochroneBank
from .models import (
    AtmosphereGrid,
    BinarySystem,
    IsochronePoint,
    IsochroneProvider,
    ObservationModel,
    Spectrum,
    Star,
)
from .physics import (
    SPEED_OF_LIGHT_KMS,
    add_noise,
    apply_instrumental_broadening,
    apply_rotational_broadening,
    doppler_shift,
    make_log_wavelength_grid,
    resample_spectrum,
    rotational_broadening_kernel,
)
from .render import render_binary, render_single_star

__all__ = [
    "AtmosphereGrid",
    "AtmosphereGridNode",
    "BinarySystem",
    "FallbackAtmosphereGrid",
    "IsochroneAgeCandidate",
    "IsochroneAgeSampler",
    "IsochroneAgeSamplingError",
    "IsochroneBank",
    "IsochronePoint",
    "IsochroneProvider",
    "LoggSkewWeight",
    "ObservationModel",
    "SPEED_OF_LIGHT_KMS",
    "Spectrum",
    "Star",
    "StellarConstraints",
    "TextAtmosphereGrid",
    "add_noise",
    "apply_instrumental_broadening",
    "apply_rotational_broadening",
    "doppler_shift",
    "make_log_wavelength_grid",
    "render_binary",
    "render_single_star",
    "resample_spectrum",
    "rotational_broadening_kernel",
]
