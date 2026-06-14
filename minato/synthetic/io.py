"""File writers for synthetic spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Spectrum


def write_ravel_txt(spectrum: Spectrum, path: str | Path) -> Path:
    """
    Write a spectrum in the whitespace-delimited text format read by RAVEL.

    The output columns are wavelength and flux, plus flux error when the
    spectrum carries one. RAVEL estimates errors itself when the file has only
    two columns.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if spectrum.error is None:
        data = np.column_stack([spectrum.wavelength, spectrum.flux])
    else:
        data = np.column_stack([spectrum.wavelength, spectrum.flux, spectrum.error])
    np.savetxt(output_path, data, fmt="%.10e")
    return output_path
