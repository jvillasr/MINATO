# Synthetic Spectra Design Note

`minato.synthetic` provides generic, in-memory primitives for constructing
synthetic spectra before passing them to tools such as RAVEL. The core package
does not know about AP18, PoWR, MIST, SDSS, BOSS filenames, or paper-specific
catalogue layouts.

## Generic Core

- `Star`, `BinarySystem`, `ObservationModel`, and `Spectrum` are lightweight
  containers.
- Atmosphere grids are backend interfaces: any object with
  `get_spectrum(star)` can provide a `Spectrum` or `(wavelength, flux)` tuple.
- Isochrones are optional. `IsochroneBank` reads a simple CSV-bank format with
  `mass_init`, `teff`, `logg`, and `radius`, while `Star.from_mass` and
  `BinarySystem.from_masses` accept any object implementing the same
  interpolation interface.
- Rendering is deliberately composed from small steps: resampling to a
  log-wavelength grid, rotational broadening, instrumental broadening, RV
  shifting, component weighting, binary summation, and seeded noise injection.
- Outputs are `Spectrum` objects first. File writers such as
  `write_ravel_txt` are bridges, not the primary API.

## Left Outside The Core

- AP18/PoWR selection rules and fallback thresholds.
- MIST-specific coeval-age priors and logg/temperature guards.
- BOSS or SDSS S/N distributions, apparent-magnitude scaling, filenames, and
  FITS header policy.
- Batch HDF5/FITS production for a specific survey or paper.

Those policies should live in project adapters that call `minato.synthetic`
with explicit atmosphere and isochrone backends.
