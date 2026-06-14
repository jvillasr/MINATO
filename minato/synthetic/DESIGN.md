# Synthetic Spectra Design Note

`minato.synthetic` provides generic, in-memory primitives for constructing
synthetic spectra before passing them to tools such as RAVEL. The core renderer
does not encode AP18/PoWR routing, MIST age priors, SDSS paths, BOSS noise
models, or paper-specific catalogue layouts.

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
- `TextAtmosphereGrid` is a small convenience adapter for users with folders of
  text atmosphere models. It can scan directories using the MINATO convention
  (`teff25000_logg4.00.txt`) or common PoWR, TLUSTY, and FASTWIND-style
  filename patterns, and it supports regex/parser/index fallbacks for local
  naming schemes.

## Text Atmosphere Grids

The simplest path is:

```python
from minato.synthetic import TextAtmosphereGrid

grid = TextAtmosphereGrid.from_directory("models/", format="auto")
```

If auto-detection fails, users can:

- pass `format="powr"`, `format="tlusty"`, or `format="fastwind"`;
- pass `filename_pattern=...` with named groups such as `teff`, `teff_kk`,
  `logg`, `logg10`, or `logg100`;
- pass a parser function that returns `{"teff": ..., "logg": ...}`;
- run `TextAtmosphereGrid.write_index_template(...)`, fill in the CSV, and
  load it with `TextAtmosphereGrid.from_index(...)`;
- symlink or rename files to the MINATO convention, for example
  `teff25000_logg4.00.txt`.

This adapter recognises model-grid files and loads wavelength/flux columns. It
does not decide whether PoWR, TLUSTY, FASTWIND, or any other grid is physically
appropriate for a given star.

For overlapping or partial grid coverage, compose several user-supplied grids
with explicit priority:

```python
from minato.synthetic import FallbackAtmosphereGrid, TextAtmosphereGrid

powr = TextAtmosphereGrid.from_directory(
    "powr_models/",
    format="powr",
    max_teff_delta=800,
    max_logg_delta=0.25,
)
tlusty = TextAtmosphereGrid.from_directory(
    "tlusty_models/",
    format="tlusty",
    max_teff_delta=1000,
    max_logg_delta=0.25,
)
grid = FallbackAtmosphereGrid([("powr", powr), ("tlusty", tlusty)])
```

MINATO tries each backend in order and falls through only when a backend raises
`LookupError`, for example because no node is close enough in `(teff, logg)`.
The user sets the order and tolerance values, and the returned spectrum records
the selected grid in metadata.

## Left Outside The Core

- AP18/PoWR selection rules and fallback thresholds.
- MIST-specific coeval-age priors and logg/temperature guards.
- BOSS or SDSS S/N distributions, apparent-magnitude scaling, filenames, and
  FITS header policy.
- Batch HDF5/FITS production for a specific survey or paper.

Those policies should live in project adapters that call `minato.synthetic`
with explicit atmosphere and isochrone backends.
