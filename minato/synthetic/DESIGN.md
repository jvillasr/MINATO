# Synthetic Spectra Design Note

`minato.synthetic` provides generic, in-memory primitives for constructing
synthetic spectra before passing them to tools such as RAVEL or SPAN. The core
renderer does not encode AP18/PoWR routing, MIST age priors, SDSS paths, BOSS
noise models, or paper-specific catalogue layouts.

## Generic Core

- `Star`, `BinarySystem`, `ObservationModel`, and `Spectrum` are lightweight
  containers.
- Atmosphere grids are backend interfaces: any object with
  `get_spectrum(star)` can provide a `Spectrum` or `(wavelength, flux)` tuple.
- Isochrones are optional. `IsochroneBank` reads a simple CSV-bank format with
  `mass_init`, `teff`, `logg`, and `radius`, while `Star.from_mass` and
  `BinarySystem.from_masses` accept any object implementing the same
  interpolation interface.
- `IsochroneAgeSampler` is an explicit population-to-spectrum helper for
  selecting a coeval age before calling `Star.from_mass` or
  `BinarySystem.from_masses`. It can validate fixed ages, sample uniformly from
  valid isochrone slices, or sample with user-provided weights.
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
- pass `file_filter=...` when one directory contains several products, such as
  normalised and calibrated spectra;
- run `TextAtmosphereGrid.write_index_template(...)`, fill in the CSV, and
  load it with `TextAtmosphereGrid.from_index(...)`;
- symlink or rename files to the MINATO convention, for example
  `teff25000_logg4.00.txt`.

This adapter recognises model-grid files and loads wavelength/flux columns. It
does not decide whether PoWR, TLUSTY, FASTWIND, or any other grid is physically
appropriate for a given star. It rejects duplicate `(teff, logg)` entries
because selecting one of several files silently would make the grid ambiguous.

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

## Rendered Fitting Grids

`render_atmosphere_grid` renders an exact set of temperature, gravity, and
rotation nodes onto one common wavelength grid. It uses the same physical
operations as `render_single_star`, rejects noise, and validates a
continuum-normalised flux scale by default before returning a
`RenderedAtmosphereGrid` in memory:

```python
from minato.synthetic import ObservationModel, render_atmosphere_grid

rendered = render_atmosphere_grid(
    grid,
    atmosphere_nodes=[(32_000, 4.0), (22_000, 4.2)],
    vsini_values=[50, 75, 100, 125],
    observation=ObservationModel(
        resolving_power=40_000,
        wavelength_min=3950,
        wavelength_max=4600,
        velocity_step=2.5,
    ),
)
```

SPAN can consume these models directly, without copied atmosphere files or a
second broadening implementation:

```python
from minato.span import AtmFit

fit = AtmFit(
    "primary.txt",
    "secondary.txt",
    grid=fit_grid,
    lrat0=0.10,
    binary=True,
    wavelength_shift=0.0,
    modelsA_grid=rendered,
    modelsB_grid=rendered,
)
results = fit.compute_chi2([4102, 4340, 4471], [4102, 4340, 4471])
```

Pass different rendered grids as `modelsA_grid` and `modelsB_grid` when the
components require different atmosphere families or compositions. The legacy
`modelsA_path` and `modelsB_path` interface remains available for existing
precomputed SPAN grids. Rendered SPAN grids may contain irregular
`(Teff, log g)` nodes, allowing physical boundaries such as PoWR's
temperature-dependent maximum gravity without inventing unavailable Cartesian
combinations.

## Isochrone Age Sampling

Age selection is a separate step from binary-orbit drawing and spectrum
rendering. By default, MINATO does not impose a temperature, gravity, radius, or
evolutionary-state prior:

```python
from minato.synthetic import IsochroneAgeSampler

sampler = IsochroneAgeSampler()
log_age, age_meta = sampler.sample(isochrone_bank=iso, m1=8.0, m2=5.6, rng=rng)
```

Users can define generic main-sequence cuts with `StellarConstraints`:

```python
from minato.synthetic import IsochroneAgeSampler, StellarConstraints

main_sequence = IsochroneAgeSampler(
    primary=StellarConstraints(logg_min=3.5),
    secondary=StellarConstraints(logg_min=3.5),
)
log_age, age_meta = main_sequence.sample(iso, m1=8.0, m2=5.6, rng=rng)
```

Hot-star or OB-specific policies should be passed explicitly by the project
that needs them:

```python
from minato.synthetic import BinarySystem
from minato.synthetic import IsochroneAgeSampler, LoggSkewWeight, StellarConstraints

hot_star = IsochroneAgeSampler(
    primary=StellarConstraints(teff_min=10_000.0, logg_min=3.0),
    secondary=StellarConstraints(logg_min=3.0, logg_max=5.5),
    secondary_low_mass_teff_max={"mass_max": 8.0, "teff_max": 20_000.0},
    require_primary_logg_lte_secondary=True,
    weight=LoggSkewWeight(mu=4.0, sigma_lo=0.25, sigma_hi=0.12),
)
log_age, age_meta = hot_star.sample(iso, m1=19.28, m2=19.05, rng=rng)
system = BinarySystem.from_masses(19.28, 19.05 / 19.28, log_age, iso)
```

The returned metadata records the selected age, the number of valid ages, the
constraints used, and the selected primary/secondary `teff`, `logg`, and
`radius`.

## Left Outside The Core

- AP18/PoWR selection rules and fallback thresholds.
- MIST-specific coeval-age priors and paper-specific logg/temperature guards.
- BOSS or SDSS S/N distributions, apparent-magnitude scaling, filenames, and
  FITS header policy.
- Batch HDF5/FITS production for a specific survey or paper.

Those policies should live in project adapters that call `minato.synthetic`
with explicit atmosphere and isochrone backends.
