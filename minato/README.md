# MINATO tutorials

Run tutorials from an installed development environment at the repository root.
Start with the small synthetic examples before using local spectra or large CPU
runs.

MINATO releases contain no atmosphere grids, trained models, spectra, fitted
results, or tutorial datasets. Tutorials must generate small synthetic inputs
during execution or ask for explicit paths to user-supplied resources. Public
TLUSTY and PoWR grids can be used after obtaining them separately and following
their own licence and citation guidance.

## Spectral analysis

1. [`span_example.ipynb`](tutorials/span_example.ipynb) - atmosphere fitting
   with `minato.span`, user-supplied model grids, and synthetic disentangled
   spectra.
2. [`working_with_atmfit_results.ipynb`](tutorials/working_with_atmfit_results.ipynb)
   - inspect and plot user-supplied atmosphere-fit results.
3. [`plot_model.ipynb`](tutorials/plot_model.ipynb) - inspect a model spectrum
   supplied by the user.

These legacy spectral-analysis notebooks are being revised to remove bundled
models, real example spectra, fixed result files, and machine-specific paths.
Their data-free execution is a `0.3.0` release gate.

## Radial velocities

1. [`ravel_SB1_rvs.ipynb`](tutorials/ravel_tutorial/ravel_SB1_rvs.ipynb) - SB1
   line fitting and radial velocities.
2. [`ravel_SB2_rvs.ipynb`](tutorials/ravel_tutorial/ravel_SB2_rvs.ipynb) - SB2
   fitting and diagnostics.
3. [`ravel_large_scale_batches.ipynb`](tutorials/ravel_tutorial/ravel_large_scale_batches.ipynb)
   - CPU-oriented batch execution.

The SB1 and SB2 notebooks still need output clean-up and a full release
validation pass. Release versions will generate their input spectra from a
small analytic synthetic example instead of reading bundled spectra.

## Synthetic spectra

1. [`create_synth_spectra.ipynb`](tutorials/create_synth_spectra.ipynb) -
   single-star, SB1, and SB2 rendering with a runnable analytic backend.
2. [`synthetic_spectra_ravel_bridge.ipynb`](tutorials/synthetic_spectra_ravel_bridge.ipynb)
   - write RAVEL-compatible synthetic observations.
3. [`isochrone_age_sampling.ipynb`](tutorials/isochrone_age_sampling.ipynb) -
   constrained coeval-age selection.

## Binary populations

1. [`binary_population_tutorial.ipynb`](tutorials/binary_population_tutorial.ipynb)
   - intrinsic draws, real and ideal cadence simulation, and inference choices.

For production inference, also read the
[`binary_population` module guide](binary_population/README.md), which documents
the validated averaged mixture-CRN workflow and CPU pool choices.

## Observing

1. [`observing_tutorial.ipynb`](tutorials/observing_tutorial.ipynb) - generate
   orbital-phase windows, assess target observability, and plot night-time
   airmass from a synthetic example.

The [`observing` module guide](observing/README.md) provides shorter API
examples and documents explicit output-file handling.

## Development-only external adaptation

The repository retains a class-based adaptation of
[Disentangling_Shift_And_Add](https://github.com/TomerShenar/Disentangling_Shift_And_Add)
under [`minato.contrib`](contrib/README.md). It is not part of the released
MINATO product, is not installed as `minato.spdis`, and is excluded from
releases. A `develop` checkout provides it as `minato.contrib.spdis`. Use the
upstream project for credit, citations, permissions, and supported usage. In
particular, cite Gonzalez & Levato (2006) for the shift-and-add algorithm and
Shenar et al. (2020, 2022) as requested by the upstream authors; do not cite
MINATO as the source of this method.
