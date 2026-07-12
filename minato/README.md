# MINATO tutorials

Run tutorials from an installed development environment at the repository root.
Start with the small synthetic examples before using local spectra or large CPU
runs.

## Spectral analysis

1. [`span_example.ipynb`](tutorials/span_example.ipynb) - atmosphere fitting
   with `minato.span`.
2. [`working_with_atmfit_results.ipynb`](tutorials/working_with_atmfit_results.ipynb)
   - inspect and plot atmosphere-fit results.
3. [`plot_model.ipynb`](tutorials/plot_model.ipynb) - inspect model spectra.

## Radial velocities

1. [`ravel_SB1_rvs.ipynb`](tutorials/ravel_tutorial/ravel_SB1_rvs.ipynb) - SB1
   line fitting and radial velocities.
2. [`ravel_SB2_rvs.ipynb`](tutorials/ravel_tutorial/ravel_SB2_rvs.ipynb) - SB2
   fitting and diagnostics.
3. [`ravel_large_scale_batches.ipynb`](tutorials/ravel_tutorial/ravel_large_scale_batches.ipynb)
   - CPU-oriented batch execution.

The SB1 and SB2 notebooks still need output clean-up and a full release
validation pass.

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

## Experimental modules

`minato.spdis` is importable from an installed package but still needs focused
tests and a clean synthetic tutorial before it can move beyond experimental
status.
