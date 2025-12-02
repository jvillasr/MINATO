## [Unreleased]
### Added
* Probabilistic SB1 fitting workflow (`fit_sb1_probmod`) that mirrors the SB2 logic, including SB1-specific line plotting, corner-plot generation, and CSV writers.
* Tools for diagnosing multimodal SB2 posteriors: HDI / mode summaries, per-component RV corner plots, Chi^2 comparison plots, and the ability to stitch together epoch-wise best fits from sequential MCMC runs.
* `profile` and `sigma_prior` parameters in `fit_sb2_probmod` / `SLfit` so users can choose Gaussian vs. Voigt profiles and tune the RV priors explicitly.
* New `binary_population` subpackage:
  * `BinaryPopulation` for intrinsic draws (masses, periods, eccentricities) with Roche/smear guards and fixed-value overrides.
  * `BinarySurveySimulator` for survey cadence/noise and RV time-series synthesis.
  * `run_mcmc` helper to fit the binary fraction via Poisson likelihood on `dRV_max` (importable as `from minato.binary_population import run_mcmc`).

### Changed
* SB2 workflow now runs a two-stage sampling procedure (full fit + RV-only refit with frozen nuisance parameters) and records which posterior summary (median / mode / HDI) was used per epoch in `fit_values.csv`.
* `SLfit` logic, docstrings, and plotting have been streamlined; both SB1 and SB2 runs now share the same probabilistic infrastructure and optionally create diagnostic corner plots only for the recovered RVs.
* `plot_lines_fit` / `plot_lines_fit_sb1` visualize both MCMC runs, Chi^2 statistics, and stitched RV solutions. Error reporting in `mcmc_results_to_file*` is more robust to bimodality.
* Lomb–Scargle utilities read SB1 products directly, handle star names without underscores, and clean up KDE imports / plotting labels.
* Restored the earlier `setup_star_directory_and_save_jds` behaviour so SB2 output directories are laid out consistently again.

### Fixed
* Plotting now respects the exact number of SB1 fits requested.
* `mcmc_results_to_file_sb1` correctly treats `rm_epochs=None`, preventing crashes when no epochs are removed.
* General small cleanups (missing imports, stale comments, path handling) uncovered while merging RVfitting into develop.

## [0.2.0] – 2025‑04‑17
### Added
* New `ravel` module:
  * Refactored code from the one used in Villaseñor+21
  * Spectral‑line Gaussian/Lorentzian fitting (SB1 & SB2)
  * Radial‑velocity computation
  * Lomb‑Scargle periodogram analysis
