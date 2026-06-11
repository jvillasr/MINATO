## [Unreleased]
### Added
* [2026-06-11] `binary_population.run_mcmc` can now fit `f_bin`, `pi`, `kappa`, and `eta` together via the opt-in `parameter_names` interface, while retaining the previous one-parameter `f_bin` default.
* [2026-06-11] `BinaryPopulation` now supports `logP_powerlaw_mode="direct"` for literature-style `p(log10 P) ∝ (log10 P)^pi` sampling over positive log-period intervals.
* [2026-05-05] Dedicated Na-doublet probabilistic fitter (`sb1_method='na'`) for per-epoch Na I D diagnostic RVs, using one shared local window and tied D1/D2 physics.
* [2026-05-05] Added `Na I 5896 (D1)` and the Paschen series entries used by the P117 workflow to the default `ravel` line dictionary.
* Probabilistic SB1 fitting workflow (`fit_sb1_probmod`) that mirrors the SB2 logic, including SB1-specific line plotting, corner-plot generation, and CSV writers.
* Tools for diagnosing multimodal SB2 posteriors: HDI / mode summaries, per-component RV corner plots, Chi^2 comparison plots, and the ability to stitch together epoch-wise best fits from sequential MCMC runs.
* `profile` and `sigma_prior` parameters in `fit_sb2_probmod` / `SLfit` so users can choose Gaussian vs. Voigt profiles and tune the RV priors explicitly.
* New `binary_population` subpackage:
  * `BinaryPopulation` for intrinsic draws (masses, periods, eccentricities) with Roche/smear guards and fixed-value overrides.
  * `BinarySurveySimulator` for survey cadence/noise and RV time-series synthesis.
  * `run_mcmc` helper to fit the binary fraction via Poisson likelihood on `dRV_max` (importable as `from minato.binary_population import run_mcmc`).
* New tutorial notebook for `minato.binary_population`: `minato/tutorials/binary_population_tutorial.ipynb`.
* Added `pyproject.toml` (initial packaging metadata) to support `uv`/`pip` workflows.
* Added missing dependencies for `ravel` (`corner`, `exojax`) to `minato_env.yml` and `pyproject.toml`.

### Changed
* [2026-05-05] P117 four-fit validation now uses Gaussian profiles for non-H stellar lines, Lorentzian profiles for H/Paschen lines, and the dedicated Na-doublet model for Na diagnostics.
* [2026-05-05] The dedicated Na-doublet diagnostic now defaults to the narrowed `5882-5905 Å` fitting window used in validation.
* [2026-05-05] Probabilistic SB1 outputs now preserve asymmetric posterior RV errors as `rv_err_minus` and `rv_err_plus` while retaining the legacy collapsed `mean_rv_er` column.
* `AGENTS.md` and `ROADMAP.md` are now tracked on `develop`, and the repository instructions document the release workflow for keeping development-only project-management files out of `main`.
* SB2 workflow now runs a two-stage sampling procedure (full fit + RV-only refit with frozen nuisance parameters) and records which posterior summary (median / mode / HDI) was used per epoch in `fit_values.csv`.
* `SLfit` logic, docstrings, and plotting have been streamlined; both SB1 and SB2 runs now share the same probabilistic infrastructure and optionally create diagnostic corner plots only for the recovered RVs.
* `plot_lines_fit` / `plot_lines_fit_sb1` visualize both MCMC runs, Chi^2 statistics, and stitched RV solutions. Error reporting in `mcmc_results_to_file*` is more robust to bimodality.
* Lomb–Scargle utilities read SB1 products directly, handle star names without underscores, and clean up KDE imports / plotting labels.
* Restored the earlier `setup_star_directory_and_save_jds` behaviour so SB2 output directories are laid out consistently again.
* `binary_population` inference: `run_mcmc` defaults to summary-only mock simulations (lighter outputs), vectorizes the Poisson log-likelihood, and supports multi-core pooling (`pool_kind="process"` default, configurable `start_method`).
* The SB1 and SB2 `ravel` tutorials now include an upfront note on setting `XLA_FLAGS` before importing `minato.ravel` on shared CPU servers.

### Fixed
* [2026-05-05] SB1 fit plots no longer swap epoch and line axes when `n_epochs == n_lines`; plots now draw a coherent high-likelihood posterior sample.
* [2026-05-05] Probabilistic SB1/Na/SB2 interpolation windows now drop non-finite or non-positive-error points before interpolation, avoiding red/NIR collapse from bad pixels.
* [2026-05-05] SB1 probabilistic fitting now samples and plots only the width parameters used by the selected profile family, avoiding unused Gaussian/Lorentzian width dimensions in corner plots.
* Plotting now respects the exact number of SB1 fits requested.
* `SLfit` now allows single-epoch runs and no longer trips over one-panel plotting in the probabilistic SB1/SB2 plotting helpers.
* `mcmc_results_to_file_sb1` correctly treats `rm_epochs=None`, preventing crashes when no epochs are removed.
* `binary_population` inference: fixed `run_mcmc` passing internal bookkeeping kwargs into the survey simulator.
* `binary_population` inference: `LogProb` is now a pickle-friendly callable to support multiprocessing pools reliably.
* `binary_population` outputs: added `n_epochs` alias for `n_eps` in survey products.
* `binary_population` draws: reduced noisy Roche-clamp reporting during MCMC runs and addressed pandas concat/fillna FutureWarnings.
* General small cleanups (missing imports, stale comments, path handling) uncovered while merging RVfitting into develop.

## [0.2.0] – 2025‑04‑17
### Added
* New `ravel` module:
  * Refactored code from the one used in Villaseñor+21
  * Spectral‑line Gaussian/Lorentzian fitting (SB1 & SB2)
  * Radial‑velocity computation
  * Lomb‑Scargle periodogram analysis
