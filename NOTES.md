# MINATO Notes

## 2026-07-15 - Restore SPAN's standard result plots

Implementation:
- Replaced the tutorial-specific one-dimensional score figure with SPAN's
  established `read_results.compute_bestfit` visualisation.
- The standard figure now marks the profile minimum at every parameter-grid
  value and uses independent panel scales so no profile nodes are hidden by
  the near-best-model display threshold.
- Rebuilt the public `read_results.plot_corr` path as a two-dimensional profile
  corner plot for current SPAN tables. It profiles over the other parameters,
  marks valid grid nodes and the global joint best model, retains blank regions
  for unavailable irregular PoWR combinations, and shares a profile
  `Delta chi-squared` colour scale across panels.
- Added `read_results.plot_corner`, which reuses the same profile preparation,
  one- and two-dimensional minimisation, interpolation, confidence levels, and
  colour scaling. The tutorial now uses this single classical corner layout,
  with one-dimensional profiles on the diagonal and two-dimensional profiles
  below it. `compute_bestfit` and `plot_corr` remain available separately.
- Added TeX-formatted fitted values and asymmetric nominal intervals above all
  seven diagonal panels, adopted `f_B` for the companion light fraction,
  substantially increased all corner-plot typography, and moved the top-panel
  scientific scale multiplier away from the result title.
- Restored MINATO's Times-first plotting style with STIX as the portable maths
  fallback. The tutorial's displayed `f_B` range is zoomed to 0.05-0.20 without
  changing the fitted 0.05-0.50 grid; other axes retain their full input-grid
  limits.
- The tutorial objective is an unweighted residual sum of squares because the
  spectra contain no per-pixel uncertainties. Dividing that score by the
  degrees of freedom would produce a mean squared residual, not reduced
  chi-squared. The original diagonal score profiles remain in place.
  One-dimensional intervals use one-parameter likelihood-ratio thresholds,
  while joint contours use the corresponding two-parameter thresholds; both
  remain explicitly nominal.
- Replaced continuous score heatmaps with nested blue-grey rank regions on
  white axes, with black boundaries, sampled grid points, red injected-value
  guides, and red best-grid markers. The tutorial shows the best 10%, 25%, and
  50% of each 2D profiled grid because its nominal 2D thresholds correspond to
  fractional score increases of only 0.022%, 0.059%, and 0.113%, below the
  coarse grid resolution. Shape-preserving PCHIP interpolation avoids
  artificial cubic zero-score plateaus. The real PoWR run
  places five of seven injected quantities inside their nominal 68% intervals;
  the faint secondary's temperature and gravity fall just outside.
- Made report-file creation explicit through `report_to`; no report is written
  by default, matching the tutorial's in-memory-only policy.
- Made the standard plot portable without an external LaTeX installation,
  fixed its single-panel case, preserved the caller's result table, and used a
  scaled polynomial basis with a valid degree cap for sparse profile grids.
- Kept the tutorial explanation explicit that its unweighted score profiles
  are useful diagnostics but do not provide formal confidence intervals.

Validation:
- Executed the complete tutorial against the real PoWR grid. All 196,000 model
  combinations and the combined corner figure, containing seven diagonal
  profiles and 21 two-dimensional panels, completed without errors or warnings;
  the best-fit RMS residual remained 0.0205.
- The committed notebook remains unexecuted with zero stored outputs and no
  report or result files are created by the standard plotting call.
- The complete Python 3.13 suite passed 83 tests with one optional RAVEL smoke
  test skipped.

## 2026-07-15 - Extend the SPAN secondary-temperature profile

Implementation:
- Positioned SPAN consistently in the main README, tutorial index, module
  description, and tutorial as a fast, transparent, grid-based
  stellar-atmosphere fitting tool for disentangled binary-star spectra.
- Extended the tutorial's secondary-temperature range from 20-26 kK to
  16-26 kK in 2 kK steps.
- The PoWR grid has 0.2 dex gravity spacing but a temperature-dependent upper
  boundary. The tutorial therefore uses `16 kK` only at `log g=3.6` and
  `18 kK` at `log g=3.6, 3.8, 4.0`; no unavailable atmosphere combinations
  are interpolated or substituted.
- Updated the in-memory SPAN path to score the valid parameter nodes present in
  an irregular `RenderedAtmosphereGrid`. A regression test verifies that a
  missing Cartesian `(Teff, log g)` combination is not evaluated.

Validation:
- The complete real-PoWR tutorial used 20 atmosphere nodes per component, 280
  rendered spectra, and 196,000 valid joint combinations. It completed with a
  best-fit RMS residual of 0.0205.
- The best solution remains `(20 kK, log g=4.0, 125 km/s)` for the secondary.
  The best 18 kK solution, at `log g=3.8` and `125 km/s`, ranks seventh overall
  with score 4.740; the new lower temperatures do not reveal a better minimum.
- The complete Python 3.13 suite passed 80 tests with one optional RAVEL smoke
  test skipped. The committed notebook remains unexecuted with empty outputs.

## 2026-07-14 - Converge the SPAN tutorial disentangling

Diagnosis:
- The complete real-PoWR tutorial reproduced the reported secondary boundary
  solution with the original 100-iteration fixtures. The best secondary node
  was `(20 kK, log g=3.6, 125 km/s)` rather than the injected
  `(22 kK, log g=4.2, 120 km/s)`.
- The downloaded PoWR files are numerically identical to the generator inputs
  over the fitted wavelength range. The light-ratio approximation also does
  not cause the discrepancy.
- H-delta and H-gamma account for about 2.14 of the 2.31 score penalty against
  the injected secondary node. These broad wings had not converged after 100
  shift-and-add iterations.

Implementation:
- Regenerated both tutorial fixtures with 500 iterations, matching the
  contributed upstream implementation's default. A 1,000-iteration control
  changed the secondary profile difference by less than 0.001.
- Set 500 as the shared generator and command-line default, recorded it in the
  fixture provenance, and added a regression test tying the command-line
  default to that record.

Validation:
- With the 500-iteration fixtures, the complete 156,800-row PoWR fit prefers
  `(20 kK, log g=4.0, 125 km/s)` for the secondary, but the injected
  `(22 kK, log g=4.2)` atmosphere is nearly tied: the one-dimensional
  temperature profile differs by 0.056 and the relevant joint secondary score
  differs by 0.080. The best total score is 4.419.
- The remaining adjacent-node ambiguity is expected for a companion supplying
  only 10.8% of the light. It is no longer the monotonic temperature/gravity
  boundary bias produced by the under-converged fixtures.
- The focused SPAN/synthetic suite passed 38 tests. The complete Python 3.13
  suite passed 79 tests with one optional RAVEL smoke test skipped.

## 2026-07-14 - Reject calibrated PoWR spectra in normalised SPAN fits

Diagnosis:
- A completed tutorial run produced profile scores with a baseline near
  `3.2e5` and temperature/gravity minima at grid boundaries. That scale is
  consistent with comparing continuum-normalised observations near one against
  PoWR calibrated logarithmic fluxes near `-6` to `-8`.
- The PoWR filename recogniser accepted both `_line.txt` and
  `_line_calib.txt` for the same `(Teff, logg)` node. When both products were
  below `powr_dir`, nearest-node selection could silently choose whichever path
  sorted first. The tutorial's `log_flux="never"` then treated logarithmic
  calibrated values as normalised flux.

Implementation:
- `TextAtmosphereGrid` now rejects duplicate atmosphere nodes and supports a
  `file_filter` for directories containing several products per model.
- `render_atmosphere_grid` validates finite, continuum-like source fluxes by
  default and reports the offending node, median flux, and source path.
- The SPAN tutorial asks users to select a directory containing one normalised
  PoWR line-spectrum grid. Flux validation remains in
  `render_atmosphere_grid`, which checks every complete source spectrum before
  SPAN receives it. The tutorial also displays an example source and model
  median-flux range and rejects a best-fit RMS residual above 0.2.
- Removed the tutorial work directory and CSV export. The fitting grid,
  complete score table, best rows, and profiles now remain in memory, and the
  committed notebook contains no stored execution output.

Validation:
- A 36-file normalised validation grid completed the full notebook. All 10
  code cells executed and all 156,800 result rows were produced without
  errors. A mixed grid is rejected as ambiguous rather than silently selecting
  one of two products for the same atmosphere node.
- The in-memory-only notebook completed again after removing the work-directory
  and CSV-export cells. A source scan found no persistent-output calls, and the
  committed notebook remains unexecuted with empty cell outputs.
- A two-node run with the actual PoWR sources recovered light fraction 0.10,
  primary `v sin i=75 km/s`, secondary `v sin i=100 km/s`, and best-fit RMS
  residual 0.0272. Supplying only a calibrated source now raises a descriptive
  error before rendering.

## 2026-07-14 - Refactor SPAN around in-memory synthetic model grids

- Added the generic `RenderedAtmosphereGrid` container and
  `render_atmosphere_grid` renderer to `minato.synthetic`. The renderer uses
  the same resampling, rotational-broadening, and instrumental-broadening path
  as `render_single_star`, validates exact atmosphere nodes, and keeps fitting
  models noiseless and in memory.
- Added component-specific `modelsA_grid` and `modelsB_grid` inputs to
  `AtmFit`, while preserving the existing model-directory API. Different
  atmosphere grids can therefore still be used for the two binary components.
- The direct binary path scores each component model once per light ratio and
  combines the two score tables afterwards. The tutorial's 156,800 joint
  combinations now require 1,400 primary and 1,120 secondary spectral
  comparisons instead of 156,800 repeated joint comparisons.
- Removed the tutorial-specific disk conversion helper, copied model files,
  and CSV model manifest. The SPAN notebook now passes the rendered synthetic
  grid directly to `AtmFit`.

Validation:
- The complete 23-cell notebook executed top to bottom with a temporary
  36-node analytic atmosphere directory standing in for the user download.
  It rendered 252 models, produced all 156,800 result rows, and raised no cell
  errors. This validates the tutorial plumbing but is not a scientific PoWR
  recovery test.
- A separate 490-combination fit used the two actual PoWR source spectra from
  which the tutorial fixtures were generated. It recovered light fraction
  0.10, primary `v sin i=75 km/s`, and secondary `v sin i=100 km/s`, with
  125 km/s nearly tied, matching the previous file-based recovery check.
- The focused synthetic/SPAN suite passed 37 tests. The complete Python 3.13
  suite passed 78 tests with one optional RAVEL smoke test skipped.
- Fresh wheel and source archives under
  `/tmp/minato-dist-powr-fluxfix-20260714` passed the release-content policy.
  An isolated wheel import confirmed `RenderedAtmosphereGrid`,
  `render_atmosphere_grid`, its normalised-flux validation, the atmosphere
  directory file filter, and the `modelsA_grid`/`modelsB_grid` SPAN inputs.

## 2026-07-14 - Add step-by-step guidance to the SPAN tutorial

- Reorganised `span_example.ipynb` into 23 explanation/code cells. Each
  calculation now states its purpose, inputs, expected output, and the next
  interpretation step.
- Clarified that `span_synthetic/provenance.json` was generated by
  `scripts/generate_span_tutorial_data.py` with the synthetic observations and
  is not a PoWR download. Its injected atmosphere values are used only for
  recovery checks, while its light fraction supplies the `lrat0` flux-scale
  reference required by the fit.
- Explicitly identifies the plotted spectra as the disentangled primary and
  secondary observations supplied to SPAN, labels both panels, and explains
  their continuum scale and different noise quality.
- Replaced unexplained tuple and integer outputs with labelled path, grid-size,
  rendered-model-count, wavelength-coverage, and result messages. The fitting
  section now defines the unweighted score, explains the separable component
  calculation, and shows injected values on readable one-dimensional profile
  plots.
- Notebook JSON, code-cell syntax, clean source outputs, the introductory
  workflow with a temporary two-model PoWR directory, and the complete
  calculation with a temporary 36-node analytic validation directory were
  checked with Python 3.13.

## 2026-07-13 - Rebuild the SPAN tutorial around synthetic O+B spectra

Scope:
- Added a development-only generator for ten noisy `R=40,000`, `S/N=100`
  composite spectra from PoWR `GAL-OB-Vd3` models `32-40` and `22-42`, a
  17-day circular orbit, and injected rotational velocities of 80 and
  120 km/s. The PoWR masses give RV semi-amplitudes of 70.4754 and
  176.7919 km/s.
- The calibrated PoWR spectra imply a secondary blue-optical light fraction of
  0.1078785. The ten epochs were disentangled for 100 iterations through the
  development-only `minato.contrib.spdis` adaptation.
- Added two provenance-tracked synthetic disentangled spectra under
  `minato/tutorials/span_synthetic/`. The original PoWR files, ten composite
  epochs, and contributed disentangling code remain outside release packages.
- Rewrote `span_example.ipynb` to request the official PoWR grid through
  `MINATO_POWR_GRID`, render an in-memory fitting grid at `R=40,000`, and
  search component temperatures, gravities, light fractions from 0.05 to 0.50,
  and equal `v sin i` steps of 25 km/s from 50 to 200 km/s.

Implementation:
- Added the generic `render_atmosphere_grid` synthetic renderer and direct
  in-memory SPAN fitting path. SPAN also supports decimal model values,
  commented spectra, wavelength-grid interpolation, explicit input shifts,
  and configurable process/chunk counts for the legacy directory path.
- Replaced the model-flux-divided residual statistic with a non-negative
  squared-residual score and corrected the H-delta fitting interval to
  4087-4130 Angstrom.
- Kept all tutorials, fixtures, development tests, atmosphere grids, and
  `minato.contrib` outside wheel and source archives. They remain available in
  the GitHub repository according to branch policy.

Validation:
- The generated primary fixture has an approximately 0.0050 fit-window RMSE
  against its injected broadened spectrum; the 10.8%-light secondary has an
  approximately 0.0412 RMSE and correspondingly weaker rotational constraint.
- An exact-node 14-model in-memory recovery grid selected light fraction 0.10 and primary
  `v sin i=75 km/s`; the secondary selected 100 km/s, with 125 km/s nearly
  tied. This is consistent with the intentionally weak companion and motivates
  the tutorial's profile plots and refined-grid guidance.
- The full Python 3.13 suite passed 74 tests with one optional RAVEL test
  skipped before the in-memory fitting refactor. Fresh wheel and source
  archives in `/tmp/minato-dist-20260713-v2` passed the release-content
  checker. The final refactored suite and package artefacts require a fresh
  validation pass.

Remaining work:
- Execute the complete 156,800-point notebook grid after the user supplies the
  full PoWR archive; the exact-node workflow and generated fixtures have been
  validated end to end.
- Complete the two remaining legacy spectral-analysis notebooks and the RAVEL
  tutorial clean-up before marking the combined roadmap item done.

## 2026-07-13 - Define external-code and data-free release boundaries

Decisions:
- The adapted shift-and-add class comes from
  `TomerShenar/Disentangling_Shift_And_Add` and is not a MINATO product. It is
  retained as `minato.contrib.spdis` for development checkouts only, removed
  from the released `minato` package surface, and excluded from packages and
  release branches.
- The upstream README credits Tomer Shenar, with contributions from Matthias
  Fabry and Julia Bodensteiner, and requests citations to Gonzalez & Levato
  (2006), Shenar et al. (2020), and Shenar et al. (2022). No software licence
  file was present in the upstream repository during this audit, so the local
  adaptation must not be redistributed under MINATO's MIT licence without an
  explicit permission and licence review.
- MINATO wheels and source archives will contain no atmosphere grids, trained
  models, spectra, fitted results, tutorials, development tests, or contributed
  code. The final `main` tree may retain only explicitly approved synthetic
  tutorial fixtures with provenance; observed spectra and scientific model
  grids remain excluded.
- Tutorials will use small synthetic spectra generated during execution where
  possible. SPAN examples may use public TLUSTY or PoWR grids obtained
  separately by users under the original terms and citation guidance.

Repository audit:
- `minato/models/` contains 2,532 tracked files and occupies about 180 MB.
- The tutorial tree contains 155 tracked non-notebook files: 82 PNG, 56 text,
  12 PDF, four CSV, and one Feather file.
- Bundled RAVEL synthetic fixtures occupy about 98 MB. Existing RAVEL output
  directories occupy about 35 MB, 13 MB, and 13 MB.
- `minato/tutorials/example_spectra/` contains two real disentangled spectra
  used by the legacy SPAN notebooks.
- Previous wheel and source builds excluded the model and tutorial trees by
  package-discovery behaviour, but included `minato.spdis`; this was not a
  sufficiently explicit release guarantee.

Implementation:
- Added `MANIFEST.in` exclusions and
  `scripts/check_release_artifacts.py`, with a matching CI step, to reject
  development records, contributed code, package data, model trees, and
  tutorial trees in release artefacts.
- Updated package import tests, the main README, tutorial index, release plan,
  roadmap, and changelog to reflect the approved boundaries.

Validation:
- Rebuilt the repository `.venv` from `uv.lock` with Python 3.13.5 and 132
  locked packages. The environment is complete and directly runnable again.
- The full Python 3.13 unit suite passed 62 tests with the opt-in
  probabilistic RAVEL smoke test skipped.
- `uv lock --check`, notebook JSON parsing, workflow YAML parsing,
  `py_compile`, and `git diff --check` passed.
- Fresh wheel and source archives were built in
  `/tmp/minato-dist-contrib-namespace-20260713`. The artefact checker passed
  both; they contain no `minato.contrib`, `minato.spdis`, model, tutorial, or
  non-Python package data.
- A no-dependency installation of the wheel in
  `/tmp/minato-wheeltest-contrib-namespace-20260713` imported `minato` from the
  isolated environment and confirmed that both `minato.contrib` and
  `minato.spdis` are absent.

Remaining work:
- Execute the complete SPAN grid after the external PoWR download, and rewrite
  the result-inspection and RAVEL notebooks without observed spectra or
  machine-specific paths.
- Clean notebook outputs and remove all model/data/output assets from the
  pending `main` merge tree, then validate that exact tree before release.
- Publish the final candidate to TestPyPI and repeat the installation check
  from the package index.

## 2026-07-13 - Trace the RAVEL SB2 indentation regression

- `git blame` and the relevant patches identify `b9cc352f` (`Tag SB1/SB2 fit
  plots with chosen profile`, 2025-11-29) as the first regression. An unrelated
  plot-filename edit dedented the wavelength-preparation lines inside the SB2
  NumPyro model but left the following profile lines indented, producing an
  immediate `IndentationError`.
- `261df900` (`Restore classic SB1 uncertainty band via lmfit`, 2025-11-30)
  made the file syntactically valid while adding profile controls by dedenting
  the remaining SB2 profile and likelihood block. This placed the block outside
  `sb2_model`, where model arguments such as `λ` were unavailable at runtime.
- Compiling the historical files reproduces the sequence: `b9cc352f` fails at
  parse time, while `261df900` parses but retains the incorrect function scope.
  Commit `1676a28` restores the complete block to `sb2_model`, and the new
  probabilistic SB2 smoke test protects the path in CI.
- The main README now previews the implemented `0.3.0` highlights and clearly
  separates them from the remaining release gates.

## 2026-07-12 - Add CI and repair the SB2 release path

Scope:
- Added `.github/workflows/ci.yml` with Python 3.12/3.13 unit-test jobs, a
  Python 3.13 reduced RAVEL SB1/SB2 smoke job, and a package-build job.
- Added `tests/test_ravel.py`. The normal suite checks a deterministic classic
  SB1 fit; `MINATO_RUN_RAVEL_SMOKE=1` enables reduced one-line, one-epoch
  probabilistic SB1 and SB2 fits.
- Fixed an indentation regression that left the SB2 profile and likelihood
  block outside the nested NumPyro model and caused `NameError: name 'λ' is
  not defined`.
- Fixed single-epoch SB2 prior shaping by retaining an explicit epoch axis.
- Made the second SB2 MCMC stage honour all public sampling and progress
  controls instead of hard-coding `4 x (1000 + 2000)` draws.
- Removed hard-coded laptop import paths from the RAVEL tutorials and pointed
  the SB2 source cells at the existing `SB2_case1` and `SB2_case2` synthetic
  directories. Existing notebook outputs were left untouched.

Validation:
- Python 3.12 and Python 3.13 normal suites passed 56 tests with one gated
  probabilistic smoke test skipped.
- The enabled RAVEL smoke suite passed both classic/probabilistic tests in
  about 17 seconds. Posterior RV arrays were finite with SB1 shape
  `(10, 1, 1)` and SB2 shape `(10, 2, 1, 1)`.
- `uv lock --check`, `pixi lock --check`, workflow YAML parsing, notebook JSON
  parsing, tutorial fixture discovery, and `git diff --check` passed.
- A fresh mamba installation resolved Python 3.13.14, NumPy 2.2.6, and Numba
  0.66.0. Installing the checkout with `--no-deps` left no broken requirements,
  and all 56 tests passed with the opt-in RAVEL smoke skipped.

Mamba lock correction:
- The first clean installation exposed a mixed-manager conflict: ExoJAX's
  PyPI dependency resolution replaced conda's NumPy 2.2.6 with NumPy 2.4.6,
  violating MINATO's declared `numpy<2.3` requirement.
- `minato_env.yml` now repeats the NumPy 2.2 constraint in its pip subsection,
  and `conda-lock.yml` was regenerated for Linux x86-64, Intel macOS, and Apple
  Silicon. Direct conda-lock parser checks reproduce all three recorded input
  hashes exactly.

Remaining validation:
- Push `develop` and confirm all GitHub Actions jobs pass.
- Run the full multi-epoch SB1/SB2 tutorials in isolated output directories;
  the reduced smoke verifies code paths, not production posterior quality.

## 2026-07-12 - Promote observing and adopt Python 3.13

Scope:
- Replaced the recovered top-level `Observing/` directory with the installable
  `minato.observing` package.
- Added phase-window generation, Astroplan observability checks, and night
  visibility plotting. Data are returned to callers; printing and CSV output
  are optional, and existing files require `overwrite=True`.
- Added focused observing tests and a clean, offline-capable tutorial at
  `minato/tutorials/observing_tutorial.ipynb`.
- Set Python 3.13 as the development baseline and declared Python 3.12-3.13
  support. Pinned NumPy to the 2.2 line across package, Pixi, and mamba
  definitions so Python 3.13 installs use binary wheels.
- Replaced `kepler.py` with a vectorised SciPy/Halley solver in
  `minato.binary_population.orbits`.
- Regenerated `uv.lock`, `pixi.lock`, and `conda-lock.yml` for the updated
  runtime and dependency policy.

Numerical and performance checks:
- Against `kepler.py`, 100,000 random elliptic orbits differed by at most
  `2.8e-13` radians after angle wrapping. A 100,000-epoch, `e=0.92` RV curve
  differed by at most `7.4e-11 km/s`.
- The isolated SciPy solver was about 2.5 times slower in that microbenchmark
  (`0.027 s` versus `0.011 s`), with no meaningful change in RV results.

Validation:
- A clean uv Python 3.13 environment installed 132 packages and passed all 54
  unit tests with NumPy 2.2.6.
- A separate clean uv Python 3.12 environment passed the same 54 tests, so both
  ends of the declared Python support range are validated.
- A locked Pixi Python 3.13 environment passed the same 54 tests.
- `uv lock --check`, `pixi lock --check`, and the three-platform conda-lock
  input-hash check passed.
- The observing tutorial executed successfully with `nbclient`; the disposable
  executed copy is `/tmp/minato-observing-tutorial-executed.ipynb`.
- `uv build` created
  `/tmp/minato-dist-py313-final/minato_astro-0.2.0-py3-none-any.whl` and the
  matching source archive. The wheel includes `minato.observing`,
  `minato.spdis`, and the SciPy orbital solver, and excludes development logs,
  tutorials, and atmosphere-model files.
- Installing that wheel with dependencies into `/tmp/minato-wheeltest` worked
  without the source checkout. Installed imports for `minato.observing`,
  `minato.binary_population`, and `minato.spdis` passed with current unlocked
  package-index dependencies as well as with the committed locks.

Remaining validation:
- A direct `conda-lock install --micromamba` attempt made no prefix progress
  and was stopped after several minutes. The lock itself is current for
  `linux-64`, `osx-64`, and `osx-arm64`; repeat the install on the laptop.
- Complete representative SB1/SB2 release workflows.
- Existing invalid-escape warnings in `span.py` and `spdis.py`, plus Python
  3.13 multiprocessing fork warnings, remain follow-up work.

## 2026-07-12 - Start 0.3.0 release preparation

Scope:
- Started release preparation from clean `develop` after the binary-population
  inference status update.
- Approved `0.3.0` rather than `1.0.0`. The official release date is defined as
  the date when the approved merge from `develop` is committed on `main`.
- Approved the distribution name `minato-astro` because `minato` is occupied
  by an unrelated PyPI project. The import namespace remains `minato`.
- Added Pixi configuration to `pyproject.toml`, generated `pixi.lock` for
  `linux-64`, `osx-64`, and `osx-arm64`, and refreshed `uv.lock`.
- Kept `minato_env.yml` as the mamba development manifest and added the local
  editable package plus `astroplan`; generated `conda-lock.yml` for `linux-64`,
  `osx-64`, and `osx-arm64`. The editable checkout is installed separately
  after a locked environment because conda-lock excludes `-e .`.
- Recovered useful `Observing/` Python sources from `archive-develop` without
  archived notebooks, figures, or the hard-coded runner. The directory remains
  outside the installable package.
- Confirmed `minato/spdis.py` was already identical to `archive-develop` and
  repaired its package-relative `myRC` import.
- Reworked the main README and tutorial index to document current modules,
  installation choices, development-only material, and tutorial status.
- Expanded `minato/tutorials/binary_population_tutorial.ipynb` and the module
  README with baseline-conditioned averaged mixture-CRN guidance and current
  process-pool choices.
- Added `RELEASE_PREPARATION_PLAN.md` with package, branch-content, version, and
  Read the Docs recommendations.

Audit findings:
- PyPI already serves an unrelated `minato` distribution, so MINATO must not
  document `pip install minato`.
- `pyproject.toml` and `minato/__init__.py` still report version `0.2.0`; do not
  bump until the release checks pass and the merge is ready.
- `CHANGELOG.md` still uses `[Unreleased]`; do not move entries to a dated
  release section until the release is approved.
- Tracked hygiene checks found no tracked `__pycache__`, `.DS_Store`, or
  token-like files.
- `minato/models/` contains many tracked atmosphere-grid files; review package
  contents before building a distribution.
- The archived observing helpers need API redesign and tests before they move
  into `minato.observing`.
- Existing SB1/SB2 notebooks contain large outputs and require a separate
  clean-up and execution pass before release.

Validation:
- `UV_CACHE_DIR=.uv-cache uv lock --check` passed with 176 resolved packages.
- `pixi lock --check` passed against `pixi.lock` for three platforms.
- `conda-lock lock --micromamba --file minato_env.yml --platform linux-64
  --platform osx-64 --platform osx-arm64` generated `conda-lock.yml`.
  A subsequent `--check-input-hash` pass confirmed all three platform specs are
  current without re-solving.
- `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s
  tests` passed 48 tests. Importing `minato.spdis` still reports archived
  invalid-escape deprecation warnings that should be cleaned before promoting
  the module beyond experimental status.
- `jq empty minato/tutorials/binary_population_tutorial.ipynb` passed. All
  notebook code cells also executed sequentially in one Python namespace with
  the expensive MCMC/CRN flags left off.
- `UV_CACHE_DIR=.uv-cache uv build --out-dir /tmp/minato-dist-2` built
  `/tmp/minato-dist-2/minato_astro-0.2.0-py3-none-any.whl` and
  `/tmp/minato-dist-2/minato_astro-0.2.0.tar.gz` without packaging warnings.
  The wheel is about 146 KB and excludes `Observing/`, development logs,
  tutorials, and the tracked atmosphere-model tree.
- `git diff --check` passed before the final logbook update.

Remaining validation:
- Create genuinely fresh environments from `uv.lock`, `pixi.lock`, and
  `conda-lock.yml`, then run the same import and unit-test smoke checks.
- Execute the notebook with `nbclient`/`nbconvert` and retained outputs in a
  disposable copy; the repository notebook intentionally remains clean.
- Run representative SB1 and SB2 workflows before release.

## 2026-06-14 - Replace synthetic-spectrum creation tutorial

Scope:
- Replaced `minato/tutorials/create_synth_spectra.ipynb` with a clean
  `minato.synthetic` tutorial.
- Removed notebook-local renderer code and hard-coded local PoWR paths from the
  tutorial.
- The new notebook demonstrates an analytic runnable backend, optional
  user-supplied `TextAtmosphereGrid`/`FallbackAtmosphereGrid` usage, single-star
  rendering, SB1/SB2 multi-epoch rendering, `write_ravel_txt`, `JDs.txt`, and
  truth-manifest writing.
- Outputs are directed to `tutorial_outputs/create_synth_spectra/` instead of
  overwriting committed RAVEL tutorial spectra.

Validation:
- `jq empty minato/tutorials/create_synth_spectra.ipynb`.
- Confirmed the notebook has 24 cells, no outputs, and no execution counts.
- Confirmed old local-path markers are absent from the notebook source.
- `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s tests`
  passed 13 tests.

## 2026-06-14 - Fallback atmosphere-grid routing

Scope:
- Added `FallbackAtmosphereGrid` for composing multiple user-supplied
  atmosphere backends in priority order.
- A backend can decline a star by raising `LookupError`, for example when its
  nearest `(teff, logg)` node exceeds user-set tolerances; MINATO then tries the
  next backend.
- Returned spectra record `selected_grid` metadata so downstream renders know
  which grid family was actually used.
- This remains explicit user policy: MINATO does not choose PoWR, TLUSTY,
  FASTWIND, or any other model family by default.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s tests`
  passed (`13` tests).
- `jq empty minato/tutorials/synthetic_spectra_ravel_bridge.ipynb` passed.
- Import smoke check for `FallbackAtmosphereGrid` and `TextAtmosphereGrid`
  passed.

## 2026-06-14 - Text atmosphere-grid adapter

Scope:
- Added `TextAtmosphereGrid` and `AtmosphereGridNode` as generic atmosphere
  backends for folders of text model spectra.
- Directory scanning recognises the MINATO filename convention
  `teff25000_logg4.00.txt` plus common PoWR, TLUSTY, and FASTWIND-style names.
- Added fallback paths for unconventional model names: custom regex patterns,
  parser functions, editable index templates, and explicit CSV indexes.
- The adapter performs nearest-neighbour selection inside the supplied grid; it
  does not encode a scientific policy for choosing between PoWR, TLUSTY,
  FASTWIND, AP18, or other model families.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s tests`
  passed (`10` tests).
- `jq empty minato/tutorials/synthetic_spectra_ravel_bridge.ipynb` passed.
- Import smoke check for `TextAtmosphereGrid.recognised_formats()` passed.

## 2026-06-14 - Generic synthetic-spectrum rendering API

Scope:
- Added `minato.synthetic` with in-memory `Spectrum`, `Star`,
  `BinarySystem`, and `ObservationModel` containers.
- Atmosphere grids are backend interfaces via `get_spectrum(star)`; AP18,
  PoWR, MIST, SDSS filenames, and BOSS noise policies remain outside the core.
- Added optional CSV-bank isochrone interpolation plus `Star.from_mass` and
  `BinarySystem.from_masses` conveniences.
- Added Doppler shifting, log-grid resampling, rotational/instrumental
  broadening, flux-weighted binary combination, seeded noise injection, and
  `write_ravel_txt` for RAVEL-compatible text output.
- Added `minato/tutorials/synthetic_spectra_ravel_bridge.ipynb` and
  `minato/synthetic/DESIGN.md`.

Validation:
- `./.venv/bin/python -m unittest discover -s tests` passed (`4` tests).

## 2026-06-11 - Four-parameter binary-population MCMC support

Scope:
- Extended `minato.binary_population.run_mcmc` from the historical
  one-parameter `f_bin` default to an opt-in parameter list supporting
  simultaneous sampling of `f_bin`, `pi`, `kappa`, and `eta`.
- Added `BinaryPopulation.logP_powerlaw_mode = "direct"` for literature-style
  `p(log10 P) proportional to (log10 P)^pi` sampling over positive log-period
  intervals. The old shifted-logP sampler remains the default.

Validation:
- AST syntax check passed for `mcmc.py` and `population.py`.
- A tiny four-parameter synthetic-cadence smoke run returned finite
  prior/likelihood values and chain shape `(2, 16, 4)`.
- A tiny legacy one-parameter smoke run returned chain shape `(2, 8, 1)`.

Paper context:
- Added for the SDSS-V multiplicity paper clean-epoch analysis, where the
  first broad-domain inference uses `log10(P/d)=0.15-3.5`, `q=0.1-1`, and
  fits `f_bin`, `pi`, `kappa`, and `eta`.

## 2026-05-05 - RAVEL integrated four-fit validation with Gaussian non-H profiles

Output:
- `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_20260505`
- report: `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_20260505/REPORT.md`

Command:
- `env MINATO_QUIET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_ENABLE_X64=True XLA_FLAGS=--xla_force_host_platform_device_count=2 MPLCONFIGDIR=benchmarks/results/.mplconfig ./.venv/bin/python benchmarks/run_p117_fourfit_validation.py --output-dir benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_20260505 --families blue,red,full,na --num-warmup 100 --num-samples 300 --num-chains 2 --chain-method parallel --max-interp-points 400`

Scope:
- Ran the fixed 10-star validation subset with all four planned families: `blue`, `red`, `full`, and `na`.
- Stellar families used non-H `profile="Gaussian"` and `Hprofile="Lorentzian"`.
- Na used the dedicated Na-doublet diagnostic model with the narrowed default window `5882-5905 Å`.
- Plots were enabled; corner plots were off for this integrated pass.
- Low-cost smoke sampler settings were `num_warmup=100`, `num_samples=300`, `num_chains=2`.

Results:
- `40/40` fits completed.
- Median runtimes were `7.36 s` for blue, `5.89 s` for red, `17.53 s` for full, and `3.38 s` for Na.
- Total wall time was `376.47 s`.
- Mode A red/full collapse did not reappear; the old repeated zero-error RV template was absent.
- Mode B Na behaviour is still the limiting case: `82244026` is mostly clean, while `73957460` remains explicitly flagged with stationary-scatter warnings and a high-RV epoch 4 (`349.73 km/s`) at this low-cost sampler depth.
- Mode C `82903408` blue plots now look coherent with the full blue line list; the earlier apparent amplitude inconsistency is resolved.

Decision:
- Use Gaussian profiles for non-H stellar lines in the P117 four-fit validation workflow.
- Keep H/Paschen lines Lorentzian.
- Keep Na separate on the dedicated Na-doublet diagnostic path with the narrowed `5882-5905 Å` window.
- This is still a functional validation at smoke-test MCMC depth, not the final production-depth or throughput benchmark.
- The previous Na-only deep run with the same narrowed window (`num_warmup=500`, `num_samples=1000`, `num_chains=4`) recovered `73957460` epoch 4 near the stationary solution, so the remaining `73957460` issue is sampler-depth sensitive rather than fixed by window narrowing alone.

Targeted follow-up for `73957460` Na:
- Reran only `73957460` / `na` in the same output tree with `num_warmup=200`, `num_samples=500`, `num_chains=4`.
- Runtime was `3.91 s`.
- Two low-likelihood chains were rejected and two chains retained.
- The current `73957460/na` products in `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_20260505` come from this targeted run.
- Epoch 4 recovered the stationary solution: `45.51 km/s` with `-6.97/+5.14 km/s` asymmetric errors.
- Targeted summary copy: `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_20260505/summary_73957460_na_200_500_4.json`.

## 2026-05-05 - RAVEL 10-star 200/500/4 comparison

Output:
- `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_200_500_4_20260505`
- comparison report: `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_200_500_4_20260505/COMPARISON_TO_LOW_COST.md`
- comparison table: `benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_200_500_4_20260505/rv_comparison_to_low_cost.csv`

Command:
- `env MINATO_QUIET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_ENABLE_X64=True XLA_FLAGS=--xla_force_host_platform_device_count=4 MPLCONFIGDIR=benchmarks/results/.mplconfig ./.venv/bin/python benchmarks/run_p117_fourfit_validation.py --output-dir benchmarks/results/p117_validation_subset_fourfit_gaussian_nonh_200_500_4_20260505 --families blue,red,full,na --num-warmup 200 --num-samples 500 --num-chains 4 --chain-method parallel --max-interp-points 400`

Scope:
- Reran all four families for all 10 validation stars with `num_warmup=200`, `num_samples=500`, `num_chains=4`.
- Stellar profile settings stayed at non-H `Gaussian`, H/Paschen `Lorentzian`.
- Na used the narrowed default window `5882-5905 Å`.
- Plots were enabled; corner plots were off.

Results:
- `40/40` fits completed.
- Total wall time was `320.03 s` in this run, compared with `376.47 s` for the previous low-cost smoke run. This should not be over-interpreted as a guaranteed speed-up because the deeper run used `XLA_FLAGS=--xla_force_host_platform_device_count=4` while the earlier integrated smoke run used `2`; it does show that `200/500/4` is not obviously too expensive at this 10-star scale.
- Median runtimes were `7.25 s` for blue, `5.80 s` for red, `12.90 s` for full, and `3.58 s` for Na.
- Stellar RV differences relative to the current low-cost output tree were small:
  - blue: max absolute shift `3.10 km/s`, median `0.75 km/s`
  - red: max absolute shift `2.81 km/s`, median `0.89 km/s`
  - full: max absolute shift `3.10 km/s`, median `0.46 km/s`
- Na gained substantially in robustness:
  - `73957460` epoch 4 stayed on the stationary solution (`45.51 km/s`) with two low-likelihood chains rejected.
  - `82244026` epochs that were pathological in the low-cost tree recovered near the stationary Na solution (`-3.00` and `-13.72 km/s` for the last two epochs).

Interpretation:
- The stellar fits do not materially change at `200/500/4` on this subset.
- Na is the main beneficiary of the deeper sampler and four chains.
- Candidate default for functional validation is now `200/500/4` for all families unless throughput tests show this is too expensive at scale. If production throughput becomes limiting, use family-specific settings with Na at `200/500/4` and stellar families at the cheaper setting.

## 2026-05-05 - RAVEL 10-star stellar profile comparison

Comparison outputs:
- `benchmarks/results/p117_blue_profile_compare_20260505/blue_voigt`
- `benchmarks/results/p117_blue_profile_compare_20260505/blue_gaussian_nonh`
- `benchmarks/results/p117_stellar_profile_compare_20260505/redfull_voigt`
- `benchmarks/results/p117_stellar_profile_compare_20260505/redfull_gaussian_nonh`
- summary report: `benchmarks/results/p117_stellar_profile_compare_20260505/REPORT.md`

Scope:
- Fixed 10-star validation subset.
- Compared non-H `profile="Voigt"` against non-H `profile="Gaussian"`.
- Kept `Hprofile="Lorentzian"` in both runs.
- Ran `blue`, `red`, and `full`; Na was not part of this stellar-profile comparison.

Results:
- `60/60` stellar fits completed across the two profile settings.
- Gaussian-minus-Voigt RV shifts were small:
  - blue: maximum absolute shift `5.49 km/s`, median absolute shift `1.52 km/s`, max shift `0.53` combined sigma
  - red: maximum absolute shift `5.39 km/s`, median absolute shift `1.88 km/s`, max shift `0.50` combined sigma
  - full: maximum absolute shift `3.15 km/s`, median absolute shift `1.01 km/s`, max shift `0.50` combined sigma
- Gaussian non-H fits were modestly faster in these smoke runs.

Interpretation:
- At this smoke-test depth, Gaussian non-H profiles do not show an RV-level regression relative to Voigt non-H profiles.
- Gaussian non-H is a credible candidate for the production stellar profile default, but final promotion should wait for qualitative plot review and the chosen production MCMC depth.
- Minor cosmetic follow-up: some three-epoch blue plots place the y-axis label too close to tick labels.

## 2026-05-05 - RAVEL SB1 width-parameter cleanup

Targeted diagnostic output:
- `benchmarks/results/mode_c_sb1_width_sampling_20260505`

Code change:
- SB1 probabilistic fitting now samples only line-width parameters that enter the selected profile likelihood.
- With `profile="Voigt"` and `Hprofile="Lorentzian"`, Balmer lines no longer sample unused Gaussian widths.
- With `profile="Gaussian"` and `Hprofile="Lorentzian"`, non-H lines sample only Gaussian widths and Balmer lines sample only Lorentzian widths.
- SB1 profile corner plots now use the same masks, so `cornerplot_wid_G.png` and `cornerplot_wid_L.png` only show active parameters.

Targeted validation:
- Reran `82903408` blue with the full blue line list for:
  - non-H `Voigt`, H `Lorentzian`
  - non-H `Gaussian`, H `Lorentzian`
- Both runs completed and produced coherent `4471` panels.
- The Gaussian-vs-Voigt RV differences were small at smoke-test depth, with epoch differences from `-2.84` to `+2.18 km/s`.
- Do not promote Gaussian non-H profiles to the default yet; run the full 10-star validation subset with both settings first.

SB2 note:
- The equivalent unused-width cleanup has not yet been applied to SB2 because the two-stage frozen-parameter model needs a separate, validated edit.

## 2026-05-05 - RAVEL mode C blue plot-axis fix

Targeted diagnostic output:
- `benchmarks/results/mode_c_sb1_axis_fix_20260505/82903408/blue`
- `benchmarks/results/mode_c_sb1_axis_fix_20260505/82244026/red_axis_check`

Diagnosis:
- The blue mode C failure for `82903408` was a plotting-axis bug, not evidence that the production blue line list had to be tailored per star.
- `82903408` has 7 epochs and the blue family has 7 lines. The SB1 plotter had a legacy shape-detection branch that swapped the line and epoch axes whenever `fλ_pred.shape[1] == n_epochs` and `fλ_pred.shape[2] == len(lines)`.
- In the ambiguous `7 x 7` case this swapped a correct current trace shaped `(sample, line, epoch, pixel)` into the wrong order, producing apparent per-epoch depth changes for a single line. That visual behaviour looked like a violation of the shared-amplitude requirement, but the model parameters were already shared per line across epochs.

Code changes:
- SB1 plotting now treats current traces as `(sample, line, epoch, pixel)` first and only swaps legacy traces when the shape is unambiguously `(sample, epoch, line, pixel)`.
- SB1 plots now draw a coherent high-likelihood posterior sample rather than a pointwise posterior median.
- When `cornerplots=True`, SB1 now writes additional profile-parameter diagnostics:
  - `cornerplot_amp.png`
  - `cornerplot_wid_G.png`
  - `cornerplot_wid_L.png`

Validation:
- Reran `82903408` blue with the full production blue list `[4026, 4102, 4144, 4340, 4388, 4471, 4713]`.
- The `4471`, `4102`, and `4340` fit panels now look coherent, with shared-depth profiles across epochs.
- Also reran `82244026` red because it has the same axis ambiguity (`5` red lines and `5` epochs); the red panels look coherent after the fix.
- The RV table is unchanged relative to the previous all-blue run, as expected for a plotting-axis fix.
- The earlier no-Hδ run is retained only as a diagnostic control; it is not the production solution.

## 2026-05-04 - RAVEL mode A/C targeted validation

Targeted validation outputs are in `benchmarks/results/mode_ac_validation_20260504`.

Scope:
- Mode A: red/full collapse recovery for `73590837`, `76657579`, `105622073`
- Mode C: blue posterior/plot diagnosis for `82903408`

Run settings:
- `MINATO_QUIET=1`, BLAS/OpenMP thread counts set to `1`
- `JAX_ENABLE_X64=True`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4`
- `chain_method="parallel"`
- `max_interp_points=400`
- plots and corner plots enabled

Validation summary:
- `8/8` targeted fits completed in `89.5 s`
- Mode A did not reproduce the old repeated pathological red/full RV template (`50.0176, -376.8281, 108.6114`) and did not produce zero uncertainties
- Mode C smoke and deep blue runs for `82903408` agreed to within `1.3 km/s` per epoch, so increasing chains/samples did not change the RV solution
- The `82903408` blue corner plot is unimodal; remaining visual concerns appear to be line-profile/depth mismatch rather than sampler failure

Code behaviour under validation:
- Finite-window sanitisation is now shared by SB1, Na, and SB2 probabilistic interpolation paths
- Probabilistic Na, SB1, and SB2 result writers now preserve asymmetric posterior errors via `rv_err_minus` and `rv_err_plus`
- Na outputs additionally record `na_quality_flag`, `na_posterior_warning`, and `na_warning_reasons`
- Generic chain rejection was not added to SB1/SB2; Na chain filtering remains diagnostic-specific

Detailed morning-review report:
- `benchmarks/results/mode_ac_validation_20260504/REPORT.md`

Follow-up mode C diagnosis:
- The original blue family `[4026, 4102, 4144, 4340, 4388, 4471, 4713]` produced visibly poor `82903408` blue plots, even though the posterior was unimodal
- Comparing line-list variants showed that dropping Hδ (`4102`) made the visual issue disappear, but this was a diagnostic clue rather than a valid production solution
- The actual cause was later identified as an SB1 plotting-axis ambiguity when `n_lines == n_epochs == 7`; see the `2026-05-05` entry
- The validation runner keeps the full blue family; do not tailor the line list per star for this failure mode

## 2026-04-17 - Na diagnostic and single-epoch support

Follow-up decisions for the planned four-fit SB1 production workflow:
- The Na fit is a diagnostic branch, not part of the stellar `blue` / `red` / `full` fits
- Rationale: the current probabilistic SB1 model enforces one shared RV per epoch across all lines in a run, so interstellar Na should remain in a separate fit family
- Added `Na I 5896` to the line dictionary alongside the existing `Na I 5890` entry, using the NIST sodium strong-line wavelengths (`5889.951 Å` and `5895.924 Å` in air)
- The Na diagnostic should be run with a Gaussian profile and narrow practical widths; this is compatible with the current `profile='Gaussian'` option, though the dedicated Na priors still need to be tuned
- MINATO now allows single-epoch `SLfit` runs instead of exiting early; this returns a single-epoch RV estimate plus shared line-profile parameters, but naturally cannot diagnose epoch-to-epoch variability

Current modelling caveat to keep in mind:
- Adding `5896` to the dictionary is enough to register the line, but a proper Na-doublet treatment is still a separate modelling step because the current simultaneous-line likelihood treats each listed line region independently, including overlapping windows

Provisional line-family definitions agreed on `2026-04-17`:
- `blue`
  - `4026, 4102, 4144, 4340, 4388, 4471, 4713`
  - dropped `4861` and `4922` relative to the historical P117 stellar set
  - keep `4102`; the apparent mode C failure was a plotting-axis bug, not a line-list failure
- `red`
  - `5876, 6678`, plus a curated subset of Paschen lines
  - drop `6562` (`Hα`) because of emission contamination concerns
  - recommended first-pass Paschen subset: `9015, 9229, 9546`
  - optional secondary Paschen candidate if the first-pass subset behaves well: `8863`
- `full`
  - start from the combined stellar blue+red set
  - keep `4026, 4102, 4144, 4340, 4388, 4471, 4713, 5876, 6678`
  - add He II lines `4542, 4686, 5412` (these are the current MINATO dictionary keys corresponding to the physical lines usually referred to as He II `4541`, `4686`, `5411`)
  - add the same first-pass Paschen subset: `9015, 9229, 9546`
  - exclude `4861` and `6562`
- `na`
  - `5890, 5896`

Implementation gaps implied by this split:
- The Na branch remains provisional until its overlapping-window handling and narrow-Gaussian priors are cleaned up

Na diagnostic modelling decision agreed on `2026-04-17`:
- Use per-epoch Na RVs for the first-pass diagnostic model
- Do not force one shared Na RV across all epochs at this stage
- Rationale: a shared-across-epochs Na RV would suppress or smear out isolated epoch-level Na shifts, which may be exactly the signal needed to flag suspicious reductions or spurious stellar RVs
- Within each epoch, the two Na lines should share the same RV
- The Na profile should be Gaussian-only
- Widths should be narrow and constrained across epochs rather than left completely free epoch-by-epoch
- Line depths can vary, but should not be given excessively broad freedom

Concrete Na fitting method agreed on `2026-04-17`:
- Implement a dedicated Na-doublet fitter rather than reusing the generic simultaneous-line SB1 machinery
- Fit one shared local wavelength window containing both D lines together, rather than treating `5890` and `5896` as two independent line regions
- Model one effective interstellar Na component per star, observed at low resolution:
  - per epoch:
    - `RV_Na[e]`
    - local continuum intercept and slope
    - one overall Na strength / depth scale
  - shared across epochs:
    - one Gaussian width
    - one D1/D2 depth ratio
- Keep the laboratory wavelengths fixed to the NIST air values for D2 (`5889.951 Å`) and D1 (`5895.924 Å`)
- Use the observed flux errors in the likelihood, but prefer the same robust Student-t likelihood family already used in the probabilistic stellar fits
- Start with a single effective component even though some sightlines may contain multiple ISM clouds
  - rationale: at `R=2000` those components will often be unresolved or only weakly separated, and the diagnostic goal is stable Na RVs rather than detailed ISM decomposition
- If residuals or obviously broadened/asymmetric profiles show that one component is inadequate, only then escalate flagged stars to a multi-component Na model

Recommended practical priors / constraints for the first implementation:
- fit window: one local region spanning both lines, roughly `5878–5899 Å` or slightly wider if needed for continuum placement
- Gaussian width prior centred on the instrumental resolution scale near `5900 Å`
  - at `R=2000`, the instrumental FWHM is about `3 Å`
  - use a narrow prior around that scale rather than the much broader generic stellar-line priors
- D1/D2 ratio should be constrained to a sensible positive range rather than left fully free
- Continuum should be local and per epoch because the diagnostic is sensitive to bad normalisation in this region

Planned MINATO integration path:
- Add a dedicated function, conceptually `fit_na_probmod(...)`, alongside `fit_sb1_probmod(...)`
- Route to it from `SLfit(...)` via a new `sb1_method='na'` mode, rather than overloading the generic `sb1_method='prob'` path
- Keep the top-level call style close to the existing SB1 workflow so the batch runner can swap fit families without bespoke wrappers

Proposed first-pass function shape:
- `fit_na_probmod(wavelengths, fluxes, f_errors, path,`
- `                shift_kms=0, wavelength_type='air', rm_epochs=None,`
- `                num_warmup=..., num_samples=..., num_chains=..., chain_method='parallel',`
- `                window=(5878, 5899), plots=True, cornerplot=True,`
- `                verbose=None, progress=None)`

Expected parameterisation inside the model:
- observed data:
  - one shared local wavelength window per epoch covering both Na lines
- fixed line centres:
  - D2 at `5889.951 Å`
  - D1 at `5895.924 Å`
- stochastic parameters:
  - per epoch:
    - `RV_Na[e]`
    - local continuum slope/intercept
    - overall Na depth scale
  - shared across epochs:
    - Gaussian FWHM
    - D1/D2 depth ratio
- outputs:
  - per-epoch Na RV summary with uncertainties
  - optional Na-only diagnostic plots / corner plots

Reason for keeping this separate from the generic SB1 fitter:
- the generic SB1 model assumes independent line windows and a line-by-line bookkeeping layout that is well suited to stellar lines, but not to one tightly coupled doublet in a shared local continuum window

Implementation status on `2026-04-17`:
- Paschen lines have been added to the default MINATO line dictionary
- `SLfit(..., sb1_method='na')` now routes to a first-pass dedicated Na-doublet probabilistic fitter
- The first implementation is intentionally conservative:
  - air wavelengths only
  - one effective Na component
  - one shared window for the D doublet
  - one RV per epoch

Planned diagnostic interpretation:
- `blue` and `red` agreement is the primary stellar consistency check
- Na is a secondary diagnostic reference, not a truth reference for the stellar systemic velocity
- stable Na with moving stellar RVs is compatible with binarity
- Na that also shifts strongly or behaves erratically points more toward suspicious/systematic cases

## 2026-04-17 - P117-style SB1 throughput optimisation plan

Goal: optimise the old P117-style SB1 batch workflow for a much larger production campaign, where each star will need four separate SB1 fits (`blue`, `red`, `full`, `Na I`) and the limiting metric is stars per hour rather than latency of one fit.

Iteration registry for the next optimisation passes:
- Iteration 0: recover the historic P117 setup and document the new production constraints
  - status: completed on `2026-04-17`
  - outcome: confirmed that the old P117 runner structure is still the correct starting point, but the new campaign must use four fit families per star and no plots
- Iteration 1: warmup/sample-depth validation on a small real subset
  - target: identify the cheapest MCMC depth that preserves RV stability and the spurious-RV diagnostics
  - planned comparison: `100/300`, `200/500`, `300/1000`, `500/2000`
- Iteration 2: worker-packing benchmark on a larger representative subset
  - target: maximise stars/hour/node with the full four-fit workflow
  - focus: `n_workers × n_cpus_per_worker × num_chains`
- Iteration 3: epoch-count batching test
  - target: determine whether grouping stars by number of epochs improves JAX compile reuse and throughput
- Iteration 4: multi-node split
  - target: validate a simple production layout across more than one astro-node once the single-node layout is fixed

Why this is a different problem from the smoke benchmark:
- The `2026-04-16` smoke benchmark only measured one reduced SB1 fit per star on three synthetic binaries with `100/300` warmup/samples and a 9-line bundle
- The historic P117 workflow already uses the right *structure* for large campaigns: multiprocessing across stars, JAX device allocation per worker, and `dict` input to `ravel.SLfit`
- For a production-scale run, total throughput will be controlled by:
  - worker packing across stars
  - repeated HDF5 I/O
  - repeated JAX compilation across different epoch counts and line-set sizes
  - warmup/sample depth needed for robust RVs and spurious-RV detection

Recovered P117 reference configuration:
- Source files:
  - `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/FEROS_followup/P117/work_summary.md`
  - `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/FEROS_followup/P117/run_minato_sb1_batch.py`
- P117 used:
  - star-level multiprocessing via `--n-workers`
  - per-worker JAX device count via `--n-cpus`
  - `num_warmup=500`
  - `num_samples=2000`
  - `num_chains=4`
  - `chain_method='parallel'`
  - `plots=False`
  - `cornerplots=True`
  - Balmer + He I line list, optionally with Paschen lines
- The new large run should differ in two immediate ways:
  - no plots at all, including no corner plots
  - four line-set fits per star rather than one

Important P117 sample statistics for benchmark design:
- `p117_targets.csv`: 185 stars, mean `3.35` epochs/star
- `p117_targets_minato.csv`: 118 stars with successful MINATO measurements
- Epoch-count distribution in `p117_targets_minato.csv`:
  - 2 epochs: 28 stars
  - 3 epochs: 58 stars
  - 4 epochs: 14 stars
  - 5 epochs: 7 stars
  - 6 epochs: 6 stars
  - 7 epochs: 2 stars
  - 8 epochs: 2 stars
  - 10 epochs: 1 star

Current interpretation from the smoke benchmark:
- For the reduced synthetic SB1 workload, best per-fit latency came from:
  - `XLA_FLAGS=--xla_force_host_platform_device_count=2`
  - `num_chains=2`
  - `chain_method='parallel'`
- Larger per-star allocations (`4` or `8` devices/chains) did not improve meaningfully
- This suggests the P117-style runner should probably use *smaller* per-worker JAX allocations and scale out with more workers instead

Planned optimisation iterations:

1. Warmup/sample-depth test on a small but real P117 subset
- Use the real P117 HDF5 input path and the four production fit families
- Compare at least:
  - `100/300`
  - `200/500`
  - `300/1000`
  - `500/2000` as the reference
- Evaluate not just runtime but:
  - `mean_rv`
  - `mean_rv_er`
  - fit failures/pathologies
  - stability of the spurious-RV diagnostics that motivate the four-fit workflow

2. Throughput test on a larger P117 subset
- Use a representative subset from `p117_targets_minato.csv`, not the full 300k-star-scale input yet
- Proposed benchmark subset: 48 stars, stratified by epoch count
  - 16 stars with 2 epochs
  - 16 stars with 3 epochs
  - 8 stars with 4 epochs
  - 8 stars with 5+ epochs
- Purpose: retain the real epoch-count mix while keeping the test manageable

3. Worker-packing benchmark
- Keep the same four-fit workflow per star
- Test combinations like:
  - `n_workers=1`, `n_cpus_per_worker=2`, `num_chains=2`
  - `n_workers=2`, `n_cpus_per_worker=2`, `num_chains=2`
  - `n_workers=4`, `n_cpus_per_worker=2`, `num_chains=2`
  - one comparison against the old heavier style:
    - `n_workers=2`, `n_cpus_per_worker=4`, `num_chains=4`
- Measure:
  - wall-clock per star
  - stars/hour/node
  - peak memory
  - any obvious filesystem contention

4. Shape-batching experiment
- Group stars by epoch count before dispatching them to workers
- Motivation: JAX compilation cost depends strongly on array shape, so batching `2`-epoch and `3`-epoch stars separately may improve compile reuse and throughput
- This is especially relevant because the production run will also switch between four fixed line-set families, each with its own line count

5. Multi-node scaling
- Only after the single-node worker-packing result is understood
- Check actual usage/availability of the astro-nodes first
- Then split the sample across nodes, for example half on one node and half on another, while keeping the validated per-node worker layout fixed

Implementation direction before the large run:
- The batch script should be restructured so that one worker:
  - loads a star once
  - prepares the arrays once
  - runs the four line-set fits sequentially in the same process
- This should reduce repeated HDF5 reads and may allow more JAX compilation reuse than launching four independent star-fit jobs

Current recommended starting point, pending the next benchmark:
- `plots=False`
- `cornerplots=False`
- try `n_cpus_per_worker=2`, `num_chains=2`, `chain_method='parallel'`
- validate warmup/sample depth before locking that in for production

Validation subset fixed on `2026-04-17` before scaling tests:
- File: `benchmarks/p117_validation_subset_20260417.csv`
- Purpose: larger functional check of the four-fit workflow before any throughput benchmark
- Selection design: 10 stars stratified by epoch count and MINATO RV amplitude, with one explicitly suspicious XCSAO-versus-MINATO case
- Selected stars:
  - `74826456`: 2-epoch moderate case
  - `105622073`: 2-epoch high-amplitude case
  - `76657579`: 3-epoch near-threshold case
  - `74829598`: 3-epoch moderate compact-baseline case
  - `74914335`: 3-epoch strong-variability case
  - `73590837`: 3-epoch smoke-test anchor
  - `73957460`: 4-epoch lower-amplitude multi-epoch stability case
  - `82334978`: 4-epoch suspicious control with extreme XCSAO `dRV` but low MINATO `dRV`
  - `82244026`: 5-epoch moderate longer-baseline case
  - `82903408`: 7-epoch high-amplitude higher-cadence stress case
- Working rule for the next iteration:
  - run the full `blue` / `red` / `full` / `na` workflow on this 10-star subset first
  - only after the qualitative behaviour looks stable across this set should the worker-packing benchmark begin

## 2026-04-16 - Launched SB1 astro-node smoke scaling matrix

Goal: execute the first smoke benchmark for `ravel` SB1 probabilistic fits on the PoWR multiepoch `R=2000`, `SNR=25` binary sample and compare chain/device scaling on three representative binaries before expanding to all 10 binaries.

Execution status:
- Smoke matrix launched on `Node-10` in tmux session `minato_sb1_scaling_20260416_run2`
- Wrapper script: `benchmarks/run_ravel_sb1_scaling_smoke.sh`
- Python driver: `benchmarks/run_ravel_sb1_scaling.py`
- Output root: `benchmarks/results/sb1_scaling_smoke_20260416_run2/`
- Top-level logs:
  - `benchmarks/results/sb1_scaling_smoke_20260416_run2/tmux.log`
  - `benchmarks/results/sb1_scaling_smoke_20260416_run2/run.log`

Smoke benchmark definition:
- Dataset: `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/simulations/multiepoch/manifest_R2000_SNR25_powr.csv`
- Representative binary subset:
  - `sim_id=0` (`q=0.74`, `vsini=101.1 km/s`)
  - `sim_id=2` (`q=0.85`, `vsini=328.8 km/s`)
  - `sim_id=6` (`q=0.51`, `vsini=136.4 km/s`)
- Configurations:
  - `N=1`, `num_chains=1`, `chain_method='sequential'`
  - `N=2`, `num_chains=2`, `chain_method='parallel'`
  - `N=4`, `num_chains=4`, `chain_method='parallel'`
  - `N=8`, `num_chains=4`, `chain_method='parallel'`
  - `N=8`, `num_chains=8`, `chain_method='parallel'`
  - `N=8`, `num_chains=4`, `chain_method='sequential'`
- Each configuration is run twice per binary (`repeat1`, `repeat2`) to capture cold/hot behaviour.

Runtime environment:
- `MINATO_QUIET=1`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `JAX_ENABLE_X64=True`
- `MPLCONFIGDIR=/nexus/posix0/MIA-astro-env/hxr/jvillasr/.tmp/matplotlib`
- `XLA_FLAGS=--xla_force_host_platform_device_count=<N>` varied by benchmark case

Sanity result before launch:
- A direct single-case run (`sim_id=0`, `N=1`, `num_chains=1`, sequential, `100/300` warmup/samples) completed successfully in `15.86 s`
- Metadata path: `benchmarks/results/sanity_sim0_run2/sim_0_repeat1_metadata.json`
- Recorded peak memory: `max_rss_kb=1081680`

## 2026-04-16 - SB1 astro-node performance benchmark plan

Goal: benchmark `ravel` SB1 probabilistic fits on the astro-nodes using the same class of low-resolution binary spectra prepared for the SpecFANN paper work, then choose a practical default setup for CPU device count and chain parallelism.

Recovered simulation provenance:
- Source repo notes: `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/NOTES.md`
- Correct dataset for `ravel` timing: `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/simulations/multiepoch/manifest_R2000_SNR25_powr.csv`
- This is the dedicated multi-epoch PoWR binary/single sample created for stacking and binary tests: 20 systems total, 10 binaries + 10 singles, 10 epochs each, `R=2000`, `SNR=25`
- Do not use `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/simulations/snr_tests_20260109_powrfix/manifest_R2000_SNR25.csv` for this benchmark; that PoWR-fix grid is only 20 single-epoch spectra and is for SpecFANN validation, not multi-epoch `ravel`

Benchmark sample definition:
- Smoke subset before full scaling:
  - `sim_id=0`: binary, `q=0.74`, `vsini=101.1 km/s`
  - `sim_id=2`: binary, `q=0.85`, `vsini=328.8 km/s`
  - `sim_id=6`: binary, `q=0.51`, `vsini=136.4 km/s`
- Full benchmark set: all 10 binary systems in the multiepoch manifest (`sim_id=0` to `9`)
- Spectra root: `/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/simulations/multiepoch/spectra_powr_minmax/R2000_SNR25/`

Data-format note:
- The PoWR simulation FITS files are custom binary-table spectra with `loglam`, `FLUX_NORM`, `IVAR_NORM`, and `MJD` in the header/table
- MINATO's built-in FITS reader is instrument-specific (`FLAMES`/`FEROS`), so the benchmark should load these spectra into the `file_type='dict'` interface rather than relying on `read_fits()`

First-pass benchmark settings:
- Hold the scientific workload fixed and vary only execution settings
- Start with the current SB1 tutorial line bundle for comparability and simplicity:
  - `lines = [4026, 4089, 4102, 4144, 4340, 4388, 4471, 4542, 4553]`
- Use:
  - `SB2=False`
  - `plots=False`
  - `cornerplots=False`
  - `verbose=False`
  - `progress=False`
  - fixed `num_warmup`, `num_samples`, `profile`, `Hprofile`, `max_interp_points`
- Suggested initial matrix:
  - baseline: `N=1`, `num_chains=1`, `chain_method='sequential'`
  - scaling: `N=2`, `num_chains=2`, `chain_method='parallel'`
  - scaling: `N=4`, `num_chains=4`, `chain_method='parallel'`
  - scaling: `N=8`, `num_chains=4`, `chain_method='parallel'`
  - scaling: `N=8`, `num_chains=8`, `chain_method='parallel'`
  - control: `N=8`, `num_chains=4`, `chain_method='sequential'`

Proposed astro-node environment:
- `export MINATO_QUIET=1`
- `export OMP_NUM_THREADS=1`
- `export OPENBLAS_NUM_THREADS=1`
- `export MKL_NUM_THREADS=1`
- `export NUMEXPR_NUM_THREADS=1`
- `export JAX_ENABLE_X64=True`
- vary `XLA_FLAGS=--xla_force_host_platform_device_count=<N>` per run

Proposed command template (not yet executed):

```bash
cd /nexus/posix0/MIA-astro-env/hxr/jvillasr/MINATO
export MINATO_QUIET=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export XLA_FLAGS=--xla_force_host_platform_device_count=4

/usr/bin/time -v python - <<'PY'
from pathlib import Path
import csv
import numpy as np
from astropy.io import fits
from minato import ravel

manifest = Path("/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/simulations/multiepoch/manifest_R2000_SNR25_powr.csv")
base_dir = manifest.parent
target_sim_id = "0"  # swap for 2, 6, or loop over 0-9

rows = [r for r in csv.DictReader(manifest.open()) if r["sim_id"] == target_sim_id]
rows.sort(key=lambda r: int(r["epoch"]))

wavelengths, fluxes, f_errors, names, jds = [], [], [], [], []
for row in rows:
    path = base_dir / row["relative_path"]
    with fits.open(path) as hdul:
        data = hdul["SPECTRUM"].data
        header = hdul["SPECTRUM"].header
        wave = 10 ** np.asarray(data["loglam"], dtype=float)
        flux = np.asarray(data["FLUX_NORM"], dtype=float)
        ivar = np.asarray(data["IVAR_NORM"], dtype=float)
        ferr = np.full_like(flux, np.inf, dtype=float)
        good = ivar > 0
        ferr[good] = 1.0 / np.sqrt(ivar[good])
        wavelengths.append(wave)
        fluxes.append(flux)
        f_errors.append(ferr)
        names.append(Path(row["filename"]).stem)
        jds.append(float(header["MJD"]))

spec_dict = {
    "wavelengths": wavelengths,
    "fluxes": fluxes,
    "f_errors": f_errors,
    "names": names,
    "jds": jds,
}

lines = [4026, 4089, 4102, 4144, 4340, 4388, 4471, 4542, 4553]

ravel.SLfit(
    spec_dict,
    data_path="",
    save_path="benchmark_outputs/",
    lines=lines,
    SB2=False,
    file_type="dict",
    plots=False,
    cornerplots=False,
    verbose=False,
    progress=False,
    num_warmup=100,
    num_samples=300,
    num_chains=4,
    chain_method="parallel",
    max_interp_points=150,
    profile="Gaussian",
    Hprofile="Gaussian",
)
PY
```

Measurement plan:
- For each configuration, record wall time and max RSS
- Do one cold run and one immediate hot run; compare hot runs for throughput
- Once the best chain/device setup is identified on the smoke subset, rerun on all 10 binaries
- If the production line list is longer than the tutorial 9-line bundle, do one follow-up spot check with that longer list before fixing defaults
