# MINATO Roadmap

Milestone-oriented to-do list for MINATO.

## Status convention

Each item ends with `[][]`.

- First bracket: fill with `STARTED` (optional) or `Done` when completed
- Second bracket: fill with completion date (`YYYY-MM-DD`)

Example: `- Add smoke-test script [Done][2025-12-15]`

## Milestone: Ready to merge to `main`

- Confirm feature scope agreed (no last-minute additions) [][]
- Ensure `CHANGELOG.md` `[Unreleased]` is accurate and complete [][]
- Add/verify minimal smoke tests (imports + one tiny “happy path” run) [Done][2026-07-12]
- Run tutorials or a curated subset end-to-end and note environment details [STARTED][2026-07-12]
- Verify all new APIs have docstrings + basic usage examples [][]
- Audit repository for committed secrets (e.g., `.token`) and remove/rotate if needed [][]
- Clean repo hygiene (remove tracked `.DS_Store`, `__pycache__`, large artifacts) [][]
- Review dependency footprint (`pyproject.toml`, `minato_env.yml`, and lock files) and remove unused requirements [STARTED][2026-07-12]
- Do fresh locked-environment install checks with mamba, uv, and Pixi [Done][2026-07-12]
- Decide/record branch policy (e.g., `develop` → `main`, required reviews) [Done][2026-04-16]

## Milestone: Next release (target: `0.3.0`)

- Approve version `0.3.0` and define the release date as the `develop`-to-`main` merge date [Done][2026-07-12]
- Approve the `minato-astro` distribution name while preserving `import minato` [Done][2026-07-12]
- Add a high-level `0.3.0` preview and release-status section to the main README [Done][2026-07-13]
- Add and validate reproducible mamba, uv, and Pixi workflows; commit exact lock files for supported platforms [Done][2026-07-12]
- Move the development baseline to Python 3.13, support Python 3.12-3.13, and remove Python 3.10-only dependencies [Done][2026-07-12]
- Bump `minato/__init__.py::__version__` to the release version [][]
- Move items from `CHANGELOG.md` `[Unreleased]` into a new dated release section [][]
- Ensure “what changed” notes exist for users (README + tutorials where needed) [STARTED][2026-07-12]
- Add a short upgrade note if outputs/files changed (paths, CSV formats, column names) [][]
- Run a full validation pass (smoke tests + a representative SB1 + SB2 run) [STARTED][2026-07-12]
- Add reduced synthetic probabilistic SB1/SB2 release smoke tests to CI [Done][2026-07-12]
- Verify `binary_population` MCMC batching behaves as expected (correct normalization) and improves wall-time performance [][]
- PRIORITY: Make `binary_population` inference scalable on large CPUs (process/MPI pool, summary-only simulation, reproducible seeding) and test on laptop + MPIA astro-nodes (small/large scale; 1-parameter `f_bin` and multi-parameter runs) in a dedicated branch [STARTED][2026-06-16]
- Add and validate an experimental mixture/common-random-number likelihood for high-`N_obs` binary-population inference [Done][2026-07-08]
- Optimise empirical-blending mixture-CRN binary-bank evaluation by grouping
  systems with shared cadence templates while preserving fixed-bank numerical
  outputs [Done][2026-07-11]
- Add an experimental star-balanced pairwise mixture-CRN likelihood with a
  `dRV_max` collapse regression test [Done][2026-07-09]
- Follow the larger binary-population inference branch plan in `BINARY_POPULATION_INFERENCE_PLAN.md`, including multi-bank averaging and time/cadence-conditioned likelihoods [Done][2026-07-08]
- Validate the pairwise mixture-CRN likelihood against the baseline-binned
  `dRV_max` reference at production scale [Done][2026-07-10]
- Reconcile `BINARY_POPULATION_INFERENCE_PLAN.md` with the merged public APIs,
  completed controlled validation, and remaining documentation/release work
  [Done][2026-07-12]
- Update `binary_population` tutorial with HPC caveats and “how to run” guidance (threads vs processes vs MPI; choosing `N_sim`, `batch_size`, `nwalkers`, `nsteps`) [STARTED][2026-07-12]
- Create/expand a tutorial for binary simulations (`minato.binary_population`) covering the main options and common workflows [Done][2025-12-15]
- Validate and complete the SB1 tutorial: Gaussian vs Voigt (`profile`), impact on results, and a working legacy `lmfit` example [][]
- Benchmark `ravel` SB1 scaling on astro-nodes using the PoWR multiepoch binary sample (`R=2000`, `SNR=25`) and document the recommended `XLA_FLAGS` / `num_chains` / `chain_method` setup [STARTED][2026-04-16]
- Benchmark and optimise the P117-style SB1 batch workflow for large campaigns: 4 fit families per star, no plots, tuned warmup/sample depth, epoch-count batching, and validated `n_workers` × `n_cpus_per_worker` packing before multi-node scale-out [STARTED][2026-04-17]
- Refresh the SB2 tutorial to match current code and include new functionality (two-stage sampling, diagnostics, stitching) [STARTED][2026-07-12]
- Make MINATO installable via `pip`, validate wheel/sdist contents, and publish the approved distribution through TestPyPI and PyPI [STARTED][2026-07-12]
- Enforce data-free wheel and source archives, and remove atmosphere models,
  observed spectra, fitted results, and unapproved generated assets from the
  pending `main` release tree [STARTED][2026-07-13]
- Rewrite the SPAN tutorial around user-downloaded PoWR models and two
  provenance-tracked synthetic disentangled fixtures; rewrite RAVEL inputs
  around spectra generated synthetically during execution [STARTED][2026-07-13]
- Classify the shift-and-add adaptation as development-only external code, add upstream attribution, and exclude it from MINATO packages and releases [Done][2026-07-13]
- Recover useful `Observing/` sources on `develop`, then redesign them as a tested `minato.observing` package before any release [Done][2026-07-12]
- Create a small Sphinx/MyST-NB documentation site and evaluate Read the Docs builds after the release tutorial set is clean [STARTED][2026-07-12]
- Start the MINATO paper draft: choose workflow (gitignored `paper/` folder vs. dedicated `paper` branch) and add minimal repo guidance [STARTED][2025-12-15]
- Tag the release in git and draft a GitHub release note (if applicable) [][]
- Decide distribution path (GitHub source, PyPI, conda, or internal) and document it [STARTED][2026-07-12]

## Backlog (not tied to a milestone yet)

- Add CI for supported-Python tests, RAVEL smoke tests, lock checking, and package builds [Done][2026-07-12]
- Add a packaging config (`pyproject.toml`) for `pip install -e .` workflows [Done][2026-07-12]
- Add SB2 support for component-specific RV uncertainties in `binary_population` (e.g., `rv_errors1`/`rv_errors2`, separate cadence/error columns, optional error ratios) [][]
- Add a `binary_population` survey-coverage generator that can create simulated `ID`/`MJD`/`mean_rv_er` cadence tables from a high-level survey design, so users do not need to hand-build coverage inputs before calling `BinarySurveySimulator.load_data(...)` [][]
- SED fitting module (from README planned features) [][]
- Automated spectral classification tooling (from README planned features) [][]
