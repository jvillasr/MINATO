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
- Add/verify minimal smoke tests (imports + one tiny “happy path” run) [][]
- Run tutorials or a curated subset end-to-end and note environment details [][]
- Verify all new APIs have docstrings + basic usage examples [][]
- Audit repository for committed secrets (e.g., `.token`) and remove/rotate if needed [][]
- Clean repo hygiene (remove tracked `.DS_Store`, `__pycache__`, large artifacts) [][]
- Review dependency footprint (`minato_env.yml`) and remove unused pins [][]
- Do a quick “fresh env” install check from `minato_env.yml` [][]
- Decide/record branch policy (e.g., `develop` → `main`, required reviews) [Done][2026-04-16]

## Milestone: Next release (target: `0.3.0`)

- Pick a version number + release date [][]
- Bump `minato/__init__.py::__version__` to the release version [][]
- Move items from `CHANGELOG.md` `[Unreleased]` into a new dated release section [][]
- Ensure “what changed” notes exist for users (README + tutorials where needed) [][]
- Add a short upgrade note if outputs/files changed (paths, CSV formats, column names) [][]
- Run a full validation pass (smoke tests + a representative SB1 + SB2 run) [][]
- Verify `binary_population` MCMC batching behaves as expected (correct normalization) and improves wall-time performance [][]
- PRIORITY: Make `binary_population` inference scalable on large CPUs (process/MPI pool, summary-only simulation, reproducible seeding) and test on laptop + MPIA astro-nodes (small/large scale; 1-parameter `f_bin` and multi-parameter runs) in a dedicated branch [STARTED][2026-06-16]
- Add and validate an experimental mixture/common-random-number likelihood for high-`N_obs` binary-population inference [STARTED][2026-06-21]
- Update `binary_population` tutorial with HPC caveats and “how to run” guidance (threads vs processes vs MPI; choosing `N_sim`, `batch_size`, `nwalkers`, `nsteps`) [][]
- Create/expand a tutorial for binary simulations (`minato.binary_population`) covering the main options and common workflows [Done][2025-12-15]
- Validate and complete the SB1 tutorial: Gaussian vs Voigt (`profile`), impact on results, and a working legacy `lmfit` example [][]
- Benchmark `ravel` SB1 scaling on astro-nodes using the PoWR multiepoch binary sample (`R=2000`, `SNR=25`) and document the recommended `XLA_FLAGS` / `num_chains` / `chain_method` setup [STARTED][2026-04-16]
- Benchmark and optimise the P117-style SB1 batch workflow for large campaigns: 4 fit families per star, no plots, tuned warmup/sample depth, epoch-count batching, and validated `n_workers` × `n_cpus_per_worker` packing before multi-node scale-out [STARTED][2026-04-17]
- Refresh the SB2 tutorial to match current code and include new functionality (two-stage sampling, diagnostics, stitching) [][]
- Make MINATO installable via `pip` (add packaging config like `pyproject.toml`, declare deps, and document install) [][]
- Start the MINATO paper draft: choose workflow (gitignored `paper/` folder vs. dedicated `paper` branch) and add minimal repo guidance [STARTED][2025-12-15]
- Tag the release in git and draft a GitHub release note (if applicable) [][]
- Decide distribution path (source-only, PyPI, conda, internal) and document it [][]

## Backlog (not tied to a milestone yet)

- Add CI (format/lint + tests) [][]
- Add a packaging config (`pyproject.toml`) for `pip install -e .` workflows [][]
- Add SB2 support for component-specific RV uncertainties in `binary_population` (e.g., `rv_errors1`/`rv_errors2`, separate cadence/error columns, optional error ratios) [][]
- SED fitting module (from README planned features) [][]
- Automated spectral classification tooling (from README planned features) [][]
