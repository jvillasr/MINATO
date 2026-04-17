# MINATO Notes

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
