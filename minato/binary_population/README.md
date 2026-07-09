# binary_population

Binary population synthesis and survey simulation for MINATO. This subpackage has three pieces:

- `BinaryPopulation`: draws intrinsic binaries (masses, periods, eccentricities, orientations) with optional fixed-value lists and safety guards (Roche limit, exposure smearing).
- `BinarySurveySimulator`: turns an intrinsic catalogue into “observed” RV time series using real cadence (if provided) or simple synthetic cadences.
- `run_mcmc`: helper that fits the binary fraction, and optionally the period, mass-ratio, and eccentricity power-law indices, using a Poisson likelihood on the distribution of `dRV_max`.

## Quick start (no real cadence)

```python
from minato.binary_population import BinaryPopulation, BinarySurveySimulator

pop = BinaryPopulation()
survey = BinarySurveySimulator(pop)

# Generate an intrinsic catalogue (all binaries here for speed)
intrinsic = pop.generate_intrinsic_sample_vectorized(N=100, f_bin=1.0)

# Simulate two-quadrature observations with a fixed RV error
obs = survey.simulate_mock_observations(
    N=100,
    f_bin=1.0,
    ideal_sampling=True,     # two quadratures
    rv_error_common=1.0      # km/s
)
print(obs[["dRV_max", "sigma_d"]].head())
```

## Using real survey cadence

```python
import pandas as pd
from minato.binary_population import BinaryPopulation, BinarySurveySimulator

# DataFrame with columns: ID, MJD, mean_rv_er
coverage_df = pd.read_csv("path/to/coverage.csv")

pop = BinaryPopulation()
survey = BinarySurveySimulator(pop)
survey.load_data(coverage_df)

obs = survey.simulate_mock_observations(
    N=500,
    f_bin=0.6,
    ideal_sampling=False     # use real cadence/errors
)
```

## Fitting the binary fraction with MCMC

```python
import numpy as np
from minato.binary_population import BinaryPopulation, BinarySurveySimulator, run_mcmc

pop = BinaryPopulation()
survey = BinarySurveySimulator(pop)

# Optional: load coverage if using real cadence
# survey.load_data(coverage_df)

dRV_real = np.loadtxt("path/to/dRV_max_values.txt")  # array of observed max RV spans

sampler = run_mcmc(
    pop,
    survey,
    dRV_real=dRV_real,
    N_sim=100_000,
    batch_size=1000,
    nwalkers=16,
    nsteps=2000,
    nthreads=4,
    # you can pass simulate kwargs, e.g. ideal_sampling=True, rv_error_common=1.0
    sim_kwargs={"ideal_sampling": True, "rv_error_common": 1.0},
)
chain = sampler.get_chain(discard=500, thin=10, flat=True)
print("median f_bin", np.median(chain))
```

## Fitting distribution indices

The default MCMC call is backward compatible and samples only `f_bin`. To fit the
first four-parameter multiplicity model, opt in with `parameter_names`:

```python
from minato.binary_population import BinaryPopulation, BinarySurveySimulator, run_mcmc

pop = BinaryPopulation()
pop.logP_min = 0.15
pop.logP_max = 3.5
pop.logP_powerlaw_mode = "direct"  # p(log10 P) ∝ (log10 P)^pi
pop.q_min = 0.1
pop.q_max = 1.0

survey = BinarySurveySimulator(pop)
survey.load_data(coverage_df)  # columns: ID, MJD, mean_rv_er

sampler = run_mcmc(
    pop,
    survey,
    dRV_real=dRV_real,
    N_sim=100_000,
    batch_size=1000,
    nwalkers=32,
    nsteps=2000,
    parameter_names=("f_bin", "pi", "kappa", "eta"),
    parameter_bounds={
        "f_bin": (0.0, 1.0),
        "pi": (-3.0, 3.0),
        "kappa": (-4.0, 4.0),
        "eta": (-0.95, 4.0),
    },
)
chain = sampler.get_chain(discard=500, thin=10, flat=True)
```

The period sampler has two modes. The default `logP_powerlaw_mode = "shifted"`
preserves older behaviour and samples
`p(x) ∝ (x - logP_min)^pi`, where `x = log10(P/day)`. The direct mode samples
`p(x) ∝ x^pi` on `[logP_min, logP_max]` and requires `logP_min > 0`; this is the
mode to use when comparing `pi` to literature-style period distributions over a
positive log-period interval.

## Experimental pairwise mixture-CRN likelihood

The pairwise CRN likelihood scores a star-balanced summary vector instead of
raw all-pair histograms. For each star and each `Delta t` bin it computes one
response, either `max_pair_significance = max(|Delta RV| / hypot(err_i, err_j))`
or `max_abs_delta_rv`. Missing `Delta t` bins are treated as unsupported cells,
not as zero-valued non-detections.

```python
import numpy as np
from minato.binary_population import (
    PairwiseSummaryConfig,
    run_averaged_mixture_crn_pairwise_mcmc,
)

config = PairwiseSummaryConfig(
    delta_time_bins=(0, 1, 7, 30, 100, 365, 1000, 3000, np.inf),
    response="max_pair_significance",
    response_bins=tuple(np.linspace(0, 20, 41)) + (np.inf,),
)

# Shape: (n_stars, n_delta_time_bins). Unsupported cells should be NaN.
observed_pairwise_summary = np.load("path/to/observed_pairwise_summary.npy")

sampler = run_averaged_mixture_crn_pairwise_mcmc(
    pop,
    survey,
    observed_pairwise_summary,
    n_single_bank=100_000,
    n_binary_bank=100_000,
    bank_seeds=(20260621, 20260622, 20260623, 20260624),
    parameter_names=("f_bin", "pi"),
    pool_kind="static_process",
)
```

This API is experimental. Validate it against the baseline-binned `dRV_max`
likelihood, bank-seed stability, and wall time before using it for paper
claims.

## Optional empirical per-epoch RV blending bias

The CRN likelihoods can consume a caller-supplied empirical RV-bias sampler for
binary epochs. MINATO does not ship or define any study-specific kernel data;
the caller owns the calibration table, filtering, binning, and out-of-range
fallback policy. During the likelihood evaluation MINATO computes
`abs(RV_2,true - RV_1,true)`, calls the supplied sampler, adds the returned
`Delta RV_blend` to the simulated primary epoch RV, and only then computes
`dRV_max` or pairwise summaries.

The sampler must either be callable or provide
`sample_bias(abs_delta_v, f_secondary, u)`, where `u` is the fixed CRN
unit-uniform draw for each epoch. If the sampler needs an effective secondary
flux fraction, pass `blending_flux_fraction` as a scalar or as a callable that
accepts `intrinsic_arrays`, `system_index`, and `n_epochs`.

```python
class MyBlendingKernel:
    metadata = {"source": "my validated calibration table"}

    def sample_bias(self, abs_delta_v, f_secondary, u):
        # Project-owned lookup/sampling logic goes here.
        ...


def effective_secondary_flux_fraction(*, intrinsic_arrays, system_index, n_epochs):
    # Project-owned flux-fraction model goes here.
    return 0.25


sampler = run_averaged_mixture_crn_mcmc(
    pop,
    survey,
    dRV_real,
    bank_seeds=(20260621, 20260622),
    parameter_names=("f_bin", "pi", "kappa", "eta"),
    blending_kernel=MyBlendingKernel(),
    blending_flux_fraction=effective_secondary_flux_fraction,
)
print(sampler.mixture_crn_likelihood.blending_metadata)
```

For process pools, the supplied kernel and flux-fraction callable must be
pickleable. Use `pool_kind="none"` or `pool_kind="thread"` while prototyping
non-pickleable local callables.

## Key options (selected)

- Fixed draws: set `M1_values`, `logP_values`, `q_values`, `e_values`, with `fixed_values_mode` = `"random"` or `"cycle"`.
- Period law: `logP_powerlaw_mode = "shifted"` for backward-compatible shifted-logP draws, or `"direct"` for `p(log10 P) ∝ (log10 P)^pi`.
- Eccentricity caps: `e_max`, `use_period_ecc_cap`; enforce with `fixed_e_enforcement` = `"clip"` or `"error"`.
- Safety: `use_roche_guard`, `roche_margin_frac`; smearing check via `use_smear_flag`, `t_exp_sec`, `dv_smear_limit`.
- Cadence modes: `ideal_sampling=False` (real cadence), `True` (two quadratures), or `"phase_uniform"` (uniform phases, needs `n_epochs` and `rv_error_common`).
- Large averaged CRN MCMC: use `pool_kind="static_process"` to parallelise
  walkers, or `pool_kind="bank_static_process"` to parallelise the
  `walker x bank` likelihood tasks when fitting several population parameters
  with multiple fixed CRN banks.

## Notes

- The old `binary_simulator` name is deprecated; keep using `minato.binary_population` going forward.
- Dependencies: NumPy, pandas, kepler, emcee. In the project’s conda environment you can run examples with `mamba run -n minato python your_script.py`.
