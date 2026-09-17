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

## Validated averaged mixture-CRN likelihood

For larger observed samples, prefer the averaged mixture/common-random-number
likelihood over the older stochastic `run_mcmc` path. It builds fixed random
banks once, pools component counts using their contributing support, and then
scores one Poisson likelihood. This is equivalent to concatenating the fixed
systems into one larger bank; for equal effective bank sizes it reduces to
averaging model probabilities. The current validated reference for
cadence-aware work is baseline-binned `dRV_max`.

```python
import numpy as np
import pandas as pd
from minato.binary_population import (
    BinaryPopulation,
    BinarySurveySimulator,
    run_averaged_mixture_crn_mcmc,
)

coverage_df = pd.read_csv("path/to/coverage.csv")  # columns: ID, MJD, mean_rv_er
dRV_real = np.loadtxt("path/to/dRV_max_values.txt")
observed_baseline_days = np.loadtxt("path/to/baseline_days_values.txt")

pop = BinaryPopulation()
pop.logP_min = 0.15
pop.logP_max = 3.5
pop.logP_powerlaw_mode = "direct"
pop.q_min = 0.1
pop.q_max = 1.0

survey = BinarySurveySimulator(pop)
survey.load_data(coverage_df)

sampler = run_averaged_mixture_crn_mcmc(
    pop,
    survey,
    dRV_real,
    n_single_bank=100_000,
    n_binary_bank=100_000,
    bank_seeds=(20260621, 20260622, 20260623, 20260624),
    parameter_names=("f_bin", "pi", "kappa", "eta"),
    parameter_bounds={
        "f_bin": (0.0, 1.0),
        "pi": (-3.0, 3.0),
        "kappa": (-4.0, 4.0),
        "eta": (-0.95, 4.0),
    },
    fixed_parameters={},
    condition_by="baseline_days",
    baseline_bins=(0, 7, 30, 100, 365, np.inf),
    observed_baseline_days=observed_baseline_days,
    pool_kind="bank_static_process",
    nthreads=48,
)
chain = sampler.get_chain(discard=500, thin=10, flat=True)
print(np.median(chain, axis=0))
```

Use `pool_kind="static_process"` to parallelise over walkers when each walker
can own all banks. Use `pool_kind="bank_static_process"` when multi-bank,
multi-parameter evaluations need the extra `walker x bank` parallelism. Keep
bank sizes, bank counts, walker counts, and process counts modest for tutorial
runs, then scale them on CPU nodes after a small deterministic smoke test. MPI
is not currently exposed by the public runner API.

## Experimental joint dRV/time mixture-CRN likelihood

`run_averaged_joint_drvmax_dtmax_crn_mcmc` retains the epoch separation of the
minimum and maximum measured RV as a second histogram axis. Pass
`condition_by="baseline_days"` to score
`P(dRV_max, dt_at_dRV_max | baseline_bin, theta)` rather than the global joint
distribution. Every observed and simulated system must enter exactly one
baseline, `dRV_max`, and time-separation bin; the non-negative `dRV_max` and
time axes are completed with zero and positive-infinity edges when needed.

```python
from minato.binary_population import run_averaged_joint_drvmax_dtmax_crn_mcmc

sampler = run_averaged_joint_drvmax_dtmax_crn_mcmc(
    pop,
    survey,
    dRV_real,
    dt_at_dRVmax_real,
    observed_baseline_days=observed_baseline_days,
    condition_by="baseline_days",
    baseline_bins=(0, 7, 150, np.inf),
    dt_bins=(0, 1, 7, 30, 100, 365, 1000, 3000, np.inf),
    bank_seeds=(20260621, 20260622, 20260623, 20260624),
    parameter_names=("f_bin", "pi"),
    pool_kind="bank_static_process",
    nthreads=48,
)
```

Banks are combined by pooling their contributing counts, not by averaging
separate log-likelihoods. Marginalising the joint probabilities over the time
axis therefore reproduces the baseline-conditioned `dRV_max` component
probabilities for the same fixed systems. Treat this API as experimental
until its recovery accuracy and runtime have been compared with the validated
baseline-only likelihood for the intended sample.

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
the signed `RV_2,true - RV_1,true` separation, calls the supplied sampler, adds
the returned `Delta RV_blend` to the simulated primary epoch RV, and only then
computes `dRV_max`, its defining-epoch separation, or pairwise summaries.

Signed kernels should provide
`sample_bias_signed(delta_v, f_secondary, u)`, where `u` is the fixed CRN
unit-uniform draw for each epoch. This supports kernels conditioned on
`abs(delta_v)` while preserving the current orbital direction when applying
the sampled correction. Existing callable kernels and objects providing
`sample_bias(abs_delta_v, f_secondary, u)` remain supported and continue to
receive absolute separation. If the sampler needs an effective secondary flux
fraction, pass `blending_flux_fraction` as a scalar or as a callable that
accepts `intrinsic_arrays`, `system_index`, and `n_epochs`.

```python
class MyBlendingKernel:
    metadata = {"source": "my validated calibration table"}

    def sample_bias_signed(self, delta_v, f_secondary, u):
        # Project-owned lookup samples beta from abs(delta_v) and f_secondary.
        beta = ...
        return beta * delta_v


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

### Roche-lobe period guard

The population mass ratio is `q = M2/M1`. Eggleton's lobe fraction takes
`M_star/M_companion`, so component 1 uses `1/q` and component 2 uses `q`.
The guard applies these fractions at periastron and requires
`RL_i >= (1 + roche_margin_frac) * R_i` for both components. This retains the
existing periastron approximation for eccentric systems. For fixed masses
and radii, the minimum period scales as `(1-e)^(-3/2)` within the existing
eccentricity floor.

For `M1=20`, `M2=10` (solar masses), `R1=8`, `R2=5` (solar radii), `e=0`
and `roche_margin_frac=0.1`, the corrected minimum is `1.891284964` days;
the reversed component assignment gave `3.038189393` days. Equal-mass
results are unchanged. Depending on which radius limits the separation,
the correction can increase or decrease the minimum for unequal masses.
Population generation and the mixture/joint and pairwise likelihoods share
this guard; existing saved populations are not modified.

## Notes

- The old `binary_simulator` name is deprecated; keep using `minato.binary_population` going forward.
- Dependencies: NumPy, pandas, kepler, emcee. In the project’s conda environment you can run examples with `mamba run -n minato python your_script.py`.
