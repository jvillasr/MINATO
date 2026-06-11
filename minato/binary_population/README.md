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

## Key options (selected)

- Fixed draws: set `M1_values`, `logP_values`, `q_values`, `e_values`, with `fixed_values_mode` = `"random"` or `"cycle"`.
- Period law: `logP_powerlaw_mode = "shifted"` for backward-compatible shifted-logP draws, or `"direct"` for `p(log10 P) ∝ (log10 P)^pi`.
- Eccentricity caps: `e_max`, `use_period_ecc_cap`; enforce with `fixed_e_enforcement` = `"clip"` or `"error"`.
- Safety: `use_roche_guard`, `roche_margin_frac`; smearing check via `use_smear_flag`, `t_exp_sec`, `dv_smear_limit`.
- Cadence modes: `ideal_sampling=False` (real cadence), `True` (two quadratures), or `"phase_uniform"` (uniform phases, needs `n_epochs` and `rv_error_common`).

## Notes

- The old `binary_simulator` name is deprecated; keep using `minato.binary_population` going forward.
- Dependencies: NumPy, pandas, kepler, emcee. In the project’s conda environment you can run examples with `mamba run -n minato python your_script.py`.
