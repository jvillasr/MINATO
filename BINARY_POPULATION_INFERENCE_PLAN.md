# Binary Population Inference Plan

This plan records the larger direction for
`feature/binary-population-mixture-crn-likelihood-20260621`.

The branch has two linked goals:

1. Make binary-population inference computationally usable for large observed samples.
2. Make the likelihood scientifically stronger by using survey time/cadence information, not only the global `dRV_max` histogram.

## Current Status

The branch currently contains an experimental mixture/common-random-number
likelihood and the fast `summary_only=True` likelihood path.

The current MINATO implementation supports:

- fixed random banks for deterministic likelihood evaluation;
- continuous mixture weighting,
  `p_model = (1 - f_bin) p_single + f_bin p_binary`;
- deterministic inverse-CDF transforms for `M1`, `logP`, `q`, and `e`;
- clean access to loaded survey cadence/error templates via
  `BinarySurveySimulator.cadence_templates()`;
- a faster real-cadence `dRV_max` summary path;
- `MixtureCRNLikelihood` and `run_mixture_crn_mcmc`;
- unit/smoke tests for the restored likelihood path.

The paper-side diagnostics showed that:

- the old stochastic likelihood was too noisy for efficient MCMC on the WP1 mock;
- single-bank mixture-CRN fixes sampler sticking and makes likelihood calls deterministic;
- the global `dRV_max` histogram constrains `f_bin` better than `pi`;
- conditioning `dRV_max` on time baseline makes `pi` more informative;
- single-bank baseline-conditioned likelihoods are jagged because finite-bank/bin noise is amplified;
- four-bank averaging before likelihood scoring produced the best diagnostic grid, with a peak near the injected `f_bin=0.6`, `pi=0.1`.

The successful paper-side reference result is:

- four fixed banks: `20260621`, `20260622`, `20260623`, `20260624`;
- `100000` single-star and `100000` binary systems per bank;
- baseline bins: `[0,7)`, `[7,30)`, `[30,100)`, `[100,365)`, `[365,inf)` days;
- best grid point: `f_bin=0.611`, `pi=0.100`.

That four-bank averaged, baseline-conditioned method is not yet implemented in
MINATO proper. It currently exists only in the paper-side grid helper.

## Phase 1: Stabilise Current Branch

- Review the WIP commit scope.
- Decide whether all restored files belong in this branch, especially
  `minato/spdis.py`.
- Clean up the current WIP commit into reviewable commits if needed.
- Keep the current single-bank mixture-CRN likelihood labelled experimental.
- Confirm public API names are acceptable before expanding the interface.

## Phase 2: Performance Foundation

- Keep the fast `summary_only=True` path for real-cadence `dRV_max`.
- Avoid repeated `DataFrame` construction during likelihood evaluation.
- Cache fixed observed histogram state where possible.
- Use fixed random banks so repeated calls at the same parameters are deterministic.
- Support multi-bank evaluation without unnecessary repeated setup work.

## Phase 3: Time/Cadence-Aware Likelihood

The current likelihood is based on a global `dRV_max` histogram.

The next target is:

```text
dRV_max conditioned on survey time/cadence information
```

First implementation target:

- baseline-binned likelihood using bins
  `[0,7)`, `[7,30)`, `[30,100)`, `[100,365)`, `[365,inf)` days.

Future scientific extensions:

- pairwise `Delta MJD/HJD` information, closer to the Sana et al. style;
- number-of-epochs conditioning;
- RV-error conditioning;
- broader cadence-class conditioning if needed.

## Phase 4: Reduce Numerical Noise

- Implement multi-bank averaging in MINATO.
- Average model probabilities or histograms across banks before scoring.
- Do not average log likelihoods after the fact.
- Start with four banks: `20260621`, `20260622`, `20260623`, `20260624`.
- Keep bank-seed diagnostics available.
- If jaggedness remains, test a simulation-uncertainty-aware likelihood rather than only increasing MCMC length.

## Phase 5: Validation

Use mock data first.

Required checks:

- repeated likelihood calls are deterministic;
- one-bank averaged mode matches single-bank mode;
- baseline-bin counts are reproducible and correctly normalised;
- four-bank averaged likelihood returns finite values;
- the four-bank baseline-conditioned grid peaks near the injected truth for the WP1 mock;
- MCMC acceptance and mobility are healthy;
- posterior summaries are stable enough under bank-seed choices;
- final corner plots recover injected `f_bin` and `pi` for the mock.

## Phase 6: MCMC And Science Products

Only after the averaged baseline-conditioned grid is smooth enough:

- run one full MCMC using the averaged baseline-conditioned likelihood;
- make the final corner plot;
- report medians, credible intervals, acceptance fraction, and autocorrelation/ESS if available;
- document fixed parameters, fitted parameters, bank sizes, bank seeds, baseline bins, and caveats.

## Phase 7: Documentation And Merge

- Update `CHANGELOG.md` under `Unreleased`.
- Keep `ROADMAP.md` status aligned with actual progress.
- Add a short usage example or tutorial note covering:
  - survey cadence input;
  - mixture-CRN likelihood;
  - bank averaging;
  - baseline-conditioned likelihood.
- Merge only after the experimental API and validation story are coherent.

## Immediate Next Step

Implement the successful paper-side method in MINATO proper:

```text
four-bank averaged + baseline-conditioned mixture-CRN likelihood
```

Use the paper-side grid evaluator as the reference implementation, but keep the
MINATO API general enough to support simulated survey-coverage tables as well
as real survey cadence tables loaded with `BinarySurveySimulator.load_data(...)`.
