# Binary Population Inference Status And Remaining Plan

This document records the completed binary-population mixture/common-random-
number (CRN) work and the remaining MINATO tasks. The original implementation
branch, `feature/binary-population-mixture-crn-likelihood-20260621`, has been
merged into `develop`. The current merged reference is commit `27fdd4e`.

The work had two linked goals:

1. Make binary-population inference computationally usable for large observed
   samples.
2. Strengthen the observational likelihood by using survey time/cadence
   information rather than only a global `dRV_max` histogram.

## Current Status

The core implementation is complete and available from the public
`minato.binary_population` API.

Implemented capabilities include:

- fixed random banks for deterministic likelihood evaluation;
- continuous mixture weighting,
  `p_model = (1 - f_bin) p_single + f_bin p_binary`;
- deterministic inverse-CDF transforms for `M1`, `logP`, `q`, and `e`;
- fast real-cadence `dRV_max` summaries without repeated DataFrame creation;
- `MixtureCRNLikelihood` and `run_mixture_crn_mcmc` for single-bank inference;
- `AveragedMixtureCRNLikelihood` and `run_averaged_mixture_crn_mcmc` for
  averaged multi-bank inference;
- optional conditioning on survey quantities, including baseline-bin
  conditioning;
- `static_process` pooling, which keeps large fixed banks resident in worker
  processes;
- `bank_static_process` pooling, which parallelises the expensive
  `walker x bank` work for multi-parameter inference;
- experimental star-balanced pairwise summaries based on time-binned
  `max |Delta RV|` or RV-error-normalised `max X_ij`;
- an optional caller-supplied per-epoch RV blending-bias hook; and
- focused regression, equivalence, multiprocessing, pairwise, and blending
  tests.

## Validation Completed

The following controlled checks have been completed in the
`multiplicity_paper` reference analysis:

- repeated CRN likelihood calls are deterministic;
- one-bank averaged evaluation agrees with the corresponding single-bank
  evaluation;
- baseline-conditioned probabilities are reproducible and finite;
- four-bank model probabilities are averaged before likelihood scoring, not
  after taking logarithms;
- the public MINATO averaged likelihood exactly reproduces all `1271` points
  of the original paper-side `4 x 100000` baseline-conditioned grid;
- a finer `17 x 25` method-comparison grid with four `100000+100000` banks
  recovers the injected `(f_bin, pi) = (0.600, 0.100)` exactly with the
  baseline-binned `dRV_max` likelihood;
- the same production-scale comparison shows that the tested pairwise
  `max X_ij`, joint `(dRV_max, Delta t_at_dRVmax)`, and Sana et al.-style
  scores do not improve recovery over baseline-binned `dRV_max`;
- full two-parameter baseline-conditioned and global `dRV_max` MCMC runs have
  completed;
- a `3000`-step four-parameter baseline-conditioned mock MCMC has completed
  with `bank_static_process` pooling; and
- paper-side real-data MCMC runs have exercised the generic blending hook and
  shared-cadence optimisation at four-bank scale.

The reference four-bank configuration remains:

- bank seeds: `20260621`, `20260622`, `20260623`, `20260624`;
- `100000` single-star and `100000` binary systems per bank; and
- baseline bins: `[0,7)`, `[7,30)`, `[30,100)`, `[100,365)`,
  `[365,inf)` days.

These values describe the validated reference case. They are not hard-coded
requirements of the public API.

## Method Status

### Validated reference

Baseline-binned `dRV_max` is the current reference likelihood for controlled
recovery and the first real-data analysis. It retains one robust `dRV_max`
summary per star and conditions its distribution on the star's total survey
baseline.

### Regression/control

Global `dRV_max` remains useful as a cadence-blind control and backwards-
compatibility check. It is not the preferred production constraint when
baseline information is available.

### Experimental

The pairwise CRN API is implemented and has passed mechanics, collapse, and
production-scale comparison tests. The currently tested time-binned
`max X_ij` scoring form did not outperform baseline-binned `dRV_max`, so it
remains experimental and should not be presented as the default likelihood.

The caller-supplied blending hook is also optional. Its numerical behaviour is
defined by the external kernel supplied by the caller; study-specific kernel
data and flux-fraction choices do not belong in MINATO core.

## Completed Phases

1. **Branch stabilisation:** completed and merged into `develop`.
2. **Performance foundation:** completed for the current fixed-bank workloads.
3. **Time/cadence-aware likelihood:** baseline conditioning is implemented;
   pairwise summaries are available experimentally.
4. **Numerical-noise reduction:** averaged multi-bank scoring is implemented,
   with both static and bank-parallel process pools.
5. **Controlled validation:** deterministic, equivalence, mock-recovery, and
   production-scale method-comparison gates are complete.
6. **MCMC products:** two- and four-parameter mock chains have completed;
   study-specific real-data interpretation remains paper-side work.
7. **Merge:** the implementation and performance changes are merged into
   `develop` and recorded in `CHANGELOG.md` and `ROADMAP.md`.

## Remaining MINATO Work

### 1. User documentation

Update the binary-population tutorial and complete the module README with one
compact end-to-end example covering:

- survey cadence input;
- fixed-bank mixture-CRN construction;
- averaged multi-bank likelihood evaluation;
- baseline conditioning through `condition_by` and `baseline_bins`;
- choosing between `static_process` and `bank_static_process`;
- fitted versus fixed population parameters; and
- the optional blending-kernel interface.

The documentation should identify baseline-binned `dRV_max` as the validated
reference and pairwise likelihoods as experimental.

### 2. Release validation

Before the next release:

- run the focused binary-population tests in the intended release environment;
- run one small tutorial-scale averaged baseline-conditioned inference;
- verify public docstrings and exported API names;
- record practical memory and CPU guidance for bank size, bank count, walker
  count, step count, and pool choice; and
- include the binary-population changes in the release notes.

### 3. Optional future research

Further pairwise work should start from a specific failure hypothesis, not by
adding more summary variants. Possible future directions include:

- richer per-star pairwise response vectors that retain more continuous
  information without weighting high-epoch stars by their raw pair count;
- explicit number-of-epochs or cadence-class conditioning;
- RV-error and cadence-pattern robustness tests; and
- likelihoods that account for finite simulation uncertainty.

These are research extensions rather than blockers for the current
baseline-binned `dRV_max` workflow.

## Paper-Side Scientific Gates

The following tasks use MINATO but are owned by the science analysis rather
than MINATO core:

- diagnose convergence and identifiability in the completed four-parameter
  mock chain;
- rerun the first LMC real-data inference with wider `pi` support;
- calibrate finite-sample reliability before reporting subtype-separated
  constraints; and
- prepare the main-method description and controlled method-comparison
  appendix.

## Immediate Next Step

Update the binary-population tutorial and module documentation to expose the
merged averaged, baseline-conditioned, and parallel APIs with clear method-
status labels.
