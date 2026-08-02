"""Joint ``dRV_max`` and time-at-maximum mixture-CRN likelihoods."""

from __future__ import annotations

import numpy as np

from .histograms import complete_nonnegative_bins, validate_nonnegative_finite
from .mcmc import (
    _env_truthy,
    _initial_walker_positions,
    _normalise_parameter_bounds,
    _normalise_parameter_names,
)
from .mixture_crn import (
    MixtureCRNLikelihood,
    _assign_condition_bins,
    _edge_label,
    _make_emcee_pool,
    _normalise_bank_seeds,
    _normalise_condition_by,
    _normalise_condition_edges,
    _template_baselines,
)


DEFAULT_DT_AT_DRVMAX_BINS = np.array(
    [0.0, 1.0, 7.0, 30.0, 100.0, 365.0, 1000.0, 3000.0, np.inf]
)


def _validate_observed_summaries(drv, dt):
    drv = validate_nonnegative_finite(drv, name="observed dRV_max")
    dt = validate_nonnegative_finite(dt, name="observed dt_at_dRVmax")
    if drv.ndim != 1 or dt.ndim != 1 or drv.shape != dt.shape:
        raise ValueError(
            "dRV_real and dt_at_dRVmax_real must be one-dimensional arrays "
            "with the same shape."
        )
    return drv, dt


def _joint_histogram(drv, dt, drv_bins, dt_bins, *, label):
    drv = validate_nonnegative_finite(drv, name=f"{label} dRV_max")
    dt = validate_nonnegative_finite(dt, name=f"{label} dt_at_dRVmax")
    if drv.shape != dt.shape:
        raise ValueError(f"{label} dRV_max and dt_at_dRVmax must have the same shape")
    histogram, _, _ = np.histogram2d(drv, dt, bins=(drv_bins, dt_bins))
    if int(histogram.sum()) != int(drv.size):
        raise RuntimeError(
            f"{label} joint histogram assigned {int(histogram.sum())} of "
            f"{int(drv.size)} systems"
        )
    return histogram


def _joint_histogram_by_condition(
    drv,
    dt,
    condition_indices,
    n_conditions,
    drv_bins,
    dt_bins,
    *,
    label,
):
    drv = validate_nonnegative_finite(drv, name=f"{label} dRV_max")
    dt = validate_nonnegative_finite(dt, name=f"{label} dt_at_dRVmax")
    condition_indices = np.asarray(condition_indices, dtype=np.int32)
    if drv.shape != dt.shape or drv.shape != condition_indices.shape:
        raise ValueError(f"{label} summaries and condition indices must have the same shape")
    invalid = (condition_indices < 0) | (condition_indices >= int(n_conditions))
    if np.any(invalid):
        raise ValueError(
            "Every joint-summary system must enter exactly one condition bin; "
            f"found {int(np.count_nonzero(invalid))} unassigned system(s)"
        )
    histogram = np.zeros(
        (int(n_conditions), len(drv_bins) - 1, len(dt_bins) - 1),
        dtype=float,
    )
    for condition_index in range(int(n_conditions)):
        mask = condition_indices == condition_index
        if np.any(mask):
            histogram[condition_index] = _joint_histogram(
                drv[mask],
                dt[mask],
                drv_bins,
                dt_bins,
                label=f"{label} condition {condition_index}",
            )
    if int(histogram.sum()) != int(drv.size):
        raise RuntimeError(f"{label} conditioned joint histogram did not conserve all systems")
    return histogram


def _normalise_joint_histograms(histogram, support_counts):
    probability = np.asarray(histogram, dtype=float).copy()
    support_counts = np.asarray(support_counts, dtype=float)
    if probability.shape[:-2] != support_counts.shape:
        raise ValueError("Joint support counts do not match the non-histogram axes")
    np.divide(
        probability,
        support_counts[..., None, None],
        out=probability,
        where=support_counts[..., None, None] > 0.0,
    )
    return probability


def _pool_joint_probability_tables(probabilities, support_counts, *, label):
    """Pool bank probabilities using system counts, preserving two histogram axes."""
    probability_array = np.asarray(probabilities, dtype=float)
    support_array = np.asarray(support_counts, dtype=float)
    if probability_array.ndim < 3:
        raise ValueError(f"{label} probabilities must include bank, dRV and time axes")
    if support_array.shape != probability_array.shape[:-2]:
        raise ValueError(
            f"{label} support counts must match every probability axis except dRV and time"
        )
    if np.any(~np.isfinite(probability_array)) or np.any(probability_array < 0.0):
        raise ValueError(f"{label} probabilities must be finite and non-negative")
    if np.any(~np.isfinite(support_array)) or np.any(support_array < 0.0):
        raise ValueError(f"{label} support counts must be finite and non-negative")
    total_support = np.sum(support_array, axis=0)
    pooled_counts = np.sum(
        probability_array * support_array[..., None, None],
        axis=0,
    )
    pooled = np.zeros_like(pooled_counts, dtype=float)
    np.divide(
        pooled_counts,
        total_support[..., None, None],
        out=pooled,
        where=total_support[..., None, None] > 0.0,
    )
    return pooled


class JointDrvmaxDtmaxCRNLikelihood(MixtureCRNLikelihood):
    """Deterministic joint likelihood, optionally conditioned on survey baseline."""

    def __init__(
        self,
        survey,
        dRV_real,
        dt_at_dRVmax_real,
        n_single_bank=100000,
        n_binary_bank=100000,
        bank_seed=None,
        single_bank=None,
        binary_bank=None,
        parameter_names=None,
        parameter_bounds=None,
        fixed_parameters=None,
        drv_bins=None,
        dt_bins=None,
        condition_by=None,
        baseline_bins=None,
        observed_baseline_days=None,
        blending_kernel=None,
        blending_flux_fraction=None,
    ):
        dRV_real, dt_at_dRVmax_real = _validate_observed_summaries(
            dRV_real,
            dt_at_dRVmax_real,
        )
        self.dt_at_dRVmax_real = dt_at_dRVmax_real
        self.dt_bins = complete_nonnegative_bins(
            DEFAULT_DT_AT_DRVMAX_BINS if dt_bins is None else dt_bins
        )
        self.condition_by = _normalise_condition_by(condition_by)
        self.condition_edges = (
            None
            if self.condition_by is None
            else _normalise_condition_edges(baseline_bins)
        )
        super().__init__(
            survey,
            dRV_real,
            n_single_bank=n_single_bank,
            n_binary_bank=n_binary_bank,
            bank_seed=bank_seed,
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            fixed_parameters=fixed_parameters,
            bins=drv_bins,
            blending_kernel=blending_kernel,
            blending_flux_fraction=blending_flux_fraction,
        )
        self.drv_bins = self.bins
        self.single_drv, self.single_dt_at_dRVmax = (
            self._simulate_single_drv_and_dt_at_max()
        )
        if self.condition_by is None:
            self.condition_labels = []
            self.n_conditions = 0
            self.observed_counts = None
            self.observed_condition_indices = None
            self.single_condition_indices = None
            self.binary_condition_indices = None
            self.single_support_counts = float(self.single_bank.size)
            self.binary_support_counts = float(self.binary_bank.size)
            self.observed_hist = _joint_histogram(
                self.dRV_real,
                self.dt_at_dRVmax_real,
                self.drv_bins,
                self.dt_bins,
                label="observed",
            )
            self.single_hist = _joint_histogram(
                self.single_drv,
                self.single_dt_at_dRVmax,
                self.drv_bins,
                self.dt_bins,
                label="single bank",
            )
            self.single_probability = self.single_hist / float(self.single_bank.size)
        else:
            self._initialise_baseline_conditioning(observed_baseline_days)
        self.n_real = self.observed_hist

    def _observed_baselines(self, observed_baseline_days):
        if observed_baseline_days is not None:
            values = validate_nonnegative_finite(
                observed_baseline_days,
                name="observed baseline_days",
            )
            if values.shape != self.dRV_real.shape:
                raise ValueError(
                    "observed_baseline_days must have the same shape as dRV_real"
                )
            return values
        template_baselines = _template_baselines(self.template_mjds)
        if template_baselines.shape == self.dRV_real.shape:
            return template_baselines
        raise ValueError(
            "observed_baseline_days is required unless the observed rows are "
            "one-to-one with the survey cadence templates"
        )

    def _bank_condition_indices(self, bank):
        template_baselines = _template_baselines(self.template_mjds)
        template_indices = np.asarray(bank.cadence.template_indices, dtype=np.int64)
        indices = _assign_condition_bins(
            template_baselines[template_indices],
            self.condition_edges,
        )
        if np.any(indices < 0):
            raise ValueError("Every simulated system must enter one baseline bin")
        return indices

    def _initialise_baseline_conditioning(self, observed_baseline_days):
        self.condition_labels = [
            _edge_label(self.condition_edges, index)
            for index in range(len(self.condition_edges) - 1)
        ]
        self.n_conditions = len(self.condition_labels)
        self.observed_condition_indices = _assign_condition_bins(
            self._observed_baselines(observed_baseline_days),
            self.condition_edges,
        )
        if np.any(self.observed_condition_indices < 0):
            raise ValueError("Every observed system must enter one baseline bin")
        self.single_condition_indices = self._bank_condition_indices(self.single_bank)
        self.binary_condition_indices = self._bank_condition_indices(self.binary_bank)
        self.observed_counts = np.bincount(
            self.observed_condition_indices,
            minlength=self.n_conditions,
        ).astype(float)
        self.single_support_counts = np.bincount(
            self.single_condition_indices,
            minlength=self.n_conditions,
        ).astype(float)
        self.binary_support_counts = np.bincount(
            self.binary_condition_indices,
            minlength=self.n_conditions,
        ).astype(float)
        self.observed_hist = _joint_histogram_by_condition(
            self.dRV_real,
            self.dt_at_dRVmax_real,
            self.observed_condition_indices,
            self.n_conditions,
            self.drv_bins,
            self.dt_bins,
            label="observed",
        )
        self.single_hist = _joint_histogram_by_condition(
            self.single_drv,
            self.single_dt_at_dRVmax,
            self.single_condition_indices,
            self.n_conditions,
            self.drv_bins,
            self.dt_bins,
            label="single bank",
        )
        self.single_probability = _normalise_joint_histograms(
            self.single_hist,
            self.single_support_counts,
        )

    def component_probabilities(self, params):
        binary_drv, binary_dt = self.simulate_binary_drv_and_dt_at_max(params)
        if self.condition_by is None:
            binary_hist = _joint_histogram(
                binary_drv,
                binary_dt,
                self.drv_bins,
                self.dt_bins,
                label="binary bank",
            )
            binary_probability = binary_hist / float(self.binary_bank.size)
        else:
            binary_hist = _joint_histogram_by_condition(
                binary_drv,
                binary_dt,
                self.binary_condition_indices,
                self.n_conditions,
                self.drv_bins,
                self.dt_bins,
                label="binary bank",
            )
            binary_probability = _normalise_joint_histograms(
                binary_hist,
                self.binary_support_counts,
            )
        return self.single_probability, binary_probability

    def expected_counts_from_binary_probability(self, params, binary_probability):
        f_bin = float(params["f_bin"])
        probability = (1.0 - f_bin) * self.single_probability + f_bin * binary_probability
        if self.condition_by is None:
            return float(self.N_obs) * probability
        return self.observed_counts[:, None, None] * probability

    def expected_counts(self, params):
        return self.expected_counts_from_binary_probability(
            params,
            self.component_probabilities(params)[1],
        )

    def log_likelihood_from_binary_probability(self, params, binary_probability):
        expected = self.expected_counts_from_binary_probability(
            params,
            binary_probability,
        ) + 1e-8
        return float(np.sum(self.observed_hist * np.log(expected) - expected))

    def log_likelihood(self, params):
        return self.log_likelihood_from_binary_probability(
            params,
            self.component_probabilities(params)[1],
        )

    def condition_summary(self):
        if self.condition_by is None:
            return []
        return [
            {
                "condition": label,
                "observed_count": int(self.observed_counts[index]),
                "single_bank_count": int(self.single_support_counts[index]),
                "binary_bank_count": int(self.binary_support_counts[index]),
                "n_banks": 1,
            }
            for index, label in enumerate(self.condition_labels)
        ]

    def delta_time_summary(self):
        observed_counts = np.sum(self.observed_hist, axis=tuple(range(self.observed_hist.ndim - 1)))
        single_counts = np.sum(self.single_hist, axis=tuple(range(self.single_hist.ndim - 1)))
        rows = []
        for index in range(self.dt_bins.size - 1):
            upper = self.dt_bins[index + 1]
            rows.append(
                {
                    "lower_days": float(self.dt_bins[index]),
                    "upper_days": None if np.isinf(upper) else float(upper),
                    "observed_count": int(observed_counts[index]),
                    "single_bank_count": int(single_counts[index]),
                    "binary_bank_size": int(self.binary_bank.size),
                }
            )
        return rows


class AveragedJointDrvmaxDtmaxCRNLikelihood:
    """Count-pooled joint likelihood across one or more fixed random banks."""

    def __init__(
        self,
        survey,
        dRV_real,
        dt_at_dRVmax_real,
        n_single_bank=100000,
        n_binary_bank=100000,
        bank_seed=None,
        bank_seeds=None,
        parameter_names=None,
        parameter_bounds=None,
        fixed_parameters=None,
        drv_bins=None,
        dt_bins=None,
        condition_by=None,
        baseline_bins=None,
        observed_baseline_days=None,
        likelihoods=None,
        blending_kernel=None,
        blending_flux_fraction=None,
    ):
        self.dRV_real, self.dt_at_dRVmax_real = _validate_observed_summaries(
            dRV_real,
            dt_at_dRVmax_real,
        )
        self.N_obs = int(self.dRV_real.size)
        self.parameter_names = _normalise_parameter_names(parameter_names)
        self.parameter_bounds = _normalise_parameter_bounds(
            self.parameter_names,
            parameter_bounds=parameter_bounds,
        )
        self.condition_by = _normalise_condition_by(condition_by)
        seeds = _normalise_bank_seeds(bank_seeds, bank_seed=bank_seed)
        if likelihoods is None:
            self.likelihoods = [
                JointDrvmaxDtmaxCRNLikelihood(
                    survey,
                    self.dRV_real,
                    self.dt_at_dRVmax_real,
                    n_single_bank=n_single_bank,
                    n_binary_bank=n_binary_bank,
                    bank_seed=seed,
                    parameter_names=self.parameter_names,
                    parameter_bounds=self.parameter_bounds,
                    fixed_parameters=fixed_parameters,
                    drv_bins=drv_bins,
                    dt_bins=dt_bins,
                    condition_by=self.condition_by,
                    baseline_bins=baseline_bins,
                    observed_baseline_days=observed_baseline_days,
                    blending_kernel=blending_kernel,
                    blending_flux_fraction=blending_flux_fraction,
                )
                for seed in seeds
            ]
        else:
            self.likelihoods = list(likelihoods)
            if not self.likelihoods:
                raise ValueError("likelihoods must contain at least one joint likelihood")
        self.bank_seeds = tuple(likelihood.bank_seed for likelihood in self.likelihoods)
        self.reference = self.likelihoods[0]
        self.survey = survey
        self.population = survey.population
        self.drv_bins = self.reference.drv_bins
        self.bins = self.drv_bins
        self.dt_bins = self.reference.dt_bins
        self.fixed_parameters = dict(self.reference.fixed_parameters)
        self.condition_edges = self.reference.condition_edges
        self.condition_labels = list(self.reference.condition_labels)
        self.n_conditions = self.reference.n_conditions
        self.observed_counts = self.reference.observed_counts
        self.observed_hist = self.reference.observed_hist
        self.n_real = self.observed_hist
        self.blending_metadata = dict(self.reference.blending_metadata)
        self._validate_likelihoods()
        if self.condition_by is None:
            single_support = [likelihood.single_bank.size for likelihood in self.likelihoods]
        else:
            single_support = [
                likelihood.single_support_counts for likelihood in self.likelihoods
            ]
            binary_support = np.sum(
                [likelihood.binary_support_counts for likelihood in self.likelihoods],
                axis=0,
            )
            needed = self.observed_counts > 0
            if np.any((np.sum(single_support, axis=0) <= 0) & needed):
                raise ValueError("Pooled single banks lack support in an observed baseline bin")
            if np.any((binary_support <= 0) & needed):
                raise ValueError("Pooled binary banks lack support in an observed baseline bin")
        self.single_probability = _pool_joint_probability_tables(
            [likelihood.single_probability for likelihood in self.likelihoods],
            single_support,
            label="single-bank joint",
        )

    @property
    def n_banks(self):
        return len(self.likelihoods)

    def _validate_likelihoods(self):
        for likelihood in self.likelihoods:
            if likelihood.parameter_names != self.parameter_names:
                raise ValueError("All joint likelihoods must use the same fitted parameters")
            if likelihood.parameter_bounds != self.parameter_bounds:
                raise ValueError("All joint likelihoods must use the same parameter bounds")
            if likelihood.fixed_parameters != self.fixed_parameters:
                raise ValueError("All joint likelihoods must use the same fixed parameters")
            if likelihood.condition_by != self.condition_by:
                raise ValueError("All joint likelihoods must use the same conditioning")
            if not np.array_equal(likelihood.drv_bins, self.drv_bins):
                raise ValueError("All joint likelihoods must use the same dRV bins")
            if not np.array_equal(likelihood.dt_bins, self.dt_bins):
                raise ValueError("All joint likelihoods must use the same time bins")
            if not np.array_equal(likelihood.observed_hist, self.observed_hist):
                raise ValueError("All joint likelihoods must use the same observed histogram")
            if self.condition_by is not None and not np.array_equal(
                likelihood.condition_edges,
                self.condition_edges,
            ):
                raise ValueError("All joint likelihoods must use the same condition bins")

    def _theta_to_params(self, theta):
        return self.reference._theta_to_params(theta)

    def bank_binary_probability(self, bank_index, params):
        bank_index = int(bank_index)
        if bank_index < 0 or bank_index >= self.n_banks:
            raise ValueError(f"bank_index={bank_index} is outside the available banks")
        return self.likelihoods[bank_index].component_probabilities(params)[1]

    def combine_binary_probabilities(self, probabilities):
        if self.condition_by is None:
            support = [likelihood.binary_bank.size for likelihood in self.likelihoods]
        else:
            support = [
                likelihood.binary_support_counts for likelihood in self.likelihoods
            ]
        return _pool_joint_probability_tables(
            probabilities,
            support,
            label="binary-bank joint",
        )

    def component_probabilities(self, params):
        binary_probability = self.combine_binary_probabilities(
            [self.bank_binary_probability(index, params) for index in range(self.n_banks)]
        )
        return self.single_probability, binary_probability

    def expected_counts_from_binary_probability(self, params, binary_probability):
        f_bin = float(params["f_bin"])
        probability = (1.0 - f_bin) * self.single_probability + f_bin * binary_probability
        if self.condition_by is None:
            return float(self.N_obs) * probability
        return self.observed_counts[:, None, None] * probability

    def expected_counts(self, params):
        return self.expected_counts_from_binary_probability(
            params,
            self.component_probabilities(params)[1],
        )

    def log_likelihood_from_binary_probability(self, params, binary_probability):
        expected = self.expected_counts_from_binary_probability(
            params,
            binary_probability,
        ) + 1e-8
        return float(np.sum(self.observed_hist * np.log(expected) - expected))

    def log_likelihood(self, params):
        return self.log_likelihood_from_binary_probability(
            params,
            self.component_probabilities(params)[1],
        )

    def __call__(self, theta):
        params = self._theta_to_params(theta)
        if params is None:
            return -np.inf
        try:
            return self.log_likelihood(params)
        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf

    def condition_summary(self):
        if self.condition_by is None:
            return []
        return [
            {
                "condition": label,
                "observed_count": int(self.observed_counts[index]),
                "single_bank_count": int(
                    sum(likelihood.single_support_counts[index] for likelihood in self.likelihoods)
                ),
                "binary_bank_count": int(
                    sum(likelihood.binary_support_counts[index] for likelihood in self.likelihoods)
                ),
                "n_banks": int(self.n_banks),
            }
            for index, label in enumerate(self.condition_labels)
        ]

    def delta_time_summary(self):
        observed_counts = np.sum(
            self.observed_hist,
            axis=tuple(range(self.observed_hist.ndim - 1)),
        )
        single_hist = np.sum(
            [likelihood.single_hist for likelihood in self.likelihoods],
            axis=0,
        )
        single_counts = np.sum(single_hist, axis=tuple(range(single_hist.ndim - 1)))
        rows = []
        for index in range(self.dt_bins.size - 1):
            upper = self.dt_bins[index + 1]
            rows.append(
                {
                    "lower_days": float(self.dt_bins[index]),
                    "upper_days": None if np.isinf(upper) else float(upper),
                    "observed_count": int(observed_counts[index]),
                    "single_bank_count": int(single_counts[index]),
                    "binary_bank_size": int(
                        sum(likelihood.binary_bank.size for likelihood in self.likelihoods)
                    ),
                    "n_banks": int(self.n_banks),
                }
            )
        return rows


def run_averaged_joint_drvmax_dtmax_crn_mcmc(
    population,
    survey,
    dRV_real,
    dt_at_dRVmax_real,
    n_single_bank=100000,
    n_binary_bank=100000,
    bank_seed=None,
    bank_seeds=None,
    nwalkers=16,
    nsteps=2000,
    nthreads=4,
    parameter_names=None,
    parameter_bounds=None,
    fixed_parameters=None,
    initial_position=None,
    initial_scatter=None,
    pool_kind="process",
    start_method=None,
    moves=None,
    progress=None,
    drv_bins=None,
    dt_bins=None,
    condition_by=None,
    baseline_bins=None,
    observed_baseline_days=None,
    blending_kernel=None,
    blending_flux_fraction=None,
):
    """Run emcee with the count-pooled joint fixed-bank likelihood."""
    parameter_names = _normalise_parameter_names(parameter_names)
    parameter_bounds = _normalise_parameter_bounds(
        parameter_names,
        parameter_bounds=parameter_bounds,
    )
    if getattr(survey, "population", None) is None:
        survey.population = population
    elif survey.population is not population:
        raise ValueError("The supplied survey is not using the supplied population")
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "run_averaged_joint_drvmax_dtmax_crn_mcmc requires emcee"
        ) from exc
    if progress is None:
        progress = not _env_truthy("MINATO_QUIET")
    log_prob = AveragedJointDrvmaxDtmaxCRNLikelihood(
        survey,
        dRV_real,
        dt_at_dRVmax_real,
        n_single_bank=n_single_bank,
        n_binary_bank=n_binary_bank,
        bank_seed=bank_seed,
        bank_seeds=bank_seeds,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
        fixed_parameters=fixed_parameters,
        drv_bins=drv_bins,
        dt_bins=dt_bins,
        condition_by=condition_by,
        baseline_bins=baseline_bins,
        observed_baseline_days=observed_baseline_days,
        blending_kernel=blending_kernel,
        blending_flux_fraction=blending_flux_fraction,
    )
    p0 = _initial_walker_positions(
        population,
        parameter_names,
        parameter_bounds,
        nwalkers,
        initial_position=initial_position,
        initial_scatter=initial_scatter,
    )
    previous_roche_report = getattr(population, "roche_guard_report", None)
    if previous_roche_report is not None:
        population.roche_guard_report = False
    pool = _make_emcee_pool(pool_kind, nthreads, log_prob, start_method=start_method)
    try:
        sampler = emcee.EnsembleSampler(
            int(nwalkers),
            len(parameter_names),
            log_prob,
            pool=pool,
            moves=moves,
        )
        try:
            sampler.run_mcmc(p0, int(nsteps), progress=bool(progress))
        except TypeError:
            sampler.run_mcmc(p0, int(nsteps))
        sampler.mixture_crn_likelihood = log_prob
        sampler.joint_drvmax_dtmax_crn_likelihood = log_prob
        return sampler
    finally:
        if pool is not None and hasattr(pool, "close"):
            pool.close()
            pool.join()
        if previous_roche_report is not None:
            population.roche_guard_report = previous_roche_report


__all__ = [
    "AveragedJointDrvmaxDtmaxCRNLikelihood",
    "DEFAULT_DT_AT_DRVMAX_BINS",
    "JointDrvmaxDtmaxCRNLikelihood",
    "run_averaged_joint_drvmax_dtmax_crn_mcmc",
]
