"""
Pairwise RV-summary mixture-CRN likelihoods.

This module extends the averaged mixture-CRN machinery from ``mixture_crn`` to
star-balanced pairwise RV summaries. It is intentionally separate from the
existing ``dRV_max`` likelihood so the current public API remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .mcmc import (
    _env_truthy,
    _initial_walker_positions,
    _normalise_parameter_bounds,
    _normalise_parameter_names,
)
from .mixture_crn import (
    _make_emcee_pool,
    _normalise_bank_seeds,
    build_mixture_crn_banks,
)


DEFAULT_PAIRWISE_DELTA_TIME_BINS = (0.0, 1.0, 7.0, 30.0, 100.0, 365.0, 1000.0, 3000.0, np.inf)
DEFAULT_PAIRWISE_DELTA_TIME_LABELS = (
    "0-1",
    "1-7",
    "7-30",
    "30-100",
    "100-365",
    "365-1000",
    "1000-3000",
    ">=3000",
)
SUPPORTED_PAIRWISE_RESPONSES = ("max_pair_significance", "max_abs_delta_rv")


@dataclass(frozen=True)
class PairwiseSummaryConfig:
    """
    Configuration for per-star, time-binned pairwise RV summaries.

    ``response`` selects the value that is histogrammed by the likelihood. The
    summary always computes both ``max_pair_significance`` and
    ``max_abs_delta_rv`` so callers can use the unscored quantity for collapse
    tests or posterior-predictive checks.
    """

    delta_time_bins: tuple[float, ...] = DEFAULT_PAIRWISE_DELTA_TIME_BINS
    response: str = "max_pair_significance"
    response_bins: tuple[float, ...] | None = None


def _normalise_pairwise_config(config=None) -> PairwiseSummaryConfig:
    if config is None:
        config = PairwiseSummaryConfig()
    if not isinstance(config, PairwiseSummaryConfig):
        raise TypeError("config must be a PairwiseSummaryConfig.")

    edges = np.asarray(config.delta_time_bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("delta_time_bins must be a one-dimensional sequence with at least two edges.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("delta_time_bins must be strictly increasing.")
    if not np.isfinite(edges[:-1]).all():
        raise ValueError("Only the final delta_time_bins edge may be infinite.")
    if config.response not in SUPPORTED_PAIRWISE_RESPONSES:
        raise ValueError(
            f"Unsupported pairwise response {config.response!r}; use one of "
            f"{SUPPORTED_PAIRWISE_RESPONSES}."
        )

    response_bins = None
    if config.response_bins is not None:
        response_bins_array = np.asarray(config.response_bins, dtype=float)
        if response_bins_array.ndim != 1 or response_bins_array.size < 2:
            raise ValueError("response_bins must be a one-dimensional sequence with at least two edges.")
        if not np.all(np.diff(response_bins_array) > 0):
            raise ValueError("response_bins must be strictly increasing.")
        if not np.isfinite(response_bins_array[:-1]).all():
            raise ValueError("Only the final response_bins edge may be infinite.")
        response_bins = tuple(float(value) for value in response_bins_array)

    return PairwiseSummaryConfig(
        delta_time_bins=tuple(float(value) for value in edges),
        response=str(config.response),
        response_bins=response_bins,
    )


def _default_response_bins(response: str) -> np.ndarray:
    if response == "max_pair_significance":
        return np.concatenate([np.linspace(0.0, 20.0, 41), [np.inf]])
    if response == "max_abs_delta_rv":
        return np.asarray(np.logspace(0.4, 3, 30), dtype=float)
    raise ValueError(f"Unsupported pairwise response {response!r}.")


def _response_bins(config: PairwiseSummaryConfig) -> np.ndarray:
    if config.response_bins is None:
        return _default_response_bins(config.response)
    return np.asarray(config.response_bins, dtype=float)


def _delta_time_labels(edges: np.ndarray) -> tuple[str, ...]:
    labels = []
    for idx in range(len(edges) - 1):
        lower = edges[idx]
        upper = edges[idx + 1]
        upper_label = "inf" if np.isinf(upper) else f"{upper:g}"
        if lower == 3000.0 and np.isinf(upper):
            labels.append(">=3000")
        else:
            labels.append(f"{lower:g}-{upper_label}")
    return tuple(labels)


def _assign_delta_time_bin(delta_time: float, edges: np.ndarray) -> int:
    index = int(np.searchsorted(edges, float(delta_time), side="right") - 1)
    if index < 0 or index >= len(edges) - 1:
        return -1
    return index


def compute_pairwise_summary(mjd, rv, rv_error, config=None) -> dict[str, np.ndarray]:
    """
    Compute a star-balanced pairwise summary for one star.

    Missing time bins have ``n_pairs == 0`` and NaN response values. These bins
    are unobserved support, not zero-valued non-detections.
    """

    config = _normalise_pairwise_config(config)
    edges = np.asarray(config.delta_time_bins, dtype=float)
    n_bins = len(edges) - 1

    mjd = np.asarray(mjd, dtype=float)
    rv = np.asarray(rv, dtype=float)
    rv_error = np.asarray(rv_error, dtype=float)
    if mjd.shape != rv.shape or mjd.shape != rv_error.shape:
        raise ValueError("mjd, rv, and rv_error must have the same shape.")

    n_pairs = np.zeros(n_bins, dtype=np.int32)
    max_abs_delta_rv = np.full(n_bins, np.nan, dtype=float)
    max_pair_significance = np.full(n_bins, np.nan, dtype=float)

    valid = np.isfinite(mjd) & np.isfinite(rv) & np.isfinite(rv_error) & (rv_error > 0.0)
    if np.count_nonzero(valid) < 2:
        return {
            "n_pairs": n_pairs,
            "max_abs_delta_rv": max_abs_delta_rv,
            "max_pair_significance": max_pair_significance,
            "has_pair": n_pairs > 0,
        }

    mjd = mjd[valid]
    rv = rv[valid]
    rv_error = rv_error[valid]
    order = np.argsort(mjd)
    mjd = mjd[order]
    rv = rv[order]
    rv_error = rv_error[order]

    for i, j in combinations(range(mjd.size), 2):
        delta_time = abs(float(mjd[j] - mjd[i]))
        bin_index = _assign_delta_time_bin(delta_time, edges)
        if bin_index < 0:
            continue
        abs_delta_rv = abs(float(rv[j] - rv[i]))
        sigma_pair = float(np.hypot(rv_error[i], rv_error[j]))
        if sigma_pair <= 0.0 or not np.isfinite(sigma_pair):
            continue
        significance = abs_delta_rv / sigma_pair
        n_pairs[bin_index] += 1
        if not np.isfinite(max_abs_delta_rv[bin_index]) or abs_delta_rv > max_abs_delta_rv[bin_index]:
            max_abs_delta_rv[bin_index] = abs_delta_rv
        if (
            not np.isfinite(max_pair_significance[bin_index])
            or significance > max_pair_significance[bin_index]
        ):
            max_pair_significance[bin_index] = significance

    return {
        "n_pairs": n_pairs,
        "max_abs_delta_rv": max_abs_delta_rv,
        "max_pair_significance": max_pair_significance,
        "has_pair": n_pairs > 0,
    }


def compute_pairwise_response(mjd, rv, rv_error, config=None) -> np.ndarray:
    """Return the configured pairwise response vector for one star."""

    config = _normalise_pairwise_config(config)
    summary = compute_pairwise_summary(mjd, rv, rv_error, config=config)
    return np.asarray(summary[config.response], dtype=float)


def compute_pairwise_response_matrix(star_mjds, star_rvs, star_rv_errors, config=None) -> np.ndarray:
    """Return an ``n_stars x n_delta_time_bins`` pairwise response matrix."""

    config = _normalise_pairwise_config(config)
    n_bins = len(config.delta_time_bins) - 1
    responses = np.full((len(star_mjds), n_bins), np.nan, dtype=float)
    for idx, (mjd, rv, rv_error) in enumerate(zip(star_mjds, star_rvs, star_rv_errors)):
        responses[idx] = compute_pairwise_response(mjd, rv, rv_error, config=config)
    return responses


def _normalise_observed_summary(observed_pairwise_summary, n_delta_time_bins: int) -> np.ndarray:
    summary = np.asarray(observed_pairwise_summary, dtype=float)
    if summary.ndim == 1:
        if n_delta_time_bins != 1:
            raise ValueError("A one-dimensional observed summary is only valid for one Delta-t bin.")
        summary = summary[:, None]
    if summary.ndim != 2 or summary.shape[1] != int(n_delta_time_bins):
        raise ValueError(
            "observed_pairwise_summary must have shape "
            f"(n_stars, {int(n_delta_time_bins)})."
        )
    return summary


def _histogram_pairwise_summary(summary: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    summary = np.asarray(summary, dtype=float)
    hist = np.zeros((summary.shape[1], len(bins) - 1), dtype=float)
    counts = np.zeros(summary.shape[1], dtype=float)
    for bin_index in range(summary.shape[1]):
        values = summary[:, bin_index]
        values = values[np.isfinite(values)]
        counts[bin_index] = float(values.size)
        if values.size:
            hist[bin_index], _ = np.histogram(values, bins=bins)
    return hist, counts


def _normalise_pairwise_histograms(hist: np.ndarray, counts: np.ndarray) -> np.ndarray:
    probability = np.asarray(hist, dtype=float).copy()
    counts = np.asarray(counts, dtype=float)
    has_support = counts > 0
    probability[has_support] /= counts[has_support, None]
    return probability


class PairwiseMixtureCRNLikelihood:
    """Single-bank mixture-CRN likelihood for pairwise RV summary vectors."""

    def __init__(
        self,
        survey,
        observed_pairwise_summary,
        n_single_bank=100000,
        n_binary_bank=100000,
        bank_seed=None,
        single_bank=None,
        binary_bank=None,
        parameter_names=None,
        parameter_bounds=None,
        fixed_parameters=None,
        config=None,
    ):
        self.survey = survey
        self.population = survey.population
        self.config = _normalise_pairwise_config(config)
        self.delta_time_bins = np.asarray(self.config.delta_time_bins, dtype=float)
        self.delta_time_labels = _delta_time_labels(self.delta_time_bins)
        self.response_bins = _response_bins(self.config)
        self.parameter_names = _normalise_parameter_names(parameter_names)
        self.parameter_bounds = _normalise_parameter_bounds(
            self.parameter_names,
            parameter_bounds=parameter_bounds,
        )
        if "f_bin" not in self.parameter_names:
            raise ValueError("PairwiseMixtureCRNLikelihood requires f_bin as a fitted parameter.")

        self.fixed_parameters = {} if fixed_parameters is None else dict(fixed_parameters)
        for name in ("pi", "kappa", "eta"):
            self.fixed_parameters.setdefault(name, float(getattr(self.population, name)))

        self.template_ids, self.template_mjds, self.template_rv_errors = survey.cadence_templates()
        if single_bank is None or binary_bank is None:
            single_bank, binary_bank = build_mixture_crn_banks(
                survey,
                n_single_bank=n_single_bank,
                n_binary_bank=n_binary_bank,
                seed=bank_seed,
            )
        self.single_bank = single_bank
        self.binary_bank = binary_bank
        self.bank_seed = bank_seed

        self.observed_summary = _normalise_observed_summary(
            observed_pairwise_summary,
            len(self.delta_time_bins) - 1,
        )
        self.N_obs = int(self.observed_summary.shape[0])
        self.observed_hist, self.observed_counts = _histogram_pairwise_summary(
            self.observed_summary,
            self.response_bins,
        )

        self.single_summary = self._simulate_single_pairwise_summary()
        self.single_hist, self.single_support_counts = _histogram_pairwise_summary(
            self.single_summary,
            self.response_bins,
        )
        self._validate_component_support(self.single_support_counts, "Single")
        self.single_probability = _normalise_pairwise_histograms(
            self.single_hist,
            self.single_support_counts,
        )

    def _validate_component_support(self, support_counts: np.ndarray, label: str) -> None:
        missing = (self.observed_counts > 0) & (support_counts <= 0)
        if np.any(missing):
            missing_labels = [
                self.delta_time_labels[idx]
                for idx in np.flatnonzero(missing)
            ]
            raise ValueError(f"{label} random bank has no support in observed Delta-t bin(s): {missing_labels}")

    def _theta_to_params(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.parameter_names),):
            return None
        params = dict(self.fixed_parameters)
        for name, value in zip(self.parameter_names, theta):
            lower, upper = self.parameter_bounds[name]
            if not (lower < float(value) < upper):
                return None
            params[name] = float(value)
        return params

    def _binary_intrinsic_arrays(self, params):
        pop = self.population
        bank = self.binary_bank
        m1 = pop.draw_M1_from_unit(bank.u_m1)
        logP = pop.draw_logP_from_unit(bank.u_logP, pi=params["pi"])
        q = pop.draw_q_from_unit(bank.u_q, kappa=params["kappa"])
        period = np.power(10.0, logP)
        eccentricity = pop.draw_e_from_unit(bank.u_e, period, eta=params["eta"])

        if getattr(pop, "use_roche_guard", True):
            period, _ = pop._enforce_roche_guard_on_P(m1, q, eccentricity, period)
            logP = np.log10(period)

        m2 = m1 * q
        cos_i = -1.0 + 2.0 * bank.u_cos_i
        i_rad = np.arccos(np.clip(cos_i, -1.0, 1.0))
        sin_i = np.sin(i_rad)
        omega_deg = 360.0 * bank.u_omega
        omega_deg[eccentricity == 0.0] = 90.0
        tp = bank.u_Tp * period

        g_factor = 4.309e-3 * 3.0857e13
        period_sec = period * 86400.0
        denom = np.power(m1 + m2, 2.0 / 3.0)
        factor = np.power(2.0 * np.pi * g_factor, 1.0 / 3.0) * np.power(period_sec, -1.0 / 3.0)
        k1 = factor * (m2 * sin_i) / denom

        return {
            "P": period,
            "e": eccentricity,
            "Tp": tp,
            "omega_deg": omega_deg,
            "K1": k1,
        }

    def _simulate_single_pairwise_summary(self):
        response = np.empty((self.single_bank.size, len(self.delta_time_bins) - 1), dtype=float)
        for idx, template_idx in enumerate(self.single_bank.cadence.template_indices):
            template_idx = int(template_idx)
            rv_errors = self.template_rv_errors[template_idx]
            rv_obs = self.single_bank.cadence.noise_unit[idx] * rv_errors
            response[idx] = compute_pairwise_response(
                self.template_mjds[template_idx],
                rv_obs,
                rv_errors,
                config=self.config,
            )
        return response

    def simulate_binary_pairwise_summary(self, params):
        arrays = self._binary_intrinsic_arrays(params)
        response = np.empty((self.binary_bank.size, len(self.delta_time_bins) - 1), dtype=float)
        bank = self.binary_bank
        pop = self.population
        for idx, template_idx in enumerate(bank.cadence.template_indices):
            template_idx = int(template_idx)
            t_array = self.template_mjds[template_idx]
            rv_errors = self.template_rv_errors[template_idx]
            v1_true = pop.rvcurve(
                t_array,
                arrays["P"][idx],
                arrays["Tp"][idx],
                arrays["e"][idx],
                arrays["omega_deg"][idx],
                0.0,
                arrays["K1"][idx],
                0.0,
                SB2=False,
            )
            rv_obs = v1_true + bank.cadence.noise_unit[idx] * rv_errors
            response[idx] = compute_pairwise_response(
                t_array,
                rv_obs,
                rv_errors,
                config=self.config,
            )
        return response

    def component_probabilities(self, params):
        binary_summary = self.simulate_binary_pairwise_summary(params)
        binary_hist, binary_support_counts = _histogram_pairwise_summary(
            binary_summary,
            self.response_bins,
        )
        self._validate_component_support(binary_support_counts, "Binary")
        binary_probability = _normalise_pairwise_histograms(
            binary_hist,
            binary_support_counts,
        )
        return self.single_probability, binary_probability

    def expected_counts(self, params):
        f_bin = float(params["f_bin"])
        single_probability, binary_probability = self.component_probabilities(params)
        probability = (1.0 - f_bin) * single_probability + f_bin * binary_probability
        return self.observed_counts[:, None] * probability

    def log_likelihood(self, params):
        expected = self.expected_counts(params) + 1e-8
        return float(np.sum(self.observed_hist * np.log(expected) - expected))

    def condition_summary(self) -> list[dict[str, int | str]]:
        rows = []
        for idx, label in enumerate(self.delta_time_labels):
            rows.append(
                {
                    "delta_time_bin": label,
                    "observed_count": int(self.observed_counts[idx]),
                    "single_bank_count": int(self.single_support_counts[idx]),
                    "n_banks": 1,
                }
            )
        return rows

    def __call__(self, theta):
        params = self._theta_to_params(theta)
        if params is None:
            return -np.inf
        try:
            return self.log_likelihood(params)
        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf


class AveragedMixtureCRNPairwiseLikelihood:
    """
    Multi-bank mixture-CRN likelihood for pairwise RV summary vectors.

    Component probabilities are averaged across banks before the Poisson
    likelihood is evaluated.
    """

    def __init__(
        self,
        survey,
        observed_pairwise_summary,
        n_single_bank=100000,
        n_binary_bank=100000,
        bank_seed=None,
        bank_seeds=None,
        parameter_names=None,
        parameter_bounds=None,
        fixed_parameters=None,
        config=None,
        likelihoods=None,
    ):
        self.survey = survey
        self.population = survey.population
        self.config = _normalise_pairwise_config(config)
        self.parameter_names = _normalise_parameter_names(parameter_names)
        self.parameter_bounds = _normalise_parameter_bounds(
            self.parameter_names,
            parameter_bounds=parameter_bounds,
        )
        if "f_bin" not in self.parameter_names:
            raise ValueError("AveragedMixtureCRNPairwiseLikelihood requires f_bin as a fitted parameter.")

        self.bank_seeds = _normalise_bank_seeds(bank_seeds, bank_seed=bank_seed)
        if likelihoods is None:
            self.likelihoods = [
                PairwiseMixtureCRNLikelihood(
                    survey,
                    observed_pairwise_summary,
                    n_single_bank=n_single_bank,
                    n_binary_bank=n_binary_bank,
                    bank_seed=seed,
                    parameter_names=self.parameter_names,
                    parameter_bounds=self.parameter_bounds,
                    fixed_parameters=fixed_parameters,
                    config=self.config,
                )
                for seed in self.bank_seeds
            ]
        else:
            self.likelihoods = list(likelihoods)
            if not self.likelihoods:
                raise ValueError("likelihoods must contain at least one PairwiseMixtureCRNLikelihood.")
            self.bank_seeds = tuple(
                getattr(likelihood, "bank_seed", None)
                for likelihood in self.likelihoods
            )

        self.reference = self.likelihoods[0]
        self.delta_time_bins = self.reference.delta_time_bins
        self.delta_time_labels = self.reference.delta_time_labels
        self.response_bins = self.reference.response_bins
        self.observed_summary = self.reference.observed_summary
        self.observed_hist = self.reference.observed_hist
        self.observed_counts = self.reference.observed_counts
        self.N_obs = self.reference.N_obs
        self.fixed_parameters = dict(self.reference.fixed_parameters)
        self._validate_likelihoods()
        self.single_probability = np.mean(
            [likelihood.single_probability for likelihood in self.likelihoods],
            axis=0,
        )
        self.single_support_counts = np.sum(
            [likelihood.single_support_counts for likelihood in self.likelihoods],
            axis=0,
        )

    @property
    def n_banks(self) -> int:
        return len(self.likelihoods)

    def _validate_likelihoods(self) -> None:
        for likelihood in self.likelihoods:
            if likelihood.parameter_names != self.parameter_names:
                raise ValueError("All averaged pairwise likelihoods must use the same fitted parameters.")
            if likelihood.parameter_bounds != self.parameter_bounds:
                raise ValueError("All averaged pairwise likelihoods must use the same parameter bounds.")
            if likelihood.fixed_parameters != self.fixed_parameters:
                raise ValueError("All averaged pairwise likelihoods must use the same fixed parameters.")
            if likelihood.config != self.reference.config:
                raise ValueError("All averaged pairwise likelihoods must use the same pairwise config.")
            if not np.array_equal(likelihood.response_bins, self.response_bins):
                raise ValueError("All averaged pairwise likelihoods must use the same response bins.")
            if not np.array_equal(likelihood.observed_hist, self.observed_hist):
                raise ValueError("All averaged pairwise likelihoods must use the same observed histogram.")
            if not np.array_equal(likelihood.observed_counts, self.observed_counts):
                raise ValueError("All averaged pairwise likelihoods must use the same observed support counts.")

    def _theta_to_params(self, theta):
        return self.reference._theta_to_params(theta)

    def component_probabilities(self, params):
        binary_probability = np.mean(
            [
                likelihood.component_probabilities(params)[1]
                for likelihood in self.likelihoods
            ],
            axis=0,
        )
        return self.single_probability, binary_probability

    def expected_counts(self, params):
        f_bin = float(params["f_bin"])
        single_probability, binary_probability = self.component_probabilities(params)
        probability = (1.0 - f_bin) * single_probability + f_bin * binary_probability
        return self.observed_counts[:, None] * probability

    def log_likelihood(self, params):
        expected = self.expected_counts(params) + 1e-8
        return float(np.sum(self.observed_hist * np.log(expected) - expected))

    def condition_summary(self) -> list[dict[str, int | str]]:
        rows = []
        for idx, label in enumerate(self.delta_time_labels):
            rows.append(
                {
                    "delta_time_bin": label,
                    "observed_count": int(self.observed_counts[idx]),
                    "single_bank_count": int(self.single_support_counts[idx]),
                    "n_banks": int(self.n_banks),
                }
            )
        return rows

    def __call__(self, theta):
        params = self._theta_to_params(theta)
        if params is None:
            return -np.inf
        try:
            return self.log_likelihood(params)
        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf


def run_averaged_mixture_crn_pairwise_mcmc(
    population,
    survey,
    observed_pairwise_summary,
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
    config=None,
):
    """Run emcee with the averaged pairwise mixture-CRN likelihood."""

    parameter_names = _normalise_parameter_names(parameter_names)
    parameter_bounds = _normalise_parameter_bounds(
        parameter_names,
        parameter_bounds=parameter_bounds,
    )
    if getattr(survey, "population", None) is None:
        survey.population = population
    elif survey.population is not population:
        raise ValueError("The provided survey instance is not using the supplied population.")

    try:
        import emcee
    except ImportError as exc:
        raise ImportError("run_averaged_mixture_crn_pairwise_mcmc requires emcee to be installed") from exc

    if progress is None:
        progress = not _env_truthy("MINATO_QUIET")

    log_prob = AveragedMixtureCRNPairwiseLikelihood(
        survey,
        observed_pairwise_summary,
        n_single_bank=n_single_bank,
        n_binary_bank=n_binary_bank,
        bank_seed=bank_seed,
        bank_seeds=bank_seeds,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
        fixed_parameters=fixed_parameters,
        config=config,
    )
    p0 = _initial_walker_positions(
        population,
        parameter_names,
        parameter_bounds,
        nwalkers,
        initial_position=initial_position,
        initial_scatter=initial_scatter,
    )

    prev_roche_report = getattr(population, "roche_guard_report", None)
    if prev_roche_report is not None:
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
        return sampler
    finally:
        if pool is not None and hasattr(pool, "close"):
            pool.close()
            pool.join()
        if prev_roche_report is not None:
            population.roche_guard_report = prev_roche_report


__all__ = [
    "AveragedMixtureCRNPairwiseLikelihood",
    "DEFAULT_PAIRWISE_DELTA_TIME_BINS",
    "DEFAULT_PAIRWISE_DELTA_TIME_LABELS",
    "PairwiseMixtureCRNLikelihood",
    "PairwiseSummaryConfig",
    "compute_pairwise_response",
    "compute_pairwise_response_matrix",
    "compute_pairwise_summary",
    "run_averaged_mixture_crn_pairwise_mcmc",
]
