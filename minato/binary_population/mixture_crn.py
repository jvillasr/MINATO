"""
Mixture-weight binary-population likelihood with common random numbers.

This module provides an experimental deterministic alternative to the standard
fresh-simulation likelihood in ``mcmc.py``. It represents the model histogram as

    (1 - f_bin) * p_single + f_bin * p_binary(pi, kappa, eta)

using fixed random banks for the single and binary components.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import numpy as np

from .mcmc import (
    DEFAULT_PARAMETER_BOUNDS,
    _env_truthy,
    _initial_walker_positions,
    _normalise_parameter_bounds,
    _normalise_parameter_names,
)


DEFAULT_BASELINE_BINS = np.array([0.0, 7.0, 30.0, 100.0, 365.0, np.inf])


@dataclass(frozen=True)
class CadenceRandomBank:
    """Fixed cadence-template choices and unit-normal RV-noise draws."""

    template_indices: np.ndarray
    noise_unit: tuple[np.ndarray, ...]

    @property
    def size(self) -> int:
        return int(self.template_indices.size)


@dataclass(frozen=True)
class SingleRandomBank:
    """Fixed random bank for the single-star component."""

    cadence: CadenceRandomBank

    @property
    def size(self) -> int:
        return self.cadence.size


@dataclass(frozen=True)
class BinaryRandomBank:
    """Fixed unit-rank bank for the binary-star component."""

    u_m1: np.ndarray
    u_logP: np.ndarray
    u_q: np.ndarray
    u_e: np.ndarray
    u_cos_i: np.ndarray
    u_omega: np.ndarray
    u_Tp: np.ndarray
    cadence: CadenceRandomBank

    @property
    def size(self) -> int:
        return int(self.u_m1.size)


@dataclass(frozen=True)
class ConditionedBankState:
    """Per-bank cached probabilities for a conditioned CRN likelihood."""

    likelihood: "MixtureCRNLikelihood"
    single_condition_indices: np.ndarray
    binary_condition_indices: np.ndarray
    single_counts: np.ndarray
    binary_counts: np.ndarray
    single_probability: np.ndarray


def _normalise_bank_seeds(bank_seeds=None, bank_seed=None) -> tuple[int | None, ...]:
    if bank_seeds is None:
        return (bank_seed,)
    if isinstance(bank_seeds, (str, bytes)):
        raise TypeError("bank_seeds must be an iterable of integers, not a string.")
    seeds = tuple(None if seed is None else int(seed) for seed in bank_seeds)
    if not seeds:
        raise ValueError("bank_seeds must contain at least one seed.")
    return seeds


def _normalise_condition_by(condition_by):
    if condition_by is None or condition_by == "none":
        return None
    if condition_by in {"baseline", "baseline_days"}:
        return "baseline_days"
    raise ValueError("condition_by must be None, 'none', 'baseline', or 'baseline_days'.")


def _normalise_condition_edges(edges) -> np.ndarray:
    edge_array = np.asarray(DEFAULT_BASELINE_BINS if edges is None else edges, dtype=float)
    if edge_array.ndim != 1 or edge_array.size < 2:
        raise ValueError("Condition-bin edges must be a one-dimensional array with at least two values.")
    if not np.all(np.diff(edge_array) > 0):
        raise ValueError("Condition-bin edges must be strictly increasing.")
    if not np.isfinite(edge_array[:-1]).all():
        raise ValueError("Only the final condition-bin edge may be infinite.")
    return edge_array


def _template_baselines(template_mjds) -> np.ndarray:
    baselines = np.empty(len(template_mjds), dtype=float)
    for idx, mjd in enumerate(template_mjds):
        mjd_array = np.asarray(mjd, dtype=float)
        baselines[idx] = float(np.nanmax(mjd_array) - np.nanmin(mjd_array))
    return baselines


def _assign_condition_bins(values, edges) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    indices = np.searchsorted(edges, values, side="right") - 1
    invalid = (indices < 0) | (indices >= len(edges) - 1) | ~np.isfinite(values)
    indices = indices.astype(np.int32, copy=False)
    indices[invalid] = -1
    return indices


def _edge_label(edges, index: int) -> str:
    lower = edges[index]
    upper = edges[index + 1]
    upper_label = "inf" if np.isinf(upper) else f"{upper:g}"
    return f"[{lower:g},{upper_label})"


def _histogram_by_condition(values, condition_indices, n_conditions, bins) -> np.ndarray:
    hist = np.zeros((int(n_conditions), len(bins) - 1), dtype=float)
    values = np.asarray(values, dtype=float)
    condition_indices = np.asarray(condition_indices, dtype=np.int32)
    for condition_index in range(int(n_conditions)):
        mask = condition_indices == condition_index
        if np.any(mask):
            hist[condition_index], _ = np.histogram(values[mask], bins=bins)
    return hist


def _normalise_condition_histograms(hist, counts) -> np.ndarray:
    probability = np.asarray(hist, dtype=float).copy()
    counts = np.asarray(counts, dtype=float)
    has_support = counts > 0
    probability[has_support] /= counts[has_support, None]
    return probability


def _build_cadence_bank(rng, n_bank, template_mjds) -> CadenceRandomBank:
    template_indices = rng.integers(0, len(template_mjds), size=int(n_bank))
    noise_unit = tuple(
        rng.standard_normal(np.asarray(template_mjds[int(template_idx)]).size)
        for template_idx in template_indices
    )
    return CadenceRandomBank(
        template_indices=np.asarray(template_indices, dtype=np.int32),
        noise_unit=noise_unit,
    )


def build_mixture_crn_banks(survey, n_single_bank, n_binary_bank, seed=None):
    """
    Build fixed single and binary random banks for a real-cadence survey.
    """
    _, template_mjds, _ = survey.cadence_templates()
    rng = np.random.default_rng(seed)
    n_binary_bank = int(n_binary_bank)
    return (
        SingleRandomBank(
            cadence=_build_cadence_bank(rng, int(n_single_bank), template_mjds),
        ),
        BinaryRandomBank(
            u_m1=rng.random(n_binary_bank),
            u_logP=rng.random(n_binary_bank),
            u_q=rng.random(n_binary_bank),
            u_e=rng.random(n_binary_bank),
            u_cos_i=rng.random(n_binary_bank),
            u_omega=rng.random(n_binary_bank),
            u_Tp=rng.random(n_binary_bank),
            cadence=_build_cadence_bank(rng, n_binary_bank, template_mjds),
        ),
    )


class MixtureCRNLikelihood:
    """
    Pickle-friendly deterministic mixture likelihood for emcee.
    """

    def __init__(
        self,
        survey,
        dRV_real,
        n_single_bank=100000,
        n_binary_bank=100000,
        bank_seed=None,
        single_bank=None,
        binary_bank=None,
        parameter_names=None,
        parameter_bounds=None,
        fixed_parameters=None,
        bins=None,
    ):
        self.survey = survey
        self.population = survey.population
        self.dRV_real = np.asarray(dRV_real, dtype=float)
        self.N_obs = int(self.dRV_real.size)
        self.parameter_names = _normalise_parameter_names(parameter_names)
        self.parameter_bounds = _normalise_parameter_bounds(
            self.parameter_names,
            parameter_bounds=parameter_bounds,
        )
        self.fixed_parameters = {} if fixed_parameters is None else dict(fixed_parameters)
        for name in ("pi", "kappa", "eta"):
            self.fixed_parameters.setdefault(name, float(getattr(self.population, name)))
        if "f_bin" not in self.parameter_names:
            raise ValueError("MixtureCRNLikelihood requires f_bin as a fitted parameter.")

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
        self.bins = np.asarray(np.logspace(0.4, 3, 30) if bins is None else bins, dtype=float)
        self.n_real, _ = np.histogram(self.dRV_real, bins=self.bins)

        self.single_drv = self._simulate_single_drv()
        self.single_hist, _ = np.histogram(self.single_drv, bins=self.bins)
        self.single_probability = self.single_hist.astype(float) / float(self.single_bank.size)

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

    def _simulate_single_drv(self):
        d_rv = np.empty(self.single_bank.size, dtype=float)
        for idx, template_idx in enumerate(self.single_bank.cadence.template_indices):
            rv_errors = self.template_rv_errors[int(template_idx)]
            rv_obs = self.single_bank.cadence.noise_unit[idx] * rv_errors
            d_rv[idx] = float(np.nanmax(rv_obs) - np.nanmin(rv_obs))
        return d_rv

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

    def simulate_binary_drv(self, params):
        arrays = self._binary_intrinsic_arrays(params)
        d_rv = np.empty(self.binary_bank.size, dtype=float)
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
            d_rv[idx] = float(np.nanmax(rv_obs) - np.nanmin(rv_obs))
        return d_rv

    def component_probabilities(self, params):
        binary_drv = self.simulate_binary_drv(params)
        binary_hist, _ = np.histogram(binary_drv, bins=self.bins)
        binary_probability = binary_hist.astype(float) / float(self.binary_bank.size)
        return self.single_probability, binary_probability

    def expected_counts(self, params):
        f_bin = float(params["f_bin"])
        single_probability, binary_probability = self.component_probabilities(params)
        probability = (1.0 - f_bin) * single_probability + f_bin * binary_probability
        return float(self.N_obs) * probability

    def log_likelihood(self, params):
        expected = self.expected_counts(params) + 1e-8
        return float(np.sum(self.n_real * np.log(expected) - expected))

    def __call__(self, theta):
        params = self._theta_to_params(theta)
        if params is None:
            return -np.inf
        try:
            return self.log_likelihood(params)
        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf


class AveragedMixtureCRNLikelihood:
    """
    Experimental multi-bank mixture-CRN likelihood.

    With ``condition_by=None`` this averages global component probabilities
    across one or more fixed random banks before applying one Poisson
    likelihood. With ``condition_by="baseline_days"`` it scores separate
    ``dRV_max`` histograms in survey-baseline bins and averages the per-bank
    conditioned probabilities before scoring.
    """

    def __init__(
        self,
        survey,
        dRV_real,
        n_single_bank=100000,
        n_binary_bank=100000,
        bank_seed=None,
        bank_seeds=None,
        parameter_names=None,
        parameter_bounds=None,
        fixed_parameters=None,
        bins=None,
        condition_by=None,
        baseline_bins=None,
        observed_baseline_days=None,
        likelihoods=None,
    ):
        self.survey = survey
        self.population = survey.population
        self.dRV_real = np.asarray(dRV_real, dtype=float)
        self.N_obs = int(self.dRV_real.size)
        self.parameter_names = _normalise_parameter_names(parameter_names)
        self.parameter_bounds = _normalise_parameter_bounds(
            self.parameter_names,
            parameter_bounds=parameter_bounds,
        )
        if "f_bin" not in self.parameter_names:
            raise ValueError("AveragedMixtureCRNLikelihood requires f_bin as a fitted parameter.")

        self.condition_by = _normalise_condition_by(condition_by)
        self.condition_edges = (
            None
            if self.condition_by is None
            else _normalise_condition_edges(baseline_bins)
        )
        self.condition_labels = (
            []
            if self.condition_edges is None
            else [
                _edge_label(self.condition_edges, idx)
                for idx in range(len(self.condition_edges) - 1)
            ]
        )
        self.n_conditions = len(self.condition_labels)
        self.bank_seeds = _normalise_bank_seeds(bank_seeds, bank_seed=bank_seed)

        if likelihoods is None:
            self.likelihoods = [
                MixtureCRNLikelihood(
                    survey,
                    self.dRV_real,
                    n_single_bank=n_single_bank,
                    n_binary_bank=n_binary_bank,
                    bank_seed=seed,
                    parameter_names=self.parameter_names,
                    parameter_bounds=self.parameter_bounds,
                    fixed_parameters=fixed_parameters,
                    bins=bins,
                )
                for seed in self.bank_seeds
            ]
        else:
            self.likelihoods = list(likelihoods)
            if not self.likelihoods:
                raise ValueError("likelihoods must contain at least one MixtureCRNLikelihood.")
            self.bank_seeds = tuple(
                getattr(likelihood, "bank_seed", None)
                for likelihood in self.likelihoods
            )

        self.reference = self.likelihoods[0]
        self.template_ids = self.reference.template_ids
        self.template_mjds = self.reference.template_mjds
        self.template_rv_errors = self.reference.template_rv_errors
        self.bins = self.reference.bins
        self.fixed_parameters = dict(self.reference.fixed_parameters)
        self._validate_likelihoods()

        if self.condition_by is None:
            self.n_real = np.asarray(self.reference.n_real, dtype=float)
            self.single_probability = np.mean(
                [likelihood.single_probability for likelihood in self.likelihoods],
                axis=0,
            )
            self.observed_counts = None
            self.observed_hist = None
            self.conditioned_bank_states = []
        else:
            self._init_baseline_conditioning(observed_baseline_days)

    @property
    def n_banks(self) -> int:
        return len(self.likelihoods)

    def _validate_likelihoods(self) -> None:
        for likelihood in self.likelihoods:
            if int(likelihood.N_obs) != self.N_obs:
                raise ValueError("All averaged likelihoods must use the same observed sample size.")
            if not np.array_equal(likelihood.bins, self.bins):
                raise ValueError("All averaged likelihoods must use the same dRV histogram bins.")
            if likelihood.parameter_names != self.parameter_names:
                raise ValueError("All averaged likelihoods must use the same fitted parameters.")
            if likelihood.parameter_bounds != self.parameter_bounds:
                raise ValueError("All averaged likelihoods must use the same parameter bounds.")
            if likelihood.fixed_parameters != self.fixed_parameters:
                raise ValueError("All averaged likelihoods must use the same fixed parameters.")
            if not np.array_equal(likelihood.n_real, self.reference.n_real):
                raise ValueError("All averaged likelihoods must use the same observed histogram.")

    def _observed_baseline_days(self, observed_baseline_days):
        if observed_baseline_days is not None:
            values = np.asarray(observed_baseline_days, dtype=float)
            if values.shape != self.dRV_real.shape:
                raise ValueError("observed_baseline_days must have the same shape as dRV_real.")
            return values

        template_baselines = _template_baselines(self.template_mjds)
        if template_baselines.shape == self.dRV_real.shape:
            return template_baselines
        raise ValueError(
            "observed_baseline_days is required unless dRV_real is one-to-one "
            "with the loaded survey cadence templates."
        )

    def _bank_condition_indices(self, likelihood, bank):
        template_baselines = _template_baselines(likelihood.template_mjds)
        template_indices = np.asarray(bank.cadence.template_indices, dtype=np.int64)
        values = template_baselines[template_indices]
        return _assign_condition_bins(values, self.condition_edges)

    def _init_baseline_conditioning(self, observed_baseline_days) -> None:
        observed_values = self._observed_baseline_days(observed_baseline_days)
        self.observed_condition_indices = _assign_condition_bins(
            observed_values,
            self.condition_edges,
        )
        if np.any(self.observed_condition_indices < 0):
            raise ValueError("Some observed baseline values fall outside the condition bins.")

        self.observed_counts = np.bincount(
            self.observed_condition_indices,
            minlength=self.n_conditions,
        ).astype(float)
        self.observed_hist = _histogram_by_condition(
            self.dRV_real,
            self.observed_condition_indices,
            self.n_conditions,
            self.bins,
        )

        needed = self.observed_counts > 0
        states = []
        for likelihood in self.likelihoods:
            single_condition_indices = self._bank_condition_indices(
                likelihood,
                likelihood.single_bank,
            )
            binary_condition_indices = self._bank_condition_indices(
                likelihood,
                likelihood.binary_bank,
            )
            single_counts = np.bincount(
                single_condition_indices[single_condition_indices >= 0],
                minlength=self.n_conditions,
            ).astype(float)
            binary_counts = np.bincount(
                binary_condition_indices[binary_condition_indices >= 0],
                minlength=self.n_conditions,
            ).astype(float)
            if np.any((single_counts <= 0) & needed):
                missing = [
                    self.condition_labels[idx]
                    for idx in np.flatnonzero((single_counts <= 0) & needed)
                ]
                raise ValueError(f"Single random bank has no systems in observed condition bin(s): {missing}")
            if np.any((binary_counts <= 0) & needed):
                missing = [
                    self.condition_labels[idx]
                    for idx in np.flatnonzero((binary_counts <= 0) & needed)
                ]
                raise ValueError(f"Binary random bank has no systems in observed condition bin(s): {missing}")

            single_hist = _histogram_by_condition(
                likelihood.single_drv,
                single_condition_indices,
                self.n_conditions,
                self.bins,
            )
            states.append(
                ConditionedBankState(
                    likelihood=likelihood,
                    single_condition_indices=single_condition_indices,
                    binary_condition_indices=binary_condition_indices,
                    single_counts=single_counts,
                    binary_counts=binary_counts,
                    single_probability=_normalise_condition_histograms(
                        single_hist,
                        single_counts,
                    ),
                )
            )

        self.conditioned_bank_states = states
        self.single_probability = np.mean(
            [state.single_probability for state in states],
            axis=0,
        )
        self.n_real = self.observed_hist

    def _theta_to_params(self, theta):
        return self.reference._theta_to_params(theta)

    def _binary_conditioned_probability(self, params):
        probabilities = []
        for state in self.conditioned_bank_states:
            binary_drv = state.likelihood.simulate_binary_drv(params)
            binary_hist = _histogram_by_condition(
                binary_drv,
                state.binary_condition_indices,
                self.n_conditions,
                self.bins,
            )
            probabilities.append(
                _normalise_condition_histograms(binary_hist, state.binary_counts)
            )
        return np.mean(probabilities, axis=0)

    def component_probabilities(self, params):
        if self.condition_by is None:
            binary_probability = np.mean(
                [
                    likelihood.component_probabilities(params)[1]
                    for likelihood in self.likelihoods
                ],
                axis=0,
            )
        else:
            binary_probability = self._binary_conditioned_probability(params)
        return self.single_probability, binary_probability

    def expected_counts(self, params):
        f_bin = float(params["f_bin"])
        single_probability, binary_probability = self.component_probabilities(params)
        probability = (1.0 - f_bin) * single_probability + f_bin * binary_probability
        if self.condition_by is None:
            return float(self.N_obs) * probability
        return self.observed_counts[:, None] * probability

    def log_likelihood(self, params):
        expected = self.expected_counts(params) + 1e-8
        if self.condition_by is None:
            observed = self.n_real
        else:
            observed = self.observed_hist
        return float(np.sum(observed * np.log(expected) - expected))

    def condition_summary(self) -> list[dict[str, int | str]]:
        if self.condition_by is None:
            return []
        rows = []
        for idx, label in enumerate(self.condition_labels):
            rows.append(
                {
                    "condition": label,
                    "observed_count": int(self.observed_counts[idx]),
                    "single_bank_count": int(
                        sum(state.single_counts[idx] for state in self.conditioned_bank_states)
                    ),
                    "binary_bank_count": int(
                        sum(state.binary_counts[idx] for state in self.conditioned_bank_states)
                    ),
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


def run_mixture_crn_mcmc(
    population,
    survey,
    dRV_real,
    n_single_bank=100000,
    n_binary_bank=100000,
    bank_seed=None,
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
    bins=None,
):
    """
    Run emcee with the deterministic mixture/common-random-number likelihood.
    """
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
        raise ImportError("run_mixture_crn_mcmc requires emcee to be installed") from exc

    if progress is None:
        progress = not _env_truthy("MINATO_QUIET")

    log_prob = MixtureCRNLikelihood(
        survey,
        dRV_real,
        n_single_bank=n_single_bank,
        n_binary_bank=n_binary_bank,
        bank_seed=bank_seed,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
        fixed_parameters=fixed_parameters,
        bins=bins,
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

    pool = None
    if nthreads and int(nthreads) > 1 and pool_kind != "none":
        if pool_kind == "thread":
            pool = ThreadPool(processes=int(nthreads))
        elif pool_kind == "process":
            if start_method is None:
                start_method = "spawn" if sys.platform in {"darwin", "win32"} else "fork"
            ctx = mp.get_context(str(start_method))
            pool = ctx.Pool(processes=int(nthreads))
        else:
            raise ValueError(f"Unsupported pool_kind={pool_kind!r}; use 'process', 'thread', or 'none'.")
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


def run_averaged_mixture_crn_mcmc(
    population,
    survey,
    dRV_real,
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
    bins=None,
    condition_by=None,
    baseline_bins=None,
    observed_baseline_days=None,
):
    """
    Run emcee with the experimental averaged mixture-CRN likelihood.

    ``bank_seeds`` may contain one or more fixed-bank seeds. Model component
    probabilities are averaged across banks before the Poisson likelihood is
    evaluated. Pass ``condition_by="baseline_days"`` to use baseline-binned
    ``dRV_max`` histograms.
    """
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
        raise ImportError("run_averaged_mixture_crn_mcmc requires emcee to be installed") from exc

    if progress is None:
        progress = not _env_truthy("MINATO_QUIET")

    log_prob = AveragedMixtureCRNLikelihood(
        survey,
        dRV_real,
        n_single_bank=n_single_bank,
        n_binary_bank=n_binary_bank,
        bank_seed=bank_seed,
        bank_seeds=bank_seeds,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
        fixed_parameters=fixed_parameters,
        bins=bins,
        condition_by=condition_by,
        baseline_bins=baseline_bins,
        observed_baseline_days=observed_baseline_days,
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

    pool = None
    if nthreads and int(nthreads) > 1 and pool_kind != "none":
        if pool_kind == "thread":
            pool = ThreadPool(processes=int(nthreads))
        elif pool_kind == "process":
            if start_method is None:
                start_method = "spawn" if sys.platform in {"darwin", "win32"} else "fork"
            ctx = mp.get_context(str(start_method))
            pool = ctx.Pool(processes=int(nthreads))
        else:
            raise ValueError(f"Unsupported pool_kind={pool_kind!r}; use 'process', 'thread', or 'none'.")
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
    "AveragedMixtureCRNLikelihood",
    "BinaryRandomBank",
    "CadenceRandomBank",
    "ConditionedBankState",
    "DEFAULT_BASELINE_BINS",
    "MixtureCRNLikelihood",
    "SingleRandomBank",
    "build_mixture_crn_banks",
    "run_averaged_mixture_crn_mcmc",
    "run_mixture_crn_mcmc",
]
