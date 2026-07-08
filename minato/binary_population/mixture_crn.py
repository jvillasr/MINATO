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


__all__ = [
    "BinaryRandomBank",
    "CadenceRandomBank",
    "MixtureCRNLikelihood",
    "SingleRandomBank",
    "build_mixture_crn_banks",
    "run_mixture_crn_mcmc",
]
