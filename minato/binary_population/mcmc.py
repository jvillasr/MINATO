import copy
import os
import sys
from collections.abc import Mapping

import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing as mp


SUPPORTED_PARAMETER_NAMES = ("f_bin", "pi", "kappa", "eta")
DEFAULT_PARAMETER_BOUNDS = {
    "f_bin": (0.0, 1.0),
    # The default pi lower bound keeps the legacy shifted-logP sampler integrable.
    "pi": (-0.95, 4.0),
    "kappa": (-4.0, 4.0),
    "eta": (-0.95, 4.0),
}
DEFAULT_INITIAL_SCATTER = {
    "f_bin": 0.05,
    "pi": 0.05,
    "kappa": 0.05,
    "eta": 0.05,
}


def _env_truthy(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        return False
    return val.strip().lower() not in {"", "0", "false", "no", "off"}


def _normalise_parameter_names(parameter_names=None) -> tuple[str, ...]:
    if parameter_names is None:
        return ("f_bin",)
    if isinstance(parameter_names, str):
        if parameter_names == "all":
            return SUPPORTED_PARAMETER_NAMES
        parameter_names = (parameter_names,)
    names = tuple(parameter_names)
    if not names:
        raise ValueError("parameter_names must contain at least 'f_bin'.")
    unknown = sorted(set(names) - set(SUPPORTED_PARAMETER_NAMES))
    if unknown:
        raise ValueError(f"Unsupported MCMC parameter names: {unknown}")
    if "f_bin" not in names:
        raise ValueError("parameter_names must include 'f_bin'.")
    return names


def _normalise_parameter_bounds(parameter_names, parameter_bounds=None) -> dict[str, tuple[float, float]]:
    bounds = dict(DEFAULT_PARAMETER_BOUNDS)
    if parameter_bounds is not None:
        for name, value in dict(parameter_bounds).items():
            if name not in SUPPORTED_PARAMETER_NAMES:
                raise ValueError(f"Unsupported parameter bound for {name!r}.")
            if len(value) != 2:
                raise ValueError(f"Bounds for {name!r} must be a two-element sequence.")
            bounds[name] = (float(value[0]), float(value[1]))
    selected = {name: bounds[name] for name in parameter_names}
    for name, (lower, upper) in selected.items():
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid bounds for {name!r}: {(lower, upper)}")
    return selected


def _theta_to_parameter_dict(theta, parameter_names) -> dict[str, float]:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(parameter_names),):
        raise ValueError(
            f"Expected theta with {len(parameter_names)} values for {parameter_names}, "
            f"got shape {theta.shape}."
        )
    return {name: float(value) for name, value in zip(parameter_names, theta)}


def _apply_population_parameters(population, theta_params: Mapping[str, float]) -> None:
    for name in ("pi", "kappa", "eta"):
        if name in theta_params:
            setattr(population, name, float(theta_params[name]))


def _survey_for_population_parameters(survey, theta_params: Mapping[str, float]):
    population_params = {
        name: value for name, value in theta_params.items() if name != "f_bin"
    }
    if not population_params:
        return survey

    # Copy the lightweight survey/population shells so thread/process pools do
    # not race on pi/kappa/eta while sharing the read-only coverage dictionary.
    survey_local = copy.copy(survey)
    population_local = copy.copy(survey.population)
    _apply_population_parameters(population_local, population_params)
    survey_local.population = population_local
    return survey_local


def _initial_value_for_parameter(name: str, population) -> float:
    if name == "f_bin":
        return 0.80
    return float(getattr(population, name))


def _initial_walker_positions(
    population,
    parameter_names,
    parameter_bounds,
    nwalkers,
    initial_position=None,
    initial_scatter=None,
) -> np.ndarray:
    if initial_position is None:
        centre = np.array(
            [_initial_value_for_parameter(name, population) for name in parameter_names],
            dtype=float,
        )
    elif isinstance(initial_position, Mapping):
        centre = np.array(
            [float(initial_position[name]) for name in parameter_names],
            dtype=float,
        )
    else:
        centre = np.asarray(initial_position, dtype=float)
        if centre.shape != (len(parameter_names),):
            raise ValueError(
                "initial_position must match the number of fitted parameters."
            )

    scatter_map = dict(DEFAULT_INITIAL_SCATTER)
    if initial_scatter is not None:
        if isinstance(initial_scatter, Mapping):
            scatter_map.update(
                {name: float(value) for name, value in initial_scatter.items()}
            )
            scatter = np.array([scatter_map[name] for name in parameter_names], dtype=float)
        else:
            scatter = np.asarray(initial_scatter, dtype=float)
            if scatter.shape == ():
                scatter = np.full(len(parameter_names), float(scatter))
            if scatter.shape != (len(parameter_names),):
                raise ValueError(
                    "initial_scatter must be scalar or match the number of fitted parameters."
                )
    else:
        scatter = np.array([scatter_map[name] for name in parameter_names], dtype=float)

    p0 = centre + scatter * np.random.randn(int(nwalkers), len(parameter_names))
    for idx, name in enumerate(parameter_names):
        lower, upper = parameter_bounds[name]
        valid = (p0[:, idx] > lower) & (p0[:, idx] < upper)
        for _ in range(100):
            if np.all(valid):
                break
            p0[~valid, idx] = centre[idx] + scatter[idx] * np.random.randn(
                np.count_nonzero(~valid)
            )
            valid = (p0[:, idx] > lower) & (p0[:, idx] < upper)
        if not np.all(valid):
            eps = np.finfo(float).eps * max(1.0, abs(upper - lower))
            p0[~valid, idx] = np.random.uniform(
                lower + eps,
                upper - eps,
                np.count_nonzero(~valid),
            )
    return p0


class LogProb:
    """
    Pickle-friendly log-probability callable for emcee pools.
    """

    def __init__(
        self,
        survey,
        N_sim,
        dRV_real,
        batch_size,
        sim_kwargs,
        parameter_names=None,
        parameter_bounds=None,
    ):
        self.survey = survey
        self.N_sim = int(N_sim)
        self.dRV_real = np.asarray(dRV_real)
        self.batch_size = int(batch_size)
        self.sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)
        self.parameter_names = _normalise_parameter_names(parameter_names)
        self.parameter_bounds = _normalise_parameter_bounds(
            self.parameter_names,
            parameter_bounds=parameter_bounds,
        )

    def __call__(self, theta):
        return log_posterior(
            theta,
            self.survey,
            self.N_sim,
            self.dRV_real,
            batch_size=self.batch_size,
            sim_kwargs=self.sim_kwargs,
            parameter_names=self.parameter_names,
            parameter_bounds=self.parameter_bounds,
        )


def log_prior(theta, parameter_names=None, parameter_bounds=None):
    """
    Uniform box prior over the fitted binary-population parameters.

    By default this preserves the original one-parameter behaviour and fits
    only ``f_bin``. Pass ``parameter_names=("f_bin", "pi", "kappa", "eta")``
    to fit the binary fraction together with the period, mass-ratio, and
    eccentricity power-law indices.
    """
    names = _normalise_parameter_names(parameter_names)
    bounds = _normalise_parameter_bounds(names, parameter_bounds=parameter_bounds)
    try:
        theta_params = _theta_to_parameter_dict(theta, names)
    except ValueError:
        return -np.inf
    for name, value in theta_params.items():
        lower, upper = bounds[name]
        if not (lower < value < upper):
            return -np.inf
    return 0.0


def compute_log_likelihood_batch(theta_params, survey, dRV_real, current_batch, sim_kwargs):
    """
    Generate a mock batch and compute Poisson log-likelihood contribution.
    """
    if "f_bin" not in theta_params:
        raise ValueError("theta_params must include f_bin.")
    f_bin = float(theta_params["f_bin"])
    if not (0.0 <= f_bin <= 1.0):
        return -np.inf

    survey_for_batch = _survey_for_population_parameters(survey, theta_params)
    sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)
    # Internal bookkeeping for normalization across batches; not a survey simulator argument.
    N_sim_total = int(sim_kwargs.pop("N_sim_total", current_batch))
    # Likelihood bookkeeping (not survey simulator arguments).
    n_real = np.asarray(sim_kwargs.pop("n_real"), dtype=float)
    N_obs = int(sim_kwargs.pop("N_obs", len(dRV_real)))
    bins = np.asarray(sim_kwargs.pop("bins", np.logspace(0.4, 3, 30)), dtype=float)

    mock_res_df = survey_for_batch.simulate_mock_observations(
        N=current_batch,
        f_bin=f_bin,
        save_sample=False,
        intrinsic_sample=None,
        **sim_kwargs,
    )
    dRV_mock = mock_res_df["dRV_max"].values

    n_mock, _ = np.histogram(dRV_mock, bins=bins)
    n_mock = n_mock.astype(float)

    scale = N_obs / float(N_sim_total)
    n_mock *= scale

    epsilon = 1e-8
    n_mock += epsilon

    return float(np.sum(n_real * np.log(n_mock) - n_mock))


def log_likelihood(
    theta,
    survey,
    N_sim,
    dRV_real,
    batch_size=1000,
    sim_kwargs=None,
    parameter_names=None,
    parameter_bounds=None,
):
    """
    Poisson log-likelihood comparing dRV_max distributions of real vs mock data.
    """
    if sim_kwargs is None:
        sim_kwargs = {}
    names = _normalise_parameter_names(parameter_names)
    theta_params = _theta_to_parameter_dict(theta, names)
    n_batches = int(np.ceil(N_sim / batch_size))
    logL_total = 0.0

    bins = np.logspace(0.4, 3, 30)
    n_real, _ = np.histogram(dRV_real, bins=bins)

    # Default to summary-only simulation during inference (faster, lower memory).
    sim_kwargs = {**sim_kwargs}
    sim_kwargs.setdefault("summary_only", True)
    sim_kwargs = {**sim_kwargs, "N_sim_total": N_sim, "n_real": n_real, "N_obs": len(dRV_real), "bins": bins}
    for i in range(n_batches):
        current_batch = batch_size if i < n_batches - 1 else (N_sim - batch_size * (n_batches - 1))
        logL_total += compute_log_likelihood_batch(
            theta_params,
            survey,
            dRV_real,
            current_batch,
            sim_kwargs,
        )

    return logL_total


def log_posterior(
    theta,
    survey,
    N_sim,
    dRV_real,
    batch_size=1000,
    sim_kwargs=None,
    parameter_names=None,
    parameter_bounds=None,
):
    lp = log_prior(theta, parameter_names=parameter_names, parameter_bounds=parameter_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(
        theta,
        survey,
        N_sim,
        dRV_real,
        batch_size=batch_size,
        sim_kwargs=sim_kwargs,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
    )


def run_mcmc(
    population,
    survey,
    dRV_real,
    N_sim,
    batch_size=1000,
    nwalkers=16,
    nsteps=2000,
    nthreads=4,
    sim_kwargs=None,
    parameter_names=None,
    parameter_bounds=None,
    initial_position=None,
    initial_scatter=None,
    pool_kind="process",  # "process" | "thread" | "none"
    start_method=None,    # e.g. "spawn" (macOS) or "fork" (Linux)
    progress=None,
):
    """
    Run emcee over binary-population parameters using a Poisson likelihood on dRV_real.

    The default keeps the historical one-parameter interface and samples only
    ``f_bin``. To fit the first paper model, pass
    ``parameter_names=("f_bin", "pi", "kappa", "eta")`` and set the
    corresponding period, mass-ratio, and eccentricity domains on ``population``
    before calling this function.
    """
    if sim_kwargs is None:
        sim_kwargs = {}
    parameter_names = _normalise_parameter_names(parameter_names)
    parameter_bounds = _normalise_parameter_bounds(
        parameter_names,
        parameter_bounds=parameter_bounds,
    )

    if progress is None:
        progress = not _env_truthy("MINATO_QUIET")

    # Keep survey/population in sync if the caller passed a different instance.
    if getattr(survey, "population", None) is None:
        survey.population = population
    elif survey.population is not population:
        raise ValueError("The provided survey instance is not using the supplied population.")

    try:
        import emcee
    except ImportError as exc:
        raise ImportError("run_mcmc requires emcee to be installed") from exc

    log_prob = LogProb(
        survey,
        N_sim,
        dRV_real,
        batch_size,
        sim_kwargs,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
    )

    ndim = len(parameter_names)
    p0 = _initial_walker_positions(
        population,
        parameter_names,
        parameter_bounds,
        nwalkers,
        initial_position=initial_position,
        initial_scatter=initial_scatter,
    )

    # Silence Roche-guard clamp reports during MCMC (very noisy in batched simulations).
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
        if pool is None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
            try:
                sampler.run_mcmc(p0, nsteps, progress=bool(progress))
            except TypeError:
                sampler.run_mcmc(p0, nsteps)
            return sampler

        # ThreadPool supports context manager; multiprocessing Pool does not always implement __enter__/__exit__.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        try:
            sampler.run_mcmc(p0, nsteps, progress=bool(progress))
        except TypeError:
            sampler.run_mcmc(p0, nsteps)
        return sampler
    finally:
        if pool is not None and hasattr(pool, "close"):
            pool.close()
            pool.join()
        if prev_roche_report is not None:
            population.roche_guard_report = prev_roche_report
