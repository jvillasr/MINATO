import os
import sys
import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing as mp


def _env_truthy(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        return False
    return val.strip().lower() not in {"", "0", "false", "no", "off"}


class LogProb:
    """
    Pickle-friendly log-probability callable for emcee pools.
    """

    def __init__(self, survey, N_sim, dRV_real, batch_size, sim_kwargs):
        self.survey = survey
        self.N_sim = int(N_sim)
        self.dRV_real = np.asarray(dRV_real)
        self.batch_size = int(batch_size)
        self.sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)

    def __call__(self, theta):
        return log_posterior(
            theta,
            self.survey,
            self.N_sim,
            self.dRV_real,
            batch_size=self.batch_size,
            sim_kwargs=self.sim_kwargs,
        )


def log_prior(theta):
    """
    Simple prior on f_bin: uniform between 0 and 1.
    """
    f_bin = theta[0]
    if 0.0 < f_bin < 1.0:
        return 0.0
    return -np.inf


def compute_log_likelihood_batch(f_bin, survey, dRV_real, current_batch, sim_kwargs):
    """
    Generate a mock batch and compute Poisson log-likelihood contribution.
    """
    sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)
    # Internal bookkeeping for normalization across batches; not a survey simulator argument.
    N_sim_total = int(sim_kwargs.pop("N_sim_total", current_batch))
    # Likelihood bookkeeping (not survey simulator arguments).
    n_real = np.asarray(sim_kwargs.pop("n_real"), dtype=float)
    N_obs = int(sim_kwargs.pop("N_obs", len(dRV_real)))
    bins = np.asarray(sim_kwargs.pop("bins", np.logspace(0.4, 3, 30)), dtype=float)

    mock_res_df = survey.simulate_mock_observations(
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


def log_likelihood(theta, survey, N_sim, dRV_real, batch_size=1000, sim_kwargs=None):
    """
    Poisson log-likelihood comparing dRV_max distributions of real vs mock data.
    """
    if sim_kwargs is None:
        sim_kwargs = {}
    f_bin = theta[0]
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
        logL_total += compute_log_likelihood_batch(f_bin, survey, dRV_real, current_batch, sim_kwargs)

    return logL_total


def log_posterior(theta, survey, N_sim, dRV_real, batch_size=1000, sim_kwargs=None):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, survey, N_sim, dRV_real, batch_size=batch_size, sim_kwargs=sim_kwargs)


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
    pool_kind="process",  # "process" | "thread" | "none"
    start_method=None,    # e.g. "spawn" (macOS) or "fork" (Linux)
    progress=None,
):
    """
    Run an emcee EnsembleSampler over f_bin using the Poisson likelihood on dRV_real.
    """
    if sim_kwargs is None:
        sim_kwargs = {}

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

    log_prob = LogProb(survey, N_sim, dRV_real, batch_size, sim_kwargs)

    ndim = 1  # only fitting f_bin
    p0 = 0.80 + 0.05 * np.random.randn(nwalkers, ndim)

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
