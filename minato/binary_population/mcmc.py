import numpy as np
from multiprocessing.pool import ThreadPool


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
    mock_res_df = survey.simulate_mock_observations(
        N=current_batch,
        f_bin=f_bin,
        save_sample=False,
        intrinsic_sample=None,
        **sim_kwargs,
    )
    dRV_mock = mock_res_df["dRV_max"].values

    nbins = 30
    bins = np.logspace(0.4, 3, nbins)

    n_real, _ = np.histogram(dRV_real, bins=bins)
    n_mock, _ = np.histogram(dRV_mock, bins=bins)
    n_mock = n_mock.astype(float)

    N_obs = len(dRV_real)
    scale = N_obs / float(sim_kwargs.get("N_sim_total", current_batch))
    n_mock *= scale

    epsilon = 1e-8
    n_mock += epsilon

    logL_batch = 0.0
    for j in range(len(n_real)):
        logL_batch += n_real[j] * np.log(n_mock[j]) - n_mock[j]
    return logL_batch


def log_likelihood(theta, survey, N_sim, dRV_real, batch_size=1000, sim_kwargs=None):
    """
    Poisson log-likelihood comparing dRV_max distributions of real vs mock data.
    """
    if sim_kwargs is None:
        sim_kwargs = {}
    f_bin = theta[0]
    n_batches = int(np.ceil(N_sim / batch_size))
    logL_total = 0.0

    sim_kwargs = {**sim_kwargs, "N_sim_total": N_sim}
    for i in range(n_batches):
        current_batch = batch_size if i < n_batches - 1 else (N_sim - batch_size * (n_batches - 1))
        logL_total += compute_log_likelihood_batch(f_bin, survey, dRV_real, current_batch, sim_kwargs)

    return logL_total


def log_posterior(theta, survey, N_sim, dRV_real, batch_size=1000, sim_kwargs=None):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, survey, N_sim, dRV_real, batch_size=batch_size, sim_kwargs=sim_kwargs)


def run_mcmc(population, survey, dRV_real, N_sim, batch_size=1000, nwalkers=16, nsteps=2000, nthreads=4, sim_kwargs=None):
    """
    Run an emcee EnsembleSampler over f_bin using the Poisson likelihood on dRV_real.
    """
    if sim_kwargs is None:
        sim_kwargs = {}

    # Keep survey/population in sync if the caller passed a different instance.
    if getattr(survey, "population", None) is None:
        survey.population = population
    elif survey.population is not population:
        raise ValueError("The provided survey instance is not using the supplied population.")

    try:
        import emcee
    except ImportError as exc:
        raise ImportError("run_mcmc requires emcee to be installed") from exc

    def log_prob(theta):
        return log_posterior(theta, survey, N_sim, dRV_real, batch_size=batch_size, sim_kwargs=sim_kwargs)

    ndim = 1  # only fitting f_bin
    p0 = 0.80 + 0.05 * np.random.randn(nwalkers, ndim)

    pool = ThreadPool(processes=nthreads) if nthreads and nthreads > 1 else None
    if pool is None:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(p0, nsteps, progress=True)
        return sampler

    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    return sampler
