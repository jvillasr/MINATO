from .population import BinaryPopulation
from .survey import BinarySurveySimulator
from .mcmc import run_mcmc, log_prior, log_likelihood, log_posterior
from .mixture_crn import MixtureCRNLikelihood, run_mixture_crn_mcmc

# Backward-compatibility alias (previous name)
BinarySimulations = BinaryPopulation

__all__ = [
    "BinaryPopulation",
    "BinarySimulations",
    "BinarySurveySimulator",
    "run_mcmc",
    "run_mixture_crn_mcmc",
    "MixtureCRNLikelihood",
    "log_prior",
    "log_likelihood",
    "log_posterior",
]
