import unittest

import numpy as np
import pandas as pd

from minato.binary_population import BinaryPopulation, BinarySurveySimulator, run_mcmc
from minato.binary_population import MixtureCRNLikelihood, run_mixture_crn_mcmc


def make_toy_survey():
    pop = BinaryPopulation()
    pop.roche_guard_report = False
    pop.logP_min = 0.15
    pop.logP_max = 3.5
    pop.logP_powerlaw_mode = "direct"
    pop.q_min = 0.1
    pop.q_max = 1.0
    survey = BinarySurveySimulator(pop)
    coverage = pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2, 3, 3, 3],
            "MJD": [60000, 60020, 60100, 60010, 60040, 60000, 60005, 60120],
            "mean_rv_er": [5, 5, 6, 4, 5, 7, 6, 8],
        }
    )
    survey.load_data(coverage)
    return pop, survey


class BinaryPopulationInferenceTests(unittest.TestCase):
    def test_fast_drv_summary_returns_finite_values(self):
        _, survey = make_toy_survey()
        np.random.seed(123)
        d_rv = survey.simulate_mock_drv_max(N=32, f_bin=0.5)

        self.assertEqual(d_rv.shape, (32,))
        self.assertTrue(np.all(np.isfinite(d_rv)))
        self.assertTrue(np.all(d_rv >= 0.0))

    def test_fast_and_legacy_summaries_have_consistent_scale(self):
        pop, survey = make_toy_survey()
        np.random.seed(123)
        intrinsic = pop.generate_intrinsic_sample_vectorized(N=256, f_bin=0.6)

        np.random.seed(456)
        fast = survey.simulate_mock_drv_max(intrinsic_sample=intrinsic)
        np.random.seed(456)
        legacy = survey.simulate_mock_observations(
            intrinsic_sample=intrinsic,
            summary_only=True,
        )["dRV_max"].to_numpy(float)

        self.assertLess(abs(np.nanmedian(fast) - np.nanmedian(legacy)), 5.0)
        self.assertLess(abs(np.nanpercentile(fast, 90) - np.nanpercentile(legacy, 90)), 25.0)

    def test_run_mcmc_smoke_shape(self):
        pop, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        np.random.seed(789)
        sampler = run_mcmc(
            pop,
            survey,
            observed,
            N_sim=20,
            batch_size=10,
            nwalkers=8,
            nsteps=2,
            nthreads=1,
            pool_kind="none",
            progress=False,
            sim_kwargs={"summary_only": True},
        )

        self.assertEqual(sampler.get_chain().shape, (2, 8, 1))

    def test_deterministic_unit_transforms_respect_domains(self):
        pop, _ = make_toy_survey()
        u = np.linspace(0.05, 0.95, 16)

        m1 = pop.draw_M1_from_unit(u)
        logp = pop.draw_logP_from_unit(u, pi=0.1)
        q = pop.draw_q_from_unit(u, kappa=0.5)
        period = 10.0 ** logp
        e = pop.draw_e_from_unit(u, period, eta=-0.4)

        self.assertTrue(np.all((m1 >= pop.M1_min) & (m1 <= pop.M1_max)))
        self.assertTrue(np.all((logp >= pop.logP_min) & (logp <= pop.logP_max)))
        self.assertTrue(np.all((q >= pop.q_min) & (q <= pop.q_max)))
        self.assertTrue(np.all(np.isfinite(e)))
        self.assertTrue(np.all(e[period < 2.0] == 0.0))
        self.assertTrue(np.all(e <= pop.e_max))

    def test_mixture_crn_likelihood_is_deterministic(self):
        pop, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        like = MixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seed=123,
            parameter_names=("f_bin", "pi"),
        )

        theta = np.array([0.6, 0.1])
        self.assertEqual(like(theta), like(theta))

    def test_mixture_crn_fbin_is_linear_mixture_weight(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        like = MixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seed=456,
            parameter_names=("f_bin", "pi"),
        )
        params = {"f_bin": 0.0, "pi": 0.0, "kappa": 0.0, "eta": -0.5}
        single_counts = like.expected_counts(params)
        params["f_bin"] = 1.0
        binary_counts = like.expected_counts(params)
        params["f_bin"] = 0.25
        mixed_counts = like.expected_counts(params)

        np.testing.assert_allclose(
            mixed_counts,
            0.75 * single_counts + 0.25 * binary_counts,
        )

    def test_run_mixture_crn_mcmc_smoke_shape(self):
        pop, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        np.random.seed(987)
        sampler = run_mixture_crn_mcmc(
            pop,
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seed=789,
            nwalkers=8,
            nsteps=2,
            nthreads=1,
            pool_kind="none",
            progress=False,
            parameter_names=("f_bin", "pi"),
            initial_position={"f_bin": 0.6, "pi": 0.0},
            initial_scatter={"f_bin": 0.03, "pi": 0.05},
        )

        self.assertEqual(sampler.get_chain().shape, (2, 8, 2))
        self.assertTrue(np.all(np.isfinite(sampler.get_log_prob())))


if __name__ == "__main__":
    unittest.main()
