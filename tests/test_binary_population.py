import unittest
import multiprocessing as mp

import numpy as np
import pandas as pd

from minato.binary_population import BinaryPopulation, BinarySurveySimulator, run_mcmc
from minato.binary_population import (
    AveragedMixtureCRNLikelihood,
    MixtureCRNLikelihood,
    run_averaged_mixture_crn_mcmc,
    run_mixture_crn_mcmc,
)


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

    def test_one_bank_averaged_mixture_matches_single_bank(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        single = MixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seed=321,
            parameter_names=("f_bin", "pi"),
        )
        averaged = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(321,),
            parameter_names=("f_bin", "pi"),
        )

        theta = np.array([0.55, 0.1])
        self.assertEqual(averaged(theta), single(theta))

    def test_baseline_conditioned_likelihood_is_deterministic(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        like = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=128,
            n_binary_bank=128,
            bank_seeds=(11, 12),
            parameter_names=("f_bin", "pi"),
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )

        theta = np.array([0.6, 0.1])
        self.assertEqual(like(theta), like(theta))

    def test_baseline_condition_counts_and_normalisation(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        like = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=256,
            n_binary_bank=256,
            bank_seeds=(21, 22),
            parameter_names=("f_bin", "pi"),
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )

        np.testing.assert_array_equal(like.observed_counts, np.array([2.0, 2.0, 0.0]))
        np.testing.assert_allclose(like.observed_hist.sum(axis=1), like.observed_counts)
        needed = like.observed_counts > 0
        np.testing.assert_allclose(like.single_probability[needed].sum(axis=1), 1.0)

        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        _, binary_probability = like.component_probabilities(params)
        np.testing.assert_allclose(binary_probability[needed].sum(axis=1), 1.0)

        summary = like.condition_summary()
        self.assertEqual(summary[0]["observed_count"], 2)
        self.assertEqual(summary[1]["observed_count"], 2)
        self.assertEqual(summary[2]["observed_count"], 0)

    def test_four_bank_baseline_conditioned_likelihood_is_finite(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        like = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=128,
            n_binary_bank=128,
            bank_seeds=(20260621, 20260622, 20260623, 20260624),
            parameter_names=("f_bin", "pi"),
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )

        grid_values = [
            like(np.array([f_bin, pi_value]))
            for f_bin in (0.5, 0.6)
            for pi_value in (-0.1, 0.1)
        ]
        self.assertTrue(np.all(np.isfinite(grid_values)))

    def test_run_averaged_mixture_crn_mcmc_smoke_shape(self):
        pop, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        np.random.seed(654)
        sampler = run_averaged_mixture_crn_mcmc(
            pop,
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(31,),
            nwalkers=8,
            nsteps=2,
            nthreads=1,
            pool_kind="none",
            progress=False,
            parameter_names=("f_bin", "pi"),
            initial_position={"f_bin": 0.6, "pi": 0.0},
            initial_scatter={"f_bin": 0.03, "pi": 0.05},
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )

        self.assertEqual(sampler.get_chain().shape, (2, 8, 2))
        self.assertTrue(np.all(np.isfinite(sampler.get_log_prob())))

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "static_process smoke uses fork")
    def test_run_averaged_mixture_crn_mcmc_static_process_smoke_shape(self):
        pop, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        np.random.seed(655)
        sampler = run_averaged_mixture_crn_mcmc(
            pop,
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(31, 32),
            nwalkers=8,
            nsteps=2,
            nthreads=2,
            pool_kind="static_process",
            start_method="fork",
            progress=False,
            parameter_names=("f_bin", "pi"),
            initial_position={"f_bin": 0.6, "pi": 0.0},
            initial_scatter={"f_bin": 0.03, "pi": 0.05},
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )

        self.assertEqual(sampler.get_chain().shape, (2, 8, 2))
        self.assertTrue(np.all(np.isfinite(sampler.get_log_prob())))


if __name__ == "__main__":
    unittest.main()
