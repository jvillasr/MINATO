import unittest
import multiprocessing as mp

import numpy as np
import pandas as pd

from minato.binary_population import BinaryPopulation, BinarySurveySimulator, run_mcmc
from minato.binary_population import (
    AveragedMixtureCRNLikelihood,
    AveragedMixtureCRNPairwiseLikelihood,
    MixtureCRNLikelihood,
    PairwiseMixtureCRNLikelihood,
    PairwiseSummaryConfig,
    compute_pairwise_summary,
    run_averaged_mixture_crn_mcmc,
    run_averaged_mixture_crn_pairwise_mcmc,
    run_mixture_crn_mcmc,
)
from minato.binary_population.mixture_crn import (
    BankParallelAveragedLogProbPool,
    BinaryRandomBank,
    CadenceRandomBank,
    SingleRandomBank,
    build_mixture_crn_banks,
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


def make_single_template_survey():
    pop = BinaryPopulation()
    pop.roche_guard_report = False
    pop.logP_min = 0.15
    pop.logP_max = 3.5
    pop.logP_powerlaw_mode = "direct"
    pop.q_min = 0.2
    pop.q_max = 1.0
    survey = BinarySurveySimulator(pop)
    coverage = pd.DataFrame(
        {
            "ID": [1, 1, 1],
            "MJD": [60000.0, 60002.0, 60008.0],
            "mean_rv_er": [1.0, 1.0, 1.0],
        }
    )
    survey.load_data(coverage)
    return pop, survey


def make_one_system_banks(blend_unit=None):
    cadence = CadenceRandomBank(
        template_indices=np.array([0], dtype=np.int32),
        noise_unit=(np.zeros(3),),
    )
    single_bank = SingleRandomBank(cadence=cadence)
    binary_bank = BinaryRandomBank(
        u_m1=np.array([0.5]),
        u_logP=np.array([0.45]),
        u_q=np.array([0.6]),
        u_e=np.array([0.0]),
        u_cos_i=np.array([0.5]),
        u_omega=np.array([0.25]),
        u_Tp=np.array([0.0]),
        cadence=cadence,
        blend_unit=blend_unit,
    )
    return single_bank, binary_bank


class FluxAwareTestBlendKernel:
    metadata = {"name": "flux-aware-test-kernel", "n_rows": 3}

    def sample_bias(self, abs_delta_v, f_secondary, u):
        if f_secondary is None:
            raise ValueError("test kernel requires f_secondary")
        return 100.0 * np.asarray(u) + 0.01 * float(f_secondary) * np.asarray(abs_delta_v)


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

    def test_averaged_binary_probability_matches_bank_mean_for_four_parameters(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.2, "eta": -0.4}
        for condition_by, observed_baselines in (
            (None, None),
            ("baseline_days", np.array([30.0, 100.0, 120.0, 30.0])),
        ):
            with self.subTest(condition_by=condition_by):
                like = AveragedMixtureCRNLikelihood(
                    survey,
                    observed,
                    n_single_bank=128,
                    n_binary_bank=128,
                    bank_seeds=(41, 42, 43),
                    parameter_names=("f_bin", "pi", "kappa", "eta"),
                    bins=np.array([0.0, 1.0e6]),
                    condition_by=condition_by,
                    baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
                    observed_baseline_days=observed_baselines,
                )

                _, binary_probability = like.component_probabilities(params)
                bank_mean = np.mean(
                    [
                        like.bank_binary_probability(bank_index, params)
                        for bank_index in range(like.n_banks)
                    ],
                    axis=0,
                )
                np.testing.assert_allclose(binary_probability, bank_mean)

                expected = like.expected_counts(params)
                from_binary = like.expected_counts_from_binary_probability(
                    params,
                    binary_probability,
                )
                np.testing.assert_allclose(expected, from_binary)

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "bank_static_process smoke uses fork")
    def test_bank_static_process_pool_matches_serial_for_four_parameter_baseline(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        like = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=128,
            n_binary_bank=128,
            bank_seeds=(51, 52, 53),
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )
        theta_values = [
            np.array([0.55, 0.1, 0.2, -0.4]),
            np.array([0.65, -0.2, -0.3, 0.1]),
            np.array([-0.1, 0.1, 0.0, -0.4]),
        ]

        pool = BankParallelAveragedLogProbPool(like, processes=2, start_method="fork")
        try:
            parallel = pool.map(like, theta_values)
        finally:
            pool.close()
            pool.join()

        serial = [like(theta) for theta in theta_values]
        np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=1.0e-12)

    def test_pairwise_summary_invariants(self):
        config = PairwiseSummaryConfig()
        summary = compute_pairwise_summary(
            mjd=np.array([0.0, 10.0]),
            rv=np.array([10.0, 40.0]),
            rv_error=np.array([3.0, 4.0]),
            config=config,
        )
        labels = ["0-1", "1-7", "7-30", "30-100", "100-365", "365-1000", "1000-3000", ">=3000"]
        idx = labels.index("7-30")

        self.assertEqual(int(summary["n_pairs"][idx]), 1)
        self.assertEqual(float(summary["max_abs_delta_rv"][idx]), 30.0)
        self.assertEqual(float(summary["max_pair_significance"][idx]), 6.0)
        self.assertFalse(bool(summary["has_pair"][labels.index("1-7")]))
        self.assertTrue(np.isnan(summary["max_abs_delta_rv"][labels.index("1-7")]))

    def test_pairwise_all_time_absdrv_matches_drvmax_likelihood(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        bins = np.array([0.0, 10.0, 20.0, 50.0, 1.0e6])
        config = PairwiseSummaryConfig(
            delta_time_bins=(0.0, np.inf),
            response="max_abs_delta_rv",
            response_bins=tuple(bins),
        )
        drv_like = MixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=96,
            n_binary_bank=96,
            bank_seed=20260709,
            parameter_names=("f_bin", "pi"),
            bins=bins,
        )
        pairwise_like = PairwiseMixtureCRNLikelihood(
            survey,
            observed[:, None],
            n_single_bank=96,
            n_binary_bank=96,
            bank_seed=20260709,
            parameter_names=("f_bin", "pi"),
            config=config,
        )

        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.4}
        np.testing.assert_allclose(
            pairwise_like.expected_counts(params)[0],
            drv_like.expected_counts(params),
        )
        self.assertAlmostEqual(
            pairwise_like(np.array([0.6, 0.1])),
            drv_like(np.array([0.6, 0.1])),
            places=10,
        )

    def test_blending_kernel_shifts_epoch_rvs_before_drvmax(self):
        pop, survey = make_single_template_survey()
        blend_unit = (np.array([0.0, 0.25, 0.75]),)
        single_bank, binary_bank = make_one_system_banks(blend_unit=blend_unit)
        kernel = FluxAwareTestBlendKernel()
        like = MixtureCRNLikelihood(
            survey,
            np.array([1.0]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            bins=np.array([0.0, 1.0e6]),
            blending_kernel=kernel,
            blending_flux_fraction=lambda **kwargs: 0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        arrays = like._binary_intrinsic_arrays(params)
        t_array = like.template_mjds[0]
        v1_true, v2_true = pop.rvcurve(
            t_array,
            arrays["P"][0],
            arrays["Tp"][0],
            arrays["e"][0],
            arrays["omega_deg"][0],
            0.0,
            arrays["K1"][0],
            arrays["K2"][0],
            SB2=True,
        )
        expected_bias = kernel.sample_bias(np.abs(v2_true - v1_true), 0.25, blend_unit[0])
        expected_drv = np.ptp(v1_true + expected_bias)

        np.testing.assert_allclose(like.simulate_binary_drv(params), [expected_drv])
        self.assertTrue(like.blending_metadata["enabled"])
        self.assertEqual(like.blending_metadata["name"], "flux-aware-test-kernel")

    def test_blending_kernel_shifts_epoch_rvs_before_pairwise_summary(self):
        pop, survey = make_single_template_survey()
        blend_unit = (np.array([0.0, 0.25, 0.75]),)
        single_bank, binary_bank = make_one_system_banks(blend_unit=blend_unit)
        kernel = FluxAwareTestBlendKernel()
        config = PairwiseSummaryConfig(
            delta_time_bins=(0.0, np.inf),
            response="max_abs_delta_rv",
            response_bins=(0.0, 10.0, 100.0, 1000.0),
        )
        like = PairwiseMixtureCRNLikelihood(
            survey,
            np.array([[1.0]]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            config=config,
            blending_kernel=kernel,
            blending_flux_fraction=lambda **kwargs: 0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        arrays = like._binary_intrinsic_arrays(params)
        t_array = like.template_mjds[0]
        v1_true, v2_true = pop.rvcurve(
            t_array,
            arrays["P"][0],
            arrays["Tp"][0],
            arrays["e"][0],
            arrays["omega_deg"][0],
            0.0,
            arrays["K1"][0],
            arrays["K2"][0],
            SB2=True,
        )
        expected_bias = kernel.sample_bias(np.abs(v2_true - v1_true), 0.25, blend_unit[0])
        expected_response = np.ptp(v1_true + expected_bias)

        np.testing.assert_allclose(
            like.simulate_binary_pairwise_summary(params),
            [[expected_response]],
        )

    def test_blending_kernel_requires_blend_unit_bank_draws(self):
        _, survey = make_single_template_survey()
        single_bank, binary_bank = make_one_system_banks(blend_unit=None)
        like = MixtureCRNLikelihood(
            survey,
            np.array([1.0]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            bins=np.array([0.0, 1.0e6]),
            blending_kernel=FluxAwareTestBlendKernel(),
            blending_flux_fraction=lambda **kwargs: 0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        with self.assertRaisesRegex(ValueError, "blend_unit"):
            like.simulate_binary_drv(params)

    def test_blend_unit_bank_draws_are_opt_in_and_preserve_existing_crn_draws(self):
        _, survey = make_single_template_survey()
        plain_single, plain_binary = build_mixture_crn_banks(
            survey,
            n_single_bank=4,
            n_binary_bank=5,
            seed=123,
        )
        blend_single, blend_binary = build_mixture_crn_banks(
            survey,
            n_single_bank=4,
            n_binary_bank=5,
            seed=123,
            include_blend_unit=True,
        )

        self.assertIsNone(plain_binary.blend_unit)
        self.assertIsNotNone(blend_binary.blend_unit)
        np.testing.assert_array_equal(
            plain_single.cadence.template_indices,
            blend_single.cadence.template_indices,
        )
        np.testing.assert_allclose(plain_binary.u_m1, blend_binary.u_m1)
        np.testing.assert_allclose(plain_binary.u_logP, blend_binary.u_logP)
        np.testing.assert_allclose(plain_binary.u_q, blend_binary.u_q)
        np.testing.assert_allclose(plain_binary.u_e, blend_binary.u_e)
        np.testing.assert_allclose(plain_binary.u_cos_i, blend_binary.u_cos_i)
        np.testing.assert_allclose(plain_binary.u_omega, blend_binary.u_omega)
        np.testing.assert_allclose(plain_binary.u_Tp, blend_binary.u_Tp)
        np.testing.assert_array_equal(
            plain_binary.cadence.template_indices,
            blend_binary.cadence.template_indices,
        )

    def test_one_bank_averaged_pairwise_matches_single_bank(self):
        _, survey = make_toy_survey()
        observed = np.array([[5.0], [12.0], [25.0], [40.0]])
        config = PairwiseSummaryConfig(
            delta_time_bins=(0.0, np.inf),
            response="max_abs_delta_rv",
            response_bins=(0.0, 10.0, 20.0, 50.0, 1.0e6),
        )
        single = PairwiseMixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seed=321,
            parameter_names=("f_bin", "pi"),
            config=config,
        )
        averaged = AveragedMixtureCRNPairwiseLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(321,),
            parameter_names=("f_bin", "pi"),
            config=config,
        )

        theta = np.array([0.55, 0.1])
        self.assertEqual(averaged(theta), single(theta))

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

    def test_run_averaged_mixture_crn_pairwise_mcmc_smoke_shape(self):
        pop, survey = make_toy_survey()
        observed = np.array([[5.0], [12.0], [25.0], [40.0]])
        config = PairwiseSummaryConfig(
            delta_time_bins=(0.0, np.inf),
            response="max_abs_delta_rv",
            response_bins=(0.0, 10.0, 20.0, 50.0, 1.0e6),
        )
        np.random.seed(656)
        sampler = run_averaged_mixture_crn_pairwise_mcmc(
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
            config=config,
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

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "bank_static_process smoke uses fork")
    def test_run_averaged_mixture_crn_mcmc_bank_static_process_smoke_shape(self):
        pop, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        np.random.seed(657)
        sampler = run_averaged_mixture_crn_mcmc(
            pop,
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(41, 42),
            nwalkers=10,
            nsteps=2,
            nthreads=2,
            pool_kind="bank_static_process",
            start_method="fork",
            progress=False,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            initial_position={"f_bin": 0.6, "pi": 0.0, "kappa": 0.0, "eta": -0.4},
            initial_scatter={"f_bin": 0.03, "pi": 0.05, "kappa": 0.05, "eta": 0.05},
            bins=np.array([0.0, 1.0e6]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )

        self.assertEqual(sampler.get_chain().shape, (2, 10, 4))
        self.assertTrue(np.all(np.isfinite(sampler.get_log_prob())))


if __name__ == "__main__":
    unittest.main()
