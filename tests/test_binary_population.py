import unittest
import multiprocessing as mp

import numpy as np
import pandas as pd

from minato.binary_population import BinaryPopulation, BinarySurveySimulator, run_mcmc
from minato.binary_population import (
    AveragedJointDrvmaxDtmaxCRNLikelihood,
    AveragedMixtureCRNLikelihood,
    AveragedMixtureCRNPairwiseLikelihood,
    JointDrvmaxDtmaxCRNLikelihood,
    MixtureCRNLikelihood,
    PairwiseMixtureCRNLikelihood,
    PairwiseSummaryConfig,
    compute_pairwise_summary,
    run_averaged_joint_drvmax_dtmax_crn_mcmc,
    run_averaged_mixture_crn_mcmc,
    run_averaged_mixture_crn_pairwise_mcmc,
    run_mixture_crn_mcmc,
)
from minato.binary_population.blending import sample_blending_bias
from minato.binary_population.joint_crn import _pool_joint_probability_tables
from minato.binary_population.mixture_crn import (
    BankParallelAveragedLogProbPool,
    BinaryRandomBank,
    CadenceRandomBank,
    SingleRandomBank,
    _pool_probability_tables,
    build_mixture_crn_banks,
)
from minato.binary_population.orbits import orbital_semi_amplitudes_kms


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


def slice_random_banks(single_bank, binary_bank, indices):
    indices = np.asarray(indices, dtype=np.int64)

    def sliced_cadence(cadence):
        return CadenceRandomBank(
            template_indices=np.asarray(cadence.template_indices)[indices],
            noise_unit=tuple(cadence.noise_unit[int(index)] for index in indices),
        )

    return (
        SingleRandomBank(cadence=sliced_cadence(single_bank.cadence)),
        BinaryRandomBank(
            u_m1=np.asarray(binary_bank.u_m1)[indices],
            u_logP=np.asarray(binary_bank.u_logP)[indices],
            u_q=np.asarray(binary_bank.u_q)[indices],
            u_e=np.asarray(binary_bank.u_e)[indices],
            u_cos_i=np.asarray(binary_bank.u_cos_i)[indices],
            u_omega=np.asarray(binary_bank.u_omega)[indices],
            u_Tp=np.asarray(binary_bank.u_Tp)[indices],
            cadence=sliced_cadence(binary_bank.cadence),
            blend_unit=(
                None
                if binary_bank.blend_unit is None
                else tuple(binary_bank.blend_unit[int(index)] for index in indices)
            ),
        ),
    )


def make_tiny_observed_summaries(survey, *, seed=20260802, n_systems=8):
    """Generate aligned dRV, defining-epoch separation and baseline arrays."""
    sample = survey.simulate_mock_observations(
        N=n_systems,
        f_bin=0.5,
        summary_only=False,
        seed=seed,
    )
    drv = sample["dRV_max"].to_numpy(dtype=float)
    dt = np.asarray(
        [
            abs(
                np.asarray(row.mjd_array, dtype=float)[np.argmax(row.rv_array)]
                - np.asarray(row.mjd_array, dtype=float)[np.argmin(row.rv_array)]
            )
            for row in sample.itertuples()
        ],
        dtype=float,
    )
    baselines = np.asarray(
        [np.ptp(np.asarray(values, dtype=float)) for values in sample["mjd_array"]],
        dtype=float,
    )
    return drv, dt, baselines


class FluxAwareTestBlendKernel:
    metadata = {"name": "flux-aware-test-kernel", "n_rows": 3}

    def sample_bias(self, abs_delta_v, f_secondary, u):
        if f_secondary is None:
            raise ValueError("test kernel requires f_secondary")
        return 100.0 * np.asarray(u) + 0.01 * float(f_secondary) * np.asarray(abs_delta_v)


class SignedFractionalTestBlendKernel:
    metadata = {"name": "signed-fractional-test-kernel", "beta": 0.05}

    def sample_bias_signed(self, delta_v, f_secondary, u):
        return self.metadata["beta"] * np.asarray(delta_v)

    def sample_bias(self, abs_delta_v, f_secondary, u):
        raise AssertionError("signed protocol must take precedence")


class BinaryPopulationInferenceTests(unittest.TestCase):
    def test_joint_path_uses_corrected_orbital_amplitudes(self):
        pop, survey = make_single_template_survey()
        pop.use_roche_guard = False
        single_bank, binary_bank = make_one_system_banks()
        binary_bank = BinaryRandomBank(
            **{
                **binary_bank.__dict__,
                "u_e": np.array([0.8]),
            }
        )
        like = JointDrvmaxDtmaxCRNLikelihood(
            survey,
            np.array([1.0]),
            np.array([2.0]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            drv_bins=np.array([0.0, np.inf]),
            dt_bins=np.array([0.0, np.inf]),
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.2, "eta": -0.4}
        arrays = like._binary_intrinsic_arrays(params)
        inclination = np.arccos(-1.0 + 2.0 * binary_bank.u_cos_i)
        expected_k1, expected_k2 = orbital_semi_amplitudes_kms(
            arrays["M1"], arrays["M2"], arrays["P"], inclination, arrays["e"]
        )

        self.assertGreater(float(arrays["e"][0]), 0.0)
        np.testing.assert_allclose(arrays["K1"], expected_k1)
        np.testing.assert_allclose(arrays["K2"], expected_k2)

    def test_joint_one_time_bin_matches_drv_likelihood(self):
        _, survey = make_toy_survey()
        observed_drv = np.array([0.0, 5.0, 25.0, 1.0e6])
        observed_dt = np.array([0.0, 20.0, 100.0, 1.0e6])
        single_bank, binary_bank = build_mixture_crn_banks(survey, 128, 128, seed=610)
        common = {
            "survey": survey,
            "dRV_real": observed_drv,
            "single_bank": single_bank,
            "binary_bank": binary_bank,
            "parameter_names": ("f_bin", "pi"),
        }
        drv_like = MixtureCRNLikelihood(
            bins=np.array([0.0, 10.0, 100.0, np.inf]),
            **common,
        )
        joint_like = JointDrvmaxDtmaxCRNLikelihood(
            dt_at_dRVmax_real=observed_dt,
            drv_bins=np.array([0.0, 10.0, 100.0, np.inf]),
            dt_bins=np.array([0.0, np.inf]),
            **common,
        )
        theta = np.array([0.6, 0.1])

        self.assertEqual(joint_like(theta), drv_like(theta))
        self.assertEqual(int(joint_like.observed_hist.sum()), observed_drv.size)
        self.assertEqual(int(joint_like.single_hist.sum()), single_bank.size)

    def test_baseline_conditioned_joint_one_time_bin_matches_drv_likelihood(self):
        _, survey = make_toy_survey()
        observed_drv = np.array([5.0, 12.0, 25.0, 40.0])
        observed_dt = np.array([20.0, 90.0, 100.0, 20.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        common = {
            "survey": survey,
            "dRV_real": observed_drv,
            "n_single_bank": 128,
            "n_binary_bank": 128,
            "bank_seeds": (611, 612),
            "parameter_names": ("f_bin", "pi"),
            "condition_by": "baseline_days",
            "baseline_bins": np.array([0.0, 50.0, 150.0, np.inf]),
            "observed_baseline_days": observed_baselines,
        }
        drv_like = AveragedMixtureCRNLikelihood(
            bins=np.array([0.0, 10.0, 30.0, np.inf]),
            **common,
        )
        joint_like = AveragedJointDrvmaxDtmaxCRNLikelihood(
            dt_at_dRVmax_real=observed_dt,
            drv_bins=np.array([0.0, 10.0, 30.0, np.inf]),
            dt_bins=np.array([0.0, np.inf]),
            **common,
        )
        theta = np.array([0.6, 0.1])

        self.assertEqual(joint_like(theta), drv_like(theta))
        np.testing.assert_array_equal(
            joint_like.observed_hist.sum(axis=(1, 2)),
            joint_like.observed_counts,
        )

    def test_joint_histograms_conserve_low_and_high_rows(self):
        _, survey = make_toy_survey()
        like = AveragedJointDrvmaxDtmaxCRNLikelihood(
            survey,
            np.array([0.0, 5.0, 1.0e9]),
            np.array([0.0, 5.0, 1.0e9]),
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(613, 614),
            parameter_names=("f_bin", "pi"),
            drv_bins=np.array([1.0, 10.0]),
            dt_bins=np.array([1.0, 10.0]),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, np.inf]),
            observed_baseline_days=np.array([0.0, 30.0, 1.0e9]),
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        single, binary = like.component_probabilities(params)
        needed = like.observed_counts > 0

        self.assertEqual(int(like.observed_hist.sum()), 3)
        np.testing.assert_allclose(single[needed].sum(axis=(1, 2)), 1.0)
        np.testing.assert_allclose(binary[needed].sum(axis=(1, 2)), 1.0)

    def test_joint_pooling_handles_unequal_banks_and_empty_condition_cells(self):
        _, survey = make_toy_survey()
        observed_drv = np.array([5.0, 12.0, 25.0, 40.0])
        observed_dt = np.array([20.0, 90.0, 100.0, 20.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        likelihoods = [
            JointDrvmaxDtmaxCRNLikelihood(
                survey,
                observed_drv,
                observed_dt,
                n_single_bank=size,
                n_binary_bank=size,
                bank_seed=seed,
                parameter_names=("f_bin", "pi"),
                drv_bins=np.array([0.0, 10.0, 30.0, np.inf]),
                dt_bins=np.array([0.0, 50.0, np.inf]),
                condition_by="baseline_days",
                baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
                observed_baseline_days=observed_baselines,
            )
            for size, seed in ((1, 615), (128, 616))
        ]
        like = AveragedJointDrvmaxDtmaxCRNLikelihood(
            survey,
            observed_drv,
            observed_dt,
            likelihoods=likelihoods,
            parameter_names=("f_bin", "pi"),
            condition_by="baseline_days",
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        bank_probabilities = [
            like.bank_binary_probability(index, params)
            for index in range(like.n_banks)
        ]
        expected = _pool_joint_probability_tables(
            bank_probabilities,
            [likelihood.binary_support_counts for likelihood in likelihoods],
            label="test",
        )

        np.testing.assert_allclose(like.component_probabilities(params)[1], expected)
        self.assertTrue(
            any(np.any(likelihood.binary_support_counts == 0) for likelihood in likelihoods)
        )
        theta = np.array([0.6, 0.1])
        self.assertEqual(like(theta), like(theta))

    def test_joint_time_marginal_matches_baseline_conditioned_drv_likelihood(self):
        _, survey = make_toy_survey()
        observed_drv = np.array([5.0, 12.0, 25.0, 40.0])
        observed_dt = np.array([20.0, 90.0, 100.0, 20.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        common = {
            "survey": survey,
            "dRV_real": observed_drv,
            "n_single_bank": 128,
            "n_binary_bank": 128,
            "bank_seeds": (620, 621),
            "parameter_names": ("f_bin", "pi"),
            "condition_by": "baseline_days",
            "baseline_bins": np.array([0.0, 50.0, 150.0, np.inf]),
            "observed_baseline_days": observed_baselines,
        }
        drv_like = AveragedMixtureCRNLikelihood(
            bins=np.array([0.0, 10.0, 30.0, np.inf]),
            **common,
        )
        joint_like = AveragedJointDrvmaxDtmaxCRNLikelihood(
            dt_at_dRVmax_real=observed_dt,
            drv_bins=np.array([0.0, 10.0, 30.0, np.inf]),
            dt_bins=np.array([0.0, 10.0, 50.0, np.inf]),
            **common,
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        drv_single, drv_binary = drv_like.component_probabilities(params)
        joint_single, joint_binary = joint_like.component_probabilities(params)

        np.testing.assert_allclose(joint_single.sum(axis=2), drv_single)
        np.testing.assert_allclose(joint_binary.sum(axis=2), drv_binary)
        np.testing.assert_array_equal(
            joint_like.observed_hist.sum(axis=2),
            drv_like.observed_hist,
        )

    def test_split_joint_population_matches_unsplit_identical_systems(self):
        _, survey = make_toy_survey()
        observed_drv = np.array([5.0, 12.0, 25.0, 40.0])
        observed_dt = np.array([20.0, 90.0, 100.0, 20.0])
        single_bank, binary_bank = build_mixture_crn_banks(survey, 128, 128, seed=622)
        common = {
            "survey": survey,
            "dRV_real": observed_drv,
            "dt_at_dRVmax_real": observed_dt,
            "parameter_names": ("f_bin", "pi"),
            "drv_bins": np.array([0.0, 10.0, 30.0, np.inf]),
            "dt_bins": np.array([0.0, 10.0, 50.0, np.inf]),
        }
        unsplit = JointDrvmaxDtmaxCRNLikelihood(
            single_bank=single_bank,
            binary_bank=binary_bank,
            **common,
        )
        pieces = []
        for indices in (np.arange(31), np.arange(31, 128)):
            piece_single, piece_binary = slice_random_banks(
                single_bank,
                binary_bank,
                indices,
            )
            pieces.append(
                JointDrvmaxDtmaxCRNLikelihood(
                    single_bank=piece_single,
                    binary_bank=piece_binary,
                    **common,
                )
            )
        split = AveragedJointDrvmaxDtmaxCRNLikelihood(
            survey,
            observed_drv,
            observed_dt,
            likelihoods=pieces,
            parameter_names=("f_bin", "pi"),
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        unsplit_components = unsplit.component_probabilities(params)
        split_components = split.component_probabilities(params)

        np.testing.assert_array_equal(split_components[0], unsplit_components[0])
        np.testing.assert_array_equal(split_components[1], unsplit_components[1])
        self.assertEqual(split.log_likelihood(params), unsplit.log_likelihood(params))

    def test_ordinary_crn_and_pairwise_paths_share_corrected_amplitudes(self):
        pop, survey = make_single_template_survey()
        pop.use_roche_guard = False
        cadence = CadenceRandomBank(
            template_indices=np.array([0], dtype=np.int32),
            noise_unit=(np.zeros(3),),
        )
        single_bank = SingleRandomBank(cadence=cadence)
        binary_bank = BinaryRandomBank(
            u_m1=np.array([0.65]),
            u_logP=np.array([0.55]),
            u_q=np.array([0.70]),
            u_e=np.array([0.80]),
            u_cos_i=np.array([0.50]),
            u_omega=np.array([0.30]),
            u_Tp=np.array([0.20]),
            cadence=cadence,
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.2, "eta": -0.4}
        drv_like = MixtureCRNLikelihood(
            survey,
            np.array([1.0]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            bins=np.array([0.0, 1.0e6]),
        )
        pairwise_like = PairwiseMixtureCRNLikelihood(
            survey,
            np.array([[1.0]]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            config=PairwiseSummaryConfig(
                delta_time_bins=(0.0, np.inf),
                response="max_abs_delta_rv",
                response_bins=(0.0, 1.0e6),
            ),
        )

        drv_arrays = drv_like._binary_intrinsic_arrays(params)
        pairwise_arrays = pairwise_like._binary_intrinsic_arrays(params)
        expected_k1, expected_k2 = orbital_semi_amplitudes_kms(
            drv_arrays["M1"],
            drv_arrays["M2"],
            drv_arrays["P"],
            np.full(1, np.pi / 2.0),
            drv_arrays["e"],
        )
        ordinary = pop.sample_orbital_extras_vectorized(
            drv_arrays["M1"],
            np.log10(drv_arrays["P"]),
            drv_arrays["q"],
            drv_arrays["e"],
            inc_mode="edge_on",
        )

        self.assertGreater(float(drv_arrays["e"][0]), 0.0)
        np.testing.assert_allclose(drv_arrays["K1"], expected_k1)
        np.testing.assert_allclose(drv_arrays["K2"], expected_k2)
        np.testing.assert_allclose(pairwise_arrays["K1"], expected_k1)
        np.testing.assert_allclose(pairwise_arrays["K2"], expected_k2)
        np.testing.assert_allclose(ordinary["K1"], expected_k1)
        np.testing.assert_allclose(ordinary["K2"], expected_k2)

    def test_probability_pooling_uses_support_counts(self):
        probabilities = np.array([[0.5, 0.5], [0.0, 1.0]])

        equal = _pool_probability_tables(
            probabilities,
            [4, 4],
            label="test",
        )
        unequal = _pool_probability_tables(
            probabilities,
            [2, 6],
            label="test",
        )

        np.testing.assert_allclose(equal, [0.25, 0.75])
        np.testing.assert_allclose(unequal, [0.125, 0.875])
        np.testing.assert_allclose(
            _pool_probability_tables(
                np.array([[0.2, 0.8], [0.2, 0.8]]),
                [1, 9],
                label="test",
            ),
            [0.2, 0.8],
        )

    def test_probability_pooling_handles_condition_specific_empty_support(self):
        probabilities = np.array(
            [
                [[0.25, 0.75], [0.0, 0.0]],
                [[0.50, 0.50], [0.10, 0.90]],
            ]
        )
        support = np.array([[4, 0], [12, 5]])

        first = _pool_probability_tables(probabilities, support, label="test")
        second = _pool_probability_tables(probabilities, support, label="test")

        np.testing.assert_allclose(first[0], [0.4375, 0.5625])
        np.testing.assert_allclose(first[1], [0.10, 0.90])
        np.testing.assert_array_equal(first, second)

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

    def test_equal_epoch_batching_matches_per_template_reference(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        like = MixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=512,
            n_binary_bank=512,
            bank_seed=127,
            parameter_names=("f_bin", "pi"),
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}

        arrays = like._binary_intrinsic_arrays(params)
        reference_drv = np.empty(like.binary_bank.size, dtype=float)
        reference_dt = np.empty(like.binary_bank.size, dtype=float)
        template_indices = np.asarray(
            like.binary_bank.cadence.template_indices,
            dtype=np.int32,
        )
        for template_index in np.unique(template_indices):
            indices = np.flatnonzero(template_indices == template_index)
            t_array = np.asarray(like.template_mjds[template_index], dtype=float)
            rv_errors = np.asarray(
                like.template_rv_errors[template_index],
                dtype=float,
            )
            noise = np.column_stack(
                [
                    like.binary_bank.cadence.noise_unit[int(index)]
                    for index in indices
                ]
            )
            rv_obs = like.population.rvcurve(
                t_array[:, None],
                arrays["P"][indices],
                arrays["Tp"][indices],
                arrays["e"][indices],
                arrays["omega_deg"][indices],
                0.0,
                arrays["K1"][indices],
                0.0,
                SB2=False,
            )
            rv_obs = rv_obs + noise * rv_errors[:, None]
            columns = np.arange(indices.size)
            maximum = np.nanargmax(rv_obs, axis=0)
            minimum = np.nanargmin(rv_obs, axis=0)
            reference_drv[indices] = (
                rv_obs[maximum, columns] - rv_obs[minimum, columns]
            )
            reference_dt[indices] = np.abs(
                t_array[maximum] - t_array[minimum]
            )

        batched_drv = like.simulate_binary_drv(params)
        actual_drv, actual_dt = like.simulate_binary_drv_and_dt_at_max(params)
        assigned_epoch_counts = np.asarray(
            [
                len(like.template_mjds[int(template_index)])
                for template_index in template_indices
            ],
            dtype=int,
        )

        self.assertEqual(
            len(like.binary_cadence_groups),
            np.unique(assigned_epoch_counts).size,
        )
        np.testing.assert_allclose(
            batched_drv,
            reference_drv,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual_drv,
            reference_drv,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_array_equal(actual_dt, reference_dt)

    def test_mixture_crn_histograms_account_for_low_and_high_tails(self):
        _, survey = make_toy_survey()
        observed = np.array([0.0, 0.5, 2.0, 12.0, 1500.0])
        like = MixtureCRNLikelihood(
            survey,
            observed,
            n_single_bank=64,
            n_binary_bank=64,
            bank_seed=124,
            parameter_names=("f_bin", "pi"),
        )

        self.assertEqual(like.bins[0], 0.0)
        self.assertTrue(np.isposinf(like.bins[-1]))
        self.assertEqual(int(like.n_real.sum()), len(observed))
        self.assertEqual(int(like.single_hist.sum()), like.single_bank.size)
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        _, binary_probability = like.component_probabilities(params)
        np.testing.assert_allclose(binary_probability.sum(), 1.0)

    def test_mixture_crn_rejects_invalid_observed_drv_values(self):
        _, survey = make_toy_survey()
        for invalid in (-1.0, np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    MixtureCRNLikelihood(
                        survey,
                        np.array([1.0, invalid]),
                        n_single_bank=8,
                        n_binary_bank=8,
                        bank_seed=125,
                        parameter_names=("f_bin", "pi"),
                    )

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

    def test_unequal_global_banks_match_count_weighted_pooling(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        likelihoods = [
            MixtureCRNLikelihood(
                survey,
                observed,
                n_single_bank=size,
                n_binary_bank=size,
                bank_seed=seed,
                parameter_names=("f_bin", "pi"),
            )
            for size, seed in ((32, 401), (96, 402))
        ]
        averaged = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            likelihoods=likelihoods,
            parameter_names=("f_bin", "pi"),
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}

        expected_single = np.average(
            [likelihood.single_probability for likelihood in likelihoods],
            axis=0,
            weights=[likelihood.single_bank.size for likelihood in likelihoods],
        )
        bank_binary = [
            likelihood.component_probabilities(params)[1]
            for likelihood in likelihoods
        ]
        expected_binary = np.average(
            bank_binary,
            axis=0,
            weights=[likelihood.binary_bank.size for likelihood in likelihoods],
        )
        actual_single, actual_binary = averaged.component_probabilities(params)

        np.testing.assert_allclose(actual_single, expected_single)
        np.testing.assert_allclose(actual_binary, expected_binary)
        np.testing.assert_allclose(actual_single.sum(), 1.0)
        np.testing.assert_allclose(actual_binary.sum(), 1.0)

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

    def test_conditioned_pooling_uses_per_cell_support_and_allows_empty_bank_cells(self):
        _, survey = make_toy_survey()
        observed = np.array([5.0, 12.0, 25.0, 40.0])
        observed_baselines = np.array([30.0, 100.0, 120.0, 30.0])
        likelihoods = [
            MixtureCRNLikelihood(
                survey,
                observed,
                n_single_bank=size,
                n_binary_bank=size,
                bank_seed=seed,
                parameter_names=("f_bin", "pi"),
                bins=np.array([0.0, 10.0, 30.0, np.inf]),
            )
            for size, seed in ((1, 501), (128, 502))
        ]
        averaged = AveragedMixtureCRNLikelihood(
            survey,
            observed,
            likelihoods=likelihoods,
            parameter_names=("f_bin", "pi"),
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, 150.0, np.inf]),
            observed_baseline_days=observed_baselines,
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        bank_binary = [
            averaged.bank_binary_probability(index, params)
            for index in range(averaged.n_banks)
        ]
        expected_binary = _pool_probability_tables(
            bank_binary,
            [state.binary_counts for state in averaged.conditioned_bank_states],
            label="test",
        )

        _, actual_binary = averaged.component_probabilities(params)
        np.testing.assert_allclose(actual_binary, expected_binary)
        needed = averaged.observed_counts > 0
        np.testing.assert_allclose(averaged.single_probability[needed].sum(axis=1), 1.0)
        np.testing.assert_allclose(actual_binary[needed].sum(axis=1), 1.0)
        self.assertTrue(
            any(
                np.any(state.single_counts[needed] == 0)
                or np.any(state.binary_counts[needed] == 0)
                for state in averaged.conditioned_bank_states
            )
        )

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

    def test_grouped_binary_drv_matches_serial_blending_path(self):
        pop, survey = make_toy_survey()
        like = MixtureCRNLikelihood(
            survey,
            np.array([1.0, 5.0, 20.0]),
            n_single_bank=12,
            n_binary_bank=48,
            bank_seed=20260710,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            bins=np.array([0.0, 10.0, 100.0, 1.0e6]),
            blending_kernel=FluxAwareTestBlendKernel(),
            blending_flux_fraction=lambda **kwargs: 0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        grouped = like.simulate_binary_drv(params)

        arrays = like._binary_intrinsic_arrays(params)
        serial = np.empty(like.binary_bank.size, dtype=float)
        for idx, template_idx in enumerate(like.binary_bank.cadence.template_indices):
            template_idx = int(template_idx)
            t_array = like.template_mjds[template_idx]
            rv_errors = like.template_rv_errors[template_idx]
            v1_true, v2_true = pop.rvcurve(
                t_array,
                arrays["P"][idx],
                arrays["Tp"][idx],
                arrays["e"][idx],
                arrays["omega_deg"][idx],
                0.0,
                arrays["K1"][idx],
                arrays["K2"][idx],
                SB2=True,
            )
            v1_true = v1_true + like._binary_blending_bias(
                arrays,
                idx,
                v1_true,
                v2_true,
            )
            rv_obs = v1_true + like.binary_bank.cadence.noise_unit[idx] * rv_errors
            serial[idx] = float(np.nanmax(rv_obs) - np.nanmin(rv_obs))

        self.assertGreater(len(like.binary_cadence_groups), 1)
        np.testing.assert_allclose(grouped, serial)

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

    def test_unequal_pairwise_banks_match_count_weighted_pooling(self):
        _, survey = make_toy_survey()
        observed = np.array([[5.0], [12.0], [25.0], [40.0]])
        config = PairwiseSummaryConfig(
            delta_time_bins=(0.0, np.inf),
            response="max_abs_delta_rv",
            response_bins=(0.0, 10.0, 20.0, 50.0, np.inf),
        )
        likelihoods = [
            PairwiseMixtureCRNLikelihood(
                survey,
                observed,
                n_single_bank=size,
                n_binary_bank=size,
                bank_seed=seed,
                parameter_names=("f_bin", "pi"),
                config=config,
                require_observed_support=False,
            )
            for size, seed in ((24, 601), (72, 602))
        ]
        averaged = AveragedMixtureCRNPairwiseLikelihood(
            survey,
            observed,
            likelihoods=likelihoods,
            parameter_names=("f_bin", "pi"),
            config=config,
        )
        params = {"f_bin": 0.6, "pi": 0.1, "kappa": 0.0, "eta": -0.5}
        binary_results = [
            likelihood.binary_component_probability(params)
            for likelihood in likelihoods
        ]
        expected_binary = _pool_probability_tables(
            [probability for probability, _ in binary_results],
            [support for _, support in binary_results],
            label="test",
        )

        actual_single, actual_binary = averaged.component_probabilities(params)
        np.testing.assert_allclose(actual_single.sum(axis=1), 1.0)
        np.testing.assert_allclose(actual_binary, expected_binary)
        np.testing.assert_allclose(actual_binary.sum(axis=1), 1.0)

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

    def test_blending_dispatcher_preserves_legacy_absolute_keyword(self):
        separation = np.array([2.0, 4.0])
        unit = np.array([0.1, 0.2])
        kernel = FluxAwareTestBlendKernel()

        bias = sample_blending_bias(
            kernel,
            abs_velocity_separation=separation,
            secondary_flux_fraction=0.25,
            blend_unit=unit,
        )

        np.testing.assert_allclose(
            bias,
            kernel.sample_bias(separation, 0.25, unit),
        )

    def test_signed_blending_kernel_rejects_absolute_only_dispatch(self):
        with self.assertRaisesRegex(ValueError, "signed blending kernel"):
            sample_blending_bias(
                SignedFractionalTestBlendKernel(),
                abs_velocity_separation=np.array([2.0]),
                secondary_flux_fraction=0.25,
                blend_unit=np.array([0.1]),
            )

    def test_signed_blending_kernel_suppresses_drvmax_before_summary(self):
        pop, survey = make_single_template_survey()
        blend_unit = (np.array([0.0, 0.25, 0.75]),)
        single_bank, binary_bank = make_one_system_banks(blend_unit=blend_unit)
        kernel = SignedFractionalTestBlendKernel()
        like = MixtureCRNLikelihood(
            survey,
            np.array([1.0]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            bins=np.array([0.0, np.inf]),
            blending_kernel=kernel,
            blending_flux_fraction=0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        arrays = like._binary_intrinsic_arrays(params)
        v1_true, v2_true = pop.rvcurve(
            like.template_mjds[0],
            arrays["P"][0],
            arrays["Tp"][0],
            arrays["e"][0],
            arrays["omega_deg"][0],
            0.0,
            arrays["K1"][0],
            arrays["K2"][0],
            SB2=True,
        )
        separation = v2_true - v1_true
        expected_bias = kernel.metadata["beta"] * separation
        expected_drv = np.ptp(v1_true + expected_bias)

        np.testing.assert_allclose(like.simulate_binary_drv(params), [expected_drv])
        np.testing.assert_array_equal(
            np.sign(expected_bias[separation != 0.0]),
            np.sign(separation[separation != 0.0]),
        )
        self.assertLess(expected_drv, np.ptp(v1_true))
        self.assertEqual(
            like.blending_metadata["sampling_protocol"],
            "signed_velocity_separation",
        )

    def test_signed_blending_kernel_suppresses_pairwise_summary(self):
        pop, survey = make_single_template_survey()
        blend_unit = (np.array([0.0, 0.25, 0.75]),)
        single_bank, binary_bank = make_one_system_banks(blend_unit=blend_unit)
        kernel = SignedFractionalTestBlendKernel()
        config = PairwiseSummaryConfig(
            delta_time_bins=(0.0, np.inf),
            response="max_abs_delta_rv",
            response_bins=(0.0, 10.0, 100.0, np.inf),
        )
        like = PairwiseMixtureCRNLikelihood(
            survey,
            np.array([[1.0]]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            config=config,
            blending_kernel=kernel,
            blending_flux_fraction=0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        arrays = like._binary_intrinsic_arrays(params)
        v1_true, v2_true = pop.rvcurve(
            like.template_mjds[0],
            arrays["P"][0],
            arrays["Tp"][0],
            arrays["e"][0],
            arrays["omega_deg"][0],
            0.0,
            arrays["K1"][0],
            arrays["K2"][0],
            SB2=True,
        )
        expected_response = np.ptp(
            v1_true + kernel.metadata["beta"] * (v2_true - v1_true)
        )

        np.testing.assert_allclose(
            like.simulate_binary_pairwise_summary(params),
            [[expected_response]],
        )
        self.assertLess(expected_response, np.ptp(v1_true))

    def test_signed_blending_kernel_precedes_joint_drvmax_dtmax_summary(self):
        pop, survey = make_single_template_survey()
        blend_unit = (np.array([0.0, 0.25, 0.75]),)
        single_bank, binary_bank = make_one_system_banks(blend_unit=blend_unit)
        kernel = SignedFractionalTestBlendKernel()
        like = JointDrvmaxDtmaxCRNLikelihood(
            survey,
            np.array([1.0]),
            np.array([2.0]),
            single_bank=single_bank,
            binary_bank=binary_bank,
            parameter_names=("f_bin", "pi", "kappa", "eta"),
            drv_bins=np.array([0.0, np.inf]),
            dt_bins=np.array([0.0, np.inf]),
            blending_kernel=kernel,
            blending_flux_fraction=0.25,
        )
        params = {"f_bin": 1.0, "pi": 0.1, "kappa": 0.0, "eta": -0.4}

        arrays = like._binary_intrinsic_arrays(params)
        t_array = np.asarray(like.template_mjds[0], dtype=float)
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
        rv_corrected = v1_true + kernel.metadata["beta"] * (v2_true - v1_true)
        max_index = int(np.argmax(rv_corrected))
        min_index = int(np.argmin(rv_corrected))

        actual_drv, actual_dt = like.simulate_binary_drv_and_dt_at_max(params)
        np.testing.assert_allclose(
            actual_drv,
            [rv_corrected[max_index] - rv_corrected[min_index]],
        )
        np.testing.assert_allclose(
            actual_dt,
            [abs(t_array[max_index] - t_array[min_index])],
        )

    def test_tiny_generated_no_blending_sample_scores_baseline_and_joint(self):
        _, survey = make_toy_survey()
        observed_drv, observed_dt, observed_baselines = make_tiny_observed_summaries(
            survey
        )
        common = {
            "survey": survey,
            "dRV_real": observed_drv,
            "n_single_bank": 64,
            "n_binary_bank": 64,
            "bank_seeds": (20260802,),
            "parameter_names": ("f_bin", "pi"),
            "condition_by": "baseline_days",
            "baseline_bins": np.array([0.0, 50.0, np.inf]),
            "observed_baseline_days": observed_baselines,
        }
        baseline = AveragedMixtureCRNLikelihood(
            bins=np.array([0.0, np.inf]),
            **common,
        )
        joint = AveragedJointDrvmaxDtmaxCRNLikelihood(
            dt_at_dRVmax_real=observed_dt,
            drv_bins=np.array([0.0, np.inf]),
            dt_bins=np.array([0.0, np.inf]),
            **common,
        )
        params = {"f_bin": 0.5, "pi": 0.0, "kappa": 0.0, "eta": -0.5}
        baseline_single, baseline_binary = baseline.component_probabilities(params)
        joint_single, joint_binary = joint.component_probabilities(params)

        np.testing.assert_allclose(joint_single.sum(axis=2), baseline_single)
        np.testing.assert_allclose(joint_binary.sum(axis=2), baseline_binary)
        self.assertTrue(np.isfinite(baseline.log_likelihood(params)))
        self.assertTrue(np.isfinite(joint.log_likelihood(params)))
        self.assertFalse(baseline.reference.blending_metadata["enabled"])
        self.assertFalse(joint.reference.blending_metadata["enabled"])

    def test_tiny_signed_blending_joint_runner_end_to_end(self):
        pop, survey = make_toy_survey()
        observed_drv, observed_dt, observed_baselines = make_tiny_observed_summaries(
            survey,
            seed=20260803,
        )
        np.random.seed(20260803)
        sampler = run_averaged_joint_drvmax_dtmax_crn_mcmc(
            pop,
            survey,
            observed_drv,
            observed_dt,
            observed_baseline_days=observed_baselines,
            condition_by="baseline_days",
            baseline_bins=np.array([0.0, 50.0, np.inf]),
            drv_bins=np.array([0.0, np.inf]),
            dt_bins=np.array([0.0, np.inf]),
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(20260803,),
            nwalkers=8,
            nsteps=2,
            nthreads=1,
            pool_kind="none",
            progress=False,
            parameter_names=("f_bin", "pi"),
            initial_position={"f_bin": 0.5, "pi": 0.0},
            initial_scatter={"f_bin": 0.03, "pi": 0.05},
            blending_kernel=SignedFractionalTestBlendKernel(),
            blending_flux_fraction=0.25,
        )

        self.assertEqual(sampler.get_chain().shape, (2, 8, 2))
        self.assertTrue(np.all(np.isfinite(sampler.get_log_prob())))
        self.assertEqual(
            sampler.joint_drvmax_dtmax_crn_likelihood.blending_metadata[
                "sampling_protocol"
            ],
            "signed_velocity_separation",
        )

    @unittest.skipUnless(
        "fork" in mp.get_all_start_methods(),
        "joint bank_static_process smoke uses fork",
    )
    def test_run_averaged_joint_drvmax_dtmax_crn_mcmc_bank_static_smoke(self):
        pop, survey = make_toy_survey()
        np.random.seed(20260804)
        sampler = run_averaged_joint_drvmax_dtmax_crn_mcmc(
            pop,
            survey,
            np.array([5.0, 12.0, 25.0, 40.0]),
            np.array([5.0, 20.0, 30.0, 100.0]),
            n_single_bank=64,
            n_binary_bank=64,
            bank_seeds=(41, 42),
            nwalkers=8,
            nsteps=2,
            nthreads=2,
            pool_kind="bank_static_process",
            start_method="fork",
            progress=False,
            parameter_names=("f_bin", "pi"),
            initial_position={"f_bin": 0.6, "pi": 0.0},
            initial_scatter={"f_bin": 0.03, "pi": 0.05},
            drv_bins=np.array([0.0, 10.0, 20.0, 50.0, np.inf]),
            dt_bins=np.array([0.0, 10.0, 50.0, np.inf]),
        )

        self.assertEqual(sampler.get_chain().shape, (2, 8, 2))
        self.assertTrue(np.all(np.isfinite(sampler.get_log_prob())))


if __name__ == "__main__":
    unittest.main()
