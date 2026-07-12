import os
import tempfile
import unittest

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np

from minato import ravel


class RavelProfileTests(unittest.TestCase):
    def test_classic_sb1_fit_recovers_synthetic_line_centre(self):
        wave = np.linspace(4548.0, 4558.0, 161)
        expected_centre = 4554.2
        flux = 1.0 - 0.30 * np.exp(
            -0.5 * ((wave - expected_centre) / 0.65) ** 2
        )
        error = np.full_like(wave, 0.01)

        result, _, _, _ = ravel.fit_sb1(
            4553,
            wave,
            flux,
            error,
            ravel.setup_line_dictionary(),
            Hlines=[],
            neblines=[],
            doubem=[],
            shift=0.0,
        )

        self.assertAlmostEqual(result.params["g1_cen"].value, expected_centre, 3)


@unittest.skipUnless(
    os.getenv("MINATO_RUN_RAVEL_SMOKE") == "1",
    "set MINATO_RUN_RAVEL_SMOKE=1 for probabilistic release smoke tests",
)
class RavelProbabilisticSmokeTests(unittest.TestCase):
    def test_single_epoch_sb1_and_sb2_runs(self):
        wave = np.linspace(4548.0, 4558.0, 61)
        error = np.full_like(wave, 0.02)
        lines = ravel.setup_line_dictionary()

        sb1_flux = 1.0 - 0.25 * np.exp(-0.5 * ((wave - 4554.0) / 0.70) ** 2)
        sb2_flux = (
            1.0
            - 0.22 * np.exp(-0.5 * ((wave - 4551.3) / 0.65) ** 2)
            - 0.16 * np.exp(-0.5 * ((wave - 4554.4) / 0.75) ** 2)
        )

        with tempfile.TemporaryDirectory() as output_dir:
            common = {
                "num_warmup": 5,
                "num_samples": 10,
                "num_chains": 1,
                "chain_method": "sequential",
                "max_interp_points": 50,
                "plots": False,
                "verbose": False,
                "progress": False,
            }
            sb1, _, _ = ravel.fit_sb1_probmod(
                [4553],
                [wave],
                [sb1_flux],
                [error],
                lines,
                [],
                [],
                output_dir,
                profile="Gaussian",
                cornerplot=False,
                **common,
            )
            sb2, _, _ = ravel.fit_sb2_probmod(
                [4553],
                [wave],
                [sb2_flux],
                [error],
                lines,
                [],
                [],
                output_dir,
                sigma_prior=150,
                profile="Gaussian",
                **common,
            )

        sb1_rv = np.asarray(sb1["Δv"])
        sb2_rv = np.asarray(sb2["Δv_τk"])
        self.assertEqual(sb1_rv.shape, (10, 1, 1))
        self.assertEqual(sb2_rv.shape, (10, 2, 1, 1))
        self.assertTrue(np.isfinite(sb1_rv).all())
        self.assertTrue(np.isfinite(sb2_rv).all())


if __name__ == "__main__":
    unittest.main()
