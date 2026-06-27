import unittest

import numpy as np

from minato.span import AtmFit


class ToyAtmFit(AtmFit):
    def __init__(self, wavelength, fluxA, fluxB, modelA, modelB, grid, lrat0):
        super().__init__(
            spectrumA=None,
            spectrumB=None,
            grid=grid,
            lrat0=lrat0,
            modelsA_path="A",
            modelsB_path="B",
            binary=True,
        )
        self._wavelength = np.asarray(wavelength, dtype=float)
        self._fluxA = np.asarray(fluxA, dtype=float)
        self._fluxB = np.asarray(fluxB, dtype=float)
        self._modelA = np.asarray(modelA, dtype=float)
        self._modelB = np.asarray(modelB, dtype=float)
        self.wavA = self._wavelength
        self.wavB = self._wavelength
        self.user_dicA = {5000: {"region": [4998.0, 5002.0], "HeH_region": []}}
        self.user_dicB = {5000: {"region": [4998.0, 5002.0], "HeH_region": []}}
        self.nparams = len(grid)
        self.cols = list(grid.keys())

    def get_flux(self):
        return self._fluxA.copy(), self._fluxB.copy()

    def get_model(self, pars, models_path=None, source=None):
        if models_path == "A":
            return self._wavelength.copy(), self._modelA.copy(), "modelA"
        if models_path == "B":
            return self._wavelength.copy(), self._modelB.copy(), "modelB"
        raise ValueError("unknown toy model path")


class SpanLightRatioLikelihoodTests(unittest.TestCase):
    def test_model_flux_to_initial_light_ratio_uses_component_scaling(self):
        fit = AtmFit(None, None, lrat0=0.30)
        model_flux = np.array([1.0, 0.8, 0.9])

        modelA = fit.model_flux_to_initial_light_ratio(model_flux, "A", 0.45)
        modelB = fit.model_flux_to_initial_light_ratio(model_flux, "B", 0.45)

        np.testing.assert_allclose(
            modelA,
            1 + (model_flux - 1) * ((1 - 0.45) / (1 - 0.30)),
        )
        np.testing.assert_allclose(
            modelB,
            1 + (model_flux - 1) * (0.45 / 0.30),
        )

    def test_compute_single_set_scores_light_ratios_in_common_noise_scale(self):
        wavelength = np.linspace(4999.0, 5001.0, 401)
        lrat0 = 0.30
        true_lrat = 0.30
        lrat_grid = [0.15, 0.30, 0.45, 0.60, 0.75]
        modelA = np.ones_like(wavelength)
        modelB = 1.0 - 0.04 * np.exp(-0.5 * ((wavelength - 5000.0) / 0.2) ** 2)

        rng = np.random.default_rng(0)
        fluxA = np.ones_like(wavelength)
        fluxB = 1 + (modelB - 1) * (true_lrat / lrat0)
        fluxB = fluxB + rng.normal(0.0, 0.02, wavelength.size)

        grid = {"lr": lrat_grid, "TA": [1], "TB": [1]}
        fit = ToyAtmFit(wavelength, fluxA, fluxB, modelA, modelB, grid, lrat0)

        corrected_scores = {
            lrat: fit.compute_single_set((lrat, 1, 1))[len(grid)]
            for lrat in lrat_grid
        }
        old_scores = {
            lrat: self._old_rescaled_observation_score(fit, lrat)
            for lrat in lrat_grid
        }

        self.assertEqual(min(corrected_scores, key=corrected_scores.get), true_lrat)
        self.assertEqual(min(old_scores, key=old_scores.get), lrat_grid[-1])

    def _old_rescaled_observation_score(self, fit, lrat):
        fluxA, fluxB = fit.get_flux()
        modelA = fit._modelA
        modelB = fit._modelB
        ratio0 = fit.lrat0

        fluxA = 1 + (fluxA - 1) * ((1 - ratio0) / (1 - lrat))
        fluxB = 1 + (fluxB - 1) * (ratio0 / lrat)
        return fit.chi2(fluxA, modelA) + fit.chi2(fluxB, modelB)


if __name__ == "__main__":
    unittest.main()
