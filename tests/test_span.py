import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from minato import read_results
from minato.span import AtmFit
from minato.contrib.spdis import SpecDisent
from minato.synthetic import RenderedAtmosphereGrid, Spectrum
from scripts.generate_span_tutorial_data import (
    DEFAULT_DISENTANGLING_ITERATIONS,
    build_parser,
    generate_inputs,
    orbital_semi_amplitudes,
)


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
    def test_hdelta_window_uses_intended_wavelengths(self):
        self.assertEqual(AtmFit.lines_dic[4102]["region"], [4087, 4130])

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

    def test_chi2_is_non_negative_for_negative_trial_flux(self):
        fit = AtmFit(None, None)
        observed = np.array([1.0, 0.8, 1.0])
        expected = np.array([1.0, -0.5, 1.0])

        score = fit.chi2(observed, expected)

        self.assertGreaterEqual(score, 0.0)
        self.assertAlmostEqual(score, 1.3**2)

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

    def test_read_spec_accepts_comments_and_explicit_zero_shift(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path_a = Path(directory) / "primary.txt"
            path_b = Path(directory) / "secondary.txt"
            data = np.array([[4000.0, 1.0], [4001.0, 0.9]])
            np.savetxt(path_a, data, header="wavelength_A normalised_flux")
            np.savetxt(path_b, data, header="wavelength_A normalised_flux")
            fit = AtmFit(path_a, path_b, wavelength_shift=0.0)

            wave_a, wave_b = fit.get_wave(shift=fit.wavelength_shift)
            flux_a, flux_b = fit.get_flux()

            np.testing.assert_allclose(wave_a, data[:, 0])
            np.testing.assert_allclose(wave_b, data[:, 0])
            np.testing.assert_allclose(flux_a, data[:, 1])
            np.testing.assert_allclose(flux_b, data[:, 1])

    def test_worker_controls_are_validated(self):
        fit = AtmFit(None, None, max_workers=2, chunksize=16)
        self.assertEqual(fit.max_workers, 2)
        self.assertEqual(fit.chunksize, 16)
        with self.assertRaisesRegex(ValueError, "max_workers"):
            AtmFit(None, None, max_workers=0)
        with self.assertRaisesRegex(ValueError, "chunksize"):
            AtmFit(None, None, chunksize=0)

    def test_model_is_interpolated_onto_observed_grid(self):
        fit = AtmFit(None, None)
        observed = np.array([4000.0, 4000.5, 4001.0])
        model_wavelength = np.array([3999.5, 4000.25, 4000.75, 4001.5])
        model_flux = np.array([1.0, 0.8, 0.9, 1.0])

        interpolated = fit.model_on_observed_grid(
            observed,
            model_wavelength,
            model_flux,
        )

        np.testing.assert_allclose(
            interpolated,
            np.interp(observed, model_wavelength, model_flux),
        )

    def _old_rescaled_observation_score(self, fit, lrat):
        fluxA, fluxB = fit.get_flux()
        modelA = fit._modelA
        modelB = fit._modelB
        ratio0 = fit.lrat0

        fluxA = 1 + (fluxA - 1) * ((1 - ratio0) / (1 - lrat))
        fluxB = 1 + (fluxB - 1) * (ratio0 / lrat)
        return fit.chi2(fluxA, modelA) + fit.chi2(fluxB, modelB)


class SpanRenderedGridTests(unittest.TestCase):
    @staticmethod
    def _absorption(wavelength, depth):
        return 1.0 - depth * np.exp(
            -0.5 * ((wavelength - 4340.0) / 0.45) ** 2
        )

    def test_binary_fit_recovers_parameters_from_in_memory_grid(self):
        wavelength = np.linspace(4318.0, 4364.0, 1001)
        models = {
            (30_000, 4.0, 50): Spectrum(
                wavelength,
                self._absorption(wavelength, 0.20),
            ),
            (32_000, 4.0, 50): Spectrum(
                wavelength,
                self._absorption(wavelength, 0.35),
            ),
            (20_000, 4.2, 75): Spectrum(
                wavelength,
                self._absorption(wavelength, 0.12),
            ),
            (22_000, 4.2, 75): Spectrum(
                wavelength,
                self._absorption(wavelength, 0.25),
            ),
        }
        primary_grid = RenderedAtmosphereGrid(
            {key: model for key, model in models.items() if key[1] == 4.0}
        )
        secondary_grid = RenderedAtmosphereGrid(
            {key: model for key, model in models.items() if key[1] == 4.2}
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            spectrum_a = Path(directory) / "primary.txt"
            spectrum_b = Path(directory) / "secondary.txt"
            np.savetxt(
                spectrum_a,
                np.column_stack([wavelength, models[(32_000, 4.0, 50)].flux]),
            )
            np.savetxt(
                spectrum_b,
                np.column_stack([wavelength, models[(22_000, 4.2, 75)].flux]),
            )

            grid = {
                "lr": [0.2, 0.3, 0.4],
                "TA": [30_000, 32_000],
                "gA": [4.0],
                "vA": [50],
                "TB": [20_000, 22_000],
                "gB": [4.2],
                "vB": [75],
            }
            fit = AtmFit(
                spectrum_a,
                spectrum_b,
                grid=grid,
                lrat0=0.3,
                binary=True,
                wavelength_shift=0.0,
                modelsA_grid=primary_grid,
                modelsB_grid=secondary_grid,
            )

            results = fit.compute_chi2([4340], [4340])
            best = results.loc[results["chi2_tot"].idxmin()]

            self.assertEqual(len(results), 12)
            self.assertAlmostEqual(best["lr"], 0.3)
            self.assertAlmostEqual(best["TA"], 32_000)
            self.assertAlmostEqual(best["gA"], 4.0)
            self.assertAlmostEqual(best["vA"], 50)
            self.assertAlmostEqual(best["TB"], 22_000)
            self.assertAlmostEqual(best["gB"], 4.2)
            self.assertAlmostEqual(best["vB"], 75)
            self.assertAlmostEqual(best["chi2_tot"], 0.0, places=12)

    def test_binary_fit_respects_irregular_atmosphere_nodes(self):
        wavelength = np.linspace(4318.0, 4364.0, 1001)
        primary_grid = RenderedAtmosphereGrid(
            {
                (32_000, 4.0, 50): Spectrum(
                    wavelength,
                    self._absorption(wavelength, 0.35),
                )
            }
        )
        secondary_grid = RenderedAtmosphereGrid(
            {
                (20_000, 4.0, 75): Spectrum(
                    wavelength,
                    self._absorption(wavelength, 0.10),
                ),
                (22_000, 4.0, 75): Spectrum(
                    wavelength,
                    self._absorption(wavelength, 0.18),
                ),
                (22_000, 4.2, 75): Spectrum(
                    wavelength,
                    self._absorption(wavelength, 0.25),
                ),
            }
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            spectrum_a = Path(directory) / "primary.txt"
            spectrum_b = Path(directory) / "secondary.txt"
            np.savetxt(
                spectrum_a,
                np.column_stack(
                    [wavelength, primary_grid.get_model(32_000, 4.0, 50).flux]
                ),
            )
            np.savetxt(
                spectrum_b,
                np.column_stack(
                    [wavelength, secondary_grid.get_model(22_000, 4.2, 75).flux]
                ),
            )
            fit = AtmFit(
                spectrum_a,
                spectrum_b,
                grid={
                    "lr": [0.3],
                    "TA": [32_000],
                    "gA": [4.0],
                    "vA": [50],
                    "TB": [20_000, 22_000],
                    "gB": [4.0, 4.2],
                    "vB": [75],
                },
                lrat0=0.3,
                binary=True,
                wavelength_shift=0.0,
                modelsA_grid=primary_grid,
                modelsB_grid=secondary_grid,
            )

            results = fit.compute_chi2([4340], [4340])
            best = results.loc[results["chi2_tot"].idxmin()]

            self.assertEqual(len(results), 3)
            self.assertFalse(
                ((results["TB"] == 20_000) & (results["gB"] == 4.2)).any()
            )
            self.assertEqual(best["TB"], 22_000)
            self.assertEqual(best["gB"], 4.2)

    def test_single_star_fit_uses_only_the_primary_in_memory_grid(self):
        wavelength = np.linspace(4318.0, 4364.0, 1001)
        rendered_grid = RenderedAtmosphereGrid(
            {
                (30_000, 4.0, 50): Spectrum(
                    wavelength,
                    self._absorption(wavelength, 0.20),
                ),
                (32_000, 4.0, 50): Spectrum(
                    wavelength,
                    self._absorption(wavelength, 0.35),
                ),
            }
        )

        with tempfile.TemporaryDirectory(dir=".") as directory:
            spectrum = Path(directory) / "star.txt"
            np.savetxt(
                spectrum,
                np.column_stack(
                    [wavelength, rendered_grid.get_model(32_000, 4.0, 50).flux]
                ),
            )
            fit = AtmFit(
                spectrum,
                None,
                grid={"T": [30_000, 32_000], "g": [4.0], "v": [50]},
                wavelength_shift=0.0,
                modelsA_grid=rendered_grid,
            )

            results = fit.compute_chi2([4340], None)
            best = results.loc[results["chi2_tot"].idxmin()]

            self.assertEqual(len(results), 2)
            self.assertAlmostEqual(best["T"], 32_000)
            self.assertAlmostEqual(best["chi2_tot"], 0.0, places=12)

    def test_rendered_grid_and_model_directories_are_mutually_exclusive(self):
        wavelength = np.array([4339.0, 4340.0, 4341.0])
        rendered_grid = RenderedAtmosphereGrid(
            {(25_000, 4.0, 50): Spectrum(wavelength, np.ones(3))}
        )

        with self.assertRaisesRegex(ValueError, "either modelsA_grid"):
            AtmFit(
                None,
                None,
                modelsA_path="models",
                modelsA_grid=rendered_grid,
            )


class SpanTutorialGenerationTests(unittest.TestCase):
    def _write_model_pair(self, directory, stem, depth, calibrated_scale):
        wavelength = np.linspace(3970.0, 4590.0, 8001)
        normalised = 1.0 - depth * np.exp(-0.5 * ((wavelength - 4340.0) / 0.4) ** 2)
        calibrated = np.log10(calibrated_scale * normalised)
        normalised_path = Path(directory) / f"{stem}_line.txt"
        calibrated_path = Path(directory) / f"{stem}_line_calib.txt"
        np.savetxt(normalised_path, np.column_stack([wavelength, normalised]))
        np.savetxt(calibrated_path, np.column_stack([wavelength, calibrated]))
        return normalised_path, calibrated_path

    def test_orbital_semi_amplitudes_follow_inverse_mass_ratio(self):
        k_a, k_b = orbital_semi_amplitudes(19.04, 7.59, 17.0)

        self.assertGreater(k_a, 0)
        self.assertGreater(k_b, k_a)
        self.assertAlmostEqual(k_b / k_a, 19.04 / 7.59)

    def test_disentangling_iteration_default_matches_tutorial_provenance(self):
        parser_default = build_parser().get_default("iterations")
        provenance_path = (
            Path(__file__).resolve().parents[1]
            / "minato"
            / "tutorials"
            / "span_synthetic"
            / "provenance.json"
        )
        provenance = json.loads(provenance_path.read_text())

        self.assertEqual(DEFAULT_DISENTANGLING_ITERATIONS, 500)
        self.assertEqual(parser_default, DEFAULT_DISENTANGLING_ITERATIONS)
        self.assertEqual(
            provenance["disentangling"]["iterations"],
            DEFAULT_DISENTANGLING_ITERATIONS,
        )

    def test_spdis_text_reader_accepts_commented_metadata(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "spectrum.txt"
            np.savetxt(
                path,
                np.array([[4000.0, 1.0], [4001.0, 0.9]]),
                header="wavelength_A normalised_flux\nphase=0.0 snr=100",
            )

            spectrum = SpecDisent.__new__(SpecDisent).read_xytable(path)

            np.testing.assert_allclose(spectrum[0], [4000.0, 4001.0])
            np.testing.assert_allclose(spectrum[1], [1.0, 0.9])

    def test_generate_inputs_writes_ten_deterministic_epochs(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            primary_normalised, primary_calibrated = self._write_model_pair(
                directory, "primary", depth=0.4, calibrated_scale=4.0
            )
            secondary_normalised, secondary_calibrated = self._write_model_pair(
                directory, "secondary", depth=0.2, calibrated_scale=1.0
            )
            output = Path(directory) / "generation"

            provenance = generate_inputs(
                primary_normalised,
                secondary_normalised,
                primary_calibrated,
                secondary_calibrated,
                output,
            )

            spectra = sorted((output / "spectra").glob("epoch_*.txt"))
            self.assertEqual(len(spectra), 10)
            self.assertAlmostEqual(provenance["secondary"]["light_fraction"], 0.2)
            self.assertEqual(provenance["observation"]["seed"], 20260713)
            self.assertTrue((output / "epochs.txt").exists())
            self.assertTrue((output / "orbital_parameters.csv").exists())
            self.assertTrue((output / "provenance.json").exists())

            first = np.loadtxt(spectra[0])
            self.assertEqual(first.shape[1], 2)
            self.assertTrue(np.all(np.isfinite(first)))


class SpanResultPlotTests(unittest.TestCase):
    @staticmethod
    def _correlation_results():
        rows = []
        for light_ratio in [0.1, 0.2, 0.3]:
            for temperature in [30_000, 32_000, 34_000]:
                for gravity in [3.8, 4.0, 4.2]:
                    if temperature == 30_000 and gravity == 4.2:
                        continue
                    rows.append(
                        {
                            "lr": light_ratio,
                            "TA": temperature,
                            "gA": gravity,
                            "chi2_tot": 1.0
                            + 20 * (light_ratio - 0.2) ** 2
                            + ((temperature - 32_000) / 2_000) ** 2
                            + 4 * (gravity - 4.0) ** 2,
                            "ndata": 100,
                        }
                    )
        return pd.DataFrame(rows)

    def test_compute_bestfit_creates_no_implicit_output_files(self):
        results = pd.DataFrame(
            {
                "lr": [0.1, 0.2, 0.3, 0.4, 0.5],
                "chi2_tot": [1.0, 1.2, 2.0, 4.0, 8.0],
                "ndata": [100] * 5,
            }
        )
        original = results.copy(deep=True)

        with tempfile.TemporaryDirectory(dir=".") as directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                with patch("builtins.print"), patch("matplotlib.pyplot.show"):
                    figure = read_results.compute_bestfit(
                        results,
                        chi2max=150,
                        show_histogram=False,
                    )
                    figure.canvas.draw()
                    output_files = list(Path.cwd().iterdir())
            finally:
                os.chdir(previous_directory)

        self.assertEqual(len(figure.axes), 1)
        profile_minima = next(
            line
            for line in figure.axes[0].lines
            if line.get_label() == "Profile minima"
        )
        np.testing.assert_allclose(profile_minima.get_xdata(), results["lr"])
        self.assertEqual(output_files, [])
        pd.testing.assert_frame_equal(results, original)

    def test_plot_corr_profiles_parameter_pairs_without_writing_files(self):
        results = self._correlation_results()
        original = results.copy(deep=True)

        with tempfile.TemporaryDirectory(dir=".") as directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                with patch("matplotlib.pyplot.show"):
                    figure = read_results.plot_corr(
                        results,
                        {
                            "lr": [0.1, 0.2, 0.3],
                            "TA": [30_000, 32_000, 34_000],
                            "gA": [3.8, 4.0, 4.2],
                        },
                        chi2col="chi2_tot",
                        grid_size=30,
                    )
                    figure.canvas.draw()
                    output_files = list(Path.cwd().iterdir())
            finally:
                os.chdir(previous_directory)

        visible_axes = [axis for axis in figure.axes if axis.get_visible()]
        self.assertEqual(len(visible_axes), 4)
        self.assertEqual(output_files, [])
        pd.testing.assert_frame_equal(results, original)

    def test_plot_corner_combines_diagonal_and_pair_profiles(self):
        results = self._correlation_results()
        original = results.copy(deep=True)
        parameters = {
            "lr": [0.1, 0.2, 0.3],
            "TA": [30_000, 32_000, 34_000],
            "gA": [3.8, 4.0, 4.2],
        }

        with tempfile.TemporaryDirectory(dir=".") as directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                with patch("matplotlib.pyplot.show"):
                    figure = read_results.plot_corner(
                        results,
                        parameters,
                        chi2col="chi2_tot",
                        interp="pchip",
                        grid_size=30,
                        parameter_limits={"lr": (0.1, 0.2)},
                        reference_values={
                            "lr": 0.15,
                            "TA": 32_000,
                            "gA": 4.0,
                        },
                        contour_mode="rank",
                    )
                    figure.canvas.draw()
                    output_files = list(Path.cwd().iterdir())
            finally:
                os.chdir(previous_directory)

        visible_axes = [axis for axis in figure.axes if axis.get_visible()]
        profile_lines = [
            line
            for axis in visible_axes
            for line in axis.lines
            if line.get_label() == "Profile minima"
        ]
        diagonal_axes = [figure.axes[index * 3 + index] for index in range(3)]
        reference_lines = [
            line
            for axis in visible_axes
            for line in axis.lines
            if line.get_color() == "#d84a3a" and line.get_linestyle() == ":"
        ]
        self.assertEqual(len(visible_axes), 6)
        self.assertEqual(len(profile_lines), 3)
        self.assertGreater(len(reference_lines), 3)
        self.assertIn(
            "Best 10% of grid",
            [text.get_text() for text in figure.legends[0].get_texts()],
        )
        self.assertTrue(
            all("^{+" in axis.get_title() and "}_{-" in axis.get_title()
                for axis in diagonal_axes)
        )
        self.assertTrue(diagonal_axes[0].get_title().startswith("$f_B ="))
        self.assertTrue(
            all(axis.title.get_fontsize() >= 22 for axis in diagonal_axes)
        )
        self.assertGreater(
            diagonal_axes[0].yaxis.get_offset_text().get_position()[1],
            1,
        )
        np.testing.assert_allclose(diagonal_axes[0].get_xlim(), [0.1, 0.2])
        np.testing.assert_allclose(figure.axes[3].get_xlim(), [0.1, 0.2])
        self.assertEqual(figure.axes[6].get_xlabel(), r"$f_B$")
        self.assertEqual(
            diagonal_axes[0].get_ylabel(),
            "Scaled score above best fit",
        )
        np.testing.assert_allclose(
            read_results._profile_confidence_levels(1),
            [1.0, 4.0, 9.0],
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            read_results._profile_confidence_levels(2),
            [2.2957, 6.1801, 11.8290],
            rtol=2e-4,
        )
        self.assertEqual(output_files, [])
        pd.testing.assert_frame_equal(results, original)


if __name__ == "__main__":
    unittest.main()
