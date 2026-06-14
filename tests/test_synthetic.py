import unittest
import tempfile
from pathlib import Path

import numpy as np

from minato.synthetic import (
    BinarySystem,
    FallbackAtmosphereGrid,
    ObservationModel,
    Spectrum,
    Star,
    TextAtmosphereGrid,
    doppler_shift,
    render_binary,
    render_single_star,
)


class ToyAtmosphereGrid:
    def __init__(self, wavelength, flux_by_label):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.flux_by_label = flux_by_label

    def get_spectrum(self, star):
        flux = self.flux_by_label.get(star.label, self.flux_by_label["default"])
        return Spectrum(self.wavelength, flux)


def gaussian_absorption(wavelength, centre, depth=0.4, width=0.25):
    return 1.0 - depth * np.exp(-0.5 * ((wavelength - centre) / width) ** 2)


class SyntheticRenderingTests(unittest.TestCase):
    def setUp(self):
        self.wavelength = np.linspace(4995.0, 5005.0, 1001)
        self.primary_flux = gaussian_absorption(self.wavelength, 5000.0)
        self.secondary_flux = 1.0 - 0.2 * np.exp(-0.5 * ((self.wavelength - 5001.0) / 0.35) ** 2)
        self.grid = ToyAtmosphereGrid(
            self.wavelength,
            {
                "default": self.primary_flux,
                "primary": self.primary_flux,
                "secondary": self.secondary_flux,
            },
        )

    def test_no_rv_shift_preserves_single_star_spectrum(self):
        star = Star(teff=25_000, logg=4.0, radius=8.0, label="default")
        observation = ObservationModel(velocity_step=None)

        spectrum = render_single_star(star, atmosphere_grid=self.grid, observation=observation)

        np.testing.assert_allclose(spectrum.wavelength, self.wavelength)
        np.testing.assert_allclose(spectrum.flux, self.primary_flux)

    def test_positive_doppler_shift_moves_feature_redward(self):
        shifted = doppler_shift(self.wavelength, self.primary_flux, rv=120.0)

        original_centre = self.wavelength[np.argmin(self.primary_flux)]
        shifted_centre = self.wavelength[np.argmin(shifted)]

        self.assertGreater(shifted_centre, original_centre)

    def test_binary_composite_flux_equals_scaled_component_sum(self):
        system = BinarySystem(
            primary=Star(teff=30_000, logg=4.1, radius=2.0, label="primary"),
            secondary=Star(teff=18_000, logg=4.2, radius=1.0, label="secondary"),
        )
        observation = ObservationModel(velocity_step=None)

        spectrum = render_binary(system, atmosphere_grid=self.grid, observation=observation)

        expected = (4.0 * self.primary_flux + 1.0 * self.secondary_flux) / 5.0
        np.testing.assert_allclose(spectrum.flux, expected)
        self.assertAlmostEqual(spectrum.metadata["primary_weight"], 0.8)
        self.assertAlmostEqual(spectrum.metadata["secondary_weight"], 0.2)

    def test_noise_is_reproducible_with_seed(self):
        star = Star(teff=25_000, logg=4.0, radius=8.0, label="default")
        observation = ObservationModel(velocity_step=None, snr=50, seed=42)

        first = render_single_star(star, atmosphere_grid=self.grid, observation=observation)
        second = render_single_star(star, atmosphere_grid=self.grid, observation=observation)
        different_seed = render_single_star(
            star,
            atmosphere_grid=self.grid,
            observation=ObservationModel(velocity_step=None, snr=50, seed=43),
        )

        np.testing.assert_allclose(first.flux, second.flux)
        np.testing.assert_allclose(first.error, second.error)
        self.assertFalse(np.allclose(first.flux, different_seed.flux))


class TextAtmosphereGridTests(unittest.TestCase):
    def _write_model(self, directory, name, offset=0.0):
        path = Path(directory) / name
        wavelength = np.array([4999.0, 5000.0, 5001.0])
        flux = np.array([1.0, 0.8 + offset, 1.0])
        np.savetxt(path, np.column_stack([wavelength, flux]))
        return path

    def _write_flux(self, directory, name, flux):
        path = Path(directory) / name
        wavelength = np.array([4999.0, 5000.0, 5001.0])
        np.savetxt(path, np.column_stack([wavelength, np.asarray(flux, dtype=float)]))
        return path

    def test_from_directory_recognises_minato_names(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self._write_model(directory, "teff25000_logg4.00.txt")
            self._write_model(directory, "teff30000_logg4.25.txt", offset=0.1)

            grid = TextAtmosphereGrid.from_directory(directory)
            spectrum = grid.get_spectrum(Star(teff=24_900, logg=4.02))

            self.assertEqual(len(grid.nodes), 2)
            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["teff"], 25_000.0)
            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["logg"], 4.0)
            np.testing.assert_allclose(spectrum.flux, [1.0, 0.8, 1.0])

    def test_known_format_recognisers_parse_common_names(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self._write_model(directory, "gal-ob-vd3_25-40_line.txt")
            self._write_model(directory, "BG30000g425v2.flux")
            self._write_model(directory, "T35000_g4.50.dat")

            powr = TextAtmosphereGrid.from_directory(directory, format="powr")
            tlusty = TextAtmosphereGrid.from_directory(directory, format="tlusty")
            fastwind = TextAtmosphereGrid.from_directory(directory, format="fastwind")

            self.assertAlmostEqual(powr.nodes[0].teff, 25_000.0)
            self.assertAlmostEqual(powr.nodes[0].logg, 4.0)
            self.assertAlmostEqual(tlusty.nodes[0].teff, 30_000.0)
            self.assertAlmostEqual(tlusty.nodes[0].logg, 4.25)
            self.assertAlmostEqual(fastwind.nodes[0].teff, 35_000.0)
            self.assertAlmostEqual(fastwind.nodes[0].logg, 4.5)

    def test_auto_powr_directory_uses_log_flux_auto_conversion(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self._write_flux(directory, "gal-ob-vd3_25-40_line.txt", [-1.0, -0.2, -1.0])

            grid = TextAtmosphereGrid.from_directory(directory)
            spectrum = grid.get_spectrum(Star(teff=25_000, logg=4.0))

            np.testing.assert_allclose(spectrum.flux, 10.0 ** np.array([-1.0, -0.2, -1.0]))
            self.assertEqual(spectrum.metadata["flux_transform"], "10**flux")

    def test_filename_pattern_handles_unconventional_names(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self._write_model(directory, "model_T22k_grav375.txt")

            grid = TextAtmosphereGrid.from_directory(
                directory,
                filename_pattern=r"T(?P<teff_kk>\d+)k_grav(?P<logg100>\d+)",
            )
            spectrum = grid.get_spectrum(Star(teff=22_000, logg=3.75))

            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["teff"], 22_000.0)
            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["logg"], 3.75)

    def test_custom_parser_handles_unconventional_names(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self._write_model(directory, "odd_model_A.txt")

            def parser(path):
                if path.name == "odd_model_A.txt":
                    return {"teff": 22_000, "logg": 3.75, "label": "manual"}
                return None

            grid = TextAtmosphereGrid.from_directory(directory, parser=parser)
            spectrum = grid.get_spectrum(Star(teff=22_100, logg=3.8))

            node = spectrum.metadata["atmosphere_node"]
            self.assertEqual(node["label"], "manual")
            self.assertAlmostEqual(node["teff"], 22_000.0)
            self.assertAlmostEqual(node["logg"], 3.75)

    def test_index_template_and_from_index_round_trip(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            self._write_model(directory, "unparsed_name.txt")
            index_path = Path(directory) / "model_index.csv"

            TextAtmosphereGrid.write_index_template(directory, index_path)
            text = index_path.read_text()
            text = text.replace("unparsed_name.txt,,", "unparsed_name.txt,28000,4.1")
            index_path.write_text(text)

            grid = TextAtmosphereGrid.from_index(index_path, root=directory)
            spectrum = grid.get_spectrum(Star(teff=28_100, logg=4.1))

            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["teff"], 28_000.0)
            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["logg"], 4.1)


class FallbackAtmosphereGridTests(unittest.TestCase):
    def _write_model(self, directory, name, depth):
        path = Path(directory) / name
        wavelength = np.array([4999.0, 5000.0, 5001.0])
        flux = np.array([1.0, 1.0 - depth, 1.0])
        np.savetxt(path, np.column_stack([wavelength, flux]))
        return path

    def test_fallback_uses_first_grid_when_node_is_acceptable(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            self._write_model(first_dir, "teff17000_logg4.00.txt", depth=0.1)
            self._write_model(second_dir, "teff17000_logg4.00.txt", depth=0.2)

            first = TextAtmosphereGrid.from_directory(first_dir, max_logg_delta=0.2)
            second = TextAtmosphereGrid.from_directory(second_dir, max_logg_delta=0.2)
            grid = FallbackAtmosphereGrid([("first", first), ("second", second)])
            spectrum = grid.get_spectrum(Star(teff=17_000, logg=4.05))

            self.assertEqual(spectrum.metadata["selected_grid"]["name"], "first")
            np.testing.assert_allclose(spectrum.flux, [1.0, 0.9, 1.0])

    def test_fallback_uses_later_grid_when_first_grid_misses_logg(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            powr_dir = Path(directory) / "powr"
            tlusty_dir = Path(directory) / "tlusty"
            powr_dir.mkdir()
            tlusty_dir.mkdir()
            self._write_model(powr_dir, "teff17000_logg3.20.txt", depth=0.1)
            self._write_model(tlusty_dir, "teff17000_logg4.20.txt", depth=0.2)

            powr = TextAtmosphereGrid.from_directory(powr_dir, max_logg_delta=0.25)
            tlusty = TextAtmosphereGrid.from_directory(tlusty_dir, max_logg_delta=0.25)
            grid = FallbackAtmosphereGrid([("powr", powr), ("tlusty", tlusty)])
            spectrum = grid.get_spectrum(Star(teff=17_000, logg=4.2))

            self.assertEqual(spectrum.metadata["selected_grid"]["name"], "tlusty")
            self.assertAlmostEqual(spectrum.metadata["atmosphere_node"]["logg"], 4.2)
            np.testing.assert_allclose(spectrum.flux, [1.0, 0.8, 1.0])

    def test_fallback_reports_all_lookup_failures(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            self._write_model(first_dir, "teff15000_logg3.00.txt", depth=0.1)
            self._write_model(second_dir, "teff25000_logg5.00.txt", depth=0.2)

            first = TextAtmosphereGrid.from_directory(first_dir, max_teff_delta=500)
            second = TextAtmosphereGrid.from_directory(second_dir, max_teff_delta=500)
            grid = FallbackAtmosphereGrid([("first", first), ("second", second)])

            with self.assertRaisesRegex(LookupError, "first: .*second:"):
                grid.get_spectrum(Star(teff=20_000, logg=4.0))


if __name__ == "__main__":
    unittest.main()
