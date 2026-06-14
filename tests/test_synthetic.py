import unittest

import numpy as np

from minato.synthetic import (
    BinarySystem,
    ObservationModel,
    Spectrum,
    Star,
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


if __name__ == "__main__":
    unittest.main()
