import tempfile
import unittest
from pathlib import Path

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.utils import iers

from minato.observing import (
    compute_phases,
    generate_phase_windows,
    plot_night_visibility,
)


class PhaseWindowTests(unittest.TestCase):
    def test_visibility_rejects_too_few_samples(self):
        with self.assertRaisesRegex(ValueError, "n_samples must be at least 2"):
            plot_night_visibility(
                "Example target",
                "18h09m17.69s",
                "-23d59m18.23s",
                "2026-07-20 20:00",
                EarthLocation.from_geodetic(-17.89 * u.deg, 28.76 * u.deg),
                n_samples=1,
            )

    def test_generate_phase_windows_has_expected_spacing(self):
        windows = generate_phase_windows(
            "2026-01-01 00:00",
            period_days=4.0,
            num_epochs=4,
            phase_tolerance=0.1,
            max_time="2026-01-05 00:00",
        )

        self.assertEqual(len(windows), 4)
        np_spacing = windows["nominal_mjd"].diff().dropna().to_numpy()
        self.assertTrue((abs(np_spacing - 1.0) < 1e-12).all())
        self.assertEqual(windows["phase"].tolist(), [0.0, 0.25, 0.5, 0.75])

    def test_compute_phases_returns_observability_and_protects_output(self):
        location = EarthLocation.from_geodetic(
            lon=-17.89 * u.deg,
            lat=28.76 * u.deg,
            height=2300 * u.m,
        )
        with (
            iers.conf.set_temp("auto_download", False),
            iers.conf.set_temp("auto_max_age", None),
            tempfile.TemporaryDirectory(dir=".") as directory,
        ):
            output = Path(directory) / "windows.csv"
            result = compute_phases(
                "2026-07-20 20:00",
                location,
                2.0,
                2,
                0.1,
                "2026-07-21 20:00",
                "18h09m17.69s",
                "-23d59m18.23s",
                "Example target",
                output_path=output,
                print_results=False,
            )

            self.assertIn("observable", result.columns)
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                compute_phases(
                    "2026-07-20 20:00",
                    location,
                    2.0,
                    2,
                    0.1,
                    "2026-07-21 20:00",
                    "18h09m17.69s",
                    "-23d59m18.23s",
                    "Example target",
                    output_path=output,
                    print_results=False,
                )


if __name__ == "__main__":
    unittest.main()
