"""Generate the synthetic binary inputs used by the SPAN tutorial.

This development-only script expects four user-downloaded PoWR spectra: the
normalised and calibrated line spectra for GAL-OB-Vd3 models 32-40 and 22-42.
It does not download or redistribute atmosphere models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from astropy.constants import G, M_sun
import astropy.units as u

from minato.synthetic.physics import (
    add_noise,
    apply_instrumental_broadening,
    apply_rotational_broadening,
    doppler_shift,
    make_log_wavelength_grid,
    resample_spectrum,
)


DEFAULT_LINES = (4009, 4026, 4102, 4121, 4144, 4233, 4267, 4340, 4388, 4471, 4553)
DEFAULT_DISENTANGLING_ITERATIONS = 500
LIGHT_RATIO_WINDOWS = (
    (3990.0, 4000.0),
    (4005.0, 4033.0),
    (4064.0, 4117.0),
    (4117.0, 4135.0),
    (4137.0, 4151.0),
    (4225.0, 4241.0),
    (4260.0, 4275.0),
    (4320.0, 4362.0),
    (4380.0, 4396.0),
    (4465.0, 4485.0),
    (4536.0, 4560.0),
)


def orbital_semi_amplitudes(
    primary_mass,
    secondary_mass,
    period_days,
    *,
    inclination_deg=90.0,
    eccentricity=0.0,
):
    """Return the primary and secondary RV semi-amplitudes in km/s."""
    if primary_mass <= 0 or secondary_mass <= 0:
        raise ValueError("stellar masses must be positive")
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    if not 0 <= eccentricity < 1:
        raise ValueError("eccentricity must be in the interval [0, 1)")

    period = float(period_days) * u.day
    mass_a = float(primary_mass) * M_sun
    mass_b = float(secondary_mass) * M_sun
    total_mass = mass_a + mass_b
    factor = (2.0 * np.pi * G / period) ** (1.0 / 3.0)
    projection = np.sin(np.deg2rad(float(inclination_deg))) / np.sqrt(
        1.0 - float(eccentricity) ** 2
    )
    k_a = (factor * mass_b / total_mass ** (2.0 / 3.0) * projection).to_value(
        u.km / u.s
    )
    k_b = (factor * mass_a / total_mass ** (2.0 / 3.0) * projection).to_value(
        u.km / u.s
    )
    return float(k_a), float(k_b)


def read_powr_spectrum(path, *, calibrated=False):
    """Read a two-column PoWR line spectrum."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"PoWR spectrum does not exist: {source}")
    data = np.loadtxt(source, comments="#", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"PoWR spectrum needs at least two columns: {source}")
    wavelength = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError(f"PoWR wavelengths must be strictly increasing: {source}")
    if calibrated:
        flux = 10.0**flux
    return wavelength, flux


def companion_light_fraction(
    wavelength,
    normalised_primary,
    normalised_secondary,
    calibrated_primary,
    calibrated_secondary,
):
    """Estimate the secondary continuum contribution over the SPAN windows."""
    floor = np.finfo(float).eps
    continuum_primary = calibrated_primary / np.clip(normalised_primary, floor, None)
    continuum_secondary = calibrated_secondary / np.clip(normalised_secondary, floor, None)
    fraction = continuum_secondary / (continuum_primary + continuum_secondary)
    selected = np.zeros_like(wavelength, dtype=bool)
    for lower, upper in LIGHT_RATIO_WINDOWS:
        selected |= (wavelength >= lower) & (wavelength <= upper)
    if not np.any(selected):
        raise ValueError("wavelength grid does not cover the SPAN light-ratio windows")
    light_fraction = float(np.nanmedian(fraction[selected]))
    if not 0 < light_fraction < 1:
        raise ValueError("calibrated PoWR spectra produced an invalid light fraction")
    return light_fraction


def sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_inputs(
    primary_normalised,
    secondary_normalised,
    primary_calibrated,
    secondary_calibrated,
    output_directory,
    *,
    primary_mass=19.04,
    secondary_mass=7.59,
    primary_teff=32_000.0,
    secondary_teff=22_000.0,
    primary_logg=4.0,
    secondary_logg=4.2,
    primary_vsini=80.0,
    secondary_vsini=120.0,
    period_days=17.0,
    inclination_deg=90.0,
    eccentricity=0.0,
    omega_deg=0.0,
    gamma=0.0,
    epoch_zero=60_000.0,
    number_of_epochs=10,
    resolving_power=40_000.0,
    snr=100.0,
    wavelength_min=3980.0,
    wavelength_max=4580.0,
    velocity_step=2.5,
    seed=20260713,
):
    """Generate ten composite spectra and their reproducibility metadata."""
    if number_of_epochs < 2:
        raise ValueError("number_of_epochs must be at least two")
    output = Path(output_directory).expanduser()
    output.mkdir(parents=True, exist_ok=False)
    spectra_directory = output / "spectra"
    spectra_directory.mkdir()

    input_paths = {
        "primary_normalised": Path(primary_normalised).expanduser(),
        "secondary_normalised": Path(secondary_normalised).expanduser(),
        "primary_calibrated": Path(primary_calibrated).expanduser(),
        "secondary_calibrated": Path(secondary_calibrated).expanduser(),
    }
    norm_a_wave, norm_a_flux = read_powr_spectrum(input_paths["primary_normalised"])
    norm_b_wave, norm_b_flux = read_powr_spectrum(input_paths["secondary_normalised"])
    cal_a_wave, cal_a_flux = read_powr_spectrum(
        input_paths["primary_calibrated"], calibrated=True
    )
    cal_b_wave, cal_b_flux = read_powr_spectrum(
        input_paths["secondary_calibrated"], calibrated=True
    )

    lower = max(
        float(wavelength_min),
        float(norm_a_wave[0]),
        float(norm_b_wave[0]),
        float(cal_a_wave[0]),
        float(cal_b_wave[0]),
    )
    upper = min(
        float(wavelength_max),
        float(norm_a_wave[-1]),
        float(norm_b_wave[-1]),
        float(cal_a_wave[-1]),
        float(cal_b_wave[-1]),
    )
    if lower >= upper:
        raise ValueError("requested wavelength range does not overlap all four PoWR spectra")
    wavelength = make_log_wavelength_grid(lower, upper, velocity_step)

    norm_a = resample_spectrum(norm_a_wave, norm_a_flux, wavelength)
    norm_b = resample_spectrum(norm_b_wave, norm_b_flux, wavelength)
    cal_a = resample_spectrum(cal_a_wave, cal_a_flux, wavelength)
    cal_b = resample_spectrum(cal_b_wave, cal_b_flux, wavelength)
    light_fraction_b = companion_light_fraction(wavelength, norm_a, norm_b, cal_a, cal_b)

    broad_a = apply_rotational_broadening(wavelength, norm_a, primary_vsini)
    broad_b = apply_rotational_broadening(wavelength, norm_b, secondary_vsini)
    broad_a = apply_instrumental_broadening(wavelength, broad_a, resolving_power)
    broad_b = apply_instrumental_broadening(wavelength, broad_b, resolving_power)

    k_a, k_b = orbital_semi_amplitudes(
        primary_mass,
        secondary_mass,
        period_days,
        inclination_deg=inclination_deg,
        eccentricity=eccentricity,
    )
    phases = np.arange(number_of_epochs, dtype=float) / float(number_of_epochs)
    epochs = float(epoch_zero) + phases * float(period_days)
    omega = np.deg2rad(float(omega_deg))
    orbital_term = np.cos(2.0 * np.pi * phases + omega) + eccentricity * np.cos(omega)
    primary_rv = float(gamma) + k_a * orbital_term
    secondary_rv = float(gamma) - k_b * orbital_term

    epoch_names = []
    velocity_rows = []
    for index, (epoch, phase, rv_a, rv_b) in enumerate(
        zip(epochs, phases, primary_rv, secondary_rv, strict=True)
    ):
        shifted_a = doppler_shift(wavelength, broad_a, rv_a)
        shifted_b = doppler_shift(wavelength, broad_b, rv_b)
        composite = (1.0 - light_fraction_b) * shifted_a + light_fraction_b * shifted_b
        noisy_flux, _ = add_noise(composite, snr, seed=seed + index)
        name = f"epoch_{index:02d}"
        epoch_names.append(name)
        np.savetxt(
            spectra_directory / f"{name}.txt",
            np.column_stack([wavelength, noisy_flux]),
            fmt="%.6f %.8f",
            header=(
                "wavelength_A normalised_flux\n"
                f"phase={phase:.6f} mjd={epoch:.6f} "
                f"rv_primary_kms={rv_a:.6f} rv_secondary_kms={rv_b:.6f} "
                f"snr={snr:g} seed={seed + index}"
            ),
        )
        velocity_rows.append((name, epoch, phase, rv_a, rv_b))

    epochs_path = output / "epochs.txt"
    with epochs_path.open("w") as handle:
        for name, epoch in zip(epoch_names, epochs, strict=True):
            handle.write(f"{name} {epoch:.8f}\n")

    orbital_path = output / "orbital_parameters.csv"
    orbital_path.write_text(
        "star_ID,P,T0,e,Omega,Gamma,K1,K2\n"
        f"span_synthetic,{period_days},{epoch_zero},{eccentricity},{omega_deg},"
        f"{gamma},{k_a:.8f},{k_b:.8f}\n"
    )
    np.savetxt(
        output / "velocities.csv",
        np.asarray([row[1:] for row in velocity_rows]),
        delimiter=",",
        fmt="%.8f",
        header="mjd,phase,rv_primary_kms,rv_secondary_kms",
        comments="",
    )

    provenance = {
        "generator": "scripts/generate_span_tutorial_data.py",
        "powr_grid": "GAL-OB-Vd3",
        "powr_grid_url": (
            "https://www.astro.physik.uni-potsdam.de/~wrh/PoWR/"
            "details_gal-ob-vink-models.php"
        ),
        "primary": {
            "model": "32-40",
            "teff_K": primary_teff,
            "logg": primary_logg,
            "mass_msun": primary_mass,
            "vsini_kms": primary_vsini,
        },
        "secondary": {
            "model": "22-42",
            "teff_K": secondary_teff,
            "logg": secondary_logg,
            "mass_msun": secondary_mass,
            "vsini_kms": secondary_vsini,
            "light_fraction": light_fraction_b,
        },
        "orbit": {
            "period_days": period_days,
            "inclination_deg": inclination_deg,
            "eccentricity": eccentricity,
            "omega_deg": omega_deg,
            "gamma_kms": gamma,
            "epoch_zero_mjd": epoch_zero,
            "primary_semi_amplitude_kms": k_a,
            "secondary_semi_amplitude_kms": k_b,
        },
        "observation": {
            "number_of_epochs": number_of_epochs,
            "resolving_power": resolving_power,
            "snr": snr,
            "wavelength_min_A": float(wavelength[0]),
            "wavelength_max_A": float(wavelength[-1]),
            "velocity_step_kms": velocity_step,
            "seed": seed,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def disentangle_inputs(
    work_directory,
    tutorial_output_directory,
    *,
    iterations=DEFAULT_DISENTANGLING_ITERATIONS,
):
    """Run the development-only shift-and-add adaptation and copy its outputs."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from minato.contrib.spdis import SpecDisent

    work = Path(work_directory).expanduser().resolve()
    tutorial_output = Path(tutorial_output_directory).expanduser().resolve()
    primary_target = tutorial_output / "span_primary_disentangled.txt"
    secondary_target = tutorial_output / "span_secondary_disentangled.txt"
    provenance_target = tutorial_output / "provenance.json"
    existing = [
        path
        for path in (primary_target, secondary_target, provenance_target)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "tutorial disentangled spectrum already exists: "
            + ", ".join(str(path) for path in existing)
        )

    provenance = json.loads((work / "provenance.json").read_text())
    spectra_directory = work / "spectra"
    disentangler = SpecDisent(
        list(DEFAULT_LINES),
        str(work / "orbital_parameters.csv"),
        str(work / "epochs.txt"),
        str(spectra_directory) + os.sep,
        extension=".txt",
    )
    primary_fraction = 1.0 - float(provenance["secondary"]["light_fraction"])
    disentangler.get_disspec(
        lguess1=primary_fraction,
        GridDis=False,
        PLOTCONV=False,
        PLOTITR=False,
        PLOTFITS=False,
        PLOTEXTREMES=False,
        NebOff=True,
        NumItrFinal=int(iterations),
    )

    generated = Path(disentangler.working_path)
    primary_source = next(generated.glob("ADIS_*.txt"))
    secondary_source = next(generated.glob("BDIS_*.txt"))
    tutorial_output.mkdir(parents=True, exist_ok=True)
    header = (
        "Synthetic disentangled spectrum for the MINATO SPAN tutorial.\n"
        "Generated from PoWR GAL-OB-Vd3 models 32-40 and 22-42; see provenance.json.\n"
        "wavelength_A normalised_flux"
    )
    for source, target in (
        (primary_source, primary_target),
        (secondary_source, secondary_target),
    ):
        data = np.loadtxt(source)
        np.savetxt(target, data[:, :2], fmt="%.6f %.8f", header=header)
    release_provenance = json.loads(json.dumps(provenance))
    for source in release_provenance["inputs"].values():
        source["path"] = Path(source["path"]).name
    release_provenance["disentangling"] = {
        "implementation": "minato.contrib.spdis development-only adaptation",
        "upstream": "https://github.com/TomerShenar/Disentangling_Shift_And_Add",
        "iterations": int(iterations),
        "lines_A": list(DEFAULT_LINES),
        "method_citations": [
            "Gonzalez & Levato (2006), A&A, 448, 283",
            "Shenar et al. (2020), A&A, 639, A6",
            "Shenar et al. (2022), A&A, 665, A148",
        ],
    }
    provenance_target.write_text(json.dumps(release_provenance, indent=2) + "\n")
    return primary_target, secondary_target


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-normalised", type=Path, required=True)
    parser.add_argument("--secondary-normalised", type=Path, required=True)
    parser.add_argument("--primary-calibrated", type=Path, required=True)
    parser.add_argument("--secondary-calibrated", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--tutorial-output-directory", type=Path)
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_DISENTANGLING_ITERATIONS,
        help=(
            "Number of shift-and-add iterations "
            f"(default: {DEFAULT_DISENTANGLING_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--skip-disentangling",
        action="store_true",
        help="Generate the ten epochs without running minato.contrib.spdis.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generate_inputs(
        args.primary_normalised,
        args.secondary_normalised,
        args.primary_calibrated,
        args.secondary_calibrated,
        args.work_directory,
    )
    if not args.skip_disentangling:
        if args.tutorial_output_directory is None:
            raise SystemExit(
                "--tutorial-output-directory is required unless --skip-disentangling is used"
            )
        disentangle_inputs(
            args.work_directory,
            args.tutorial_output_directory,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
