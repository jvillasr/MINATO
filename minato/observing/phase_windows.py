"""Plan phase-resolved observing windows and assess target visibility."""

from pathlib import Path
import re

import astropy.units as u
import pandas as pd
from astroplan import (
    AirmassConstraint,
    AltitudeConstraint,
    AtNightConstraint,
    FixedTarget,
    is_observable,
)
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ._utils import resolve_observer


def generate_phase_windows(
    first_time,
    period_days,
    num_epochs,
    phase_tolerance,
    max_time,
):
    """Return evenly spaced orbital-phase windows as a pandas DataFrame."""
    period_days = float(period_days)
    num_epochs = int(num_epochs)
    phase_tolerance = float(phase_tolerance)
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if phase_tolerance < 0:
        raise ValueError("phase_tolerance must be non-negative")

    first_time = _as_utc_time(first_time)
    max_time = _as_utc_time(max_time)
    if max_time <= first_time:
        raise ValueError("max_time must be later than first_time")

    phase_step_days = period_days / num_epochs
    tolerance_days = phase_step_days * phase_tolerance
    rows = []
    for phase_index in range(num_epochs):
        nominal = first_time + phase_index * phase_step_days * u.day
        cycle = 0
        while nominal < max_time:
            window_start = nominal - tolerance_days * u.day
            window_end = nominal + tolerance_days * u.day
            rows.append(
                {
                    "phase_index": phase_index + 1,
                    "phase": phase_index / num_epochs,
                    "cycle": cycle,
                    "nominal_mjd": float(nominal.mjd),
                    "nominal_utc": nominal.utc.isot,
                    "window_start_mjd": float(window_start.mjd),
                    "window_end_mjd": float(window_end.mjd),
                }
            )
            cycle += 1
            nominal = nominal + period_days * u.day
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["nominal_mjd", "phase_index", "cycle"], ignore_index=True)
    )


def assess_observability(
    windows,
    location,
    ra,
    dec,
    name="Target",
    *,
    twilight="nautical",
    altitude=(10.0, 90.0),
    airmass_max=2.5,
    time_grid_resolution_min=15.0,
):
    """Add an ``observable`` column using Astroplan constraints."""
    observer = resolve_observer(location)
    target = FixedTarget(
        SkyCoord(ra, dec, unit=(u.hourangle, u.deg)),
        name=str(name),
    )
    constraints = [
        AltitudeConstraint(float(altitude[0]) * u.deg, float(altitude[1]) * u.deg),
        AirmassConstraint(float(airmass_max)),
        _twilight_constraint(twilight),
    ]

    result = windows.copy()
    observable = []
    for row in result.itertuples(index=False):
        time_range = Time(
            [row.window_start_mjd, row.window_end_mjd],
            format="mjd",
            scale="utc",
        )
        value = is_observable(
            constraints,
            observer,
            target,
            time_range=time_range,
            time_grid_resolution=float(time_grid_resolution_min) * u.minute,
        )
        observable.append(bool(value[0]))
    result["observable"] = observable
    return result


def compute_phases(
    time,
    location,
    period,
    num_epochs,
    phase_tolerance,
    max_date,
    ra,
    dec,
    name,
    twilight="nautical",
    alt_min=10,
    alt_max=90,
    airmass_max=2.5,
    save_table=False,
    *,
    output_path=None,
    overwrite=False,
    print_results=True,
):
    """Generate and assess phase windows using the archived convenience API.

    Set ``save_table=True`` to use ``<name>_phase_windows.csv``, pass a path as
    ``save_table``, or provide ``output_path`` explicitly. Existing files are
    protected unless ``overwrite=True``.
    """
    windows = generate_phase_windows(
        time,
        period,
        num_epochs,
        phase_tolerance,
        max_date,
    )
    result = assess_observability(
        windows,
        location,
        ra,
        dec,
        name,
        twilight=twilight,
        altitude=(alt_min, alt_max),
        airmass_max=airmass_max,
    )
    if print_results:
        _print_schedule(result)

    destination = _output_path(name, save_table, output_path)
    if destination is not None:
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    return result


def _as_utc_time(value):
    if isinstance(value, Time):
        return value.utc
    return Time(value, scale="utc")


def _twilight_constraint(twilight):
    constraints = {
        "civil": AtNightConstraint.twilight_civil,
        "nautical": AtNightConstraint.twilight_nautical,
        "astronomical": AtNightConstraint.twilight_astronomical,
    }
    try:
        return constraints[str(twilight).lower()]()
    except KeyError as exc:
        choices = ", ".join(sorted(constraints))
        raise ValueError(f"twilight must be one of: {choices}") from exc


def _output_path(name, save_table, output_path):
    if output_path is not None:
        return Path(output_path)
    if save_table is False or save_table is None:
        return None
    if save_table is True:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "target"
        return Path(f"{slug}_phase_windows.csv")
    return Path(save_table)


def _print_schedule(result):
    for row in result.itertuples(index=False):
        status = "observable" if row.observable else "not observable"
        print(
            f"OB{row.phase_index:02d} cycle {row.cycle:02d}: "
            f"{row.nominal_utc} UTC ({status})"
        )
