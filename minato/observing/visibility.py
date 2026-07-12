"""Night-visibility plotting for one or more targets."""

from datetime import timezone

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, SkyCoord, get_body
from astropy.time import Time
from matplotlib import dates

from ._utils import resolve_observer


def plot_night_visibility(
    names,
    ra,
    dec,
    time,
    location,
    *,
    n_samples=100,
    show_moon=True,
):
    """Plot target airmass through the civil-twilight night.

    ``location`` may be an Astroplan ``Observer``, an Astropy
    ``EarthLocation``, or an Astropy site-registry name. The function returns
    ``(figure, axes, coordinates)`` and never calls ``plt.show``.
    """
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    target_names, target_ra, target_dec = _normalise_targets(names, ra, dec)
    observer = resolve_observer(location)
    reference_time = Time(time, scale="utc", location=observer.location)

    civil_pm = observer.twilight_evening_civil(reference_time, which="nearest")
    civil_am = observer.twilight_morning_civil(civil_pm, which="next")
    nautical_pm = observer.twilight_evening_nautical(civil_pm, which="nearest")
    nautical_am = observer.twilight_morning_nautical(civil_pm, which="next")
    astronomical_pm = observer.twilight_evening_astronomical(
        civil_pm, which="nearest"
    )
    astronomical_am = observer.twilight_morning_astronomical(civil_pm, which="next")

    observe_time = civil_pm + (civil_am - civil_pm) * np.linspace(
        0.0,
        1.0,
        n_samples,
    )
    frame = AltAz(obstime=observe_time, location=observer.location)
    plot_times = observe_time.to_datetime(timezone=timezone.utc)

    figure, axes = plt.subplots(figsize=(8, 5))
    coordinates = {}
    for target_name, target_ra_value, target_dec_value in zip(
        target_names,
        target_ra,
        target_dec,
    ):
        target = SkyCoord(
            ra=target_ra_value,
            dec=target_dec_value,
            unit=(u.hourangle, u.deg),
        )
        target_altaz = target.transform_to(frame)
        airmass = _visible_airmass(target_altaz)
        coordinates[target_name] = target_altaz
        axes.plot(plot_times, airmass, label=target_name, lw=2)

    if show_moon:
        moon_altaz = get_body("moon", observe_time, location=observer.location).transform_to(
            frame
        )
        axes.plot(
            plot_times,
            _visible_airmass(moon_altaz),
            color="0.55",
            label="Moon",
            lw=1.5,
        )

    for evening, morning, alpha in (
        (civil_pm, civil_am, 0.08),
        (nautical_pm, nautical_am, 0.08),
        (astronomical_pm, astronomical_am, 0.08),
    ):
        axes.axvspan(
            evening.to_datetime(timezone=timezone.utc),
            morning.to_datetime(timezone=timezone.utc),
            color="black",
            alpha=alpha,
            zorder=0,
        )

    axes.set_xlim(plot_times[0], plot_times[-1])
    axes.set_ylim(3.0, 1.0)
    axes.set_xlabel("Time (UTC)")
    axes.set_ylabel("Airmass")
    axes.grid(alpha=0.3)
    axes.legend()
    axes.xaxis.set_major_locator(dates.HourLocator(interval=1))
    axes.xaxis.set_major_formatter(dates.DateFormatter("%Hh", tz=timezone.utc))
    figure.tight_layout()
    return figure, axes, coordinates


def NVTC(name, RA, DEC, time0, location):
    """Compatibility alias for the archived night-visibility calculator."""
    return plot_night_visibility(name, RA, DEC, time0, location)


def _visible_airmass(altaz):
    values = np.asarray(altaz.secz.value, dtype=float)
    return np.where((altaz.alt.deg > 0.0) & (values >= 1.0), values, np.nan)


def _normalise_targets(names, ra, dec):
    target_ra = _as_list(ra)
    target_dec = _as_list(dec)
    if len(target_ra) != len(target_dec):
        raise ValueError("ra and dec must describe the same number of targets")
    if isinstance(names, str):
        target_names = [names] * len(target_ra)
    else:
        target_names = list(names)
    if len(target_names) != len(target_ra):
        raise ValueError("names, ra, and dec must describe the same number of targets")
    return target_names, target_ra, target_dec


def _as_list(values):
    if isinstance(values, str) or np.isscalar(values):
        return [values]
    return list(values)
