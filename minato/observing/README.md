# Observing planning

`minato.observing` provides reusable phase-scheduling and night-visibility
tools built from the useful parts of the archived top-level `Observing/`
directory.

## Phase windows

Generate evenly spaced orbital-phase windows without site calculations:

```python
from minato.observing import generate_phase_windows

windows = generate_phase_windows(
    first_time="2026-07-20 20:00",
    period_days=3.2,
    num_epochs=8,
    phase_tolerance=0.1,
    max_time="2026-08-20 20:00",
)
```

Use `compute_phases(...)` to add Astroplan altitude, airmass, and twilight
constraints. It returns a pandas DataFrame. CSV output is opt-in and refuses to
overwrite existing files unless `overwrite=True`.

## Night visibility

```python
from astropy.coordinates import EarthLocation
from minato.observing import plot_night_visibility

location = EarthLocation.from_geodetic(-17.89, 28.76, 2300)
figure, axes, coordinates = plot_night_visibility(
    "Example target",
    "18h09m17.69s",
    "-23d59m18.23s",
    "2026-07-20 12:00",
    location,
)
```

The plotting function returns Matplotlib objects and does not call
`plt.show()`, so callers control display and output paths. `NVTC` remains as a
compatibility alias for the archived function name.
