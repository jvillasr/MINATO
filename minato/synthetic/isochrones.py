"""Optional isochrone helpers for synthetic-spectrum inputs."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .models import IsochronePoint


class IsochroneBank:
    """
    Load a simple CSV bank of isochrones and interpolate by mass and log age.

    Files are discovered as ``*logage*.csv`` and must include the columns
    ``mass_init``, ``teff``, ``logg``, and ``radius``. This matches the useful
    subset of the SDSS/MIST bank format without making MIST a hard dependency.
    """

    required_columns = ("mass_init", "teff", "logg", "radius")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"isochrone bank does not exist: {self.path}")

        ages: list[float] = []
        tables: list[dict[str, np.ndarray]] = []
        for filename in sorted(self.path.glob("*logage*.csv")):
            match = re.search(r"logage(\d+(?:\.\d+)?)", filename.stem)
            if match is None:
                continue
            table = np.genfromtxt(filename, delimiter=",", names=True)
            names = table.dtype.names or ()
            missing = [name for name in self.required_columns if name not in names]
            if missing:
                raise ValueError(f"{filename} is missing required columns: {missing}")
            data = {
                name: np.atleast_1d(np.asarray(table[name], dtype=float))
                for name in self.required_columns
            }
            order = np.argsort(data["mass_init"])
            for name in data:
                data[name] = data[name][order]
            ages.append(float(match.group(1)))
            tables.append(data)

        if not ages:
            raise ValueError(f"no isochrone CSV files found in {self.path}")

        order = np.argsort(ages)
        self.ages = np.asarray(ages, dtype=float)[order]
        self.tables = [tables[index] for index in order]

    def _interpolate_table(self, table: dict[str, np.ndarray], mass: float) -> IsochronePoint:
        masses = table["mass_init"]
        if mass < masses[0] or mass > masses[-1]:
            raise ValueError(
                f"mass {mass} is outside this isochrone range "
                f"({masses[0]}-{masses[-1]})"
            )
        return IsochronePoint(
            teff=float(np.interp(mass, masses, table["teff"])),
            logg=float(np.interp(mass, masses, table["logg"])),
            radius=float(np.interp(mass, masses, table["radius"])),
        )

    def interpolate(self, mass: float, log_age: float) -> IsochronePoint:
        """Return linearly interpolated parameters at ``mass`` and ``log_age``."""

        mass = float(mass)
        log_age = float(log_age)
        if log_age < self.ages[0] or log_age > self.ages[-1]:
            raise ValueError(
                f"log_age {log_age} is outside the bank range "
                f"({self.ages[0]}-{self.ages[-1]})"
            )

        upper = int(np.searchsorted(self.ages, log_age, side="left"))
        if upper < len(self.ages) and self.ages[upper] == log_age:
            return self._interpolate_table(self.tables[upper], mass)
        lower = max(upper - 1, 0)
        upper = min(upper, len(self.ages) - 1)
        if lower == upper:
            return self._interpolate_table(self.tables[lower], mass)

        point_lower = self._interpolate_table(self.tables[lower], mass)
        point_upper = self._interpolate_table(self.tables[upper], mass)
        age_lower = self.ages[lower]
        age_upper = self.ages[upper]
        weight = (log_age - age_lower) / (age_upper - age_lower)

        return IsochronePoint(
            teff=(1.0 - weight) * point_lower.teff + weight * point_upper.teff,
            logg=(1.0 - weight) * point_lower.logg + weight * point_upper.logg,
            radius=(1.0 - weight) * point_lower.radius + weight * point_upper.radius,
        )
