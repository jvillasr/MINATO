"""Isochrone age-selection helpers for synthetic-spectrum inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .models import IsochronePoint, IsochroneProvider


PointConstraint = Callable[[IsochronePoint], bool]
AgeConstraint = Callable[["IsochroneAgeCandidate"], bool]
AgeWeight = Callable[["IsochroneAgeCandidate"], float]


class IsochroneAgeSamplingError(ValueError):
    """Raised when no configured isochrone age can be selected."""


@dataclass(frozen=True)
class StellarConstraints:
    """Bounds for accepting one interpolated isochrone point."""

    teff_min: float | None = None
    teff_max: float | None = None
    logg_min: float | None = None
    logg_max: float | None = None
    radius_min: float | None = None
    radius_max: float | None = None
    predicate: PointConstraint | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_bounds("teff", self.teff_min, self.teff_max)
        _validate_bounds("logg", self.logg_min, self.logg_max)
        _validate_bounds("radius", self.radius_min, self.radius_max)

    def matches(self, point: IsochronePoint) -> bool:
        """Return whether ``point`` satisfies all configured bounds."""

        if self.teff_min is not None and point.teff < self.teff_min:
            return False
        if self.teff_max is not None and point.teff > self.teff_max:
            return False
        if self.logg_min is not None and point.logg < self.logg_min:
            return False
        if self.logg_max is not None and point.logg > self.logg_max:
            return False
        if self.radius_min is not None and point.radius < self.radius_min:
            return False
        if self.radius_max is not None and point.radius > self.radius_max:
            return False
        if self.predicate is not None and not bool(self.predicate(point)):
            return False
        return True

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the configured constraints."""

        description: dict[str, Any] = {}
        for name in (
            "teff_min",
            "teff_max",
            "logg_min",
            "logg_max",
            "radius_min",
            "radius_max",
        ):
            value = getattr(self, name)
            if value is not None:
                description[name] = float(value)
        if self.predicate is not None:
            description["predicate"] = _callable_name(self.predicate)
        return description

    @property
    def is_empty(self) -> bool:
        """Return whether no bound or predicate is configured."""

        return not self.describe()


@dataclass(frozen=True)
class IsochroneAgeCandidate:
    """One coeval isochrone-age candidate for a primary and optional secondary."""

    log_age: float
    m1: float
    primary: IsochronePoint
    m2: float | None = None
    secondary: IsochronePoint | None = None


@dataclass(frozen=True)
class LoggSkewWeight:
    """Asymmetric Gaussian weight in ``logg`` for age sampling."""

    mu: float = 4.0
    sigma_lo: float = 0.25
    sigma_hi: float = 0.12
    component: str = "primary"

    def __post_init__(self) -> None:
        if self.sigma_lo <= 0:
            raise ValueError("sigma_lo must be positive")
        if self.sigma_hi <= 0:
            raise ValueError("sigma_hi must be positive")
        if self.component not in {"primary", "secondary", "both"}:
            raise ValueError("component must be 'primary', 'secondary', or 'both'")

    def __call__(self, candidate: IsochroneAgeCandidate) -> float:
        points = _candidate_points(candidate, self.component)
        weight = 1.0
        for point in points:
            delta = point.logg - self.mu
            sigma = self.sigma_lo if delta < 0 else self.sigma_hi
            weight *= float(np.exp(-0.5 * (delta / sigma) ** 2))
        return weight

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the weight."""

        return {
            "type": type(self).__name__,
            "mu": float(self.mu),
            "sigma_lo": float(self.sigma_lo),
            "sigma_hi": float(self.sigma_hi),
            "component": self.component,
        }


@dataclass(frozen=True)
class _LowMassTeffLimit:
    mass_max: float
    teff_max: float

    def __post_init__(self) -> None:
        if self.mass_max <= 0:
            raise ValueError("secondary_low_mass_teff_max['mass_max'] must be positive")
        if self.teff_max <= 0:
            raise ValueError("secondary_low_mass_teff_max['teff_max'] must be positive")

    def matches(self, mass: float, point: IsochronePoint) -> bool:
        if mass <= self.mass_max and point.teff > self.teff_max:
            return False
        return True

    def describe(self) -> dict[str, float]:
        return {"mass_max": float(self.mass_max), "teff_max": float(self.teff_max)}


class IsochroneAgeSampler:
    """
    Select a coeval log-age for one star or binary before spectrum rendering.

    With no constraints and no weight, the sampler draws uniformly from the
    isochrone bank ages where all requested masses can be interpolated. Passing
    ``fixed_log_age`` validates and returns one explicit age, reproducing direct
    ``Star.from_mass(..., log_age=...)`` behaviour without making sampling
    implicit in ``Star``.
    """

    def __init__(
        self,
        *,
        primary: StellarConstraints | PointConstraint | None = None,
        secondary: StellarConstraints | PointConstraint | None = None,
        secondary_low_mass_teff_max: Mapping[str, float] | None = None,
        require_primary_logg_lte_secondary: bool = False,
        weight: AgeWeight | None = None,
        fixed_log_age: float | None = None,
        constraint: AgeConstraint | None = None,
        on_failure: str = "raise",
    ) -> None:
        if on_failure not in {"raise", "return_none"}:
            raise ValueError("on_failure must be 'raise' or 'return_none'")
        self.primary = _normalise_stellar_constraints(primary)
        self.secondary = _normalise_stellar_constraints(secondary)
        self._secondary_configured = secondary is not None
        self.secondary_low_mass_teff_max = _normalise_low_mass_teff_limit(
            secondary_low_mass_teff_max
        )
        self.require_primary_logg_lte_secondary = bool(require_primary_logg_lte_secondary)
        self.weight = weight
        self.fixed_log_age = None if fixed_log_age is None else float(fixed_log_age)
        self.constraint = constraint
        self.on_failure = on_failure

    @classmethod
    def fixed(cls, log_age: float, **kwargs: Any) -> "IsochroneAgeSampler":
        """Return a sampler that validates and returns one explicit age."""

        return cls(fixed_log_age=log_age, **kwargs)

    def sample(
        self,
        isochrone_bank: IsochroneProvider,
        m1: float,
        m2: float | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[float | None, dict[str, Any]]:
        """
        Select one shared log-age and return ``(log_age, metadata)``.

        For sampled ages, ``isochrone_bank`` must expose an ``ages`` sequence in
        addition to the ``interpolate(mass, log_age)`` method. Fixed-age sampling
        only requires ``interpolate``.
        """

        m1 = float(m1)
        m2 = None if m2 is None else float(m2)
        if m1 <= 0:
            raise ValueError("m1 must be positive")
        if m2 is not None and m2 <= 0:
            raise ValueError("m2 must be positive")
        if m2 is None and self._uses_secondary_policy:
            raise ValueError("m2 is required when secondary constraints are configured")

        candidate_ages = self._candidate_ages(isochrone_bank)
        valid_candidates: list[IsochroneAgeCandidate] = []
        rejections: list[tuple[float, str]] = []

        for log_age in candidate_ages:
            candidate, rejection = self._candidate_at_age(isochrone_bank, m1, m2, log_age)
            if rejection is None and candidate is not None:
                rejection = self._rejection_reason(candidate)
            if rejection is None and candidate is not None:
                valid_candidates.append(candidate)
            else:
                rejections.append((float(log_age), str(rejection)))

        if not valid_candidates:
            metadata = self._failure_metadata(
                candidate_ages=candidate_ages,
                rejections=rejections,
            )
            if self.on_failure == "return_none":
                return None, metadata
            raise IsochroneAgeSamplingError(self._failure_message(m1, m2, metadata))

        if self.fixed_log_age is not None:
            selected = valid_candidates[0]
            weights = None
            selection_mode = "fixed"
        elif self.weight is None:
            generator = rng if rng is not None else np.random.default_rng()
            selected = valid_candidates[int(generator.choice(len(valid_candidates)))]
            weights = None
            selection_mode = "uniform"
        else:
            generator = rng if rng is not None else np.random.default_rng()
            weights = self._weight_values(valid_candidates)
            probabilities = weights / np.sum(weights)
            selected = valid_candidates[
                int(generator.choice(len(valid_candidates), p=probabilities))
            ]
            selection_mode = "weighted"

        metadata = self._metadata(
            selected=selected,
            candidate_ages=candidate_ages,
            valid_candidates=valid_candidates,
            rejections=rejections,
            weights=weights,
            selection_mode=selection_mode,
        )
        return float(selected.log_age), metadata

    @property
    def _uses_secondary_policy(self) -> bool:
        return (
            self._secondary_configured
            or self.secondary_low_mass_teff_max is not None
            or self.require_primary_logg_lte_secondary
        )

    def _candidate_ages(self, isochrone_bank: IsochroneProvider) -> np.ndarray:
        if self.fixed_log_age is not None:
            return np.asarray([self.fixed_log_age], dtype=float)
        if not hasattr(isochrone_bank, "ages"):
            raise TypeError(
                "isochrone_bank must expose an 'ages' sequence for sampled age selection"
            )
        candidate_ages = np.asarray(getattr(isochrone_bank, "ages"), dtype=float)
        if candidate_ages.ndim != 1 or candidate_ages.size == 0:
            raise ValueError("isochrone_bank.ages must be a non-empty one-dimensional sequence")
        return candidate_ages

    def _candidate_at_age(
        self,
        isochrone_bank: IsochroneProvider,
        m1: float,
        m2: float | None,
        log_age: float,
    ) -> tuple[IsochroneAgeCandidate | None, str | None]:
        try:
            primary = isochrone_bank.interpolate(mass=m1, log_age=float(log_age))
        except ValueError as exc:
            return None, f"primary interpolation failed: {exc}"
        secondary = None
        if m2 is not None:
            try:
                secondary = isochrone_bank.interpolate(mass=m2, log_age=float(log_age))
            except ValueError as exc:
                return None, f"secondary interpolation failed: {exc}"
        return (
            IsochroneAgeCandidate(
                log_age=float(log_age),
                m1=m1,
                m2=m2,
                primary=primary,
                secondary=secondary,
            ),
            None,
        )

    def _rejection_reason(self, candidate: IsochroneAgeCandidate) -> str | None:
        if not self.primary.matches(candidate.primary):
            return f"primary constraints failed: {_point_summary(candidate.primary)}"
        if self._secondary_configured and candidate.secondary is not None:
            if not self.secondary.matches(candidate.secondary):
                return f"secondary constraints failed: {_point_summary(candidate.secondary)}"
        if (
            self.secondary_low_mass_teff_max is not None
            and candidate.secondary is not None
            and candidate.m2 is not None
            and not self.secondary_low_mass_teff_max.matches(candidate.m2, candidate.secondary)
        ):
            return (
                "secondary low-mass teff limit failed: "
                f"m2={candidate.m2:g}, {_point_summary(candidate.secondary)}"
            )
        if (
            self.require_primary_logg_lte_secondary
            and candidate.secondary is not None
            and candidate.primary.logg > candidate.secondary.logg
        ):
            return (
                "primary logg exceeds secondary logg: "
                f"primary={candidate.primary.logg:g}, secondary={candidate.secondary.logg:g}"
            )
        if self.constraint is not None and not bool(self.constraint(candidate)):
            return f"age constraint failed: {_callable_name(self.constraint)}"
        return None

    def _weight_values(self, candidates: list[IsochroneAgeCandidate]) -> np.ndarray:
        weights = np.asarray([float(self.weight(candidate)) for candidate in candidates], dtype=float)
        if np.any(~np.isfinite(weights)):
            raise IsochroneAgeSamplingError("age weights must be finite")
        if np.any(weights < 0):
            raise IsochroneAgeSamplingError("age weights must be non-negative")
        if float(np.sum(weights)) <= 0:
            raise IsochroneAgeSamplingError("at least one valid age must have positive weight")
        return weights

    def _metadata(
        self,
        *,
        selected: IsochroneAgeCandidate,
        candidate_ages: np.ndarray,
        valid_candidates: list[IsochroneAgeCandidate],
        rejections: list[tuple[float, str]],
        weights: np.ndarray | None,
        selection_mode: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "selected_age": float(selected.log_age),
            "log_age": float(selected.log_age),
            "selection_mode": selection_mode,
            "candidate_age_count": int(candidate_ages.size),
            "valid_age_count": len(valid_candidates),
            "valid_log_ages": [float(candidate.log_age) for candidate in valid_candidates],
            "constraints": self.describe(),
            "selected_primary": _point_dict(selected.primary),
            "selected_secondary": (
                None if selected.secondary is None else _point_dict(selected.secondary)
            ),
            "rejection_counts": _rejection_counts(rejections),
        }
        if weights is not None:
            metadata["weights"] = {
                float(candidate.log_age): float(weight)
                for candidate, weight in zip(valid_candidates, weights)
            }
        return metadata

    def _failure_metadata(
        self,
        *,
        candidate_ages: np.ndarray,
        rejections: list[tuple[float, str]],
    ) -> dict[str, Any]:
        return {
            "selected_age": None,
            "log_age": None,
            "selection_mode": "fixed" if self.fixed_log_age is not None else "sampled",
            "candidate_age_count": int(candidate_ages.size),
            "valid_age_count": 0,
            "valid_log_ages": [],
            "constraints": self.describe(),
            "rejection_counts": _rejection_counts(rejections),
            "rejection_examples": [
                {"log_age": float(log_age), "reason": reason}
                for log_age, reason in rejections[:5]
            ],
        }

    def _failure_message(
        self,
        m1: float,
        m2: float | None,
        metadata: Mapping[str, Any],
    ) -> str:
        masses = f"m1={m1:g}" if m2 is None else f"m1={m1:g}, m2={m2:g}"
        examples = metadata.get("rejection_examples", [])
        example_text = "; ".join(
            f"log_age={item['log_age']}: {item['reason']}" for item in examples
        )
        if example_text:
            example_text = f" Examples: {example_text}"
        return (
            "No valid isochrone age found for "
            f"{masses} across {metadata['candidate_age_count']} candidate age(s). "
            f"Constraints: {metadata['constraints']}.{example_text}"
        )

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the sampler configuration."""

        description: dict[str, Any] = {
            "primary": self.primary.describe(),
            "secondary": self.secondary.describe() if self._secondary_configured else None,
            "secondary_low_mass_teff_max": (
                None
                if self.secondary_low_mass_teff_max is None
                else self.secondary_low_mass_teff_max.describe()
            ),
            "require_primary_logg_lte_secondary": self.require_primary_logg_lte_secondary,
            "fixed_log_age": self.fixed_log_age,
            "weight": _weight_description(self.weight),
            "constraint": None if self.constraint is None else _callable_name(self.constraint),
            "on_failure": self.on_failure,
        }
        return description


def _validate_bounds(name: str, lower: float | None, upper: float | None) -> None:
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"{name}_min must be <= {name}_max")


def _normalise_stellar_constraints(
    constraints: StellarConstraints | PointConstraint | None,
) -> StellarConstraints:
    if constraints is None:
        return StellarConstraints()
    if isinstance(constraints, StellarConstraints):
        return constraints
    if callable(constraints):
        return StellarConstraints(predicate=constraints)
    raise TypeError("stellar constraints must be StellarConstraints, a callable, or None")


def _normalise_low_mass_teff_limit(
    value: Mapping[str, float] | None,
) -> _LowMassTeffLimit | None:
    if value is None:
        return None
    missing = [name for name in ("mass_max", "teff_max") if name not in value]
    if missing:
        raise ValueError(f"secondary_low_mass_teff_max is missing keys: {missing}")
    return _LowMassTeffLimit(
        mass_max=float(value["mass_max"]),
        teff_max=float(value["teff_max"]),
    )


def _candidate_points(
    candidate: IsochroneAgeCandidate,
    component: str,
) -> tuple[IsochronePoint, ...]:
    if component == "primary":
        return (candidate.primary,)
    if component == "secondary":
        if candidate.secondary is None:
            raise ValueError("secondary logg weight requires m2")
        return (candidate.secondary,)
    if candidate.secondary is None:
        raise ValueError("both-component logg weight requires m2")
    return (candidate.primary, candidate.secondary)


def _point_dict(point: IsochronePoint) -> dict[str, float]:
    return {
        "teff": float(point.teff),
        "logg": float(point.logg),
        "radius": float(point.radius),
    }


def _point_summary(point: IsochronePoint) -> str:
    return f"teff={point.teff:g}, logg={point.logg:g}, radius={point.radius:g}"


def _rejection_counts(rejections: list[tuple[float, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, reason in rejections:
        key = reason.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _callable_name(function: Callable[..., Any]) -> str:
    return getattr(function, "__name__", type(function).__name__)


def _weight_description(weight: AgeWeight | None) -> Any:
    if weight is None:
        return None
    describe = getattr(weight, "describe", None)
    if callable(describe):
        return describe()
    return _callable_name(weight)
