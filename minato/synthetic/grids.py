"""Text-file atmosphere-grid adapters for synthetic spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .models import Spectrum, Star


ParameterParser = Callable[[Path], Mapping[str, Any] | None]


@dataclass(frozen=True)
class AtmosphereGridNode:
    """One atmosphere-model file and its grid parameters."""

    teff: float
    logg: float
    path: Path
    format: str = "custom"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FormatRecogniser:
    name: str
    patterns: tuple[re.Pattern[str], ...]


_SUPPORTED_FORMATS = ("auto", "minato", "powr", "tlusty", "fastwind")


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_FORMAT_RECOGNISERS = {
    "minato": _FormatRecogniser(
        "minato",
        (
            _compile(
                r"^teff(?P<teff>\d+(?:\.\d+)?)"
                r"[_-]logg(?P<logg>\d+(?:\.\d+)?)"
            ),
        ),
    ),
    "powr": _FormatRecogniser(
        "powr",
        (
            _compile(
                r"gal-ob-vd3[_-](?P<teff_kk>\d+(?:\.\d+)?)"
                r"[-_](?P<logg10>\d+)(?:[_-]line(?:[_-]calib)?)?"
            ),
        ),
    ),
    "tlusty": _FormatRecogniser(
        "tlusty",
        (
            _compile(r"(?:^|[^a-z0-9])(?:[ob]?g|t)?(?P<teff>\d{5})g(?P<logg100>\d{3})"),
            _compile(
                r"(?:^|[^a-z0-9])(?P<teff_kk>\d{2,3})k"
                r"[_-]?g(?P<logg100>\d{3})"
            ),
        ),
    ),
    "fastwind": _FormatRecogniser(
        "fastwind",
        (
            _compile(
                r"(?:teff|t)[_-]?(?P<teff>\d{4,6}(?:\.\d+)?)"
                r".*?(?:logg|g)[_-]?(?P<logg>\d+(?:\.\d+)?)"
            ),
            _compile(
                r"(?:teff|t)[_-]?(?P<teff_kk>\d{2,3}(?:\.\d+)?)k"
                r".*?(?:logg|g)[_-]?(?P<logg>\d+(?:\.\d+)?)"
            ),
            _compile(
                r"(?:teff|t)[_-]?(?P<teff>\d{4,6}(?:\.\d+)?)"
                r".*?(?:logg|g)[_-]?(?P<logg100>\d{3})"
            ),
        ),
    ),
}


class TextAtmosphereGrid:
    """
    Nearest-neighbour atmosphere-grid backend for text model spectra.

    The default directory scanner recognises a compact MINATO convention
    (``teff25000_logg4.00.txt``) and common PoWR, TLUSTY, and FASTWIND-style
    names. It only chooses the nearest node inside one supplied grid; it does
    not decide which physical grid family should be used for a star.
    """

    def __init__(
        self,
        nodes: Iterable[AtmosphereGridNode],
        *,
        wavelength_column: int = 0,
        flux_column: int = 1,
        teff_scale: float = 1000.0,
        logg_scale: float = 0.1,
        max_teff_delta: float | None = None,
        max_logg_delta: float | None = None,
        log_flux: str = "never",
    ):
        self.nodes = tuple(nodes)
        if not self.nodes:
            raise ValueError("TextAtmosphereGrid needs at least one atmosphere node")
        self.wavelength_column = int(wavelength_column)
        self.flux_column = int(flux_column)
        self.teff_scale = float(teff_scale)
        self.logg_scale = float(logg_scale)
        self.max_teff_delta = max_teff_delta
        self.max_logg_delta = max_logg_delta
        self.log_flux = log_flux
        if self.teff_scale <= 0 or self.logg_scale <= 0:
            raise ValueError("teff_scale and logg_scale must be positive")
        if log_flux not in {"never", "auto", "always"}:
            raise ValueError("log_flux must be 'never', 'auto', or 'always'")
        self._cache: dict[Path, Spectrum] = {}

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        format: str = "auto",
        recursive: bool = True,
        filename_pattern: str | re.Pattern[str] | None = None,
        parser: ParameterParser | None = None,
        extensions: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> "TextAtmosphereGrid":
        """
        Build a grid by scanning a directory of text atmosphere-model files.

        ``format`` may be ``"auto"``, ``"minato"``, ``"powr"``, ``"tlusty"``,
        or ``"fastwind"``. For unconventional names, pass either a regex with
        named groups or a parser function returning ``{"teff": ..., "logg": ...}``.
        """

        root_path = Path(root)
        if format not in _SUPPORTED_FORMATS:
            raise ValueError(f"format must be one of {_SUPPORTED_FORMATS}")
        if not root_path.exists():
            raise FileNotFoundError(f"atmosphere-grid directory does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"atmosphere-grid path is not a directory: {root_path}")

        if filename_pattern is not None and parser is not None:
            raise ValueError("use either filename_pattern or parser, not both")
        files = _find_model_files(root_path, recursive=recursive, extensions=extensions)

        if parser is not None:
            format_name = "custom" if format == "auto" else format
            nodes = _nodes_from_parser(files, parser, format_name=format_name)
        elif filename_pattern is not None:
            if isinstance(filename_pattern, str):
                pattern = re.compile(filename_pattern, re.IGNORECASE)
            else:
                pattern = filename_pattern
            nodes = _nodes_from_pattern(files, pattern, format_name="custom")
        else:
            nodes = _nodes_from_known_formats(files, format=format)

        if not nodes:
            raise ValueError(_no_nodes_message(root_path, format))

        if "log_flux" not in kwargs and (
            format == "powr" or all(node.format == "powr" for node in nodes)
        ):
            kwargs["log_flux"] = "auto"
        return cls(nodes, **kwargs)

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        *,
        root: str | Path | None = None,
        **kwargs: Any,
    ) -> "TextAtmosphereGrid":
        """Build a grid from a CSV index with ``path``, ``teff``, and ``logg`` columns."""

        index_file = Path(index_path)
        base = Path(root) if root is not None else index_file.parent
        with index_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "teff", "logg"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{index_file} is missing required columns: {sorted(missing)}")
            nodes = []
            for row in reader:
                if not row.get("path") or not row.get("teff") or not row.get("logg"):
                    continue
                metadata = {key: value for key, value in row.items() if key not in required}
                nodes.append(
                    AtmosphereGridNode(
                        teff=float(row["teff"]),
                        logg=float(row["logg"]),
                        path=base / row["path"],
                        format=metadata.get("format", "index") or "index",
                        metadata=metadata,
                    )
                )
        return cls(nodes, **kwargs)

    @staticmethod
    def write_index_template(
        root: str | Path,
        output_path: str | Path,
        *,
        recursive: bool = True,
        extensions: Iterable[str] | None = None,
    ) -> Path:
        """
        Write an editable CSV index template for files that MINATO cannot parse.

        Users can fill in ``teff`` and ``logg`` and then load the result with
        ``TextAtmosphereGrid.from_index``.
        """

        root_path = Path(root)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        files = _find_model_files(root_path, recursive=recursive, extensions=extensions)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "teff", "logg", "format"])
            writer.writeheader()
            for filename in files:
                writer.writerow(
                    {
                        "path": filename.relative_to(root_path).as_posix(),
                        "teff": "",
                        "logg": "",
                        "format": "",
                    }
                )
        return output

    @staticmethod
    def recognised_formats() -> tuple[str, ...]:
        """Return recognised directory-scan format names."""

        return _SUPPORTED_FORMATS

    def nearest_node(self, star: Star) -> AtmosphereGridNode:
        """Return the nearest grid node for ``star`` in Teff-logg space."""

        teff = np.array([node.teff for node in self.nodes], dtype=float)
        logg = np.array([node.logg for node in self.nodes], dtype=float)
        score = ((teff - star.teff) / self.teff_scale) ** 2 + (
            (logg - star.logg) / self.logg_scale
        ) ** 2
        node = self.nodes[int(np.argmin(score))]
        d_teff = abs(node.teff - float(star.teff))
        d_logg = abs(node.logg - float(star.logg))
        if self.max_teff_delta is not None and d_teff > self.max_teff_delta:
            raise LookupError(
                f"nearest atmosphere node is {d_teff:.0f} K away in teff, "
                f"exceeding max_teff_delta={self.max_teff_delta}"
            )
        if self.max_logg_delta is not None and d_logg > self.max_logg_delta:
            raise LookupError(
                f"nearest atmosphere node is {d_logg:.2f} dex away in logg, "
                f"exceeding max_logg_delta={self.max_logg_delta}"
            )
        return node

    def get_spectrum(self, star: Star) -> Spectrum:
        """Load and return the nearest text model spectrum for ``star``."""

        node = self.nearest_node(star)
        if node.path not in self._cache:
            self._cache[node.path] = self._load_node(node)
        cached = self._cache[node.path]
        metadata = dict(cached.metadata)
        metadata["atmosphere_node"] = {
            "teff": float(node.teff),
            "logg": float(node.logg),
            "path": str(node.path),
            "format": node.format,
            **node.metadata,
        }
        return Spectrum(cached.wavelength.copy(), cached.flux.copy(), metadata=metadata)

    def _load_node(self, node: AtmosphereGridNode) -> Spectrum:
        if not node.path.exists():
            raise FileNotFoundError(f"atmosphere model file does not exist: {node.path}")
        data = np.loadtxt(node.path, comments="#", ndmin=2)
        required_column = max(self.wavelength_column, self.flux_column)
        if data.shape[1] <= required_column:
            raise ValueError(
                f"{node.path} has {data.shape[1]} columns, but column "
                f"{required_column} was requested"
            )
        wavelength = np.asarray(data[:, self.wavelength_column], dtype=float)
        flux = np.asarray(data[:, self.flux_column], dtype=float)
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        flux = flux[order]
        if self.log_flux == "always" or (self.log_flux == "auto" and np.nanmax(flux) < 0):
            flux = 10.0**flux
            flux_transform = "10**flux"
        else:
            flux_transform = "none"
        return Spectrum(
            wavelength,
            flux,
            metadata={
                "source_path": str(node.path),
                "source_format": node.format,
                "flux_transform": flux_transform,
            },
        )


class FallbackAtmosphereGrid:
    """
    Try multiple atmosphere-grid backends in user-defined priority order.

    Each backend must implement ``get_spectrum(star)``. A backend may decline a
    star by raising ``LookupError``; the next backend is then tried. Other
    exceptions propagate because they usually indicate a malformed grid or file.
    """

    def __init__(self, grids: Iterable[tuple[str, Any]]):
        self.grids = tuple(grids)
        if not self.grids:
            raise ValueError("FallbackAtmosphereGrid needs at least one named backend")
        for name, grid in self.grids:
            if not name:
                raise ValueError("fallback grid names must be non-empty")
            if not hasattr(grid, "get_spectrum"):
                raise TypeError(f"fallback grid {name!r} must define get_spectrum(star)")

    def get_spectrum(self, star: Star) -> Spectrum:
        """Return the first spectrum whose backend accepts ``star``."""

        failures = []
        for index, (name, grid) in enumerate(self.grids):
            try:
                spectrum = grid.get_spectrum(star)
            except LookupError as exc:
                failures.append(f"{name}: {exc}")
                continue
            metadata = dict(spectrum.metadata)
            metadata["selected_grid"] = {
                "name": name,
                "priority_index": index,
                "backend": type(grid).__name__,
            }
            return Spectrum(
                spectrum.wavelength.copy(),
                spectrum.flux.copy(),
                error=None if spectrum.error is None else spectrum.error.copy(),
                metadata=metadata,
            )

        detail = " | ".join(failures) if failures else "no grid attempted"
        raise LookupError(
            f"No atmosphere grid matched Teff={star.teff}, logg={star.logg}. {detail}"
        )


def _find_model_files(
    root: Path,
    *,
    recursive: bool,
    extensions: Iterable[str] | None,
) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    normalised_extensions = None
    if extensions is not None:
        normalised_extensions = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in extensions
        )
    files = []
    for path in iterator:
        if not path.is_file() or path.name.startswith("."):
            continue
        if normalised_extensions is not None and path.suffix.lower() not in normalised_extensions:
            continue
        files.append(path)
    return sorted(files)


def _nodes_from_parser(
    files: Iterable[Path],
    parser: ParameterParser,
    *,
    format_name: str,
) -> list[AtmosphereGridNode]:
    nodes = []
    for path in files:
        parsed = parser(path)
        if not parsed:
            continue
        nodes.append(_node_from_mapping(path, parsed, format_name=format_name))
    return nodes


def _nodes_from_pattern(
    files: Iterable[Path],
    pattern: re.Pattern[str],
    *,
    format_name: str,
) -> list[AtmosphereGridNode]:
    nodes = []
    for path in files:
        match = pattern.search(path.name)
        if match is None:
            continue
        params = _parameters_from_match(match)
        nodes.append(_node_from_mapping(path, params, format_name=format_name))
    return nodes


def _nodes_from_known_formats(files: Iterable[Path], *, format: str) -> list[AtmosphereGridNode]:
    recognisers = (
        _FORMAT_RECOGNISERS.values()
        if format == "auto"
        else (_FORMAT_RECOGNISERS[format],)
    )
    nodes = []
    for path in files:
        for recogniser in recognisers:
            node = _node_for_recogniser(path, recogniser)
            if node is not None:
                nodes.append(node)
                break
    return nodes


def _node_for_recogniser(path: Path, recogniser: _FormatRecogniser) -> AtmosphereGridNode | None:
    for pattern in recogniser.patterns:
        match = pattern.search(path.name)
        if match is None:
            continue
        params = _parameters_from_match(match)
        return _node_from_mapping(path, params, format_name=recogniser.name)
    return None


def _parameters_from_match(match: re.Match[str]) -> dict[str, float]:
    groups = {key: value for key, value in match.groupdict().items() if value is not None}
    if "teff" in groups:
        teff = float(groups["teff"])
    elif "teff_kk" in groups:
        teff = 1000.0 * float(groups["teff_kk"])
    else:
        raise ValueError("filename pattern must provide teff or teff_kk")

    if "logg" in groups:
        logg = _parse_logg_value(groups["logg"])
    elif "logg10" in groups:
        logg = float(groups["logg10"]) / 10.0
    elif "logg100" in groups:
        logg = float(groups["logg100"]) / 100.0
    else:
        raise ValueError("filename pattern must provide logg, logg10, or logg100")
    return {"teff": teff, "logg": logg}


def _parse_logg_value(value: str) -> float:
    parsed = float(value)
    if parsed >= 100:
        return parsed / 100.0
    if parsed >= 10:
        return parsed / 10.0
    return parsed


def _node_from_mapping(
    path: Path,
    mapping: Mapping[str, Any],
    *,
    format_name: str,
) -> AtmosphereGridNode:
    if "teff" not in mapping or "logg" not in mapping:
        raise ValueError("atmosphere-grid parser must return teff and logg")
    metadata = {key: value for key, value in mapping.items() if key not in {"teff", "logg"}}
    return AtmosphereGridNode(
        teff=float(mapping["teff"]),
        logg=float(mapping["logg"]),
        path=path,
        format=str(mapping.get("format", format_name)),
        metadata=metadata,
    )


def _no_nodes_message(root: Path, format: str) -> str:
    return (
        f"Could not infer teff/logg for any atmosphere models in {root} "
        f"with format={format!r}. Use one of: pass filename_pattern=..., "
        "pass parser=..., create an index with TextAtmosphereGrid.write_index_template(...), "
        "or rename/symlink files as teff25000_logg4.00.txt."
    )
