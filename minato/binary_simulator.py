"""Deprecated compatibility shim for binary_population.

Import `minato.binary_population` instead of this module. This file re-exports
all public symbols for backward compatibility.
"""

from .binary_population import *  # noqa: F401,F403
import warnings as _warnings

_warnings.warn(
    "minato.binary_simulator is deprecated; use minato.binary_population instead.",
    DeprecationWarning,
    stacklevel=2,
)
