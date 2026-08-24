"""Education and Research 3D Radiative Transfer Toolbox.

The top-level package is intentionally lightweight. Scientific dependencies are
loaded when a public subpackage is accessed rather than during ``import er3t``.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


try:
    __version__ = version("er3t")
except PackageNotFoundError:  # Running directly from an unpackaged source tree.
    __version__ = "0+unknown"


_LAZY_SUBMODULES = ("cli", "common", "core", "io", "pre", "rtm", "sat", "visualization")
__all__ = ["__version__", *_LAZY_SUBMODULES]


def __getattr__(name: str) -> Any:
    """Load a public subpackage only when it is requested."""

    if name in _LAZY_SUBMODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
