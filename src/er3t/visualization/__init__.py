"""Plotting and interactive visualization APIs."""

from importlib import import_module
from typing import Any


__all__ = ["plot", "grid", "interactive"]


def __getattr__(name: str) -> Any:
    if name == "interactive":
        value = import_module(f"{__name__}.interactive")
    elif name in {"plot", "grid"}:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
