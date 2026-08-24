"""Interfaces to supported radiative-transfer models."""

from importlib import import_module
from typing import Any


__all__ = ["drt", "lrt", "mca", "shd"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
