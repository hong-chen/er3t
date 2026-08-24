"""Package resources and user-writable working directories."""

from __future__ import annotations

import os
from pathlib import Path


def package_dir() -> Path:
    """Return the installed ``er3t`` package directory."""

    return Path(__file__).resolve().parents[1]


def repository_dir() -> Path:
    """Return the source checkout directory when running from a checkout."""

    return package_dir().parent


def data_dir() -> Path:
    """Return the bundled data directory."""

    return package_dir() / "data"


def cache_dir(*, create: bool = False) -> Path:
    """Return the user-writable EaR³T cache directory.

    ``ER3T_CACHE_DIR`` takes precedence, followed by the platform's conventional
    cache variables. The directory is created only when explicitly requested.
    """

    configured = os.environ.get("ER3T_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "er3t"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "er3t"

    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def resource_path(*parts: str, must_exist: bool = False) -> Path:
    """Resolve a path inside the installed package data directory."""

    path = data_dir().joinpath(*parts)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"EaR³T resource does not exist: {path}")
    return path
