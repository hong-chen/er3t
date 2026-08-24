"""Education and Research 3D Radiative Transfer Toolbox.

The top-level package intentionally avoids importing the scientific stack. Public
subpackages and names historically re-exported from :mod:`er3t.common` are loaded
on first access so existing ``import er3t`` workflows keep working.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


try:
    __version__ = version("er3t")
except PackageNotFoundError:  # Running directly from an unpackaged source tree.
    __version__ = "0+unknown"


_LAZY_SUBMODULES = frozenset({"cli", "common", "dev", "pre", "rtm", "util", "vis"})
_COMMON_EXPORTS = frozenset(
    {
        "f_dtype",
        "i_dtype",
        "has_shdom",
        "has_mcarats",
        "has_libradtran",
        "has_token",
        "has_netcdf4",
        "has_hdf4",
        "has_hdf5",
        "has_xarray",
        "has_mpi",
        "fdir_er3t",
        "fdir_data",
        "fdir_data_solar",
        "fdir_data_abs",
        "fdir_data_pha",
        "fdir_data_atmmod",
        "fdir_data_slit",
        "fdir_data_ssfr",
        "fdir_data_tmp",
        "fdir_logs",
        "fdir_examples",
        "fdir_projects",
        "fdir_tests",
        "params",
        "logger",
        "references",
    }
)

__all__ = ["__version__", *_LAZY_SUBMODULES, *_COMMON_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load compatibility attributes only when they are requested."""

    if name in _LAZY_SUBMODULES:
        value = import_module(f"{__name__}.{name}")
    elif name in _COMMON_EXPORTS:
        value = getattr(import_module(f"{__name__}.common"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
