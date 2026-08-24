"""Runtime capability detection without importing optional scientific modules."""

from dataclasses import dataclass
import importlib.util
import os
import shutil


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Optional executables, libraries, and credentials visible to this process."""

    shdom: bool
    mcarats: bool
    libradtran: bool
    earthdata_token: bool
    netcdf4: bool
    hdf4: bool
    hdf5: bool
    xarray: bool
    mpi: bool


def detect_capabilities() -> Capabilities:
    """Return a fresh capability snapshot.

    Detection is intentionally performed on request instead of at package import;
    callers that change environment variables during a session can re-check.
    """

    return Capabilities(
        shdom="SHDOM_EXE" in os.environ,
        mcarats="MCARATS_V010_EXE" in os.environ,
        libradtran="LIBRADTRAN_V2_DIR" in os.environ,
        earthdata_token="EARTHDATA_TOKEN" in os.environ,
        netcdf4=importlib.util.find_spec("netCDF4") is not None,
        hdf4=importlib.util.find_spec("pyhdf") is not None,
        hdf5=importlib.util.find_spec("h5py") is not None,
        xarray=importlib.util.find_spec("xarray") is not None,
        mpi=shutil.which("mpirun") is not None,
    )
