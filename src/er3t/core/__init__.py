"""Small, dependency-light building blocks shared by EaR3T components."""

from .capabilities import Capabilities, detect_capabilities
from .config import Settings, default_settings
from .logging import configure_logging, get_logger, start_log_session
from .files import get_all_files, get_all_folders
from ._utilities import calculate_raa, find_nearest, move_correlate
from .numerics import (
    cal_ext,
    cal_geodesic_dist,
    cal_geodesic_lonlat,
    cal_mol_ext,
    cal_mol_ext_atm,
    cal_rho_air,
    cal_r_twostream,
    cal_sol_ang,
    cal_sol_fac,
    cal_t_twostream,
    check_equidistant,
    downscale,
    grid_by_dxdy,
    grid_by_extent,
    grid_by_lonlat,
    get_lay_index,
    mmr2vmr,
    nice_array_str,
    unpack_uint_to_bits,
)
from .references import add_reference, get_references, print_references
from ._utilities import print_reference
from .resources import (
    cache_dir,
    data_dir,
    package_dir,
    repository_dir,
    resource_path,
)

__all__ = [
    "Capabilities",
    "Settings",
    "cache_dir",
    "calculate_raa",
    "cal_ext",
    "cal_geodesic_dist",
    "cal_geodesic_lonlat",
    "cal_mol_ext",
    "cal_mol_ext_atm",
    "cal_rho_air",
    "cal_r_twostream",
    "cal_sol_ang",
    "cal_sol_fac",
    "cal_t_twostream",
    "check_equidistant",
    "configure_logging",
    "data_dir",
    "default_settings",
    "detect_capabilities",
    "downscale",
    "find_nearest",
    "get_logger",
    "get_all_files",
    "get_all_folders",
    "add_reference",
    "get_references",
    "grid_by_dxdy",
    "grid_by_extent",
    "grid_by_lonlat",
    "get_lay_index",
    "mmr2vmr",
    "move_correlate",
    "nice_array_str",
    "package_dir",
    "repository_dir",
    "resource_path",
    "start_log_session",
    "print_references",
    "print_reference",
    "unpack_uint_to_bits",
]
