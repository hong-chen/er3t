"""Numerical and geospatial primitives used across EaR³T."""

# This module is the stable home for these functions while their implementation
# is split out of the historical utility file. New code should import here.
from ._utilities import (
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

__all__ = [
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
    "downscale",
    "grid_by_dxdy",
    "grid_by_extent",
    "grid_by_lonlat",
    "get_lay_index",
    "mmr2vmr",
    "nice_array_str",
    "unpack_uint_to_bits",
]
