"""Compatibility namespace for EaR3T utility functions.

Utility modules are loaded lazily so using a numerical helper does not also load
every satellite reader and its optional dependencies.
"""

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "util": (
        "get_all_files", "get_all_folders", "load_h5", "check_equal",
        "check_equidistant", "send_email", "nice_array_str",
        "h5dset_to_pydict", "dtime_to_jday", "jday_to_dtime",
        "get_data_nc", "get_data_h4", "find_nearest", "move_correlate",
        "grid_by_extent", "grid_by_lonlat", "grid_by_dxdy", "get_doy_tag",
        "add_reference", "print_reference", "combine_alt", "get_lay_index",
        "downscale", "upscale_2d", "mmr2vmr", "cal_rho_air",
        "cal_sol_fac", "cal_sol_ang", "cal_mol_ext_atm", "mol_ext_wvl",
        "cal_mol_ext", "cal_ext", "cal_r_twostream", "cal_t_twostream",
        "cal_geodesic_dist", "cal_geodesic_lonlat", "format_time",
        "region_parser", "parse_geojson", "unpack_uint_to_bits",
    ),
    "modis": (
        "modis_l1b", "modis_l2", "modis_35_l2", "modis_03", "modis_04",
        "modis_07", "modis_09", "modis_09a1", "modis_43a1", "modis_43a3",
        "modis_tiff", "upscale_modis_lonlat", "download_modis_rgb",
        "download_modis_https", "cal_sinusoidal_grid",
        "get_sinusoidal_grid_tag",
    ),
    "viirs": (
        "viirs_03", "viirs_l1b", "viirs_cldprop_l2", "viirs_09a1",
        "viirs_43ma3", "viirs_43ma4",
    ),
    "ahi": ("ahi_l2",),
    "abi": ("abi_l2",),
    "oco2": (
        "oco2_rad_nadir", "oco2_std", "oco2_met", "get_fnames_from_web",
        "get_dtime_from_xml",
    ),
    "daac": (
        "format_satname", "get_token_earthdata", "gen_file_earthdata",
        "get_command_earthdata", "get_fname_geometa", "delete_file",
        "get_local_file", "get_online_file", "get_nsidc_file_list",
        "final_file_check", "read_geometa", "cal_proj_xy_geometa",
        "cal_lon_lat_utc_geometa", "cal_sec_offset_abi", "get_satfile_tag",
        "download_laads_https", "download_lance_https",
        "download_nsidc_https", "download_oco2_https",
        "download_worldview_image",
    ),
}

_SYMBOL_MODULES = {
    symbol: module_name
    for module_name, symbols in _MODULE_EXPORTS.items()
    for symbol in symbols
}

__all__ = [*_MODULE_EXPORTS, *_SYMBOL_MODULES]


def __getattr__(name: str) -> Any:
    module_name = name if name in _MODULE_EXPORTS else _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module

    if name == module_name:
        return module

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
