"""
EaR³T Example 06 — MODIS Data Download and Pre-processing
==========================================================

What this script does
---------------------
Downloads five MODIS products for a given date and region from NASA LAADS,
grids them onto a common 250 m domain, and saves everything needed for the
3D radiance simulation in Example 07 into a single HDF5 file (modis_scene_input.h5).
A four-panel overview figure is also produced.

Cloud field treatment
---------------------
MODIS L2 cloud properties (COT, CER, CTH) are available at 1 km resolution.
This script interpolates them to the 250 m target grid using bilinear
interpolation ('linear'), which tapers cloud edges smoothly rather than
repeating coarse 1 km blocks.  The original full-research code (02_modis_rad-
sim.py) goes further: it runs an IPA retrieval to derive 250 m COT from the
250 m reflectance field.  That step is intentionally omitted here to keep
the example simple — using L2 COT directly is a valid and physically
transparent starting point.

Outputs
-------
* data/06_modis_data_download/modis_scene_input.h5  — gridded inputs for Example 07
* 06_modis_data_download-overview_<wl>nm.png — 4-panel overview

Required environment variable
------------------------------
    export EARTHDATA_TOKEN="<your-token>"
    # Get one at https://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens

Student controls — edit the block below
----------------------------------------
"""

import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import h5py

import er3t


# ============================================================
#  STUDENT CONTROL BLOCK — modify these for different scenes
# ============================================================
date       = datetime.datetime(2019, 9, 2)           # MODIS-Aqua overpass date
extent     = [-108.6, -107.4, 37.4, 38.6]           # [lon_min, lon_max, lat_min, lat_max]
wavelength = 650                                     # nm  (650 = MODIS Band 1, 620–670 nm)
dx         = 250.0                                   # m, target grid spacing in x
dy         = 250.0                                   # m, target grid spacing in y
satellite  = 'aqua'                                  # 'aqua' (MYD) or 'terra' (MOD)
overwrite  = False                                   # set True to re-download / re-process
# ============================================================


_NAME     = os.path.splitext(os.path.basename(__file__))[0]
_FDIR     = os.path.join('data', _NAME)
_FNAME_H5 = os.path.join(_FDIR, 'modis_scene_input.h5')


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _sinusoidal_pixel_lonlat(fname, Nx=2400, Ny=2400):
    """Return (lon, lat) arrays for every pixel centre in a MODIS sinusoidal tile.

    Uses the pure-NumPy inverse sinusoidal projection (no cartopy required).
    Works for any MCD43Ax HDF file whose tile tag is encoded in the filename,
    e.g. MCD43A1.A2019245.h09v05.061.xxx.hdf → h=9, v=5.

    Returns
    -------
    lon, lat : ndarray of shape (Ny, Nx)
    """
    R = 6_371_007.181
    index_str = os.path.basename(fname).split('.')[2]        # e.g. 'h09v05'
    ih = int(index_str[1:3])
    iv = int(index_str[4:])

    x_min = np.deg2rad(-180.0) * R
    x_max = np.deg2rad( 180.0) * R
    y_max = np.deg2rad(  90.0) * R
    y_min = np.deg2rad( -90.0) * R
    tile_w = (x_max - x_min) / 36.0
    tile_h = (y_max - y_min) / 18.0

    xlo = x_min +  ih      * tile_w
    xhi = x_min + (ih + 1) * tile_w
    yhi = y_max -  iv      * tile_h        # north edge (higher y)
    ylo = y_max - (iv + 1) * tile_h        # south edge (lower y)

    x_tmp = np.linspace(xlo, xhi, Nx + 1)
    y_tmp = np.linspace(yhi, ylo, Ny + 1)  # north → south
    x_mid = (x_tmp[1:] + x_tmp[:-1]) / 2.0
    y_mid = (y_tmp[1:] + y_tmp[:-1]) / 2.0
    XX, YY = np.meshgrid(x_mid, y_mid)     # shape (Ny, Nx)

    lat = np.rad2deg(YY / R)
    with np.errstate(invalid='ignore', divide='ignore'):
        lon = np.where(np.abs(np.cos(YY / R)) > 1e-10,
                       np.rad2deg(XX / (R * np.cos(YY / R))),
                       np.nan)
    return lon, lat


def _hdf4_read_scaled(fname, varname):
    """Read one variable from an HDF4 file, apply scale/offset, fill → NaN."""
    from pyhdf.SD import SD, SDC
    f    = SD(fname, SDC.READ)
    ds   = f.select(varname)
    attr = ds.attributes()
    data = ds[:].astype(np.float64)
    f.end()
    if '_FillValue' in attr:
        data[data == float(attr['_FillValue'])] = np.nan
    if 'add_offset'    in attr:
        data -= float(attr['add_offset'])
    if 'scale_factor'  in attr:
        data *= float(attr['scale_factor'])
    return data


def _read_mcd43a1(fnames, extent, Nx=2400, Ny=2400):
    """Read MCD43A1 BRDF parameters, returning 1-D arrays of valid pixels.

    Replaces ``er3t.util.modis_43a1`` with a cartopy-free implementation.
    Channels: 0=620-670 nm (Band 1), 1=841-876 nm (Band 2), ...
    """
    lon_all  = []
    lat_all  = []
    fiso_all = []
    fvol_all = []
    fgeo_all = []

    lon_range = [extent[0] - 0.01, extent[1] + 0.01]
    lat_range = [extent[2] - 0.01, extent[3] + 0.01]
    Nchan = 7

    for fname in fnames:
        lon2d, lat2d = _sinusoidal_pixel_lonlat(fname, Nx=Nx, Ny=Ny)
        mask = ((lon2d >= lon_range[0]) & (lon2d <= lon_range[1]) &
                (lat2d >= lat_range[0]) & (lat2d <= lat_range[1]))
        n = mask.sum()
        if n == 0:
            continue
        fiso = np.zeros((Nchan, n), dtype=np.float32)
        fvol = np.zeros((Nchan, n), dtype=np.float32)
        fgeo = np.zeros((Nchan, n), dtype=np.float32)
        for ic in range(Nchan):
            raw = _hdf4_read_scaled(fname, 'BRDF_Albedo_Parameters_Band%d' % (ic + 1))
            fiso[ic, :] = raw[mask, 0]
            fvol[ic, :] = raw[mask, 1]
            fgeo[ic, :] = raw[mask, 2]
        lon_all.append(lon2d[mask])
        lat_all.append(lat2d[mask])
        fiso_all.append(fiso)
        fvol_all.append(fvol)
        fgeo_all.append(fgeo)

    return {
        'lon':   {'data': np.concatenate(lon_all)  if lon_all  else np.array([])},
        'lat':   {'data': np.concatenate(lat_all)  if lat_all  else np.array([])},
        'f_iso': {'data': np.hstack(fiso_all)       if fiso_all else np.zeros((Nchan, 0))},
        'f_vol': {'data': np.hstack(fvol_all)       if fvol_all else np.zeros((Nchan, 0))},
        'f_geo': {'data': np.hstack(fgeo_all)       if fgeo_all else np.zeros((Nchan, 0))},
    }


def _read_mcd43a3(fnames, extent, Nx=2400, Ny=2400):
    """Read MCD43A3 white-sky surface albedo, returning 1-D arrays of valid pixels.

    Replaces ``er3t.util.modis_43a3`` with a cartopy-free implementation.
    """
    lon_all = []
    lat_all = []
    wsa_all = []

    lon_range = [extent[0] - 0.01, extent[1] + 0.01]
    lat_range = [extent[2] - 0.01, extent[3] + 0.01]
    Nchan = 7

    for fname in fnames:
        lon2d, lat2d = _sinusoidal_pixel_lonlat(fname, Nx=Nx, Ny=Ny)
        mask = ((lon2d >= lon_range[0]) & (lon2d <= lon_range[1]) &
                (lat2d >= lat_range[0]) & (lat2d <= lat_range[1]))
        n = mask.sum()
        if n == 0:
            continue
        wsa = np.zeros((Nchan, n), dtype=np.float32)
        for ic in range(Nchan):
            raw = _hdf4_read_scaled(fname, 'Albedo_WSA_Band%d' % (ic + 1))
            vals = raw[mask]
            vals[(vals > 1.0) | (vals < 0.0)] = np.nan
            wsa[ic, :] = vals
        lon_all.append(lon2d[mask])
        lat_all.append(lat2d[mask])
        wsa_all.append(wsa)

    return {
        'lon': {'data': np.concatenate(lon_all) if lon_all else np.array([])},
        'lat': {'data': np.concatenate(lat_all) if lat_all else np.array([])},
        'wsa': {'data': np.hstack(wsa_all)       if wsa_all else np.zeros((Nchan, 0))},
    }


def _get_sinusoidal_tags_numpy(lon, lat):
    """Return MODIS sinusoidal tile tag(s) covering the given lon/lat arrays.

    This is a pure-NumPy fallback for ``er3t.util.modis.get_sinusoidal_grid_tag``
    that does not require cartopy.  It uses the standard MODIS sinusoidal
    projection definition (sphere radius R = 6,371,007.181 m).

    Returns a list of strings like ['h09v05'].
    """
    R = 6_371_007.181                          # MODIS sphere radius [m]
    lon_r = np.deg2rad(np.asarray(lon).ravel())
    lat_r = np.deg2rad(np.asarray(lat).ravel())

    x = lon_r * np.cos(lat_r) * R             # sinusoidal x [m]
    y = lat_r * R                              # sinusoidal y [m]

    # Full-grid bounds in x and y
    x_min = np.deg2rad(-180.0) * R            # ≈ -20,015,109 m
    x_max = np.deg2rad( 180.0) * R            # ≈ +20,015,109 m
    y_max = np.deg2rad(  90.0) * R            # ≈ +10,007,554 m
    y_min = np.deg2rad( -90.0) * R            # ≈ -10,007,554 m

    tile_w = (x_max - x_min) / 36.0           # ≈ 1,111,950 m per tile
    tile_h = (y_max - y_min) / 18.0           # ≈ 1,111,950 m per tile

    tags = []
    for ih in range(36):
        for iv in range(18):
            xlo = x_min + ih       * tile_w
            xhi = x_min + (ih + 1) * tile_w
            ylo = y_max - (iv + 1) * tile_h   # south edge (lower y)
            yhi = y_max - iv       * tile_h   # north edge (higher y)
            inside = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
            if inside.any():
                tags.append('h%02dv%02d' % (ih, iv))
    return tags


# ─────────────────────────────────────────────────────────────
#  DOWNLOAD
# ─────────────────────────────────────────────────────────────

def download_modis_products(date, extent, fdir, satellite='aqua', overwrite=False):
    """Download all MODIS products needed for the 3D radiance simulation.

    Products downloaded (~390 MB total):
      MYD02QKM  — 250 m L1B radiance / reflectance
      MYD03     — geolocation (SZA, SAA, VZA, VAA, surface height)
      MYD06_L2  — 1 km cloud product (COT, CER, CTH)
      MCD43A1   — BRDF parameters (fiso, fvol, fgeo) for MODIS Band 1
      MCD43A3   — white-sky surface albedo
      RGB image — true-colour Worldview tile (no auth required)

    Parameters
    ----------
    date      : datetime  — overpass date
    extent    : list      — [lon_min, lon_max, lat_min, lat_max]
    fdir      : str       — directory for downloaded files
    satellite : str       — 'aqua' or 'terra'
    overwrite : bool      — re-download even if files exist

    Returns
    -------
    fnames : dict  — lists of file paths, keyed by product
    """
    os.makedirs(fdir, exist_ok=True)

    lon0 = np.linspace(extent[0], extent[1], 100)
    lat0 = np.linspace(extent[2], extent[3], 100)
    lon, lat = np.meshgrid(lon0, lat0, indexing='ij')

    if satellite.lower() == 'aqua':
        tags_03  = '61/MYD03'
        tags_l2  = '61/MYD06_L2'
        tags_02  = '61/MYD02QKM'
    else:
        tags_03  = '61/MOD03'
        tags_l2  = '61/MOD06_L2'
        tags_02  = '61/MOD02QKM'

    fnames = {'mod_rgb': [], 'mod_03': [], 'mod_l2': [], 'mod_02': [],
              'mod_43a1': [], 'mod_43a3': []}

    # Worldview RGB (no Earthdata token needed)
    print('Downloading RGB imagery ...')
    fnames['mod_rgb'] = [er3t.util.download_worldview_image(
        date, extent, fdir_out=fdir, satellite=satellite,
        instrument='modis', coastline=True)]

    # L1B / L2 / geolocation (Earthdata token required)
    print('Finding overpass granule IDs ...')
    filename_tags_03 = er3t.util.get_satfile_tag(
        date, lon, lat, satellite=satellite, instrument='modis')
    print('  Found %d overpass(es).' % len(filename_tags_03))

    for ftag in filename_tags_03:
        fnames['mod_03'] += er3t.util.download_laads_https(
            date, tags_03, ftag, day_interval=1, fdir_out=fdir, run=True)
        fnames['mod_l2'] += er3t.util.download_laads_https(
            date, tags_l2, ftag, day_interval=1, fdir_out=fdir, run=True)
        fnames['mod_02'] += er3t.util.download_laads_https(
            date, tags_02, ftag, day_interval=1, fdir_out=fdir, run=True)

    # MODIS surface (BRDF + white-sky albedo)
    print('Downloading surface reflectance products ...')
    try:
        filename_tags_43 = er3t.util.modis.get_sinusoidal_grid_tag(lon, lat)
        print('  Sinusoidal grid tile(s) for this region: %s' % filename_tags_43)
    except Exception as _e:
        print('  WARNING: get_sinusoidal_grid_tag raised %s: %s' % (type(_e).__name__, _e))
        filename_tags_43 = []

    if not filename_tags_43:
        # Cartopy may be unavailable or returned empty; fall back to a pure-NumPy calculation
        # of the MODIS sinusoidal tile(s) covering the requested extent.
        print('  Falling back to NumPy-only sinusoidal tile calculation ...')
        filename_tags_43 = _get_sinusoidal_tags_numpy(lon, lat)
        print('  Sinusoidal grid tile(s) (fallback): %s' % filename_tags_43)
    if not filename_tags_43:
        print('  WARNING: could not determine sinusoidal grid tile — '
              'MCD43A1/MCD43A3 will not be downloaded.')
    for ftag in filename_tags_43:
        print('  Fetching MCD43A1/%s ...' % ftag)
        result_43a1 = er3t.util.download_laads_https(
            date, '61/MCD43A1', ftag, day_interval=1, fdir_out=fdir, run=True)
        print('    → download_laads_https returned: %s' % result_43a1)
        fnames['mod_43a1'] += result_43a1
        print('  Fetching MCD43A3/%s ...' % ftag)
        result_43a3 = er3t.util.download_laads_https(
            date, '61/MCD43A3', ftag, day_interval=1, fdir_out=fdir, run=True)
        print('    → download_laads_https returned: %s' % result_43a3)
        fnames['mod_43a3'] += result_43a3

    # download_laads_https returns [] for files that were skipped (already on disk).
    # Fall back to a directory scan so fnames is always populated on repeat runs.
    import glob
    sat_prefix = 'MYD' if satellite.lower() == 'aqua' else 'MOD'
    _fallback = {
        'mod_03':  sat_prefix + '03*.hdf',
        'mod_l2':  sat_prefix + '06_L2*.hdf',
        'mod_02':  sat_prefix + '02QKM*.hdf',
        'mod_43a1': 'MCD43A1*.hdf',
        'mod_43a3': 'MCD43A3*.hdf',
    }
    for key, pattern in _fallback.items():
        if not fnames[key]:
            fnames[key] = sorted(glob.glob(os.path.join(fdir, pattern)))
            if fnames[key]:
                print('  %s: found %d file(s) on disk.' % (key, len(fnames[key])))

    return fnames


# ─────────────────────────────────────────────────────────────
#  PRE-PROCESSING
# ─────────────────────────────────────────────────────────────

def _interp_cloud_field(lon0, lat0, data0, extent, dx, dy):
    """Grid a cloud-property field (COT, CER, or CTH) to the 250 m domain.

    Strategy
    --------
    Clear-sky pixels have NaN in the L2 product.  We replace NaN with 0
    before bilinear interpolation so that cloud edges taper smoothly to zero
    rather than creating abrupt 1 km steps.  Any slightly negative values
    produced at the very edge of the interpolation are clipped to 0.

    Note: bilinear ('linear') interpolation smooths cloud boundaries compared
    to the nearest-neighbour approach used by the full research code.  This is
    physically reasonable — real cloud edges are not perfectly sharp — and
    avoids the 1 km staircase artefact seen with nearest-neighbour at 250 m.
    """
    # Replace NaN (clear sky) with 0 so the interpolation tapers to 0 at edges
    data0_filled = np.where(np.isnan(data0), 0.0, data0)

    _, _, data_2d = er3t.util.grid_by_dxdy(
        lon0, lat0, data0_filled,
        extent=extent, dx=dx, dy=dy,
        method='linear', Ngrid_limit=4, fill_value=0.0)

    return np.maximum(data_2d, 0.0)   # clip any sub-zero artefacts


def preprocess_to_h5(fnames, date, extent, wvl, dx, dy, fname_h5, overwrite=False):
    """Read downloaded MODIS files, grid to 250 m, and save to HDF5.

    The output file contains:
      extent                  (4,)           — [lon_min, lon_max, lat_min, lat_max]
      lon, lat                (Nx, Ny)       — 250 m grid coordinates
      mod/rgb                 (H, W, 4)      — RGBA Worldview image
      mod/rad/rad_NNNN        (Nx, Ny)       — L1B spectral radiance [W m⁻² sr⁻¹ µm⁻¹]
      mod/rad/ref_NNNN        (Nx, Ny)       — L1B reflectance (0–1)
      mod/geo/sza/saa/vza/vaa (Nx, Ny)       — illumination/view geometry [deg]
      mod/geo/sfh             (Nx, Ny)       — surface elevation [km]
      mod/cld/cot_l2          (Nx, Ny)       — L2 COT, interpolated to 250 m
      mod/cld/cer_l2          (Nx, Ny)       — L2 CER [µm], interpolated to 250 m
      mod/cld/cth_l2          (Nx, Ny)       — L2 CTH [km], interpolated to 250 m
      mod/cld/cot_ipa         (Nx, Ny)       — alias for cot_l2 (used by Example 07)
      mod/cld/cer_ipa         (Nx, Ny)       — alias for cer_l2
      mod/cld/cth_ipa         (Nx, Ny)       — alias for cth_l2
      mod/sfc/fiso_43_NNNN    (Nx, Ny)       — BRDF isotropic kernel weight
      mod/sfc/fvol_43_NNNN    (Nx, Ny)       — BRDF volumetric kernel weight
      mod/sfc/fgeo_43_NNNN    (Nx, Ny)       — BRDF geometric kernel weight
      mod/sfc/alb_43_NNNN     (Nx, Ny)       — white-sky albedo
    """
    # Check whether an existing file is complete (has the minimum required datasets).
    # We check specific leaf datasets — not just group existence — because a partial run
    # creates all HDF5 groups before filling them, which would fool a group-only check.
    _required_datasets = [
        'lon', 'lat', 'extent',
        'mod/rad/rad_%4.4d' % wvl,
        'mod/geo/sza',
        'mod/cld/cot_l2',
        'mod/sfc/alb_43_%4.4d' % wvl,   # written last — best proxy for a complete run
    ]
    if os.path.exists(fname_h5) and not overwrite:
        try:
            with h5py.File(fname_h5, 'r') as _f:
                _complete = all(k in _f for k in _required_datasets)
        except Exception:
            _complete = False
        if _complete:
            print('modis_scene_input.h5 already exists and is complete — skipping '
                  '(set overwrite=True to reprocess).')
            return
        else:
            print('modis_scene_input.h5 exists but is incomplete — reprocessing.')
            os.remove(fname_h5)

    # MODIS band index for this wavelength
    wvl_key = {650: 0, 860: 1}
    if wvl not in wvl_key:
        raise ValueError('Only 650 nm and 860 nm are supported (MODIS Bands 1 and 2).')
    idx_wvl = wvl_key[wvl]

    f0 = h5py.File(fname_h5, 'w')
    f0['extent'] = extent

    g     = f0.create_group('mod')
    g_geo = g.create_group('geo')
    g_rad = g.create_group('rad')
    g_cld = g.create_group('cld')
    g_sfc = g.create_group('sfc')

    # ── RGB ──────────────────────────────────────────────────
    import matplotlib.image as mpl_img
    rgb = mpl_img.imread(fnames['mod_rgb'][0])
    g['rgb'] = rgb
    print('  RGB: done.')

    # ── L1B radiance / reflectance ────────────────────────────
    modl1b = er3t.util.modis_l1b(fnames=fnames['mod_02'], extent=extent)
    lon0   = modl1b.data['lon']['data']
    lat0   = modl1b.data['lat']['data']
    ref0   = modl1b.data['ref']['data'][idx_wvl, ...]
    rad0   = modl1b.data['rad']['data'][idx_wvl, ...]

    lon_2d, lat_2d, ref_2d = er3t.util.grid_by_dxdy(
        lon0, lat0, ref0, extent=extent, dx=dx, dy=dy, method='nearest')
    lon_2d, lat_2d, rad_2d = er3t.util.grid_by_dxdy(
        lon0, lat0, rad0, extent=extent, dx=dx, dy=dy, method='nearest')

    g_rad['rad_%4.4d' % wvl] = rad_2d
    g_rad['ref_%4.4d' % wvl] = ref_2d
    f0['lon'] = lon_2d
    f0['lat'] = lat_2d
    print('  L1B radiance / reflectance: done.')

    # ── Geometry (SZA, SAA, VZA, VAA, surface height) ────────
    mod03 = er3t.util.modis_03(
        fnames=fnames['mod_03'], extent=extent, vnames=['Height'])
    lon0  = mod03.data['lon']['data']
    lat0  = mod03.data['lat']['data']

    for key, arr, grp_key in [
            ('sza', mod03.data['sza']['data'],            'sza'),
            ('saa', mod03.data['saa']['data'],            'saa'),
            ('vza', mod03.data['vza']['data'],            'vza'),
            ('vaa', mod03.data['vaa']['data'],            'vaa'),
            ('sfh', mod03.data['height']['data']/1000.0,  'sfh'),
    ]:
        arr_clipped = arr.copy()
        if key == 'sfh':
            arr_clipped[arr_clipped < 0.0] = np.nan
        _, _, arr_2d = er3t.util.grid_by_dxdy(
            lon0, lat0, arr_clipped,
            extent=extent, dx=dx, dy=dy, method='linear')
        g_geo[grp_key] = arr_2d

    print('  Geolocation / geometry: done.')

    # ── Cloud properties (COT, CER, CTH) — bilinear from 1 km ─
    modl2 = er3t.util.modis_l2(
        fnames=fnames['mod_l2'], extent=extent,
        vnames=['cloud_top_height_1km'])
    lon0  = modl2.data['lon']['data']
    lat0  = modl2.data['lat']['data']

    cot0  = modl2.data['cot']['data']
    cer0  = modl2.data['cer']['data']
    cth0  = modl2.data['cloud_top_height_1km']['data'] / 1000.0   # → km
    cth0[cth0 <= 0.0] = np.nan

    cot_2d = _interp_cloud_field(lon0, lat0, cot0, extent, dx, dy)
    cer_2d = _interp_cloud_field(lon0, lat0, cer0, extent, dx, dy)
    cth_2d = _interp_cloud_field(lon0, lat0, cth0, extent, dx, dy)

    # Store L2 fields
    g_cld['cot_l2'] = cot_2d
    g_cld['cer_l2'] = cer_2d
    g_cld['cth_l2'] = cth_2d

    # Alias as '_ipa' so Example 07 reads them directly without changes
    g_cld['cot_ipa'] = cot_2d
    g_cld['cer_ipa'] = cer_2d
    g_cld['cth_ipa'] = cth_2d

    print('  Cloud properties (bilinear, 1 km → 250 m): done.')

    # ── Surface BRDF (MCD43A1) + white-sky albedo (MCD43A3) ──
    # We use our own NumPy-based readers (_read_mcd43a1, _read_mcd43a3) rather
    # than er3t.util.modis_43a1/43a3 because the er3t readers rely on cartopy's
    # transform_points, which can return NaN for the MODIS sinusoidal projection
    # in some cartopy versions.
    if not fnames['mod_43a1']:
        raise FileNotFoundError(
            'No MCD43A1 files found. The surface BRDF download may have failed.\n'
            'Check the Step 1 output above for the sinusoidal grid tile and download messages.\n'
            'You can also try running with --overwrite to force a fresh download.')
    if not fnames['mod_43a3']:
        raise FileNotFoundError(
            'No MCD43A3 files found. The surface albedo download may have failed.')

    sfc43a1 = _read_mcd43a1(fnames['mod_43a1'], extent)
    lo = sfc43a1['lon']['data']
    la = sfc43a1['lat']['data']
    if lo.size == 0:
        raise ValueError('_read_mcd43a1 returned no pixels within extent — '
                         'check that the MCD43A1 tile covers the requested region.')

    for coef, key in [
            (sfc43a1['f_iso']['data'][idx_wvl, :], 'fiso_43_%4.4d' % wvl),
            (sfc43a1['f_vol']['data'][idx_wvl, :], 'fvol_43_%4.4d' % wvl),
            (sfc43a1['f_geo']['data'][idx_wvl, :], 'fgeo_43_%4.4d' % wvl),
    ]:
        _, _, c_2d = er3t.util.grid_by_dxdy(
            lo, la, coef, extent=extent, dx=dx, dy=dy,
            method='nearest', Ngrid_limit=4)
        c_2d = np.where(np.isnan(c_2d) | np.isinf(c_2d) | (c_2d < 0) | (c_2d > 1),
                        0.0, c_2d)
        g_sfc[key] = c_2d

    sfc43a3 = _read_mcd43a3(fnames['mod_43a3'], extent)
    lo = sfc43a3['lon']['data']
    la = sfc43a3['lat']['data']
    alb0  = sfc43a3['wsa']['data'][idx_wvl, :]
    _, _, alb_2d = er3t.util.grid_by_dxdy(
        lo, la, alb0, extent=extent, dx=dx, dy=dy,
        method='nearest', Ngrid_limit=4)
    alb_2d = np.where(np.isnan(alb_2d) | np.isinf(alb_2d) | (alb_2d < 0) | (alb_2d > 1),
                      0.0, alb_2d)
    g_sfc['alb_43_%4.4d' % wvl] = alb_2d

    print('  Surface BRDF / albedo: done.')

    f0.close()
    print('\nSaved: %s' % fname_h5)


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_modis_overview(fname_h5, wvl, name_tag):
    """Four-panel overview of the downloaded and gridded MODIS data.

    Panel layout
    ------------
    Top-left     — true-colour RGB (Worldview)
    Top-right    — cloud optical thickness (COT, 250 m)
    Bottom-left  — solar zenith angle (SZA)
    Bottom-right — white-sky surface albedo at wavelength wvl
    """
    f = h5py.File(fname_h5, 'r')
    extent  = f['extent'][...]
    lon     = f['lon'][...]
    lat     = f['lat'][...]
    rgb     = f['mod/rgb'][...]
    cot     = f['mod/cld/cot_l2'][...]
    sza     = f['mod/geo/sza'][...]
    alb     = f['mod/sfc/alb_43_%4.4d' % wvl][...]
    f.close()

    # lon/lat edges of the image for imshow extent
    lon_min, lon_max = extent[0], extent[1]
    lat_min, lat_max = extent[2], extent[3]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MODIS Scene Overview — %s nm' % wvl, fontsize=13)

    tick_kw = dict(labelsize=8)

    # ── RGB ──────────────────────────────────────────────────
    ax = axes[0, 0]
    # The PNG was saved by matplotlib/cartopy with row 0 = top = north,
    # so use origin='upper' to keep north at the top.
    ax.imshow(rgb, extent=[lon_min, lon_max, lat_min, lat_max],
              origin='upper', aspect='auto')
    ax.set_title('True-colour RGB (Worldview)', fontsize=10)
    ax.set_xlabel('Longitude [°]', fontsize=9)
    ax.set_ylabel('Latitude [°]', fontsize=9)
    ax.tick_params(**tick_kw)

    # ── COT ──────────────────────────────────────────────────
    ax = axes[0, 1]
    cot_plot = np.where(cot <= 0, np.nan, cot)
    cm = ax.pcolormesh(lon, lat, cot_plot, cmap='YlOrBr', vmin=0, vmax=40,
                       shading='auto')
    fig.colorbar(cm, ax=ax, label='COT')
    ax.set_title('Cloud Optical Thickness (250 m, L2)', fontsize=10)
    ax.set_xlabel('Longitude [°]', fontsize=9)
    ax.set_ylabel('Latitude [°]', fontsize=9)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(**tick_kw)

    # ── SZA ──────────────────────────────────────────────────
    ax = axes[1, 0]
    cm = ax.pcolormesh(lon, lat, sza, cmap='plasma', vmin=20, vmax=60,
                       shading='auto')
    fig.colorbar(cm, ax=ax, label='SZA [°]')
    ax.set_title('Solar Zenith Angle', fontsize=10)
    ax.set_xlabel('Longitude [°]', fontsize=9)
    ax.set_ylabel('Latitude [°]', fontsize=9)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(**tick_kw)

    # ── Surface albedo ────────────────────────────────────────
    ax = axes[1, 1]
    cm = ax.pcolormesh(lon, lat, alb, cmap='Greens', vmin=0, vmax=0.3,
                       shading='auto')
    fig.colorbar(cm, ax=ax, label='Albedo')
    ax.set_title('White-sky Surface Albedo (%d nm)' % wvl, fontsize=10)
    ax.set_xlabel('Longitude [°]', fontsize=9)
    ax.set_ylabel('Latitude [°]', fontsize=9)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(**tick_kw)

    plt.tight_layout()
    fname_png = '%s-overview_%dnm.png' % (name_tag, wvl)
    plt.savefig(fname_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: %s' % fname_png)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download and pre-process MODIS data for the 3D radiance example.')
    parser.add_argument('--date',   type=str,   default=None,
        help='Overpass date YYYY-MM-DD  (default: %s)' % date.strftime('%Y-%m-%d'))
    parser.add_argument('--extent', type=float, nargs=4,
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        default=None,
        help='Spatial extent in decimal degrees  (default: %.1f %.1f %.1f %.1f)' % tuple(extent))
    parser.add_argument('--satellite', type=str, default=satellite,
        help="'aqua' or 'terra'  (default: %s)" % satellite)
    parser.add_argument('--overwrite', action='store_true',
        help='Re-download / re-process even if files already exist.')
    parser.add_argument('--plot-only', action='store_true',
        help='Skip download and pre-processing; re-generate the figure from an existing h5 file.')
    args = parser.parse_args()

    # Apply CLI overrides
    _date    = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date   else date
    _extent  = args.extent  if args.extent  else extent
    _sat     = args.satellite
    _ow      = args.overwrite or overwrite

    # Derive output paths from date/extent so different scenes don't collide
    tag_ext  = '(%.2f,%.2f,%.2f,%.2f)' % tuple(_extent)
    tag_date = _date.strftime('%Y-%m-%d')
    _name    = _NAME + '_' + tag_date + '_' + tag_ext
    _fdir    = os.path.join('data', _NAME, tag_date + tag_ext)
    _fname_h5 = os.path.join(_fdir, 'modis_scene_input.h5')

    os.makedirs(_fdir, exist_ok=True)

    if args.plot_only:
        if not os.path.isfile(_fname_h5):
            sys.exit('ERROR: --plot-only requires an existing h5 file at %s' % _fname_h5)
        print('\n[--plot-only] Skipping download and pre-processing.')
    else:
        # Step 1: download
        print('\n[Step 1] Downloading MODIS products ...')
        fnames = download_modis_products(_date, _extent, _fdir, satellite=_sat, overwrite=_ow)

        # Step 2: pre-process
        print('\n[Step 2] Pre-processing to %s ...' % _fname_h5)
        preprocess_to_h5(fnames, _date, _extent, wavelength, dx, dy, _fname_h5, overwrite=_ow)

    # Step 3: overview figure
    print('\n[Step 3] Generating overview figure ...')
    plot_modis_overview(_fname_h5, wavelength, _name)

    print('\nDone.  Run 07_modis_radiance.py --predata %s to simulate.' % _fname_h5)


if __name__ == '__main__':
    # Running without CLI arguments uses the student control block values above.
    # To process a different scene:
    #   python 06_modis_data_download.py --date 2019-09-02 --extent -108.6 -107.4 37.4 38.6
    main()
