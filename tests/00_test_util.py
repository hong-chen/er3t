"""
EaR³T — Utility tests  (00_test_util.py)
=========================================
Tests for er3t.util: gridding, job allocation, and satellite data download.

Summer school usage
-------------------
Run offline tests (no internet or token required):

    cd $ERTDIR/er3t
    conda activate er3t
    python tests/00_test_util.py

Run download tests (requires EARTHDATA_TOKEN):

    python tests/00_test_util.py --download

You do NOT need to run the download tests now.  The examples (01–05) work
without network access.  Come back here after the examples if you want to
test the full download pipeline.

Set up your token:
    https://konradsebastian.github.io/er3t-edu/install.html#earthdata-token
"""

import os
import sys
import argparse
import datetime

import numpy as np

import matplotlib as mpl
mpl.use('Agg')           # non-interactive backend — no plt.show() blocking
import matplotlib.pyplot as plt

import er3t
from er3t.util import grid_by_extent


# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='EaR³T utility tests')
parser.add_argument('--download', action='store_true',
                    help='Also run download tests (requires EARTHDATA_TOKEN)')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def test_grid_by_dxdy():
    """Test uniform-dx/dy re-gridding of lon/lat data (offline)."""

    extent_lonlat = [125.0, 127.0, 35.0, 37.0]

    lon_1d = np.linspace(extent_lonlat[0], extent_lonlat[1], 201)
    lat_1d = np.linspace(extent_lonlat[2], extent_lonlat[3], 201)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')
    data_2d = lon_2d**2 + lat_2d**2

    lon_out, lat_out, data_out = er3t.util.grid_by_dxdy(lon_2d, lat_2d, data_2d)

    assert lon_out.shape == lat_out.shape == data_out.shape, \
        'Output arrays must have matching shapes'

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(lon_out, lat_out, s=4, c='k', lw=0)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.set_title('grid_by_dxdy — regridded points')
    fig.savefig('test_grid_by_dxdy.png', bbox_inches='tight')
    plt.close(fig)

    print('  ✓  test_grid_by_dxdy passed  →  test_grid_by_dxdy.png')


def test_allocate_jobs():
    """Test job-weight allocation helper (offline)."""

    try:
        weights = np.array([
            14824075, 14483931, 13811633, 12822922, 11540979,
             9995858,  8223786,  6266341,  4349287,   752063,
              676158,   599970,   523397,   446830,   363135,  319635,
        ])
        weights_in = np.tile(weights, 3)
        indices_out = er3t.dev.rearrange_jobs(5, weights_in)
        assert indices_out is not None, 'rearrange_jobs returned None'
        print('  ✓  test_allocate_jobs passed')
    except AttributeError:
        print('  —  test_allocate_jobs skipped (er3t.dev.rearrange_jobs not available in this build)')


def test_download_worldview():
    """
    Download Worldview RGB images via NASA DAAC.

    Requires EARTHDATA_TOKEN to be set.  Downloads four images (MODIS Aqua,
    MODIS Terra, VIIRS SNPP, VIIRS NOAA-20) for a small region over the
    central US.
    """

    from er3t.util.daac import download_worldview_image

    date   = datetime.datetime(2022, 5, 18)
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]

    download_worldview_image(date, extent, fdir_out='tmp-data/00',
                           instrument='modis', satellite='aqua',   fmt='png')
    download_worldview_image(date, extent, fdir_out='tmp-data/00',
                           instrument='modis', satellite='terra',  fmt='png')
    download_worldview_image(date, extent, fdir_out='tmp-data/00',
                           instrument='viirs', satellite='snpp',   fmt='h5')
    download_worldview_image(date, extent, fdir_out='tmp-data/00',
                           instrument='viirs', satellite='noaa20', fmt='h5')

    print('  ✓  test_download_worldview passed  →  files in tmp-data/00/')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    os.makedirs('tmp-data/00', exist_ok=True)

    print()
    print('── Offline tests ────────────────────────────────────────')
    test_grid_by_dxdy()
    test_allocate_jobs()

    print()
    print('── Download tests ───────────────────────────────────────')
    if args.download:
        token = os.environ.get('EARTHDATA_TOKEN', '')
        if not token:
            print('  ✗  EARTHDATA_TOKEN is not set.')
            print('     Set it up at:')
            print('     https://konradsebastian.github.io/er3t-edu/install.html#earthdata-token')
            sys.exit(1)
        test_download_worldview()
    else:
        print('  —  skipped  (re-run with  --download  to test NASA data access)')
        print()
        print('     You can skip this for now.  Examples 01–05 work without a token.')
        print('     Run this test after the examples once you have set up:')
        print('       export EARTHDATA_TOKEN="<your-token>"  # bash/zsh')
        print('       setenv EARTHDATA_TOKEN "<your-token>"  # tcsh/csh')

    print()
