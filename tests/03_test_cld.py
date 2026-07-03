"""
EaR³T — Cloud property tests  (03_test_cld.py)
===============================================
Tests for er3t.pre.cld: LES cloud ingestion (offline) and MODIS satellite
cloud data download (requires Earthdata token).

Summer school usage
-------------------
Run the offline LES test (no internet needed):

    cd $ERTDIR/er3t
    conda activate er3t
    python tests/03_test_cld.py

Run the MODIS download test (requires EARTHDATA_TOKEN):

    python tests/03_test_cld.py --download

The LES cloud test uses the example data bundled with er3t (data/les.nc).
You do NOT need to run the download test before examples 01–05 — those use
the same LES data file.  The download test exercises the full satellite-data
retrieval pipeline; run it once you have your Earthdata token set up.

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

from er3t.pre.atm import atm_atmmod
from er3t.pre.cld import cld_les
import er3t.common


# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='EaR³T cloud property tests')
parser.add_argument('--download', action='store_true',
                    help='Also run MODIS download test (requires EARTHDATA_TOKEN)')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def test_cld_les(fdir):
    """
    Test LES cloud ingestion using the bundled example LES NetCDF file.
    Offline — no internet or token needed.
    """

    fname_nc  = '%s/data/les.nc' % er3t.common.fdir_examples
    fname_les = '%s/les.pk' % fdir

    if not os.path.exists(fname_nc):
        print('  ✗  LES data file not found: %s' % fname_nc)
        print('     Run install-examples.sh first (Step 2 of install guide).')
        sys.exit(1)

    dz  = 0.2
    dnz = int(dz / 0.04)
    cld0 = cld_les(fname_nc=fname_nc, fname=fname_les,
                   coarsen=[1, 1, dnz], overwrite=True)

    ext_2d = np.sum(cld0.lay['extinction']['data'], axis=-1)
    assert ext_2d.min() >= 0.0, 'Extinction must be non-negative'

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ext_2d.T, origin='lower', cmap='Blues')
    plt.colorbar(im, ax=ax, label='Column extinction [m⁻¹]')
    ax.set_xlabel('X index')
    ax.set_ylabel('Y index')
    ax.set_title('LES cloud column extinction')
    fig.savefig('%s/test_cld_les.png' % fdir, bbox_inches='tight')
    plt.close(fig)

    print('  ✓  test_cld_les passed  →  %s/test_cld_les.png' % fdir)


def test_download_modis(fdir):
    """
    Download MODIS Level-2 cloud products for a sample date/scene.
    Requires EARTHDATA_TOKEN to be set.
    """

    from er3t.util.modis import download_modis_https

    date = datetime.datetime(2017, 8, 25)
    dataset_tags = ['61/MYD02QKM', '61/MYD03', '61/MYD06_L2']
    filename_tag = '.2035.'

    print('  Downloading MODIS files for %s ...' % date.strftime('%Y-%m-%d'))
    for dataset_tag in dataset_tags:
        download_modis_https(date, dataset_tag, filename_tag,
                             day_interval=1, fdir_out='%s/modis' % fdir, run=True)
    print('  ✓  test_download_modis passed  →  files in %s/modis/' % fdir)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    fdir = os.path.abspath('tmp-data/03')
    os.makedirs(fdir, exist_ok=True)

    print()
    print('── Offline tests ────────────────────────────────────────')
    test_cld_les(fdir)

    print()
    print('── Download tests ───────────────────────────────────────')
    if args.download:
        token = os.environ.get('EARTHDATA_TOKEN', '')
        if not token:
            print('  ✗  EARTHDATA_TOKEN is not set.')
            print('     Set it up at:')
            print('     https://konradsebastian.github.io/er3t-edu/install.html#earthdata-token')
            sys.exit(1)
        test_download_modis(fdir)
    else:
        print('  —  skipped  (re-run with  --download  to test MODIS data access)')
        print()
        print('     You can skip this for now.  Examples 01–05 work without a token.')

    print()
