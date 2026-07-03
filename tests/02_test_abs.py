"""
EaR³T — Absorption tests  (02_test_abs.py)
===========================================
Tests for er3t.pre.abs using the REPTRAN gas absorption database.

Summer school usage
-------------------
Run from the er3t root directory:

    cd $ERTDIR/er3t
    conda activate er3t
    python tests/02_test_abs.py

No internet access or Earthdata token required.
Requires the REPTRAN database (installed by install.sh in Step 2).
"""

import os
import sys
import numpy as np

import matplotlib as mpl
mpl.use('Agg')           # non-interactive backend — no plt.show() blocking
import matplotlib.pyplot as plt

import er3t


def test_abs_reptran(fdir='tmp-data/02'):
    """
    Test REPTRAN gas absorption database via er3t.pre.abs.abs_rep.

    Computes the absorption spectrum at four representative wavelengths
    spanning the solar range (300–2400 nm) and saves a profile plot.
    """

    os.makedirs(fdir, exist_ok=True)

    # US standard atmosphere
    levels   = np.linspace(0.0, 20.0, 41)
    fname_atm = '%s/atm.pk' % fdir
    atm0 = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)
    alt0 = atm0.lay['altitude']['data']

    # Test four wavelengths: UV edge, visible, near-IR O2-A, near-IR
    test_wavelengths = [350.0, 500.0, 760.0, 1600.0]

    print()
    print('  Testing REPTRAN at %d wavelengths ...' % len(test_wavelengths))

    results = {}
    for wvl in test_wavelengths:
        try:
            abs1 = er3t.pre.abs.abs_rep(wavelength=wvl, target='coarse', atm_obj=atm0)
            # column-weighted absorption coefficient at each layer
            coef = np.array([
                (abs1.coef['abso_coef']['data'][j, :] * abs1.coef['weight']['data']).sum()
                for j in range(alt0.size)
            ])
            results[wvl] = coef
            print('    ✓  %.0f nm — peak abs. coef = %.3e m⁻¹' % (wvl, coef.max()))
        except Exception as e:
            print('    ✗  %.0f nm — %s' % (wvl, e))
            sys.exit(1)

    # Save absorption profile plot
    fig, ax = plt.subplots(figsize=(6, 7))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    for i, (wvl, coef) in enumerate(results.items()):
        ax.plot(coef, alt0, color=colors[i], lw=1.5, label='%.0f nm' % wvl)
    ax.set_xlabel('Absorption Coefficient [m⁻¹]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('REPTRAN Absorption Profiles')
    ax.legend()
    fig.savefig('%s/test_abs_reptran.png' % fdir, bbox_inches='tight')
    plt.close(fig)
    print()
    print('  ✓  test_abs_reptran passed  →  %s/test_abs_reptran.png' % fdir)


if __name__ == '__main__':

    test_abs_reptran()
