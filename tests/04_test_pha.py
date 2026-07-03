"""
EaR³T — Phase function tests  (04_test_pha.py)
===============================================
Tests for er3t.pre.pha: Mie phase function computation for water clouds.

Summer school usage
-------------------
Run from the er3t root directory:

    cd $ERTDIR/er3t
    conda activate er3t
    python tests/04_test_pha.py

No internet access or Earthdata token required.
Produces PNG figures in tmp-data/04/.
"""

import os
import sys
import datetime
import numpy as np

import matplotlib as mpl
mpl.use('Agg')           # non-interactive backend — no plt.show() blocking
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from er3t.pre.pha import pha_mie_wc as pha_mie


# ─────────────────────────────────────────────────────────────────────────────
def test_pha_mie(wavelength=500.0, fdir='tmp-data/04'):
    """
    Compute Mie phase function and optical properties for liquid water clouds
    at a given wavelength.  Checks that the asymmetry parameter is in the
    expected physical range (0.8–0.95) and saves diagnostic plots.
    """

    os.makedirs(fdir, exist_ok=True)

    pha0 = pha_mie(wavelength=wavelength, overwrite=True)

    g   = pha0.data['asy']['data']
    ssa = pha0.data['ssa']['data']
    ref = pha0.data['ref']['data']

    # Sanity checks
    g_mean = float(g.mean())
    assert 0.80 < g_mean < 0.95, \
        'Mean asymmetry parameter %.3f out of expected range [0.80, 0.95]' % g_mean
    assert np.all(ssa >= 0.999), \
        'Single-scattering albedo below 0.999 — unexpected absorption in water cloud'

    print('  ✓  pha_mie_wc @ %.0f nm  mean g = %.3f  mean SSA = %.4f' % (
          wavelength, g_mean, float(ssa.mean())))

    # ── Phase function plot ───────────────────────────────────────────────
    refs_plot = [1, 5, 10, 15, 20]
    ang       = pha0.data['ang']['data']
    colors    = mpl.cm.viridis(np.linspace(0.0, 1.0, len(refs_plot)))
    patches   = []

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, r in enumerate(refs_plot):
        idx = np.argmin(np.abs(r - ref))
        ax.plot(ang, pha0.data['pha']['data'][:, idx],
                color=colors[i], lw=1.5)
        patches.append(mpatches.Patch(color=colors[i],
                                      label='CER = %d µm' % r))
    ax.set_yscale('log')
    ax.set_xlim((0, 180))
    ax.set_ylim((1e-2, 1e4))
    ax.set_xlabel('Scattering Angle [°]')
    ax.set_ylabel('Phase Function P(Θ)')
    ax.set_title('Mie Phase Function @ %.0f nm' % wavelength)
    ax.legend(handles=patches, loc='upper right')
    fname_pha = '%s/test_pha_mie_phase_%04dnm.png' % (fdir, int(wavelength))
    fig.savefig(fname_pha, bbox_inches='tight')
    plt.close(fig)

    # ── Asymmetry parameter vs. CER ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ref, g,             color='k',    lw=2.0, label='g (full)')
    ax.plot(ref, pha0.data['asy_']['data'],
                                color='gray', lw=2.0, label="g' (δ-fit)")
    for i, r in enumerate(refs_plot):
        ax.axvline(r, ls='--', color=colors[i], lw=1.5, alpha=0.8)
    ax.set_xlim((0, 25))
    ax.set_ylim((0.75, 0.95))
    ax.set_xlabel('Cloud Effective Radius [µm]')
    ax.set_ylabel('Asymmetry Parameter g')
    ax.set_title('Asymmetry Parameter @ %.0f nm' % wavelength)
    ax.legend()
    fname_asy = '%s/test_pha_mie_asymmetry_%04dnm.png' % (fdir, int(wavelength))
    fig.savefig(fname_asy, bbox_inches='tight')
    plt.close(fig)

    print('  Plots: %s  %s' % (fname_pha, fname_asy))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    fdir = 'tmp-data/04'
    print()
    print('── Mie phase function tests ─────────────────────────────')
    test_pha_mie(wavelength=500.0, fdir=fdir)
    test_pha_mie(wavelength=650.0, fdir=fdir)
    print()
    print('All phase function tests passed.')
    print()
