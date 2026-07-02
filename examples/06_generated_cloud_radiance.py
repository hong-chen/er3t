"""
EaR³T Example 06 — Radiance from a Randomly Generated Cloud Field
==================================================================

What this script does
---------------------
Generates a synthetic cloud field from scratch using EaR³T's built-in
hemispheric cloud generator (cld_gen_hem), then simulates nadir radiance at
satellite altitude.  No external data file needed beyond the er3t core data.

Outputs
-------
* HDF5 results in  tmp-data/06_generated_cloud_radiance/
* PNG plot         06_generated_cloud_radiance-..._<solver>.png
  Three-panel figure:
    (1) 3D view of cloud-top height
    (2) 2D map of cloud optical thickness (COT)
    (3) 2D map of simulated nadir radiance

Key physics / "aha" moments
----------------------------
Here you control the cloud geometry directly.  Try different cloud fractions
and size distributions to see how cloud morphology affects the radiance field.
Notice how isolated cumulus clouds produce strong 3D effects (bright halos,
shadowed surroundings) that would be missed by IPA.

The cloud generator creates hemispheroidal (dome-shaped) clouds with
configurable:
  cloud_frac_tgt — fraction of columns that are cloudy (0–1)
  radii          — list of cloud radii [km]
  weights        — probability of each radius (must sum to 1)
  altitude       — vertical extent of cloud layer [km array]

Student controls — edit the block below
-----------------------------------------
"""

import os
import sys
import h5py
import time
import numpy as np
import datetime
import multiprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

import er3t


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     STUDENT CONTROL BLOCK                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣

wavelength = 650.0      # Wavelength in nm.  Try: 400, 500, 650, 860, 1640

photons    = 1e6        # Monte Carlo photons per spectral bin.
                        #   1e6  → fast, lower quality image
                        #   1e7  → noticeably cleaner (recommended for this example)
                        #   1e8  → production quality

solver     = '3D'       # '3D' or 'IPA'  — compare to see the 3D bias signature!

# ── Cloud generator parameters ─────────────────────────────────────────────
# Try changing these to create different cloud scenes:
cloud_frac_tgt = 0.2        # Target cloud fraction (0–1).  Try: 0.05, 0.2, 0.5
cloud_radii    = [1.0, 2.0, 4.0]   # Cloud radii [km].  Larger = fewer big clouds
cloud_weights  = [0.6, 0.3, 0.1]   # Probability of each radius (must sum to 1)
cloud_alt_base = 2.0        # Cloud base altitude [km]
cloud_alt_top  = 5.0        # Cloud top altitude [km]  (dz=0.2 km layers)
w2h_ratio      = 2.0        # Width-to-height ratio.  Large = flat (Sc), small = tall (Cu)
# ──────────────────────────────────────────────────────────────────────────

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Internal configuration (do not need to change) ────────────────────────
name_tag = '06_generated_cloud_radiance'
fdir0    = er3t.common.fdir_examples
Ncpu     = max(1, multiprocessing.cpu_count() - 2)
rcParams['font.size'] = 14
# ──────────────────────────────────────────────────────────────────────────


def example_06_rad_cld_gen_hem(
        wavelength=wavelength,
        solver=solver,
        overwrite=True,
        plot=True
        ):

    """
    Radiance field for a randomly generated hemispheroidal cloud field
    at satellite altitude of 705 km.  Mie phase function is used.
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object (higher vertical resolution than other examples)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    levels = np.linspace(0.0, 20.0, 101)    # 0.2 km resolution
    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_abs = '%s/abs.pk' % fdir
    abs0      = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, target='medium', overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object using the cloud generator
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # <radii>          : radius of each cloud type in km
    # <weights>        : probability of each radius (sum to 1)
    # <altitude>       : vertical levels of the cloud layer in km
    # <cloud_frac_tgt> : target fraction of cloudy pixels
    # <w2h_ratio>      : width-to-height ratio (larger = flatter clouds)
    # <min_dist>       : minimum inter-cloud distance [km]
    # <overlap>        : whether clouds can overlap
    fname_cld = '%s/cld.pk' % fdir
    cld0 = er3t.pre.cld.cld_gen_hem(
            fname=fname_cld,
            Nx=200,
            Ny=200,
            dx=0.2,
            dy=0.2,
            radii=cloud_radii,
            weights=cloud_weights,
            altitude=np.arange(cloud_alt_base, cloud_alt_top + 0.01, 0.2),
            cloud_frac_tgt=cloud_frac_tgt,
            w2h_ratio=w2h_ratio,
            min_dist=0.2,
            overlap=False,
            overwrite=overwrite
            )

    print('Cloud fraction achieved: %.3f' % cld0.cloud_frac)

    # After run, cld0 contains:
    #   .cloud_frac           : actual achieved cloud fraction
    #   .lev['cot_2d']['data']: column-integrated optical thickness map (Nx, Ny)
    #   .lev['cth_2d']['data']: cloud-top height map (Nx, Ny)
    #   .x_3d, .y_3d          : 3D coordinate arrays
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mca_sca object — Mie phase function for water clouds
    #╭────────────────────────────────────────────────────────────────────────────╮#
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength)
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mcarats 1d and 3d "atmosphere"
    #╭────────────────────────────────────────────────────────────────────────────╮#
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm3d0  = er3t.rtm.mca.mca_atm_3d(fname='%s/mca_atm_3d.bin' % fdir, cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, overwrite=overwrite)

    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mcarats object and run (radiance mode)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    mca0 = er3t.rtm.mca.mcarats_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            target='radiance',
            surface_albedo=0.2,
            sca=sca,
            solar_zenith_angle=29.162360459281544,
            solar_azimuth_angle=-63.16777636586792,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            sensor_altitude=705000.0,
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            photons=photons,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=overwrite
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # collect output
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = '%s/mca-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.mca.mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(16, 5.0))
        cmap = mpl.colormaps['jet'].copy()

        ax1 = fig.add_subplot(131, projection='3d')
        cs = ax1.plot_surface(cld0.x_3d[:, :, 0], cld0.y_3d[:, :, 0], cld0.lev['cth_2d']['data'], cmap=cmap, alpha=0.8, antialiased=False)
        ax1.set_zlim((0, 10))
        ax1.set_xlabel('X [km]')
        ax1.set_ylabel('Y [km]')
        ax1.set_zlabel('Z [km]')
        ax1.set_title('Cloud Top Height (3D View)\ncf = %.2f' % cld0.cloud_frac)

        ax2 = fig.add_subplot(132)
        cs = ax2.imshow(cld0.lev['cot_2d']['data'].T, cmap=cmap, origin='lower', vmin=0.0, vmax=80.0)
        ax2.set_xlabel('X Index')
        ax2.set_ylabel('Y Index')
        ax2.set_title('Cloud Optical Thickness')
        plt.colorbar(cs, ax=ax2, label='COT')

        ax3 = fig.add_subplot(133)
        cs = ax3.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
        ax3.set_xlabel('X Index')
        ax3.set_ylabel('Y Index')
        ax3.set_title('Radiance at %.2f nm (%s Mode)' % (wavelength, solver))

        plt.subplots_adjust(wspace=0.4)
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        print('Plot saved: %s' % fname_png)
    #╰────────────────────────────────────────────────────────────────────────────╯#




if __name__ == '__main__':

    print('=' * 60)
    print('EaR³T Example 06 — Generated Cloud Radiance')
    print('  wavelength     = %.1f nm' % wavelength)
    print('  solver         = %s' % solver)
    print('  photons        = %.0e' % photons)
    print('  Ncpu           = %d' % Ncpu)
    print('  cloud_frac_tgt = %.2f' % cloud_frac_tgt)
    print('  radii          = %s km  (weights: %s)' % (cloud_radii, cloud_weights))
    print('  altitude       = %.1f – %.1f km' % (cloud_alt_base, cloud_alt_top))
    print('  w2h_ratio      = %.1f' % w2h_ratio)
    print('=' * 60)
    print()
    print('Tip: change cloud_frac_tgt and w2h_ratio to create different cloud scenes.')
    print('     Run with solver="3D" and solver="IPA" to see the 3D effect on isolated clouds.')
    print()

    example_06_rad_cld_gen_hem(
            wavelength=wavelength,
            solver=solver,
            )
