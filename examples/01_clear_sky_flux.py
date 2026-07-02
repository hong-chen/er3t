"""
EaR³T Example 01 — Clear-Sky Flux Profile
==========================================

What this script does
---------------------
Simulates the vertical profile of solar irradiance (flux) through a clear
atmosphere using the MCARaTS 3D Monte Carlo solver.  No clouds, no aerosol —
just the atmosphere and gas absorption.

Outputs
-------
* HDF5 results in  tmp-data/01_clear_sky_flux/
* PNG plot         01_clear_sky_flux-example_01_flux_clear_sky_<solver>.png
  Vertical profiles of upwelling, total downwelling, direct downwelling,
  and diffuse downwelling flux (W m⁻² nm⁻¹).  Shaded bands show ±1σ Monte
  Carlo noise — wide bands mean you need more photons.

Key physics
-----------
Even without clouds the atmosphere scatters and absorbs solar photons.
At visible wavelengths (e.g. 650 nm) Rayleigh scattering is weak, so the
direct beam barely changes with altitude.  Try wavelength=400 nm to see
much stronger Rayleigh scattering and a large diffuse component.

Noise visualisation
-------------------
The simulation runs Nrun independent Monte Carlo batches.  The spread between
them (standard deviation, shown as shaded bands) is purely statistical noise
from having too few photons — it is NOT a physical uncertainty.  Increase
photons until the bands are barely visible relative to the mean profiles.
Rule of thumb: ±1σ < 1% of the mean = good quality result.

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
import matplotlib.patches as mpatches
from matplotlib import rcParams

import er3t


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     STUDENT CONTROL BLOCK                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  Edit these values to explore how the simulation changes.              ║
# ╠══════════════════════════════════════════════════════════════════════════╣

wavelength = 650.0      # Wavelength in nm.
                        # Try: 400 (strong Rayleigh), 500, 650 (visible), 860, 1640

photons    = 1e5        # Monte Carlo photons per spectral bin.
                        # *** Start here to see the noise bands clearly! ***
                        # Then increase step by step to watch them shrink:
                        #   1e4  → very wide bands (almost useless result)
                        #   1e5  → clearly visible bands (good starting point)
                        #   1e6  → narrow bands, Max noise ~0.03%
                        #   1e7  → very narrow, ~0.01%
                        #   1e8  → production quality, barely visible
                        # The shaded ±1σ regions in the output plot are the
                        # noise — watch them shrink as you increase photons.

Nrun       = 3          # Number of independent MC runs (used to estimate noise).
                        # Each run uses a different random seed.
                        # More runs → better noise estimate, but Nrun × longer runtime.
                        # Keep at 3 for testing; 5–10 for publishable uncertainty bars.

solver     = '3D'       # Radiative transfer solver mode.
                        #   '3D'  → full 3D Monte Carlo (MCARaTS)
                        #   'IPA' → Independent Pixel Approximation (1D per column)
                        # For this clear-sky 1D atmosphere both give identical results.
                        # The difference shows up only with inhomogeneous clouds (Ex. 02+).

surface_albedo = 0.05   # Surface reflectance (Lambertian), 0–1.
                        # Try: 0.05 (ocean), 0.15 (vegetation), 0.5 (sand/desert), 0.9 (snow)
                        # Upwelling flux at the surface increases linearly with albedo.

solar_zenith_angle  = 30.0   # Solar zenith angle in degrees (0° = sun overhead, 90° = horizon).
                              # Try: 0, 30, 45, 60, 75.  Higher SZA → longer path, more scattering.

solar_azimuth_angle = 0.0    # Solar azimuth in degrees.
                              # Does NOT affect this 1D clear-sky result — the atmosphere
                              # is horizontally uniform so no direction is special.
                              # (It matters in 3D cloud simulations with asymmetric geometry.)

z_top    = 20.0         # Top of atmosphere in km.  Default 20 km covers the stratosphere.
                        # Reducing to 10 km speeds up the run (fewer layers) but misses
                        # stratospheric ozone absorption.

n_levels = 21           # Number of altitude interfaces (= number of layers + 1).
                        # Levels are evenly spaced from 0 to z_top.
                        # 21 levels → 1 km resolution.  Try 41 for 0.5 km resolution.

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Internal configuration (do not need to change) ────────────────────────
name_tag = '01_clear_sky_flux'
fdir0    = er3t.common.fdir_examples
Ncpu     = max(1, multiprocessing.cpu_count() - 2)   # leave 2 CPUs free
rcParams['font.size'] = 14
# ──────────────────────────────────────────────────────────────────────────


def example_01_flux_clear_sky(
        wavelength=wavelength,
        solver=solver,
        overwrite=True,
        plot=True
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interfaces in km
    levels    = np.linspace(0.0, z_top, n_levels)

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % fdir

    # absorption object (REPTRAN — publicly distributed gas absorption database)
    abs0      = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, target='medium', overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mcarats 1d and 3d "atmosphere", can represent aerosol, cloud, atmosphere
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # homogeneous 1d mcarats "atmosphere"
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    # data can be accessed at
    #     atm1d0.nml[ig]['Atm_zgrd0']['data']
    #     atm1d0.nml[ig]['Atm_wkd0']['data']
    #     atm1d0.nml[ig]['Atm_mtprof']['data']
    #     atm1d0.nml[ig]['Atm_tmp1d']['data']
    #     atm1d0.nml[ig]['Atm_nkd']['data']
    #     atm1d0.nml[ig]['Atm_nz']['data']
    #     atm1d0.nml[ig]['Atm_ext1d']['data']
    #     atm1d0.nml[ig]['Atm_abs1d']['data']
    #     atm1d0.nml[ig]['Atm_omg1d']['data']
    #     atm1d0.nml[ig]['Atm_apf1d']['data']

    # make them into python list, can contain more than one 1d or 3d mcarats "atmosphere"
    atm_1ds   = [atm1d0]
    atm_3ds   = []
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define mcarats object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # run mcarats
    mca0 = er3t.rtm.mca.mcarats_ng(
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            fdir='%s/%4.4d/flux_%s' % (fdir, wavelength, solver.lower()),
            target='flux',
            Nrun=Nrun,
            surface_albedo=surface_albedo,
            solar_zenith_angle=solar_zenith_angle,
            solar_azimuth_angle=solar_azimuth_angle,
            photons=photons,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=overwrite
            )

    # data can be accessed at
    #     mca0.Nrun
    #     mca0.Ng
    #     mca0.nml         (Nrun, Ng), e.g., mca0.nml[0][0], namelist for the first g of the first run
    #     mca0.fnames_inp  (Nrun, Ng), e.g., mca0.fnames_inp[0][0], input file name
    #     mca0.fnames_out  (Nrun, Ng), e.g., mca0.fnames_out[0][0], output file name
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # collect all Nrun individual runs, then compute mean and std manually
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # mode='all' returns every run stacked in the last dimension (shape: Nz × Nrun)
    fname_h5_all = '%s/mca-out-flux-%s_%s_all.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0_all = er3t.rtm.mca.mca_out_ng(fname_h5_all, mca_obj=mca0, abs_obj=abs0, mode='all', squeeze=True, verbose=True, overwrite=overwrite)

    # compute mean and std across runs (last axis)
    # shape of each array: (Nz, Nrun) → mean/std shape: (Nz,)
    f_up_mean     = np.mean(out0_all.data['f_up']['data'],           axis=-1)
    f_dn_mean     = np.mean(out0_all.data['f_down']['data'],         axis=-1)
    f_dir_mean    = np.mean(out0_all.data['f_down_direct']['data'],  axis=-1)
    f_dif_mean    = np.mean(out0_all.data['f_down_diffuse']['data'], axis=-1)

    f_up_std      = np.std(out0_all.data['f_up']['data'],           axis=-1, ddof=1)
    f_dn_std      = np.std(out0_all.data['f_down']['data'],         axis=-1, ddof=1)
    f_dir_std     = np.std(out0_all.data['f_down_direct']['data'],  axis=-1, ddof=1)
    f_dif_std     = np.std(out0_all.data['f_down_diffuse']['data'], axis=-1, ddof=1)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot — mean profiles with ±1σ noise bands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        z   = atm0.lev['altitude']['data']   # altitude grid for plotting

        # report noise level for each component
        def noise_pct(mean, std):
            peak = np.abs(mean).max()
            return (std.max() / peak * 100.0) if peak > 0 else 0.0

        print('')
        print('─── Monte Carlo noise (std / mean, larger = need more photons) ───')
        print('  f_down  (total)  : %7.3f%%' % noise_pct(f_dn_mean,  f_dn_std))
        print('  f_down  (direct) : %7.3f%%' % noise_pct(f_dir_mean, f_dir_std))
        print('  f_down  (diffuse): %7.3f%%' % noise_pct(f_dif_mean, f_dif_std))
        print('  f_up             : %7.3f%%' % noise_pct(f_up_mean,  f_up_std))
        print('  target for science: < 1%%   |  acceptable exploration: < 5%%')
        print('')

        fig, ax1 = plt.subplots(figsize=(5, 8))

        colors = {'up': 'red', 'down': 'blue', 'direct': 'green', 'diffuse': 'deeppink'}

        # mean lines
        ax1.plot(f_up_mean,  z, color=colors['up'],     lw=1.5)
        ax1.plot(f_dn_mean,  z, color=colors['down'],   lw=1.5)
        ax1.plot(f_dir_mean, z, color=colors['direct'], lw=1.5)
        ax1.plot(f_dif_mean, z, color=colors['diffuse'],lw=1.5)

        # ±1σ shaded noise bands
        ax1.fill_betweenx(z, f_up_mean  - f_up_std,  f_up_mean  + f_up_std,  color=colors['up'],     alpha=0.25)
        ax1.fill_betweenx(z, f_dn_mean  - f_dn_std,  f_dn_mean  + f_dn_std,  color=colors['down'],   alpha=0.25)
        ax1.fill_betweenx(z, f_dir_mean - f_dir_std, f_dir_mean + f_dir_std, color=colors['direct'], alpha=0.25)
        ax1.fill_betweenx(z, f_dif_mean - f_dif_std, f_dif_mean + f_dif_std, color=colors['diffuse'],alpha=0.25)

        ax1.set_xlabel('Flux [$\\mathrm{W\\,m^{-2}\\,nm^{-1}}$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_ylim((0.0, z_top))
        ax1.set_xlim(left=0.0)

        patches_legend = [
            mpatches.Patch(color=colors['up'],     label='Up'),
            mpatches.Patch(color=colors['down'],   label='Down (total)'),
            mpatches.Patch(color=colors['direct'], label='Down (direct)'),
            mpatches.Patch(color=colors['diffuse'],label='Down (diffuse)'),
        ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=11)

        ax1.set_title(
            'Clear Sky (%s), SZA=%.0f°\n'
            '$\\lambda$=%.0f nm,  albedo=%.2f,  photons=%.0e'
            % (solver, solar_zenith_angle, wavelength, surface_albedo, photons)
        )

        plt.tight_layout()
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        print('Plot saved: %s' % fname_png)
    #╰────────────────────────────────────────────────────────────────────────────╯#




if __name__ == '__main__':

    print('=' * 60)
    print('EaR³T Example 01 — Clear-Sky Flux')
    print('  wavelength          = %.1f nm'  % wavelength)
    print('  solver              = %s'        % solver)
    print('  photons             = %.0e  (increase to reduce noise bands)'  % photons)
    print('  Nrun                = %d'        % Nrun)
    print('  surface_albedo      = %.2f'      % surface_albedo)
    print('  solar_zenith_angle  = %.1f deg'  % solar_zenith_angle)
    print('  solar_azimuth_angle = %.1f deg'  % solar_azimuth_angle)
    print('  z_top               = %.1f km'   % z_top)
    print('  n_levels            = %d'        % n_levels)
    print('  Ncpu                = %d'        % Ncpu)
    print('=' * 60)
    print()

    example_01_flux_clear_sky(
            wavelength=wavelength,
            solver=solver,
            )
