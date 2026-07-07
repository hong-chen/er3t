"""
EaR³T Example 07 — MODIS Radiance Self-Consistency Check
=========================================================

What this script does
---------------------
Loads the pre-processed MODIS data file produced by Example 06 and runs a
3D Monte Carlo radiance simulation with MCARaTS.  The simulated radiance is
then compared pixel-by-pixel with what MODIS actually measured, producing a
radiance map pair and a density scatter plot.

The comparison reproduces Appendix 2 of Chen et al. (2022), "3D Radiative
Transfer in Cloudy Atmospheres", Atmos. Meas. Tech., 15, 1813–1836.

Scientific story
----------------
The retrieved MODIS cloud state (optical thickness, effective radius, cloud
top height from MYD06_L2) is used as input to the 3D RT model.  If the
physics in the model and the retrieval algorithm are self-consistent, the
simulated and measured radiances should match closely (points on or near the
1:1 line in the scatter plot).  Deviations reveal where 1D retrieval
assumptions break down: cloud edges, shadows, and strongly heterogeneous
cloud fields show the largest offsets between 3D simulation and observation.

plot_only mode
--------------
Set  plot_only = True  to skip the simulation and regenerate figures from
existing output files (e.g., after adjusting colour scales).

Outputs
-------
* 07_modis_radiance-comparison_<wl>nm.png  — 3-panel figure:
    Left   — MODIS measured radiance
    Centre — EaR³T simulated 3D radiance
    Right  — density scatter: measured vs simulated (log-scale density)
* data/07_modis_radiance/post-data.h5  — saves simulation results

Student controls — edit the block below
----------------------------------------
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FixedLocator
import h5py

import er3t


# ============================================================
#  STUDENT CONTROL BLOCK
# ============================================================
# Path to modis_scene_input.h5 produced by 06_modis_data_download.py
# (default: same date/region as that script's defaults)
fname_predata  = os.path.join(
    'data', '06_modis_data_download',
    '2019-09-02(-108.60,-107.40,37.40,38.60)',
    'modis_scene_input.h5')

wavelength     = 650      # nm — must match what 06 used
photons        = 1e4      # photons per run; increase for better statistics
                          # (original paper used 1e8; expect noisier maps at 1e5)
Nrun           = 16        # independent MC runs; std dev across runs = noise estimate
                          # higher Nrun → more reliable noise estimate, total cost ∝ Nrun
ncpu           = 8        # number of CPU cores for MCARaTS
plot_only      = False    # True = skip simulation, regenerate figures only
overwrite      = False    # True = re-run simulation even if output exists
# ============================================================


_NAME  = os.path.splitext(os.path.basename(__file__))[0]
_FDIR  = os.path.join('data', _NAME)


# ─────────────────────────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────────────────────────

def run_simulation(fname_predata, wvl, photons, ncpu, fdir_out, Nrun=3, overwrite=False):
    """Run MCARaTS 3D radiance simulation using MODIS-derived cloud/surface inputs.

    The simulation geometry (SZA, SAA, VZA, VAA) and cloud field (COT, CER,
    CTH) are read from modis_scene_input.h5.  The gas absorption is computed with
    abs_rep (REPTRAN correlated-k, MODIS Aqua Band 1 parameterisation).
    Surface BRDF is the Ross-Li kernel model from MCD43A1.

    Parameters
    ----------
    fname_predata : str   — path to modis_scene_input.h5 from 06_modis_data_download.py
    wvl           : float — wavelength in nm (650 or 860)
    photons       : float — number of photons per MCARaTS run
    ncpu          : int   — number of CPU cores for MCARaTS
    fdir_out      : str   — directory for temporary MCARaTS files
    Nrun          : int   — number of independent MC runs; std dev across runs = noise estimate
    overwrite     : bool  — re-run even if output exists

    Returns
    -------
    fname_out : str — path to MCARaTS output HDF5 file
    """
    fname_out = os.path.join(fdir_out, 'mca-out-rad-modis-3d_%.4fnm.h5' % wvl)
    if os.path.exists(fname_out) and not overwrite:
        print('Simulation output already exists — skipping (set overwrite=True to re-run).')
        return fname_out

    os.makedirs(fdir_out, exist_ok=True)

    f = h5py.File(fname_predata, 'r')
    extent   = f['extent'][...]
    cot_2d   = f['mod/cld/cot_ipa'][...]
    cer_2d   = f['mod/cld/cer_ipa'][...]
    cth_2d   = f['mod/cld/cth_ipa'][...]
    sza      = float(f['mod/geo/sza'][...].mean())
    saa      = float(f['mod/geo/saa'][...].mean())
    vza      = float(f['mod/geo/vza'][...].mean())
    vaa      = float(f['mod/geo/vaa'][...].mean())
    fiso     = f['mod/sfc/fiso_43_%4.4d' % wvl][...]
    fvol     = f['mod/sfc/fvol_43_%4.4d' % wvl][...]
    fgeo     = f['mod/sfc/fgeo_43_%4.4d' % wvl][...]
    f.close()

    dx    = 250.0   # m, must match 07a
    dy    = 250.0
    Nx, Ny = cot_2d.shape
    extent_xy = [0.0, dx * Nx / 1000.0, 0.0, dy * Ny / 1000.0]   # km

    # ── Atmosphere ───────────────────────────────────────────
    levels    = np.arange(0.0, 20.1, 0.5)
    fname_atm = os.path.join(fdir_out, 'atm.pk')
    fname_prof = os.path.join(er3t.common.fdir_data_atmmod, 'afglus.dat')
    atm0      = er3t.pre.atm.atm_atmmod(
        levels=levels, fname=fname_atm,
        fname_atmmod=fname_prof, overwrite=overwrite)

    # ── Gas absorption (REPTRAN, MODIS Aqua Band 1) ──────────
    fname_abs = os.path.join(fdir_out, 'abs.pk')
    abs0      = er3t.pre.abs.abs_rep(
        wavelength=wvl, fname=fname_abs,
        target='modis', band_name='modis_aqua_b01',
        atm_obj=atm0, overwrite=overwrite)

    # ── Surface (Ross-Li BRDF) ────────────────────────────────
    coef_dict = {'fiso': fiso, 'fvol': fvol, 'fgeo': fgeo,
                 'dx': dx / 1000.0, 'dy': dy / 1000.0}   # km
    fname_sfc = os.path.join(fdir_out, 'sfc.pk')
    sfc0      = er3t.pre.sfc.sfc_2d_gen(
        sfc_dict=coef_dict, fname=fname_sfc, overwrite=overwrite)
    sfc_2d    = er3t.rtm.mca.mca_sfc_2d(
        atm_obj=atm0, sfc_obj=sfc0,
        fname=os.path.join(fdir_out, 'mca_sfc_2d.bin'),
        overwrite=overwrite)

    # ── Cloud field ───────────────────────────────────────────
    # Cloud geometrical thickness: 1 km for low clouds; high clouds
    # (CTH > 4 km) have base at 3 km.
    cgt_2d = np.zeros_like(cth_2d)
    cgt_2d[cth_2d > 0.0] = 1.0
    cgt_2d[cth_2d > 4.0] = cth_2d[cth_2d > 4.0] - 3.0

    fname_cld = os.path.join(fdir_out, 'cld.pk')
    cld0      = er3t.pre.cld.cld_gen_cop(
        fname=fname_cld,
        cot=cot_2d, cer=cer_2d, cth=cth_2d, cgt=cgt_2d,
        dz=atm0.lay['thickness']['data'][0],
        extent_xy=extent_xy,
        atm_obj=atm0,
        overwrite=overwrite)

    # ── Phase function ────────────────────────────────────────
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wvl, overwrite=overwrite)
    sca  = er3t.rtm.mca.mca_sca(
        pha_obj=pha0,
        fname=os.path.join(fdir_out, 'mca_sca.bin'),
        overwrite=overwrite)

    # ── MCARaTS inputs ────────────────────────────────────────
    atm3d0  = er3t.rtm.mca.mca_atm_3d(
        cld_obj=cld0, atm_obj=atm0, pha_obj=pha0,
        fname=os.path.join(fdir_out, 'mca_atm_3d.bin'))
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

    # ── Run MCARaTS ───────────────────────────────────────────
    mca0 = er3t.rtm.mca.mcarats_ng(
        date=datetime.datetime(2019, 9, 2),   # used for Earth–Sun distance
        atm_1ds=[atm1d0],
        atm_3ds=[atm3d0],
        surface=sfc_2d,
        sca=sca,
        Ng=abs0.Ng,
        target='radiance',
        solar_zenith_angle   = sza,
        solar_azimuth_angle  = saa,
        sensor_zenith_angle  = vza,
        sensor_azimuth_angle = vaa,
        fdir=os.path.join(fdir_out, '%.4fnm' % wvl),
        Nrun=Nrun,
        weights=abs0.coef['weight']['data'],
        photons=photons,
        solver='3D',
        Ncpu=ncpu,
        mp_mode='py',
        overwrite=True)   # always run — the early-return above handles the skip-if-done logic

    # ── Save output ───────────────────────────────────────────
    out0 = er3t.rtm.mca.mca_out_ng(
        fname=fname_out,
        mca_obj=mca0, abs_obj=abs0,
        mode='mean', squeeze=True, verbose=True,
        overwrite=overwrite)

    return fname_out


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_comparison(fname_predata, fname_sim, wvl, name_tag, fdir_out,
                    photons=None, nrun=None, elapsed=None):
    """Three-panel radiance comparison figure.

    Left   — MODIS measured radiance map
    Centre — EaR³T simulated 3D radiance map
    Right  — log-density scatter: measured vs simulated
              Points near the 1:1 line indicate self-consistency.
    """
    # Load observations
    f = h5py.File(fname_predata, 'r')
    extent  = f['extent'][...]
    lon     = f['lon'][...]
    lat     = f['lat'][...]
    rad_obs = f['mod/rad/rad_%4.4d' % wvl][...]
    sza     = float(f['mod/geo/sza'][...].mean())
    saa     = float(f['mod/geo/saa'][...].mean())
    vza     = float(f['mod/geo/vza'][...].mean())
    vaa     = float(f['mod/geo/vaa'][...].mean())
    f.close()

    # Load simulation
    f = h5py.File(fname_sim, 'r')
    rad_sim     = f['mean/rad'][...]
    rad_sim_std = f['mean/rad_std'][...]
    f.close()

    # Save comparison data
    os.makedirs(fdir_out, exist_ok=True)
    fname_post = os.path.join(fdir_out, 'post-data.h5')
    with h5py.File(fname_post, 'w') as fp:
        fp['wvl']         = wvl
        fp['lon']         = lon
        fp['lat']         = lat
        fp['extent']      = extent
        fp['rad_obs']     = rad_obs
        fp['rad_sim_3d']  = rad_sim
        fp['rad_sim_std'] = rad_sim_std

    # ── Plot ─────────────────────────────────────────────────
    lon_min, lon_max = extent[0], extent[1]
    lat_min, lat_max = extent[2], extent[3]

    # Inner domain mask (strip 0.1° border to avoid edge artefacts)
    inner = ((lon >= lon_min + 0.1) & (lon <= lon_max - 0.1) &
             (lat >= lat_min + 0.1) & (lat <= lat_max - 0.1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    suptitle  = 'MODIS Radiance Self-Consistency — %d nm\n' % wvl
    suptitle += ('SZA = %.1f°,  SAA = %.1f°,  VZA = %.1f°,  VAA = %.1f°'
                 % (sza, saa, vza, vaa))
    if photons is not None:
        suptitle += ',  photons = %.0e' % photons
    if nrun is not None:
        suptitle += ',  Nrun = %d' % nrun
    if elapsed is not None:
        mins, secs = divmod(int(elapsed), 60)
        suptitle += ',  wall time = %dm %02ds' % (mins, secs)
    fig.suptitle(suptitle, fontsize=11, y=1.02)

    vmin, vmax = 0.0, 0.5   # radiance colour scale [W m⁻² sr⁻¹ µm⁻¹]

    tick_lon = FixedLocator(np.arange(-180, 181, 0.5))
    tick_lat = FixedLocator(np.arange(-90, 91, 0.5))

    # ── Top row: measured and simulated radiance maps ─────────
    for ax, data, title in [
            (axes[0, 0], rad_obs, 'MODIS Measured'),
            (axes[0, 1], rad_sim, 'EaR³T Simulated (3D)'),
    ]:
        cm = ax.pcolormesh(lon, lat, data, cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(cm, ax=ax, label='Radiance [W m⁻² sr⁻¹ µm⁻¹]',
                     fraction=0.046, pad=0.04)
        ax.set_xlim(lon_min + 0.05, lon_max - 0.05)
        ax.set_ylim(lat_min + 0.05, lat_max - 0.05)
        ax.xaxis.set_major_locator(tick_lon)
        ax.yaxis.set_major_locator(tick_lat)
        ax.set_xlabel('Longitude [°]', fontsize=9)
        ax.set_ylabel('Latitude [°]', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)

    # ── Bottom-left: density scatter ──────────────────────────
    ax = axes[1, 0]
    obs_flat = rad_obs[inner].ravel()
    sim_flat = rad_sim[inner].ravel()

    bins = np.arange(-0.01, 0.81, 0.005)
    heatmap, xe, ye = np.histogram2d(obs_flat, sim_flat, bins=(bins, bins))
    XX, YY = np.meshgrid((xe[:-1] + xe[1:]) / 2.0,
                          (ye[:-1] + ye[1:]) / 2.0, indexing='ij')
    levels = np.concatenate([
        np.arange(1, 10, 1),
        np.arange(10, 200, 10),
        np.arange(200, 1000, 100),
        np.arange(1000, 10001, 5000)])

    ax.contourf(XX, YY, heatmap, levels,
                extend='both', locator=ticker.LogLocator(), cmap='jet')
    ax.plot([0, 0.7], [0, 0.7], lw=1.2, ls='--', color='gray', zorder=4,
            label='1:1')
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel('Measured Radiance [W m⁻² sr⁻¹ µm⁻¹]', fontsize=9)
    ax.set_ylabel('Simulated 3D Radiance [W m⁻² sr⁻¹ µm⁻¹]', fontsize=9)
    ax.set_title('Density Scatter (log scale)', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ── Bottom-right: Monte Carlo relative noise (std / mean) ─
    ax = axes[1, 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        noise = np.where(rad_sim > 0, rad_sim_std / rad_sim * 100.0, np.nan)
    cm = ax.pcolormesh(lon, lat, noise, cmap='hot_r', vmin=0, vmax=50,
                       shading='auto')
    fig.colorbar(cm, ax=ax, label='MC noise σ/μ [%]',
                 fraction=0.046, pad=0.04)
    ax.set_xlim(lon_min + 0.05, lon_max - 0.05)
    ax.set_ylim(lat_min + 0.05, lat_max - 0.05)
    ax.xaxis.set_major_locator(tick_lon)
    ax.yaxis.set_major_locator(tick_lat)
    ax.set_xlabel('Longitude [°]', fontsize=9)
    ax.set_ylabel('Latitude [°]', fontsize=9)
    ax.set_title('MC Noise (σ/μ across Nrun runs)', fontsize=10)
    ax.tick_params(labelsize=8)
    # annotate median noise over cloudy pixels
    cloudy = noise[rad_obs > 0.05]
    if cloudy.size:
        med_noise = np.nanmedian(cloudy)
        ax.text(0.03, 0.97, 'median (cloudy) = %.1f %%' % med_noise,
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    fname_png = '%s-comparison_%dnm.png' % (name_tag, wvl)
    plt.savefig(fname_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: %s' % fname_png)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run 3D MODIS radiance simulation and compare with observations.')
    parser.add_argument('--predata', type=str, default=None,
        help='Path to modis_scene_input.h5 from 06_modis_data_download.py  (default: %s)' % fname_predata)
    parser.add_argument('--photons', type=float, default=None,
        help='Number of photons  (default: %.0e)' % photons)
    parser.add_argument('--ncpu', type=int, default=None,
        help='Number of CPU cores for MCARaTS  (default: %d)' % ncpu)
    parser.add_argument('--nrun', type=int, default=None,
        help='Number of independent MC runs; std dev across runs = noise estimate  (default: %d)' % Nrun)
    parser.add_argument('--plot-only', action='store_true',
        help='Skip simulation and regenerate figures from existing output.')
    parser.add_argument('--overwrite', action='store_true',
        help='Re-run simulation even if output already exists.')
    args = parser.parse_args()

    _fname_pre = args.predata  if args.predata  else fname_predata
    _photons   = args.photons  if args.photons  else photons
    _ncpu      = args.ncpu     if args.ncpu     else ncpu
    _nrun      = args.nrun     if args.nrun     else Nrun
    _plot_only = args.plot_only or plot_only
    _overwrite = args.overwrite or overwrite

    if not os.path.exists(_fname_pre):
        print('ERROR: modis_scene_input.h5 not found at %s' % _fname_pre)
        print('Run 06_modis_data_download.py first to download and pre-process the data.')
        sys.exit(1)

    # Derive output directory from the pre-data path
    scene_dir  = os.path.dirname(_fname_pre)
    scene_tag  = os.path.basename(scene_dir)
    _fdir_tmp  = os.path.join('tmp-data', _NAME, scene_tag,
                              'sim-%06.1fnm' % wavelength, '3d')
    _fdir_out  = os.path.join(_FDIR, scene_tag)
    _name_tag  = _NAME + '_' + scene_tag

    fname_sim = os.path.join(_fdir_tmp,
                             'mca-out-rad-modis-3d_%.4fnm.h5' % wavelength)

    elapsed = None
    if not _plot_only:
        print('\n[Step 1] Running 3D radiance simulation ...')
        print('  photons = %.0e,  Nrun = %d,  ncpu = %d' % (_photons, _nrun, _ncpu))
        t0 = time.time()
        fname_sim = run_simulation(
            _fname_pre, wavelength, _photons, _ncpu,
            _fdir_tmp, Nrun=_nrun, overwrite=_overwrite)
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        print('  Simulation wall time: %dm %02ds' % (mins, secs))
    else:
        if not os.path.exists(fname_sim):
            print('ERROR: simulation output not found — cannot use plot_only mode.')
            print('Run without --plot-only first.')
            sys.exit(1)
        print('plot_only mode: using existing simulation output.')

    print('\n[Step 2] Generating comparison figure ...')
    plot_comparison(_fname_pre, fname_sim, wavelength, _name_tag, _fdir_out,
                    photons=_photons, nrun=_nrun, elapsed=elapsed)

    print('\nDone.')


if __name__ == '__main__':
    main()
