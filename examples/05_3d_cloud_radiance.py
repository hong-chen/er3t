"""
EaR³T Example 05 — 3D Cloud Radiance (Satellite View, LES Field)
=================================================================

What this script does
---------------------
Simulates what a satellite *would observe* looking down through an
inhomogeneous LES cloud field.  Both the 3D and IPA solvers run in
a single call, producing three output figures.

plot_only mode
--------------
Set  plot_only = True  in the student control block to skip the
radiative-transfer calculation entirely and regenerate all figures
directly from the HDF5 files saved by a previous run.  This lets
you tweak colour scales, add annotations, or change axis limits
without waiting for new MC simulations.

Required files for plot_only mode (created on first full run):
    tmp-data/05_3d_cloud_radiance/.../mca-out-rad-3d.h5
    tmp-data/05_3d_cloud_radiance/.../mca-out-rad-ipa.h5
    tmp-data/05_3d_cloud_radiance/.../scene_info.npz

Outputs
-------
* <name>_comparison_<wl>nm.png  (4-panel)
    Top-left    — 3D reflectance image
    Top-right   — IPA reflectance image
    Bottom-left — Difference map (3D − IPA), diverging blue–white–red
    Bottom-right— Pixel-to-pixel scatter: 3D vs IPA (hexbin density)

* <name>_map_<wl>nm.png  (2-panel)
    Left  — Liquid water path [g m⁻²]
    Right — Cloud optical thickness (COT, vertically integrated extinction)

* <name>_r_tau_<wl>nm.png  (1-panel)
    IPA reflectance vs COT (hexbin density); shows how reflectance
    increases with cloud optical depth under the independent-pixel
    approximation

Key physics / "aha" moments
----------------------------
  - IPA: thick-cloud pixels look brighter, thin-cloud pixels darker.
    Cloud edges appear sharp — no lateral photon transport.
  - 3D: bright halos appear next to cloud edges (photons scatter
    sideways into open sky from the cloud interior).  Optically thick
    cloud tops appear slightly darker than in IPA (photon escape
    through cloud sides).
  - Difference map (3D − IPA): red = 3D brighter, blue = 3D darker.
  - r(tau) plot: IPA reflectance increases monotonically with COT;
    the 3D scatter shows additional horizontal transport effects.
    Thick-cloud pixels tend to fall below the 1:1 line (3D darker);
    cloud-edge / clear-sky pixels tend to fall above it (3D brighter).
  - Try sensor_zenith_angle = 30–45°: shadows become dramatic and the
    3D–IPA difference grows substantially.

Reflectance: R = π L / (E₀ cos θ_s)

Requires
--------
  data/00_er3t_mca/aux/les.nc  (~209 MB, downloaded via install-examples.sh)

Student controls — edit the block below
-----------------------------------------
"""

import os
import sys
import numpy as np
import datetime
import multiprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import h5py

import er3t


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     STUDENT CONTROL BLOCK                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣

plot_only = False       # set True to skip RT and re-plot from saved HDF5 files
                        # True  → skip RT entirely; regenerate all figures
                        #         from the HDF5 files saved by the last run.
                        #         Use this to tweak plots without re-running MC.
                        # False → run full MC simulation (then save outputs).

wavelength = 650.0      # Wavelength [nm].  Try: 400, 500, 650, 860, 1640.

photons    = 1e8        # Monte Carlo photons per spectral bin.
                        #   1e6 → fast (~minutes per solver); grainy image
                        #   1e7 → noticeably cleaner (recommended)
                        #   1e8 → production quality
                        # Script runs BOTH solvers → total ≈ 2× single-solver.

Nrun       = 3          # Independent MC runs per solver.

surface_albedo = 0.03   # Surface reflectance.
                        #   0.03 → ocean (default)
                        #   0.05 → green vegetation
                        #   0.20 → sand / bright desert
                        #   0.80 → fresh snow

solar_zenith_angle  = 30.0  # Solar zenith angle [°].
                              #   0  → overhead sun
                              #   30 → moderate (default)
                              #   60 → low sun → long shadows, strong 3D effects

solar_azimuth_angle = 45.0  # Solar azimuth [°], clockwise from North.

sensor_zenith_angle  = 0.0  # Sensor (satellite) viewing angle [°].
                              #   0  → nadir (default)
                              #   20 → moderate off-nadir
                              #   45 → oblique — shadows become dramatic!

sensor_azimuth_angle = 0.0  # Sensor azimuth [°] (relevant when VZA > 0).

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Internal configuration ────────────────────────────────────────────────
name_tag = '05_3d_cloud_radiance'
fdir0    = er3t.common.fdir_examples
Ncpu     = max(1, multiprocessing.cpu_count() - 2)
rcParams['font.size'] = 13
# ─────────────────────────────────────────────────────────────────────────


# ── Two-stream analytical functions (Coakley & Chylek 1975, JAS) ─────────────
# Reference: http://www.atmo.arizona.edu/students/courselinks/
#            spring09/atmo551b/2-stream%20solution.pdf

def _twostream_no_abs(tau, a=0.03, g=0.85, mu=0.866):
    """Conservative two-stream (w=1). r = (tau+a*x)/(tau+x)."""
    x = 2 * mu / (1. - g) / (1. - a)
    return (tau + a * x) / (tau + x)


def _coakley_black_surface(tau, w=0.9999, g=0.85, mu=0.866):
    """Coakley & Chylek (1975) eq. 13 — two-stream, black surface."""
    beta = (1 - g) / 2.
    U    = np.sqrt((1 - w + 2 * w * beta) / (1 - w))
    al   = np.sqrt(1 - w) * np.sqrt(1 - w + 2 * w * beta)
    ep   = np.exp( al * tau / mu)
    em   = np.exp(-al * tau / mu)
    num  = (U + 1) * (U - 1) * (ep - em)
    den  = (U + 1)**2 * ep - (U - 1)**2 * em
    return num / den


def _coakley_with_surface(tau, w=0.9999, g=0.85, mu=0.866, a=0.03):
    """
    Coakley & Chylek (1975) eq. 13 + adding formula for Lambertian surface:
        R_total = R_c + T_c^2 * a / (1 - R_c * a),  T_c ≈ 1 - R_c
    """
    R_c = _coakley_black_surface(tau, w=w, g=g, mu=mu)
    T_c = 1.0 - R_c
    return np.clip(R_c + T_c**2 * a / (1.0 - R_c * a), 0.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────


def run_radiance_simulation(
        solver,
        atm0, abs0, atm_1ds, atm_3ds, sca,
        fdir,
        Nrun=3,
        overwrite=True
        ):
    """Run MCARaTS radiance simulation for one solver. Returns output object."""
    mca0 = er3t.rtm.mca.mcarats_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            target='radiance',
            surface_albedo=surface_albedo,
            sca=sca,
            solar_zenith_angle=solar_zenith_angle,
            solar_azimuth_angle=solar_azimuth_angle,
            sensor_zenith_angle=sensor_zenith_angle,
            sensor_azimuth_angle=sensor_azimuth_angle,
            sensor_altitude=705000.0,
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=Nrun,
            photons=photons,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=overwrite
            )

    fname_h5 = '%s/mca-out-rad-%s.h5' % (fdir, solver.lower())
    out0 = er3t.rtm.mca.mca_out_ng(
            fname=fname_h5,
            mca_obj=mca0, abs_obj=abs0,
            mode='mean', squeeze=True,
            verbose=True, overwrite=overwrite
            )
    return out0


def example_05_rad_les_cloud_3d(
        wavelength=wavelength,
        overwrite=True,
        plot=True,
        plot_only=plot_only
        ):
    """
    Run (or reload) 3D and IPA nadir radiance simulations over the LES cloud
    field and produce three figure files.
    """

    fdir = '%s/tmp-data/%s/example_05_rad_les_cloud_3d' % (fdir0, name_tag)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    fname_h5_3d   = '%s/mca-out-rad-3d.h5'  % fdir
    fname_h5_ipa  = '%s/mca-out-rad-ipa.h5' % fdir
    fname_scene   = '%s/scene_info.npz'      % fdir

    # ── Branch: plot_only (reload) or full RT run ─────────────────────────────
    if plot_only:
        for f in [fname_h5_3d, fname_h5_ipa, fname_scene]:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    'plot_only=True but required file not found:\n  %s\n'
                    'Run once with plot_only=False to generate the RT output.'
                    % f)

        print('plot_only mode: loading saved outputs from\n  %s' % fdir)

        scene = np.load(fname_scene)
        lwp   = scene['lwp']
        cot   = scene['cot']
        x_km  = scene['x_km']
        y_km  = scene['y_km']
        Nx    = int(scene['Nx'])
        Ny    = int(scene['Ny'])

        with h5py.File(fname_h5_3d, 'r') as h:
            rad_3d = h['mean/rad'][:]
            toa    = float(h['mean/toa'][()])
        with h5py.File(fname_h5_ipa, 'r') as h:
            rad_ipa = h['mean/rad'][:]

    else:
        # ── Atmosphere ────────────────────────────────────────────────────────
        levels    = np.linspace(0.0, 20.0, 21)
        fname_atm = '%s/atm.pk' % fdir
        atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm,
                                             overwrite=overwrite)

        fname_abs = '%s/abs.pk' % fdir
        abs0      = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs,
                                          atm_obj=atm0, target='medium',
                                          overwrite=overwrite)

        # ── Cloud field from LES ──────────────────────────────────────────────
        fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
        fname_les = '%s/les.pk' % fdir
        cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les,
                                          coarsen=[1, 1, 25], overwrite=overwrite)

        # ── Derive LWP and COT from cloud object ──────────────────────────────
        dx       = float(cld0.lay['dx']['data'])
        dy       = float(cld0.lay['dy']['data'])
        Nx       = int(cld0.lay['nx']['data'])
        Ny       = int(cld0.lay['ny']['data'])
        x_km     = (np.arange(Nx) + 0.5) * dx
        y_km     = (np.arange(Ny) + 0.5) * dy

        lwc_3d   = cld0.lay['lwc']['data']          # (Nx, Ny, Nz)  kg m⁻³
        ext_3d   = cld0.lay['extinction']['data']   # (Nx, Ny, Nz)  m⁻¹
        thick_km = cld0.lay['thickness']['data']    # (Nz,)          km
        if thick_km.ndim == 1:
            thick_km = thick_km[np.newaxis, np.newaxis, :]

        # LWP [g m⁻²]: lwc[kg/m³] × dz[km] × 1000[m/km] × 1000[g/kg]
        lwp = (lwc_3d * thick_km * 1e6).sum(axis=2)
        # COT [–]:    ext[m⁻¹]  × dz[km] × 1000[m/km]
        cot = (ext_3d * thick_km * 1e3).sum(axis=2)

        # Save scene info so plot_only mode can reload without the cloud pickle
        np.savez(fname_scene, lwp=lwp, cot=cot,
                 x_km=x_km, y_km=y_km, Nx=Nx, Ny=Ny)
        print('Scene info saved: %s' % fname_scene)

        # ── Mie phase function ────────────────────────────────────────────────
        pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength)
        sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0,
                                      fname='%s/mca_sca.bin' % fdir,
                                      overwrite=overwrite)

        atm3d0 = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0,
                                           fname='%s/mca_atm_3d.bin' % fdir,
                                           overwrite=overwrite)
        atm1d0 = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
        atm_1ds = [atm1d0]
        atm_3ds = [atm3d0]

        # ── Run both solvers ──────────────────────────────────────────────────
        print('\n── Running 3D solver ──')
        out_3d  = run_radiance_simulation('3D',  atm0, abs0, atm_1ds, atm_3ds,
                                          sca, fdir, Nrun=Nrun, overwrite=overwrite)
        print('\n── Running IPA solver ──')
        out_ipa = run_radiance_simulation('IPA', atm0, abs0, atm_1ds, atm_3ds,
                                          sca, fdir, Nrun=Nrun, overwrite=overwrite)

        toa     = np.sum(abs0.coef['solar']['data'] * abs0.coef['weight']['data'])
        rad_3d  = out_3d.data['rad']['data']
        rad_ipa = out_ipa.data['rad']['data']

    # ── Reflectance  R = π L / (E₀ cos θ_s) ─────────────────────────────────
    mu0  = np.cos(np.deg2rad(solar_zenith_angle))
    norm = np.pi / (toa * mu0) if toa * mu0 > 0 else 1.0
    ref_3d  = rad_3d  * norm
    ref_ipa = rad_ipa * norm
    diff    = ref_3d  - ref_ipa

    extent_km = [0.0, Nx * (x_km[1] - x_km[0]) + x_km[0] - (x_km[1] - x_km[0]) / 2,
                 0.0, Ny * (y_km[1] - y_km[0]) + y_km[0] - (y_km[1] - y_km[0]) / 2]
    # Simpler: pixel size is uniform, so domain = [0, Nx*dx, 0, Ny*dy]
    dx_val = x_km[1] - x_km[0]
    dy_val = y_km[1] - y_km[0]
    extent_km = [0.0, Nx * dx_val, 0.0, Ny * dy_val]

    print('\n── Domain-mean reflectance ──────────────────────────────────────')
    print('  3D:   %.4f' % ref_3d.mean())
    print('  IPA:  %.4f' % ref_ipa.mean())
    print('  mean |3D − IPA|: %.4f' % np.abs(diff).mean())
    print('  max  |3D − IPA|: %.4f' % np.abs(diff).max())

    if not plot:
        return dict(ref_3d=ref_3d, ref_ipa=ref_ipa, diff=diff,
                    lwp=lwp, cot=cot)

    vmax_ref  = min(1.0, float(np.percentile(np.maximum(ref_3d, ref_ipa), 99.5)))
    vlim_diff = max(0.01, float(np.percentile(np.abs(diff), 99)))

    im_kw = dict(origin='lower', extent=extent_km, aspect='equal',
                 interpolation='none')

    suptitle_str = (
        'LES Cloud Radiance: 3D vs IPA  —  %.0f nm\n'
        'SZA = %.0f°,  SAA = %.0f°,  VZA = %.0f°,  VAA = %.0f°,  '
        'albedo = %.2f,  photons = %.0e'
        % (wavelength,
           solar_zenith_angle, solar_azimuth_angle,
           sensor_zenith_angle, sensor_azimuth_angle,
           surface_albedo, photons)
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 1 — 4-panel comparison (reflectance images, diff, scatter)
    # ═════════════════════════════════════════════════════════════════════════
    from matplotlib.gridspec import GridSpec

    fig1 = plt.figure(figsize=(15, 12))
    gs   = GridSpec(2, 2, figure=fig1, wspace=0.30, hspace=0.40)
    ax_3d   = fig1.add_subplot(gs[0, 0])
    ax_ipa  = fig1.add_subplot(gs[0, 1])
    ax_diff = fig1.add_subplot(gs[1, 0])
    ax_scat = fig1.add_subplot(gs[1, 1])

    im1 = ax_3d.imshow(np.transpose(ref_3d),
                       cmap='Greys_r', vmin=0.0, vmax=vmax_ref, **im_kw)
    fig1.colorbar(im1, ax=ax_3d, label='Reflectance', shrink=0.85)
    ax_3d.set_xlabel('x [km]'); ax_3d.set_ylabel('y [km]')
    ax_3d.set_title('3D reflectance\n%.0f nm,  SZA=%.0f°,  VZA=%.0f°'
                    % (wavelength, solar_zenith_angle, sensor_zenith_angle))

    im2 = ax_ipa.imshow(np.transpose(ref_ipa),
                        cmap='Greys_r', vmin=0.0, vmax=vmax_ref, **im_kw)
    fig1.colorbar(im2, ax=ax_ipa, label='Reflectance', shrink=0.85)
    ax_ipa.set_xlabel('x [km]'); ax_ipa.set_ylabel('y [km]')
    ax_ipa.set_title('IPA reflectance\n%.0f nm,  SZA=%.0f°,  VZA=%.0f°'
                     % (wavelength, solar_zenith_angle, sensor_zenith_angle))

    im3 = ax_diff.imshow(np.transpose(diff),
                         cmap='RdBu_r', vmin=-vlim_diff, vmax=vlim_diff, **im_kw)
    cb3 = fig1.colorbar(im3, ax=ax_diff,
                        label='3D − IPA  (reflectance)', shrink=0.85)
    cb3.ax.axhline(0, color='k', lw=0.8, ls='--')
    ax_diff.set_xlabel('x [km]'); ax_diff.set_ylabel('y [km]')
    ax_diff.set_title('Difference: 3D − IPA\n'
                      'Red = 3D brighter  |  Blue = 3D darker')

    hb = ax_scat.hexbin(ref_ipa.ravel(), ref_3d.ravel(),
                        gridsize=120, cmap='viridis',
                        bins='log', mincnt=1,
                        extent=[0, vmax_ref, 0, vmax_ref])
    fig1.colorbar(hb, ax=ax_scat, label='log₁₀(pixel count)', shrink=0.85)
    ax_scat.plot([0, vmax_ref], [0, vmax_ref], color='red', lw=1.5, ls='--', label='1:1')
    ax_scat.set_xlabel('IPA reflectance')
    ax_scat.set_ylabel('3D reflectance')
    ax_scat.set_title('Pixel-to-pixel: 3D vs IPA\n'
                      'Above 1:1 = 3D brighter,  below = 3D darker')
    ax_scat.set_xlim(0, vmax_ref); ax_scat.set_ylim(0, vmax_ref)
    ax_scat.set_aspect('equal')
    ax_scat.legend(fontsize=10); ax_scat.grid(True, alpha=0.25)
    fig1.suptitle(suptitle_str, fontsize=12, y=1.01)

    fname1 = '%s-comparison_%.0fnm.png' % (name_tag, wavelength)
    fig1.savefig(fname1, bbox_inches='tight')
    plt.close(fig1)
    print('Comparison saved: %s' % fname1)

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 2 — Cloud scene map: LWP and COT
    # ═════════════════════════════════════════════════════════════════════════
    fig2, (ax_lwp, ax_cot) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.subplots_adjust(wspace=0.30)

    im_lwp = ax_lwp.imshow(np.transpose(lwp), cmap='Blues',
                           vmin=0, vmax=np.percentile(lwp, 99.5),
                           origin='lower', extent=extent_km,
                           aspect='equal', interpolation='none')
    fig2.colorbar(im_lwp, ax=ax_lwp, label='LWP [g m$^{-2}$]', shrink=0.85)
    ax_lwp.set_xlabel('x [km]'); ax_lwp.set_ylabel('y [km]')
    ax_lwp.set_title('Liquid Water Path  [g m$^{-2}$]')

    im_cot = ax_cot.imshow(np.transpose(cot), cmap='viridis',
                           vmin=0, vmax=np.percentile(cot, 99.5),
                           origin='lower', extent=extent_km,
                           aspect='equal', interpolation='none')
    fig2.colorbar(im_cot, ax=ax_cot, label='COT  [–]', shrink=0.85)
    ax_cot.set_xlabel('x [km]'); ax_cot.set_ylabel('y [km]')
    ax_cot.set_title('Cloud Optical Thickness (COT)\n'
                     '= vertical integral of extinction')

    fig2.suptitle('LES cloud scene — %.0f nm' % wavelength,
                  fontsize=12, y=1.01)

    fname2 = '%s-map_%.0fnm.png' % (name_tag, wavelength)
    fig2.savefig(fname2, bbox_inches='tight')
    plt.close(fig2)
    print('Map saved:        %s' % fname2)

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 3 — R(tau): IPA reflectance vs COT + two-stream theory
    # ═════════════════════════════════════════════════════════════════════════
    from scipy.optimize import curve_fit

    cot_flat = cot.ravel()
    ipa_flat = ref_ipa.ravel()

    cot_max = float(np.percentile(cot_flat[cot_flat > 0], 99.5)) if (cot_flat > 0).any() else 50.0
    cot_max = max(cot_max, 1.0)

    tau_theory = np.linspace(0, cot_max, 500)
    g_liq      = 0.85    # asymmetry parameter, liquid water clouds (visible)
    w_liq      = 0.9999  # single-scattering albedo, liquid water (visible)

    r_noabs = _twostream_no_abs(tau_theory,      a=surface_albedo, g=g_liq, mu=mu0)
    r_black = _coakley_black_surface(tau_theory, w=w_liq, g=g_liq, mu=mu0)
    r_surf  = _coakley_with_surface(tau_theory,  w=w_liq, g=g_liq, mu=mu0, a=surface_albedo)

    # Fit Coakley with surface to binned MC median (a fixed, g and w free)
    n_bins   = 120
    cot_bins = np.linspace(0, cot_max, n_bins + 1)
    bin_ctrs = 0.5 * (cot_bins[:-1] + cot_bins[1:])
    bin_med  = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (cot_flat >= cot_bins[i]) & (cot_flat < cot_bins[i + 1])
        if mask.sum() > 20:
            bin_med[i] = np.median(ipa_flat[mask])
    valid  = np.isfinite(bin_med)
    x_fit  = bin_ctrs[valid]
    y_fit  = bin_med[valid]

    def _fit_fn(tau, w, g):
        return _coakley_with_surface(tau, w=w, g=g, mu=mu0, a=surface_albedo)

    try:
        popt, _ = curve_fit(_fit_fn, x_fit, y_fit,
                            p0=[0.9999, 0.85],
                            bounds=([0.5, 0.0], [0.999999, 0.9999]),
                            maxfev=5000)
        w_fit, g_fit = popt
        r_fit   = _coakley_with_surface(tau_theory, w=w_fit, g=g_fit,
                                        mu=mu0, a=surface_albedo)
        fit_ok  = True
        print('Coakley fit:  w = %.5f,  g = %.3f' % (w_fit, g_fit))
    except Exception as e:
        fit_ok = False
        print('Coakley fit failed: %s' % e)

    fig3, ax_r = plt.subplots(1, 1, figsize=(9, 6))

    hb = ax_r.hexbin(cot_flat, ipa_flat,
                     gridsize=80, cmap='Blues',
                     bins='log', mincnt=1,
                     extent=[0, cot_max, 0, 1.0])
    fig3.colorbar(hb, ax=ax_r, label='log₁₀(pixel count)', shrink=0.85)

    ax_r.plot(tau_theory, r_surf,  color='firebrick',  lw=2.0, ls='-',
              label='Coakley eq. 23  (w=%.4f, g=%.2f, a=%.2f)' % (w_liq, g_liq, surface_albedo))
    ax_r.plot(tau_theory, r_black, color='darkorange',  lw=1.5, ls='--',
              label='Coakley eq. 13  (w=%.4f, g=%.2f, a=0)' % (w_liq, g_liq))
    ax_r.plot(tau_theory, r_noabs, color='green',       lw=1.5, ls=':',
              label='Conservative 2-stream  (w=1, g=%.2f, a=%.2f)' % (g_liq, surface_albedo))
    if fit_ok:
        ax_r.plot(tau_theory, r_fit, color='black', lw=2.5, ls='-',
                  label='Coakley fit  (w=%.5f, g=%.3f, a=%.2f fixed)'
                  % (w_fit, g_fit, surface_albedo))

    ax_r.set_xlabel('Cloud Optical Thickness (COT)')
    ax_r.set_ylabel('IPA Reflectance')
    ax_r.set_xlim(0, cot_max)
    ax_r.set_ylim(0, 1.0)
    ax_r.set_title('IPA Reflectance vs COT\n%.0f nm,  SZA = %.0f°,  albedo = %.2f  '
                   '— analytical two-stream overlaid'
                   % (wavelength, solar_zenith_angle, surface_albedo))
    ax_r.legend(fontsize=9, loc='lower right')
    ax_r.grid(True, alpha=0.25)

    fig3.tight_layout()

    fname3 = '%s-r_tau_%.0fnm.png' % (name_tag, wavelength)
    fig3.savefig(fname3, bbox_inches='tight')
    plt.close(fig3)
    print('R(tau) saved:     %s' % fname3)

    return dict(ref_3d=ref_3d, ref_ipa=ref_ipa, diff=diff,
                lwp=lwp, cot=cot)


if __name__ == '__main__':

    import time
    t_start = time.time()

    print('=' * 65)
    print('EaR³T Example 05 — 3D Cloud Radiance (Satellite View)')
    print('  plot_only           = %s' % plot_only)
    print('  wavelength          = %.1f nm'  % wavelength)
    if not plot_only:
        print('  photons             = %.0e (× 2 solvers)' % photons)
        print('  Nrun                = %d'        % Nrun)
    print('  surface_albedo      = %.2f'      % surface_albedo)
    print('  solar_zenith_angle  = %.1f°'     % solar_zenith_angle)
    print('  solar_azimuth_angle = %.1f°'     % solar_azimuth_angle)
    print('  sensor_zenith_angle = %.1f°'     % sensor_zenith_angle)
    print('  sensor_azimuth_angle= %.1f°'     % sensor_azimuth_angle)
    if not plot_only:
        print('  Ncpu                = %d'        % Ncpu)
    print('=' * 65)
    print()
    if plot_only:
        print('Fast plot-only mode: loading existing HDF5 outputs.')
    else:
        print('Full run: 3D + IPA solvers → three output figures.')
    print('  *_comparison_*.png — reflectance images, diff map, scatter')
    print('  *_map_*.png        — LWP and COT maps')
    print('  *_r_tau_*.png      — reflectance and 3D bias vs optical depth')
    print()
    print('Key experiments:')
    print('  plot_only=True      → tweak plots instantly without re-running RT')
    print('  sensor_zenith_angle=30° → oblique view, dramatic shadows')
    print('  solar_zenith_angle=60°  → low sun, amplified 3D effects')
    print('  surface_albedo=0.8      → snow surface, lower cloud contrast')
    print()

    example_05_rad_les_cloud_3d(wavelength=wavelength, plot_only=plot_only)

    elapsed = time.time() - t_start
    print()
    print('=' * 65)
    print('Total elapsed time: %d min %02d sec' % (int(elapsed) // 60, int(elapsed) % 60))
    print('=' * 65)
