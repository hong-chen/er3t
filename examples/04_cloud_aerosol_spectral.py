"""
EaR³T Example 04 — 3D Cloud + 1D Aerosol Layer, Spectral Flux
==============================================================

What this script does
---------------------
Extends Example 03 into the spectral dimension.  The same LES cloud scene and
1D aerosol layer are used as before, but calculations are looped over multiple
wavelengths.  After all wavelengths have been run, a three-panel figure is
produced that lets you explore both spatial and spectral structure simultaneously.

Outputs
-------
* Per-wavelength HDF5 results in  tmp-data/04_cloud_aerosol_spectral/
    mca-out-flux-3d_<wl>nm.h5
    mca-out-flux-ipa_<wl>nm.h5
* PNG plot   04_cloud_aerosol_spectral-<wavelength list>nm.png
  Layout (two rows × three columns):
    Row 1, Col 1 — Liquid water path map [g m⁻²] in km coordinates
                   (colour scale capped at 200 g m⁻²), output locations marked.
    Row 1, Col 2 — Upwelling (F↑) normalised spectra at each output point — 3D.
    Row 1, Col 3 — Upwelling (F↑) normalised spectra at each output point — IPA.
    Row 2, Col 1 — Transmitted irradiance map: F↓ at the surface (3D,
                   first wavelength in the list), output locations marked.
    Row 2, Col 2 — Downwelling (F↓ total and diffuse) normalised spectra — 3D.
    Row 2, Col 3 — Downwelling (F↓ total and diffuse) normalised spectra — IPA.

Key physics / "aha" moments
----------------------------
  - The spectral dependence of cloud transmittance is weak in the visible
    but strengthens strongly in the near-IR (liquid water absorption
    around 1640 nm cuts downwelling dramatically).
  - Aerosol effects are most visible at shorter wavelengths (Rayleigh-like
    scattering; strong absorption by black carbon in the UV–visible).
  - The 3D–IPA difference is roughly wavelength-independent over the cloud,
    because it is driven by horizontal photon transport that depends on
    cloud geometry, not wavelength.
  - Varying z_out lets you sample below / inside / above the cloud and see
    how the spectral and 3D-vs-IPA signatures change with altitude.

Requires
--------
  data/00_er3t_mca/aux/les.nc  (~209 MB, downloaded via install-examples.sh)

Student controls — edit the block below
-----------------------------------------
"""

import os
import sys
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

# ── Wavelengths ───────────────────────────────────────────────────────────
wavelengths = [450., 555., 650.0]
                        # Wavelengths to simulate [nm].
                        # Each wavelength runs both solvers independently.
                        # Typical choices: 400 (UV–blue), 500, 650 (red),
                        # 860 (near-IR), 1640 (strong liquid-water absorption).
                        # Fewer wavelengths → faster; add more to fill in the spectrum.

# ── Output (x, y, z) locations ────────────────────────────────────────────
x_out = [10]   # x coordinate(s) of output location [km].
                        # Domain is 48 × 48 km.  x_out and y_out can be
                        # scalar-like lists or longer arrays — one spectrum
                        # per location is plotted in Panel 2.
y_out = [25]   # y coordinate(s) of output location [km].
z_out = [5]   # z coordinate(s) of output location [km].
                        # Set z_out[i] to 0 for the surface, ~1–2 km for
                        # inside the cloud, or a higher value to sample
                        # above the cloud.

# ── MC settings ───────────────────────────────────────────────────────────
photons    = 1e8        # Monte Carlo photons per spectral bin.
Nrun       = 10          # Independent MC runs per solver (for noise reduction).

# ── Scene parameters ──────────────────────────────────────────────────────
surface_albedo      = 0.3   # Surface reflectance.
solar_zenith_angle  = 30.0  # Solar zenith angle [°].
solar_azimuth_angle = 45.0  # Solar azimuth [°], clockwise from North.

# ── Aerosol layer ─────────────────────────────────────────────────────────
aod   = 0.4             # Aerosol optical depth (total column).
                        #   0.0 → no aerosol.
ssa   = 0.9             # Single scattering albedo.
asy   = 0.6             # Asymmetry parameter.
z_bot = 0.0             # Aerosol layer bottom altitude [km].
z_top = 2.0             # Aerosol layer top altitude [km].

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Internal configuration ────────────────────────────────────────────────
name_tag = '04_cloud_aerosol_spectral'
fdir0    = er3t.common.fdir_examples
Ncpu     = max(1, multiprocessing.cpu_count() - 2)
rcParams['font.size'] = 13
# ─────────────────────────────────────────────────────────────────────────


def run_flux_simulation(
        solver,
        atm0, abs0, atm_1ds, atm_3ds,
        fdir, wl,
        Nrun=3,
        overwrite=True
        ):
    """
    Run MCARaTS flux simulation for one solver at one wavelength.
    Returns the output object (mca_out_ng).
    """
    mca0 = er3t.rtm.mca.mcarats_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            Nrun=Nrun,
            surface_albedo=surface_albedo,
            solar_zenith_angle=solar_zenith_angle,
            solar_azimuth_angle=solar_azimuth_angle,
            fdir='%s/%.0fnm/flux_%s' % (fdir, wl, solver.lower()),
            photons=photons,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=overwrite
            )

    fname_h5 = '%s/mca-out-flux-%s_%.0fnm.h5' % (fdir, solver.lower(), wl)
    out0 = er3t.rtm.mca.mca_out_ng(
            fname=fname_h5,
            mca_obj=mca0, abs_obj=abs0,
            mode='mean', squeeze=True,
            verbose=True, overwrite=overwrite
            )
    return out0


def example_04_spectral(
        overwrite=True,
        plot=True
        ):
    """
    Run 3D and IPA flux simulations over multiple wavelengths on an LES cloud
    field with a 1D aerosol layer, and produce a 3-panel spectral figure.
    """

    fdir = '%s/tmp-data/%s/example_04_spectral' % (fdir0, name_tag)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # ── Validate output location arrays ──────────────────────────────────────
    x_arr = np.atleast_1d(np.array(x_out, dtype=float))
    y_arr = np.atleast_1d(np.array(y_out, dtype=float))
    z_arr = np.atleast_1d(np.array(z_out, dtype=float))
    n_pts = len(x_arr)
    if not (len(y_arr) == n_pts and len(z_arr) == n_pts):
        raise ValueError('x_out, y_out, z_out must all have the same length.')

    # ── Build atmosphere (shared across wavelengths) ──────────────────────────
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # ── Build cloud (shared across wavelengths) ───────────────────────────────
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    fname_les = '%s/les.pk' % fdir
    cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les,
                                     coarsen=[1, 1, 25], overwrite=overwrite)

    # 3D MCARaTS atmosphere (cloud geometry; same for all wavelengths)
    atm3d0 = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0,
                                      fname='%s/mca_atm_3d.bin' % fdir,
                                      overwrite=overwrite)
    atm_3ds = [atm3d0]

    # ── Derive pixel grid from cloud object ───────────────────────────────────
    dx = float(cld0.lay['dx']['data'])        # km per pixel in x
    dy = float(cld0.lay['dy']['data'])        # km per pixel in y
    Nx = int(cld0.lay['nx']['data'])
    Ny = int(cld0.lay['ny']['data'])

    # Pixel-centre coordinates [km]
    x_km = (np.arange(Nx) + 0.5) * dx
    y_km = (np.arange(Ny) + 0.5) * dy

    # Convert (x_out, y_out, z_out) to pixel / level indices
    xi_arr = np.clip(np.floor(x_arr / dx).astype(int), 0, Nx - 1)
    yi_arr = np.clip(np.floor(y_arr / dy).astype(int), 0, Ny - 1)

    alt_prof = atm0.lev['altitude']['data']   # 0, 1, … 20 km (21 values)
    zi_arr   = np.array([np.argmin(np.abs(alt_prof - z)) for z in z_arr])

    # Compute TOA normalisation denominator: cos(SZA)
    mu0 = np.cos(np.deg2rad(solar_zenith_angle))

    # ── Liquid water path (for Panel 1) ──────────────────────────────────────
    # lwc [kg m⁻³], thickness [km] → LWP [g m⁻²]
    lwc       = cld0.lay['lwc']['data']            # (Nx, Ny, Nz_cld) kg/m³
    thick_km  = cld0.lay['thickness']['data']      # may be 1-D (Nz_cld)
    if thick_km.ndim == 1:
        thick_km = thick_km[np.newaxis, np.newaxis, :]   # broadcast to (1,1,Nz)
    lwp = (lwc * thick_km * 1e3).sum(axis=2) * 1e3       # kg/m³ × km × 1000 m/km × 1000 g/kg → g/m²

    # ── Spectral loop ─────────────────────────────────────────────────────────
    results = {}   # keyed by wavelength [nm]
    wl_list = sorted(wavelengths)

    for wl in wl_list:
        print('\n' + '=' * 60)
        print('Wavelength: %.0f nm' % wl)
        print('=' * 60)

        # Gas absorption (per wavelength)
        fname_abs = '%s/abs_%.0fnm.pk' % (fdir, wl)
        abs0 = er3t.pre.abs.abs_rep(wavelength=wl, fname=fname_abs,
                                    atm_obj=atm0, target='medium',
                                    overwrite=overwrite)

        # 1D atmosphere (gas + aerosol; per wavelength because abs0 differs)
        atm1d0 = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
        if aod > 0.0:
            aer_ext = aod / (z_top - z_bot) / 1000.0
            atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy,
                                   z_bottom=z_bot, z_top=z_top)
        atm_1ds = [atm1d0]

        # TOA irradiance for this wavelength (W m⁻² nm⁻¹); cos(SZA) applied separately
        toa = np.sum(abs0.coef['solar']['data'] * abs0.coef['weight']['data'])

        print('\n── Running 3D solver ──')
        out_3d  = run_flux_simulation('3D',  atm0, abs0, atm_1ds, atm_3ds,
                                      fdir, wl, Nrun=Nrun, overwrite=overwrite)
        print('\n── Running IPA solver ──')
        out_ipa = run_flux_simulation('IPA', atm0, abs0, atm_1ds, atm_3ds,
                                      fdir, wl, Nrun=Nrun, overwrite=overwrite)

        results[wl] = dict(out_3d=out_3d, out_ipa=out_ipa, toa=toa, abs0=abs0)

    # ── Visualisation ─────────────────────────────────────────────────────────
    if not plot:
        return results

    flux_keys    = ['f_up', 'f_down', 'f_down_diffuse']
    flux_labels  = ['$F_{\\uparrow}$', '$F_{\\downarrow}$ total', '$F_{\\downarrow}$ diffuse']
    flux_linestyles = ['-', '--', ':']
    flux_markers    = ['o', 's', '^']

    # Output locations → maximally distinct colours (red, blue, green, orange, …)
    _loc_palette = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e',
                    '#9467bd', '#8c564b', '#e377c2', '#17becf']
    loc_colors = [_loc_palette[i % len(_loc_palette)] for i in range(n_pts)]

    # Pre-compute TOA-normalised spectra: spec[i_pt][wl][solver_key][flux_key]
    # Also store ±1σ standard deviation (from Nrun independent MC runs).
    spec = {}
    for i_pt in range(n_pts):
        xi, yi, zi = xi_arr[i_pt], yi_arr[i_pt], zi_arr[i_pt]
        spec[i_pt] = {}
        for wl in wl_list:
            r    = results[wl]
            norm = r['toa'] * mu0 if r['toa'] * mu0 > 0 else 1.0
            spec[i_pt][wl] = {
                '3d':      {k: r['out_3d'].data[k]['data'][xi, yi, zi]       / norm
                            for k in flux_keys},
                'ipa':     {k: r['out_ipa'].data[k]['data'][xi, yi, zi]      / norm
                            for k in flux_keys},
                '3d_std':  {k: r['out_3d'].data[k + '_std']['data'][xi, yi, zi]  / norm
                            for k in flux_keys},
                'ipa_std': {k: r['out_ipa'].data[k + '_std']['data'][xi, yi, zi] / norm
                            for k in flux_keys},
            }

    def _plot_spectra(ax, solver_key, title, flux_subset='all'):
        """Plot normalised spectra for one solver on ax.
        flux_subset: 'all' | 'up' (F↑ only) | 'down' (F↓ total + diffuse).
        Colour encodes output location; linestyle + marker encodes flux component."""
        if flux_subset == 'up':
            _keys = flux_keys[:1]; _lbls = flux_labels[:1]
            _mrks = flux_markers[:1]; _lss  = flux_linestyles[:1]
        elif flux_subset == 'down':
            _keys = flux_keys[1:]; _lbls = flux_labels[1:]
            _mrks = flux_markers[1:]; _lss  = flux_linestyles[1:]
        else:
            _keys = flux_keys; _lbls = flux_labels
            _mrks = flux_markers; _lss = flux_linestyles

        std_key = solver_key + '_std'
        for i_pt in range(n_pts):
            for k, lbl, mrk, ls in zip(_keys, _lbls, _mrks, _lss):
                vals = np.array([spec[i_pt][wl][solver_key][k] for wl in wl_list])
                stds = np.array([spec[i_pt][wl][std_key][k]    for wl in wl_list])
                ax.plot(wl_list, vals,
                        color=loc_colors[i_pt], marker=mrk, ms=6,
                        lw=2.0, ls=ls)
                ax.fill_between(wl_list,
                                vals - stds, vals + stds,
                                color=loc_colors[i_pt], alpha=0.18, linewidth=0)

        # Legend: flux components (linestyle + marker), then locations (colour)
        handles = []
        for k, lbl, mrk, ls in zip(_keys, _lbls, _mrks, _lss):
            handles.append(mpl.lines.Line2D([0], [0], color='k',
                                            marker=mrk, ms=6, lw=2, ls=ls,
                                            label=lbl))
        for i_pt in range(n_pts):
            handles.append(mpl.patches.Patch(
                color=loc_colors[i_pt],
                label='loc%d  (%.1f, %.1f, %.1f km)' %
                      (i_pt + 1, x_arr[i_pt], y_arr[i_pt], z_arr[i_pt])))

        ax.legend(handles=handles, fontsize=8, loc='upper right', framealpha=0.9)
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Irradiance / (TOA × cos SZA)')
        ax.set_title(title)
        ax.set_xlim(min(wl_list) - 50, max(wl_list) + 50)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)

    # ── Cross-section helper (first wavelength, first location) ─────────────
    xs_flux_colors  = ['firebrick', 'steelblue', 'seagreen']
    xs_flux_lss     = ['-', '--', ':']
    xs_flux_markers = ['o', 's', '^']

    def _plot_xsection(ax, out_obj, solver_label):
        """x cross-section of all flux components / TOA at y=y_arr[0], z=zi_arr[0].
        First wavelength and first output location only."""
        wl0   = wl_list[0]
        r0    = results[wl0]
        norm0 = r0['toa'] * mu0 if r0['toa'] * mu0 > 0 else 1.0
        yi0   = yi_arr[0]
        zi0   = zi_arr[0]
        z_km_xs = alt_prof[zi0]

        for k, lbl, col, ls, mrk in zip(
                flux_keys, flux_labels, xs_flux_colors, xs_flux_lss, xs_flux_markers):
            vals = out_obj.data[k]['data'][:, yi0, zi0] / norm0
            ax.plot(x_km, vals, color=col, lw=1.8, ls=ls, label=lbl)
            ax.scatter(x_km[xi_arr[0]], vals[xi_arr[0]],
                       marker=mrk, s=80, color=col, zorder=5,
                       edgecolors='k', linewidths=0.5)

        ax.axvline(x_km[xi_arr[0]], color='dimgray', lw=0.9, ls=':',
                   label='x₀ = %.1f km' % x_km[xi_arr[0]])
        ax.set_xlabel('x [km]')
        ax.set_ylabel('Flux / TOA')
        ax.set_xlim(x_km[0], x_km[-1])
        ax.set_ylim(bottom=0)
        ax.set_title('x cross-section — %s\n'
                     'y = %.1f km,  z = %.1f km,  %.0f nm'
                     % (solver_label, y_km[yi0], z_km_xs, wl0))
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.25)

    fig, axes = plt.subplots(2, 4, figsize=(26, 13))
    fig.subplots_adjust(wspace=0.35, hspace=0.48)

    # ── Row 0, Col 0: LWP map ────────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.pcolormesh(x_km, y_km, lwp.T, cmap='jet', shading='auto',
                       vmin=0, vmax=200)
    fig.colorbar(im, ax=ax, label='LWP [g m$^{-2}$]', shrink=0.85)
    for i in range(n_pts):
        ax.scatter(x_arr[i], y_arr[i],
                   color=loc_colors[i], s=200, marker='*', zorder=5,
                   edgecolors='k', linewidths=0.5,
                   label='loc%d  (%.1f, %.1f) km' % (i + 1, x_arr[i], y_arr[i]))
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_title('Liquid Water Path  [g m$^{-2}$]\n(capped at 200 g m$^{-2}$)')
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')

    # ── Row 0, Col 1–2: IPA upwelling and downwelling spectra ────────────────
    _plot_spectra(axes[0, 1], 'ipa',
                  'Upwelling $F_{\\uparrow}$ — IPA\n(colour = location)',
                  flux_subset='up')
    _plot_spectra(axes[0, 2], 'ipa',
                  'Downwelling $F_{\\downarrow}$ — IPA\n(colour = location)',
                  flux_subset='down')

    # ── Row 0, Col 3: IPA x cross-section ────────────────────────────────────
    _plot_xsection(axes[0, 3], results[wl_list[0]]['out_ipa'], 'IPA')

    # ── Row 1, Col 0: Transmitted irradiance map (3D, surface, first wl) ─────
    wl_map    = wl_list[0]
    r_map     = results[wl_map]
    norm_map  = r_map['toa'] * mu0 if r_map['toa'] * mu0 > 0 else 1.0
    f_dn_surf = r_map['out_3d'].data['f_down']['data'][:, :, 0] / norm_map
    ax = axes[1, 0]
    im2 = ax.pcolormesh(x_km, y_km, f_dn_surf.T, cmap='jet', shading='auto',
                        vmin=0, vmax=1)
    fig.colorbar(im2, ax=ax,
                 label='$F_{\\downarrow}$ / (TOA × cos SZA)', shrink=0.85)
    for i in range(n_pts):
        ax.scatter(x_arr[i], y_arr[i],
                   color=loc_colors[i], s=200, marker='*', zorder=5,
                   edgecolors='k', linewidths=0.5)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_title('Transmitted irradiance — 3D\n'
                 '$F_{\\downarrow}$ at surface,  %.0f nm' % wl_map)
    ax.set_aspect('equal')

    # ── Row 1, Col 1–2: 3D upwelling and downwelling spectra ─────────────────
    _plot_spectra(axes[1, 1], '3d',
                  'Upwelling $F_{\\uparrow}$ — 3D\n(colour = location)',
                  flux_subset='up')
    _plot_spectra(axes[1, 2], '3d',
                  'Downwelling $F_{\\downarrow}$ — 3D\n(colour = location)',
                  flux_subset='down')

    # ── Row 1, Col 3: 3D x cross-section ─────────────────────────────────────
    _plot_xsection(axes[1, 3], results[wl_list[0]]['out_3d'], '3D')

    # ── Sync y limits: same flux direction, IPA (row 0) vs 3D (row 1) ────────
    for ax_a, ax_b in [(axes[0, 1], axes[1, 1]),   # upwelling IPA vs 3D
                       (axes[0, 2], axes[1, 2]),   # downwelling IPA vs 3D
                       (axes[0, 3], axes[1, 3])]:  # x-section IPA vs 3D
        ymax = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
        for ax in (ax_a, ax_b):
            ax.set_ylim(0, ymax)

    # ── Supertitle and save ───────────────────────────────────────────────────
    wl_str = '_'.join(['%.0f' % w for w in wl_list])
    fig.suptitle(
        'LES Cloud + Aerosol — Spectral Flux: 3D vs IPA\n'
        '$\\lambda$ = [%s] nm,  SZA = %.0f°,  albedo = %.2f,  photons = %.0e\n'
        'AOD = %.2f  SSA = %.2f  asy = %.2f  z$_{aer}$ = [%.1f–%.1f km]'
        % (', '.join(['%.0f' % w for w in wl_list]),
           solar_zenith_angle, surface_albedo, photons,
           aod, ssa, asy, z_bot, z_top),
        fontsize=11, y=1.02
    )

    fname_png = '%s-comparison_%snm.png' % (name_tag, wl_str)
    plt.savefig(fname_png, bbox_inches='tight')
    plt.close(fig)
    print('\nPlot saved: %s' % fname_png)

    return results


if __name__ == '__main__':

    print('=' * 65)
    print('EaR³T Example 04 — Cloud + Aerosol Spectral Flux (3D vs IPA)')
    print('  wavelengths         =', wavelengths, 'nm')
    print('  photons             = %.0e (× 2 solvers × %d wavelengths)'
          % (photons, len(wavelengths)))
    print('  Nrun                = %d' % Nrun)
    print('  surface_albedo      = %.2f' % surface_albedo)
    print('  solar_zenith_angle  = %.1f°' % solar_zenith_angle)
    print('  solar_azimuth_angle = %.1f°' % solar_azimuth_angle)
    print('  Aerosol:  AOD=%.2f  SSA=%.2f  asy=%.2f  z=[%.1f–%.1f km]'
          % (aod, ssa, asy, z_bot, z_top))
    print('  Output locations:')
    for i in range(len(x_out)):
        print('    loc%d: (x=%.1f, y=%.1f, z=%.1f) km' % (i+1, x_out[i], y_out[i], z_out[i]))
    print('  Ncpu                = %d' % Ncpu)
    print('=' * 65)
    print()
    print('Runs BOTH 3D and IPA solvers for EACH wavelength.')
    print('Eight-panel figure (2×4):')
    print('  Row 1 (IPA): LWP map         | F_up IPA   | F_down IPA   | x cross-section IPA')
    print('  Row 2 (3D):  Transmitted map  | F_up 3D    | F_down 3D    | x cross-section 3D ')
    print('  Cross-sections: first wavelength, first output location, all flux components.')
    print()
    print('Key experiments:')
    print('  aod=0.0             → cloud only; see pure spectral cloud signal')
    print('  add 500, 1240 nm    → fill in the visible–NIR spectrum')
    print('  z_out=[0, 2, 5]     → compare surface / in-cloud / above-cloud spectra')
    print('  ssa=0.7             → absorbing aerosol; prominent at short wavelengths')
    print()

    t_start = time.perf_counter()
    example_04_spectral()
    t_elapsed = time.perf_counter() - t_start
    print('\nTotal wall-clock time: %dm %04.1fs' % (int(t_elapsed // 60), t_elapsed % 60))
