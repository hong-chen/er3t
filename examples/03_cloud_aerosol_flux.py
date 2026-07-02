"""
EaR³T Example 03 — 3D Cloud + 1D Aerosol Layer, Flux (3D vs IPA)
=================================================================

What this script does
---------------------
Builds on Example 02 by adding a horizontally homogeneous (1D) aerosol layer
to the LES cloud scene, then runs both 3D and IPA solvers and produces three
separate figures:

Outputs
-------
* HDF5 results in  tmp-data/03_cloud_aerosol_flux/

* <name>_map_<wl>nm.png
    2 rows × 3 columns of horizontal flux maps at a chosen altitude level.
    Axes in km.  A star marks the user-specified location (x0, y0).
    Row 1: 3D solver  — F_up | F_down (total) | F_down (diffuse)
    Row 2: IPA solver — F_up | F_down (total) | F_down (diffuse)

* <name>_profile_<wl>nm.png
    Two vertical profile panels side by side:
    Left  — domain-mean profiles (3D solid, IPA dashed)
    Right — profiles at the point (x0, y0) (3D solid, IPA dashed)
    A horizontal dotted line marks the map altitude level.

* <name>_xsection_<wl>nm.png
    Two cross-section panels:
    Left  — flux / TOA along the full x range at y = y0, z = z_index
    Right — flux / TOA along the full y range at x = x0, z = z_index
    A vertical dashed line marks the (x0, y0) position; stars on each curve
    mark the exact point value.

Key physics / "aha" moments
----------------------------
Compare with Example 02 (set aod=0.0) to isolate the aerosol effect:
  - An absorbing aerosol (SSA < 1) reduces downwelling and warms the
    aerosol layer.  A scattering aerosol (SSA ≈ 1) redirects flux but
    absorbs little — the domain-mean surface downwelling changes less
    than you might expect.
  - The diffuse panel is especially sensitive: aerosol converts direct-beam
    photons into diffuse ones, visible as a shift between columns 2 and 3.
  - 3D always produces more diffuse radiation than IPA because photons
    escape from cloud sides and scatter through neighbouring clear columns —
    a horizontal transport effect IPA cannot capture regardless of aerosol
    position or loading.
  - Place the aerosol layer inside the cloud (z_bot ≈ z_top of cloud) to
    see a direct aerosol–cloud interaction; place it well above the cloud
    to see indirect illumination effects.
  - The xsection panels show how flux varies horizontally: thick-cloud
    columns suppress downwelling (shadows) while clear-sky columns
    between clouds can receive MORE flux than IPA predicts (photon leakage
    from cloud sides).

Aerosol parameters you can tune (in the student control block):
  aod   — aerosol optical depth (total column; controls layer thickness)
  ssa   — single scattering albedo (1.0 = pure scattering, 0.0 = pure absorber)
  asy   — asymmetry parameter (0 = isotropic, ~0.7 = typical dust/smoke)
  z_bot, z_top — bottom and top altitudes of the layer [km]; can overlap the cloud

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

import er3t


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     STUDENT CONTROL BLOCK                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣

wavelength = 650.0      # Wavelength in nm.  Try: 400, 500, 650, 860, 1640

photons    = 1e8        # Monte Carlo photons per spectral bin.
                        #   5e5  → fast (~minutes per solver)
                        #   1e6  → medium quality
                        #   1e7  → production quality
                        # Script runs BOTH solvers → total ≈ 2× single-solver time.

Nrun       = 10         # Independent MC runs per solver.

z_index    = 0          # Vertical level index for the horizontal map panels.
                        # Atmosphere has 21 interfaces at 0, 1, … 20 km.
                        # → z_index 0  = 0 km   (surface — shadows most visible here)
                        # → z_index 3  = 3 km   (above cloud — try this!)
                        # → z_index 10 = 10 km  (mid-troposphere)

surface_albedo      = 0.8    # Surface reflectance.  Try: 0.03 (ocean), 0.5 (sand/desert)
solar_zenith_angle  = 30.0   # Solar zenith angle [°].  Try: 0, 30, 45, 60
solar_azimuth_angle = 45.0   # Solar azimuth [°], clockwise from North.

# ── Output location ───────────────────────────────────────────────────────
x0 = 10.0               # x coordinate of the output location [km].
                        # Domain spans 0–48 km in x.
                        # A star is plotted at (x0, y0) on all six maps,
                        # the profile at this point is shown alongside the
                        # domain-mean, and the cross-sections pass through it.

y0 = 25.0               # y coordinate of the output location [km].
                        # Domain spans 0–48 km in y.

# ── Aerosol layer parameters ──────────────────────────────────────────────
aod   = 0.4             # Aerosol optical depth (total column).
                        #   0.0 → no aerosol (reproduce Example 02 exactly)
                        #   0.2 → light haze
                        #   0.4 → moderate aerosol (default)
                        #   1.0 → very thick (e.g., heavy smoke or desert dust)

ssa   = 0.9             # Single scattering albedo.
                        #   1.0 → purely scattering (no absorption)
                        #   0.9 → clean continental aerosol (default)
                        #   0.7 → absorbing aerosol (black carbon, urban smoke)

asy   = 0.6             # Asymmetry parameter g (Henyey-Greenstein phase function).
                        #   0.0 → isotropic scattering
                        #   0.6 → typical marine / continental aerosol (default)
                        #   0.7 → mineral dust

z_bot = 5.0             # Aerosol layer bottom altitude [km].
                        #   Cloud occupies roughly 0.5–2.5 km.
                        #   5–8 km (default) → aerosol decoupled above the cloud.
                        #   Try 0.5–2.5 km to collocate aerosol with the cloud.

z_top = 8.0             # Aerosol layer top altitude [km].
                        #   Thicker layer (larger z_top − z_bot) → lower extinction
                        #   for the same AOD.  Try: 5–6 km (thin) vs 5–12 km (deep).

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Internal configuration (do not need to change) ────────────────────────
name_tag = '03_cloud_aerosol_flux'
fdir0    = er3t.common.fdir_examples
Ncpu     = max(1, multiprocessing.cpu_count() - 2)
rcParams['font.size'] = 13
# ──────────────────────────────────────────────────────────────────────────


def run_flux_simulation(
        solver,
        atm0, abs0, atm_1ds, atm_3ds,
        fdir,
        Nrun=3,
        overwrite=True
        ):
    """
    Run MCARaTS flux simulation for one solver and return the output object.
    Atmosphere, absorption, cloud, and aerosol objects are passed in pre-built
    so the heavy setup is not repeated for each solver.
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
            fdir='%s/%4.4d/flux_%s' % (fdir, wavelength, solver.lower()),
            photons=photons,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=overwrite
            )

    fname_h5 = '%s/mca-out-flux-%s.h5' % (fdir, solver.lower())
    out0 = er3t.rtm.mca.mca_out_ng(
            fname=fname_h5,
            mca_obj=mca0, abs_obj=abs0,
            mode='mean', squeeze=True,
            verbose=True, overwrite=overwrite
            )
    return out0


def example_03_flux_les_cloud_aerosol(
        wavelength=wavelength,
        overwrite=True,
        plot=True
        ):

    """
    Run 3D and IPA flux simulations on an LES cloud field with a 1D aerosol
    layer, and produce three figure files: maps, profiles, and cross-sections.

    Requires data/00_er3t_mca/aux/les.nc
    """

    fdir = '%s/tmp-data/%s/example_03_flux_les_cloud_aerosol' % (fdir0, name_tag)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # ── Build scene (shared between both solvers) ─────────────────────────────
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    fname_abs = '%s/abs.pk' % fdir
    abs0      = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs,
                                      atm_obj=atm0, target='medium', overwrite=overwrite)

    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    fname_les = '%s/les.pk' % fdir
    cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les,
                                      coarsen=[1, 1, 25], overwrite=overwrite)

    atm3d0 = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0,
                                       fname='%s/mca_atm_3d.bin' % fdir,
                                       overwrite=overwrite)

    atm1d0 = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    if aod > 0.0:
        aer_ext = aod / (z_top - z_bot) / 1000.0
        atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy,
                               z_bottom=z_bot, z_top=z_top)
        print('Aerosol layer: AOD=%.2f  SSA=%.2f  asy=%.2f  z=[%.1f–%.1f km]'
              % (aod, ssa, asy, z_bot, z_top))
    else:
        print('Aerosol: none (aod=0) — reproducing Example 02 result')

    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]

    # ── Run both solvers ──────────────────────────────────────────────────────
    print('\n── Running 3D solver ──')
    out_3d  = run_flux_simulation('3D',  atm0, abs0, atm_1ds, atm_3ds,
                                  fdir, Nrun=Nrun, overwrite=overwrite)
    print('\n── Running IPA solver ──')
    out_ipa = run_flux_simulation('IPA', atm0, abs0, atm_1ds, atm_3ds,
                                  fdir, Nrun=Nrun, overwrite=overwrite)

    if not plot:
        return

    # ── Shared quantities ─────────────────────────────────────────────────────
    alt_prof = atm0.lev['altitude']['data']   # 21 level interfaces, 0–20 km
    z_km     = alt_prof[z_index]              # altitude of the map level

    # Cloud pixel grid
    dx = float(cld0.lay['dx']['data'])        # km per pixel in x
    dy = float(cld0.lay['dy']['data'])        # km per pixel in y
    Nx = int(cld0.lay['nx']['data'])
    Ny = int(cld0.lay['ny']['data'])
    x_km = (np.arange(Nx) + 0.5) * dx        # pixel-centre x coords [km]
    y_km = (np.arange(Ny) + 0.5) * dy        # pixel-centre y coords [km]
    extent_km = [0.0, Nx * dx, 0.0, Ny * dy] # for imshow

    # Output location → pixel indices
    xi0 = int(np.clip(np.floor(x0 / dx), 0, Nx - 1))
    yi0 = int(np.clip(np.floor(y0 / dy), 0, Ny - 1))
    # Pixel-centre km of the chosen pixel (for labelling)
    x0_c = x_km[xi0]
    y0_c = y_km[yi0]

    # TOA irradiance for normalisation in xsection
    toa = np.sum(abs0.coef['solar']['data'] * abs0.coef['weight']['data'])

    flux_keys   = ['f_up', 'f_down', 'f_down_diffuse']
    flux_labels = ['$F_{\\uparrow}$', '$F_{\\downarrow}$ total', '$F_{\\downarrow}$ diffuse']
    flux_colors = ['firebrick', 'steelblue', 'seagreen']

    suptitle_str = (
        'LES Cloud + Aerosol Flux: 3D vs IPA\n'
        '$\\lambda$=%.0f nm,  SZA=%.0f°,  albedo=%.2f,  photons=%.0e\n'
        'AOD=%.2f  SSA=%.2f  asy=%.2f  z$_{aer}$=[%.1f–%.1f km]'
        % (wavelength, solar_zenith_angle, surface_albedo, photons,
           aod, ssa, asy, z_bot, z_top)
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 1 — Map  (2×3 grid, km axes, star at (x0, y0))
    # ═════════════════════════════════════════════════════════════════════════
    vmin, vmax = 0.0, 1.6

    fig_map, axes_map = plt.subplots(2, 3, figsize=(18, 10))
    fig_map.subplots_adjust(wspace=0.25, hspace=0.40)

    map_panels = [
        (axes_map[0, 0], out_3d,  'f_up',          '3D',  '$F_{\\uparrow}$'),
        (axes_map[0, 1], out_3d,  'f_down',         '3D',  '$F_{\\downarrow}$ (total)'),
        (axes_map[0, 2], out_3d,  'f_down_diffuse', '3D',  '$F_{\\downarrow}$ (diffuse)'),
        (axes_map[1, 0], out_ipa, 'f_up',           'IPA', '$F_{\\uparrow}$'),
        (axes_map[1, 1], out_ipa, 'f_down',         'IPA', '$F_{\\downarrow}$ (total)'),
        (axes_map[1, 2], out_ipa, 'f_down_diffuse', 'IPA', '$F_{\\downarrow}$ (diffuse)'),
    ]

    ims = []
    for ax, out, key, slabel, flabel in map_panels:
        data = np.transpose(out.data[key]['data'][:, :, z_index])
        im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax,
                       origin='lower', extent=extent_km, aspect='equal',
                       interpolation='none')
        # Star and horizontal guide line at output location
        ax.axhline(y0_c, color='white', lw=1.0, ls='--', alpha=0.8, zorder=4)
        ax.scatter(x0_c, y0_c, marker='*', s=200, color='white',
                   edgecolors='black', linewidths=0.7, zorder=5)
        ax.set_title('%s  —  %s\nz = %.1f km' % (slabel, flabel, z_km), fontsize=11)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ims.append(im)

    all_map_axes = axes_map.ravel().tolist()
    fig_map.colorbar(ims[0], ax=all_map_axes, location='bottom',
                     shrink=0.6, pad=0.08,
                     label='Flux  [W m$^{-2}$ nm$^{-1}$]')
    fig_map.text(0.005, 0.73, '3D',  va='center', ha='left',
                 fontsize=13, fontweight='bold', rotation=90)
    fig_map.text(0.005, 0.29, 'IPA', va='center', ha='left',
                 fontsize=13, fontweight='bold', rotation=90)
    fig_map.suptitle(suptitle_str + '\n★ = (%.1f, %.1f) km' % (x0_c, y0_c),
                     fontsize=11, y=1.01)

    fname_map = '%s-map_%.0fnm.png' % (name_tag, wavelength)
    fig_map.savefig(fname_map, bbox_inches='tight')
    plt.close(fig_map)
    print('Map saved:       %s' % fname_map)

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 2 — Profile  (domain-mean | point at (x0, y0))
    # ═════════════════════════════════════════════════════════════════════════
    prof_mean_3d  = {k: out_3d.data[k]['data'].mean(axis=(0, 1))  for k in flux_keys}
    prof_mean_ipa = {k: out_ipa.data[k]['data'].mean(axis=(0, 1)) for k in flux_keys}
    prof_pt_3d    = {k: out_3d.data[k]['data'][xi0, yi0, :]       for k in flux_keys}
    prof_pt_ipa   = {k: out_ipa.data[k]['data'][xi0, yi0, :]      for k in flux_keys}

    def _draw_profile(ax, prof_3d, prof_ipa, title):
        """Draw a single vertical profile panel."""
        if aod > 0.0:
            ax.axhspan(z_bot, z_top, color='wheat', alpha=0.50, zorder=0,
                       label='Aerosol (%.1f–%.1f km)' % (z_bot, z_top))
        ax.axhspan(0.5, 2.5, color='lightsteelblue', alpha=0.35, zorder=0,
                   label='Cloud (~0.5–2.5 km)')
        for k, lbl, col in zip(flux_keys, flux_labels, flux_colors):
            ax.plot(prof_3d[k],  alt_prof, color=col, lw=2.0, ls='-',  marker='o', ms=4,
                    label='%s — 3D' % lbl)
            ax.plot(prof_ipa[k], alt_prof, color=col, lw=1.5, ls='--', marker='s', ms=4,
                    label='%s — IPA' % lbl)
        ax.axhline(z_km, color='dimgray', lw=0.9, ls=':',
                   label='map level (%.0f km)' % z_km)
        ax.set_xlabel('Flux [W m$^{-2}$ nm$^{-1}$]', fontsize=11)
        ax.set_ylabel('Altitude [km]', fontsize=11)
        ax.set_ylim(0, 20)
        ax.set_xlim(left=0)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.0, loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.25)

    fig_prof, (ax_mean, ax_pt) = plt.subplots(1, 2, figsize=(14, 8),
                                               sharey=True)
    fig_prof.subplots_adjust(wspace=0.15)

    _draw_profile(ax_mean, prof_mean_3d, prof_mean_ipa,
                  'Domain-mean vertical profile\n3D (—) vs IPA (- -)')
    _draw_profile(ax_pt,   prof_pt_3d,   prof_pt_ipa,
                  'Point profile at (%.1f, %.1f) km\n3D (—) vs IPA (- -)' % (x0_c, y0_c))

    fig_prof.suptitle(suptitle_str, fontsize=11, y=1.01)

    fname_prof = '%s-profile_%.0fnm.png' % (name_tag, wavelength)
    fig_prof.savefig(fname_prof, bbox_inches='tight')
    plt.close(fig_prof)
    print('Profile saved:   %s' % fname_prof)

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 3 — Cross-sections  (x slice at y=y0 | y slice at x=x0)
    # ═════════════════════════════════════════════════════════════════════════
    fig_xs, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 6))
    fig_xs.subplots_adjust(wspace=0.30)

    def _draw_xsection(ax, coords_km, pt_km, coord_label, pt_label,
                       slices_3d, slices_ipa):
        """Draw one cross-section panel.
        slices_3d / slices_ipa: dict {flux_key: 1-D array along the cross-section}.
        """
        for k, lbl, col in zip(flux_keys, flux_labels, flux_colors):
            vals_3d  = slices_3d[k]  / toa
            vals_ipa = slices_ipa[k] / toa
            ax.plot(coords_km, vals_3d,  color=col, lw=2.0, ls='-',  label='%s 3D'  % lbl)
            ax.plot(coords_km, vals_ipa, color=col, lw=1.5, ls='--', label='%s IPA' % lbl)
            # Star at the chosen point
            idx_pt = np.argmin(np.abs(coords_km - pt_km))
            ax.scatter(coords_km[idx_pt], vals_3d[idx_pt],
                       marker='*', s=150, color=col,
                       edgecolors='k', linewidths=0.5, zorder=5)

        ax.axvline(pt_km, color='dimgray', lw=0.9, ls=':',
                   label='%s = %.1f km' % (pt_label, pt_km))
        ax.set_xlabel('%s [km]' % coord_label)
        ax.set_ylabel('Flux / TOA')
        ax.set_xlim(coords_km[0], coords_km[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85, ncol=2)

    # x cross-section: all x at y = yi0, z = z_index
    x_slices_3d  = {k: out_3d.data[k]['data'][:, yi0, z_index]  for k in flux_keys}
    x_slices_ipa = {k: out_ipa.data[k]['data'][:, yi0, z_index] for k in flux_keys}
    _draw_xsection(ax_x, x_km, x0_c, 'x', 'x₀',
                   x_slices_3d, x_slices_ipa)
    ax_x.set_title('x cross-section  at  y = %.1f km,  z = %.1f km\n'
                   '3D (—) vs IPA (- -)' % (y0_c, z_km))

    # y cross-section: all y at x = xi0, z = z_index
    y_slices_3d  = {k: out_3d.data[k]['data'][xi0, :, z_index]  for k in flux_keys}
    y_slices_ipa = {k: out_ipa.data[k]['data'][xi0, :, z_index] for k in flux_keys}
    _draw_xsection(ax_y, y_km, y0_c, 'y', 'y₀',
                   y_slices_3d, y_slices_ipa)
    ax_y.set_title('y cross-section  at  x = %.1f km,  z = %.1f km\n'
                   '3D (—) vs IPA (- -)' % (x0_c, z_km))

    fig_xs.suptitle(suptitle_str + '\n★ = (%.1f, %.1f) km,  z = %.1f km'
                    % (x0_c, y0_c, z_km), fontsize=11, y=1.02)

    fname_xs = '%s-xsection_%.0fnm.png' % (name_tag, wavelength)
    fig_xs.savefig(fname_xs, bbox_inches='tight')
    plt.close(fig_xs)
    print('X-section saved: %s' % fname_xs)

    # ── Console summary ────────────────────────────────────────────────────────
    print('\n── Domain-mean flux at z=%.0f km (z_index=%d) ───────────────────' % (z_km, z_index))
    for label, out in [('3D ', out_3d), ('IPA', out_ipa)]:
        f_up  = out.data['f_up']['data'][:, :, z_index].mean()
        f_dn  = out.data['f_down']['data'][:, :, z_index].mean()
        f_dif = out.data['f_down_diffuse']['data'][:, :, z_index].mean()
        print('  %s:  F_up=%.4f  F_down=%.4f  (direct=%.4f  diffuse=%.4f)  [W/m²/nm]'
              % (label, f_up, f_dn, f_dn - f_dif, f_dif))
    print()
    print('── Point flux at (%.1f, %.1f) km,  z=%.0f km ───────────────────'
          % (x0_c, y0_c, z_km))
    for label, out in [('3D ', out_3d), ('IPA', out_ipa)]:
        f_up  = out.data['f_up']['data'][xi0, yi0, z_index]
        f_dn  = out.data['f_down']['data'][xi0, yi0, z_index]
        f_dif = out.data['f_down_diffuse']['data'][xi0, yi0, z_index]
        print('  %s:  F_up=%.4f  F_down=%.4f  (direct=%.4f  diffuse=%.4f)  [W/m²/nm]'
              % (label, f_up, f_dn, f_dn - f_dif, f_dif))
    print('  Tip: set aod=0.0 to reproduce Example 02 and compare.')


if __name__ == '__main__':

    print('=' * 60)
    print('EaR³T Example 03 — Cloud + Aerosol Flux (3D vs IPA)')
    print('  wavelength          = %.1f nm'  % wavelength)
    print('  photons             = %.0e (× 2 solvers)' % photons)
    print('  Nrun                = %d'        % Nrun)
    print('  z_index             = %d  (z = %.0f km)' % (z_index, z_index))
    print('  surface_albedo      = %.2f'      % surface_albedo)
    print('  solar_zenith_angle  = %.1f°'     % solar_zenith_angle)
    print('  solar_azimuth_angle = %.1f°'     % solar_azimuth_angle)
    print('  Output location     = (%.1f, %.1f) km' % (x0, y0))
    print('  Aerosol:  AOD=%.2f  SSA=%.2f  asy=%.2f  z=[%.1f–%.1f km]'
          % (aod, ssa, asy, z_bot, z_top))
    print('  Ncpu                = %d'        % Ncpu)
    print('=' * 60)
    print()
    print('Runs BOTH 3D and IPA solvers → three output figures:')
    print('  *_map_*.png       — 6 flux maps with km axes and location star')
    print('  *_profile_*.png   — domain-mean and point vertical profiles')
    print('  *_xsection_*.png  — x and y cross-sections at (x0,y0), flux/TOA')
    print()
    print('Key experiments:')
    print('  aod=0.0         → reproduce Example 02 (cloud only)')
    print('  ssa=0.9 vs 0.7  → scattering vs absorbing aerosol')
    print('  z_bot=2.0       → aerosol near cloud top (strong interaction)')
    print('  z_index=3       → view above-cloud level instead of surface')
    print('  x0, y0          → move the star to a cloudy vs clear-sky pixel')
    print()

    import time
    t_start = time.time()

    example_03_flux_les_cloud_aerosol(wavelength=wavelength)

    elapsed = time.time() - t_start
    print()
    print('=' * 60)
    print('Total elapsed time: %d min %02d sec' % (int(elapsed) // 60, int(elapsed) % 60))
    print('=' * 60)
