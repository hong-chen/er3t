"""
EaR³T Example 02 — 3D Cloud Flux (LES Cloud Field)
====================================================

What this script does
---------------------
Simulates the 2D horizontal distribution of solar irradiance (flux) at the
surface through an inhomogeneous cloud field derived from a Large-Eddy
Simulation (LES).  Both solvers are run on the same scene and the results
are shown side-by-side in a 4-panel comparison figure:

     ┌─────────────────┬─────────────────┐
     │  3D   │  F_up   │  3D   │ F_down  │   ← Row 1: full 3D RT
     ├─────────────────┼─────────────────┤
     │  IPA  │  F_up   │  IPA  │ F_down  │   ← Row 2: Independent Pixel
     └─────────────────┴─────────────────┘

Outputs
-------
* HDF5 results in  tmp-data/02_3d_cloud_flux/
* PNG plot         02_3d_cloud_flux-comparison_<wavelength>nm.png

Key physics / "aha" moments
----------------------------
3D vs IPA: same cloud field, same photon count, different assumption.
  - IPA: each vertical column is treated as an infinite homogeneous slab.
    No photon ever moves horizontally. Each pixel is independent.
  - 3D: photons scatter sideways between columns — the physical reality.

Results look similar in the domain mean but the spatial *patterns* differ:
  - IPA tends to OVERESTIMATE reflection at thick-cloud centres
    (no photons can escape sideways).
  - 3D produces bright halos next to cloud edges and smoother gradients
    (photons leaked out sideways brighten the neighbouring clear sky).
These biases directly contaminate satellite cloud optical thickness
retrievals and are the core motivation for EaR³T.

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

photons    = 1e6        # Monte Carlo photons per spectral bin.
                        #   1e6  → fast (~minutes per solver),  lower quality
                        #   1e7  → medium quality (recommended for comparison)
                        #   1e8  → production quality (~hours)
                        # Note: the script runs BOTH solvers, so total runtime
                        # is approximately 2× the single-solver time.

Nrun       = 3          # Number of independent MC runs per solver.
                        # Each run uses a different random seed.
                        # More runs → smoother images but proportionally longer runtime.
                        # 3 is a good default; try 1 for speed, 5 for extra smoothness.

z_index    = 0          # Vertical level index for plotting.
                        # The atmosphere has 21 levels: np.linspace(0, 20, 21)
                        # → z_index 0 =  0 km  (surface)
                        # → z_index 1 =  1 km
                        # → z_index 2 =  2 km  (just above cloud base ~1-2 km)
                        # → z_index 3 =  3 km  (above-cloud level — try this!)
                        # → z_index 10 = 10 km (mid-troposphere)
                        # At z_index=0 you see surface flux patterns (shadows).
                        # At z_index=3 you see above-cloud radiation field.

surface_albedo      = 0.03   # Surface reflectance.  Try: 0.03 (ocean), 0.5 (sand/desert)
                              # Higher albedo → stronger upwelling pattern (F_up map becomes
                              # much more structured).  Compare 0.03 vs 0.5 to see how
                              # surface brightness amplifies 3D cloud shadow effects.

solar_zenith_angle  = 30.0   # Solar zenith angle [°].  Try: 0 (overhead), 30, 45, 60
                              # Controls shadow LENGTH: high SZA → long shadows → large
                              # spatial separation between bright cloud tops and dark shadows.
                              # High SZA also amplifies the 3D vs IPA difference because
                              # photons must travel through more cloud edges at oblique angles.

solar_azimuth_angle = 45.0   # Solar azimuth angle [°], clockwise from North.
                              # Controls shadow DIRECTION in the x–y map.
                              # Try: 0, 45, 90, 180 — clouds cast shadows in opposite direction.
                              # The LES cloud field is not symmetric, so changing this angle
                              # moves the shadow pattern across the domain.

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Internal configuration (do not need to change) ────────────────────────
name_tag = '02_3d_cloud_flux'
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
    The atmosphere, absorption, and cloud objects are passed in (pre-built)
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


def example_02_flux_les_cloud_3d(
        wavelength=wavelength,
        overwrite=True,
        plot=True
        ):

    """
    Run 3D and IPA flux simulations on the same LES cloud field and
    produce a 4-panel side-by-side comparison figure.

    Requires data/00_er3t_mca/aux/les.nc
    """

    fdir = '%s/tmp-data/%s/example_02_flux_les_cloud_3d' % (fdir0, name_tag)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # ── Build scene (shared between both solvers) ─────────────────────────────
    # Atmosphere
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # Absorption (REPTRAN)
    fname_abs = '%s/abs.pk' % fdir
    abs0      = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, target='medium', overwrite=overwrite)

    # Cloud field from LES
    #   coarsen=[1,1,25]: average 25 LES vertical levels into 1 model layer
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    fname_les = '%s/les.pk' % fdir
    cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[1, 1, 25], overwrite=overwrite)

    # Assemble MCARaTS atmosphere objects (same for both solvers)
    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]

    # ── Run both solvers ───────────────────────────────────────────────────────
    print('\n── Running 3D solver ──')
    out_3d  = run_flux_simulation('3D',  atm0, abs0, atm_1ds, atm_3ds, fdir, Nrun=Nrun, overwrite=overwrite)

    print('\n── Running IPA solver ──')
    out_ipa = run_flux_simulation('IPA', atm0, abs0, atm_1ds, atm_3ds, fdir, Nrun=Nrun, overwrite=overwrite)

    # ── 4-panel comparison figure ─────────────────────────────────────────────
    if plot:
        z_km      = atm0.lev['altitude']['data'][z_index]
        vmin, vmax = 0.0, 1.6

        fname_png = '%s-comparison_%4.0fnm.png' % (name_tag, wavelength)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                                 gridspec_kw={'wspace': 0.08, 'hspace': 0.35})

        panels = [
            (axes[0, 0], out_3d,  'f_up',   '3D',  '$F_{\\uparrow}$'),
            (axes[0, 1], out_3d,  'f_down', '3D',  '$F_{\\downarrow}$'),
            (axes[1, 0], out_ipa, 'f_up',   'IPA', '$F_{\\uparrow}$'),
            (axes[1, 1], out_ipa, 'f_down', 'IPA', '$F_{\\downarrow}$'),
        ]

        ims = []
        for ax, out, key, solver_label, flux_label in panels:
            data = np.transpose(out.data[key]['data'][:, :, z_index])
            im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title('%s  —  %s at z=%d km' % (solver_label, flux_label, z_km), fontsize=12)
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
            ims.append(im)

        # shared colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.75])
        fig.colorbar(ims[0], cax=cbar_ax, label='Flux  [W m$^{-2}$ nm$^{-1}$]')

        # row labels
        fig.text(0.01, 0.73, '3D', va='center', ha='left', fontsize=14, fontweight='bold', rotation=90)
        fig.text(0.01, 0.29, 'IPA', va='center', ha='left', fontsize=14, fontweight='bold', rotation=90)

        fig.suptitle(
            'LES Cloud Flux Comparison: 3D vs IPA\n'
            '$\\lambda$=%.0f nm,  SZA=%.0f°,  albedo=%.2f,  photons=%.0e'
            % (wavelength, solar_zenith_angle, surface_albedo, photons),
            fontsize=13
        )

        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        print('\nPlot saved: %s' % fname_png)

        # Print domain-mean comparison
        print('\n── Domain-mean flux at z=%.0f km (z_index=%d) ──────────────────────' % (z_km, z_index))
        for label, out in [('3D ', out_3d), ('IPA', out_ipa)]:
            f_up  = out.data['f_up']['data'][:, :, z_index].mean()
            f_dn  = out.data['f_down']['data'][:, :, z_index].mean()
            print('  %s:  F_up = %.4f   F_down = %.4f   F_net = %.4f  [W/m²/nm]'
                  % (label, f_up, f_dn, f_dn - f_up))
        print('  (Domain means should be nearly identical — the *patterns* differ!)')




if __name__ == '__main__':

    print('=' * 60)
    print('EaR³T Example 02 — 3D Cloud Flux (LES), 3D vs IPA comparison')
    print('  wavelength          = %.1f nm'  % wavelength)
    print('  photons             = %.0e (× 2 solvers)'  % photons)
    print('  Nrun                = %d'        % Nrun)
    print('  z_index             = %d  (z = %.0f km)' % (z_index, z_index))
    print('  surface_albedo      = %.2f'      % surface_albedo)
    print('  solar_zenith_angle  = %.1f°'     % solar_zenith_angle)
    print('  solar_azimuth_angle = %.1f°'     % solar_azimuth_angle)
    print('  Ncpu                = %d'        % Ncpu)
    print('=' * 60)
    print()
    print('This script runs BOTH 3D and IPA solvers automatically.')
    print('The output is a 4-panel figure with rows = solver, cols = F_up / F_down.')
    print('Look at the PATTERNS, not just the means — the domain means are similar!')
    print('Tip: change z_index to 2 or 3 to view the above-cloud radiation field.')
    print()

    example_02_flux_les_cloud_3d(wavelength=wavelength)
