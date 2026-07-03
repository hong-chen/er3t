"""
EaR³T Example 05 — Plot-only version
======================================
Reads the HDF5 and scene files written by 05_3d_cloud_radiance.py
and regenerates all three figures.  No er3t, no RT, no long waits.

Required input files (created by 05_3d_cloud_radiance.py):
    <fdir>/mca-out-rad-3d.h5
    <fdir>/mca-out-rad-ipa.h5
    <fdir>/scene_info.npz

Outputs
-------
  05_3d_cloud_radiance-comparison_<wl>nm.png  — 4-panel reflectance comparison
  05_3d_cloud_radiance-map_<wl>nm.png         — LWP and COT maps
  05_3d_cloud_radiance-r_tau_<wl>nm.png       — IPA reflectance vs COT
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import h5py

rcParams['font.size'] = 13


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     STUDENT CONTROL BLOCK                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣

# -- Path to the folder produced by 05_3d_cloud_radiance.py ----------------
fdir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'tmp-data', '05_3d_cloud_radiance',
    'example_05_rad_les_cloud_3d'
)

# -- These must match the values used in the RT run -------------------------
wavelength         = 650.0   # [nm]
solar_zenith_angle = 30.0    # [°]
solar_azimuth_angle = 45.0   # [°]
sensor_zenith_angle = 0.0    # [°]
sensor_azimuth_angle = 0.0   # [°]
surface_albedo     = 0.03    # [–]

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Load RT output ────────────────────────────────────────────────────────
fname_3d    = os.path.join(fdir, 'mca-out-rad-3d.h5')
fname_ipa   = os.path.join(fdir, 'mca-out-rad-ipa.h5')
fname_scene = os.path.join(fdir, 'scene_info.npz')

for f in [fname_3d, fname_ipa, fname_scene]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            'Required file not found:\n  %s\n'
            'Run 05_3d_cloud_radiance.py first to generate RT output.' % f)

with h5py.File(fname_3d, 'r') as h:
    rad_3d = h['mean/rad'][:]
    toa    = float(h['mean/toa'][()])

with h5py.File(fname_ipa, 'r') as h:
    rad_ipa = h['mean/rad'][:]

scene = np.load(fname_scene)
lwp   = scene['lwp']
cot   = scene['cot']
x_km  = scene['x_km']
y_km  = scene['y_km']
Nx    = int(scene['Nx'])
Ny    = int(scene['Ny'])

print('Loaded:')
print('  rad_3d  shape: %s' % str(rad_3d.shape))
print('  rad_ipa shape: %s' % str(rad_ipa.shape))
print('  lwp     shape: %s' % str(lwp.shape))
print('  toa = %.4f' % toa)


# ── Reflectance  R = π L / (E₀ cos θ_s) ─────────────────────────────────
mu0  = np.cos(np.deg2rad(solar_zenith_angle))
norm = np.pi / (toa * mu0) if toa * mu0 > 0 else 1.0
ref_3d  = rad_3d  * norm
ref_ipa = rad_ipa * norm
diff    = ref_3d  - ref_ipa

dx_val    = x_km[1] - x_km[0]
dy_val    = y_km[1] - y_km[0]
extent_km = [0.0, Nx * dx_val, 0.0, Ny * dy_val]

print('\nDomain-mean reflectance:')
print('  3D:          %.4f' % ref_3d.mean())
print('  IPA:         %.4f' % ref_ipa.mean())
print('  mean|3D-IPA|:%.4f' % np.abs(diff).mean())

name_tag = '05_3d_cloud_radiance'

vmax_ref  = min(1.0, float(np.percentile(np.maximum(ref_3d, ref_ipa), 99.5)))
vlim_diff = max(0.01, float(np.percentile(np.abs(diff), 99)))

im_kw = dict(origin='lower', extent=extent_km, aspect='equal',
             interpolation='none')

suptitle_str = (
    'LES Cloud Radiance: 3D vs IPA  —  %.0f nm\n'
    'SZA = %.0f°,  SAA = %.0f°,  VZA = %.0f°,  VAA = %.0f°,  albedo = %.2f'
    % (wavelength,
       solar_zenith_angle, solar_azimuth_angle,
       sensor_zenith_angle, sensor_azimuth_angle,
       surface_albedo)
)


# ═════════════════════════════════════════════════════════════════════════
# Figure 1 — 4-panel comparison (reflectance images, diff, scatter)
# ═════════════════════════════════════════════════════════════════════════
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
print('\nComparison saved: %s' % fname1)


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

fig2.suptitle('LES cloud scene — %.0f nm' % wavelength, fontsize=12, y=1.01)

fname2 = '%s-map_%.0fnm.png' % (name_tag, wavelength)
fig2.savefig(fname2, bbox_inches='tight')
plt.close(fig2)
print('Map saved:        %s' % fname2)


# ═════════════════════════════════════════════════════════════════════════
# Two-stream analytical curves  (Coakley & Chylek 1975, JAS)
# Reference: http://www.atmo.arizona.edu/students/courselinks/
#            spring09/atmo551b/2-stream%20solution.pdf
#
# Assumptions for liquid water clouds at visible wavelengths:
#   g  = 0.85   asymmetry parameter
#   w  = 0.9999 single-scattering albedo (nearly conservative)
#   mu = cos(SZA)
#   a  = surface_albedo
# ═════════════════════════════════════════════════════════════════════════

def twostream_no_abs(tau, a=0.03, g=0.85, mu=0.866):
    """
    Conservative two-stream (w=1, no absorption). Simple closed form.
    r = (tau + a*x) / (tau + x),  x = 2*mu / ((1-g)*(1-a))
    Bounded: r → a as tau→0, r → 1 as tau→∞.
    """
    x = 2 * mu / (1. - g) / (1. - a)
    return (tau + a * x) / (tau + x)


def coakley_black_surface(tau, w=0.9999, g=0.85, mu=0.866):
    """
    Coakley & Chylek (1975) eq. 13 — full two-stream, black surface (a=0).
    Bounded: R_c → 0 as tau→0, R_c → (U-1)/(U+1) < 1 as tau→∞.
    """
    beta = (1 - g) / 2.
    U    = np.sqrt((1 - w + 2 * w * beta) / (1 - w))
    al   = np.sqrt(1 - w) * np.sqrt(1 - w + 2 * w * beta)
    ep   = np.exp( al * tau / mu)
    em   = np.exp(-al * tau / mu)
    num  = (U + 1) * (U - 1) * (ep - em)
    den  = (U + 1)**2 * ep - (U - 1)**2 * em
    return num / den


def coakley_with_surface(tau, w=0.9999, g=0.85, mu=0.866, a=0.03):
    """
    Coakley & Chylek (1975) eq. 13 for cloud reflectance R_c, then
    adding formula to include Lambertian surface:
        R_total = R_c + T_c^2 * a / (1 - R_c * a)
    with T_c ≈ 1 - R_c (valid for near-conservative scattering, w ≈ 1).
    Bounded: R_total → a as tau→0, R_total → 1 as tau→∞.
    """
    R_c = coakley_black_surface(tau, w=w, g=g, mu=mu)
    T_c = 1.0 - R_c   # near-conservative approximation
    return np.clip(R_c + T_c**2 * a / (1.0 - R_c * a), 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════
# Figure 3 — R(tau): IPA reflectance vs COT + two-stream theory
# ═════════════════════════════════════════════════════════════════════════
cot_flat = cot.ravel()
ipa_flat = ref_ipa.ravel()

cot_max = float(np.percentile(cot_flat[cot_flat > 0], 99.5)) if (cot_flat > 0).any() else 50.0
cot_max = max(cot_max, 1.0)

# Analytical curves over the same COT range
tau_theory = np.linspace(0, cot_max, 500)
mu0        = np.cos(np.deg2rad(solar_zenith_angle))
g_liq      = 0.85    # asymmetry parameter, liquid water clouds (visible)
w_liq      = 0.9999  # single-scattering albedo, liquid water (visible)

r_noabs  = twostream_no_abs(tau_theory,     a=surface_albedo, g=g_liq, mu=mu0)
r_black  = coakley_black_surface(tau_theory, w=w_liq, g=g_liq, mu=mu0)
r_surf   = coakley_with_surface(tau_theory,  w=w_liq, g=g_liq, mu=mu0, a=surface_albedo)

# Fit Coakley eq. 23 to the binned MC median (a fixed, g and w free)
from scipy.optimize import curve_fit

n_bins    = 120
cot_bins  = np.linspace(0, cot_max, n_bins + 1)
bin_ctrs  = 0.5 * (cot_bins[:-1] + cot_bins[1:])
bin_med   = np.full(n_bins, np.nan)
for i in range(n_bins):
    mask = (cot_flat >= cot_bins[i]) & (cot_flat < cot_bins[i + 1])
    if mask.sum() > 20:
        bin_med[i] = np.median(ipa_flat[mask])
valid   = np.isfinite(bin_med)
x_fit   = bin_ctrs[valid]
y_fit   = bin_med[valid]

def _coakley_fit(tau, w, g):
    return coakley_with_surface(tau, w=w, g=g, mu=mu0, a=surface_albedo)

try:
    popt, _ = curve_fit(_coakley_fit, x_fit, y_fit,
                        p0=[0.9999, 0.85],
                        bounds=([0.5, 0.0], [0.999999, 0.9999]),
                        maxfev=5000)
    w_fit, g_fit = popt
    r_fit   = coakley_with_surface(tau_theory, w=w_fit, g=g_fit,
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
