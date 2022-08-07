import os
import h5py
import numpy as np
import datetime
import time
from scipy.io import readsav
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_gen_hem as cld_gen
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca
from er3t.util import cal_r_twostream


def cal_mca_rad_cld_gen_hem(
        fdir='tmp-data/06_cld-gen_rad-sim/cld_gen_hem',
        wavelength=650.0,
        solver='3D',
        overwrite=True,
        plot=True
        ):

    """
    Calculating radiance field for a randomly generated cloud field (nadir radiance at
    the satellite altitude of 705km)

    Mie phase function is used instead of HG.
    """

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    # =============================================================================
    levels    = np.linspace(0.0, 20.0, 201)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =============================================================================


    # define an absorption object
    # =============================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # =============================================================================


    # define an cloud object (use cloud generator)
    # =============================================================================
    fname_cld = '%s/cld.pk' % fdir
    cld0 = cld_gen(
            fname=fname_cld,
            radii=[1.0, 2.0, 4.0],
            weights=[0.6, 0.3, 0.1],
            altitude=np.arange(2.0, 5.01, 0.1),
            cloud_frac_tgt=0.2,
            w2h_ratio=2.0,
            min_dist=1.5,
            overlap=False,
            overwrite=overwrite
            )
    # =============================================================================

    # define mca_sca object
    # =============================================================================
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =============================================================================


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # =============================================================================
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm3d0  = mca_atm_3d(fname='%s/mca_atm_3d.bin' % fdir, cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, overwrite=overwrite)

    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]
    # =============================================================================


    # define mcarats object
    # =============================================================================
    # run mcarats
    mca0 = mcarats_ng(
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
            fdir='%s/%4.4d/cld3d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            photons=1e8,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=12,
            mp_mode='py',
            overwrite=overwrite
            )
    # =============================================================================


    # define mcarats output object
    # =============================================================================
    fname_h5 = '%s/mca-out-rad-%s_cld3d.h5' % (fdir, solver.lower())
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    # =============================================================================


    # plot
    # =============================================================================
    if plot:
        fname_png = '06_cld-gen_rad-sim_%s.png' % solver.lower()

        fig = plt.figure(figsize=(16, 5.0))

        ax1 = fig.add_subplot(131, projection='3d')
        cmap = mpl.cm.get_cmap('jet').copy()
        cs = ax1.plot_surface(cld0.x_3d[:, :, 0], cld0.y_3d[:, :, 0], cld0.lev['cth_2d']['data'], cmap=cmap, alpha=0.8, antialiased=False)
        ax1.set_zlim((0, 10))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D View')

        ax2 = fig.add_subplot(132)
        cs = ax2.imshow(cld0.lev['cot_2d']['data'].T, cmap=cmap, origin='lower', vmin=0.0, vmax=80.0)
        ax2.set_xlabel('X Index')
        ax2.set_ylabel('Y Index')
        ax2.set_title('Cloud Optical Thickness')

        ax3 = fig.add_subplot(133)
        cs = ax3.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
        ax3.set_xlabel('X Index')
        ax3.set_ylabel('Y Index')
        ax3.set_title('Radiance at %.2f nm (%s Mode)' % (wavelength, solver))

        plt.subplots_adjust(wspace=0.4)
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    # =============================================================================


def plot_cld_gen_hem(
        fdir='tmp-data/06_cld-gen_rad-sim/cld_gen_hem',
        overwrite=True,
        plot=True
        ):

    """
    Visualize a cloud field generated from the cloud generator
    """

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an cloud object (use cloud generator)
    # =============================================================================
    fname_cld = '%s/cld.pk' % fdir
    cld0 = cld_gen(
            fname=fname_cld,
            radii=[1.0, 2.0, 4.0],
            weights=[0.6, 0.3, 0.1],
            altitude=np.arange(2.0, 5.01, 0.1),
            cloud_frac_tgt=0.2,
            w2h_ratio=2.0,
            min_dist=1.5,
            overlap=False,
            overwrite=overwrite
            )
    # =============================================================================

    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if plot:
        fname_png = '06_cld-gen_rad-sim.png'

        fig = plt.figure(figsize=(12, 5.5))

        ax1 = fig.add_subplot(121, projection='3d')
        cmap = mpl.cm.get_cmap('jet').copy()
        cs = ax1.plot_surface(cld0.x_3d[:, :, 0], cld0.y_3d[:, :, 0], cld0.lev['cth_2d']['data'], cmap=cmap, alpha=0.8, antialiased=False)
        ax1.set_zlim((0, 10))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D View')

        ax2 = fig.add_subplot(122)
        cs = ax2.imshow(cld0.lev['cot_2d']['data'].T, cmap=cmap, origin='lower', vmin=0.0, vmax=80.0)
        ax2.set_xlabel('X Index')
        ax2.set_ylabel('Y Index')
        ax2.set_title('Cloud Optical Thickness')

        plt.subplots_adjust(wspace=0.4)
        # plt.savefig(fname_png, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------


if __name__ == '__main__':


    # cal_mca_rad_cld_gen_hem(overwrite=True)

    plot_cld_gen_hem()

    pass
