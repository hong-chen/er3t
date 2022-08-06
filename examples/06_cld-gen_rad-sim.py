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




def rad_sim_cloud_3d(
        fdir='tmp-data/06_cld-gen_rad-sim/test',
        wavelength=650.0,
        solver='3D',
        overwrite=True,
        plot=True
        ):

    """
    Similar to test_04 but for calculating radiance fields for a randomly generated cloud field (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG
    """

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 201)

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

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
    # ------------------------------------------------------------------------------------------------------


    # define an absorption object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % fdir

    # absorption object
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    # ------------------------------------------------------------------------------------------------------


    # define an cloud object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # file name of the pickle file for cloud
    fname_cld = '%s/cld.pk' % fdir

    # cloud object
    cld0 = cld_gen(fname=fname_cld, radii=[1.0, 2.0, 4.0], weights=[0.6, 0.3, 0.1], altitude=np.arange(1.0, 4.01, 0.1), cloud_frac_tgt=0.2, w2h_ratio=2.0, min_dist=1.5, overwrite=overwrite, overlap=False)

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    # ------------------------------------------------------------------------------------------------------


    # define mca_sca object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(fname='%s/mca_atm_3d.bin' % fdir, cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, overwrite=overwrite)
    # data can be accessed at
    #     atm3d0.nml['Atm_nx']['data']
    #     atm3d0.nml['Atm_ny']['data']
    #     atm3d0.nml['Atm_dx']['data']
    #     atm3d0.nml['Atm_dy']['data']
    #     atm3d0.nml['Atm_nz3']['data']
    #     atm3d0.nml['Atm_iz3l']['data']
    #     atm3d0.nml['Atm_tmpa3d']['data']
    #     atm3d0.nml['Atm_abst3d']['data']
    #     atm3d0.nml['Atm_extp3d']['data']
    #     atm3d0.nml['Atm_omgp3d']['data']
    #     atm3d0.nml['Atm_apfp3d']['data']

    # homogeneous 1d mcarats "atmosphere"
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
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
    atm_3ds   = [atm3d0]
    # ------------------------------------------------------------------------------------------------------


    # define mcarats object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # run mcarats
    mca0 = mcarats_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            target='radiance',
            surface_albedo=0.1,
            sca=sca,
            solar_zenith_angle=60.0,
            solar_azimuth_angle=45.0,
            sensor_zenith_angle=30.0,
            sensor_azimuth_angle=60.0,
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

    # data can be accessed at
    #     mca0.Nrun
    #     mca0.Ng
    #     mca0.nml         (Nrun, Ng), e.g., mca0.nml[0][0], namelist for the first g of the first run
    #     mca0.fnames_inp  (Nrun, Ng), e.g., mca0.fnames_inp[0][0], input file name for the first g of the first run
    #     mca0.fnames_out  (Nrun, Ng), e.g., mca0.fnames_out[0][0], output file name for the first g of the first run
    # ------------------------------------------------------------------------------------------------------


    # define mcarats output object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # read mcarats output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/mca-out-rad-%s_cld3d.h5' % (fdir, solver.lower())
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    # ------------------------------------------------------------------------------------------------------


def test_rad_cloud_3d(
        overwrite=True,
        plot=True
        ):

    """
    Test
    """

    fname_cld = '%s/cld.pk' % fdir
    cld0 = cld_gen(fname=fname_cld, radii=[1.0, 2.0, 4.0], weights=[0.6, 0.3, 0.1], altitude=np.arange(1.0, 4.01, 0.1), cloud_frac_tgt=0.2, w2h_ratio=2.0, min_dist=1.5, overwrite=overwrite, overlap=False)


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if plot:
        fname_png = '06_cld-gen_rad-sim_%s.png' % solver.lower()

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
        plt.savefig(fname_png, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------


if __name__ == '__main__':


    rad_sim_cloud_3d(overwrite=True)
    pass
