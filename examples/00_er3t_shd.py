"""
by Hong Chen (hong.chen.cu@gmail.com)

This code has been tested under:
    1) Linux on 2023-06-27 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64
"""

import os
import sys
import warnings
import h5py
import time
import numpy as np
import datetime
import time
from scipy.io import readsav
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


import er3t




# global variables
#╭────────────────────────────────────────────────────────────────────────────╮#
name_tag = '00_er3t_shd'
fdir0 = er3t.common.fdir_examples
Ncpu = 4
rcParams['font.size'] = 14
#╰────────────────────────────────────────────────────────────────────────────╯#



def example_05_rad_les_cloud_3d(
        wavelength=650.0,
        solver='3D',
        overwrite=True,
        plot=True
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 4.0, 0.2), np.arange(4.0, 20.1, 2.0))

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

    # absorption object
    # abs0      = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='modis', band_name='modis_aqua_b01', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the netcdf file
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % (er3t.common.fdir_examples)

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[4, 4, 5], overwrite=overwrite)

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
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File('/Users/hchen/Work/mygit/er3t/projects/data/02_modis_rad-sim/pre-data.h5', 'r')
    sfc_dict = {
            'dx': 0.12,
            'dy': 0.1,
            'fiso': f['mod/sfc/fiso_43_0650'][...][:400, :480],
            'fvol': f['mod/sfc/fvol_43_0650'][...][:400, :480],
            'fgeo': f['mod/sfc/fgeo_43_0650'][...][:400, :480],
            }
    f.close()

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc_les.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd_les.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp_les.txt' % fdir, overwrite=False)
    atm_3ds = [atm3d0]
    sys.exit()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.0, 5.0)
    # vza = np.repeat(30.0, vaa.size)
    vaa = np.arange(45.0, 46.0, 1.0)
    vza = np.repeat(0.0, vaa.size)

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Ng=abs0.Ng,
            Niter=200,
            target='radiance',
            solar_zenith_angle=30.0,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='mpi',
            overwrite=overwrite,
            force=False,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    # fname_h5 = '%s/mca-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    # out0 = er3t.rtm.mca.mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        ax1.set_title('Radiance at %.2f nm (%s Mode)' % (wavelength, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def example_06_rad_cld_gen_hem(
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

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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

    # absorption object
    # abs0      = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='modis', band_name='modis_aqua_b01', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object (use cloud generator)
    #
    # <radii>         : a list of radius (in km) that will be assigned to clouds based on
    #                   probability defined by <weights>
    # <altitude>      : defines where the clouds will be vertically located (in km), altitude[0] is the cloud base
    # <cloud_frac_tgt>: the target cloud fraction, number of cloudy pixels divided by total number of pixels in (x, y)
    # <w2h_ratio>     :the width to height ratio of the clouds, the larger the value, the flatter the clouds
    # <min_dist>      : minimum distance between clouds (in km)
    # <overlap>       : whether clouds can overlap each other
    # <overwrite>     : whether to overwrite
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir
    cld0 = er3t.pre.cld.cld_gen_hem(
            fname=fname_cld,
            Nx=200,
            Ny=200,
            dx=0.2,
            dy=0.2,
            radii=[1.0, 2.0, 4.0],
            weights=[0.6, 0.3, 0.1],
            altitude=np.arange(2.0, 6.01, 0.2),
            cloud_frac_tgt=0.2,
            w2h_ratio=2.0,
            min_dist=0.2,
            overlap=False,
            overwrite=overwrite
            )

    # print(atm0.lay['altitude']['data'])
    # print(cld0.lay['altitude']['data'])
    # sys.exit()

    # after run, the cld0 will contain
    #
    # .fname     : absolute file path to the pickle file
    # .clouds    : a list of Python dictionaries that represent all the hemispherical clouds
    # .cloud_frac: cloud fraction of the created scene
    # .x_3d
    # .y_3d
    # .z_3d      : x, y, z in 3D
    # .space_3d  : 0 and 1 values in (x, y, z), 1 indicates cloudy
    # .x_2d
    # .y_2d      : x, y in 2D
    # .min_dist  : minimum distance between clouds
    # .w2h_ratio : width to height ration of the clouds
    #
    # .lay
    #         ['x']
    #         ['y']
    #         ['z']
    #         ['nx']
    #         ['ny']
    #         ['nz']
    #         ['dx']
    #         ['dy']
    #         ['dz']
    #         ['altitude']
    #         ['thickness']
    #         ['temperature']   (x, y, z)
    #         ['extinction']    (x, y, z)
    #         ['cot']           (x, y, z)
    #         ['cer']           (x, y, z)
    # .lev
    #         ['altitude']
    #         ['cot_2d']        (x, y)
    #         ['cth_2d']        (x, y)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate property file for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd_hem.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp_hem.txt' % fdir, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa = np.arange(0.0, 360.0, 5.0)
    vza = np.repeat(30.0, vaa.size)

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=0.1,
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle=30.0,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_res_dx=cld0.lay['dx']['data'],
            sensor_res_dy=cld0.lay['dy']['data'],
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#
    sys.exit()


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-rad-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/mca-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.mca.mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        ax1.set_title('Radiance at %.2f nm (%s Mode)' % (wavelength, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



if __name__ == '__main__':

    warnings.warn('\nUnder active development ...')

    # radiance simulation
    #╭────────────────────────────────────────────────────────────────────────────╮#
    example_05_rad_les_cloud_3d(solver='3D')
    example_05_rad_les_cloud_3d(solver='IPA')
    # example_06_rad_cld_gen_hem()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    pass
