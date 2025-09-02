"""
by Hong Chen (hong.chen.cu@gmail.com)
   Ken Hirata (ken.hirata@colorado.edu)

This code has been tested under:
    1) MacBook Air M2 on 2025-05-14 by Hong Chen
"""

import os
import sys
import warnings
import h5py
import time
import numpy as np
import datetime
import multiprocessing as mp
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


import er3t




# global variables
#╭────────────────────────────────────────────────────────────────────────────╮#
name_tag = '00_er3t_shd'
fdir0 = er3t.common.fdir_examples
Ncpu = 4
rcParams['font.size'] = 14
#╰────────────────────────────────────────────────────────────────────────────╯#



def example_01_rad_atm1d_clear_over_land(
        wavelength=550.0,
        solver='IPA',
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
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
        fname=fname_cld,
        cot=np.array([0.0]).reshape((1, 1)),
        cer=np.array([1.0]).reshape((1, 1)),
        cth=np.array([1.0]).reshape((1, 1)),
        cgt=np.array([0.5]).reshape((1, 1)),
        dz=0.1,
        extent_xy=[0.0, 1.0, 0.0, 1.0],
        atm_obj=atm0,
        overwrite=overwrite
            )

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
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([0.12472048343113448]).reshape((1, 1)),
            'fvol': np.array([0.05460690884637945]).reshape((1, 1)),
            'fgeo': np.array([0.03384929843579787]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.1, 2.0)
    # vza = np.repeat(30.0, vaa.size)

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(1.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    sza = 30.0
    saa = 0.0
    raa = er3t.util.util.calculate_raa(saa=saa, vaa=vaa, forward_scattering='positive')

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(2024, 5, 18),
        atm_1ds=atm_1ds,
        atm_3ds=atm_3ds,
        surface=sfc_2d,
        Niter=1000,
        Nmu=64,
        Nphi=128,
        sol_acc=1.0e-6,
        target='radiance',
        solar_zenith_angle=sza,
        solar_azimuth_angle=saa,
        sensor_zenith_angles=vza,
        sensor_azimuth_angles=vaa,
        sensor_altitude=705.0,
        sensor_dx=cld0.lay['dx']['data'],
        sensor_dy=cld0.lay['dy']['data'],
        fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
        solver=solver,
        Ncpu=1,
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


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(raa).reshape(vaa_2d.shape), vza_2d, data, cmap='seismic')
        # cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='RdBu', vmin=0.3, vmax=0.6)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')
        cbar.ax.set_title('Radiance')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def example_02_rad_atm1d_clear_over_ocean(
        wavelength=550.0,
        windspeed=1.0,
        pigment=0.01,
        sza=30.0,
        solver='IPA',
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
    fdir='%s/tmp-data/%s/%s/%04.1f_%04.2f_%04.1f' % (fdir0, name_tag, _metadata['Function'], windspeed, pigment, sza)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
        fname=fname_cld,
        cot=np.array([0.0]).reshape((1, 1)),
        cer=np.array([1.0]).reshape((1, 1)),
        cth=np.array([1.5]).reshape((1, 1)),
        cgt=np.array([1.0]).reshape((1, 1)),
        dz=0.1,
        extent_xy=[0.0, 1.0, 0.0, 1.0],
        atm_obj=atm0,
        overwrite=overwrite
            )

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
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([windspeed]).reshape((1, 1)),
            'pigment': np.array([pigment]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.1, 2.0)
    # vza = np.repeat(30.0, vaa.size)

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(1.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    saa = 0.0
    raa = er3t.util.util.calculate_raa(saa=saa, vaa=vaa, forward_scattering='positive')

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(2024, 5, 18),
        atm_1ds=atm_1ds,
        atm_3ds=atm_3ds,
        surface=sfc_2d,
        Niter=1000,
        Nmu=64,
        Nphi=128,
        sol_acc=1.0e-6,
        target='radiance',
        solar_zenith_angle=sza,
        solar_azimuth_angle=saa,
        sensor_zenith_angles=vza,
        sensor_azimuth_angles=vaa,
        sensor_altitude=705.0,
        sensor_dx=cld0.lay['dx']['data'],
        sensor_dy=cld0.lay['dy']['data'],
        fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
        solver=solver,
        Ncpu=1,
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


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(raa).reshape(vaa_2d.shape), vza_2d, data, cmap='seismic')
        # cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='RdBu', vmin=0.3, vmax=0.6)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')
        cbar.ax.set_title('Radiance')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def example_03_rad_atm1d_clear_over_snow(
        wavelength=555.0,
        solver='IPA',
        overwrite=True,
        plot=True,
        mode='afgl'
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
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    if mode == 'afgl':
        atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=None, fname_atmmod="er3t/data/atmmod/afglss.dat", overwrite=overwrite)

    elif mode == 'arcsix':
        atm0      = er3t.pre.atm.ARCSIXAtmModel(levels=levels, config_file=os.path.join(er3t.common.fdir_er3t, 'er3t/pre/atm/', 'arcsix_atm_profile_config.yaml'), verbose=1, fname_out='data/test_data/arcsix_atm_profile_output.h5')

    else:
        raise ValueError('Error [example_03_rad_atm1d_clear_over_snow] `mode` should be afgl or arcsix')


#     atm0      = er3t.pre.atm.ARCSIXAtmModel(levels=levels, config_file=os.path.join(er3t.common.fdir_er3t, 'er3t/pre/atm/', 'arcsix_atm_profile_config.yaml'), verbose=1, fname_out='data/test_data/arcsix_atm_profile_output.h5')

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
        fname=fname_cld,
        cot=np.array([0.0]).reshape((1, 1)),
        cer=np.array([1.0]).reshape((1, 1)),
        cth=np.array([1.5]).reshape((1, 1)),
        cgt=np.array([1.0]).reshape((1, 1)),
        dz=0.1,
        extent_xy=[0.0, 1.0, 0.0, 1.0],
        atm_obj=atm0,
        overwrite=overwrite
            )

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
    # sfc_dict = {
    #         'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
    #         'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
    #         'fiso': np.array([0.986]).reshape((1, 1)),
    #         'fgeo': np.array([0.033]).reshape((1, 1)),
    #         'fvol': np.array([0.000]).reshape((1, 1)),
    #         'fj'  : np.array([0.447]).reshape((1, 1)),
    #         'alpha': np.array([0.3]).reshape((1, 1)),
    #         }

    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'alb': np.array([0.9]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.1, 2.0)
    # vza = np.repeat(30.0, vaa.size)

    vaa_1d = np.arange(0.0, 359.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    sza = 63.0
    saa = 0.0
    raa = er3t.util.util.calculate_raa(saa=saa, vaa=vaa, forward_scattering='positive')

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(2024, 6, 5),
        atm_1ds=atm_1ds,
        atm_3ds=atm_3ds,
        surface=sfc_2d,
        Niter=1000,
        Nmu=32,
        Nphi=64,
        sol_acc=1.0e-6,
        target='radiance',
        solar_zenith_angle=sza,
        solar_azimuth_angle=saa,
        sensor_zenith_angles=vza,
        sensor_azimuth_angles=vaa,
        sensor_altitude=705.0,
        sensor_dx=cld0.lay['dx']['data'],
        sensor_dy=cld0.lay['dy']['data'],
        fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
        solver=solver,
        Ncpu=1,
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


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s_wvl%s_mode%s.h5' % (fdir, solver.lower(), _metadata['Function'], int(wavelength), mode)
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s_wvl%s_mode%s.png' % (name_tag, _metadata['Function'], solver.lower(), int(wavelength), mode)

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(raa).reshape(vaa_2d.shape), vza_2d, data, cmap='seismic', vmin=0.35, vmax=0.6)
        # cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='RdBu', vmin=0.3, vmax=0.6)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')
        cbar.ax.set_title('Radiance')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        fig.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        print('Saved figure in: ', fname_png)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def example_04_rad_atm1d_cloud_over_ocean(
        wavelength=550.0,
        cot=10.0,
        cer=12.0,
        sza=30.0,
        windspeed=1.0,
        pigment=0.01,
        solver='IPA',
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
    fdir='%s/tmp-data/%s/%s/%04.1f_%04.1f_%04.1f' % (fdir0, name_tag, _metadata['Function'], cot, cer, sza)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
        fname=fname_cld,
        cot=np.array([cot]).reshape((1, 1)),
        cer=np.array([cer]).reshape((1, 1)),
        cth=np.array([1.5]).reshape((1, 1)),
        cgt=np.array([1.0]).reshape((1, 1)),
        dz=0.1,
        extent_xy=[0.0, 1.0, 0.0, 1.0],
        atm_obj=atm0,
        overwrite=overwrite
            )

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
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([windspeed]).reshape((1, 1)),
            'pigment': np.array([pigment]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.1, 2.0)
    # vza = np.repeat(30.0, vaa.size)

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(1.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(2024, 5, 18),
        atm_1ds=atm_1ds,
        atm_3ds=atm_3ds,
        surface=sfc_2d,
        Niter=1000,
        Nmu=64,
        Nphi=128,
        sol_acc=1.0e-6,
        target='radiance',
        solar_zenith_angle=sza,
        solar_azimuth_angle=0.0,
        sensor_zenith_angles=vza,
        sensor_azimuth_angles=vaa,
        sensor_altitude=705.0,
        sensor_dx=cld0.lay['dx']['data'],
        sensor_dy=cld0.lay['dy']['data'],
        fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
        solver=solver,
        Ncpu=1,
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


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
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
    # fdir='%s/tmp-data/%s/%s_fine-res' % (fdir0, name_tag, _metadata['Function'])
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the netcdf file
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % (er3t.common.fdir_examples)

    # file name of the pickle file for cloud
    fname_les = '%s/cld.pk' % fdir

    # cloud object
    cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[4, 4, 5], overwrite=overwrite)
    # cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[1, 1, 5], overwrite=overwrite)

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

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    interval = 2.0
    levels_cld = cld0.lev['altitude']['data']
    levels = np.append(levels_cld, np.arange(((int(levels_cld[-1])//interval)+1)*interval, 20.1, interval))

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='modis', band_name='modis_aqua_b01', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File('%s/data/pre-data.h5' % (er3t.common.fdir_examples), 'r')
    fiso = f['mod/sfc/fiso_43_0650'][...][:400, :480]
    fvol = f['mod/sfc/fvol_43_0650'][...][:400, :480]
    fgeo = f['mod/sfc/fgeo_43_0650'][...][:400, :480]

    lon, lat = np.meshgrid(np.linspace(0.0, 48.0, 400), np.linspace(0.0, 48.0, 480), indexing='ij')
    x, y, fiso = er3t.util.grid_by_lonlat(lon, lat, fiso, lon_1d=cld0.lay['x']['data'], lat_1d=cld0.lay['y']['data'], method='cubic')
    x, y, fvol = er3t.util.grid_by_lonlat(lon, lat, fvol, lon_1d=cld0.lay['x']['data'], lat_1d=cld0.lay['y']['data'], method='cubic')
    x, y, fgeo = er3t.util.grid_by_lonlat(lon, lat, fgeo, lon_1d=cld0.lay['x']['data'], lat_1d=cld0.lay['y']['data'], method='cubic')

    sfc_dict = {
            'dx': cld0.lay['dx']['data'],
            'dy': cld0.lay['dy']['data'],
            'fiso': fiso,
            'fvol': fvol,
            'fgeo': fgeo,
            }
    f.close()

    # sfc_dict = {
    #         'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
    #         'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
    #         'windspeed': np.array([8.0]).reshape((1, 1)),
    #         'pigment': np.array([0.01]).reshape((1, 1)),
    #         }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd_%4.4d.txt' % (fdir, wavelength), overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp_les.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.0, 5.0)
    # vza = np.repeat(30.0, vaa.size)
    vaa = np.arange(45.0, 46.0, 1.0)
    vza = np.repeat(0.0, vaa.size)

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(2024, 5, 18),
        atm_1ds=atm_1ds,
        atm_3ds=atm_3ds,
        surface=sfc_2d,
        Ng=abs0.Ng,
        Niter=200,
        split_acc=0.001,
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
        force=True,
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
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    # fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    # out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
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
    levels = np.append(np.arange(0.0, 6.0, 0.4), np.arange(6.0, 20.1, 2.0))

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

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
        dx=0.4,
        dy=0.4,
        radii=[1.0, 2.0, 4.0],
        weights=[0.6, 0.3, 0.1],
        altitude=np.arange(2.0, 6.01, 0.4),
        cloud_frac_tgt=0.2,
        w2h_ratio=2.0,
        min_dist=0.2,
        overlap=False,
        overwrite=overwrite
            )

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

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([8.0]).reshape((1, 1)),
            'pigment': np.array([0.01]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate property file for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc_hem.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd_hem.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp_hem.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
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
        surface=sfc_2d,
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
        force=True,
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
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        for ivaa in range(vaa.size):
            fname_png = '%s-%s_%s_%02d.png' % (name_tag, _metadata['Function'], solver.lower(), ivaa)

            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            cs = ax1.imshow(np.transpose(out0.data['rad']['data'][:, :, ivaa]), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
            ax1.set_xlabel('X Index')
            ax1.set_ylabel('Y Index')
            ax1.set_title('Radiance at %.2f nm, vaa=%.f deg (%s Mode)' % (wavelength, vaa[ivaa], solver))
            plt.savefig(fname_png, bbox_inches='tight')
            plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def example_07_at3d_rad_cloud_merra(
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

    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the netcdf file
    fname_nc  = '/Users/hchen/Downloads/MERRA2_400.inst3_3d_asm_Np.20240518.nc4'

    # file name of the pickle file for cloud
    fname_cld = '%s/cld.pk' % fdir

    # cloud object
    cld0      = er3t.pre.cld.cld_merra(fname_nc=fname_nc, fname=fname_cld, coarsen=[1, 1, 1], overwrite=overwrite)

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

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    # print(cld0.lay['altitude']['data'])
    # print(cld0.lev['altitude']['data']+cld0.lev['thickness']['data']/2.0)
    # print(cld0.lay['altitude']['data'])
    # print(cld0.lev['altitude']['data'])
    # print(cld0.lay['thickness']['data'])
    # print(cld0.lev['thickness']['data']+cld0.lev['thickness']['data']/2.0)

    interval = 4.0
    levels_cld = cld0.lev['altitude']['data']
    levels = np.append(levels_cld, np.arange(((int(levels_cld[-1])//interval)+1)*interval, 20.1, interval))

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
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([8.0]).reshape((1, 1)),
            'pigment': np.array([0.01]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd_%4.4d.txt' % (fdir, wavelength), overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=False)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.0, 5.0)
    # vza = np.repeat(30.0, vaa.size)
    vaa = np.arange(45.0, 46.0, 1.0)
    vza = np.repeat(0.0, vaa.size)

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(2024, 5, 8),
        atm_1ds=atm_1ds,
        atm_3ds=atm_3ds,
        surface=sfc_2d,
        Ng=abs0.Ng,
        Niter=10,
        split_acc=0.0,
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
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    # fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    # out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
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


def _process_single_wavelength(args):
    """
    Helper function to process a single wavelength for multiprocessing.

    Parameters:
    -----------
    args : tuple
        Tuple containing (i_wvl, wavelength, and all other parameters)

    Returns:
    --------
    tuple : (i_wvl, radiance_data)
        Index and radiance results for this wavelength
    """
    (i_wvl, wavelength, solar_zenith_angles, solar_azimuth_angles,
     viewing_zenith_angles, viewing_azimuth_angles, atm_mode, solver,
     surface_type, output_dir, overwrite, verbose) = args

    # Import er3t inside the worker process to avoid pickling issues
    import er3t
    import numpy as np
    import datetime
    import os

    n_sza = len(solar_zenith_angles)
    n_saa = len(solar_azimuth_angles)
    n_vza = len(viewing_zenith_angles)
    n_vaa = len(viewing_azimuth_angles)

    # Initialize results for this wavelength
    wavelength_results = np.zeros((n_sza, n_saa, n_vza, n_vaa))

    if verbose:
        print(f"Worker {i_wvl}: Processing wavelength {wavelength:.1f} nm")

    # Create subdirectory for this wavelength
    wvl_dir = f'{output_dir}/wvl_{wavelength:.1f}'
    if not os.path.exists(wvl_dir):
        os.makedirs(wvl_dir)

    # Set up atmosphere and surface objects
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # Create atmosphere object
    if atm_mode == 'afgl':
        atm0 = er3t.pre.atm.atm_atmmod(levels=levels, fname=None,
                                       fname_atmmod="er3t/data/atmmod/afglss.dat",
                                       overwrite=overwrite)
    elif atm_mode == 'arcsix':
        atm0 = er3t.pre.atm.ARCSIXAtmModel(
            levels=levels,
            config_file=os.path.join(er3t.common.fdir_er3t, 'er3t/pre/atm/', 'arcsix_atm_profile_config.yaml'),
            verbose=0,  # Reduce verbosity for multiple calculations
            fname_out=f'{wvl_dir}/arcsix_atm_profile.h5'
        )
    else:
        raise ValueError(f"atm_mode should be 'afgl' or 'arcsix', got '{atm_mode}'")

    # Create absorption object
    fname_abs = f'{wvl_dir}/abs_{wavelength:.1f}.pk'
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs,
                                target='fine', atm_obj=atm0, overwrite=overwrite)

    # Create cloud object (clear sky - zero optical thickness)
    fname_cld = f'{wvl_dir}/cld.pk'
    cld0 = er3t.pre.cld.cld_gen_cop(
        fname=fname_cld,
        cot=np.array([0.0]).reshape((1, 1)),
        cer=np.array([1.0]).reshape((1, 1)),
        cth=np.array([1.5]).reshape((1, 1)),
        cgt=np.array([1.0]).reshape((1, 1)),
        dz=0.1,
        extent_xy=[0.0, 1.0, 0.0, 1.0],
        atm_obj=atm0,
        overwrite=overwrite
    )

    # Create surface object based on surface type
    if surface_type == 'snow':
        sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([0.986]).reshape((1, 1)),
            'fgeo': np.array([0.033]).reshape((1, 1)),
            'fvol': np.array([0.000]).reshape((1, 1)),
            'fj'  : np.array([0.447]).reshape((1, 1)),
            'alpha': np.array([0.3]).reshape((1, 1)),
        }
    elif surface_type == 'land':
        sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([0.129]).reshape((1, 1)),
            'fgeo': np.array([0.074]).reshape((1, 1)),
            'fvol': np.array([0.055]).reshape((1, 1)),
            'fj'  : np.array([0.417]).reshape((1, 1)),
            'alpha': np.array([1.5]).reshape((1, 1)),
        }
    elif surface_type == 'ocean':
        sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([0.02]).reshape((1, 1)),
            'fgeo': np.array([0.0]).reshape((1, 1)),
            'fvol': np.array([0.0]).reshape((1, 1)),
            'fj'  : np.array([0.0]).reshape((1, 1)),
            'alpha': np.array([0.0]).reshape((1, 1)),
        }

    elif surface_type == 'lambertian':
        sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'alb': np.array([0.9]).reshape((1, 1)),
            }

    else:
        raise ValueError(f"surface_type should be 'snow', 'land', or 'ocean', got '{surface_type}'")

    fname_sfc = f'{wvl_dir}/sfc.pk'
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)

    # Generate SHDOM input files
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0,
                                      fname=f'{wvl_dir}/shdom-sfc.txt', overwrite=overwrite)
    atm1d0 = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0,
                                      fname=f'{wvl_dir}/shdom-ckd.txt', overwrite=overwrite)
    atm3d0 = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0,
                                      fname=f'{wvl_dir}/shdom-prp.txt',
                                      fname_atm_1d=atm1d0.fname, overwrite=overwrite)

    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]

    # Loop over solar geometries
    for i_sza, sza in enumerate(solar_zenith_angles):
        for i_saa, saa in enumerate(solar_azimuth_angles):

            # Create viewing geometry arrays
            vaa_2d, vza_2d = np.meshgrid(viewing_azimuth_angles, viewing_zenith_angles, indexing='ij')
            vaa = vaa_2d.ravel()
            vza = vza_2d.ravel()

            # Calculate relative azimuth angles
            raa = er3t.util.util.calculate_raa(saa=saa, vaa=vaa, forward_scattering='positive')

            # Run SHDOM
            shd_dir = f'{wvl_dir}/shdom_sza{sza:.1f}_saa{saa:.1f}'
            shd0 = er3t.rtm.shd.shdom_ng(
                date=datetime.datetime(2024, 6, 5),
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                surface=sfc_2d,
                Niter=1000,
                Nmu=32,
                Nphi=64,
                sol_acc=1.0e-6,
                target='radiance',
                solar_zenith_angle=sza,
                solar_azimuth_angle=saa,
                sensor_zenith_angles=vza,
                sensor_azimuth_angles=vaa,
                sensor_altitude=705.0,
                sensor_dx=cld0.lay['dx']['data'],
                sensor_dy=cld0.lay['dy']['data'],
                fdir=shd_dir,
                solver=solver,
                Ncpu=1,  # Use single CPU per wavelength worker
                mp_mode='mpi',
                overwrite=overwrite,
                force=True,
            )

            # Read SHDOM output
            fname_h5 = f'{shd_dir}/shd_output.h5'
            out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0,
                                           mode='mean', squeeze=True, verbose=False, overwrite=overwrite)

            # Store results
            output_dat = out0.data['rad']['data']
            radiance_data = output_dat.reshape(vaa_2d.shape)
            wavelength_results[i_sza, i_saa, :, :] = radiance_data

    if verbose:
        print(f"Worker {i_wvl}: Completed wavelength {wavelength:.1f} nm")

    return (i_wvl, wavelength_results)


def calculate_spectral_radiance(
        wavelengths=[550.0, 650.0, 940.0],
        solar_zenith_angles=np.arange(0, 91, 30),
        viewing_zenith_angles=np.arange(0, 91, 30),
        solar_azimuth_angles=np.arange(0, 360, 60),
        viewing_azimuth_angles=np.arange(0, 360, 60),
        atm_mode='afgl',
        solver='IPA',
        surface_type='snow',
        output_dir=None,
        overwrite=True,
        verbose=True,
        n_cpus=None
        ):
    """
    Calculate spectral radiance for given viewing geometries and wavelengths
    based on example_03_rad_atm1d_clear_over_snow.
    Uses multiprocessing to parallelize calculations across wavelengths.

    Parameters:
    -----------
    wavelengths : list or array-like, default [550.0, 650.0, 750.0]
        Array of wavelengths in nm
    solar_zenith_angles : array-like, default np.arange(0, 91, 10)
        Array of solar zenith angles in degrees (0-90)
    viewing_zenith_angles : array-like, default np.arange(0, 91, 10)
        Array of viewing zenith angles in degrees (0-90)
    solar_azimuth_angles : array-like, default np.arange(0, 360, 30)
        Array of solar azimuth angles in degrees (0-359)
    viewing_azimuth_angles : array-like, default np.arange(0, 360, 30)
        Array of viewing azimuth angles in degrees (0-359)
    atm_mode : str, default 'afgl'
        Atmospheric model mode ('afgl' or 'arcsix')
    solver : str, default 'IPA'
        Radiative transfer solver ('IPA' or 'MCA')
    surface_type : str, default 'snow'
        Surface type ('snow', 'land', 'ocean')
    output_dir : str, optional
        Output directory. If None, uses default tmp-data structure
    overwrite : bool, default True
        Whether to overwrite existing files
    verbose : bool, default True
        Whether to print progress information
    n_cpus : int, optional
        Number of CPUs to use for parallel processing. If None, uses min(cpu_count(), n_wavelengths)

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'radiance': 5D array (wavelengths, sza, saa, vza, vaa)
        - 'wavelengths': wavelength array
        - 'solar_zenith_angles': solar zenith angle array
        - 'solar_azimuth_angles': solar azimuth angle array
        - 'viewing_zenith_angles': viewing zenith angle array
        - 'viewing_azimuth_angles': viewing azimuth angle array
        - 'metadata': calculation metadata
    """

    # Convert inputs to numpy arrays
    wavelengths = np.array(wavelengths)
    solar_zenith_angles = np.array(solar_zenith_angles)
    viewing_zenith_angles = np.array(viewing_zenith_angles)
    solar_azimuth_angles = np.array(solar_azimuth_angles)
    viewing_azimuth_angles = np.array(viewing_azimuth_angles)

    # Create output directory
    if output_dir is None:
        output_dir = f'{fdir0}/tmp-data/{name_tag}/spectral_radiance_calculation'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize results array
    n_wvl = len(wavelengths)
    n_sza = len(solar_zenith_angles)
    n_saa = len(solar_azimuth_angles)
    n_vza = len(viewing_zenith_angles)
    n_vaa = len(viewing_azimuth_angles)

    radiance_results = np.zeros((n_wvl, n_sza, n_saa, n_vza, n_vaa))

    if verbose:
        total_calculations = n_wvl * n_sza * n_saa
        print("Starting spectral radiance calculations...")
        print(f"Wavelengths: {n_wvl}")
        print(f"Solar geometries: {n_sza} x {n_saa} = {n_sza * n_saa}")
        print(f"Viewing geometries: {n_vza} x {n_vaa} = {n_vza * n_vaa}")
        print(f"Total calculations: {total_calculations}")

    # Determine number of CPUs to use
    if n_cpus is None:
        n_cpus = min(mp.cpu_count(), n_wvl)
    else:
        n_cpus = min(n_cpus, mp.cpu_count(), n_wvl)

    if verbose:
        print(f"Using {n_cpus} CPUs for {n_wvl} wavelengths")
        print(f"Available CPUs: {mp.cpu_count()}")

    # Prepare arguments for multiprocessing
    args_list = []
    for i_wvl, wavelength in enumerate(wavelengths):
        args_list.append((
            i_wvl, wavelength, solar_zenith_angles, solar_azimuth_angles,
            viewing_zenith_angles, viewing_azimuth_angles, atm_mode, solver,
            surface_type, output_dir, overwrite, verbose
        ))

    # Process wavelengths in parallel
    if verbose:
        print("\nStarting parallel wavelength processing...")

    # Use multiprocessing Pool to process wavelengths in parallel
    # Note: This should work properly when called from __main__ or in notebooks
    try:
        with mp.Pool(processes=n_cpus) as pool:
            results_list = pool.map(_process_single_wavelength, args_list)
    except Exception as e:
        if verbose:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to sequential processing...")
        # Fallback to sequential processing
        results_list = []
        for args in args_list:
            results_list.append(_process_single_wavelength(args))

    # Collect results
    if verbose:
        print("Collecting results from parallel processes...")

    for i_wvl, wavelength_results in results_list:
        radiance_results[i_wvl, :, :, :, :] = wavelength_results

    # Prepare results dictionary
    results = {
        'radiance': radiance_results,
        'wavelengths': wavelengths,
        'solar_zenith_angles': solar_zenith_angles,
        'solar_azimuth_angles': solar_azimuth_angles,
        'viewing_zenith_angles': viewing_zenith_angles,
        'viewing_azimuth_angles': viewing_azimuth_angles,
        'metadata': {
            'atm_mode': atm_mode,
            'solver': solver,
            'surface_type': surface_type,
            'output_dir': output_dir,
            'calculation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_calculations': total_calculations
        }
    }

    if verbose:
        print("\nSpectral radiance calculation completed!")
        print(f"Results shape: {radiance_results.shape}")
        print(f"Output saved to: {output_dir}")

    return results


# def plot_spectral_radiance(results, output_filename=None, figsize=(12, 8), dpi=300):
#     """
#     Create a spectral plot of radiance results and save to file.

#     Parameters:
#     -----------
#     results : dict
#         Results dictionary from calculate_spectral_radiance function
#     output_filename : str, optional
#         Output filename for the plot. If None, generates automatic filename
#     figsize : tuple, default (12, 8)
#         Figure size in inches (width, height)
#     dpi : int, default 300
#         Resolution for saved figure

#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         The created figure object
#     """

#     # Extract data from results
#     radiance = results['radiance']
#     wavelengths = results['wavelengths']
#     solar_zenith_angles = results['solar_zenith_angles']
#     solar_azimuth_angles = results['solar_azimuth_angles']
#     viewing_zenith_angles = results['viewing_zenith_angles']
#     viewing_azimuth_angles = results['viewing_azimuth_angles']
#     metadata = results['metadata']

#     # Create figure with subplots
#     fig, axes = plt.subplots(2, 2, figsize=figsize)
#     fig.suptitle(f'Spectral Radiance Analysis\n'
#                 f'Surface: {metadata["surface_type"].title()}, '
#                 f'Atmosphere: {metadata["atm_mode"].upper()}, '
#                 f'Solver: {metadata["solver"]}', fontsize=14, fontweight='bold')

#     # Plot 1: Spectral radiance vs wavelength (averaged over all geometries)
#     ax1 = axes[0, 0]
#     mean_radiance = np.mean(radiance, axis=(1, 2, 3, 4))
#     std_radiance = np.std(radiance, axis=(1, 2, 3, 4))

#     ax1.errorbar(wavelengths, mean_radiance, yerr=std_radiance,
#                 marker='o', linestyle='-', linewidth=2, markersize=6,
#                 capsize=5, capthick=2, label='Mean ± Std')
#     ax1.set_xlabel('Wavelength (nm)')
#     ax1.set_ylabel('Radiance')
#     ax1.set_title('Spectral Radiance\n(averaged over all geometries)')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

#     # Plot 2: Radiance vs solar zenith angle (for first wavelength)
#     ax2 = axes[0, 1]
#     if len(wavelengths) > 0 and len(solar_zenith_angles) > 1:
#         wvl_idx = 0  # First wavelength
#         # Average over azimuth angles and viewing angles
#         radiance_vs_sza = np.mean(radiance[wvl_idx, :, :, :, :], axis=(1, 2, 3))

#         ax2.plot(solar_zenith_angles, radiance_vs_sza, 'o-', linewidth=2, markersize=6)
#         ax2.set_xlabel('Solar Zenith Angle (°)')
#         ax2.set_ylabel('Radiance')
#         ax2.set_title(f'Radiance vs Solar Zenith Angle\n(λ = {wavelengths[wvl_idx]:.1f} nm)')
#         ax2.grid(True, alpha=0.3)
#     else:
#         ax2.text(0.5, 0.5, 'Insufficient SZA data\nfor plotting',
#                 ha='center', va='center', transform=ax2.transAxes)
#         ax2.set_title('Solar Zenith Angle Dependence')

#     # Plot 3: Radiance vs viewing zenith angle (for first wavelength)
#     ax3 = axes[1, 0]
#     if len(wavelengths) > 0 and len(viewing_zenith_angles) > 1:
#         wvl_idx = 0  # First wavelength
#         # Average over solar angles and azimuth angles
#         radiance_vs_vza = np.mean(radiance[wvl_idx, :, :, :, :], axis=(0, 1, 3))

#         ax3.plot(viewing_zenith_angles, radiance_vs_vza, 's-', linewidth=2, markersize=6, color='orange')
#         ax3.set_xlabel('Viewing Zenith Angle (°)')
#         ax3.set_ylabel('Radiance')
#         ax3.set_title(f'Radiance vs Viewing Zenith Angle\n(λ = {wavelengths[wvl_idx]:.1f} nm)')
#         ax3.grid(True, alpha=0.3)
#     else:
#         ax3.text(0.5, 0.5, 'Insufficient VZA data\nfor plotting',
#                 ha='center', va='center', transform=ax3.transAxes)
#         ax3.set_title('Viewing Zenith Angle Dependence')

#     # Plot 4: Summary statistics table
#     ax4 = axes[1, 1]
#     ax4.axis('off')

#     # Create summary statistics
#     summary_data = []
#     for i, wvl in enumerate(wavelengths):
#         wvl_radiance = radiance[i, :, :, :, :]
#         summary_data.append([
#             f'{wvl:.1f}',
#             f'{np.mean(wvl_radiance):.3f}',
#             f'{np.std(wvl_radiance):.3f}',
#             f'{np.min(wvl_radiance):.3f}',
#             f'{np.max(wvl_radiance):.3f}'
#         ])

#     # Create table
#     table_headers = ['Wavelength\n(nm)', 'Mean\nRadiance', 'Std\nRadiance', 'Min\nRadiance', 'Max\nRadiance']
#     table = ax4.table(cellText=summary_data, colLabels=table_headers,
#                      cellLoc='center', loc='center', bbox=[0, 0.3, 1, 0.6])
#     table.auto_set_font_size(False)
#     table.set_fontsize(9)
#     table.scale(1, 1.5)

#     # Style the table
#     for i in range(len(table_headers)):
#         table[(0, i)].set_facecolor('#40466e')
#         table[(0, i)].set_text_props(weight='bold', color='white')

#     ax4.set_title('Radiance Statistics\n(W m⁻² sr⁻¹ nm⁻¹)', pad=20)

#     # Add metadata text
#     metadata_text = (f"Calculation Date: {metadata['calculation_date']}\n"
#                     f"Total Calculations: {metadata['total_calculations']}\n"
#                     f"Geometries: {len(solar_zenith_angles)} SZA × {len(solar_azimuth_angles)} SAA × "
#                     f"{len(viewing_zenith_angles)} VZA × {len(viewing_azimuth_angles)} VAA")
#     ax4.text(0.5, 0.1, metadata_text, ha='center', va='center',
#             transform=ax4.transAxes, fontsize=8,
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

#     # Adjust layout
#     plt.tight_layout()

#     # Generate filename if not provided
#     if output_filename is None:
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         wvl_range = f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm" if len(wavelengths) > 1 else f"{wavelengths[0]:.0f}nm"
#         output_filename = f'spectral_radiance_{metadata["surface_type"]}_{metadata["atm_mode"]}_{wvl_range}_{timestamp}.png'

#     # Ensure output directory exists
#     output_dir = os.path.dirname(output_filename) if os.path.dirname(output_filename) else metadata.get('output_dir', '.')
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Save figure
#     full_path = os.path.join(output_dir, os.path.basename(output_filename))
#     fig.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor='white')

#     print(f"\nSpectral radiance plot saved to: {full_path}")

#     return fig

def plot_spectral_radiance(results, output_filename=None, figsize=(20, 8), dpi=300):
    """
    Create a single spectral plot of radiance vs wavelength and save to file.

    Parameters:
    -----------
    results : dict
        Results dictionary from calculate_spectral_radiance function
    output_filename : str, optional
        Output filename for the plot. If None, generates automatic filename
    figsize : tuple, default (10, 8)
        Figure size in inches (width, height)
    dpi : int, default 300
        Resolution for saved figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    # Extract data from results
    radiance = results['radiance']
    wavelengths = results['wavelengths']
    solar_zenith_angles = results['solar_zenith_angles']
    solar_azimuth_angles = results['solar_azimuth_angles']
    viewing_zenith_angles = results['viewing_zenith_angles']
    viewing_azimuth_angles = results['viewing_azimuth_angles']
    metadata = results['metadata']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Calculate mean radiance across all geometries for each wavelength
    mean_radiance = np.mean(radiance, axis=(1, 2, 3, 4))
    std_radiance = np.std(radiance, axis=(1, 2, 3, 4))

    # Plot spectral radiance with error bars
    ax.errorbar(wavelengths, mean_radiance, yerr=std_radiance,
                linestyle='-', linewidth=2, markersize=0,
                capsize=0, capthick=0, color='darkblue',
                ecolor='lightblue', label='Mean ± Std')

    # Set labels and title
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Radiance', fontsize=12, fontweight='bold')
    ax.set_title('Spectral Radiance', fontsize=14, fontweight='bold', pad=20)

    # set ylim for easy comparison between multiple runs
    ax.set_ylim([-0.02, 0.8])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Define major atmospheric absorption bands
    # absorb_bands = [[649.0, 662.0], [676.0, 697.0], [707.0, 729.0], [743.0, 779.0],
    # [796.0, 830.0], [911.0, 976.0], [1108.0, 1193.0], [1262.0, 1298.0], [1324.0, 1470.0],
    # [1800.0, 1960.0]]

    absorption_bands = {
        'O2-A Band': {'range': (755, 770), 'color': 'red', 'alpha': 0.2},
        'H2O (720nm)': {'range': (707, 729), 'color': 'turquoise', 'alpha': 0.2},
        'O2-B Band': {'range': (676, 697), 'color': 'red', 'alpha': 0.2},
        'Chappuis (O3)': {'range': (400, 650), 'color': 'orange', 'alpha': 0.1},
        'H2O (820nm)': {'range': (796, 830), 'color': 'turquoise', 'alpha': 0.2},
        'H2O (940nm)': {'range': (911, 976), 'color': 'turquoise', 'alpha': 0.2},
        'H2O (1130nm)': {'range': (1108, 1193), 'color': 'turquoise', 'alpha': 0.2},
        'O2 (1270nm)': {'range': (1262, 1298), 'color': 'red', 'alpha': 0.2},
        'H2O (1380nm)': {'range': (1324, 1470), 'color': 'turquoise', 'alpha': 0.2},
        'H2O (1870nm)': {'range': (1800, 1960), 'color': 'turquoise', 'alpha': 0.2},
        'CO2 (2060nm)': {'range': (2040, 2080), 'color': 'green', 'alpha': 0.2},
    }

    # Get y-axis limits for shading
    y_min, y_max = ax.get_ylim()

    # Plot absorption bands that overlap with our wavelength range
    wvl_min, wvl_max = wavelengths.min(), wavelengths.max()
    legend_handles = []
    plotted_species = set()

    for band_name, band_info in absorption_bands.items():
        band_min, band_max = band_info['range']

        # Check if band overlaps with our wavelength range
        if band_max >= wvl_min and band_min <= wvl_max:
            # Clip band to our wavelength range
            plot_min = max(band_min, wvl_min)
            plot_max = min(band_max, wvl_max)

            # Add shaded region
            ax.axvspan(plot_min, plot_max,
                      color=band_info['color'],
                      alpha=band_info['alpha'],
                      zorder=0)

            # Determine species for legend (extract from band name)
            if 'O2' in band_name or 'O₂' in band_name:
                species = 'O₂ Absorption'
                color = 'red'
            elif 'H2O' in band_name or 'H₂O' in band_name:
                species = 'H₂O Absorption'
                color = 'blue'
            elif 'Chappuis' in band_name or 'O3' in band_name:
                species = 'O₃ Chappuis Band'
                color = 'orange'
            elif 'CO2' in band_name:
                species = 'CO₂ Absorption'
                color = 'green'
            else:
                continue

            # Add to legend if not already added
            # if species not in plotted_species:
            #     legend_handles.append(plt.Rectangle((0, 0), 1, 1,
            #                                       facecolor=color,
            #                                       alpha=0.3,
            #                                       label=species))
            #     plotted_species.add(species)

    # Add absorption band annotations for major bands in range
    # annotation_offset = 0
    # for band_name, band_info in absorption_bands.items():
    #     band_min, band_max = band_info['range']
    #     band_center = (band_min + band_max) / 2

    #     # Only annotate if band center is in our range and it's a major band
    #     major_bands = ['O2-A', 'O2-B', 'H2O (940nm)', 'Chappuis']
    #     in_range = wvl_min <= band_center <= wvl_max
    #     is_major = any(keyword in band_name for keyword in major_bands)
    #     if in_range and is_major:

    #         # Position annotation
    #         y_pos = y_max - 0.05 * (y_max - y_min) - annotation_offset * 0.04 * (y_max - y_min)

    #         ax.annotate(band_name,
    #                    xy=(band_center, y_pos),
    #                    xytext=(0, -10),
    #                    textcoords='offset points',
    #                    ha='center', va='top',
    #                    fontsize=8,
    #                    bbox=dict(boxstyle='round,pad=0.2',
    #                            facecolor=band_info['color'],
    #                            alpha=0.6),
    #                    arrowprops=dict(arrowstyle='->',
    #                                  connectionstyle='arc3,rad=0'))
    #         annotation_offset += 1

    # Format geometry information
    def format_angle_range(angles):
        """Format angle array for display"""
        if len(angles) == 1:
            return f"{angles[0]:.1f}°"
        else:
            return f"{angles.min():.1f}° - {angles.max():.1f}°"

    # Create information text box
    info_text = (
        f"Surface: {metadata['surface_type'].title()}\n"
        f"Atmosphere: {metadata['atm_mode'].upper()}\n"
        f"Solver: {metadata['solver']}\n"
        f"SZA: {format_angle_range(solar_zenith_angles)}\n"
        f"SAA: {format_angle_range(solar_azimuth_angles)}\n"
        f"VZA: {format_angle_range(viewing_zenith_angles)}\n"
        f"VAA: {format_angle_range(viewing_azimuth_angles)}\n"
    )

    # Add text box to plot
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="none"),
            fontsize=10)

    # Add legend for absorption bands if any were plotted
    if legend_handles:
        # Add the main radiance line to legend
        legend_handles.insert(0, plt.Line2D([0], [0], color='darkblue', linewidth=2,
                                          marker='o', markersize=4, label='Spectral Radiance'))

        # Create legend
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
                 framealpha=0.9, fancybox=True)

    # Adjust layout
    plt.tight_layout()

    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        wvl_range = f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm" if len(wavelengths) > 1 else f"{wavelengths[0]:.0f}nm"
        output_filename = f'spectral_radiance_{metadata["surface_type"]}_{metadata["atm_mode"]}_{wvl_range}_{timestamp}.png'

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename) if os.path.dirname(output_filename) else metadata.get('output_dir', '.')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save figure
    full_path = os.path.join(output_dir, os.path.basename(output_filename))
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    print(f"\nSpectral radiance plot saved to: {full_path}")

    return fig

if __name__ == '__main__':

    warnings.warn('\nUnder active development ...')

    # radiance simulation
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # example_01_rad_atm1d_clear_over_land()
    # example_02_rad_atm1d_clear_over_ocean()
    # example_03_rad_atm1d_clear_over_snow(wavelength=555.0, mode='afgl')
    results = calculate_spectral_radiance(wavelengths=np.arange(400, 2001, 1),
    solar_zenith_angles=np.array([63]),
    viewing_zenith_angles=np.array([0]),
    viewing_azimuth_angles=np.array([0]),
    solar_azimuth_angles=np.array([0]),
    atm_mode='afgl',
    n_cpus=64,
    surface_type='lambertian'
    )

    # Create and save spectral plot
    fig = plot_spectral_radiance(results)

    results = calculate_spectral_radiance(wavelengths=np.arange(400, 2001, 1),
    solar_zenith_angles=np.array([63]),
    viewing_zenith_angles=np.array([0]),
    viewing_azimuth_angles=np.array([0]),
    solar_azimuth_angles=np.array([0]),
    atm_mode='arcsix',
    n_cpus=64,
    surface_type='lambertian'
    )

    # Create and save spectral plot
    fig = plot_spectral_radiance(results)

    results = calculate_spectral_radiance(wavelengths=np.arange(400, 2001, 1),
    solar_zenith_angles=np.array([63]),
    viewing_zenith_angles=np.array([0]),
    viewing_azimuth_angles=np.array([0]),
    solar_azimuth_angles=np.array([0]),
    atm_mode='afgl',
    n_cpus=64,
    surface_type='snow'
    )

    # Create and save spectral plot
    fig = plot_spectral_radiance(results)

    results = calculate_spectral_radiance(wavelengths=np.arange(400, 2001, 1),
    solar_zenith_angles=np.array([63]),
    viewing_zenith_angles=np.array([0]),
    viewing_azimuth_angles=np.array([0]),
    solar_azimuth_angles=np.array([0]),
    atm_mode='arcsix',
    n_cpus=64,
    surface_type='snow'
    )

    # Create and save spectral plot
    fig = plot_spectral_radiance(results)

    # example_04_rad_atm1d_cloud_over_ocean()


    # example_05_rad_les_cloud_3d(solver='IPA')
    # example_05_rad_les_cloud_3d(solver='3D')
    # example_06_rad_cld_gen_hem()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # for windspeed in np.arange(1.0, 24.1, 0.1):
    #     example_02_rad_atm1d_clear_over_ocean(windspeed=windspeed)
    # for sza in np.append(np.arange(0.0, 90.0, 3.0), 89.9):
    #     example_02_rad_atm1d_clear_over_ocean(sza=sza)
    # for cer in np.arange(1.0, 25.1, 1.0):
    #     example_03_rad_atm1d_cloud_over_ocean(cer=cer)
    # for cot in np.concatenate((np.arange(0.1, 1.0, 0.1), np.arange(1.0, 10.0, 1.0), np.arange(10.0, 50.1, 5.0))):
    #     example_03_rad_atm1d_cloud_over_ocean(cot=cot)
    # for sza in np.append(np.arange(0.0, 90.0, 3.0), 89.9):
    #     example_03_rad_atm1d_cloud_over_ocean(sza=sza)

    # example_07_at3d_rad_cloud_merra(wavelength=650.0)
    # example_07_at3d_rad_cloud_merra(wavelength=550.0)
    # example_07_at3d_rad_cloud_merra(wavelength=450.0)

    pass
