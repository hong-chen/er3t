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
        sza=30.0,
        saa=0.0,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    1D clear over land (LSRT) simulation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir = f"{fdir0}/tmp-data/{name_tag}/{_metadata['Function']}/{int(wavelength):04d}"

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = f"{fdir}/atm.pk"

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
    fname_abs = f"{fdir}/abs.pk"

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
    fname_cld = f"{fdir}/cld.pk"

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

    fname_sfc = f"{fdir}/sfc.pk"
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname=f"{fdir}/shdom-sfc.txt", overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname=f"{fdir}/shdom-ckd.txt", overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname=f"{fdir}/shdom-prp.txt", fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
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
            fdir=f"{fdir}/rad_{solver.lower()}",
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

    fname_h5 = f"{fdir}/shd-out-rad-{solver.lower()}_{_metadata['Function']}.h5"
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = f"{name_tag}-{_metadata['Function']}_{solver.lower()}.png"

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.04, location='bottom')

        ax1.set_title(f"Radiance at {wavelength:.1f} nm (SZA={sza:.1f}$^\\circ$, {solver} Mode)")
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
        saa=0.0,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    1D clear over ocean (Ross-Sea) simulation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir = f"{fdir0}/tmp-data/{name_tag}/{_metadata['Function']}/{windspeed:04.1f}_{pigment:04.2f}_{sza:04.1f}/{int(wavelength):04d}"

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = f"{fdir}/atm.pk"

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
    fname_abs = f"{fdir}/abs.pk"

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
    fname_cld = f"{fdir}/cld.pk"

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

    fname_sfc = f"{fdir}/sfc.pk"
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname=f"{fdir}/shdom-sfc.txt", overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname=f"{fdir}/shdom-ckd.txt", overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname=f"{fdir}/shdom-prp.txt", fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
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
            fdir=f"{fdir}/rad_{solver.lower()}",
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

    fname_h5 = f"{fdir}/shd-out-rad-{solver.lower()}_{_metadata['Function']}.h5"
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = f"{name_tag}-{_metadata['Function']}_{solver.lower()}.png"

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.04, location='bottom')

        ax1.set_title(f"Radiance at {wavelength:.1f} nm (SZA={sza:.1f}$^\\circ$, {solver} Mode)")
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def example_03_rad_atm1d_clear_over_snow(
        wavelength=555.0,
        sza=63.0,
        saa=0.0,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    1D clear over snow (LSRT-J) simulation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir = f"{fdir0}/tmp-data/{name_tag}/{_metadata['Function']}/{int(wavelength):04d}"

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = f"{fdir}/atm.pk"

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
    fname_abs = f"{fdir}/abs.pk"

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
    fname_cld = f"{fdir}/cld.pk"

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
            'fiso': np.array([0.986]).reshape((1, 1)),
            'fgeo': np.array([0.033]).reshape((1, 1)),
            'fvol': np.array([0.000]).reshape((1, 1)),
            'fj'  : np.array([0.447]).reshape((1, 1)),
            'alpha': np.array([0.3]).reshape((1, 1)),
            }

    fname_sfc = f"{fdir}/sfc.pk"
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname=f"{fdir}/shdom-sfc.txt", overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname=f"{fdir}/shdom-ckd.txt", overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname=f"{fdir}/shdom-prp.txt", fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    raa = er3t.util.util.calculate_raa(saa=saa, vaa=vaa, forward_scattering='positive')

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
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
            fdir=f"{fdir}/rad_{solver.lower()}",
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

    fname_h5 = f"{fdir}/shd-out-rad-{solver.lower()}_{_metadata['Function']}.h5"
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = f"{name_tag}-{_metadata['Function']}_{solver.lower()}.png"

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.04, location='bottom')

        ax1.set_title(f"Radiance at {wavelength:.1f} nm (SZA={sza:.1f}$^\\circ$, {solver} Mode)")
        fig.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
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
        saa=0.0,
        windspeed=1.0,
        pigment=0.01,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    1D water cloud (Mie) over ocean (Ross-Sea) simulation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir = f"{fdir0}/tmp-data/{name_tag}/{_metadata['Function']}/{cot:04.1f}_{cer:04.1f}_{sza:04.1f}"

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = f"{fdir}/atm.pk"

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
    fname_abs = f"{fdir}/abs.pk"

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
    fname_cld = f"{fdir}/cld.pk"

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

    fname_sfc = f"{fdir}/sfc.pk"
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname=f"{fdir}/shdom-sfc.txt", overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname=f"{fdir}/shdom-ckd.txt", overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname=f"{fdir}/shdom-prp.txt", fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
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
            Nmu=32,
            Nphi=64,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir=f"{fdir}/rad_{solver.lower()}",
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

    fname_h5 = f"{fdir}/shd-out-rad-{solver.lower()}_{_metadata['Function']}.h5"
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = f"{name_tag}-{_metadata['Function']}_{solver.lower()}.png"

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.04, location='bottom')

        ax1.set_title(f"Radiance at {wavelength:.1f} nm (SZA={sza:.1f}$^\\circ$, {solver} Mode)")
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
    sys.exit()

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


def read_cam_viirs(
        data={},
        ):

    fname_geo = '/Users/hchen/Work/mygit/cam/products/data/rad-viirs_raw/ICDBG/2021/138/ICDBG_j01_d20210518_t0527521_e0529166_b18115_c20241029025037855295_oebc_ops.h5'
    fname_rad = '/Users/hchen/Work/mygit/cam/products/data/rad-viirs_raw/IVCDB/2021/138/IVCDB_j01_d20210518_t0527521_e0529166_b18115_c20241029025037855295_oebc_ops.h5'

    with h5py.File(fname_rad, 'r') as f_rad:

        rad = f_rad['All_Data/VIIRS-DualGain-Cal-IP_All/radiance_3'][...]/1000.0

    with h5py.File(fname_geo, 'r') as f_geo:

        utc = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/MidTime'][...]
        lon = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/Longitude'][...]
        lat = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/Latitude'][...]
        sza = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/SolarZenithAngle'][...]
        saa = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/SolarAzimuthAngle'][...]
        sza = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/SolarZenithAngle'][...]
        saa = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/SolarAzimuthAngle'][...]
        vza = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/SatelliteZenithAngle'][...]
        vaa = f_geo['All_Data/VIIRS-MOD-UNAGG-GEO_All/SatelliteAzimuthAngle'][...]

    logic_nan = (lon>180.0) | (lon<-180.0) | (lat>90.0) | (lat<-90.0) | (rad<0.0)
    rad[logic_nan] = np.nan

    jday0 = er3t.util.dtime_to_jday(datetime.datetime(1958, 1, 1))
    jday = jday0 + utc.mean()/86400.0/1.0e6
    data['utc_cam'] = (jday-int(jday)) * 24.0 * 60.0 # UTC minute
    data['lon_cam'] = lon
    data['lat_cam'] = lat
    data['sza_cam'] = sza
    data['saa_cam'] = saa
    data['vza_cam'] = vza
    data['vaa_cam'] = vaa
    data['rad_cam'] = rad
    data['extent'] = [110.3821792602539, 127.21611785888672, -1.8479857444763184, 5.583887100219727]

    return data


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
    fdir='%s/tmp-data/%s/%s/%4.4d' % (fdir0, name_tag, _metadata['Function'], wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the netcdf file
    fname_nc  = '/Users/hchen/Downloads/MERRA2_400.inst3_3d_asm_Np.20210518.nc4'

    # file name of the pickle file for cloud
    fname_cld = '%s/cld.pk' % fdir

    # cloud object
    cld0      = er3t.pre.cld.cld_merra(fname_nc=fname_nc, fname=fname_cld, coarsen=[1, 1, 1], overwrite=overwrite)
    cld0.lay['extinction']['data'][...] = 0.0

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
    fname_abs = '%s/abs.pk' % (fdir)

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
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='/Users/hchen/Work/mygit/cam/products/data/shdom-sfc_globe-mix.txt', overwrite=False, force=False)
    sfc_2d.nml['header']['data'] = 'X'
    sfc_2d.nml['NX']['data'] = 360
    sfc_2d.nml['NY']['data'] = 180
    sfc_2d.nml['dx']['data'] = cld0.lay['dx']['data']*cld0.lay['nx']['data']/sfc_2d.nml['NX']['data']
    sfc_2d.nml['dy']['data'] = cld0.lay['dy']['data']*cld0.lay['ny']['data']/sfc_2d.nml['NY']['data']

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % (fdir), overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=False)
    atm3d0.nml['PROPFILE']['data'] = '/Users/hchen/Work/mygit/shdom/data/shdom-prp_atm-clear.txt'
    atm3d0.nml['NZ']['data'] = 10
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.0, 5.0)
    # vza = np.repeat(30.0, vaa.size)
    # vaa = np.arange(45.0, 46.0, 1.0)
    # vza = np.repeat(0.0, vaa.size)

    data = read_cam_viirs()
    lon = data['lon_cam'][::5, ::5]
    lat = data['lat_cam'][::5, ::5]
    vaa = data['vaa_cam'][::5, ::5]
    vza = data['vza_cam'][::5, ::5]
    # print(vza.min())
    # print(vza.max())
    # print(vaa.min())
    # print(vaa.max())
    lon_mean = lon.mean()
    lat_mean = lat.mean()
    # sensor_xpos = (lon_mean+180.0)*cld0.lay['dx']['data']
    # sensor_ypos = (lat_mean+ 90.0)*cld0.lay['dy']['data']
    sensor_xpos = 25000.0
    sensor_ypos = 25000.0
    # print(sensor_xpos)
    # print(sensor_ypos)

    # read in camera
    #╭────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────╯#

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 8),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Ng=abs0.Ng,
            # Niter=10,
            Niter=0,
            split_acc=1.0e-4,
            target='radiance',
            solar_zenith_angle=30.0,
            solar_azimuth_angle=0.0,
            # sensor_type='radiometer',
            # sensor_type='sensor',
            sensor_type='camera2',
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_xpos=sensor_xpos,
            sensor_ypos=sensor_ypos,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='mpi',
            overwrite=True,
            force=False,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#

    os.system("open tmp-data/00_er3t_shd/example_07_at3d_rad_cloud_merra/0550/rad_3d/shdom-out_g-000.pgm")
    sys.exit()

    # processing
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    # plot = False
    if plot:
        data_cam = np.fromfile('tmp-data/00_er3t_shd/example_07_at3d_rad_cloud_merra/0550/rad_3d/shdom-out_g-000.txt.sHdOm-out', dtype='<f4').reshape(lon.shape)
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        cs = ax1.scatter(lon, lat, c=data_cam, s=2, lw=0.0, cmap='jet')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        # cbar.set_label('', rotation=270, labelpad=4.0)
        # cbar.set_ticks([])
        # cax.axis('off')
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f'{_metadata_['Function']}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#
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




if __name__ == '__main__':

    warnings.warn('\nUnder active development ...')

    # radiance simulation
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # example_01_rad_atm1d_clear_over_land()
    # example_02_rad_atm1d_clear_over_ocean()
    # example_03_rad_atm1d_clear_over_snow()
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
    example_07_at3d_rad_cloud_merra(wavelength=550.0, overwrite=False)
    # example_07_at3d_rad_cloud_merra(wavelength=450.0)

    pass
