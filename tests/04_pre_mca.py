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
from er3t.pre.cld import cld_les, cld_sat
from er3t.pre.sfc import sfc_sat

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng
from er3t.util import send_email, cal_r_twostream
from er3t.util.modis import modis_l1b, modis_l2, modis_09a1, grid_modis_by_extent, grid_modis_by_lonlat, download_modis_https, get_sinusoidal_grid_tag, get_doy_tag



def test_flux_clear_sky(fdir, wavelength=650.0, solver='3D', overwrite=True):

    """
    A test run for clear sky case
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    atm_3ds   = []
    # ------------------------------------------------------------------------------------------------------


    # define mcarats object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # run mcarats
    mca0 = mcarats_ng(
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            fdir='%s/%4.4d/clear-sky/flux_%s' % (fdir, wavelength, solver.lower()),
            target='flux',
            Nrun=3,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=0.0,
            photons=1e6,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=14,
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
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = 'mca-out-flux-%s_clear-sky.h5' % solver.lower()
    out0 = mca_out_ng(fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['f_up']['data']
    #     out0.data['f_down']['data']
    #     out0.data['f_down_direct']['data']
    #     out0.data['f_down_diffuse']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_png = 'mca-out-flux-%s_clear-sky.png' % solver.lower()

    fig = plt.figure(figsize=(4, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(out0.data['f_up']['data']          , atm0.lev['altitude']['data'], color='red')
    ax1.plot(out0.data['f_down']['data']        , atm0.lev['altitude']['data'], color='blue')
    ax1.plot(out0.data['f_down_direct']['data'] , atm0.lev['altitude']['data'], color='green')
    ax1.plot(out0.data['f_down_diffuse']['data'], atm0.lev['altitude']['data'], color='pink')
    ax1.set_xlabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_ylabel('Altitude [km]')
    ax1.set_ylim((0.0, 20.0))
    ax1.set_xlim(left=0.0)

    patches_legend = [
                mpatches.Patch(color='red'  , label='Up'),
                mpatches.Patch(color='blue' , label='Down'),
                mpatches.Patch(color='green', label='Down Direct'),
                mpatches.Patch(color='pink' , label='Down Diffuse')
                ]
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

    ax1.set_title('Clear Sky (%s Mode), Flux Profile' % solver)
    plt.savefig(fname_png, bbox_inches='tight')
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------

    # send email
    # receiver = 'chenhong.cu@gmail.com'
    # send_email(content='test_flux_clear_sky complete.', files=[fname_png, fname_h5], receiver=receiver)



def test_flux_with_les_cloud3d(fdir, wavelength=650.0, solver='3D', overwrite=True, plot=False):

    """
    A test run for calculating flux fields using LES data

    To run this test, we will need data/les.nc
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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
    # file name of the netcdf file
    fname_nc  = 'data/les.nc'

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=overwrite)

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


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
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
            Nrun=3,
            surface_albedo=0.03,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=45.0,
            fdir='%s/%4.4d/les_cld3d/flux_%s' % (fdir, wavelength, solver.lower()),
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
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = 'mca-out-flux-%s_les_cld3d.h5' % solver.lower()
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['f_up']['data']
    #     out0.data['f_down']['data']
    #     out0.data['f_down_direct']['data']
    #     out0.data['f_down_diffuse']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if plot:
        z_index = 4
        fname_png = 'mca-out-flux-%s_les_cld3d.png' % solver.lower()

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(np.transpose(out0.data['f_up']['data'][:, :, z_index]), cmap='jet', vmin=0.0, vmax=1.6, origin='lower')
        plt.colorbar(cs)
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        ax1.set_title('LES 3D Cloud (%s Mode), Upwelling Flux at %d km' % (solver, atm0.lev['altitude']['data'][z_index]))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        # receiver = 'chenhong.cu@gmail.com'
        # send_email(content='test_flux_with_les_cloud complete.', files=[fname_png], receiver=receiver)
    # ------------------------------------------------------------------------------------------------------



def test_radiance_with_les_cloud3d(fdir, wavelength=650.0, solver='3D', overwrite=True, plot=False):

    """
    A test run for calculating radiance fields using LES data (nadir radiance at the satellite altitude of 705km)

    To run this test, we will need data/les.nc
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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
    # file name of the netcdf file
    fname_nc  = 'data/les.nc'

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=overwrite)

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


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
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
            surface_albedo=0.03,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=45.0,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            sensor_altitude=705000.0,
            fdir='%s/%4.4d/les_cld3d/rad_%s' % (fdir, wavelength, solver.lower()),
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

    fname_h5 = 'mca-out-rad-%s_les_cld3d.h5' % solver.lower()
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if plot:
        fname_png = 'mca-out-rad-%s_les_cld3d.png' % solver.lower()

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
        plt.colorbar(cs)
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        ax1.set_title('LES 3D Cloud (%s Mode), Radiance at %.2f nm' % (solver, wavelength))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        # receiver = 'chenhong.cu@gmail.com'
        # send_email(content='test_radiance_with_les_cloud complete.', files=[fname_png], receiver=receiver)
    # ------------------------------------------------------------------------------------------------------



def test_radiance_with_modis_cloud_and_surface(fdir, wavelength=650.0, solver='3D', overwrite=True):

    """
    A test run for calculating radiance fields using MODIS L1B calibrated radiance and MOD09A1 8 day surface reflectance data

    To run this test, we will need data/MYD02QKM.A2017237.2035.061.2018034164525.hdf
                                   data/MYD06_L2.A2017237.2035.061.2018038220417.hdf
                                   data/MOD09A1.A2017233.h08v05.006.2017249210521.hdf
                                   data/MOD09A1.A2017233.h08v06.006.2017249205504.hdf
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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
    extent      = [-112.0, -111.0, 29.4, 30.4]

    fname_l1b   = 'data/MYD02QKM.A2017237.2035.061.2018034164525.hdf'
    fname_l2    = 'data/MYD06_L2.A2017237.2035.061.2018038220417.hdf'

    l1b = modis_l1b(fnames=[fname_l1b], extent=extent)
    l2  = modis_l2(fnames=[fname_l2], extent=extent, vnames=['Solar_Zenith', 'Solar_Azimuth', 'Sensor_Zenith', 'Sensor_Azimuth'])

    lon_2d, lat_2d, ref_2d = grid_modis_by_extent(l1b.data['lon']['data'], l1b.data['lat']['data'], l1b.data['ref']['data'][0, ...], extent=extent)
    lon_2d, lat_2d, rad_2d = grid_modis_by_extent(l1b.data['lon']['data'], l1b.data['lat']['data'], l1b.data['rad']['data'][0, ...], extent=extent)

    logic_nan = (np.isnan(ref_2d)) | (np.isnan(rad_2d))
    ref_2d[logic_nan] = 0.0
    rad_2d[logic_nan] = 0.0

    a0         = np.median(ref_2d)
    mu0        = np.cos(np.deg2rad(l2.data['solar_zenith']['data'].mean()))
    threshold  = a0
    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=0.2, mu=mu0)

    indices    = np.where(ref_2d>threshold)
    indices_x  = indices[0]
    indices_y  = indices[1]

    cot_2d = np.zeros_like(ref_2d)
    cer_2d = np.zeros_like(ref_2d); cer_2d[...] = 1.0
    for i in range(indices_x.size):
        cot_2d[indices_x[i], indices_y[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
        cer_2d[indices_x[i], indices_y[i]] = 12.0

    l1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees', data=lon_2d)
    l1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees', data=lat_2d)
    l1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'    , data=cot_2d*8)
    l1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'  , data=cer_2d)

    fname_cld   = '%s/modis.pk' % fdir
    cld0        = cld_sat(sat_obj=l1b, fname=fname_cld, cth=3.0, overwrite=overwrite)

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



    # define a surface object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fnames = ['data/MOD09A1.A2017233.h08v05.006.2017249210521.hdf', 'data/MOD09A1.A2017233.h08v06.006.2017249205504.hdf']
    mod09 = modis_09a1(fnames=fnames, extent=extent)
    lon_2d, lat_2d, surf_ref_2d = grid_modis_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][0, ...], extent=extent)
    logic_nan = np.isnan(surf_ref_2d)
    surf_ref_2d[logic_nan] = 0.0

    mod09.data['alb_2d'] = dict(data=surf_ref_2d, name='Surface albedo', units='N/A')
    mod09.data['lon_2d'] = dict(data=lon_2d, name='Longitude', units='degrees')
    mod09.data['lat_2d'] = dict(data=lat_2d, name='Latitude' , units='degrees')

    fname_sfc   = '%s/sfc.pk' % fdir
    sfc0 = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=extent, overwrite=overwrite)

    sfc_2d = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/sfc_2d.bin' % fdir)
    # ------------------------------------------------------------------------------------------------------


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
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
            date=datetime.datetime(2017, 8, 25),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sfc_2d=sfc_2d,
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = l2.data['solar_zenith']['data'].mean(),
            solar_azimuth_angle  = l2.data['solar_azimuth']['data'].mean(),
            sensor_zenith_angle  = l2.data['sensor_zenith']['data'].mean(),
            sensor_azimuth_angle = l2.data['sensor_azimuth']['data'].mean(),
            sensor_altitude=705000.0,
            fdir='%s/%4.4d/modis/rad_%s' % (fdir, wavelength, solver.lower()),
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
    # out0 = mca_out_ng(fname='mca-out-flux-3d_modis.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_modis.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_modis.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    out0 = mca_out_ng(fname='mca-out-rad-%s_modis.h5' % solver.lower(), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['f_up']['data']
    #     out0.data['f_down']['data']
    #     out0.data['f_down_direct']['data']
    #     out0.data['f_down_diffuse']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_png = 'mca-out-rad-%s_modis.png' % solver.lower()

    fig = plt.figure(figsize=(18, 5.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    cs = ax1.imshow(rad_2d.T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3)
    ax1.set_xlabel('X Index')
    ax1.set_ylabel('Y Index')
    ax1.set_title('Radiance from MODIS L1B (650nm)')

    cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3)
    ax2.set_xlabel('X Index')
    ax2.set_ylabel('Y Index')
    ax2.set_title('Radiance from ER3T 3D (650 nm)')

    # logic = (cot_2d>0.0)
    # ax3.scatter(rad_2d[logic], out0.data['rad']['data'][logic], c='black', s=10, lw=0.0)
    # ax3.scatter(rad_2d[np.logical_not(logic)], out0.data['rad']['data'][np.logical_not(logic)], c='gray', s=10, lw=0.0)

    logic = (cot_2d>0.0)

    # heatmap
    xedges = np.linspace(0.0, 0.6, 60)
    yedges = np.linspace(0.0, 0.6, 60)
    heatmap, xedges, yedges = np.histogram2d(rad_2d[logic].ravel(), out0.data['rad']['data'][logic].ravel(), bins=(xedges, yedges))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)

    min_val = heatmap[heatmap>0].min()
    max_val = heatmap.max()
    levels_heat = np.linspace(min_val, max_val, 21)
    levels_heat = np.concatenate((np.linspace(levels_heat[0], levels_heat[1], 10), levels_heat[2:]))

    cs2 = ax3.contourf(XX, YY, heatmap, levels=levels_heat, cmap='jet', extend='both')
    cmap = cs2.get_cmap()
    cmap.set_under('white')
    cs2.set_cmap(cmap)


    ax3.axvline(0.1, color='gray', ls='--')
    ax3.set_xlim((0.0, 0.6))
    ax3.set_ylim((0.0, 0.6))
    ax3.set_xlabel('Radiance (ER3T 3D)')
    ax3.set_ylabel('Radiance (MODIS L1B)')

    plt.savefig(fname_png, bbox_inches='tight')
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------

    # receiver = 'chenhong.cu@gmail.com'
    # send_email(content='test_radiance_with_modis_cloud_and_surface complete.', files=[fname_png], receiver=receiver)



def test_flux_with_les_cloud3d_aerosol1d(fdir, wavelength=650.0, solver='3D', overwrite=True):

    """
    A test run for calculating flux fields using LES data

    To run this test, we will need data/les.nc
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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
    # file name of the netcdf file
    fname_nc  = 'data/les.nc'

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=overwrite)

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


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
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

    # add homogeneous 1d mcarats "atmosphere", aerosol layer
    aod    = 0.4 # aerosol optical depth
    ssa    = 0.9 # aerosol single scattering albedo
    asy    = 0.6 # aerosol asymmetry parameter
    z_bot  = 5.0 # altitude of layer bottom in km
    z_top  = 8.0 # altitude of layer top in km
    aer_ext = aod / (atm0.lay['thickness']['data'].sum()*1000.0)

    atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy, z_bottom=z_bot, z_top=z_top)
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
            Nrun=3,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=45.0,
            fdir='%s/%4.4d/les_cld3d_aer1d/flux_%s' % (fdir, wavelength, solver.lower()),
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
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = 'mca-out-flux-%s_les_cld3d_aer1d.h5' % solver.lower()
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['f_up']['data']
    #     out0.data['f_down']['data']
    #     out0.data['f_down_direct']['data']
    #     out0.data['f_down_diffuse']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    z_index = 4
    fname_png = 'mca-out-flux-%s_les_cld3d_aer1d.png' % solver.lower()

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    cs = ax1.imshow(np.transpose(out0.data['f_up']['data'][:, :, z_index]), cmap='jet', vmin=0.0, vmax=1.6, origin='lower')
    plt.colorbar(cs)
    ax1.set_xlabel('X Index')
    ax1.set_ylabel('Y Index')
    ax1.set_title('LES 3D Cloud (%s Mode), Upwelling Flux at %d km' % (solver, atm0.lev['altitude']['data'][z_index]))
    plt.savefig(fname_png, bbox_inches='tight')
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------

    # receiver = 'chenhong.cu@gmail.com'
    # send_email(content='test_flux_with_les_cloud complete.', files=[fname_png], receiver=receiver)



def test_flux_with_les_cloud3d_aerosol3d(fdir, wavelength=650.0, solver='3D', overwrite=True, plot=False):

    """
    A test run for calculating flux fields using LES data

    To run this test, we will need data/les.nc
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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
    # file name of the netcdf file
    fname_nc  = 'data/les.nc'

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=overwrite)

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


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, overwrite=False)

    # 3d aerosol
    # ================================================================================
    ext3d = np.zeros_like(atm3d0.nml['Atm_extp3d']['data'][:, :, :, 0])
    ext3d[:, :, 0] = 0.00012
    ext3d[:, :, 1] = 0.00008

    omg3d = np.zeros_like(atm3d0.nml['Atm_extp3d']['data'][:, :, :, 0])
    omg3d[...] = 0.85

    apf3d = np.zeros_like(atm3d0.nml['Atm_extp3d']['data'][:, :, :, 0])
    apf3d[...] = 0.6
    atm3d0.add_mca_3d_atm(ext3d=ext3d, omg3d=omg3d, apf3d=apf3d)
    # ================================================================================

    atm3d0.gen_mca_3d_atm_file(fname='%s/mca_atm_3d.bin' % fdir)
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
            Nrun=3,
            surface_albedo=0.03,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=45.0,
            fdir='%s/%4.4d/les_cld3d_aer3d/flux_%s' % (fdir, wavelength, solver.lower()),
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
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = mca_out_ng(fname='mca-out-flux-3d_les.h5', mca_obj=mca0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = 'mca-out-flux-%s_les_cld3d_aer3d.h5' % solver.lower()
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['f_up']['data']
    #     out0.data['f_down']['data']
    #     out0.data['f_down_direct']['data']
    #     out0.data['f_down_diffuse']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if plot:
        for z_index in range(8):
            fname_png = 'mca-out-flux-%s_les_cld3d_aer3d_%2.2d.png' % (solver.lower(), z_index)

            fig = plt.figure(figsize=(12, 6))

            ax1 = fig.add_subplot(121)
            cs = ax1.imshow(np.transpose(out0.data['f_up']['data'][:, :, z_index]), cmap='jet', vmin=0.0, vmax=1.6, origin='lower')
            ax1.set_xlabel('X Index')
            ax1.set_ylabel('Y Index')
            ax1.set_title('LES 3D Cloud (%s Mode), Upwelling Flux at %d km' % (solver, atm0.lev['altitude']['data'][z_index]))

            ax2 = fig.add_subplot(122)
            cs = ax2.imshow(np.transpose(out0.data['f_down']['data'][:, :, z_index]), cmap='jet', vmin=0.0, vmax=1.6, origin='lower')
            ax2.set_xlabel('X Index')
            ax2.set_ylabel('Y Index')
            ax2.set_title('LES 3D Cloud (%s Mode), Downwelling Flux at %d km' % (solver, atm0.lev['altitude']['data'][z_index]))

            plt.savefig(fname_png, bbox_inches='tight')
            plt.close(fig)
    # ------------------------------------------------------------------------------------------------------



def test_radiance_with_les_cloud3d_aerosol3d(fdir, wavelength=650.0, solver='3D', overwrite=True, plot=False):

    """
    A test run for calculating radiance fields using LES data (nadir radiance at the satellite altitude of 705km)

    To run this test, we will need data/les.nc
    """


    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

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
    # file name of the netcdf file
    fname_nc  = 'data/les.nc'

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=overwrite)

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


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inhomogeneous 3d mcarats "atmosphere"
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, overwrite=False)

    # 3d aerosol
    # ================================================================================
    ext3d = np.zeros_like(atm3d0.nml['Atm_extp3d']['data'][:, :, :, 0])
    ext3d[:, :, 0] = 0.00012
    ext3d[:, :, 1] = 0.00008

    omg3d = np.zeros_like(atm3d0.nml['Atm_extp3d']['data'][:, :, :, 0])
    omg3d[...] = 0.85

    apf3d = np.zeros_like(atm3d0.nml['Atm_extp3d']['data'][:, :, :, 0])
    apf3d[...] = 0.6
    atm3d0.add_mca_3d_atm(ext3d=ext3d, omg3d=omg3d, apf3d=apf3d)
    # ================================================================================

    atm3d0.gen_mca_3d_atm_file(fname='%s/mca_atm_3d.bin' % fdir)
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
            surface_albedo=0.03,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=45.0,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            sensor_altitude=705000.0,
            fdir='%s/%4.4d/les_cld3d_aer3d/rad_%s' % (fdir, wavelength, solver.lower()),
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

    fname_h5 = 'mca-out-rad-%s_les_cld3d_aer3d.h5' % solver.lower()
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    # ------------------------------------------------------------------------------------------------------


    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if plot:
        fname_png = 'mca-out-rad-%s_les_cld3d_aer3d.png' % solver.lower()

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
        plt.colorbar(cs)
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        ax1.set_title('Radiance at %.2f nm (%s Mode)' % (solver, wavelength))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        # receiver = 'chenhong.cu@gmail.com'
        # send_email(content='test_radiance_with_les_cloud complete.', files=[fname_png], receiver=receiver)
    # ------------------------------------------------------------------------------------------------------



def main():


    # create tmp-data/04 directory if it does not exist
    fdir = os.path.abspath('tmp-data/04')
    if not os.path.exists(fdir):
        os.makedirs(fdir)


    # for solver in ['3D', 'IPA']:
    #     test_flux_clear_sky(fdir, wavelength=650.0, solver=solver, overwrite=True)


    for solver in ['3D', 'IPA']:
        test_flux_with_les_cloud3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)


    # for solver in ['3D', 'IPA']:
    #     test_radiance_with_les_cloud3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)


    # for solver in ['3D', 'IPA']:
    #     test_radiance_with_modis_cloud_and_surface(fdir, wavelength=650.0, solver=solver, overwrite=True)


    # for solver in ['3D', 'IPA']:
    #     test_flux_with_les_cloud3d_aerosol1d(fdir, wavelength=650.0, solver=solver, overwrite=True)


    # for solver in ['3D', 'IPA']:
    #     test_flux_with_les_cloud3d_aerosol3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)


    # for solver in ['3D', 'IPA']:
    #     test_radiance_with_les_cloud3d_aerosol3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)



if __name__ == '__main__':

    main()
