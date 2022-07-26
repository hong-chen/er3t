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

    To run this test, we will need data/test_mca/les.nc
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
    fname_nc  = 'data/test_mca/les.nc'

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

    To run this test, we will need data/test_mca/les.nc
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
    fname_nc  = 'data/test_mca/les.nc'

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



def test_flux_with_les_cloud3d_aerosol1d(fdir, wavelength=650.0, solver='3D', overwrite=True):

    """
    A test run for calculating flux fields using LES data

    To run this test, we will need data/test_mca/les.nc
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
    fname_nc  = 'data/test_mca/les.nc'

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

    To run this test, we will need data/test_mca/les.nc
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
    fname_nc  = 'data/test_mca/les.nc'

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

    To run this test, we will need data/test_mca/les.nc
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
    fname_nc  = 'data/test_mca/les.nc'

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
    fdir = os.path.abspath('tmp-data/test_mca')
    if not os.path.exists(fdir):
        os.makedirs(fdir)


    # for solver in ['3D', 'IPA']:
    #     test_flux_clear_sky(fdir, wavelength=650.0, solver=solver, overwrite=True)


    for solver in ['3D', 'IPA']:
        test_flux_with_les_cloud3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)


    # for solver in ['3D', 'IPA']:
    #     test_radiance_with_les_cloud3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)


    # for solver in ['3D', 'IPA']:
    #     test_flux_with_les_cloud3d_aerosol1d(fdir, wavelength=650.0, solver=solver, overwrite=True)


    # for solver in ['3D', 'IPA']:
    #     test_flux_with_les_cloud3d_aerosol3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)


    # for solver in ['3D', 'IPA']:
    #     test_radiance_with_les_cloud3d_aerosol3d(fdir, wavelength=500.0, solver=solver, overwrite=True, plot=False)



if __name__ == '__main__':

    main()
