import os
import sys
import glob
import datetime
import warnings
import tqdm
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# mpl.use('Agg')


import er3t


name_tag = '00_er3t_bmk'

params = {
                         'date': datetime.datetime(2014, 9, 11),
                  'wavelength' : 650.0,
               'surface_albedo': 0.03,
          'atmospheric_profile': '%s/afglt.dat' % er3t.common.fdir_data_atmmod,
          'solar_zenith_angle' : 28.2797,
         'solar_azimuth_angle' : 238.9053,
         'sensor_zenith_angle' : 0.0,
         # 'sensor_zenith_angle' : 6.2445,
        'sensor_azimuth_angle' : 261.9049,
             'sensor_altitude' : 705000.0,
            'cloud_top_height' : 2.0,
 'cloud_geometrical_thickness' : 1.0,
                      'Nphoton': 5e6,
                         'Ncpu': 'auto',
                         'cer0': 10.0,
                         'cot' : np.concatenate((              \
                                 np.arange(0.0, 2.0, 0.5),     \
                                 np.arange(2.0, 30.0, 2.0),    \
                                 np.arange(30.0, 60.0, 5.0),   \
                                 np.arange(60.0, 100.0, 10.0), \
                                 np.arange(100.0, 401.0, 50.0) \
                                )),
                }
#


def test_00_util():

    command = 'lss %s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    os.system(command)

    command = 'lss %s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    os.system(command)





def test_01_flux_one_clear(wavelengh, plot=True):

    params = {
                    'date': datetime.datetime(2023, 5, 26),
         'atmosphere_file': '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
          'surface_albedo': 0.03,
      'solar_zenith_angle': 0.0,
              'wavelength': wavelength,
         'output_altitude': np.arange(0.0, 35.1, 0.5),
         }

    data_lrt = lrt_flux_one_clear(params)

    data_mca = mca_flux_one_clear(params)

    error = np.abs(data_mca['f_down']-data_lrt['f_down'])/data_lrt['f_down']*100.0

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Wavelength %.1f nm [Error %.1f%%]' % (params['wavelength'], error.mean()))
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(121)
        ax1.plot(data_lrt['f_up']          , params['output_altitude'], color='red'    , lw=3.0, alpha=0.6, ls='--')
        ax1.plot(data_lrt['f_down_diffuse'], params['output_altitude'], color='magenta', lw=3.0, alpha=0.6, ls='--')
        ax1.errorbar(data_mca['f_up']          , params['output_altitude'], xerr=data_mca['f_up_std']          , color='red'     , lw=1.0, alpha=1.0)
        ax1.errorbar(data_mca['f_down_diffuse'], params['output_altitude'], xerr=data_mca['f_down_diffuse_std'],  color='magenta', lw=1.0, alpha=1.0)
        ax1.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax1.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_xlim(0.0, 0.5)

        ax2 = fig.add_subplot(122)
        ax2.plot(data_lrt['f_down']       , params['output_altitude'], color='blue', lw=3.0, alpha=0.6, ls='--')
        ax2.plot(data_lrt['f_down_direct'], params['output_altitude'], color='cyan', lw=3.0, alpha=0.6, ls='--')
        ax2.errorbar(data_mca['f_down']       , params['output_altitude'], xerr=data_mca['f_down_std']       , color='blue', lw=1.0, alpha=1.0)
        ax2.errorbar(data_mca['f_down_direct'], params['output_altitude'], xerr=data_mca['f_down_direct_std'], color='cyan', lw=1.0, alpha=1.0)
        ax2.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax2.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax2.set_xlim(0.0, 2.0)
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%06.1fnm.png' % (_metadata['Function'], params['wavelength']), bbox_inches='tight', metadata=_metadata)
        # plt.show()
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def check_solar():

    data_kurudz = np.loadtxt('%s/solar/kurudz_1.0nm.dat' % er3t.common.fdir_data)
    data_abs16g = np.loadtxt('%s/solar/solar_16g.dat' % er3t.common.fdir_data)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax1.scatter(data_abs16g[:, 0], data_abs16g[:, 1]/1000.0, s=10, c='r', lw=0.0)
        ax1.scatter(data_kurudz[:, 0], data_kurudz[:, 1]/1000.0, s=4, c='k', lw=0.0)
        ax1.set_xlim((200, 2400))
        ax1.set_ylim((0, 2.4))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')

        patches_legend = [
                          mpatches.Patch(color='black' , label='Kurudz 1nm'), \
                          mpatches.Patch(color='red'   , label='ABS_16G'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # plt.savefig('%s.png' % _metadata_['Function'], bbox_inches='tight', metadata=_metadata_)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#
    print(data_kurudz)
    print(data_abs16g)

    sys.exit()







def lrt_rad_one_clear(params):

    """
    libRadtran radiance calculation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (er3t.common.fdir_examples, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = params['atmosphere_file']
    # lrt_cfg['mol_abs_param'] = 'lowtran'
    # lrt_cfg['mol_abs_param'] = 'reptran coarse'
    # lrt_cfg['solar_file']    = '%s/solar/solar_16g.dat' % (er3t.common.fdir_data)
    # lrt_cfg['mol_abs_param'] = 'reptran_channel modis_aqua_b01'
    # lrt_cfg['output_process'] = 'per_band'

    init = er3t.rtm.lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp,
            date        = params['date'],
            surface_albedo     = params['surface_albedo'],
            solar_zenith_angle = params['solar_zenith_angle'],
            solar_azimuth_angle = params['solar_azimuth_angle'],
            sensor_zenith_angle = params['sensor_zenith_angle'],
            sensor_azimuth_angle = params['sensor_azimuth_angle'],
            wavelength         = params['wavelength'],
            output_altitude    = params['output_altitude'],
            lrt_cfg            = lrt_cfg,
            # input_dict_extra   = {
                # 'output_process': 'integrate',
                # 'output_process': 'sum',
                # },
            # mute_list = ['slit_function_file', 'spline', 'wavelength'],
            # mute_list = ['slit_function_file', 'spline', 'source solar'],
            # mute_list = ['slit_function_file', 'spline'],
            )
    er3t.rtm.lrt.lrt_run(init)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_rad([init])

    data = {
                'rad': np.squeeze(data0.rad),
            }

    return data


def test_00_solar_old():

    wvl = np.arange(300.0, 2500.1, 1.0)

    # get solar
    #╭────────────────────────────────────────────────────────────────────────────╮#
    solar_crk_16g = np.zeros_like(wvl)
    solar_rep_f = np.zeros_like(wvl)
    solar_rep_m = np.zeros_like(wvl)
    solar_rep_c = np.zeros_like(wvl)
    solar_kurudz_1nm = np.zeros_like(wvl)

    levels = np.linspace(0.0, 20.0, 41)
    atm0 = er3t.pre.atm.atm_atmmod(levels=levels)

    for i, wvl0 in enumerate(tqdm.tqdm(wvl)):
        # abs_crk0 = er3t.pre.abs.abs_16g(wavelength=wvl0, atm_obj=atm0)
        abs_rep_f0 = er3t.pre.abs.abs_rep(wavelength=wvl0, target='fine', atm_obj=atm0)
        # abs_rep_m0 = er3t.pre.abs.abs_rep(wavelength=wvl0, target='medium', atm_obj=atm0)
        # abs_rep_c0 = er3t.pre.abs.abs_rep(wavelength=wvl0, target='coarse', atm_obj=atm0)

        # solar_crk_16g[i] = np.sum(abs_crk0.coef['solar']['data'] * abs_crk0.coef['weight']['data']) * 1000.0
        solar_rep_f[i] = np.sum(abs_rep_f0.coef['solar']['data'] * abs_rep_f0.coef['weight']['data']) * 1000.0
        # solar_rep_m[i] = np.sum(abs_rep_m0.coef['solar']['data'] * abs_rep_m0.coef['weight']['data']) * 1000.0
        # solar_rep_c[i] = np.sum(abs_rep_c0.coef['solar']['data'] * abs_rep_c0.coef['weight']['data']) * 1000.0
        print('%5d %.6e' % (wvl0, solar_rep_f[i]))
    #╰────────────────────────────────────────────────────────────────────────────╯#
    sys.exit()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # ax1.plot(wvl, solar_crk_16g, lw=2, c='k')
        ax1.plot(wvl, solar_rep_f, lw=2, c='k')
        # ax1.plot(wvl, (solar_rep_f-solar_crk_16g)/solar_crk_16g*100.0, lw=2, c='r')
        # ax1.plot(wvl, (solar_rep_m-solar_crk_16g)/solar_crk_16g*100.0, lw=2, c='g')
        # ax1.plot(wvl, (solar_rep_c-solar_crk_16g)/solar_crk_16g*100.0, lw=2, c='b')
        ax1.axhline(0.0, color='gray', ls=':')
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
        fname_fig = '%s_%s.png' % (_metadata_['Date'], _metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def test_00_solar():

    wvl = np.arange(300.0, 1200.1, 1.0)

    data_sol_krz = np.loadtxt('%s/kurudz_1.0nm.dat' % er3t.common.fdir_data_solar)
    data_sol_16g = np.loadtxt('%s/solar_16g_1.0nm.dat' % er3t.common.fdir_data_solar)
    # data_sol_16g = np.loadtxt('%s/solar_16g_1.0nm.dat' % er3t.common.fdir_data_solar)
    data_sol_rep = np.loadtxt('%s/solar_rep_f.dat' % er3t.common.fdir_data_solar)

    sol_krz = np.interp(wvl, data_sol_krz[:, 0], data_sol_krz[:, 1])
    sol_16g = np.interp(wvl, data_sol_16g[:, 0], data_sol_16g[:, 1])
    sol_rep = np.interp(wvl, data_sol_rep[:, 0], data_sol_rep[:, 1])


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(18, 4))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl, sol_krz/sol_16g, lw=2, c='k')
        ax1.plot(wvl, sol_rep/sol_16g, lw=2, c='r')
        # ax1.plot(wvl, sol_16g, lw=2, c='r')
        # ax1.plot(wvl, (solar_rep_f-solar_crk_16g)/solar_crk_16g*100.0, lw=2, c='r')
        # ax1.plot(wvl, (solar_rep_m-solar_crk_16g)/solar_crk_16g*100.0, lw=2, c='g')
        # ax1.plot(wvl, (solar_rep_c-solar_crk_16g)/solar_crk_16g*100.0, lw=2, c='b')
        ax1.axhline(1.0, color='gray', ls=':')
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
        fname_fig = '%s_%s.png' % (_metadata_['Date'], _metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def test_01_rad_one_clear(wavelength, plot=True):

    params = {
                    'date': datetime.datetime(2019, 8, 15),
         'atmosphere_file': '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
          'surface_albedo': 0.1,
      'solar_zenith_angle': 18.7,
     'solar_azimuth_angle': -53.6,
     'sensor_zenith_angle': 51.5,
    'sensor_azimuth_angle': 81.3,
              'wavelength': wavelength,
         'output_altitude': 'toa',
         }

    data_lrt = lrt_rad_one_clear(params)
    print('Wavelength: %d nm' % wavelength)
    print('libRadtran: %.8f' % data_lrt['rad'])

    # data_mca = mca_rad_one_clear(params)

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def test_02_rad_cloud(params, overwrite=False):

    # run mcarats
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir = '%s/mca_ipa_%07.2fnm_%04.2f_%07.2f_%07.2f_%07.2f_%07.2f_cbh-%05.2fkm_cth-%05.2fkm' % ( \
             'tmp-data',
             params['wavelength'], params['surface_albedo'], \
             params['solar_zenith_angle'], params['solar_azimuth_angle'], \
             params['sensor_zenith_angle'], params['sensor_azimuth_angle'], \
             params['cloud_top_height']-params['cloud_geometrical_thickness'], params['cloud_top_height'], \
             )

    f_mca = er3t.rtm.mca.func_ref_vs_cot(
            params['cot'],
            cer0=params['cer0'],
            fdir=fdir,
            date=params['date'],
            wavelength=params['wavelength'],
            surface=params['surface_albedo'],
            solar_zenith_angle=params['solar_zenith_angle'],
            solar_azimuth_angle=params['solar_azimuth_angle'],
            sensor_zenith_angle=params['sensor_zenith_angle'],
            sensor_azimuth_angle=params['sensor_azimuth_angle'],
            sensor_altitude=params['sensor_altitude'],
            cloud_top_height=params['cloud_top_height'],
            cloud_geometrical_thickness=params['cloud_geometrical_thickness'],
            atmospheric_profile=params['atmospheric_profile'],
            Nphoton=params['Nphoton'],
            Ncpu=params['Ncpu'],
            overwrite=overwrite,
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # run libRadtran
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir = '%s/lrt_ipa_%07.2fnm_%04.2f_%07.2f_%07.2f_%07.2f_%07.2f_cbh-%05.2fkm_cth-%05.2fkm' % ( \
             'tmp-data',
             params['wavelength'], params['surface_albedo'], \
             params['solar_zenith_angle'], params['solar_azimuth_angle'], \
             params['sensor_zenith_angle'], params['sensor_azimuth_angle'], \
             params['cloud_top_height']-params['cloud_geometrical_thickness'], params['cloud_top_height'], \
             )

    f_lrt = er3t.rtm.lrt.func_ref_vs_cot(
            params['cot'],
            cer0=params['cer0'],
            fdir=fdir,
            date=params['date'],
            wavelength=params['wavelength'],
            surface_albedo=params['surface_albedo'],
            solar_zenith_angle=params['solar_zenith_angle'],
            solar_azimuth_angle=params['solar_azimuth_angle'],
            sensor_zenith_angle=params['sensor_zenith_angle'],
            sensor_azimuth_angle=params['sensor_azimuth_angle'],
            sensor_altitude=params['sensor_altitude'],
            cloud_top_height=params['cloud_top_height'],
            cloud_geometrical_thickness=params['cloud_geometrical_thickness'],
            atmospheric_profile=params['atmospheric_profile'],
            Ncpu=params['Ncpu'],
            overwrite=True,
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # ax1.scatter(f_mca.cot, f_mca.ref, s=2, c='k', lw=0.0)

        ax1.fill_between(f_mca.cot, f_mca.ref-f_mca.ref_std, f_mca.ref+f_mca.ref_std, lw=0, color='b')
        ax1.plot(f_lrt.cot, f_lrt.ref, lw=2, color='r')
        ax1.axhline(1.0, color='gray', ls=':')
        ax1.set_xlabel('COT')
        ax1.set_ylabel('Reflectance')
        patches_legend = [
                         mpatches.Patch(color='red'   , label='libRadtran'), \
                         mpatches.Patch(color='blue'  , label='MCARaTS'), \
                         ]
        ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)

        # ax1.plot(f_lrt.cot, (f_lrt.ref-f_mca.ref)/f_mca.ref*100.0, lw=2, color='r')
        # ax1.axhline(0.0, color='gray', ls=':')
        # ax1.set_ylim((-50, 50))
        # ax1.set_xlabel('COT')
        # ax1.set_ylabel('Bias [%]')

        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = 'test.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return






def lrt_flux_one(params):

    """
    libRadtran flux calculation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (er3t.common.fdir_examples, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = params['atmosphere_file']
    lrt_cfg['mol_abs_param'] = 'reptran coarse'

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = params['cloud_optical_thickness']
    cld_cfg['cloud_effective_radius']  = params['cloud_effective_radius']
    cld_cfg['cloud_altitude'] = np.arange(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height']+0.01, 0.1)

    init = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp,
            date        = params['date'],
            surface_albedo     = params['surface_albedo'],
            solar_zenith_angle = params['solar_zenith_angle'],
            wavelength         = params['wavelength'],
            output_altitude    = params['output_altitude'],
            lrt_cfg            = lrt_cfg,
            cld_cfg            = cld_cfg,
            # input_dict_extra   = {
                # 'output_process': 'integrate',
                # 'output_process': 'sum',
                # },
            # mute_list = ['slit_function_file', 'spline', 'wavelength'],
            # mute_list = ['source solar'],
            # mute_list = ['slit_function_file', 'spline'],
            )
    er3t.rtm.lrt.lrt_run(init)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx([init])

    data = {
                'f_up': np.squeeze(data0.f_up),
              'f_down': np.squeeze(data0.f_down)-np.squeeze(data0.f_up),
      'f_down_diffuse': np.squeeze(data0.f_down_diffuse),
       'f_down_direct': np.squeeze(data0.f_down_direct),
            }

    return data

def mca_flux_one(
        params,
        solver='3D',
        overwrite=True,
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (er3t.common.fdir_examples, name_tag, _metadata['Function'])
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=params['output_altitude'], fname=fname_atm, fname_atmmod=params['atmosphere_file'], overwrite=overwrite)

    fname_abs = '%s/abs_%06.1fnm.pk' % (fdir, params['wavelength'])
    abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='coarse', atm_obj=atm0, overwrite=overwrite)
    # abs0.coef['abso_coef']['data'][...] = 0.0

    fname_cld = '%s/cld.pk' % fdir
    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([params['cloud_optical_thickness']]).reshape((1, 1)),
            cer=np.array([params['cloud_effective_radius']]).reshape((1, 1)),
            cth=np.array([params['cloud_top_height']]).reshape((1, 1)),
            cgt=np.array([params['cloud_geometric_thickness']]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    pha0 = er3t.pre.pha.pha_mie_wc_shd(wavelength=params['wavelength'])
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds   = [atm1d0]

    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, overwrite=overwrite)
    atm_3ds   = [atm3d0]

    mca0 = er3t.rtm.mca.mcarats_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sca=sca,
            Ng=abs0.Ng,
            fdir='%s/%4.4d/flux_%s' % (fdir, params['wavelength'], solver.lower()),
            target='flux',
            Nrun=3,
            solar_zenith_angle=params['solar_zenith_angle'],
            photons=params['photons'],
            weights=abs0.coef['weight']['data'],
            surface=params['surface_albedo'],
            solver=solver,
            mp_mode='py',
            overwrite=overwrite
            )

    fname_h5 = '%s/mca-out-flux-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.mca.mca_out_ng(fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    data = {
      'f_up': out0.data['f_up']['data'],\
      # 'f_down': out0.data['f_down']['data'],\
      'f_down': (out0.data['f_down']['data']-out0.data['f_up']['data']),\
      'f_down_diffuse': out0.data['f_down_diffuse']['data'],\
      'f_down_direct': out0.data['f_down_direct']['data'],\
      'f_up_std': out0.data['f_up_std']['data'],\
      'f_down_std': out0.data['f_down_std']['data'],\
      'f_down_diffuse_std': out0.data['f_down_diffuse_std']['data'],\
      'f_down_direct_std': out0.data['f_down_direct_std']['data'],\
            }

    return data

def shd_flux_one(
        params,
        solver='3D',
        overwrite=True,
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (er3t.common.fdir_examples, name_tag, _metadata['Function'])
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=params['output_altitude'], fname=fname_atm, fname_atmmod=params['atmosphere_file'], overwrite=overwrite)

    fname_abs = '%s/abs_%06.1fnm.pk' % (fdir, params['wavelength'])
    abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='coarse', atm_obj=atm0, overwrite=overwrite)
    # abs0.coef['abso_coef']['data'][...] = 0.0

    fname_cld = '%s/cld.pk' % fdir
    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([params['cloud_optical_thickness']]).reshape((1, 1)),
            cer=np.array([params['cloud_effective_radius']]).reshape((1, 1)),
            cth=np.array([params['cloud_top_height']]).reshape((1, 1)),
            cgt=np.array([params['cloud_geometric_thickness']]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds   = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]

    shd0 = er3t.rtm.shd.shdom_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            fdir='%s/%4.4d/flux_%s' % (fdir, params['wavelength'], solver.lower()),
            target='flux',
            solar_zenith_angle=params['solar_zenith_angle'],
            sol_acc=1e-7,
            surface=params['surface_albedo'],
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    fname = shd0.fnames_out[0]
    out0 = er3t.rtm.shd.get_shd_data_out(fname)[0, 0, :, 0, :]

    data = {
      'f_up': out0[:, 0],\
              'f_down': (out0[:, 1]+out0[:, 2]-out0[:, 0]),\
      'f_down_diffuse': out0[:, 1],\
      'f_down_direct': out0[:, 2],\
            }

    return data




def test_100_flux_one_clear(wavelength, plot=True):

    params = {
                            'date': datetime.datetime(2024, 5, 18),
                 'atmosphere_file': '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
                  'surface_albedo': 0.03,
              'solar_zenith_angle': 0.0,
                      'wavelength': wavelength,
         'cloud_optical_thickness': 0.0,
          'cloud_effective_radius': 1.0,
                'cloud_top_height': 1.5,
       'cloud_geometric_thickness': 1.0,
                         'photons': 1.0e7,
                 'output_altitude': np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 30.1, 2.0)),
         }

    data_lrt = lrt_flux_one(params)

    data_shd = shd_flux_one(params, overwrite=True)

    data_mca = mca_flux_one(params, overwrite=False)

    # error = np.abs(data_mca['f_down']-data_lrt['f_down'])/data_lrt['f_down']*100.0
    error = np.abs(data_mca['f_down']-data_shd['f_down'])/data_shd['f_down']*100.0

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Wavelength %.1f nm [Error %.1f%%]' % (params['wavelength'], error.mean()))
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(121)
        ax1.plot(data_lrt['f_up']          , params['output_altitude'], color='red'    , lw=1.0, alpha=1.0, ls='--')
        ax1.plot(data_lrt['f_down_diffuse'], params['output_altitude'], color='magenta', lw=1.0, alpha=1.0, ls='--')
        ax1.plot(data_shd['f_up']          , params['output_altitude'], color='red'    , lw=1.0, alpha=1.0, ls='-')
        ax1.plot(data_shd['f_down_diffuse'], params['output_altitude'], color='magenta', lw=1.0, alpha=1.0, ls='-')
        ax1.plot(data_mca['f_up']          , params['output_altitude'], color='red'    , lw=2.0, alpha=0.6, ls=':')
        ax1.plot(data_mca['f_down_diffuse'], params['output_altitude'], color='magenta', lw=2.0, alpha=0.6, ls=':')
        # ax1.errorbar(data_mca['f_up']          , params['output_altitude'], xerr=data_mca['f_up_std']          , color='red'     , lw=1.0, alpha=1.0)
        # ax1.errorbar(data_mca['f_down_diffuse'], params['output_altitude'], xerr=data_mca['f_down_diffuse_std'],  color='magenta', lw=1.0, alpha=1.0)
        ax1.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax1.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_ylabel('Altitude [km]')
        # ax1.set_xlim(0.0, 0.5)
        # ax1.set_ylim((0.0, 3.0))
        ax1.set_ylim((0.0, 30.0))

        ax2 = fig.add_subplot(122)
        ax2.plot(data_lrt['f_down']       , params['output_altitude'], color='blue', lw=1.0, alpha=1.0, ls='--')
        ax2.plot(data_lrt['f_down_direct'], params['output_altitude'], color='cyan', lw=1.0, alpha=1.0, ls='--')
        ax2.plot(data_shd['f_down']       , params['output_altitude'], color='blue', lw=1.0, alpha=1.0, ls='-')
        ax2.plot(data_shd['f_down_direct'], params['output_altitude'], color='cyan', lw=1.0, alpha=1.0, ls='-')
        ax2.plot(data_mca['f_down']       , params['output_altitude'], color='blue', lw=2.0, alpha=0.6, ls=':')
        ax2.plot(data_mca['f_down_direct'], params['output_altitude'], color='cyan', lw=2.0, alpha=0.6, ls=':')
        # ax2.errorbar(data_mca['f_down']       , params['output_altitude'], xerr=data_mca['f_down_std']       , color='blue', lw=1.0, alpha=1.0)
        # ax2.errorbar(data_mca['f_down_direct'], params['output_altitude'], xerr=data_mca['f_down_direct_std'], color='cyan', lw=1.0, alpha=1.0)
        ax2.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax2.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        # ax2.set_xlim(0.0, 2.0)
        # ax2.set_ylim((0.0, 3.0))
        ax2.set_ylim((0.0, 30.0))

        ax1.axhspan(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height'], color='gray', lw=0.0, alpha=0.3)
        ax2.axhspan(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height'], color='gray', lw=0.0, alpha=0.3)
        #╰──────────────────────────────────────────────────────────────╯#

        patches_legend = [
                          mpatches.Patch(color='red'    , label='Up'), \
                          mpatches.Patch(color='magenta', label='Down-Diffuse'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        patches_legend = [
                          mpatches.Patch(color='blue', label='Down'), \
                          mpatches.Patch(color='cyan', label='Down-Direct'), \
                         ]
        ax2.legend(handles=patches_legend, loc='upper left', fontsize=12)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%06.1fnm.png' % (_metadata['Function'], params['wavelength']), bbox_inches='tight', metadata=_metadata)
        plt.show()
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def test_200_flux_one_cloud(wavelength, plot=True):

    params = {
                            'date': datetime.datetime(2024, 5, 18),
                 'atmosphere_file': '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
                  'surface_albedo': 0.03,
              'solar_zenith_angle': 0.0,
                      'wavelength': wavelength,
         'cloud_optical_thickness': 1.0,
          'cloud_effective_radius': 12.0,
                'cloud_top_height': 1.5,
       'cloud_geometric_thickness': 1.0,
                         'photons': 1.0e7,
                 'output_altitude': np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 30.1, 2.0)),
         }

    data_lrt = lrt_flux_one(params)

    data_shd = shd_flux_one(params, overwrite=True)

    data_mca = mca_flux_one(params, overwrite=False)

    # error = np.abs(data_mca['f_down']-data_lrt['f_down'])/data_lrt['f_down']*100.0
    error = np.abs(data_mca['f_down']-data_shd['f_down'])/data_shd['f_down']*100.0

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Wavelength %.1f nm [Error %.1f%%]' % (params['wavelength'], error.mean()))
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(121)
        ax1.plot(data_lrt['f_up']          , params['output_altitude'], color='red'    , lw=1.0, alpha=1.0, ls='--')
        ax1.plot(data_lrt['f_down_diffuse'], params['output_altitude'], color='magenta', lw=1.0, alpha=1.0, ls='--')
        ax1.plot(data_shd['f_up']          , params['output_altitude'], color='red'    , lw=1.0, alpha=1.0, ls='-')
        ax1.plot(data_shd['f_down_diffuse'], params['output_altitude'], color='magenta', lw=1.0, alpha=1.0, ls='-')
        ax1.plot(data_mca['f_up']          , params['output_altitude'], color='red'    , lw=2.0, alpha=0.6, ls=':')
        ax1.plot(data_mca['f_down_diffuse'], params['output_altitude'], color='magenta', lw=2.0, alpha=0.6, ls=':')
        # ax1.errorbar(data_mca['f_up']          , params['output_altitude'], xerr=data_mca['f_up_std']          , color='red'     , lw=1.0, alpha=1.0)
        # ax1.errorbar(data_mca['f_down_diffuse'], params['output_altitude'], xerr=data_mca['f_down_diffuse_std'],  color='magenta', lw=1.0, alpha=1.0)
        ax1.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax1.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_ylabel('Altitude [km]')
        # ax1.set_xlim(0.0, 0.5)
        ax1.set_ylim((0.0, 3.0))

        ax2 = fig.add_subplot(122)
        ax2.plot(data_lrt['f_down']       , params['output_altitude'], color='blue', lw=1.0, alpha=1.0, ls='--')
        ax2.plot(data_lrt['f_down_direct'], params['output_altitude'], color='cyan', lw=1.0, alpha=1.0, ls='--')
        ax2.plot(data_shd['f_down']       , params['output_altitude'], color='blue', lw=1.0, alpha=1.0, ls='-')
        ax2.plot(data_shd['f_down_direct'], params['output_altitude'], color='cyan', lw=1.0, alpha=1.0, ls='-')
        ax2.plot(data_mca['f_down']       , params['output_altitude'], color='blue', lw=2.0, alpha=0.6, ls=':')
        ax2.plot(data_mca['f_down_direct'], params['output_altitude'], color='cyan', lw=2.0, alpha=0.6, ls=':')
        # ax2.errorbar(data_mca['f_down']       , params['output_altitude'], xerr=data_mca['f_down_std']       , color='blue', lw=1.0, alpha=1.0)
        # ax2.errorbar(data_mca['f_down_direct'], params['output_altitude'], xerr=data_mca['f_down_direct_std'], color='cyan', lw=1.0, alpha=1.0)
        ax2.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax2.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        # ax2.set_xlim(0.0, 2.0)
        ax2.set_ylim((0.0, 3.0))

        ax1.axhspan(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height'], color='gray', lw=0.0, alpha=0.3)
        ax2.axhspan(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height'], color='gray', lw=0.0, alpha=0.3)
        #╰──────────────────────────────────────────────────────────────╯#

        patches_legend = [
                          mpatches.Patch(color='red'    , label='Up'), \
                          mpatches.Patch(color='magenta', label='Down-Diffuse'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        patches_legend = [
                          mpatches.Patch(color='blue', label='Net (Down-Up)'), \
                          mpatches.Patch(color='cyan', label='Down-Direct'), \
                         ]
        ax2.legend(handles=patches_legend, loc='upper left', fontsize=12)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%06.1fnm.png' % (_metadata['Function'], params['wavelength']), bbox_inches='tight', metadata=_metadata)
        plt.show()
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



if __name__ == '__main__':

    warnings.warn('\nUnder active development ...')

    if er3t.common.has_mcarats & er3t.common.has_libradtran:

        # test_00_solar_old()
        # test_00_solar()
        # for wavelength in [470.0, 555.0, 659.0, 772.0, 1621.0, 2079.0]:
        # for wavelength in [772.0, 1621.0, 2079.0]:
        #     test_01_rad_one_clear(wavelength, plot=True)
        # for wavelength in [650.0]:
        #     test_01_flux_one_clear(wavelength)
        # test_02_rad_cloud(params, overwrite=False)

        # test_100_flux_one_clear(650.0, plot=True)
        test_200_flux_one_cloud(650.0, plot=True)

    else:

        msg = '\nError [00_er3t_bmk.py]: Needs to have both <MCARaTS> and <libRadtran> to be installed for performing benchmark tests.'
        raise OSError(msg)
