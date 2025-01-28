import os
import sys
import glob
import datetime
import warnings
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



def test_00_util():

    command = 'lss %s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    os.system(command)

    command = 'lss %s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    os.system(command)





def lrt_flux_one_clear(params):

    """
    libRadtran flux calculation
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

    init = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp,
            date        = params['date'],
            surface_albedo     = params['surface_albedo'],
            solar_zenith_angle = params['solar_zenith_angle'],
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

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx([init])

    data = {
                'f_up': np.squeeze(data0.f_up),
              'f_down': np.squeeze(data0.f_down),
      'f_down_diffuse': np.squeeze(data0.f_down_diffuse),
       'f_down_direct': np.squeeze(data0.f_down_direct),
            }

    return data

def mca_flux_one_clear(
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
    abs0      = er3t.pre.abs.abs_16g(wavelength=params['wavelength'], fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)
    # abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='modis', band_name='modis_aqua_b01', atm_obj=atm0, overwrite=overwrite)

    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds   = [atm1d0]

    mca0 = er3t.rtm.mca.mcarats_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            Ng=abs0.Ng,
            fdir='%s/%4.4d/flux_%s' % (fdir, params['wavelength'], solver.lower()),
            target='flux',
            Nrun=3,
            solar_zenith_angle=params['solar_zenith_angle'],
            photons=1e7,
            weights=abs0.coef['weight']['data'],
            surface_albedo=params['surface_albedo'],
            solver=solver,
            mp_mode='py',
            overwrite=overwrite
            )

    fname_h5 = '%s/mca-out-flux-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.mca.mca_out_ng(fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    data = {
      'f_up': out0.data['f_up']['data'],\
      'f_down': out0.data['f_down']['data'],\
      'f_down_diffuse': out0.data['f_down_diffuse']['data'],\
      'f_down_direct': out0.data['f_down_direct']['data'],\
      'f_up_std': out0.data['f_up_std']['data'],\
      'f_down_std': out0.data['f_down_std']['data'],\
      'f_down_diffuse_std': out0.data['f_down_diffuse_std']['data'],\
      'f_down_direct_std': out0.data['f_down_direct_std']['data'],\
            }

    return data

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






if __name__ == '__main__':

    warnings.warn('Under active development ...')

    if er3t.common.has_mcarats & er3t.common.has_libradtran:

        # check_solar()
        # for wavelength in [470.0, 555.0, 659.0, 772.0, 1621.0, 2079.0]:
        for wavelength in [772.0, 1621.0, 2079.0]:
            test_01_rad_one_clear(wavelength, plot=True)
        # for wavelength in [650.0]:
        #     test_01_flux_one_clear(wavelength)

    else:

        msg = '\nError [00_er3t_bmk.py]: Needs to have both <MCARaTS> and <libRadtran> to be installed for performing benchmark tests.'
        raise OSError(msg)
