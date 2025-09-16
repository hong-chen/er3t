import datetime
import os
import sys
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import cartopy.crs as ccrs
# mpl.use('Agg')

import er3t



def anim_phase_mie(index, cer0=1.0, Nstart=5, Nend=751):

    angles = np.linspace(0.0, 180.0, 1800)
    pha_r_ = er3t.pre.pha.pha_mie_wc_shd(wavelength=650.0, Npmom_max=1000, angles=angles, overwrite=True)
    pha_g_ = er3t.pre.pha.pha_mie_wc_shd(wavelength=550.0, Npmom_max=1000, angles=angles, overwrite=True)
    pha_b_ = er3t.pre.pha.pha_mie_wc_shd(wavelength=450.0, Npmom_max=1000, angles=angles, overwrite=True)
    index_cer = np.argmin(np.abs(pha_r_.data['ref']['data']-cer0))

    for Npmom_max in np.arange(Nstart, Nend, 5):

        pmom_r = pha_r_.data['pmom']['data'][index_cer, :Npmom_max]
        logic_r = ~np.isnan(pmom_r)
        pha_r = er3t.pre.pha.legendre2phase(pmom_r[logic_r], angle=angles, lrt=False, normalize=True, deltascaling=False)

        pmom_g = pha_g_.data['pmom']['data'][index_cer, :Npmom_max]
        logic_g = ~np.isnan(pmom_g)
        pha_g = er3t.pre.pha.legendre2phase(pmom_g[logic_g], angle=angles, lrt=False, normalize=True, deltascaling=False)

        pmom_b = pha_b_.data['pmom']['data'][index_cer, :Npmom_max]
        logic_b = ~np.isnan(pmom_b)
        pha_b = er3t.pre.pha.legendre2phase(pmom_b[logic_b], angle=angles, lrt=False, normalize=True, deltascaling=False)

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(8, 4))
            # plot1
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(111)
            ax1.plot(pha_r_.data['ang']['data'], pha_r, color='r', lw=3.0)
            ax1.plot(pha_g_.data['ang']['data'], pha_g, color='g', lw=1.8)
            ax1.plot(pha_b_.data['ang']['data'], pha_b, color='b', lw=0.9)

            ax1.plot(pha_r_.data['ang']['data'], pha_r_.data['pha']['data'][:, index_cer], color='r', lw=3.0, alpha=0.2, ls='-')
            ax1.plot(pha_g_.data['ang']['data'], pha_g_.data['pha']['data'][:, index_cer], color='g', lw=1.8, alpha=0.2, ls='-')
            ax1.plot(pha_b_.data['ang']['data'], pha_b_.data['pha']['data'][:, index_cer], color='b', lw=0.9, alpha=0.2, ls='-')

            ax1.set_xlim((-10, 190))
            ax1.set_ylim((1.0e-2, 1.0e5))
            ax1.set_yscale('log')

            ax1.xaxis.set_major_locator(FixedLocator(np.arange(0.0, 180.1, 30.0)))
            ax1.set_xlabel('Angle [$^\\circ$]')
            ax1.set_ylabel('Phase Function')
            ax1.set_title('Using %d Legendre Coefficients (CER=%d$\\mu m$) ...' % (Npmom_max, cer0))
            #╰──────────────────────────────────────────────────────────────╯#

            # polar plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax_polar = ax1.inset_axes([0.25, 0.43, 0.5, 0.5], polar=True)
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)

            ang_full = np.append(pha_r_.data['ang']['data'], 180.0+pha_r_.data['ang']['data'][1:]) % 360.0
            pha_r_full = np.append(pha_r, pha_r[:-1][::-1])
            pha_g_full = np.append(pha_g, pha_g[:-1][::-1])
            pha_b_full = np.append(pha_b, pha_b[:-1][::-1])

            ax_polar.plot(np.deg2rad(ang_full), pha_r_full, color='r', lw=0.5)
            ax_polar.plot(np.deg2rad(ang_full), pha_g_full, color='g', lw=0.5)
            ax_polar.plot(np.deg2rad(ang_full), pha_b_full, color='b', lw=0.5)
            ax_polar.set_rscale('log')
            ax_polar.set_rlim((1.0e-2, 1.0e5))
            ax_polar.tick_params(axis='x', labelsize=8, pad=-2)
            ax_polar.tick_params(axis='y', labelsize=8)
            ax_polar.set_yticks([])
            #╰──────────────────────────────────────────────────────────────╯#

            patches_legend = [
                             mpatches.Patch(color='red'   , label='650 nm'), \
                             mpatches.Patch(color='green' , label='550 nm'), \
                             mpatches.Patch(color='blue'  , label='450 nm'), \
                             ]
            ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.35, wspace=0.35)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
            fname_fig = '%4.4d_%3.3d_%s_cer-%2.2d.png' % (index, Npmom_max, 'phase', cer0)
            plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
            #╰──────────────────────────────────────────────────────────────╯#
            # plt.show()
            # sys.exit()
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#

def main_phase_mie():

    # anim_phase_mie(index, cer0=12.0, Nstart=5, Nend=751)

    index = 0
    for cer0 in np.arange(1.0, 25.1, 1.0):
        anim_phase_mie(index, cer0=cer0, Nstart=1000, Nend=1001)
        index += 1
    for cer0 in np.arange(24.0, 0.9, -1.0):
        anim_phase_mie(index, cer0=cer0, Nstart=1000, Nend=1001)
        index += 1


def lrt_flux_spec(
        params,
        surface='lambertian',
        overwrite=False,
        ):

    """
    libRadtran flux calculation
    """

    name_tag = 'spec_abs'

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s/cot-%04.1f_cer-%04.1f' % (er3t.common.fdir_examples, name_tag, _metadata['Function'], params['cloud_optical_thickness'], params['cloud_effective_radius'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = params['atmosphere_file']
    lrt_cfg['mol_abs_param'] = 'reptran fine'
    lrt_cfg['number_of_streams'] = 32

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = f"{fdir_tmp}/cloud.txt"
    cld_cfg['cloud_optical_thickness'] = params['cloud_optical_thickness']
    cld_cfg['cloud_effective_radius']  = params['cloud_effective_radius']
    cld_cfg['cloud_altitude'] = np.arange(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height']+0.01, 0.1)

    inits_flux = []

    for wvl0 in params['wavelengths']:

        fname_out = '%s/output_flux_%4.4d.txt' % (fdir_tmp, wvl0)
        if (not overwrite) and (not os.path.exists(fname_out)):
            overwrite = True

        mute_list = ['source solar', 'slit_function_file', 'wavelength', 'spline', 'albedo']

        input_dict_extra = {
                'wavelength_add': '%.1f %.1f' % (wvl0, wvl0),
                }

        if params['extra'] is not None:
            for key in params['extra'].keys():
                input_dict_extra[key] = params['extra'][key]

        if surface == 'land':
            input_dict_extra['brdf_ambrals iso'] = params['f_iso']
            input_dict_extra['brdf_ambrals vol'] = params['f_vol']
            input_dict_extra['brdf_ambrals geo'] = params['f_geo']
        elif surface == 'ocean':
            input_dict_extra['brdf_cam'] = 'u10 %.2f' % params['windspeed']
        else:
            mute_list.pop()

        init_flux = er3t.rtm.lrt.lrt_init_mono_flx(
                input_file  = '%s/input_flux_%4.4d.txt' % (fdir_tmp, wvl0),
                output_file = fname_out,
                date        = params['date'],
                surface_albedo = params['surface_albedo'],
                solar_zenith_angle   = params['solar_zenith_angle'],
                wavelength         = wvl0,
                output_altitude    = params['output_altitude'],
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg,
                mute_list          = mute_list,
                input_dict_extra   = input_dict_extra,
                )
        inits_flux.append(init_flux)

    if overwrite:
        er3t.rtm.lrt.lrt_run_mp(inits_flux)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx(inits_flux)

    data = {
              'f_down': np.squeeze(data0.f_down),
              'f_down_diffuse': np.squeeze(data0.f_down_diffuse),
            }

    return data

def test_100_flux_spec(
        wavelengths,
        cot,
        cer,
        icount,
        surface='lambertian',
        plot=True,
        overwrite=True,
        ):

    params = {
                            'date': datetime.datetime(2024, 5, 18),
                 'atmosphere_file': '%s/afglss.dat' % er3t.common.fdir_data_atmmod,
                  'surface_albedo': 0.8,
              'solar_zenith_angle': 60.0,
             'solar_azimuth_angle': 0.0,
                 'sensor_altitude': 120.0,
             'sensor_zenith_angle': 0.0,
            'sensor_azimuth_angle': np.array([0.0]),
                     'wavelengths': wavelengths,
         'cloud_optical_thickness': cot,
          'cloud_effective_radius': cer,
                'cloud_top_height': 1.5,
       'cloud_geometric_thickness': 1.0,
                         'photons': 1.0e6,
                           'f_iso': 0.12472048343113448,
                           'f_vol': 0.05460690884637945,
                           'f_geo': 0.03384929843579787,
                       'windspeed': 1.0,
                         'pigment': 0.01,
                           'extra': None,
                 # 'output_altitude': np.concatenate((np.arange(0.0, 8.0, 0.1), np.arange(8.0, 16.0, 0.2), np.arange(16.0, 30.0, 0.5), np.arange(30.0, 60.1, 1.0))),
                 'output_altitude': np.concatenate((np.arange(0.0, 1.0, 0.01), np.arange(1.0, 10.0, 0.1), np.arange(10.0, 60.1, 1.0))),
         }

    if params['cloud_optical_thickness'] > 0.0:
        params['photons'] = 1.0e9

    # gases = ['O3', 'O2', 'H2O', 'CO2', 'NO2', 'BRO', 'OCLO', 'HCHO', 'O4', 'SO2', 'CH4', 'N2O', 'CO', 'N2']
    gases = ['O3', 'O2', 'O4', 'H2O', 'CO2', 'NO2', 'CH4', 'N2O', 'CO', 'N2']
    for gas in ['all']+gases:

        if gas != 'all':
            extra = {}
            for gas0 in gases:
                if (gas0 != gas):
                    if ((gas=='O4') and (gas0 != 'O2')):
                            extra[f"mol_modify {gas0}"] = "0.0 DU"
                    else:
                        extra[f"mol_modify {gas0}"] = "0.0 DU"
            params['extra'] = extra
            print(gas)
            print(extra)
            print()

        data_lrt = lrt_flux_spec(params, surface=surface, overwrite=overwrite)

        # save h5 file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname = f"data/data_flux_{gas}.h5"
        h5f = h5py.File(fname, 'w')
        h5d_f_down = h5f.create_dataset('f_down', data=data_lrt['f_down'], compression='gzip', compression_opts=9, chunks=True)
        h5d_f_down = h5f.create_dataset('f_down_diffuse', data=data_lrt['f_down_diffuse'], compression='gzip', compression_opts=9, chunks=True)
        h5d_wvl = h5f.create_dataset('wvl', data=wavelengths, compression='gzip', compression_opts=9, chunks=True)
        h5d_alt = h5f.create_dataset('alt', data=params['output_altitude'], compression='gzip', compression_opts=9, chunks=True)
        h5f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

def anim_gas_absorption(index):

    # gases = ['O3', 'O2', 'H2O', 'CO2', 'NO2', 'BRO', 'OCLO', 'HCHO', 'O4', 'SO2', 'CH4', 'N2O', 'CO', 'N2']

    gases_list = ['N2', 'O2', 'O3', 'O4', 'H2O', 'CO2', 'CO', 'NO2', 'N2O', 'CH4']

    gases = {
            'N2': {'name':'N$_{2}$'},
            'O2': {'name':'O$_{2}$'},
            'O3': {'name':'O$_{3}$'},
            'O4': {'name':'O$_{4}$'},
            'H2O': {'name':'H$_{2}$O'},
            'CO2': {'name':'CO$_{2}$'},
            'CO': {'name':'CO'},
            'NO2': {'name':'NO$_{2}$'},
            'N2O': {'name':'N$_{2}$O'},
            'CH4': {'name':'CH$_{4}$'},
            }
    cmap = mpl.colormaps['jet']
    colors = cmap(np.linspace(0.0, 1.0, len(gases)+2))
    colors = colors[1:-1, :]

    data = er3t.util.load_h5("data/data_flux_all.h5")

    atm0 = er3t.pre.atm.atm_atmmod(levels=data['alt'], fname=None, fname_atmmod=f"{er3t.common.fdir_data_atmmod}/afglss.dat", overwrite=True)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle(f"At {data['alt'][index]:.1f} km", fontsize=24)
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax2 = ax1.inset_axes([0.60, 0.15, 0.2, 0.83])

        patches_legend = []
        for i, gas in enumerate(gases_list):
            data_gas = er3t.util.load_h5(f"data/data_flux_{gas}.h5")
            if gas == 'O2':
                data_gas0 = er3t.util.load_h5("data/data_flux_O4.h5")
                ax1.fill_between(data['wvl'], data_gas0['f_down'][index, :], data_gas0['f_down'][-1, :], facecolor=colors[i], lw=0.0, alpha=1.0)
            else:
                ax1.fill_between(data['wvl'], data_gas['f_down'][index, :], data_gas['f_down'][-1, :], facecolor=colors[i], lw=0.0, alpha=1.0)

            if gas.lower() in atm0.lev.keys():
                ax2.plot(atm0.lev[gas.lower()]['data'], atm0.lev['altitude']['data'], lw=1.0, color=colors[i])
                gas_concentration = atm0.lev[gas.lower()]['data'][index:].sum()
            else:
                if gas == 'O4':
                    gas_concentration = atm0.lev['o2']['data'][index:].sum()**2
            patches_legend.append(mpatches.Patch(color=colors[i], label=f"{gases[gas]['name']} [{gas_concentration:.1E}]"))

        gas_concentration = atm0.lev['air']['data'][index:].sum()
        patches_legend.append(mpatches.Patch(color='gray', label=f"Air [{gas_concentration:.1E}]"))

        ax1.plot(data['wvl'], data['f_down'][index, :], lw=0.5, color='k')
        ax1.fill_between(data['wvl'], 0.0, data['f_down_diffuse'][index, :], lw=0.0, facecolor='gray', alpha=1.0)

        ax1.set_xlim((300, 3200.000001))
        ax1.set_ylim((0, 1.20000001))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 3201, 400)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 1.21, 0.2)))

        ax2.axhline(data['alt'][index], color='black', lw=0.5)
        ax2.set_ylim((0.1, 60.0))
        # ax2.set_xlim((0.1, 1000.0))
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_xlabel('[cm$^{-3}$]', fontsize=12)
        ax2.set_ylabel(f"{data['alt'][index]:.1f} [km]", fontsize=12)
        #╰──────────────────────────────────────────────────────────────╯#

        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f'{240-index:04d}_{_metadata_['Function']}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_gas_absorption():

    # wavelengths = np.arange(300.0, 3201.0, 5.0)
    # test_100_flux_spec(wavelengths, 0.0, 1.0, 100)

    for index in np.arange(241)[::-1]:
    # for index in np.arange(0, 1)[::-1]:
        anim_gas_absorption(index)

if __name__ == '__main__':

    # main_phase_mie()

    main_gas_absorption()

    pass
