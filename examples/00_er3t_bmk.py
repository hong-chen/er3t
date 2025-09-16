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



def test_00_solar():

    wvl = np.arange(300.0, 1200.1, 1.0)

    data_sol_krz = np.loadtxt('%s/kurudz_1.0nm.dat' % er3t.common.fdir_data_solar)
    data_sol_16g = np.loadtxt('%s/solar_16g_1.0nm.dat' % er3t.common.fdir_data_solar)
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



def lrt_flux_one(
        params,
        overwrite=False,
        ):

    """
    libRadtran flux calculation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}/{params['wavelength']:04.0f}"
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

    fname_out = '%s/output.txt' % fdir_tmp
    if (not overwrite) and (not os.path.exists(fname_out)):
        overwrite = True

    mute_list = ['source solar', 'slit_function_file', 'wavelength', 'spline']
    input_dict_extra = {
            'wavelength_add': f"{params['wavelength']:.1f} {params['wavelength']:.1f}"
            }
    init = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = fname_out,
            date        = params['date'],
            surface_albedo     = params['surface_albedo'],
            solar_zenith_angle = params['solar_zenith_angle'],
            wavelength         = params['wavelength'],
            output_altitude    = params['output_altitude'],
            lrt_cfg            = lrt_cfg,
            cld_cfg            = cld_cfg,
            mute_list          = mute_list,
            input_dict_extra   = input_dict_extra,
            )

    if overwrite:
        er3t.rtm.lrt.lrt_run(init)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx([init])

    data = {
                'f_up': np.squeeze(data0.f_up),
              'f_down': np.squeeze(data0.f_down),
              'f_net' : np.squeeze(data0.f_down)-np.squeeze(data0.f_up),
      'f_down_diffuse': np.squeeze(data0.f_down_diffuse),
       'f_down_direct': np.squeeze(data0.f_down_direct),
            }

    return data

def mca_flux_one(
        params,
        f_toa=None,
        solver='IPA',
        overwrite=False,
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}/{params['wavelength']:04.0f}"
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    fname_atm = f"{fdir_tmp}/atm.pk"
    atm0      = er3t.pre.atm.atm_atmmod(levels=params['output_altitude'], fname=fname_atm, fname_atmmod=params['atmosphere_file'], overwrite=overwrite)

    fname_abs = f"{fdir_tmp}/abs.pk"
    abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)
    # rescale TOA flux
    if f_toa is not None:
        f_toa_ = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()
        abs0.coef['solar']['data'] = f_toa/f_toa_ * abs0.coef['solar']['data']

    fname_cld = f"{fdir_tmp}/cld.pk"
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
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname=f"{fdir_tmp}/mca_sca.bin", overwrite=overwrite)

    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds   = [atm1d0]

    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname=f"{fdir_tmp}/mca_atm_3d.bin", overwrite=overwrite)
    atm_3ds   = [atm3d0]

    fdir = f"{fdir_tmp}/flux_{solver.lower()}"
    if (not overwrite) and (not os.path.exists(fdir)):
        overwrite = True

    mca0 = er3t.rtm.mca.mcarats_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sca=sca,
            Ng=abs0.Ng,
            fdir=fdir,
            target='flux',
            Nrun=3,
            solar_zenith_angle=params['solar_zenith_angle'],
            photons=params['photons'],
            weights=abs0.coef['weight']['data'],
            surface=params['surface_albedo'],
            solver=solver,
            mp_mode='mpi',
            overwrite=overwrite
            )

    fname_h5 = f"{fdir_tmp}/mca-out-flux-{solver.lower()}_{_metadata['Function']}.h5"
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
      'f_net': (out0.data['f_down']['data']-out0.data['f_up']['data']),\
      'f_net_std': np.sqrt(out0.data['f_down_std']['data']**2 + out0.data['f_up_std']['data']**2),\
            }

    return data

def shd_flux_one(
        params,
        solver='IPA',
        f_toa=None,
        overwrite=False,
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}/{params['wavelength']:04.0f}"
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    fname_atm = f"{fdir_tmp}/atm.pk"
    atm0      = er3t.pre.atm.atm_atmmod(levels=params['output_altitude'], fname=fname_atm, fname_atmmod=params['atmosphere_file'], overwrite=overwrite)

    fname_abs = f"{fdir_tmp}/abs.pk"
    abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)
    if f_toa is not None:
        f_toa_ = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()
        abs0.coef['solar']['data'] = f_toa/f_toa_ * abs0.coef['solar']['data']

    fname_cld = f"{fdir_tmp}/cld.pk"
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

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname=f"{fdir_tmp}/shdom-ckd.txt", overwrite=overwrite)
    atm_1ds   = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname=f"{fdir_tmp}/shdom-prp.txt", fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]

    fdir = f"{fdir_tmp}/flux_{solver.lower()}"
    if (not overwrite) and (not os.path.exists(fdir)):
        overwrite = True

    shd0 = er3t.rtm.shd.shdom_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            fdir=fdir,
            target='flux',
            Niter=1000,
            Nmu=16,
            Nphi=32,
            solar_zenith_angle=params['solar_zenith_angle'],
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            sol_acc=1e-6,
            split_acc=1e-6,
            surface=params['surface_albedo'],
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    fname = shd0.fnames_out[0]
    out0_ = er3t.rtm.shd.get_shd_data_out(fname)[:, :, :, 0, :]
    Nx, Ny, Nz, Nv = out0_.shape
    out0 = np.zeros((Nz, Nv), dtype=np.float32)
    for iz in np.arange(Nz):
        for iv in np.arange(Nv):
            out0[iz, iv] = np.mean(out0_[:, :, iz, iv])

    data = {
      'f_up': out0[:, 0],\
      'f_down': (out0[:, 1]+out0[:, 2]),\
      'f_net': (out0[:, 1]+out0[:, 2]-out0[:, 0]),\
      'f_down_diffuse': out0[:, 1],\
      'f_down_direct': out0[:, 2],\
            }

    return data

def test_100_flux_one(
        wavelength,
        cot,
        cer,
        icount,
        plot=True,
        overwrite=False,
        ):

    params = {
                            'date': datetime.datetime(2024, 5, 18),
                 'atmosphere_file': '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
                  'surface_albedo': 0.05,
              'solar_zenith_angle': 30.0,
                      'wavelength': wavelength,
         'cloud_optical_thickness': cot,
          'cloud_effective_radius': cer,
                           'Niter': icount,
                'cloud_top_height': 1.5,
       'cloud_geometric_thickness': 1.0,
                         'photons': 1.0e7,
                 'output_altitude': np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0)),
         }

    data_lrt = lrt_flux_one(params, overwrite=overwrite)
    f_toa = data_lrt['f_down'][-1]/np.cos(np.deg2rad(params['solar_zenith_angle']))/er3t.util.cal_sol_fac(params['date'])

    data_shd = shd_flux_one(params, f_toa=f_toa, overwrite=overwrite)

    data_mca = mca_flux_one(params, f_toa=f_toa, overwrite=overwrite)

    error_shd_up = np.nanmean(np.abs(data_lrt['f_up']-data_shd['f_up'])/data_lrt['f_up']*100.0)
    error_mca_up = np.nanmean(np.abs(data_lrt['f_up']-data_mca['f_up'])/data_lrt['f_up']*100.0)
    error_shd_net = np.nanmean(np.abs(data_lrt['f_net']-data_shd['f_net'])/data_lrt['f_net']*100.0)
    error_mca_net = np.nanmean(np.abs(data_lrt['f_net']-data_mca['f_net'])/data_lrt['f_net']*100.0)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('COT=%.1f, CER=%.1f $\\mu m$' % (params['cloud_optical_thickness'], params['cloud_effective_radius']))
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(121)
        ax1.plot(data_lrt['f_up']          , params['output_altitude'], color='black', lw=1.0, alpha=1.0, ls='-', zorder=0)
        ax1.plot(data_lrt['f_down_diffuse'], params['output_altitude'], color='black', lw=1.0, alpha=1.0, ls='-', zorder=0)
        ax1.fill_betweenx(params['output_altitude'], data_mca['f_up']-data_mca['f_up_std']                    , data_mca['f_up']+data_mca['f_up_std']                    , color='blue', lw=0.2, alpha=1.0, zorder=1)
        ax1.fill_betweenx(params['output_altitude'], data_mca['f_down_diffuse']-data_mca['f_down_diffuse_std'], data_mca['f_down_diffuse']+data_mca['f_down_diffuse_std'], color='blue', lw=0.2, alpha=1.0, zorder=1)
        ax1.plot(data_shd['f_up']          , params['output_altitude'], color='red', lw=0.5, alpha=1.0, ls='-', zorder=2)
        ax1.plot(data_shd['f_down_diffuse'], params['output_altitude'], color='red', lw=0.5, alpha=1.0, ls='-', zorder=2)
        ax1.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax1.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_ylabel('Altitude [km]')
        ax1.set_title('Down Diffuse & Upwelling')
        ax1.set_yscale('log')
        ax1.set_ylim((0.1, 40.0))
        ax1.set_xlim((0.0, 0.1*(f_toa//0.1 + 1)))

        ax2 = fig.add_subplot(122)
        ax2.plot(data_lrt['f_net']        , params['output_altitude'], color='black', lw=1.0, alpha=1.0, ls='-', zorder=0)
        ax2.plot(data_lrt['f_down_direct'], params['output_altitude'], color='black', lw=1.0, alpha=1.0, ls='-', zorder=0)
        ax2.fill_betweenx(params['output_altitude'], data_mca['f_net']-data_mca['f_net_std']                , data_mca['f_net']+data_mca['f_net_std']                , color='blue', lw=0.2, alpha=1.0, zorder=1)
        ax2.fill_betweenx(params['output_altitude'], data_mca['f_down_direct']-data_mca['f_down_direct_std'], data_mca['f_down_direct']+data_mca['f_down_direct_std'], color='blue', lw=0.2, alpha=1.0, zorder=1)
        ax2.plot(data_shd['f_net']        , params['output_altitude'], color='red', lw=0.5, alpha=1.0, ls='-', zorder=2)
        ax2.plot(data_shd['f_down_direct'], params['output_altitude'], color='red', lw=0.5, alpha=1.0, ls='-', zorder=2)
        ax2.set_ylim((params['output_altitude'][0], params['output_altitude'][-1]))
        ax2.set_xlabel('Flux Density [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax2.set_title('Down Direct & Net')
        ax2.set_yscale('log')
        ax2.set_ylim((0.1, 40.0))
        ax2.set_xlim((0.0, 0.1*(f_toa//0.1 + 1)))

        if params['cloud_optical_thickness'] > 0.0:
            ax1.axhspan(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height'], color='gray', lw=0.0, alpha=0.2, zorder=0)
            ax2.axhspan(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height'], color='gray', lw=0.0, alpha=0.2, zorder=0)
        #╰──────────────────────────────────────────────────────────────╯#

        patches_legend = [
                          mpatches.Patch(color='black', label='libRadtran'), \
                          mpatches.Patch(color='blue' , label='MCARaTS (%.1f%%)' % error_mca_up), \
                          mpatches.Patch(color='red'  , label='SHDOM (%.1f%%)' % error_shd_up), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper center', fontsize=12)

        patches_legend = [
                          mpatches.Patch(color='black', label='libRadtran'), \
                          mpatches.Patch(color='blue' , label='MCARaTS (%.1f%%)' % error_mca_net), \
                          mpatches.Patch(color='red'  , label='SHDOM (%.1f%%)' % error_shd_net), \
                         ]
        ax2.legend(handles=patches_legend, loc='upper center', fontsize=12)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%4.4d_%s_%06.1fnm_cot-%05.1f_cer-%05.1f.png' % (icount, _metadata['Function'], params['wavelength'], params['cloud_optical_thickness'], params['cloud_effective_radius']), bbox_inches='tight', metadata=_metadata)
        plt.show()
        # plt.close()
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def lrt_rad_one(
        params,
        surface='ocean',
        overwrite=False,
        ):

    """
    libRadtran radiance calculation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}/{params['wavelength']:04.0f}"
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = params['atmosphere_file']
    # lrt_cfg['rte_solver']      = 'fdisort1'
    lrt_cfg['rte_solver']      = 'disort'
    lrt_cfg['mol_abs_param'] = 'reptran fine'
    lrt_cfg['number_of_streams'] = 32

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = f"{fdir_tmp}/cloud.txt"
    cld_cfg['cloud_optical_thickness'] = params['cloud_optical_thickness']
    cld_cfg['cloud_effective_radius']  = params['cloud_effective_radius']
    cld_cfg['cloud_altitude'] = np.arange(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height']+0.01, 0.1)

    fname_out = f"{fdir_tmp}/output_rad.txt"
    if (not overwrite) and (not os.path.exists(fname_out)):
        overwrite = True

    mute_list = ['source solar', 'slit_function_file', 'wavelength', 'spline', 'albedo']

    input_dict_extra = {
            'wavelength_add': f"{params['wavelength']:.1f} {params['wavelength']:.1f}"
            }
    if surface == 'land':
        input_dict_extra['brdf_ambrals iso'] = params['f_iso']
        input_dict_extra['brdf_ambrals vol'] = params['f_vol']
        input_dict_extra['brdf_ambrals geo'] = params['f_geo']
    elif surface == 'ocean':
        input_dict_extra['brdf_cam'] = f"u10 {params['windspeed']:.8f}"
    else:
        mute_list.pop()

    inits = []
    init_rad = er3t.rtm.lrt.lrt_init_mono_rad(
            input_file  = f"{fdir_tmp}/input_rad.txt",
            output_file = fname_out,
            date        = params['date'],
            surface_albedo = params['surface_albedo'],
            solar_zenith_angle   = params['solar_zenith_angle'],
            solar_azimuth_angle  = 0.0,
            sensor_zenith_angle  = params['sensor_zenith_angle'],
            sensor_azimuth_angle = params['sensor_azimuth_angle'],
            wavelength         = params['wavelength'],
            output_altitude    = params['output_altitude'][-1],
            lrt_cfg            = lrt_cfg,
            cld_cfg            = cld_cfg,
            mute_list          = mute_list,
            input_dict_extra   = input_dict_extra,
            )
    inits.append(init_rad)

    init_flx = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file  = f"{fdir_tmp}/input_flx.txt",
            output_file = f"{fdir_tmp}/output_flx.txt",
            date        = params['date'],
            surface_albedo = params['surface_albedo'],
            solar_zenith_angle = params['solar_zenith_angle'],
            wavelength         = params['wavelength'],
            output_altitude    = params['output_altitude'][-1],
            lrt_cfg            = lrt_cfg,
            cld_cfg            = None,
            mute_list          = mute_list,
            input_dict_extra   = input_dict_extra,
            )
    inits.append(init_flx)

    if overwrite:
        er3t.rtm.lrt.lrt_run_mp(inits)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx([init_flx])
    data1 = er3t.rtm.lrt.lrt_read_uvspec_rad([init_rad])

    data = {
              'f_down': np.squeeze(data0.f_down),
              'rad': np.squeeze(data1.rad),
            }

    return data

def mca_rad_one(
        params,
        f_toa=None,
        solver='3D',
        surface='ocean',
        overwrite=False,
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}/{params['wavelength']:04.0f}"
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    fname_atm = f"{fdir_tmp}/atm.pk"
    atm0      = er3t.pre.atm.atm_atmmod(levels=params['output_altitude'], fname=fname_atm, fname_atmmod=params['atmosphere_file'], overwrite=overwrite)

    fname_abs = f"{fdir_tmp}/abs.pk"
    abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)
    # rescale TOA flux
    if f_toa is not None:
        f_toa_ = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()
        abs0.coef['solar']['data'] = f_toa/f_toa_ * abs0.coef['solar']['data']

    fname_cld = f"{fdir_tmp}/cld.pk"
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

    pha0 = er3t.pre.pha.pha_mie_wc_shd(wavelength=params['wavelength'], overwrite=overwrite)
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname=f"{fdir_tmp}/mca_sca.bin", overwrite=overwrite)

    if surface == 'ocean':
        sfc_dict = er3t.pre.sfc.cal_ocean_brdf(
                wvl=params['wavelength'],
                u10=np.array([params['windspeed']]).reshape((1, 1)),
                pcl=np.array([params['pigment']]).reshape((1, 1)),
                whitecaps=True,
                )
    elif surface == 'land':
        sfc_dict = {}
        sfc_dict['fiso'] = np.array([params['f_iso']]).reshape((1, 1))
        sfc_dict['fvol'] = np.array([params['f_vol']]).reshape((1, 1))
        sfc_dict['fgeo'] = np.array([params['f_geo']]).reshape((1, 1))
    elif surface == 'lambertian':
        sfc_dict = {}
        sfc_dict['alb'] = np.array([params['surface_albedo']]).reshape((1, 1))

    sfc_dict['dx'] = cld0.lay['dx']['data']*cld0.lay['nx']['data']
    sfc_dict['dy'] = cld0.lay['dy']['data']*cld0.lay['ny']['data']

    fname_sfc = f"{fdir_tmp}/sfc.pk"
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)

    sfc_2d = er3t.rtm.mca.mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname=f"{fdir_tmp}/mca_sfc_2d.bin", overwrite=overwrite)

    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds   = [atm1d0]

    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname=f"{fdir_tmp}/mca_atm_3d.bin", overwrite=overwrite)
    atm_3ds   = [atm3d0]

    rad = np.zeros_like(params['sensor_azimuth_angle'])
    rad_std = np.zeros_like(params['sensor_azimuth_angle'])
    for i in range(rad.size):

        fdir = f"{fdir_tmp}/rad_{solver.lower()}_{i:04d}"
        # if (not overwrite) and (not os.path.exists(fdir)):
        #     overwrite = True

        mca0 = er3t.rtm.mca.mcarats_ng(
                date=params['date'],
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                sca=sca,
                Ng=abs0.Ng,
                fdir=fdir,
                target='rad',
                Nrun=3,
                solar_zenith_angle=params['solar_zenith_angle'],
                solar_azimuth_angle=0.0,
                sensor_zenith_angle=params['sensor_zenith_angle'],
                sensor_azimuth_angle=params['sensor_azimuth_angle'][i],
                sensor_altitude=params['output_altitude'][-1],
                photons=params['photons'],
                weights=abs0.coef['weight']['data'],
                surface=sfc_2d,
                solver=solver,
                mp_mode='mpi',
                overwrite=overwrite
                )

        fname_h5 = f"{fdir_tmp}/mca-out-rad-{solver.lower()}_{_metadata['Function']}_{i:04d}.h5"
        out0 = er3t.rtm.mca.mca_out_ng(fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

        rad[i] = out0.data['rad']['data']
        rad_std[i] = out0.data['rad_std']['data']

    data = {
      'rad': rad,\
      'rad_std': rad_std,\
      'Ng': abs0.Ng,\
            }

    return data

def shd_rad_one(
        params,
        solver='IPA',
        f_toa=None,
        surface='ocean',
        overwrite=False,
        ):

    """
    A test run for clear sky case
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}/{params['wavelength']:04.0f}"
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    fname_atm = f"{fdir_tmp}/atm.pk"
    atm0      = er3t.pre.atm.atm_atmmod(levels=params['output_altitude'], fname=fname_atm, fname_atmmod=params['atmosphere_file'], overwrite=overwrite)

    fname_abs = f"{fdir_tmp}/abs.pk"
    abs0      = er3t.pre.abs.abs_rep(wavelength=params['wavelength'], fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)
    # rescale TOA flux
    if f_toa is not None:
        f_toa_ = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()
        abs0.coef['solar']['data'] = f_toa/f_toa_ * abs0.coef['solar']['data']

    fname_cld = f"{fdir_tmp}/cld.pk"
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

    if surface == 'ocean':
        sfc_dict = {}
        sfc_dict['windspeed'] = np.array([params['windspeed']]).reshape((1, 1))
        sfc_dict['pigment'] = np.array([params['pigment']]).reshape((1, 1))
    elif surface == 'land':
        sfc_dict = {}
        sfc_dict['fiso'] = np.array([params['f_iso']]).reshape((1, 1))
        sfc_dict['fvol'] = np.array([params['f_vol']]).reshape((1, 1))
        sfc_dict['fgeo'] = np.array([params['f_geo']]).reshape((1, 1))
    elif surface == 'lambertian':
        sfc_dict = {}
        sfc_dict['alb'] = np.array([params['surface_albedo']]).reshape((1, 1))

    sfc_dict['dx'] = cld0.lay['dx']['data']*cld0.lay['nx']['data']
    sfc_dict['dy'] = cld0.lay['dy']['data']*cld0.lay['ny']['data']

    fname_sfc = f"{fdir_tmp}/sfc.pk"
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)

    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir_tmp, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir_tmp, overwrite=overwrite)
    atm_1ds   = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir_tmp, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]

    fdir = f"{fdir_tmp}/rad_{solver.lower()}"
    if (not overwrite) and (not os.path.exists(fdir)):
        overwrite = True

    shd0 = er3t.rtm.shd.shdom_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            fdir=fdir,
            target='rad',
            Niter=1000,
            Nmu=32,
            Nphi=64,
            solar_zenith_angle=params['solar_zenith_angle'],
            solar_azimuth_angle=0.0,
            sensor_azimuth_angles=params['sensor_azimuth_angle'],
            sensor_zenith_angles=np.repeat(params['sensor_zenith_angle'], params['sensor_azimuth_angle'].size),
            sensor_altitude=params['output_altitude'][-1],
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            sol_acc=1e-6,
            split_acc=1e-6,
            surface=sfc_2d,
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    fname_h5 = None
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    data = {
            'rad': out0.data['rad']['data'],\
            'Ng': abs0.Ng,
            }

    return data

def test_100_rad_one(
        wavelength,
        cot,
        cer,
        icount,
        surface='ocean',
        plot=True,
        overwrite=False,
        ):

    params = {
                            'date': datetime.datetime(2024, 5, 18),
                 'atmosphere_file': f"{er3t.common.fdir_data_atmmod}/afglss.dat",
                  'surface_albedo': 0.8,
              'solar_zenith_angle': 30.0,
             'solar_azimuth_angle': 0.0,
                 'sensor_altitude': 120.0,
             'sensor_zenith_angle': 30.0,
            'sensor_azimuth_angle': np.arange(0.0, 180.1, 5.0),
            # 'sensor_azimuth_angle': np.array([0.0]),
                 'sensor_altitude': 120.0,
                      'wavelength': wavelength,
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
                 'output_altitude': np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0)),
         }

    if params['cloud_optical_thickness'] > 0.0:
        params['photons'] = 1.0e8

    data_lrt = lrt_rad_one(params, surface=surface, overwrite=overwrite)
    f_toa = data_lrt['f_down']/np.cos(np.deg2rad(params['solar_zenith_angle']))/er3t.util.cal_sol_fac(params['date'])

    data_mca = mca_rad_one(params, f_toa=f_toa, surface=surface, overwrite=overwrite)

    data_shd = shd_rad_one(params, f_toa=f_toa, surface=surface, overwrite=overwrite)

    # add the other half (180.0 - 360.0)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    params['sensor_azimuth_angle'] = np.append(params['sensor_azimuth_angle'], 180.0+params['sensor_azimuth_angle'][1:])
    data_lrt['rad'] = np.append(data_lrt['rad'], data_lrt['rad'][:-1][::-1])

    data_mca['rad'] = np.append(data_mca['rad'], data_mca['rad'][:-1][::-1])
    data_mca['rad_std'] = np.append(data_mca['rad_std'], data_mca['rad_std'][:-1][::-1])

    data_shd['rad'] = np.append(data_shd['rad'], data_shd['rad'][:-1][::-1])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    error_shd_rad = np.nanmean(np.abs(data_lrt['rad']-data_shd['rad'])/data_lrt['rad']*100.0)
    error_mca_rad = np.nanmean(np.abs(data_lrt['rad']-data_mca['rad'])/data_lrt['rad']*100.0)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 5))
        # fig.suptitle('COT=%.1f, CER=%.1f $\\mu m$' % (params['cloud_optical_thickness'], params['cloud_effective_radius']))
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax1.plot(params['sensor_azimuth_angle'], data_lrt['rad'], color='k' , lw=3.0, alpha=0.9, ls='-', zorder=0)
        ax1.fill_between(params['sensor_azimuth_angle'], data_mca['rad']-data_mca['rad_std'], data_mca['rad']+data_mca['rad_std'], color='blue', lw=1.0, alpha=1.0, zorder=1)
        ax1.plot(params['sensor_azimuth_angle'], data_shd['rad'], color='r' , lw=1.0, alpha=1.0, ls='-', zorder=2)
        ax1.set_xlabel('Viewing Azimuth Angle [$^\\circ$]')
        ax1.set_ylabel('Radiance [$\\mathrm{W m^{-2} nm^{-1} sr^{-1}}$]')
        ax1.set_xlim((0, 360.0))
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(0.0, 360.1, 60.0)))

        ax2 = ax1.twinx()

        diff1 = (data_mca['rad']-data_mca['rad_std']-data_lrt['rad'])/data_lrt['rad'] * 100.0
        diff2 = (data_mca['rad']+data_mca['rad_std']-data_lrt['rad'])/data_lrt['rad'] * 100.0
        ax2.fill_between(params['sensor_azimuth_angle'], diff1, diff2, color='cyan', lw=0.75, alpha=1.0, zorder=1)

        diff = (data_shd['rad']-data_lrt['rad'])/data_lrt['rad'] * 100.0
        ax2.plot(params['sensor_azimuth_angle'], diff, color='magenta', lw=1.0, alpha=1.0, ls='-', zorder=2)

        ax2.set_ylim((-100.0, 100.0))

        ax2.axhline(0.0, color='gray', ls='--', zorder=0)

        ax2.set_ylabel('Difference [%]', rotation=270.0, labelpad=16)
        #╰──────────────────────────────────────────────────────────────╯#

        patches_legend = [
                          mpatches.Patch(color='black', label='libRadtran'), \
                          mpatches.Patch(color='blue' , label='MCARaTS'), \
                          mpatches.Patch(color='red'  , label='SHDOM'), \
                          mpatches.Patch(color='cyan'   , label='MCARaTS Diff.'), \
                          mpatches.Patch(color='magenta', label='SHDOM Diff.'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%4.4d_%s_%06.1fnm_cot-%05.1f_cer-%05.1f.png' % (icount, _metadata['Function'], params['wavelength'], params['cloud_optical_thickness'], params['cloud_effective_radius']), bbox_inches='tight', metadata=_metadata)
        # plt.close()
        plt.show()
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#



def lrt_rad_spec_slit(
        params,
        surface='lambertian',
        overwrite=False,
        ):

    """
    libRadtran radiance calculation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s/cot-%04.1f_cer-%04.1f' % (er3t.common.fdir_examples, name_tag, _metadata['Function'], params['cloud_optical_thickness'], params['cloud_effective_radius'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = params['atmosphere_file']
    # lrt_cfg['mol_abs_param'] = 'reptran fine'
    lrt_cfg['mol_abs_param'] = 'reptran coarse'
    lrt_cfg['number_of_streams'] = 32

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = params['cloud_optical_thickness']
    cld_cfg['cloud_effective_radius']  = params['cloud_effective_radius']
    cld_cfg['cloud_altitude'] = np.arange(params['cloud_top_height']-params['cloud_geometric_thickness'], params['cloud_top_height']+0.01, 0.1)

    inits_rad = []
    inits_flx = []

    for wvl0 in params['wavelengths']:

        fname_out = '%s/output_rad_%4.4d.txt' % (fdir_tmp, wvl0)
        if (not overwrite) and (not os.path.exists(fname_out)):
            overwrite = True

        # mute_list = ['source solar', 'slit_function_file', 'wavelength', 'spline', 'albedo']
        mute_list = ['source solar', 'albedo']

        input_dict_extra = {
                # 'wavelength_add': '%.1f %.1f' % (wvl0, wvl0),
                }

        if surface == 'land':
            input_dict_extra['brdf_ambrals iso'] = params['f_iso']
            input_dict_extra['brdf_ambrals vol'] = params['f_vol']
            input_dict_extra['brdf_ambrals geo'] = params['f_geo']
        elif surface == 'ocean':
            input_dict_extra['brdf_cam'] = 'u10 %.2f' % params['windspeed']
        else:
            mute_list.pop()

        init_rad = er3t.rtm.lrt.lrt_init_mono_rad(
                input_file  = '%s/input_rad_%4.4d.txt' % (fdir_tmp, wvl0),
                output_file = fname_out,
                date        = params['date'],
                surface_albedo = params['surface_albedo'],
                solar_zenith_angle   = params['solar_zenith_angle'],
                solar_azimuth_angle  = 0.0,
                sensor_zenith_angle  = params['sensor_zenith_angle'],
                sensor_azimuth_angle = params['sensor_azimuth_angle'],
                wavelength         = wvl0,
                output_altitude    = params['sensor_altitude'],
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg,
                mute_list          = mute_list,
                input_dict_extra   = input_dict_extra,
                )
        inits_rad.append(init_rad)

        init_flx = er3t.rtm.lrt.lrt_init_mono_flx(
                input_file  = '%s/input_flx_%4.4d.txt' % (fdir_tmp, wvl0),
                output_file = '%s/output_flx_%4.4d.txt' % (fdir_tmp, wvl0),
                date        = params['date'],
                surface_albedo = params['surface_albedo'],
                solar_zenith_angle = params['solar_zenith_angle'],
                wavelength         = wvl0,
                output_altitude    = params['output_altitude'][-1],
                lrt_cfg            = lrt_cfg,
                cld_cfg            = None,
                mute_list          = mute_list,
                input_dict_extra   = input_dict_extra,
                )
        inits_flx.append(init_flx)

    if overwrite:
        er3t.rtm.lrt.lrt_run_mp(inits_rad+inits_flx)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx(inits_flx)
    data1 = er3t.rtm.lrt.lrt_read_uvspec_rad(inits_rad)

    data = {
              'f_down': np.squeeze(data0.f_down),
              'rad': np.squeeze(data1.rad),
            }

    return data

def lrt_rad_spec(
        params,
        surface='lambertian',
        overwrite=False,
        ):

    """
    libRadtran radiance calculation
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = f"{er3t.common.fdir_examples}/tmp-data/{name_tag}/{_metadata['Function']}/cot-{params['cloud_optical_thickness']:04.1f}_cer-{params['cloud_effective_radius']:04.1f}"
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

    inits_rad = []
    inits_flx = []

    for wvl0 in params['wavelengths']:

        fname_out = f"{fdir_tmp}/output_rad_{wvl0:04.0f}.txt"
        if (not overwrite) and (not os.path.exists(fname_out)):
            overwrite = True

        mute_list = ['source solar', 'slit_function_file', 'wavelength', 'spline', 'albedo']

        input_dict_extra = {
                'wavelength_add': f"{wvl0:.1f} {wvl0:.1f}",
                }

        if surface == 'land':
            input_dict_extra['brdf_ambrals iso'] = params['f_iso']
            input_dict_extra['brdf_ambrals vol'] = params['f_vol']
            input_dict_extra['brdf_ambrals geo'] = params['f_geo']
        elif surface == 'ocean':
            input_dict_extra['brdf_cam'] = f"u10 {params['windspeed']:.8f}"
        else:
            mute_list.pop()

        init_rad = er3t.rtm.lrt.lrt_init_mono_rad(
                input_file  = f"{fdir_tmp}/input_rad_{wvl0:04.0f}.txt",
                output_file = fname_out,
                date        = params['date'],
                surface_albedo = params['surface_albedo'],
                solar_zenith_angle   = params['solar_zenith_angle'],
                solar_azimuth_angle  = 0.0,
                sensor_zenith_angle  = params['sensor_zenith_angle'],
                sensor_azimuth_angle = params['sensor_azimuth_angle'],
                wavelength         = wvl0,
                output_altitude    = params['sensor_altitude'],
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg,
                mute_list          = mute_list,
                input_dict_extra   = input_dict_extra,
                )
        inits_rad.append(init_rad)

        init_flx = er3t.rtm.lrt.lrt_init_mono_flx(
                input_file  = f"{fdir_tmp}/input_flx_{wvl0:04.0f}.txt",
                output_file = f"{fdir_tmp}/output_flx_{wvl0:04.0f}.txt",
                date        = params['date'],
                surface_albedo = params['surface_albedo'],
                solar_zenith_angle = params['solar_zenith_angle'],
                wavelength         = wvl0,
                output_altitude    = params['output_altitude'][-1],
                lrt_cfg            = lrt_cfg,
                cld_cfg            = None,
                mute_list          = mute_list,
                input_dict_extra   = input_dict_extra,
                )
        inits_flx.append(init_flx)

    if overwrite:
        er3t.rtm.lrt.lrt_run_mp(inits_rad+inits_flx)

    data0 = er3t.rtm.lrt.lrt_read_uvspec_flx(inits_flx)
    data1 = er3t.rtm.lrt.lrt_read_uvspec_rad(inits_rad)

    data = {
              'f_down': np.squeeze(data0.f_down),
              'rad': np.squeeze(data1.rad),
            }

    return data

def shd_rad_spec(
        params,
        solver='IPA',
        f_toa=None,
        surface='ocean',
        overwrite=False,
        ):

    """
    A test run for clear sky case
    """

    out0 = np.zeros(params['wavelengths'].size, dtype=np.float32)
    out1 = np.zeros(params['wavelengths'].size, dtype=np.float32)

    for i, wavelength in enumerate(params['wavelengths']):
        params['wavelength'] = wavelength
        data_shd = shd_rad_one(params, f_toa=f_toa[i], surface=surface, overwrite=overwrite)
        out0[i] = data_shd['rad']
        out1[i] = data_shd['Ng']

    data = {
      'rad': out0,\
      'Ng': out1,\
            }

    return data

def mca_rad_spec(
        params,
        solver='3D',
        f_toa=None,
        surface='ocean',
        overwrite=False,
        ):

    """
    A test run for clear sky case
    """

    out0 = np.zeros(params['wavelengths'].size, dtype=np.float32)
    out1 = np.zeros(params['wavelengths'].size, dtype=np.float32)
    out2 = np.zeros(params['wavelengths'].size, dtype=np.float32)

    for i, wavelength in enumerate(params['wavelengths']):
        params['wavelength'] = wavelength
        data_mca = mca_rad_one(params, f_toa=f_toa[i], surface=surface, overwrite=overwrite)
        out0[i] = data_mca['rad']
        out1[i] = data_mca['rad_std']
        out2[i] = data_mca['Ng']

    data = {
      'rad': out0,\
      'rad_std': out1,\
      'Ng': out2,\
            }

    return data

def test_100_rad_spec(
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
                 'atmosphere_file': f"{er3t.common.fdir_data_atmmod}/afglss.dat",
                  'surface_albedo': 0.8,
              'solar_zenith_angle': 63.0,
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
                 'output_altitude': np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0)),
         }

    if params['cloud_optical_thickness'] > 0.0:
        params['photons'] = 1.0e9

    # data_lrt_slit = lrt_rad_spec_slit(params, surface=surface, overwrite=False)
    data_lrt = lrt_rad_spec(params, surface=surface, overwrite=False)
    f_toa = data_lrt['f_down']/np.cos(np.deg2rad(params['solar_zenith_angle']))/er3t.util.cal_sol_fac(params['date'])

    data_mca = mca_rad_spec(params, f_toa=f_toa, surface=surface, overwrite=False)

    data_shd = shd_rad_spec(params, f_toa=f_toa, surface=surface, overwrite=False)
    print('libRadtran:', np.trapz(data_lrt['rad'], x=params['wavelengths']))
    print('MCARaTS:', np.trapz(data_mca['rad'], x=params['wavelengths']))
    print('SHDOM:', np.trapz(data_shd['rad'], x=params['wavelengths']))
    sys.exit()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 5))
        # fig.suptitle('COT=%.1f, CER=%.1f $\\mu m$' % (params['cloud_optical_thickness'], params['cloud_effective_radius']))
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        # ax1.plot(params['wavelengths'], data_lrt_slit['rad'], color='black', lw=1.5, alpha=1.0, ls='-', zorder=0)
        ax1.plot(params['wavelengths'], data_lrt['rad'], color='black', lw=3.0, alpha=0.9, ls='-', zorder=0)
        ax1.fill_between(params['wavelengths'], data_mca['rad']-data_mca['rad_std'], data_mca['rad']+data_mca['rad_std'], color='blue', lw=1.0, alpha=1.0, zorder=1)
        ax1.plot(params['wavelengths'], data_shd['rad'], color='red'  , lw=0.8, alpha=1.0, ls='-', zorder=2)

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Radiance [$\\mathrm{W m^{-2} nm^{-1} sr^{-1}}$]')
        ax1.set_ylim((0.0, 0.3))

        ax2 = ax1.twinx()

        diff1 = (data_mca['rad']-data_mca['rad_std']-data_lrt['rad'])/data_lrt['rad'] * 100.0
        diff2 = (data_mca['rad']+data_mca['rad_std']-data_lrt['rad'])/data_lrt['rad'] * 100.0
        ax2.fill_between(params['wavelengths'], diff1, diff2, color='cyan', lw=0.75, alpha=1.0, zorder=1)

        diff = (data_shd['rad']-data_lrt['rad'])/data_lrt['rad'] * 100.0
        ax2.plot(params['wavelengths'], diff, color='magenta', lw=1.0, alpha=1.0, ls='-', zorder=2)

        ax2.set_ylim((-100.0, 100.0))

        ax2.axhline(0.0, color='gray', ls='--', zorder=0)

        ax2.set_ylabel('Difference [%]', rotation=270.0, labelpad=16)

        # ax3 = ax1.twinx()
        # ax3.plot(params['wavelengths'], data_shd['Ng'], color='orange', lw=1.0, alpha=1.0, ls='-', zorder=0)
        # ax3.set_ylim((-12, 12))
        # ax3.axis('off')
        #╰──────────────────────────────────────────────────────────────╯#

        patches_legend = [
                          mpatches.Patch(color='black', label='libRadtran'), \
                          mpatches.Patch(color='blue' , label='MCARaTS'), \
                          mpatches.Patch(color='red'  , label='SHDOM'), \
                          mpatches.Patch(color='cyan'     , label='MCARaTS Diff.'), \
                          mpatches.Patch(color='magenta'  , label='SHDOM Diff.'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper center', fontsize=14)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fname_png = f"{icount:04d}_{_metadata['Function']}_cot-{params['cloud_optical_thickness']:05.1f}_cer-{params['cloud_effective_radius']:05.1f}.png"
        fig.savefig(fname_png, bbox_inches='tight', metadata=_metadata)
        # plt.close()
        plt.show()
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#



if __name__ == '__main__':

    warnings.warn('\nUnder active development ...')

    if er3t.common.has_mcarats & er3t.common.has_libradtran:

        # test_00_solar()

        # test_100_rad_one(555.0, 0.0, 1.0, 100, surface='ocean', plot=True, overwrite=True)
        # test_100_rad_one(555.0, 0.0, 1.0, 100, surface='land', plot=True, overwrite=True)
        # test_100_rad_one(555.0, 10.0, 12.0, 100, surface='ocean', plot=True, overwrite=True)
        # test_100_flux_one(555.0, 10.0, 12.0, 100, plot=True, overwrite=True)

        # icount = 0
        # for cot in np.concatenate((np.arange(0.0, 1.0, 0.2), np.arange(1.0, 8.1, 2.0), np.arange(10.0, 50.1, 5.0))):
        # for cot in np.arange(25.0, 50.1, 5.0):
        #     for cer in np.arange(1.0, 25.1, 2.0):
        #         test_100_flux_one(2130.0, cot, cer, icount, plot=True, overwrite=False)
        #         icount += 1

        # test_100_flux_one(2130.0, 50.0, 9.0, 100, plot=True, overwrite=True)

        wavelengths = np.arange(300.0, 3201.0, 5.0)
        test_100_rad_spec(wavelengths, 0.0, 1.0, 100, overwrite=False)

    else:

        msg = '\nError [00_er3t_bmk.py]: Needs to have both <MCARaTS> and <libRadtran> to be installed for performing benchmark tests.'
        raise OSError(msg)
