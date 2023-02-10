import os
import sys
import glob
import datetime
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
# import cartopy.crs as ccrs
# mpl.use('Agg')

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_les, cld_sat
from er3t.pre.cld import cld_gen_hem as cld_gen
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca
from er3t.util import cal_r_twostream

import er3t.common


# global variables
#/-----------------------------------------------------------------------------\
name_tag = os.path.relpath(__file__).replace('.py', '')
photons = 1e8
Ncpu    = 12
#\-----------------------------------------------------------------------------/





def test_pha_mie(
        wavelength=650.0,
        cloud_effective_radius=5.0,
        solver='3d',
        overwrite=True,
        plot=True,
        ):

    """
    Test Mie phase function
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='tmp-data/%s/%s/wvl-%4.4dnm/cer-%04.1f' % (name_tag, _metadata['Function'], wavelength, cloud_effective_radius)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #/-----------------------------------------------------------------------------\
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    #\-----------------------------------------------------------------------------/


    # define an absorption object
    #/-----------------------------------------------------------------------------\
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    #\-----------------------------------------------------------------------------/


    # define an cloud object
    #/-----------------------------------------------------------------------------\
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    fname_les = '%s/les.pk' % fdir
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[1, 1, 25], overwrite=overwrite)
    cld0.lay['cer']['data'][...] = cloud_effective_radius
    #\-----------------------------------------------------------------------------/


    # define mca_sca object
    #/-----------------------------------------------------------------------------\
    pha0 = pha_mie(wavelength=wavelength)

    # for key in pha0.data.keys():
    #     data0 = pha0.data[key]['data']
    #     if isinstance(data0, np.ndarray):
    #         print(key, data0.shape)


    if False:

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(pha0.data['pha']['data'].T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        #\--------------------------------------------------------------/#
        #/--------------------------------------------------------------\#
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<pha_wvl-%4.4dnm_cer-%04.1fmm>.png' % (_metadata['Function'], wavelength, cloud_effective_radius), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#

    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)

    if False:

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(pha0.data['pha']['data'].T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        #\--------------------------------------------------------------/#
        #/--------------------------------------------------------------\#
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        cbar.set_label('', rotation=270, labelpad=4.0)
        cbar.set_ticks([])
        cax.axis('off')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<pha_wvl-%4.4dnm_cer-%04.1fmm>.png' % (_metadata['Function'], wavelength, cloud_effective_radius), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#
    #\-----------------------------------------------------------------------------/


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    #/-----------------------------------------------------------------------------\
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, overwrite=overwrite)
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]
    #\-----------------------------------------------------------------------------/

def test_pha_mie_run(
        wavelength=650.0,
        cloud_effective_radius=5.0,
        solver='3d',
        overwrite=True,
        plot=True,
        ):

    """
    Test Mie phase function
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='tmp-data/%s/%s/wvl-%4.4dnm/cer-%04.1f' % (name_tag, _metadata['Function'], wavelength, cloud_effective_radius)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #/-----------------------------------------------------------------------------\
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    #\-----------------------------------------------------------------------------/


    # define an absorption object
    #/-----------------------------------------------------------------------------\
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    #\-----------------------------------------------------------------------------/


    # define an cloud object
    #/-----------------------------------------------------------------------------\
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples
    fname_les = '%s/les.pk' % fdir
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[1, 1, 25], overwrite=overwrite)
    cld0.lay['cer']['data'][...] = cloud_effective_radius
    #\-----------------------------------------------------------------------------/
    print(cld0.lay['cer']['data'].mean())


    # define mca_sca object
    #/-----------------------------------------------------------------------------\
    pha0 = pha_mie(wavelength=wavelength)

    # for key in pha0.data.keys():
    #     data0 = pha0.data[key]['data']
    #     if isinstance(data0, np.ndarray):
    #         print(key, data0.shape)


    if False:

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(pha0.data['pha']['data'].T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        #\--------------------------------------------------------------/#
        #/--------------------------------------------------------------\#
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<pha_wvl-%4.4dnm_cer-%04.1fmm>.png' % (_metadata['Function'], wavelength, cloud_effective_radius), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#

    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)

    if plot:

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(pha0.data['pha']['data'].T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        #\--------------------------------------------------------------/#
        #/--------------------------------------------------------------\#
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        cbar.set_label('', rotation=270, labelpad=4.0)
        cbar.set_ticks([])
        cax.axis('off')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<pha_wvl-%4.4dnm_cer-%04.1fmm>.png' % (_metadata['Function'], wavelength, cloud_effective_radius), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#
    #\-----------------------------------------------------------------------------/


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    #/-----------------------------------------------------------------------------\
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, overwrite=overwrite)
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]
    #\-----------------------------------------------------------------------------/


    # define mcarats object
    #/-----------------------------------------------------------------------------\
    # run mcarats
    mca0 = mcarats_ng(
            date=datetime.datetime(2017, 8, 13),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            target='radiance',
            surface_albedo=0.03,
            sca=sca,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=45.0,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            sensor_altitude=705000.0,
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            photons=photons,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=overwrite
            )
    #\-----------------------------------------------------------------------------/


    # define mcarats output object
    #/-----------------------------------------------------------------------------\
    fname_h5 = '%s/mca-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = mca_out_ng(fname=fname_h5, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    #\-----------------------------------------------------------------------------/


    # plot
    #/-----------------------------------------------------------------------------\
    if plot:
        fname_png = '%s-%s_%s_wvl-%4.4dnm_cer-%04.1fmm.png' % (name_tag, _metadata['Function'], solver.lower(), wavelength, cloud_effective_radius)

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='jet', vmin=0.0, vmax=0.3, origin='lower')
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        ax1.set_title('Radiance at %.2f nm (%s Mode)' % (wavelength, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #\-----------------------------------------------------------------------------/

    # References
    #/-----------------------------------------------------------------------------\
    print('\nReferences:')
    print('-'*80)
    for reference in er3t.common.references:
        print(reference)
        print('-'*80)
    print()
    #\-----------------------------------------------------------------------------/


def figure_asy(wavelength=650.0, refs=[1, 5, 10, 15, 20]):

    pha0 = pha_mie(wavelength=wavelength)

    # figure
    #/----------------------------------------------------------------------------\#
    plt.close('all')
    fig = plt.figure(figsize=(7.5, 7))
    #/--------------------------------------------------------------\#

    ax1 = fig.add_subplot(111)


    ax1.plot(pha0.data['ref']['data'], pha0.data['asy']['data'], color='k', lw=2.0) #, extent=extent, vmin=0.0, vmax=0.5)

    colors = mpl.cm.jet(np.linspace(0.0, 1.0, len(refs)))
    patches_legend = []
    for i, ref0 in enumerate(refs):
        ax1.axvline(ref0, ls='--', color=colors[i, ...], lw=2.0, alpha=0.8)
        patch0 = mpatches.Patch(color=colors[i, ...], label='CER=%d$\mu m$' % ref0)
        patches_legend.append(patch0)

    ax1.set_xlim((0, 25))
    ax1.set_ylim((0.75, 0.9))
    ax1.set_xlabel('Cloud Effective Radius [$\mu m$]')
    ax1.set_ylabel('Asymmetry Parameter (g)')
    #\--------------------------------------------------------------/#

    ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)

    # save figure
    #/--------------------------------------------------------------\#
    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    plt.savefig('%s_wvl-%4.4dnm.png' % (_metadata['Function'], wavelength), bbox_inches='tight', metadata=_metadata)
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    figure_asy(wavelength=650.0, refs=[1, 5, 10, 15, 20])

    # test_pha_mie_run(cloud_effective_radius=5.0, overwrite=True)
    # test_pha_mie_run(cloud_effective_radius=20.0, overwrite=True)
