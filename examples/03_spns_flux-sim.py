import os
import sys
import glob
import copy
from collections import OrderedDict
import h5py
import numpy as np
import datetime
import time
import pickle
import multiprocessing as mp
from scipy.io import readsav
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

import cartopy.crs as ccrs
from owslib.wmts import WebMapTileService

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_les, cld_sat
from er3t.pre.pha import pha_mie_wc as pha_mie

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng
from er3t.rtm.mca_v010 import mca_sca
from er3t.util import send_email
from er3t.util.himawari import himawari_l2
from er3t.util.modis import grid_modis_by_extent





def plot_flux_comp_time_series(flt_sim0):

    fig = plt.figure(figsize=(12, 10))
    # fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()

    ax3 = fig.add_subplot(212)
    ax4 = ax3.twinx()

    for flt_trk in flt_sim0.flt_trks:

        ax1.plot(flt_trk['tmhr'], flt_trk['f-up_mca-ipa'], c='b', lw=2.5)
        ax1.plot(flt_trk['tmhr'], flt_trk['f-up_mca-3d'] ,  c='r', lw=1.5)
        ax1.scatter(flt_trk['tmhr'], flt_trk['f-up_ssfr'],  c='k', s=1)
        ax2.plot(flt_trk['tmhr'], flt_trk['alt'], c='orange', lw=1.5, alpha=0.7)

        ax3.plot(flt_trk['tmhr'], flt_trk['f-down_mca-ipa'], c='b', lw=2.5)
        ax3.plot(flt_trk['tmhr'], flt_trk['f-down_mca-3d'] ,  c='r', lw=1.5)
        ax3.scatter(flt_trk['tmhr'], flt_trk['f-down_ssfr'],  c='k', s=1)
        ax4.plot(flt_trk['tmhr'], flt_trk['alt'], c='orange', lw=1.5, alpha=0.7)


    ax1.set_ylim((0.0, 1.5))
    ax1.set_xlabel('UTC [hour]')
    ax1.set_ylabel('$F_\\uparrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('Flux Comparison at %.2f nm' % sim0.wvl)
    patches_legend = [
                mpatches.Patch(color='black' , label='SSFR'),
                mpatches.Patch(color='red'   , label='MCARaTS 3D'),
                mpatches.Patch(color='blue'  , label='MCARaTS IPA'),
                mpatches.Patch(color='orange', label='Altitude')
                ]
    ax2.set_ylabel('Altitude [km]', color='orange', rotation=270, labelpad=20)
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)
    ax2.set_ylim((0.0, 8.0))

    ax3.set_ylabel('$F_\downarrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax3.set_xlabel('UTC [hour]')
    patches_legend = [
                mpatches.Patch(color='black' , label='SSFR'),
                mpatches.Patch(color='red'   , label='MCARaTS 3D'),
                mpatches.Patch(color='blue'  , label='MCARaTS IPA'),
                mpatches.Patch(color='orange', label='Altitude')
                ]
    ax4.set_ylabel('Altitude [km]', color='orange', rotation=270, labelpad=20)
    ax3.legend(handles=patches_legend, loc='upper right', fontsize=12)
    ax3.set_ylim((0.0, 2.0))
    ax4.set_ylim((0.0, 8.0))

    plt.savefig('flux_comp_%.2fnm.png' % sim0.wvl, bbox_inches='tight')
    plt.show()
    # plt.close(fig)
    # ---------------------------------------------------------------------

def plot_flux_2d(flt_sim0):

    N = len(flt_sim0.flt_trks)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    for i in range(N):

        flt_trk      = flt_sim0.flt_trks[i]
        sat_img      = flt_sim0.sat_imgs[i]

        ax1.imshow(sat_img['cot']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=50.0, alpha=0.1, zorder=0)
        ax2.imshow(sat_img['f-up_mca-3d']   , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax3.imshow(sat_img['f-up_mca-ipa']  , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax4.imshow(sat_img['cer']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=24.0, alpha=0.1, zorder=0)
        ax5.imshow(sat_img['f-down_mca-3d'] , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax6.imshow(sat_img['f-down_mca-ipa'], extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)

        ax1.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax2.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax3.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax4.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax5.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax6.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)

    ax1.set_title('COT')
    ax2.set_title('$F_\\uparrow$ 3D')
    ax3.set_title('$F_\\uparrow$ IPA')
    ax4.set_title('CER')
    ax5.set_title('$F_\downarrow$ 3D')
    ax6.set_title('$F_\downarrow$ IPA')

    plt.show()

def plot_flux_comp_scatter(flt_sim0):


    fig = plt.figure(figsize=(6.5, 6))
    ax1 = fig.add_subplot(111)

    for flt_trk in flt_sim0.flt_trks:

        if flt_trk['alt0'] > 5.0:
            ax1.scatter(flt_trk['f-up_ssfr'], flt_trk['f-up_mca-ipa'],  c='b', s=5.0, alpha=0.5)
            ax1.scatter(flt_trk['f-up_ssfr'], flt_trk['f-up_mca-3d'] ,  c='r', s=2.5)

    ax1.plot([0, 2], [0, 2], lw=1.0, ls='--', c='gray')
    ax1.set_xlabel('SSFR $F_\\uparrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_ylabel('MCARaTS $F_\\uparrow$ [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('Flux at %.2f nm' % flt_sim0.wvl)
    ax1.set_xlim((0.0, 1.2))
    ax1.set_ylim((0.0, 1.2))

    patches_legend = [
                mpatches.Patch(color='red'   , label='3D'),
                mpatches.Patch(color='blue'  , label='IPA')
                ]

    ax1.legend(handles=patches_legend, loc='upper left', fontsize=12)
    plt.savefig('flux_comp_scatter_%.2fnm.png' % flt_sim0.wvl, bbox_inches='tight')
    plt.show()

def plot_flux_2d(flt_sim0):

    N = len(flt_sim0.flt_trks)

    # fig = plt.figure(figsize=(12, 10))
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0, 2]
    ax4 = axes[1, 0]
    ax5 = axes[1, 1]
    ax6 = axes[1, 2]

    for i in range(N):

        flt_trk      = flt_sim0.flt_trks[i]
        sat_img      = flt_sim0.sat_imgs[i]

        ax1.imshow(sat_img['cot']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=50.0, alpha=0.1, zorder=0)
        ax2.imshow(sat_img['f-up_mca-3d']   , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax3.imshow(sat_img['f-up_mca-ipa']  , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax4.imshow(sat_img['cer']           , extent=sat_img['extent'], cmap='Greys_r', vmin=0.0, vmax=24.0, alpha=0.1, zorder=0)
        ax5.imshow(sat_img['f-down_mca-3d'] , extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)
        ax6.imshow(sat_img['f-down_mca-ipa'], extent=sat_img['extent'], cmap='jet'    , vmin=0.0, vmax=1.4 , alpha=0.1, zorder=0)

        ax1.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax2.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax3.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax4.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax5.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)
        ax6.plot(flt_trk['lon'], flt_trk['lat'], c='k', lw=1.0, zorder=2)

    ax1.set_title('COT')
    ax2.set_title('$F_\\uparrow$ 3D')
    ax3.set_title('$F_\\uparrow$ IPA')
    ax4.set_title('CER')
    ax5.set_title('$F_\downarrow$ 3D')
    ax6.set_title('$F_\downarrow$ IPA')

    plt.show()



def gen_himawari_true_color_image(
        dtime,
        extent,
        layer_name='AHI_Himawari8_CorrectedReflectance_TrueColor_HIGHRES',
        web_url='http://geoworldview.ssec.wisc.edu/onearth/wmts/epsg4326/wmts.cgi',
        proj=ccrs.PlateCarree(),
        fdir='.',
        run=True
        ):

    dtime_s = dtime.strftime('%Y-%m-%dT%H:%M:%SZ')
    fname = '%s/himawari_true_color_%s_%s.png' % (fdir, dtime_s, '-'.join(['%.2f' % extent0 for extent0 in extent]))

    if run:
        wmts = WebMapTileService(web_url, version='1.0.0')

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection=proj)
        ax1.add_wmts(wmts, layer_name, wmts_kwargs={'TIME': dtime_s})
        ax1.coastlines(resolution='50m', color='black', linewidth=0.5, alpha=0.8)
        ax1.set_extent(extent, crs=proj)
        ax1.outline_patch.set_visible(False)
        ax1.axis('off')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return fname

def check_continuity(data, threshold=0.1):

    data = np.append(data[0], data)

    return (np.abs(data[1:]-data[:-1]) < threshold)

def get_jday_himawari(fnames):

    """
    Get UTC time in hour from the satellite (HIMAWARI) file name

    Input:
        fnames: Python list, file paths of all the satellite data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname)
        strings  = filename.split('_')
        date_s   = strings[2]
        time_s   = strings[3]
        dtime_s  = date_s+time_s

        jday0 = (datetime.datetime.strptime(dtime_s, '%Y%m%d%H%M') - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0
        jday.append(jday0)

    return np.array(jday)

def partition_flight_track(flt_trk, tmhr_interval=0.1, margin_x=1.0, margin_y=1.0):

    """
    Input:
        flt_trk: Python dictionary that contains
            ['jday']: numpy array, UTC time in hour
            ['tmhr']: numpy array, UTC time in hour
            ['lon'] : numpy array, longitude
            ['lat'] : numpy array, latitude
            ['alt'] : numpy array, altitude
            ['sza'] : numpy array, solar zenith angle
            [...]   : numpy array, other data variables, e.g., 'f_up_0600'

        tmhr_interval=: float, time interval of legs to be partitioned, default=0.1
        margin_x=     : float, margin in x (longitude) direction that to be used to
                        define the rectangular box to contain cloud field, default=1.0
        margin_y=     : float, margin in y (latitude) direction that to be used to
                        define the rectangular box to contain cloud field, default=1.0


    Output:
        flt_trk_segments: Python list that contains data for each partitioned leg in Python dictionary, e.g., legs[i] contains
            [i]['jday'] : numpy array, UTC time in hour
            [i]['tmhr'] : numpy array, UTC time in hour
            [i]['lon']  : numpy array, longitude
            [i]['lat']  : numpy array, latitude
            [i]['alt']  : numpy array, altitude
            [i]['sza']  : numpy array, solar zenith angle
            [i]['jday0']: mean value
            [i]['tmhr0']: mean value
            [i]['lon0'] : mean value
            [i]['lat0'] : mean value
            [i]['alt0'] : mean value
            [i]['sza0'] : mean value
            [i][...]    : numpy array, other data variables
    """

    jday_interval = tmhr_interval/24.0

    jday_start = jday_interval * (flt_trk['jday'][0]//jday_interval)
    jday_end   = jday_interval * (flt_trk['jday'][-1]//jday_interval + 1)

    jday_edges = np.arange(jday_start, jday_end+jday_interval, jday_interval)

    flt_trk_segments = []

    for i in range(jday_edges.size-1):

        logic      = (flt_trk['jday']>=jday_edges[i]) & (flt_trk['jday']<=jday_edges[i+1]) & (np.logical_not(np.isnan(flt_trk['sza'])))
        if logic.sum() > 0:

            flt_trk_segment = {}

            for key in flt_trk.keys():
                flt_trk_segment[key]     = flt_trk[key][logic]
                if key in ['jday', 'tmhr', 'lon', 'lat', 'alt', 'sza']:
                    flt_trk_segment[key+'0'] = np.nanmean(flt_trk_segment[key])

            flt_trk_segment['extent'] = np.array([np.nanmin(flt_trk_segment['lon'])-margin_x, \
                                                  np.nanmax(flt_trk_segment['lon'])+margin_x, \
                                                  np.nanmin(flt_trk_segment['lat'])-margin_y, \
                                                  np.nanmax(flt_trk_segment['lat'])+margin_y])

            flt_trk_segments.append(flt_trk_segment)

    return flt_trk_segments

def run_mcarats_single(
        index,
        fname_sat,
        extent,
        solar_zenith_angle,
        cloud_top_height=None,
        fdir='tmp-data',
        wavelength=532.0,
        date=datetime.datetime.now(),
        target='flux',
        solver='3D',
        photons=5e6,
        Ncpu=14,
        overwrite=True,
        quiet=False
        ):

    """
    Run MCARaTS with specified inputs (a general function from 04_pre_mca.py)
    """

    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm_%3.3d.pk' % (fdir, index)
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    # define an absorption object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_abs = '%s/abs_%3.3d.pk' % (fdir, index)
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    # define an cloud object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_cld = '%s/cld_him_%3.3d.pk' % (fdir, index)

    if overwrite:
        him0      = himawari_l2(fnames=[fname_sat], extent=extent, vnames=['cld_height_acha'])
        lon_2d, lat_2d, cot_2d = grid_modis_by_extent(him0.data['lon']['data'], him0.data['lat']['data'], him0.data['cot']['data'], extent=extent)
        lon_2d, lat_2d, cer_2d = grid_modis_by_extent(him0.data['lon']['data'], him0.data['lat']['data'], him0.data['cermg']['data'], extent=extent)
        cot_2d[cot_2d>100.0] = 100.0
        cer_2d[cer_2d==0.0] = 1.0
        him0.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
        him0.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
        him0.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d)
        him0.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d)

        if cloud_top_height is None:
            lon_2d, lat_2d, cth_2d = grid_modis_by_extent(him0.data['lon']['data'], him0.data['lat']['data'], him0.data['cld_height_acha']['data'], extent=extent)
            cth_2d[cth_2d<0.0]  = 0.0; cth_2d /= 1000.0
            him0.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d)
            cloud_top_height = him0.data['cth_2d']['data']
        cld0 = cld_sat(sat_obj=him0, fname=fname_cld, cth=cloud_top_height, cgt=1.0, dz=(levels[1]-levels[0]), overwrite=overwrite)
    else:
        cld0 = cld_sat(fname=fname_cld, overwrite=overwrite)
    # ----------------------------------------------------------------------------------------------------

    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)


    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]

    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, quiet=quiet, overwrite=overwrite)
    atm_3ds = [atm3d0]
    # ------------------------------------------------------------------------------------------------------


    # define mcarats object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    mca0 = mcarats_ng(
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sca=sca,
            date=date,
            weights=abs0.coef['weight']['data'],
            solar_zenith_angle=solar_zenith_angle,
            fdir='%s/%.2fnm/himawari/%s/%3.3d' % (fdir, wavelength, solver.lower(), index),
            Nrun=3,
            photons=photons,
            solver=solver,
            target=target,
            Ncpu=Ncpu,
            mp_mode='py',
            quiet=quiet,
            overwrite=overwrite
            )
    # ------------------------------------------------------------------------------------------------------


    # define mcarats output object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    out0 = mca_out_ng(fname='%s/mca-out-%s-%s_himawari_%3.3d.h5' % (fdir, target.lower(), solver.lower(), index), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, quiet=quiet, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    return atm0, cld0, out0

def interpolate_3d_to_flight_track(flt_trk, data_3d):

    """
    Extract radiative properties along flight track from MCARaTS outputs

    Input:
        flt_trk: Python dictionary
            ['tmhr']: UTC time in hour
            ['lon'] : longitude
            ['lat'] : latitude
            ['alt'] : altitude

        data_3d: Python dictionary
            ['lon']: longitude
            ['lat']: latitude
            ['alt']: altitude
            [...]  : other variables that contain 3D data field

    Output:
        flt_trk:
            [...]: add interpolated data from data_3d[...]
    """

    points = np.transpose(np.vstack((flt_trk['lon'], flt_trk['lat'], flt_trk['alt'])))

    lon_field = data_3d['lon']
    lat_field = data_3d['lat']
    dlon    = lon_field[1]-lon_field[0]
    dlat    = lat_field[1]-lat_field[0]
    lon_trk = flt_trk['lon']
    lat_trk = flt_trk['lat']
    indices_lon = np.int_(np.round((lon_trk-lon_field[0])/dlon, decimals=0))
    indices_lat = np.int_(np.round((lat_trk-lat_field[0])/dlat, decimals=0))

    indices_lon = np.int_(flt_trk['lon']-data_3d['lon'][0])

    for key in data_3d.keys():
        if key not in ['tmhr', 'lon', 'lat', 'alt']:
            f_interp     = RegularGridInterpolator((data_3d['lon'], data_3d['lat'], data_3d['alt']), data_3d[key])
            flt_trk[key] = f_interp(points)

            flt_trk['%s-alt-all' % key] = data_3d[key][indices_lon, indices_lat, :]

    return flt_trk

class flt_sim:

    def __init__(
            self,
            date=datetime.datetime.now(),
            photons=2e7,
            Ncpu=16,
            wavelength=None,
            flt_trks=None,
            sat_imgs=None,
            fname=None,
            overwrite=False,
            overwrite_rtm=False,
            quiet=False,
            verbose=False
            ):

        self.date      = date
        self.photons   = photons
        self.Ncpu      = Ncpu
        self.wvl       = wavelength
        self.flt_trks  = flt_trks
        self.sat_imgs  = sat_imgs
        self.overwrite = overwrite
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run(overwrite=overwrite_rtm)
            self.dump(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [flt_sim]: Please check if \'%s\' exists or provide \'wavelength\', \'flt_trks\', and \'sat_imgs\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'flt_trks') and hasattr(obj, 'sat_imgs'):
                if self.verbose:
                    print('Message [flt_sim]: Loading %s ...' % fname)
                self.wvl      = obj.wvl
                self.fname    = obj.fname
                self.flt_trks = obj.flt_trks
                self.sat_imgs = obj.sat_imgs
            else:
                sys.exit('Error   [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, overwrite=True):

        N = len(self.flt_trks)

        for i in range(N):

            print('%3.3d/%3.3d' % (i, N))

            flt_trk = self.flt_trks[i]
            sat_img = self.sat_imgs[i]

            try:
                atm0, cld_him0, mca_out_ipa0 = run_mcarats_single(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='IPA', fdir='tmp-data/%s/%09.4fnm' % (self.date.strftime('%Y%m%d'), self.wvl), photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)
                atm0, cld_him0, mca_out_3d0  = run_mcarats_single(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='3D' , fdir='tmp-data/%s/%09.4fnm' % (self.date.strftime('%Y%m%d'), self.wvl), photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)

                self.sat_imgs[i]['lon'] = cld_him0.lay['lon']['data']
                self.sat_imgs[i]['lat'] = cld_him0.lay['lat']['data']
                self.sat_imgs[i]['cot'] = cld_him0.lay['cot']['data']
                self.sat_imgs[i]['cer'] = cld_him0.lay['cer']['data']

                lon_sat = self.sat_imgs[i]['lon'][:, 0]
                lat_sat = self.sat_imgs[i]['lat'][0, :]
                dlon    = lon_sat[1]-lon_sat[0]
                dlat    = lat_sat[1]-lat_sat[0]
                lon_trk = self.flt_trks[i]['lon']
                lat_trk = self.flt_trks[i]['lat']
                indices_lon = np.int_(np.round((lon_trk-lon_sat[0])/dlon, decimals=0))
                indices_lat = np.int_(np.round((lat_trk-lat_sat[0])/dlat, decimals=0))
                self.flt_trks[i]['cot'] = self.sat_imgs[i]['cot'][indices_lon, indices_lat]
                self.flt_trks[i]['cer'] = self.sat_imgs[i]['cer'][indices_lon, indices_lat]

                if 'cth' in cld_him0.lay.keys():
                    self.sat_imgs[i]['cth'] = cld_him0.lay['cth']['data']
                    self.flt_trks[i]['cth'] = self.sat_imgs[i]['cth'][indices_lon, indices_lat]

                data_3d_mca = {
                    'lon'         : cld_him0.lay['lon']['data'][:, 0],
                    'lat'         : cld_him0.lay['lat']['data'][0, :],
                    'alt'         : atm0.lev['altitude']['data'],
                    }

                index_h = np.argmin(np.abs(atm0.lev['altitude']['data']-flt_trk['alt0']))
                if atm0.lev['altitude']['data'][index_h] > flt_trk['alt0']:
                    index_h -= 1
                if index_h < 0:
                    index_h = 0

                for key in mca_out_3d0.data.keys():
                    if key in ['f_down', 'f_down_diffuse', 'f_down_direct', 'f_up', 'toa']:
                        if 'toa' not in key:
                            vname = key.replace('_', '-') + '_mca-3d'
                            self.sat_imgs[i][vname] = mca_out_3d0.data[key]['data'][..., index_h]
                            data_3d_mca[vname] = mca_out_3d0.data[key]['data']
                for key in mca_out_ipa0.data.keys():
                    if key in ['f_down', 'f_down_diffuse', 'f_down_direct', 'f_up', 'toa']:
                        if 'toa' not in key:
                            vname = key.replace('_', '-') + '_mca-ipa'
                            self.sat_imgs[i][vname] = mca_out_ipa0.data[key]['data'][..., index_h]
                            data_3d_mca[vname] = mca_out_ipa0.data[key]['data']

                self.flt_trks[i] = interpolate_3d_to_flight_track(flt_trk, data_3d_mca)

            except:

                pass

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

def first_run(date,
        wavelength=532.0,
        spns=True,
        run_rtm=True,
        run_plt=True,
        fdir_sat='/argus/field/camp2ex/2019/p3/sat',
        fdir_ssfr='data/ssfr'):

    date_s = date.strftime('%Y%m%d')

    fdir = os.path.abspath('tmp-data/%s/%09.4fnm' % (date_s, wavelength))
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # prepare the data
    # ==================================================================================================
    # get all the avaiable satellite data (HIMAWARI) and calculate the time in hour for each file
    date_prev = (date-datetime.timedelta(days=1)).strftime('%Y%m%d')
    date_next = (date+datetime.timedelta(days=1)).strftime('%Y%m%d')
    fnames_himawari_all = sorted(glob.glob('%s/ftp.ssec.wisc.edu/camp2ex/clavrx/%s/*%s*.nc' % (fdir_sat, date_prev, date_prev))) +\
                      sorted(glob.glob('%s/ftp.ssec.wisc.edu/camp2ex/clavrx/%s/*%s*.nc' % (fdir_sat, date_s, date_s))) +\
                      sorted(glob.glob('%s/ftp.ssec.wisc.edu/camp2ex/clavrx/%s/*%s*.nc' % (fdir_sat, date_next, date_next)))
    jday_himawari_all   = get_jday_himawari(fnames_himawari_all)


    # flt_trks
    fname_ssfr = '%s/SSFR_%s_IWG.h5' % (fdir_ssfr, date_s)
    if not os.path.exists(fname_ssfr):
        sys.exit('Error   [first_run]: Cannot find SSFR data.')

    f_ssfr = h5py.File(fname_ssfr, 'r')
    jday   = f_ssfr['jday'][...]
    sza    = f_ssfr['sza'][...]
    lon    = f_ssfr['lon'][...]
    lat    = f_ssfr['lat'][...]
    pit    = f_ssfr['pitch'][...]
    rol    = f_ssfr['roll'][...]

    logic = (jday>=jday_himawari_all[0]) & (jday<=jday_himawari_all[-1]) & \
            (np.logical_not(np.isnan(jday))) & (np.logical_not(np.isnan(sza))) & \
            check_continuity(lon) & check_continuity(lat)

    flt_trk = {}
    flt_trk['jday'] = jday[logic]
    flt_trk['lon']  = lon[logic]
    flt_trk['lat']  = lat[logic]
    flt_trk['sza']  = sza[logic]
    flt_trk['tmhr'] = f_ssfr['tmhr'][...][logic]
    flt_trk['alt']  = f_ssfr['alt'][...][logic]/1000.0

    flt_trk['f-up_ssfr']   = f_ssfr['nad_flux'][...][:, np.argmin(np.abs(f_ssfr['nad_wvl'][...]-wavelength))][logic]
    flt_trk['f-down_ssfr'] = f_ssfr['zen_flux'][...][:, np.argmin(np.abs(f_ssfr['zen_wvl'][...]-wavelength))][logic]
    logic_turn = (np.abs(pit[logic])>5.0) | (np.abs(rol[logic])>5.0)
    flt_trk['f-up_ssfr'][logic_turn]   = np.nan
    flt_trk['f-down_ssfr'][logic_turn] = np.nan

    if spns:
        flt_trk['f-down-diffuse_spns']= f_ssfr['spns_dif_flux'][:, np.argmin(np.abs(f_ssfr['spns_wvl'][...]-wavelength))][logic]
        flt_trk['f-down_spns']= f_ssfr['spns_tot_flux'][:, np.argmin(np.abs(f_ssfr['spns_wvl'][...]-wavelength))][logic]
    f_ssfr.close()

    indices = np.where((jday_himawari_all>=(flt_trk['jday'].min()-20.0/1440.0)) & (jday_himawari_all<=(flt_trk['jday'].max()+20.0/1440.0)))[0]
    fnames_himawari = [fnames_himawari_all[i] for i in indices]
    jday_himawari   = jday_himawari_all[indices]

    extent_img = [int(flt_trk['lon'].min()-0.5), int(flt_trk['lon'].max()+0.5)+1, int(flt_trk['lat'].min()-0.5), int(flt_trk['lat'].max()+0.5)+1]

    if extent_img[2] < 6:
        extent_img[3] = extent_img[3] + (6-extent_img[2])
        extent_img[2] = 6

    fnames_img = []
    for fname in fnames_himawari:
        filename = os.path.basename(fname)
        strings  = filename.split('_')
        date_s0   = strings[2]
        time_s0   = strings[3]
        dtime_s0  = date_s0+time_s0
        dtime0 = datetime.datetime.strptime(dtime_s0, '%Y%m%d%H%M')
        fnames_img.append(gen_himawari_true_color_image(dtime0, extent_img, fdir='data/him_img', run=run_plt))

    # partition flight track
    flt_trks = partition_flight_track(flt_trk, tmhr_interval=0.05, margin_x=1.0, margin_y=1.0)

    # sat_imgs
    sat_imgs = []
    for i in range(len(flt_trks)):
        sat_img = {}

        index0  = np.argmin(np.abs(jday_himawari-flt_trks[i]['jday0']))
        sat_img['fname']  = fnames_himawari[index0]
        sat_img['extent'] = flt_trks[i]['extent']
        sat_img['extent_img'] = extent_img
        sat_img['fname_img']  = fnames_img[index0]

        sat_imgs.append(sat_img)
    # ==================================================================================================

    sim0 = flt_sim(date=date, wavelength=wavelength, flt_trks=flt_trks, sat_imgs=sat_imgs, fname='data/flt_sim_%09.4fnm_%s.pk' % (wavelength, date_s), overwrite=True, overwrite_rtm=run_rtm)

    # os.system('rm -rf %s' % fdir)

def plot_video_frame(statements, test=False):

    colors = OrderedDict()
    colors['SSFR']           = 'black'
    colors['SPN-S Diffuse']  = 'gray'
    colors['RTM 3D']         = 'red'
    colors['RTM IPA']        = 'blue'
    colors['RTM 3D Diffuse'] = 'green'
    colors['Altitude']       = 'orange'
    colors['COT']            = 'purple'


    flt_sim0, index_trk, index_pnt, n = statements

    tmhr_length  = 0.5
    tmhr_current = flt_sim0.flt_trks[index_trk]['tmhr'][index_pnt]
    tmhr_past    = tmhr_current-tmhr_length

    cot_min = 0
    cot_max = 30
    cot_cmap = mpl.cm.get_cmap('Greys_r')
    cot_norm = mpl.colors.Normalize(vmin=cot_min, vmax=cot_max)

    fig = plt.figure(figsize=(15, 5))

    gs = gridspec.GridSpec(2, 11)

    ax = fig.add_subplot(gs[:, :])
    ax.axis('off')


    ax_map = fig.add_subplot(gs[:, :4])
    divider = make_axes_locatable(ax_map)
    ax_sza = divider.append_axes('right', size='5%', pad=0.0)

    ax_fdn = fig.add_subplot(gs[0, 5:])
    ax_fup = fig.add_subplot(gs[1, 5:])

    ax_all = fig.add_subplot(gs[:, 5:])
    ax_all.axis('off')
    ax_alt = ax_all.twinx()
    ax_cot = ax_all.twinx()


    fig.subplots_adjust(hspace=0.0, wspace=1.0)

    for itrk in range(index_trk+1):

        flt_trk      = flt_sim0.flt_trks[itrk]
        sat_img      = flt_sim0.sat_imgs[itrk]

        vnames_flt = flt_trk.keys()
        vnames_sat = sat_img.keys()

        if itrk == index_trk:
            alpha = 0.9

            if 'cot' in vnames_flt:
                cot0 = flt_trk['cot'][index_pnt]
                # color0 = cot_cmap(cot_norm(cot0))
                # ax_map.scatter(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt], facecolor=color0, s=50, lw=1.0, edgecolor=colors['COT'], zorder=3)
                ax_map.scatter(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt], facecolor=colors['COT'], s=40, lw=0.0, zorder=3)
                ax_map.text(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt]-0.28, 'COT %.2f' % cot0, color=colors['COT'], ha='center', va='center', fontsize=12, zorder=4)
                ax_cot.fill_between(flt_trk['tmhr'][:index_pnt+1], flt_trk['cot'][:index_pnt+1], facecolor=colors['COT'], alpha=0.25, lw=0.0, zorder=1)

            if 'f-up_mca-ipa' in vnames_flt:
                ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_mca-ipa'][:index_pnt+1], c=colors['RTM IPA'], s=4, lw=0.0, zorder=1)
            if 'f-up_mca-3d' in vnames_flt:
                ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_mca-3d'][:index_pnt+1] , c=colors['RTM 3D'] , s=4, lw=0.0, zorder=2)

            if 'f-down_mca-ipa' in vnames_flt:
                ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_mca-ipa'][:index_pnt+1], c=colors['RTM IPA'], s=4, lw=0.0, zorder=1)
            if 'f-down_mca-3d' in vnames_flt:
                ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_mca-3d'][:index_pnt+1] , c=colors['RTM 3D'] , s=4, lw=0.0, zorder=2)
            if 'f-down-diffuse_mca-3d' in vnames_flt:
                ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-diffuse_mca-3d'][:index_pnt+1], c=colors['RTM 3D Diffuse'], s=2, lw=0.0, zorder=3)

            logic_red = (flt_trk['tmhr'][:index_pnt]>tmhr_past) & (flt_trk['tmhr'][:index_pnt]<=tmhr_current)
            logic_green = np.logical_not(logic_red)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_red]  , flt_trk['lat'][:index_pnt][logic_red], c='r', s=2 , lw=0.0, zorder=2)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_green], flt_trk['lat'][:index_pnt][logic_green], c='g', s=2 , lw=0.0, zorder=2)

            ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_ssfr'][:index_pnt+1]   , c=colors['SSFR']   , s=4, lw=0.0, zorder=3)
            ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_ssfr'][:index_pnt+1]   , c=colors['SSFR']   , s=4, lw=0.0, zorder=5)
            ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-diffuse_spns'][:index_pnt+1]  , c=colors['SPN-S Diffuse'] , s=2, lw=0.0, zorder=4)
            ax_alt.fill_between(flt_trk['tmhr'][:index_pnt+1], flt_trk['alt'][:index_pnt+1], facecolor=colors['Altitude'], alpha=0.35, lw=1.0, zorder=0)

            if ('fname_img' in vnames_sat) and ('extent_img' in vnames_sat):
                img = mpimg.imread(sat_img['fname_img'])
                ax_map.imshow(img, extent=sat_img['extent_img'], origin='upper', aspect='auto', zorder=0)
                region = sat_img['extent_img']

            # if 'cot' in vnames_sat:
            #     ax_map.imshow(sat_img['cot'].T, extent=sat_img['extent'], cmap='Greys_r', origin='lower', vmin=cot_min, vmax=cot_max, alpha=alpha, aspect='auto', zorder=1)

        else:
            alpha = 0.4

            logic = (flt_trk['tmhr']<tmhr_current) & (flt_trk['tmhr']>=tmhr_past)

            if logic.sum() > 0:

                if 'cot' in vnames_flt:
                    ax_cot.fill_between(flt_trk['tmhr'], flt_trk['cot'], facecolor=colors['COT'], alpha=0.25, lw=0.0, zorder=1)

                if 'f-up_mca-ipa' in vnames_flt:
                    ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-up_mca-ipa'], c=colors['RTM IPA'], s=4, lw=0.0, zorder=1)
                if 'f-up_mca-3d' in vnames_flt:
                    ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-up_mca-3d'] , c=colors['RTM 3D'] , s=4, lw=0.0, zorder=2)

                if 'f-down_mca-ipa' in vnames_flt:
                    ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down_mca-ipa']       , c=colors['RTM IPA']       , s=4, lw=0.0, zorder=1)
                if 'f-down_mca-3d' in vnames_flt:
                    ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down_mca-3d']        , c=colors['RTM 3D']        , s=4, lw=0.0, zorder=2)
                if 'f-down-diffuse_mca-3d' in vnames_flt:
                    ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down-diffuse_mca-3d'], c=colors['RTM 3D Diffuse'], s=2, lw=0.0, zorder=3)

                ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-up_ssfr']            , c=colors['SSFR']          , s=4, lw=0.0, zorder=3)
                ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down_ssfr']          , c=colors['SSFR']          , s=4, lw=0.0, zorder=5)
                ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down-diffuse_spns']  , c=colors['SPN-S Diffuse'] , s=2, lw=0.0, zorder=4)

                ax_alt.fill_between(flt_trk['tmhr'], flt_trk['alt'], facecolor=colors['Altitude'], alpha=0.35, lw=1.0, zorder=0)

            logic_red = logic.copy()
            logic_green = np.logical_not(logic_red)
            ax_map.scatter(flt_trk['lon'][logic_red]  , flt_trk['lat'][logic_red], c='r', s=2 , lw=0.0, zorder=2)
            ax_map.scatter(flt_trk['lon'][logic_green], flt_trk['lat'][logic_green], c='g', s=2 , lw=0.0, zorder=2)

    ax_fup.axvline(flt_trk['tmhr'][index_pnt], lw=1.0, color='gray')
    ax_fdn.axvline(flt_trk['tmhr'][index_pnt], lw=1.0, color='gray')

    dtime0 = datetime.datetime(1, 1, 1) + datetime.timedelta(days=flt_trk['jday'][index_pnt]-1)
    fig.suptitle('%s (%.2f nm)' % (dtime0.strftime('%Y-%m-%d %H:%M:%S'), flt_sim0.wvl), y=1.02, fontsize=20)

    # map plot settings
    # =======================================================================================================
    ax_map.set_xlim(region[:2])
    ax_map.set_ylim(region[2:])
    ax_map.set_xlabel('Longitude [$^\circ$]')
    ax_map.set_ylabel('Latitude [$^\circ$]')
    # =======================================================================================================

    # sun elevation plot settings
    # =======================================================================================================
    ax_sza.set_ylim((0.0, 90.0))
    ax_sza.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 90.1, 30.0)))
    ax_sza.axhline(90.0-flt_trk['sza'][index_pnt], lw=2.0, color='r')
    ax_sza.xaxis.set_ticks([])
    ax_sza.yaxis.tick_right()
    ax_sza.yaxis.set_label_position('right')
    ax_sza.set_ylabel('Sun Elevation [$^\circ$]', rotation=270.0, labelpad=18)
    # =======================================================================================================

    # upwelling flux plot settings
    # =======================================================================================================
    ax_fup.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_fup.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_fup.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_fup.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 1.1, 0.5)))
    ax_fup.set_ylim((0.0, 1.5))
    ax_fup.set_xlabel('UTC [hour]')
    ax_fup.set_ylabel('$F_\\uparrow [\mathrm{W m^{-2} nm^{-1}}]$')

    ax_fup.set_zorder(ax_alt.get_zorder()+1)
    ax_fup.set_zorder(ax_cot.get_zorder()+2)
    ax_fup.patch.set_visible(False)
    # =======================================================================================================

    # downwelling flux plot settings
    # =======================================================================================================
    ax_fdn.xaxis.set_ticks([])
    ax_fdn.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_fdn.set_ylim((0.0, 3.0))
    ax_fdn.set_ylabel('$F_\downarrow [\mathrm{W m^{-2} nm^{-1}}]$')
    ax_fdn.set_zorder(ax_alt.get_zorder()+1)
    ax_fdn.set_zorder(ax_cot.get_zorder()+2)
    ax_fdn.patch.set_visible(False)
    # =======================================================================================================

    # altitude plot settings
    # =======================================================================================================
    ax_all.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))

    ax_alt.set_frame_on(True)
    for spine in ax_alt.spines.values():
        spine.set_visible(False)
    ax_alt.spines['right'].set_visible(True)

    ax_alt.xaxis.set_ticks([])
    ax_alt.set_ylim((0.0, 10.0))
    ax_alt.yaxis.tick_right()
    ax_alt.yaxis.set_label_position('right')
    ax_alt.set_ylabel('Altitude [km]', rotation=270.0, labelpad=8)
    # =======================================================================================================

    # cot plot settings
    # =======================================================================================================
    ax_cot.spines['right'].set_position(('axes', 1.1))
    ax_cot.set_frame_on(True)
    for spine in ax_cot.spines.values():
        spine.set_visible(False)
    ax_cot.spines['right'].set_visible(True)

    ax_cot.set_ylim((cot_min, cot_max))
    ax_cot.xaxis.set_ticks([])
    ax_cot.yaxis.tick_right()
    ax_cot.yaxis.set_label_position('right')
    ax_cot.set_ylabel('Cloud Optical Thickness', rotation=270.0, labelpad=18)
    # =======================================================================================================

    # legend plot settings
    # =======================================================================================================
    patches_legend = []
    for key in colors.keys():
        patches_legend.append(mpatches.Patch(color=colors[key], label=key))
    ax.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=16)
    # =======================================================================================================

    plt.savefig('tmp-graph/%5.5d.png' % n, bbox_inches='tight')
    if test:
        plt.show()
    else:
        plt.close(fig)

def create_video(date, wavelength=532.0):

    date_s = date.strftime('%Y%m%d')

    fdir = 'tmp-graph'
    if os.path.exists(fdir):
        os.system('rm -rf %s' % fdir)
    os.makedirs(fdir)

    flt_sim0 = flt_sim(fname='data/flt_sim_%.2fnm_%s.pk' % (wavelength, date_s), overwrite=False)

    Ntrk        = len(flt_sim0.flt_trks)
    indices_trk = np.array([], dtype=np.int32)
    indices_pnt = np.array([], dtype=np.int32)
    for itrk in range(Ntrk):
        indices_trk = np.append(indices_trk, np.repeat(itrk, flt_sim0.flt_trks[itrk]['tmhr'].size))
        indices_pnt = np.append(indices_pnt, np.arange(flt_sim0.flt_trks[itrk]['tmhr'].size))

    Npnt        = indices_trk.size
    indices     = np.arange(Npnt)

    interval = 10
    indices_trk = indices_trk[::interval]
    indices_pnt = indices_pnt[::interval]
    indices     = indices[::interval]

    statements = zip([flt_sim0]*indices_trk.size, indices_trk, indices_pnt, indices)

    with mp.Pool(processes=15) as pool:
        r = list(tqdm(pool.imap(plot_video_frame, statements), total=indices_trk.size))

    # make video
    os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "tmp-graph/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p video/camp2ex_%s.mp4' % date_s)

def save_h5(date,
        wavelength=532.0,
        vnames=['jday', 'lon', 'lat', 'sza', \
            'tmhr', 'alt', 'f-up_ssfr', 'f-down_ssfr', 'f-down-diffuse_spns', 'f-down_spns', \
            'cot', 'cer', 'cth', \
            'f-down_mca-3d', 'f-down-diffuse_mca-3d', 'f-down-direct_mca-3d', 'f-up_mca-3d',\
            'f-down_mca-3d-alt-all', 'f-down-diffuse_mca-3d-alt-all', 'f-down-direct_mca-3d-alt-all', 'f-up_mca-3d-alt-all',\
            'f-down_mca-ipa', 'f-down-diffuse_mca-ipa', 'f-down-direct_mca-ipa', 'f-up_mca-ipa']):

    date_s = date.strftime('%Y%m%d')

    fname      = 'data/flt_sim_%09.4fnm_%s.pk' % (wavelength, date_s)
    flt_sim0   = flt_sim(fname=fname)

    fname_h5   = fname.replace('.pk', '.h5')
    f = h5py.File(fname_h5, 'w')

    for vname in vnames:

        if 'alt-all' in vname:


            for i in range(len(flt_sim0.flt_trks)):

                flt_trk = flt_sim0.flt_trks[i]

                if i == 0:
                    if vname in flt_trk.keys():
                        data0 = flt_trk[vname]
                    else:
                        data0 = np.repeat(np.nan, 21*flt_trk['jday'].size).reshape((-1, 21))
                else:
                    if vname in flt_trk.keys():
                        data0 = np.vstack((data0, flt_trk[vname]))
                    else:
                        data0 = np.vstack((data0, np.repeat(np.nan, 21*flt_trk['jday'].size).reshape((-1, 21))))

        else:
            data0 = np.array([], dtype=np.float64)
            for flt_trk in flt_sim0.flt_trks:
                if vname in flt_trk.keys():
                    data0 = np.append(data0, flt_trk[vname])
                else:
                    data0 = np.append(data0, np.repeat(np.nan, flt_trk['jday'].size))

        f[vname] = data0

    f.close()

    # fname_des = '/data/hong/share/%s' % os.path.basename(fname_h5)
    # os.system('cp %s %s' % (fname_h5, fname_des))
    # os.system('chmod 777 %s' % fname_des)

def test(date):

    # Test
    flt_sim0   = flt_sim(fname='data/flt_sim_532.00nm_20190904.pk')


    # statements = [flt_sim0, 110, 115, 19920]
    # plot_video_frame(statements, test=True)


if __name__ == '__main__':

    date = datetime.datetime(2019, 9, 27)
    sim0 = flt_sim(date=date, wavelength=532.0, fname='data-in-field/flt_sim_532.00nm_20190927.pk', overwrite=False, overwrite_rtm=False)
    for i in range(len(sim0.flt_trks)):
        if (sim0.flt_trks[i]['tmhr'].mean()>27.0) & (sim0.flt_trks[i]['tmhr'].mean()<27.2):
            # =============================================================================
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            ax1.imshow(sim0.sat_imgs[i]['cot'].T, origin='lower', extent=sim0.sat_imgs[i]['extent'])
            ax1.scatter(sim0.flt_trks[i]['lon'], sim0.flt_trks[i]['lat'], c='r')
            # ax1.set_xlim(())
            # ax1.set_ylim(())
            # ax1.set_xlabel('')
            # ax1.set_ylabel('')
            # ax1.set_title('')
            # plt.savefig('test.png', bbox_inches='tight')
            plt.show()
            # =============================================================================

    exit()

    run_rtm=False
    run_plt=False

    dates = [
        datetime.datetime(2019, 8, 27),
        datetime.datetime(2019, 9, 19),
        datetime.datetime(2019, 8, 29),
        datetime.datetime(2019, 8, 30),
        datetime.datetime(2019, 9, 4) ,
        datetime.datetime(2019, 9, 6) ,
        datetime.datetime(2019, 9, 8) ,
        datetime.datetime(2019, 9, 13),
        datetime.datetime(2019, 9, 15),
        datetime.datetime(2019, 9, 16),
        datetime.datetime(2019, 9, 21),
        datetime.datetime(2019, 9, 23),
        datetime.datetime(2019, 9, 25),
        datetime.datetime(2019, 9, 27),
        datetime.datetime(2019, 9, 29),
        datetime.datetime(2019, 10, 1),
        datetime.datetime(2019, 10, 3),
        datetime.datetime(2019, 10, 5)]

    wavelength = 745.0

    for date in dates:
        first_run(date, run_rtm=run_rtm, run_plt=run_plt, wavelength=wavelength, spns=True)
        # create_video(date)
        save_h5(date, wavelength=wavelength)
