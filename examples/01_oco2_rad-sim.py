#!/bin/env python

import os
import sys
import glob
import pickle
import h5py
from pyhdf.SD import SD, SDC
import numpy as np
import datetime
import time
from scipy.io import readsav
from scipy import interpolate
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g, abs_oco_idl
from er3t.pre.cld import cld_sat
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie # newly added for phase function
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_09a1, grid_modis_by_extent, grid_modis_by_lonlat, download_modis_https, get_sinusoidal_grid_tag, get_filename_tag, download_modis_rgb
from er3t.util.oco2 import oco2_std, download_oco2_https
from er3t.util import cal_r_twostream

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng
from er3t.rtm.mca_v010 import mca_sca # newly added for phase function



class satellite_download:

    def __init__(
            self,
            date=None,
            extent=None,
            fname=None,
            fdir_out='data',
            overwrite=False,
            quiet=False,
            verbose=False):

        self.date     = date
        self.extent   = extent
        self.fdir_out = fdir_out
        self.quiet    = quiet
        self.verbose  = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((date is not None) and (extent is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((date is not None) and (extent is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run()
            self.dump(fname)

        elif (((date is not None) and (extent is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [satellite_download]: Please check if \'%s\' exists or provide \'date\' and \'extent\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'fnames') and hasattr(obj, 'extent') and hasattr(obj, 'fdir_out') and hasattr(obj, 'date'):
                if self.verbose:
                    print('Message [satellite_download]: Loading %s ...' % fname)
                self.date     = obj.date
                self.extent   = obj.extent
                self.fnames   = obj.fnames
                self.fdir_out = obj.fdir_out
            else:
                sys.exit('Error   [satellite_download]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, run=True):

        lon = np.array(self.extent[:2])
        lat = np.array(self.extent[2:])

        self.fnames = {}

        self.fnames['mod_rgb'] = [download_modis_rgb(self.date, self.extent, fdir=self.fdir_out, which='aqua', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02_1km'] = []
        self.fnames['mod_02_hkm'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        filename_tags_03 = get_filename_tag(self.date, lon, lat, satID='aqua')
        for filename_tag in filename_tags_03:
            fnames_l2 = download_modis_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02_1km = download_modis_https(self.date, '61/MYD021KM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02_hkm = download_modis_https(self.date, '61/MYD02HKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02 = download_modis_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_03 = download_modis_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02_1km'] += fnames_02_1km
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03

        # MOD09A1 surface reflectance product
        self.fnames['mod_09'] = []
        filename_tags_09 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = download_modis_https(self.date, '6/MOD09A1', filename_tag, day_interval=8, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_09'] += fnames_09

        # OCO2 std and met file
        self.fnames['oco_std'] = []
        self.fnames['oco_met'] = []
        self.fnames['oco_l1b'] = []
        for filename_tag in filename_tags_03:
            dtime = datetime.datetime.strptime(filename_tag, 'A%Y%j.%H%M') + datetime.timedelta(minutes=7.0)
            fnames_std = download_oco2_https(dtime, 'OCO2_L2_Standard.10r', fdir_out=self.fdir_out, run=run)
            fnames_met = download_oco2_https(dtime, 'OCO2_L2_Met.10r'     , fdir_out=self.fdir_out, run=run)
            fnames_l1b = download_oco2_https(dtime, 'OCO2_L1B_Science.10r', fdir_out=self.fdir_out, run=run)
            self.fnames['oco_std'] += fnames_std
            self.fnames['oco_met'] += fnames_met
            self.fnames['oco_l1b'] += fnames_l1b

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



def para_corr(lon0, lat0, vza, vaa, cld_h, sfc_h, R_earth=6378000.0, verbose=True):

    """
    Parallax correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    vza : sensor zenith angle [degree]
    vaa : sensor azimuth angle [degree]
    cld_h: cloud height [meter]
    sfc_h: surface height [meter]
    """

    if verbose:
        print('Message [para_corr]: Please make sure the units of \'cld_h\' and \'sfc_h\' are in \'meter\'.')

    dist = (cld_h-sfc_h)*np.tan(np.deg2rad(vza))

    delta_lon = dist*np.sin(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0
    delta_lat = dist*np.cos(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat

def wind_corr(lon0, lat0, u, v, dt, R_earth=6378000.0, verbose=True):

    """
    Wind correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    u   : meridional wind [meter/second], positive when eastward
    v   : zonal wind [meter/second], positive when northward
    dt  : delta time [second]
    """

    if verbose:
        print('Message [wind_corr]: Please make sure the units of \'u\' and \'v\' are in \'meter/second\' and \'dt\' in \'second\'.')
        print('Message [wind_corr]: U: %.4f m/s; V: %.4f m/s; Time offset: %.2f s' % (u, v, dt))

    delta_lon = (u*dt) / (np.pi*R_earth) * 180.0
    delta_lat = (v*dt) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat


def pre_cld(sat, cth=None, scale_factor=1.0, solver='3D'):

    # retrieve 1. cloud top height; 2. sensor zenith; 3. sensor azimuth for MODIS L1B (250nm) data from MODIS L2 (5km resolution)
    # ===================================================================================
    modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent, vnames=['Sensor_Zenith', 'Sensor_Azimuth', 'Cloud_Top_Height'])
    logic_cth  = (modl2.data['cloud_top_height']['data']>0.0)
    lon0       = modl2.data['lon_5km']['data']
    lat0       = modl2.data['lat_5km']['data']
    cth0       = modl2.data['cloud_top_height']['data']/1000.0 # units: km

    mod03      = modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    logic_sfh  = (mod03.data['height']['data']>0.0)
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sfh1       = mod03.data['height']['data']/1000.0 # units: km
    sza1       = mod03.data['sza']['data']
    saa1       = mod03.data['saa']['data']
    vza1       = mod03.data['vza']['data']
    vaa1       = mod03.data['vaa']['data']

    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, ref_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)
    lon_2d, lat_2d, rad_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)

    a0         = np.median(ref_2d)
    mu0        = np.cos(np.deg2rad(sza1.mean()))

    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)

    threshold  = a0 * 2.0
    indices    = np.where(ref_2d>threshold)
    indices_x  = indices[0]
    indices_y  = indices[1]
    lon        = lon_2d[indices_x, indices_y]
    lat        = lat_2d[indices_x, indices_y]


    # parallax correction
    # ====================================================================================================
    if cth is None:
        points     = np.transpose(np.vstack((lon0[logic_cth], lat0[logic_cth])))
        cth        = interpolate.griddata(points, cth0[logic_cth], (lon, lat), method='nearest')

        cth_2d = np.zeros_like(lon_2d)
        cth_2d[indices_x, indices_y] = cth
        modl1b.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d)

    points     = np.transpose(np.vstack((lon1[logic_sfh], lat1[logic_sfh])))
    sfh        = interpolate.griddata(points, sfh1[logic_sfh], (lon, lat), method='nearest')

    points     = np.transpose(np.vstack((lon1, lat1)))
    vza        = interpolate.griddata(points, vza1, (lon, lat), method='nearest')
    vaa        = interpolate.griddata(points, vaa1, (lon, lat), method='nearest')

    if solver == '3D':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, cth*1000.0, sfh*1000.0)
    elif solver == 'IPA':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    # ====================================================================================================


    # wind correction
    # ====================================================================================================
    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    utc_oco_byte = f['SoundingGeometry/sounding_time_string'][...][logic]
    f.close()
    utc_oco = np.zeros(utc_oco_byte.size, dtype=np.float64)
    for i, utc_oco_byte0 in enumerate(utc_oco_byte):
        utc_oco_str0 = utc_oco_byte0.decode('utf-8').split('.')[0]
        utc_oco[i] = (datetime.datetime.strptime(utc_oco_str0, '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1993, 1, 1)).total_seconds()

    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]

    f = h5py.File(sat.fnames['oco_met'][0], 'r')
    lon_oco_met = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_met = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_met>=sat.extent[0]) & (lon_oco_met<=sat.extent[1]) & (lat_oco_met>=sat.extent[2]) & (lat_oco_met<=sat.extent[3])
    u_oco = f['Meteorology/windspeed_u_met'][...][logic]
    v_oco = f['Meteorology/windspeed_v_met'][...][logic]
    f.close()

    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.median(u_oco), np.median(v_oco), utc_oco.mean()-utc_mod.mean())
    # ====================================================================================================

    lon_1d = lon_2d[:, 0]
    indices_x_new = np.int_(np.round((lon_corr-lon_1d[0])/(((lon_1d[1:]-lon_1d[:-1])).mean()), decimals=0))
    lat_1d = lat_2d[0, :]
    indices_y_new = np.int_(np.round((lat_corr-lat_1d[0])/(((lat_1d[1:]-lat_1d[:-1])).mean()), decimals=0))

    Nx, Ny = ref_2d.shape
    cot_2d_l1b = np.zeros_like(ref_2d)
    cer_2d_l1b = np.zeros_like(ref_2d); cer_2d_l1b[...] = 1.0
    for i in range(indices_x.size):
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
            cer_2d_l1b[indices_x_new[i], indices_y_new[i]] = 12.0

    modl1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
    modl1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
    modl1b.data['rad_2d'] = dict(name='Gridded radiance'                , units='W/m^2/nm/sr', data=rad_2d)
    modl1b.data['ref_2d'] = dict(name='Gridded reflectance'             , units='N/A'        , data=ref_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b*scale_factor)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d_l1b)

    return modl1b

def cal_mca_rad_oco2_old(date, tag, sat, wavelength, fname_idl=None, cth=None, photons=2e10, scale_factor=1.0, sfc_scale=True, sfc_replace=True, fdir='tmp-data', solver='3D', overwrite=True):

    """
    Calculate OCO2 radiance using cloud (MODIS level 1b) and surface properties (MOD09A1) from MODIS
    """

    # atm object
    # =================================================================================
    levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object, in the future, we will implement OCO2 MET file for this
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_oco_idl(wavelength=wavelength, fname=fname_abs, fname_idl=fname_idl, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sfc object
    # =================================================================================
    mod09     = pre_sfc(sat, tag, scale=sfc_scale, replace=sfc_replace)

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_sca object (newly added for phase function)
    # =================================================================================
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    modl1b    = pre_cld(sat, cth=cth, scale_factor=scale_factor, solver=solver)
    fname_cld = '%s/cld.pk' % fdir

    if cth is None:
        cth0 = modl1b.data['cth_2d']['data']
    else:
        cth0 = cth
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=0.5, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================


    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    sza = f['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
    saa = f['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
    vza = f['SoundingGeometry/sounding_zenith'][...][logic].mean()
    vaa = f['SoundingGeometry/sounding_azimuth'][...][logic].mean()
    f.close()


    # run mcarats
    mca0 = mcarats_ng(
            date=date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sfc_2d=sfc_2d,
            sca=sca, # newly added for phase function
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = sza,
            solar_azimuth_angle  = saa,
            sensor_zenith_angle  = vza,
            sensor_azimuth_angle = vaa,
            fdir='%s/%.4fnm/oco2/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=photons,
            solver=solver,
            Ncpu=8,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-oco2-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    oco_std0 = oco2_std(fnames=sat.fnames['oco_std'], extent=sat.extent)

    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    cs = ax1.imshow(modl1b.data['rad_2d']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3, extent=sat.extent, zorder=0)
    ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
    ax1.set_xlabel('Longitude [$^\circ$]')
    ax1.set_ylabel('Latitude [$^\circ$]')
    ax1.set_xlim(sat.extent[:2])
    ax1.set_ylim(sat.extent[2:])
    ax1.set_title('MODIS Chanel 1')

    cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
    ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
    ax2.set_xlabel('Longitude [$^\circ$]')
    ax2.set_ylabel('Latitude [$^\circ$]')
    ax2.set_xlim(sat.extent[:2])
    ax2.set_ylim(sat.extent[2:])
    ax2.set_title('MCARaTS %s' % solver)
    plt.subplots_adjust(hspace=0.5)
    if cth is not None:
        plt.savefig('%s/mca-out-rad-modis-%s_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
    else:
        plt.savefig('%s/mca-out-rad-modis-%s_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------


    if solver.lower() == 'ipa':

        sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=False)
        sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)

        cld0    = cld_sat(fname=fname_cld, overwrite=False)
        cld0.lay['extinction']['data'][...] = 0.0
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
        atm_3ds = [atm3d0]

        # run mcarats
        mca0 = mcarats_ng(
                date=date,
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                sfc_2d=sfc_2d,
                Ng=abs0.Ng,
                target='radiance',
                solar_zenith_angle   = sza,
                solar_azimuth_angle  = saa,
                sensor_zenith_angle  = vza,
                sensor_azimuth_angle = vaa,
                fdir='%s/%.4fnm/oco2/rad_%s0' % (fdir, wavelength, solver.lower()),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=photons,
                solver=solver,
                Ncpu=8,
                mp_mode='py',
                overwrite=overwrite
                )

        # mcarats output
        out0 = mca_out_ng(fname='%s/mca-out-rad-oco2-%s0_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

        # plot
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        fig = plt.figure(figsize=(12, 5.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cs = ax1.imshow(modl1b.data['rad_2d']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3, extent=sat.extent, zorder=0)
        ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_xlim(sat.extent[:2])
        ax1.set_ylim(sat.extent[2:])
        ax1.set_title('MODIS Chanel 1')

        cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
        ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax2.set_xlabel('Longitude [$^\circ$]')
        ax2.set_ylabel('Latitude [$^\circ$]')
        ax2.set_xlim(sat.extent[:2])
        ax2.set_ylim(sat.extent[2:])
        ax2.set_title('MCARaTS %s' % solver)
        plt.subplots_adjust(hspace=0.5)
        if cth is not None:
            plt.savefig('%s/mca-out-rad-modis-%s0_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
        else:
            plt.savefig('%s/mca-out-rad-modis-%s0_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
        plt.close(fig)
        # ------------------------------------------------------------------------------------------------------



def convert_photon_unit(data_photon, wavelength, scale_factor=2.0):

    c = 299792458.0
    h = 6.62607015e-34
    wavelength = wavelength * 1e-9
    data = data_photon/1000.0*c*h/wavelength*scale_factor

    return data

class oco2_rad_nadir:

    def __init__(self, sat):

        self.fname_l1b = sat.fnames['oco_l1b'][0]
        self.fname_std = sat.fnames['oco_std'][0]

        self.extent = sat.extent

        # =================================================================================
        self.cal_wvl()
        # after this, the following three functions will be created
        # Input: index, range from 0 to 7, e.g., 0, 1, 2, ..., 7
        # self.get_wvl_o2_a(index)
        # self.get_wvl_co2_weak(index)
        # self.get_wvl_co2_strong(index)
        # =================================================================================

        # =================================================================================
        self.get_index(self.extent)
        # after this, the following attributes will be created
        # self.index_s: starting index
        # self.index_e: ending index
        # =================================================================================

        # =================================================================================
        self.overlap(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.logic_l1b
        # self.lon_l1b
        # self.lat_l1b
        # =================================================================================

        # =================================================================================
        self.get_data(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.rad_o2_a
        # self.rad_co2_weak
        # self.rad_co2_strong
        # =================================================================================

    def cal_wvl(self, Nchan=1016):

        """
        Oxygen A band: centered at 765 nm
        Weak CO2 band: centered at 1610 nm
        Strong CO2 band: centered at 2060 nm
        """

        f = h5py.File(self.fname_l1b, 'r')
        wvl_coef = f['InstrumentHeader/dispersion_coef_samp'][...]
        f.close()

        Nspec, Nfoot, Ncoef = wvl_coef.shape

        wvl_o2_a       = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_weak   = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_strong = np.zeros((Nfoot, Nchan), dtype=np.float64)

        chan = np.arange(1, Nchan+1)
        for i in range(Nfoot):
            for j in range(Ncoef):
                wvl_o2_a[i, :]       += wvl_coef[0, i, j]*chan**j
                wvl_co2_weak[i, :]   += wvl_coef[1, i, j]*chan**j
                wvl_co2_strong[i, :] += wvl_coef[2, i, j]*chan**j

        wvl_o2_a       *= 1000.0
        wvl_co2_weak   *= 1000.0
        wvl_co2_strong *= 1000.0

        self.get_wvl_o2_a       = lambda index: wvl_o2_a[index, :]
        self.get_wvl_co2_weak   = lambda index: wvl_co2_weak[index, :]
        self.get_wvl_co2_strong = lambda index: wvl_co2_strong[index, :]

    def get_index(self, extent):

        if extent is None:
            self.index_s = 0
            self.index_e = None
        else:
            f = h5py.File(self.fname_l1b, 'r')
            lon_l1b     = f['SoundingGeometry/sounding_longitude'][...]
            lat_l1b     = f['SoundingGeometry/sounding_latitude'][...]

            logic = (lon_l1b>=extent[0]) & (lon_l1b<=extent[1]) & (lat_l1b>=extent[2]) & (lat_l1b<=extent[3])
            indices = np.where(np.sum(logic, axis=1)>0)[0]
            self.index_s = indices[0]
            self.index_e = indices[-1]

    def overlap(self, index_s=0, index_e=None, lat0=0.0, lon0=0.0):

        f       = h5py.File(self.fname_l1b, 'r')
        if index_e is None:
            lon_l1b = f['SoundingGeometry/sounding_longitude'][...][index_s:, ...]
            lat_l1b = f['SoundingGeometry/sounding_latitude'][...][index_s:, ...]
            snd_id  = f['SoundingGeometry/sounding_id'][...][index_s:, ...]
        else:
            lon_l1b     = f['SoundingGeometry/sounding_longitude'][...][index_s:index_e, ...]
            lat_l1b     = f['SoundingGeometry/sounding_latitude'][...][index_s:index_e, ...]
            snd_id_l1b  = f['SoundingGeometry/sounding_id'][...][index_s:index_e, ...]
        f.close()

        shape    = lon_l1b.shape
        lon_l1b  = lon_l1b
        lat_l1b  = lat_l1b

        f       = h5py.File(self.fname_std, 'r')
        lon_std = f['RetrievalGeometry/retrieval_longitude'][...]
        lat_std = f['RetrievalGeometry/retrieval_latitude'][...]
        xco2_std= f['RetrievalResults/xco2'][...]
        snd_id_std = f['RetrievalHeader/sounding_id'][...]
        sfc_pres_std = f['RetrievalResults/surface_pressure_fph'][...]
        f.close()

        self.logic_l1b = np.in1d(snd_id_l1b, snd_id_std).reshape(shape)

        self.lon_l1b   = lon_l1b
        self.lat_l1b   = lat_l1b
        self.snd_id    = snd_id_l1b

        xco2      = np.zeros_like(self.lon_l1b); xco2[...] = np.nan
        sfc_pres  = np.zeros_like(self.lon_l1b); sfc_pres[...] = np.nan

        for i in range(xco2.shape[0]):
            for j in range(xco2.shape[1]):
                logic = (snd_id_std==snd_id_l1b[i, j])
                if logic.sum() == 1:
                    xco2[i, j] = xco2_std[logic]
                    sfc_pres[i, j] = sfc_pres_std[logic]
                elif logic.sum() > 1:
                    sys.exit('Error   [oco_rad_nadir]: More than one point is found.')

        self.xco2      = xco2
        self.sfc_pres  = sfc_pres

    def get_data(self, index_s=0, index_e=None):

        f       = h5py.File(self.fname_l1b, 'r')
        if index_e is None:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:, ...]
        else:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:index_e, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:index_e, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:index_e, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:index_e, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:index_e, ...]

        for i in range(8):
            self.rad_o2_a[:, i, :]       = convert_photon_unit(self.rad_o2_a[:, i, :]      , self.get_wvl_o2_a(i))
            self.rad_co2_weak[:, i, :]   = convert_photon_unit(self.rad_co2_weak[:, i, :]  , self.get_wvl_co2_weak(i))
            self.rad_co2_strong[:, i, :] = convert_photon_unit(self.rad_co2_strong[:, i, :], self.get_wvl_co2_strong(i))
        f.close()



def australia_case(band_tag):

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2015, 12, 6)
    extent = [144.0, 146.0, -27.0, -25.0]

    name_tag = 'australia_%s' % date.strftime('%Y%m%d')
    # ===============================================================

    # create data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================

    # create tmp-data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    # ===============================================================

    # read out wavelength information from Sebastian's absorption file
    # ===============================================================
    fname_idl = 'data/atm_abs_%s_11.out' % band_tag
    f = readsav(fname_idl)
    wvls = f.lamx*1000.0
    # ===============================================================

    # run calculations for each wavelength
    # ===============================================================
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            cal_mca_rad_oco2(date, band_tag, sat0, wavelength, fname_idl=fname_idl, cth=None, scale_factor=1.0, sfc_scale=True, sfc_replace=True, fdir=fdir_tmp, solver=solver, overwrite=True, photons=1e7)
    # ===============================================================

    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    cdata_all(date, band_tag, fdir_tmp, fname_idl, sat0)
    # ===============================================================

def amazon_case(band_tag):

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2015, 6, 22)
    # extent = [-56.7, -55.7, -7.2, -3.8]
    # extent = [-56.7, -55.7, -6.0, -5.0]
    extent = [-56.6, -56.2, -4.55, -4.0]

    name_tag = 'amazon_%s' % date.strftime('%Y%m%d')
    # ===============================================================

    # create data/amazon_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================

    # create tmp-data/amazon_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    # ===============================================================

    # read out wavelength information from Sebastian's absorption file
    # ===============================================================
    fname_idl = 'data/atm_abs_%s_11.out' % band_tag
    f = readsav(fname_idl)
    wvls = f.lamx*1000.0
    # ===============================================================

    # run calculations for each wavelength
    # ===============================================================
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            if solver == 'IPA':
                photons = 1e9
            elif solver == '3D':
                photons = 2e8

            cal_mca_rad_oco2(date, band_tag, sat0, wavelength, fname_idl=fname_idl, cth=None, scale_factor=1.0, sfc_scale=True, sfc_replace=True, fdir=fdir_tmp, solver=solver, overwrite=True, photons=photons)
    # ===============================================================

    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    cdata_all(date, band_tag, fdir_tmp, fname_idl, sat0)
    # ===============================================================

def cdata_sat_case_02():

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2019, 6, 3)
    extent = [-116.2, -115.5, 40.8, 41.5]

    name_tag = 'case_02'
    # ===============================================================

    # create data/case_02 directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================

def cdata_sat_case_03():

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2019, 9, 4)
    extent = [-109.4, -108.3, 40.9, 42.0]

    name_tag = 'case_03'
    # ===============================================================

    # create data/case_03 directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================


class sat_tmp:

    def __init__(self, data):

        self.data = data


def cal_mca_rad_oco2(date, tag, sat, wavelength, fname_idl=None, cth=None, photons=2e10, scale_factor=1.0, sfc_scale=True, sfc_replace=True, fdir='tmp-data', solver='3D', overwrite=True):

    """
    Calculate OCO2 radiance using cloud (MODIS level 1b) and surface properties (MOD09A1) from MODIS
    """

    # atm object
    # =================================================================================
    levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object, in the future, we will implement OCO2 MET file for this
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_oco_idl(wavelength=wavelength, fname=fname_abs, fname_idl=fname_idl, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sfc object
    # =================================================================================
    # mod09     = pre_sfc(sat, tag, scale=sfc_scale, replace=sfc_replace)

    data = {}
    f = h5py.File('data_sat_case_01.h5', 'r')
    data['alb_2d'] = dict(data=f['mod/sfc/alb_0770'][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')
    f.close()

    fname_sfc = '%s/sfc.pk' % fdir
    mod09 = sat_tmp(data)
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================

    # mca_sca object (newly added for phase function)
    # =================================================================================
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    data = {}
    f = h5py.File('data_sat_case_01.h5', 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['mod/rad/lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['mod/rad/lat'][...])
    data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_2s'][...]*scale_factor)
    data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_l2'][...])
    data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_l2'][...])
    f.close()

    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    if cth is None:
        cth0 = modl1b.data['cth_2d']['data']
    else:
        cth0 = cth
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=1.0, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================


    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    sza = f['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
    saa = f['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
    vza = f['SoundingGeometry/sounding_zenith'][...][logic].mean()
    vaa = f['SoundingGeometry/sounding_azimuth'][...][logic].mean()
    f.close()


    # run mcarats
    mca0 = mcarats_ng(
            date=date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sfc_2d=sfc_2d,
            sca=sca, # newly added for phase function
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = sza,
            solar_azimuth_angle  = saa,
            sensor_zenith_angle  = vza,
            sensor_azimuth_angle = vaa,
            fdir='%s/%.4fnm/oco2/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=photons,
            solver=solver,
            Ncpu=14,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-oco2-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    oco_std0 = oco2_std(fnames=sat.fnames['oco_std'], extent=sat.extent)

    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if False:
        fig = plt.figure(figsize=(12, 5.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cs = ax1.imshow(modl1b.data['rad_2d']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3, extent=sat.extent, zorder=0)
        ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_xlim(sat.extent[:2])
        ax1.set_ylim(sat.extent[2:])
        ax1.set_title('MODIS Chanel 1')

        cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
        ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax2.set_xlabel('Longitude [$^\circ$]')
        ax2.set_ylabel('Latitude [$^\circ$]')
        ax2.set_xlim(sat.extent[:2])
        ax2.set_ylim(sat.extent[2:])
        ax2.set_title('MCARaTS %s' % solver)
        plt.subplots_adjust(hspace=0.5)
        if cth is not None:
            plt.savefig('%s/mca-out-rad-modis-%s_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
        else:
            plt.savefig('%s/mca-out-rad-modis-%s_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
        plt.close(fig)
    # ------------------------------------------------------------------------------------------------------


    # if solver.lower() == 'ipa':
    if False:

        sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=False)
        sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)

        cld0    = cld_sat(fname=fname_cld, overwrite=False)
        cld0.lay['extinction']['data'][...] = 0.0
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
        atm_3ds = [atm3d0]

        # run mcarats
        mca0 = mcarats_ng(
                date=date,
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                sfc_2d=sfc_2d,
                Ng=abs0.Ng,
                target='radiance',
                solar_zenith_angle   = sza,
                solar_azimuth_angle  = saa,
                sensor_zenith_angle  = vza,
                sensor_azimuth_angle = vaa,
                fdir='%s/%.4fnm/oco2/rad_%s0' % (fdir, wavelength, solver.lower()),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=photons,
                solver=solver,
                Ncpu=8,
                mp_mode='py',
                overwrite=overwrite
                )

        # mcarats output
        out0 = mca_out_ng(fname='%s/mca-out-rad-oco2-%s0_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

        # plot
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if False:
            fig = plt.figure(figsize=(12, 5.5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            cs = ax1.imshow(modl1b.data['rad_2d']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3, extent=sat.extent, zorder=0)
            ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
            ax1.set_xlabel('Longitude [$^\circ$]')
            ax1.set_ylabel('Latitude [$^\circ$]')
            ax1.set_xlim(sat.extent[:2])
            ax1.set_ylim(sat.extent[2:])
            ax1.set_title('MODIS Chanel 1')

            cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
            ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
            ax2.set_xlabel('Longitude [$^\circ$]')
            ax2.set_ylabel('Latitude [$^\circ$]')
            ax2.set_xlim(sat.extent[:2])
            ax2.set_ylim(sat.extent[2:])
            ax2.set_title('MCARaTS %s' % solver)
            plt.subplots_adjust(hspace=0.5)
            if cth is not None:
                plt.savefig('%s/mca-out-rad-modis-%s0_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
            else:
                plt.savefig('%s/mca-out-rad-modis-%s0_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
            plt.close(fig)
        # ------------------------------------------------------------------------------------------------------





def func(x, a):
    return a*x

def create_sfc_alb_2d(x_ref, y_ref, data_ref, x_bkg_2d, y_bkg_2d, data_bkg_2d, scale=True, replace=True):

    points = np.transpose(np.vstack((x_bkg_2d.ravel(), y_bkg_2d.ravel())))
    data_bkg = interpolate.griddata(points, data_bkg_2d.ravel(), (x_ref, y_ref), method='nearest')

    if scale:
        popt, pcov = curve_fit(func, data_bkg, data_ref)
        slope = popt[0]
    else:
        slope = 1.0

    data_2d = data_bkg_2d*slope

    dx = x_bkg_2d[1, 0] - x_bkg_2d[0, 0]
    dy = y_bkg_2d[0, 1] - y_bkg_2d[0, 0]

    if replace:
        indices_x = np.int_(np.round((x_ref-x_bkg_2d[0, 0])/dx, decimals=0))
        indices_y = np.int_(np.round((y_ref-y_bkg_2d[0, 0])/dy, decimals=0))
        data_2d[indices_x, indices_y] = data_ref

    return data_2d


def pre_cld_oco_backup(sat, cth=None, scale_factor=1.0, solver='3D'):

    # retrieve 1. cloud top height; 2. sensor zenith; 3. sensor azimuth for MODIS L1B (250nm) data from MODIS L2 (5km resolution)
    # ===================================================================================
    modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent, vnames=['Sensor_Zenith', 'Sensor_Azimuth', 'Cloud_Top_Height'])
    logic_cth  = (modl2.data['cloud_top_height']['data']>0.0)
    lon0       = modl2.data['lon_5km']['data']
    lat0       = modl2.data['lat_5km']['data']
    cth0       = modl2.data['cloud_top_height']['data']/1000.0 # units: km

    mod03      = modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    logic_sfh  = (mod03.data['height']['data']>0.0)
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sfh1       = mod03.data['height']['data']/1000.0 # units: km
    sza1       = mod03.data['sza']['data']
    saa1       = mod03.data['saa']['data']
    vza1       = mod03.data['vza']['data']
    vaa1       = mod03.data['vaa']['data']

    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, ref_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][1, ...], extent=sat.extent)
    lon_2d, lat_2d, rad_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][1, ...], extent=sat.extent)

    a0         = np.median(ref_2d)
    mu0        = np.cos(np.deg2rad(sza1.mean()))

    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)

    threshold  = a0 * 1.0
    indices    = np.where(ref_2d>threshold)
    indices_x  = indices[0]
    indices_y  = indices[1]
    lon        = lon_2d[indices_x, indices_y]
    lat        = lat_2d[indices_x, indices_y]


    # parallax correction
    # ====================================================================================================
    if cth is None:
        points     = np.transpose(np.vstack((lon0[logic_cth], lat0[logic_cth])))
        cth_tmp    = interpolate.griddata(points, cth0[logic_cth], (lon, lat), method='cubic')

        cth_2d = np.zeros_like(lon_2d)
        cth_2d[indices_x, indices_y] = cth_tmp

    points     = np.transpose(np.vstack((lon1[logic_sfh], lat1[logic_sfh])))
    sfh        = interpolate.griddata(points, sfh1[logic_sfh], (lon, lat), method='cubic')

    points     = np.transpose(np.vstack((lon1, lat1)))
    vza        = interpolate.griddata(points, vza1, (lon, lat), method='cubic')
    vaa        = interpolate.griddata(points, vaa1, (lon, lat), method='cubic')
    vza[...] = np.nanmean(vza)
    vaa[...] = np.nanmean(vaa)

    if solver == '3D':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, cth_tmp*1000.0, sfh*1000.0)
        # lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    elif solver == 'IPA':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    # ====================================================================================================


    # wind correction
    # ====================================================================================================
    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    utc_oco_byte = f['SoundingGeometry/sounding_time_string'][...][logic]
    f.close()
    utc_oco = np.zeros(utc_oco_byte.size, dtype=np.float64)
    for i, utc_oco_byte0 in enumerate(utc_oco_byte):
        utc_oco_str0 = utc_oco_byte0.decode('utf-8').split('.')[0]
        utc_oco[i] = (datetime.datetime.strptime(utc_oco_str0, '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1993, 1, 1)).total_seconds()

    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]

    f = h5py.File(sat.fnames['oco_met'][0], 'r')
    lon_oco_met = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_met = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_met>=sat.extent[0]) & (lon_oco_met<=sat.extent[1]) & (lat_oco_met>=sat.extent[2]) & (lat_oco_met<=sat.extent[3])
    u_oco = f['Meteorology/windspeed_u_met'][...][logic]
    v_oco = f['Meteorology/windspeed_v_met'][...][logic]
    f.close()

    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.median(u_oco), np.median(v_oco), utc_oco.mean()-utc_mod.mean())
    # ====================================================================================================

    # modis 1D retrieval from L2 product
    # =============================================================
    modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent)
    lon_2d, lat_2d, cer_2d_l2 = grid_modis_by_lonlat(modl2.data['lon']['data'], modl2.data['lat']['data'], modl2.data['cer']['data'], lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='linear')
    cer_2d_l2[cer_2d_l2<1.0] = 1.0
    # =============================================================

    lon_1d = lon_2d[:, 0]
    indices_x_new = np.int_(np.round((lon_corr-lon_1d[0])/(((lon_1d[1:]-lon_1d[:-1])).mean()), decimals=0))
    lat_1d = lat_2d[0, :]
    indices_y_new = np.int_(np.round((lat_corr-lat_1d[0])/(((lat_1d[1:]-lat_1d[:-1])).mean()), decimals=0))

    Nx, Ny = ref_2d.shape
    cot_2d_l1b = np.zeros_like(ref_2d)
    cer_2d_l1b = np.zeros_like(ref_2d); cer_2d_l1b[...] = 1.0
    for i in range(indices_x.size):
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
            cer_2d_l1b[indices_x_new[i], indices_y_new[i]] = cer_2d_l2[np.argmin(np.abs(lon_2d[:, 0]-lon_2d[indices_x[i], indices_y[i]])), np.argmin(np.abs(lat_2d[0, :]-lat_2d[indices_x[i], indices_y[i]]))]

    modl1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
    modl1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
    modl1b.data['rad_2d'] = dict(name='Gridded radiance'                , units='W/m^2/nm/sr', data=rad_2d)
    modl1b.data['ref_2d'] = dict(name='Gridded reflectance'             , units='N/A'        , data=ref_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b*scale_factor)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d_l1b)

    if cth is None:
        cth_2d_l2 = np.zeros_like(ref_2d)
        for i in range(indices_x.size):
            if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
                cth_2d_l2[indices_x_new[i], indices_y_new[i]] = cth_2d[indices_x[i], indices_y[i]]

        modl1b.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d_l2)

    return modl1b

def pre_cld_oco(sat, cth=None, scale_factor=1.0, solver='3D'):

    # retrieve 1. cloud top height; 2. sensor zenith; 3. sensor azimuth for MODIS L1B (250nm) data from MODIS L2 (5km resolution)
    # ===================================================================================
    modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent, vnames=['Sensor_Zenith', 'Sensor_Azimuth', 'Cloud_Top_Height'])
    logic_cth  = (modl2.data['cloud_top_height']['data']>0.0)
    lon0       = modl2.data['lon_5km']['data']
    lat0       = modl2.data['lat_5km']['data']
    cth0       = modl2.data['cloud_top_height']['data']/1000.0 # units: km

    mod03      = modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    logic_sfh  = (mod03.data['height']['data']>0.0)
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sfh1       = mod03.data['height']['data']/1000.0 # units: km
    sza1       = mod03.data['sza']['data']
    saa1       = mod03.data['saa']['data']
    vza1       = mod03.data['vza']['data']
    vaa1       = mod03.data['vaa']['data']

    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, ref_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)
    lon_2d, lat_2d, rad_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)

    a0         = np.median(ref_2d)
    mu0        = np.cos(np.deg2rad(sza1.mean()))

    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)

    # =============================================================================
    mod_rgb = mpl_img.imread(sat.fnames['mod_rgb'][0])

    lon_rgb0 = np.linspace(sat.extent[0], sat.extent[1], mod_rgb.shape[1]+1)
    lat_rgb0 = np.linspace(sat.extent[2], sat.extent[3], mod_rgb.shape[0]+1)
    lon_rgb = (lon_rgb0[1:]+lon_rgb0[:-1])/2.0
    lat_rgb = (lat_rgb0[1:]+lat_rgb0[:-1])/2.0

    mod_r = mod_rgb[:, :, 0]
    mod_g = mod_rgb[:, :, 1]
    mod_b = mod_rgb[:, :, 2]

    logic_rgb_nan0 = (mod_r<=(np.median(mod_r)*1.06)) |\
                     (mod_g<=(np.median(mod_g)*1.06)) |\
                     (mod_b<=(np.median(mod_b)*1.06))
    logic_rgb_nan = np.flipud(logic_rgb_nan0).T

    x0_rgb = lon_rgb[0]
    y0_rgb = lat_rgb[0]
    dx_rgb = lon_rgb[1] - x0_rgb
    dy_rgb = lat_rgb[1] - y0_rgb

    indices_x = np.int_(np.round((lon_2d-x0_rgb)/dx_rgb, decimals=0))
    indices_y = np.int_(np.round((lat_2d-y0_rgb)/dy_rgb, decimals=0))

    logic_ref_nan = logic_rgb_nan[indices_x, indices_y]

    indices    = np.where(logic_ref_nan!=1)
    indices_x  = indices[0]
    indices_y  = indices[1]
    lon        = lon_2d[indices_x, indices_y]
    lat        = lat_2d[indices_x, indices_y]
    # =============================================================================


    # cld_logic = np.ones_like(ref_2d)
    # cld_logic[ref_2d<=threshold] = np.nan

    # # =============================================================================
    # mod_rgb = mpl_img.imread(sat.fnames['mod_rgb'][0])

    # # modis 1D retrieval from L2 product
    # # =============================================================
    # modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent)
    # lon_2d, lat_2d, cot_2d_l2 = grid_modis_by_lonlat(modl2.data['lon']['data'], modl2.data['lat']['data'], modl2.data['cot']['data'], lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='nearest')
    # cot_2d_l2[cot_2d_l2<0.0] = 0.0
    # # =============================================================================

    # f = h5py.File('test.h5', 'w')
    # f['extent'] = sat.extent
    # f['rgb'] = mod_rgb
    # f['ref'] = ref_2d
    # f['cot'] = cot_2d_l2
    # f['a0'] = a0
    # f['mu0'] = mu0
    # f.close()
    # exit()
    # =============================================================================


    # parallax correction
    # ====================================================================================================
    if cth is None:
        points     = np.transpose(np.vstack((lon0[logic_cth], lat0[logic_cth])))
        cth_tmp    = interpolate.griddata(points, cth0[logic_cth], (lon, lat), method='cubic')

        cth_2d = np.zeros_like(lon_2d)
        cth_2d[indices_x, indices_y] = cth_tmp

    points     = np.transpose(np.vstack((lon1[logic_sfh], lat1[logic_sfh])))
    sfh        = interpolate.griddata(points, sfh1[logic_sfh], (lon, lat), method='cubic')

    points     = np.transpose(np.vstack((lon1, lat1)))
    vza        = interpolate.griddata(points, vza1, (lon, lat), method='cubic')
    vaa        = interpolate.griddata(points, vaa1, (lon, lat), method='cubic')
    vza[...] = np.nanmean(vza)
    vaa[...] = np.nanmean(vaa)

    if solver == '3D':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, cth_tmp*1000.0, sfh*1000.0)
        # lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    elif solver == 'IPA':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    # ====================================================================================================


    # wind correction
    # ====================================================================================================
    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    utc_oco_byte = f['SoundingGeometry/sounding_time_string'][...][logic]
    f.close()
    utc_oco = np.zeros(utc_oco_byte.size, dtype=np.float64)
    for i, utc_oco_byte0 in enumerate(utc_oco_byte):
        utc_oco_str0 = utc_oco_byte0.decode('utf-8').split('.')[0]
        utc_oco[i] = (datetime.datetime.strptime(utc_oco_str0, '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1993, 1, 1)).total_seconds()

    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]

    f = h5py.File(sat.fnames['oco_met'][0], 'r')
    lon_oco_met = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_met = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_met>=sat.extent[0]) & (lon_oco_met<=sat.extent[1]) & (lat_oco_met>=sat.extent[2]) & (lat_oco_met<=sat.extent[3])
    u_oco = f['Meteorology/windspeed_u_met'][...][logic]
    v_oco = f['Meteorology/windspeed_v_met'][...][logic]
    f.close()

    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.median(u_oco), np.median(v_oco), utc_oco.mean()-utc_mod.mean())
    # ====================================================================================================

    # modis 1D retrieval from L2 product
    # =============================================================
    modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent)
    lon_2d, lat_2d, cer_2d_l2 = grid_modis_by_lonlat(modl2.data['lon']['data'], modl2.data['lat']['data'], modl2.data['cer']['data'], lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='linear')
    cer_2d_l2[cer_2d_l2<1.0] = 1.0
    # =============================================================

    lon_1d = lon_2d[:, 0]
    indices_x_new = np.int_(np.round((lon_corr-lon_1d[0])/(((lon_1d[1:]-lon_1d[:-1])).mean()), decimals=0))
    lat_1d = lat_2d[0, :]
    indices_y_new = np.int_(np.round((lat_corr-lat_1d[0])/(((lat_1d[1:]-lat_1d[:-1])).mean()), decimals=0))

    Nx, Ny = ref_2d.shape
    cot_2d_l1b = np.zeros_like(ref_2d)
    cer_2d_l1b = np.zeros_like(ref_2d); cer_2d_l1b[...] = 1.0
    for i in range(indices_x.size):
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
            cer_2d_l1b[indices_x_new[i], indices_y_new[i]] = cer_2d_l2[np.argmin(np.abs(lon_2d[:, 0]-lon_2d[indices_x[i], indices_y[i]])), np.argmin(np.abs(lat_2d[0, :]-lat_2d[indices_x[i], indices_y[i]]))]

    cot_2d_l1b[(cer_2d_l1b<1.5)&(lat_2d>38.0)] = 0.0
    cot_2d_l1b[(cer_2d_l1b<1.5)&(lon_2d<-108.5)] = 0.0

    modl1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
    modl1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
    modl1b.data['rad_2d'] = dict(name='Gridded radiance'                , units='W/m^2/nm/sr', data=rad_2d)
    modl1b.data['ref_2d'] = dict(name='Gridded reflectance'             , units='N/A'        , data=ref_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b*scale_factor)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d_l1b)

    if cth is None:
        cth_2d_l2 = np.zeros_like(ref_2d)
        for i in range(indices_x.size):
            if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
                cth_2d_l2[indices_x_new[i], indices_y_new[i]] = cth_2d[indices_x[i], indices_y[i]]

        cth_2d_l2[(cer_2d_l1b<1.5)&(lat_2d>38.0)] = 0.0
        cth_2d_l2[(cer_2d_l1b<1.5)&(lon_2d<-108.5)] = 0.0
        modl1b.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d_l2)

    return modl1b

def pre_sfc(sat, tag, version='10r', scale=True, replace=True):

    if version == '7':
        vnames = [
                'AlbedoResults/albedo_o2_fph',              # 0.77 microns
                'AlbedoResults/albedo_slope_o2',
                'AlbedoResults/albedo_strong_co2_fph',      # 2.06 microns
                'AlbedoResults/albedo_slope_strong_co2',
                'AlbedoResults/albedo_weak_co2_fph',        # 1.615 microns
                'AlbedoResults/albedo_slope_weak_co2'
                  ]
    elif version == '10' or version == '10r':
        vnames = [
                'BRDFResults/brdf_reflectance_o2',              # 0.77 microns
                'BRDFResults/brdf_reflectance_slope_o2',
                'BRDFResults/brdf_reflectance_strong_co2',      # 2.06 microns
                'BRDFResults/brdf_reflectance_slope_strong_co2',
                'BRDFResults/brdf_reflectance_weak_co2',        # 1.615 microns
                'BRDFResults/brdf_reflectance_slope_weak_co2'
                  ]
    else:
        exit('Error   [pre_sfc]: Cannot recognize version \'%s\'.' % version)

    oco = oco2_std(fnames=sat.fnames['oco_std'], vnames=vnames, extent=sat.extent)
    oco_lon = oco.data['lon']['data']
    oco_lat = oco.data['lat']['data']

    if version == '7':

        if tag.lower() == 'o2a':
            oco_sfc_alb = oco.data['albedo_o2_fph']['data']
        elif tag.lower() == 'wco2':
            oco_sfc_alb = oco.data['albedo_weak_co2_fph']['data']
        elif tag.lower() == 'sco2':
            oco_sfc_alb = oco.data['albedo_strong_co2_fph']['data']

    elif version == '10' or version == '10r':

        if tag.lower() == 'o2a':
            oco_sfc_alb = oco.data['brdf_reflectance_o2']['data']
        elif tag.lower() == 'wco2':
            oco_sfc_alb = oco.data['brdf_reflectance_weak_co2']['data']
        elif tag.lower() == 'sco2':
            oco_sfc_alb = oco.data['brdf_reflectance_strong_co2']['data']

    else:
        exit('Error   [cdata_sfc_alb]: Cannot recognize version \'%s\'.' % version)

    logic = (oco_sfc_alb>0.0) & (oco_lon>=sat.extent[0]) & (oco_lon<=sat.extent[1]) & (oco_lat>=sat.extent[2]) & (oco_lat<=sat.extent[3])
    oco_lon = oco_lon[logic]
    oco_lat = oco_lat[logic]
    oco_sfc_alb = oco_sfc_alb[logic]

    # band 1: 620  - 670  nm, index 0
    # band 2: 841  - 876  nm, index 1
    # band 3: 459  - 479  nm, index 2
    # band 4: 545  - 565  nm, index 3
    # band 5: 1230 - 1250 nm, index 4
    # band 6: 1628 - 1652 nm, index 5
    # band 7: 2105 - 2155 nm, index 6
    mod = modis_09a1(fnames=sat.fnames['mod_09'], extent=sat.extent)

    points = np.transpose(np.vstack((mod.data['lon']['data'], mod.data['lat']['data'])))
    if tag.lower() == 'o2a':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][1, :], extent=sat.extent)
    elif tag.lower() == 'wco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][5, :], extent=sat.extent)
    elif tag.lower() == 'sco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][6, :], extent=sat.extent)

    oco_sfc_alb_2d = create_sfc_alb_2d(oco_lon, oco_lat, oco_sfc_alb, lon_2d, lat_2d, mod_sfc_alb_2d, scale=scale, replace=replace)

    mod.data['alb_2d'] = dict(data=oco_sfc_alb_2d, name='Surface albedo', units='N/A')
    mod.data['lon_2d'] = dict(data=lon_2d, name='Longitude', units='degrees')
    mod.data['lat_2d'] = dict(data=lat_2d, name='Latitude' , units='degrees')

    f = h5py.File('oco_mod_sfc.h5', 'w')
    f['alb_2d'] = oco_sfc_alb_2d
    f['lon_2d'] = lon_2d
    f['lat_2d'] = lat_2d
    f['alb_mod_2d'] = mod_sfc_alb_2d
    f['alb_oco_1d'] = oco_sfc_alb
    f['lon_oco_1d'] = oco_lon
    f['lat_oco_1d'] = oco_lat
    f.close()

    return mod

def cdata_sat_case_01(band_tag):

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2019, 9, 2)
    extent = [-109.6, -106.5, 35.9, 39.0]

    name_tag = 'case_01'
    # ===============================================================

    # create data/case_01 directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=True)
    # ===============================================================

    # ==================================================================================================
    oco = oco2_rad_nadir(sat0)

    wvl_o2a  = np.zeros_like(oco.rad_o2_a      , dtype=np.float64)
    wvl_wco2 = np.zeros_like(oco.rad_co2_weak  , dtype=np.float64)
    wvl_sco2 = np.zeros_like(oco.rad_co2_strong, dtype=np.float64)
    for i in range(oco.rad_o2_a.shape[0]):
        for j in range(oco.rad_o2_a.shape[1]):
            wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
            wvl_wco2[i, j, :] = oco.get_wvl_co2_weak(j)
            wvl_sco2[i, j, :] = oco.get_wvl_co2_strong(j)
    # ==================================================================================================


    f0 = h5py.File('data_sat_case_01.h5', 'w')
    f0['extent'] = sat0.extent

    # MODIS L1B
    # ==================================================================================================
    # 2130 nm
    # modl1b = modis_l1b(fnames=sat0.fnames['mod_02_hkm'], extent=sat0.extent)
    # lon_2d, lat_2d, rad_2d_mod = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][4, ...], extent=sat0.extent)
    # 860 nm
    modl1b = modis_l1b(fnames=sat0.fnames['mod_02'], extent=sat0.extent)
    lon_2d, lat_2d, rad_2d_mod = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][1, ...], extent=sat0.extent)

    g = f0.create_group('mod')
    g1 = g.create_group('rad')
    g1['lon'] = lon_2d
    g1['lat'] = lat_2d
    g1['rad_0860'] = rad_2d_mod
    print('haha')
    # ==================================================================================================

    # ==================================================================================================
    mod0 = pre_cld_oco(sat0)
    g2 = g.create_group('cld')
    g2['cot_2s'] = mod0.data['cot_2d']['data']
    g2['cer_l2'] = mod0.data['cer_2d']['data']
    g2['cth_l2'] = mod0.data['cth_2d']['data']
    print('haha')
    # ==================================================================================================

    # MODIS RGB
    # ==================================================================================================
    mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
    g['rgb'] = mod_rgb
    print('haha')
    # ==================================================================================================

    # ==================================================================================================
    mod_sfc = pre_sfc(sat0, 'o2a', scale=True, replace=True)

    g2 = g.create_group('sfc')
    g2['lon'] = mod_sfc.data['lon_2d']['data']
    g2['lat'] = mod_sfc.data['lat_2d']['data']
    g2['alb_0770'] = mod_sfc.data['alb_2d']['data']
    print('haha')
    # ==================================================================================================


    # OCO L1B
    # ==================================================================================================
    g = f0.create_group('oco')
    g['lon'] = oco.lon_l1b
    g['lat'] = oco.lat_l1b
    g['sza'] = oco.sza
    g['saa'] = oco.saa
    g['logic'] = oco.logic_l1b

    g['snd_id'] = oco.snd_id
    g['xco2']   = oco.xco2
    g['sfc_pres'] = oco.sfc_pres

    g1 = g.create_group('o2a')
    g1['rad']  = oco.rad_o2_a
    g1['wvl']  = wvl_o2a

    g2 = g.create_group('wco2')
    g2['rad']  = oco.rad_co2_weak
    g2['wvl']  = wvl_wco2

    g3 = g.create_group('sco2')
    g3['rad_oco']  = oco.rad_co2_strong
    g3['wvl_oco']  = wvl_sco2
    print('haha')
    # ==================================================================================================

    f0.close()

def run_er3t_case_01(band_tag):

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2019, 9, 2)
    extent = [-109.6, -106.5, 35.9, 39.0]
    name_tag = 'case_01'
    fdir_data = 'data/%s' % name_tag
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================

    # create tmp-data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    # ===============================================================

    # read out wavelength information from Sebastian's absorption file
    # ===============================================================
    fname_idl = 'data/atm_abs_%s_11.out' % band_tag
    f = readsav(fname_idl)
    wvls = f.lamx*1000.0
    # ===============================================================

    # run calculations for each wavelength
    # ===============================================================
    index = np.argmin(np.abs(wvls-770.0))
    for wavelength in [wvls[index]]:
        print(wavelength)
        # for solver in ['3D', 'IPA']:
        for solver in ['IPA']:
            if solver == 'IPA':
                photons = 1e9
            elif solver == '3D':
                photons = 2e8
            cal_mca_rad_oco2(date, band_tag, sat0, wavelength, fname_idl=fname_idl, cth=None, fdir=fdir_tmp, solver=solver, overwrite=True, photons=photons)
    # ===============================================================

def cdata_sat_case_01_single_wavelength(wvl0=768.5151):

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(2019, 9, 2)
    extent = [-109.6, -106.5, 35.9, 39.0]

    name_tag = 'case_01'
    # ===============================================================

    # create data/case_01 directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================

    # ==================================================================================================
    oco = oco2_rad_nadir(sat0)

    wvl_o2a  = np.zeros_like(oco.rad_o2_a      , dtype=np.float64)
    wvl_wco2 = np.zeros_like(oco.rad_co2_weak, dtype=np.float64)
    wvl_sco2 = np.zeros_like(oco.rad_co2_strong, dtype=np.float64)
    for i in range(oco.rad_o2_a.shape[0]):
        for j in range(oco.rad_o2_a.shape[1]):
            wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
            wvl_wco2[i, j, :] = oco.get_wvl_co2_weak(j)
            wvl_sco2[i, j, :] = oco.get_wvl_co2_strong(j)
    # ==================================================================================================


    f = h5py.File('data_oco_%9.4fnm.h5' % wvl0, 'w')
    f['extent'] = sat0.extent

    # MODIS L1B
    # ==================================================================================================
    modl1b = modis_l1b(fnames=sat0.fnames['mod_02'], extent=sat0.extent)
    lon_2d, lat_2d, rad_2d_mod = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][1, ...], extent=sat0.extent)

    g = f.create_group('mod')
    g1 = g.create_group('rad')
    g1['lon'] = lon_2d
    g1['lat'] = lat_2d
    g1['rad_0860'] = rad_2d_mod
    # ==================================================================================================

    # ==================================================================================================
    mod0 = pre_cld_oco(sat0)
    g2 = g.create_group('cld')
    g2['cot_2s'] = mod0.data['cot_2d']['data']
    g2['cer_l2'] = mod0.data['cer_2d']['data']
    g2['cth_l2'] = mod0.data['cth_2d']['data']
    # ==================================================================================================

    # MODIS RGB
    # ==================================================================================================
    mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
    g['rgb'] = mod_rgb
    # ==================================================================================================

    # ==================================================================================================
    mod_sfc = pre_sfc(sat0, 'o2a', scale=True, replace=True)

    g2 = g.create_group('sfc')
    g2['lon'] = mod_sfc.data['lon_2d']['data']
    g2['lat'] = mod_sfc.data['lat_2d']['data']
    g2['alb_0770'] = mod_sfc.data['alb_2d']['data']
    # ==================================================================================================

    # MCA
    # ==================================================================================================
    rad_mca_ipa0 = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1]), dtype=np.float64)
    rad_mca_ipa  = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1]), dtype=np.float64)
    rad_mca_3d   = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1]), dtype=np.float64)

    rad_mca_ipa0_std = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1]), dtype=np.float64)
    rad_mca_ipa_std  = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1]), dtype=np.float64)
    rad_mca_3d_std   = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1]), dtype=np.float64)

    fname = '/data/hong/work/05_er3t/01_oco-sim/tmp-data/case_01/o2a/mca-out-rad-oco2-ipa0_%.4fnm.h5' % (wvl0)
    f = h5py.File(fname, 'r')
    rad_ipa0     = f['mean/rad'][...]
    rad_ipa0_std = f['mean/rad_std'][...]
    f.close()

    fname = '/data/hong/work/05_er3t/01_oco-sim/tmp-data/case_01/o2a/mca-out-rad-oco2-ipa_%.4fnm.h5' % (wvl0)
    f = h5py.File(fname, 'r')
    rad_ipa     = f['mean/rad'][...]
    rad_ipa_std = f['mean/rad_std'][...]
    f.close()

    fname = '/data/hong/work/05_er3t/01_oco-sim/tmp-data/case_01/o2a/mca-out-rad-oco2-3d_%.4fnm.h5' % (wvl0)
    f = h5py.File(fname, 'r')
    rad_3d     = f['mean/rad'][...]
    rad_3d_std = f['mean/rad_std'][...]
    toa0       = f['mean/toa'][...]
    photons    = f['mean/N_photon'][...]
    f.close()

    for i in range(wvl_o2a.shape[0]):
        for j in range(wvl_o2a.shape[1]):
            lon0 = oco.lon_l1b[i, j]
            lat0 = oco.lat_l1b[i, j]

            index_lon = np.argmin(np.abs(lon_2d[:, 0]-lon0))
            index_lat = np.argmin(np.abs(lat_2d[0, :]-lat0))

            rad_mca_ipa0[i, j] = rad_ipa0[index_lon, index_lat]
            rad_mca_ipa[i, j]  = rad_ipa[index_lon, index_lat]
            rad_mca_3d[i, j]   = rad_3d[index_lon, index_lat]

            rad_mca_ipa0_std[i, j] = rad_ipa0_std[index_lon, index_lat]
            rad_mca_ipa_std[i, j]  = rad_ipa_std[index_lon, index_lat]
            rad_mca_3d_std[i, j]   = rad_3d_std[index_lon, index_lat]

    logic_nan = np.logical_not(oco.logic_l1b)
    rad_mca_ipa0[logic_nan] = np.nan
    rad_mca_ipa[logic_nan] = np.nan
    rad_mca_3d[logic_nan] = np.nan
    rad_mca_ipa0_std[logic_nan] = np.nan
    rad_mca_ipa_std[logic_nan] = np.nan
    rad_mca_3d_std[logic_nan] = np.nan
    g = f.create_group('mca')
    g['lon'] = oco.lon_l1b
    g['lat'] = oco.lat_l1b
    g['toa'] = toa0
    g['Np'] = photons
    g['sza'] = oco.sza[oco.logic_l1b].mean()
    g['saa'] = oco.saa[oco.logic_l1b].mean()
    g['rad_3d'] = rad_mca_3d
    g['rad_ipa'] = rad_mca_ipa
    g['rad_ipa0'] = rad_mca_ipa0
    g['rad_3d_std'] = rad_mca_3d_std
    g['rad_ipa_std'] = rad_mca_ipa_std
    g['rad_ipa0_std'] = rad_mca_ipa0_std
    g['wvl'] = wvl0
    g['rad_3d_domain'] = rad_3d
    g['rad_3d_domain_std'] = rad_3d_std

    # ==================================================================================================


    # OCO L1B
    # ==================================================================================================
    g = f.create_group('oco')
    g['lon'] = oco.lon_l1b
    g['lat'] = oco.lat_l1b
    g['sza'] = oco.sza
    g['saa'] = oco.saa
    g['logic'] = oco.logic_l1b

    g['snd_id'] = oco.snd_id
    g['xco2']   = oco.xco2
    g['sfc_pres'] = oco.sfc_pres

    g1 = g.create_group('o2a')
    g1['rad']  = oco.rad_o2_a
    g1['wvl']  = wvl_o2a

    g2 = g.create_group('wco2')
    g2['rad']  = oco.rad_co2_weak
    g2['wvl']  = wvl_wco2

    g3 = g.create_group('sco2')
    g3['rad_oco']  = oco.rad_co2_strong
    g3['wvl_oco']  = wvl_sco2
    # ==================================================================================================

    f.close()



if __name__ == '__main__':

    # cdata_sat_case_01('o2a')
    run_er3t_case_01('o2a')
    # cdata_sat_case_01_single_wavelength(wvl0=768.5151)
