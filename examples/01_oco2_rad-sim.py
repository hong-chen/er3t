"""
by Hong Chen (hong.chen.cu@gmail.com)

This code serves as an example code to reproduce 3D/IPA OCO-2 radiance simulation for App. 1 in Chen et al. (2022).

The processes include:
    1) automatically download and pre-process satellite data products (~2.2 GB data will be downloaded and
       stored under data/01_oco2_rad-sim/download) from NASA data archive
        a) MODIS-Aqua_rgb_2019-09-02_(-109.60,-106.50,35.90,39.00).png
        b) MYD02QKM.A2019245.2025.061.2019246161115.hdf
        c) MYD03.A2019245.2025.061.2019246155053.hdf
        d) MYD06_L2.A2019245.2025.061.2019246164334.hdf
        e) MYD09A1.A2019241.h09v05.006.2019250044127.hdf
        f) oco2_L1bScND_27502a_190902_B10003r_200220035234.h5
        g) oco2_L2MetND_27502a_190902_B10003r_200124030754.h5
        h) oco2_L2StdND_27502a_190902_B10004r_200226231039.h5

    2) run simulation
        a) 3D mode
        b) IPA mode

    3) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2022-10-19 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64

"""

import os
import sys
import pickle
import h5py
from pyhdf.SD import SD, SDC
import numpy as np
import datetime
from scipy.io import readsav
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
from matplotlib.ticker import FixedLocator
import matplotlib.patches as mpatches


import er3t


# global variables
#/--------------------------------------------------------------\#
name_tag = os.path.relpath(__file__).replace('.py', '')
photon_sim = 1e8
#\--------------------------------------------------------------/#


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

        lon0 = np.linspace(self.extent[0], self.extent[1], 100)
        lat0 = np.linspace(self.extent[2], self.extent[3], 100)
        lon, lat = np.meshgrid(lon0, lat0, indexing='ij')

        self.fnames = {}

        self.fnames['mod_rgb'] = [er3t.util.download_worldview_rgb(self.date, self.extent, fdir_out=self.fdir_out, satellite='aqua', instrument='modis', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_03'] = []
        self.fnames['mod_l2'] = []
        self.fnames['mod_02'] = []
        filename_tags_03 = er3t.util.get_satfile_tag(self.date, lon, lat, satellite='aqua', instrument='modis')
        for filename_tag in filename_tags_03:
            fnames_03     = er3t.util.download_laads_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_l2     = er3t.util.download_laads_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02     = er3t.util.download_laads_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)

            self.fnames['mod_03'] += fnames_03
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02'] += fnames_02

        # MOD09A1 surface reflectance product
        self.fnames['mod_09'] = []
        filename_tags_09 = er3t.util.get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = er3t.util.download_laads_https(self.date, '6/MYD09A1', filename_tag, day_interval=8, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_09'] += fnames_09

        # OCO2 std and met file
        self.fnames['oco_std'] = []
        self.fnames['oco_met'] = []
        self.fnames['oco_l1b'] = []
        for filename_tag in filename_tags_03:
            dtime = datetime.datetime.strptime(filename_tag, 'A%Y%j.%H%M') + datetime.timedelta(minutes=7.0)
            fnames_std = er3t.util.download_oco2_https(dtime, 'OCO2_L2_Standard.10r', fdir_out=self.fdir_out, run=run)
            fnames_met = er3t.util.download_oco2_https(dtime, 'OCO2_L2_Met.10r'     , fdir_out=self.fdir_out, run=run)
            fnames_l1b = er3t.util.download_oco2_https(dtime, 'OCO2_L1B_Science.10r', fdir_out=self.fdir_out, run=run)
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

    delta_lon = (u*dt) / (np.pi*R_earth) * 180.0
    delta_lat = (v*dt) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat

def pre_cld_oco2(sat, scale_factor=1.0, solver='3D'):

    # Extract
    #   1. cloud top height (cth, 5km resolution);
    #   2. solar zenith and azimuth angles (sza and saa, 1km resolution);
    #   3. sensor zenith and azimuth angles (vza and vaa, 1km resolution);
    #   4. surface height (sfc, 1km resolution)
    # ===================================================================================
    modl2      = er3t.util.modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent, vnames=['Sensor_Zenith', 'Sensor_Azimuth', 'Cloud_Top_Height'])
    logic_cth  = (modl2.data['cloud_top_height']['data']>0.0)
    lon0       = modl2.data['lon_5km']['data']
    lat0       = modl2.data['lat_5km']['data']
    cth0       = modl2.data['cloud_top_height']['data']/1000.0 # units: km

    mod03      = er3t.util.modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    logic_sfh  = (mod03.data['height']['data']>0.0)
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sfh1       = mod03.data['height']['data']/1000.0 # units: km
    sza1       = mod03.data['sza']['data']
    saa1       = mod03.data['saa']['data']
    vza1       = mod03.data['vza']['data']
    vaa1       = mod03.data['vaa']['data']
    # ===================================================================================


    # Process MODIS reflectance at 650 nm (250m resolution)
    # ===================================================================================
    modl1b = er3t.util.modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, ref_2d = er3t.util.grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)
    # ===================================================================================


    # Find cloudy pixels based on MODIS RGB imagery and upscale/downscale to 250m resolution
    # =============================================================================
    mod_rgb = mpl_img.imread(sat.fnames['mod_rgb'][0])

    lon_rgb0 = np.linspace(sat.extent[0], sat.extent[1], mod_rgb.shape[1]+1)
    lat_rgb0 = np.linspace(sat.extent[2], sat.extent[3], mod_rgb.shape[0]+1)
    lon_rgb = (lon_rgb0[1:]+lon_rgb0[:-1])/2.0
    lat_rgb = (lat_rgb0[1:]+lat_rgb0[:-1])/2.0

    mod_r = mod_rgb[:, :, 0]
    mod_g = mod_rgb[:, :, 1]
    mod_b = mod_rgb[:, :, 2]

    # special note: threshold of 1.06 is hard-coded for this particular case;
    #               improvement will be made by Yu-Wen Chen and/or Katey Dong
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


    # Upscale CTH from 5km (L2) to 250m resolution
    # ===================================================================================
    points     = np.transpose(np.vstack((lon0[logic_cth], lat0[logic_cth])))
    cth = interpolate.griddata(points, cth0[logic_cth], (lon, lat), method='cubic')
    cth_2d_l2 = np.zeros_like(lon_2d)
    cth_2d_l2[indices_x, indices_y] = cth
    # ===================================================================================


    # Upscale cloud effective radius from 1km (L2) to 250m resolution
    # =============================================================
    modl2 = er3t.util.modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent)
    lon_2d, lat_2d, cer_2d_l2 = er3t.util.grid_by_lonlat(modl2.data['lon']['data'], modl2.data['lat']['data'], modl2.data['cer']['data'], lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='linear')
    cer_2d_l2[cer_2d_l2<1.0] = 1.0
    # =============================================================


    # Parallax correction (for the cloudy pixels detected previously)
    # ====================================================================================================
    points     = np.transpose(np.vstack((lon1[logic_sfh], lat1[logic_sfh])))
    sfh        = interpolate.griddata(points, sfh1[logic_sfh], (lon, lat), method='cubic')

    points     = np.transpose(np.vstack((lon1, lat1)))
    vza        = interpolate.griddata(points, vza1, (lon, lat), method='cubic')
    vaa        = interpolate.griddata(points, vaa1, (lon, lat), method='cubic')
    vza[...] = np.nanmean(vza)
    vaa[...] = np.nanmean(vaa)

    if solver == '3D':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, cth*1000.0, sfh*1000.0)
    elif solver == 'IPA':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    # ====================================================================================================


    # Wind correction (for the cloudy pixels detected previously)
    # ====================================================================================================
    # estimate average OCO-2 passing time for the scene
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

    # estimate average MODIS passing time for the scene
    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]

    # extract wind speed (10m wind)
    f = h5py.File(sat.fnames['oco_met'][0], 'r')
    lon_oco_met = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_met = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_met>=sat.extent[0]) & (lon_oco_met<=sat.extent[1]) & (lat_oco_met>=sat.extent[2]) & (lat_oco_met<=sat.extent[3])
    u_oco = f['Meteorology/windspeed_u_met'][...][logic]
    v_oco = f['Meteorology/windspeed_v_met'][...][logic]
    f.close()

    # wind correction based on the different time between OCO-2 and MODIS
    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.median(u_oco), np.median(v_oco), utc_oco.mean()-utc_mod.mean())
    # ====================================================================================================



    # Cloud optical property
    #  1) cloud optical thickness: MODIS 650 reflectance -> two-stream approximation -> cloud optical thickness
    #  2) cloud effective radius: from MODIS L2 cloud product (upscaled to 250m resolution from raw 1km resolution)
    #  3) cloud top height: from MODIS L2 cloud product
    #
    #   special note: for this particular case, saturation was found on MODIS 860 nm reflectance
    # ===================================================================================
    # two-stream
    a0         = np.median(ref_2d)
    mu0        = np.cos(np.deg2rad(sza1.mean()))
    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)

    # lon/lat shift due to parallax and wind correction
    lon_1d = lon_2d[:, 0]
    indices_x_new = np.int_(np.round((lon_corr-lon_1d[0])/(((lon_1d[1:]-lon_1d[:-1])).mean()), decimals=0))
    lat_1d = lat_2d[0, :]
    indices_y_new = np.int_(np.round((lat_corr-lat_1d[0])/(((lat_1d[1:]-lat_1d[:-1])).mean()), decimals=0))

    # assign COT, CER, CTH for every cloudy pixel (after parallax and wind correction)
    Nx, Ny = ref_2d.shape
    cot_2d_l1b = np.zeros_like(ref_2d)
    cer_2d_l1b = np.zeros_like(ref_2d); cer_2d_l1b[...] = 1.0
    cth_2d_l1b = np.zeros_like(ref_2d)
    for i in range(indices_x.size):
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            # COT from two-stream
            cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
            # CER from closest CER from MODIS L2 cloud product
            cer_2d_l1b[indices_x_new[i], indices_y_new[i]] = cer_2d_l2[indices_x[i], indices_y[i]]
            # CTH from closest CTH from MODIS L2 cloud product
            cth_2d_l1b[indices_x_new[i], indices_y_new[i]] = cth_2d_l2[indices_x[i], indices_y[i]]

    # special note: secondary cloud/clear-sky filter that is hard-coded for this particular case
    # ===================================================================================
    cot_2d_l1b[(cer_2d_l1b<1.5)&(lat_2d>38.0)] = 0.0
    cot_2d_l1b[(cer_2d_l1b<1.5)&(lon_2d<-108.5)] = 0.0
    cer_2d_l1b[(cer_2d_l1b<1.5)&(lat_2d>38.0)] = 1.0
    cer_2d_l1b[(cer_2d_l1b<1.5)&(lon_2d<-108.5)] = 1.0
    cth_2d_l1b[(cer_2d_l1b<1.5)&(lat_2d>38.0)] = 0.0
    cth_2d_l1b[(cer_2d_l1b<1.5)&(lon_2d<-108.5)] = 0.0
    # ===================================================================================

    # store data for return
    # ===================================================================================
    modl1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
    modl1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
    modl1b.data['ref_2d'] = dict(name='Gridded reflectance'             , units='N/A'        , data=ref_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b*scale_factor)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micron'     , data=cer_2d_l1b)
    modl1b.data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=cth_2d_l1b)
    # ===================================================================================

    return modl1b




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

    print('slope:', slope)
    data_2d = data_bkg_2d*slope

    dx = x_bkg_2d[1, 0] - x_bkg_2d[0, 0]
    dy = y_bkg_2d[0, 1] - y_bkg_2d[0, 0]

    if replace:
        indices_x = np.int_(np.round((x_ref-x_bkg_2d[0, 0])/dx, decimals=0))
        indices_y = np.int_(np.round((y_ref-y_bkg_2d[0, 0])/dy, decimals=0))
        data_2d[indices_x, indices_y] = data_ref

    return data_2d

def pre_sfc_oco2(sat, tag, version='10r', scale=True, replace=True):

    # Read in OCO-2 BRDF data
    if version == '10' or version == '10r':
        vnames = [
                'BRDFResults/brdf_reflectance_o2',              # 0.77 microns
                'BRDFResults/brdf_reflectance_slope_o2',
                'BRDFResults/brdf_reflectance_strong_co2',      # 2.06 microns
                'BRDFResults/brdf_reflectance_slope_strong_co2',
                'BRDFResults/brdf_reflectance_weak_co2',        # 1.615 microns
                'BRDFResults/brdf_reflectance_slope_weak_co2'
                  ]
    else:
        exit('Error   [pre_sfc_oco2]: Cannot recognize version \'%s\'.' % version)

    oco = er3t.util.oco2_std(fnames=sat.fnames['oco_std'], vnames=vnames, extent=sat.extent)

    # BRDF reflectance as surface albedo
    if version == '10' or version == '10r':
        if tag.lower() == 'o2a':
            oco_sfc_alb = oco.data['brdf_reflectance_o2']['data']
        elif tag.lower() == 'wco2':
            oco_sfc_alb = oco.data['brdf_reflectance_weak_co2']['data']
        elif tag.lower() == 'sco2':
            oco_sfc_alb = oco.data['brdf_reflectance_strong_co2']['data']
    else:
        exit('Error   [cdata_sfc_alb]: Cannot recognize version \'%s\'.' % version)

    # Longitude and latitude
    oco_lon = oco.data['lon']['data']
    oco_lat = oco.data['lat']['data']

    # Select data within the specified region defined by <sat.extent>
    logic = (oco_sfc_alb>0.0) & (oco_lon>=sat.extent[0]) & (oco_lon<=sat.extent[1]) & (oco_lat>=sat.extent[2]) & (oco_lat<=sat.extent[3])
    oco_lon = oco_lon[logic]
    oco_lat = oco_lat[logic]
    oco_sfc_alb = oco_sfc_alb[logic]

    # Extract and grid MODIS surface reflectance
    #   band 1: 620  - 670  nm, index 0
    #   band 2: 841  - 876  nm, index 1
    #   band 3: 459  - 479  nm, index 2
    #   band 4: 545  - 565  nm, index 3
    #   band 5: 1230 - 1250 nm, index 4
    #   band 6: 1628 - 1652 nm, index 5
    #   band 7: 2105 - 2155 nm, index 6
    mod = er3t.util.modis_09a1(fnames=sat.fnames['mod_09'], extent=sat.extent)
    points = np.transpose(np.vstack((mod.data['lon']['data'], mod.data['lat']['data'])))
    if tag.lower() == 'o2a':
        lon_2d, lat_2d, mod_sfc_alb_2d = er3t.util.grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][1, :], extent=sat.extent)
        wvl = 770
    elif tag.lower() == 'wco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = er3t.util.grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][5, :], extent=sat.extent)
        wvl = 1615
    elif tag.lower() == 'sco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = er3t.util.grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][6, :], extent=sat.extent)
        wvl = 2060

    # Scale all MODIS reflectance based on the relationship between collocated MODIS surface reflectance and OCO-2 BRDF reflectance
    # Replace MODIS surface reflectance with OCO-2 BRDF reflectance at OCO-2 measurement locations
    oco_sfc_alb_2d = create_sfc_alb_2d(oco_lon, oco_lat, oco_sfc_alb, lon_2d, lat_2d, mod_sfc_alb_2d, scale=scale, replace=replace)

    mod.data['alb_2d'] = dict(data=oco_sfc_alb_2d, name='Surface albedo', units='N/A')
    mod.data['lon_2d'] = dict(data=lon_2d        , name='Longitude'     , units='degrees')
    mod.data['lat_2d'] = dict(data=lat_2d        , name='Latitude'      , units='degrees')
    mod.data['wvl']    = dict(data=wvl           , name='Wavelength'    , units='nm')

    return mod




class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad(sat, wavelength, fname_idl, fdir='tmp-data', solver='3D', overwrite=False):

    """
    Simulate OCO-2 radiance
    """


    # atm object
    # =================================================================================
    levels = np.arange(0.0, 20.1, 0.5)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================


    # abs object
    # special note: in the future, we will implement OCO2 MET file for this
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = er3t.pre.abs.abs_oco_idl(wavelength=wavelength, fname=fname_abs, fname_idl=fname_idl, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================


    # sfc object
    # =================================================================================
    data = {}
    f = h5py.File('data/01_oco2_rad-sim/pre-data.h5', 'r')
    data['alb_2d'] = dict(data=f['mod/sfc/alb_0770'][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')
    f.close()

    fname_sfc = '%s/sfc.pk' % fdir
    mod09 = sat_tmp(data)
    sfc0      = er3t.pre.sfc.sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = er3t.rtm.mca.mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    data = {}
    f = h5py.File('data/01_oco2_rad-sim/pre-data.h5', 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['mod/rad/lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['mod/rad/lat'][...])
    data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_2s'][...])
    data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_l2'][...])
    data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_l2'][...])
    f.close()

    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    cth0 = modl1b.data['cth_2d']['data']
    cld0      = er3t.pre.cld.cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=1.0, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_sca object
    # =================================================================================
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================


    # solar zenith/azimuth angles and sensor zenith/azimuth angles
    # =================================================================================
    f = h5py.File('data/01_oco2_rad-sim/pre-data.h5', 'r')
    sza = f['oco/sza'][...][f['oco/logic'][...]].mean()
    saa = f['oco/saa'][...][f['oco/logic'][...]].mean()
    vza = f['oco/vza'][...][f['oco/logic'][...]].mean()
    vaa = f['oco/vaa'][...][f['oco/logic'][...]].mean()
    f.close()
    # =================================================================================


    # run mcarats
    # =================================================================================
    mca0 = er3t.rtm.mca.mcarats_ng(
            date=sat.date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sfc_2d=sfc_2d,
            sca=sca,
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = sza,
            solar_azimuth_angle  = saa,
            sensor_zenith_angle  = vza,
            sensor_azimuth_angle = vaa,
            fdir='%s/%.4fnm/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=photon_sim,
            solver=solver,
            Ncpu=8,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = er3t.rtm.mca.mca_out_ng(fname='%s/mca-out-rad-oco2-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    # =================================================================================


def main_pre_old():

    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    fdir_data = os.path.abspath('data/%s/download' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    #\----------------------------------------------------------------------------/#



    # download satellite data based on given date and region
    #/----------------------------------------------------------------------------\#
    date   = datetime.datetime(2019, 9, 2)
    extent = [-109.6, -106.5, 35.9, 39.0]

    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#



    # pre-process downloaded data
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('data/%s/pre-data.h5' % name_tag, 'w')
    f0['extent'] = sat0.extent

    # MODIS data groups in the HDF file
    #/--------------------------------------------------------------\#
    g = f0.create_group('mod')
    g1 = g.create_group('rad')
    g2 = g.create_group('cld')
    g3 = g.create_group('sfc')
    #\--------------------------------------------------------------/#

    # MODIS RGB
    #/--------------------------------------------------------------\#
    mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
    g['rgb'] = mod_rgb

    print('Message [pre_data]: the processing of MODIS RGB imagery is complete.')
    #\--------------------------------------------------------------/#

    # cloud optical properties
    #/--------------------------------------------------------------\#
    mod0 = pre_cld_oco2(sat0)
    g1['lon'] = mod0.data['lon_2d']['data']
    g1['lat'] = mod0.data['lat_2d']['data']
    g2['cot_2s'] = mod0.data['cot_2d']['data']
    g2['cer_l2'] = mod0.data['cer_2d']['data']
    g2['cth_l2'] = mod0.data['cth_2d']['data']

    print('Message [pre_data]: the processing of cloud optical properties is complete.')
    #\--------------------------------------------------------------/#

    # surface albedo
    #/--------------------------------------------------------------\#
    mod_sfc = pre_sfc_oco2(sat0, 'o2a', scale=True, replace=True)

    g3['lon'] = mod_sfc.data['lon_2d']['data']
    g3['lat'] = mod_sfc.data['lat_2d']['data']
    g3['alb_%4.4d' % mod_sfc.data['wvl']['data']] = mod_sfc.data['alb_2d']['data']

    print('Message [pre_data]: the processing of surface albedo is complete.')
    #\--------------------------------------------------------------/#


    # OCO-2 data groups in the HDF file
    #/--------------------------------------------------------------\#
    gg = f0.create_group('oco')
    gg1 = gg.create_group('o2a')
    #\--------------------------------------------------------------/#

    # Read OCO-2 radiance and wavelength data
    #/--------------------------------------------------------------\#
    oco = er3t.util.oco2_rad_nadir(sat0)

    wvl_o2a  = np.zeros_like(oco.rad_o2_a, dtype=np.float64)
    for i in range(oco.rad_o2_a.shape[0]):
        for j in range(oco.rad_o2_a.shape[1]):
            wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
    #\--------------------------------------------------------------/#

    # OCO L1B
    #/--------------------------------------------------------------\#
    gg['lon'] = oco.lon_l1b
    gg['lat'] = oco.lat_l1b
    gg['sza'] = oco.sza
    gg['saa'] = oco.saa
    gg['vza'] = oco.vza
    gg['vaa'] = oco.vaa
    gg['logic']  = oco.logic_l1b
    gg['snd_id'] = oco.snd_id
    gg1['rad']   = oco.rad_o2_a
    gg1['wvl']   = wvl_o2a
    print('Message [pre_data]: the processing of OCO-2 radiance is complete.')
    #\--------------------------------------------------------------/#

    f0.close()
    #\----------------------------------------------------------------------------/#

def main_pre(wvl=params['wavelength']):

    # 1) Download and pre-process MODIS data products
    # MODIS data products will be downloaded at <data/02_modis_rad-sim/download>
    # pre-processed data will be saved at <data/02_modis_rad-sim/pre_data.h5>,
    # which will contain
    #   extent ------------ : Dataset  (4,)
    #   lat --------------- : Dataset  (1196, 1196)
    #   lon --------------- : Dataset  (1196, 1196)
    #   mod/cld/cer_l2 ---- : Dataset  (1196, 1196)
    #   mod/cld/cot_l2 ---- : Dataset  (1196, 1196)
    #   mod/cld/cth_l2 ---- : Dataset  (1196, 1196)
    #   mod/geo/saa ------- : Dataset  (1196, 1196)
    #   mod/geo/sfh ------- : Dataset  (1196, 1196)
    #   mod/geo/sza ------- : Dataset  (1196, 1196)
    #   mod/geo/vaa ------- : Dataset  (1196, 1196)
    #   mod/geo/vza ------- : Dataset  (1196, 1196)
    #   mod/rad/rad_0650 -- : Dataset  (1196, 1196)
    #   mod/rad/ref_0650 -- : Dataset  (1196, 1196)
    #   mod/rgb ----------- : Dataset  (1386, 1386, 4)
    #   mod/sfc/alb_09 ---- : Dataset  (666, 666)
    #   mod/sfc/alb_43 ---- : Dataset  (666, 666)
    #   mod/sfc/lat ------- : Dataset  (666, 666)
    #   mod/sfc/lon ------- : Dataset  (666, 666)
    #
    #/----------------------------------------------------------------------------\#
    # cdata_sat_raw(wvl=wvl, plot=True)
    #\----------------------------------------------------------------------------/#


    # apply IPA method to retrieve cloud optical thickness (COT) from MODIS radiance
    # so new COT has higher spatial resolution (250m) than COT from MODIS L2 cloud product
    # notes: the IPA method uses "reflectance vs cot" obtained from the same RT model
    #        used for 3D radiance self-consistency check to ensure their physical processes
    #        are consistent
    # additional data will be saved at <data/02_modis_rad-sim/pre_data.h5>,
    # which are
    #   mod/cld/cer_ipa ---- : Dataset  (1196, 1196)
    #   mod/cld/cot_ipa ---- : Dataset  (1196, 1196)
    #   mod/cld/cth_ipa ---- : Dataset  (1196, 1196)
    #/----------------------------------------------------------------------------\#
    cdata_cld_ipa(wvl=wvl, plot=True)
    #\----------------------------------------------------------------------------/#

def main_sim():

    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    name_tag = os.path.relpath(__file__).replace('.py', '')
    fdir_data = os.path.abspath('data/%s/download' % name_tag)
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#


    # create tmp-data/01_oco2_rad-sim directory if it does not exist
    #/----------------------------------------------------------------------------\#
    fdir_tmp = os.path.abspath('tmp-data/%s' % (name_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    #\----------------------------------------------------------------------------/#


    # read out wavelength information from absorption file
    #/----------------------------------------------------------------------------\#
    fname_idl = 'data/%s/aux/atm_abs_o2a_11.out' % (name_tag)
    f = readsav(fname_idl)
    wvls = f.lamx*1000.0
    #\----------------------------------------------------------------------------/#


    # run radiance simulations under both 3D and IPA modes
    #/----------------------------------------------------------------------------\#
    index = np.argmin(np.abs(wvls-770.0))
    wavelength = wvls[index]
    for solver in ['3D', 'IPA']:
        cal_mca_rad(sat0, wavelength, fname_idl, fdir=fdir_tmp, solver=solver, overwrite=True)
    #\----------------------------------------------------------------------------/#

def main_post(plot=False):

    wvl0 = 768.5151

    # read in OCO-2 measured radiance
    #/----------------------------------------------------------------------------\#
    f = h5py.File('data/%s/pre-data.h5' % name_tag, 'r')
    extent = f['extent'][...]
    wvl_oco = f['oco/o2a/wvl'][...]
    lon_oco = f['oco/lon'][...]
    lat_oco = f['oco/lat'][...]
    rad_oco = f['oco/o2a/rad'][...][:, :, np.argmin(np.abs(wvl_oco[0, 0, :]-wvl0))]
    logic_oco = f['oco/logic'][...]
    lon_2d = f['mod/rad/lon'][...]
    lat_2d = f['mod/rad/lat'][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # read in EaR3T simulations (3D and IPA)
    #/----------------------------------------------------------------------------\#
    fname = 'tmp-data/%s/mca-out-rad-oco2-3d_%.4fnm.h5' % (name_tag, wvl0)
    f = h5py.File(fname, 'r')
    rad_3d     = f['mean/rad'][...]
    rad_3d_std = f['mean/rad_std'][...]
    f.close()

    fname = 'tmp-data/%s/mca-out-rad-oco2-ipa_%.4fnm.h5' % (name_tag, wvl0)
    f = h5py.File(fname, 'r')
    rad_ipa    = f['mean/rad'][...]
    rad_ipa_std = f['mean/rad_std'][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # collocate EaR3T simulations (2D domain) to OCO-2 measurement locations
    #/----------------------------------------------------------------------------\#
    rad_mca_3d = np.zeros_like(rad_oco)
    rad_mca_3d_std = np.zeros_like(rad_oco)
    rad_mca_ipa = np.zeros_like(rad_oco)
    rad_mca_ipa_std = np.zeros_like(rad_oco)

    for i in range(lon_oco.shape[0]):
        for j in range(lon_oco.shape[1]):
            lon0 = lon_oco[i, j]
            lat0 = lat_oco[i, j]

            index_lon = np.argmin(np.abs(lon_2d[:, 0]-lon0))
            index_lat = np.argmin(np.abs(lat_2d[0, :]-lat0))

            rad_mca_ipa[i, j]  = rad_ipa[index_lon, index_lat]
            rad_mca_3d[i, j]   = rad_3d[index_lon, index_lat]

            rad_mca_ipa_std[i, j]  = rad_ipa_std[index_lon, index_lat]
            rad_mca_3d_std[i, j]   = rad_3d_std[index_lon, index_lat]
    #\----------------------------------------------------------------------------/#


    # save data
    #/----------------------------------------------------------------------------\#
    f = h5py.File('data/%s/post-data.h5' % name_tag, 'w')
    f['wvl'] = wvl0
    f['lon'] = lon_oco
    f['lat'] = lat_oco
    f['rad_obs'] = rad_oco
    f['rad_sim_3d'] = rad_mca_3d
    f['rad_sim_ipa'] = rad_mca_ipa
    f['rad_sim_3d_std'] = rad_mca_3d_std
    f['rad_sim_ipa_std'] = rad_mca_ipa_std
    f.close()
    #\----------------------------------------------------------------------------/#



    if plot:

        # average over latitude grid (0.01 degree)
        #/----------------------------------------------------------------------------\#
        lat0 = np.arange(37.0, 39.01, 0.01)
        lat = (lat0[1:] + lat0[:-1])/2.0

        oco_rad = np.zeros_like(lat)
        oco_rad_std = np.zeros_like(lat)

        mca_rad_3d = np.zeros_like(lat)
        mca_rad_3d_std = np.zeros_like(lat)

        mca_rad_ipa = np.zeros_like(lat)
        mca_rad_ipa_std = np.zeros_like(lat)

        for i in range(lat.size):
            logic = (lat_oco>=lat0[i]) & (lat_oco<lat0[i+1])
            oco_rad[i] = np.mean(rad_oco[logic])
            oco_rad_std[i] = np.std(rad_oco[logic])

            mca_rad_3d[i] = np.mean(rad_mca_3d[logic])
            mca_rad_3d_std[i] = np.std(rad_mca_3d[logic])

            mca_rad_ipa[i] = np.mean(rad_mca_ipa[logic])
            mca_rad_ipa_std[i] = np.std(rad_mca_ipa[logic])
        #\----------------------------------------------------------------------------/#


        # plot
        #/----------------------------------------------------------------------------\#
        fig = plt.figure(figsize=(10, 6.18))
        ax1 = fig.add_subplot(111)
        ax1.plot(lat, mca_rad_ipa, color='b', lw=1.5, alpha=0.5)
        ax1.fill_between(lat, oco_rad-oco_rad_std, oco_rad+oco_rad_std, color='k', alpha=0.5, lw=0.0)
        ax1.plot(lat, oco_rad, color='k', lw=1.5, alpha=0.8)
        ax1.fill_between(lat, mca_rad_3d-mca_rad_3d_std, mca_rad_3d+mca_rad_3d_std, color='r', alpha=0.5, lw=0.0)
        ax1.plot(lat, mca_rad_3d, color='r', lw=1.5, alpha=0.8)
        ax1.set_xlim((37.0, 39.0))
        ax1.set_ylim((0.0, 0.4))
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax1.set_xlabel('Latitude [$^\circ$]')
        ax1.set_ylabel('Radiance [$\mathrm{W m^{-2} nm^{-1} sr^{-1}}$]')

        patches_legend = [
                    mpatches.Patch(color='black' , label='OCO-2'),
                    mpatches.Patch(color='red'   , label='RTM 3D'),
                    mpatches.Patch(color='blue'  , label='RTM IPA')
                    ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        plt.savefig('%s.png' % name_tag, bbox_inches='tight')
        plt.close(fig)
        #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    # Step 1. Download and Pre-process data, after run
    #   a. <pre-data.h5> will be created under data/01_oco2_rad-sim
    #/----------------------------------------------------------------------------\#
    main_pre()
    #\----------------------------------------------------------------------------/#

    # Step 2. Use EaR3T to run radiance simulations for OCO-2, after run
    #   a. <mca-out-rad-oco2-3d_768.5151nm.h5>  will be created under tmp-data/01_oco2_rad-sim
    #   b. <mca-out-rad-oco2-ipa_768.5151nm.h5> will be created under tmp-data/01_oco2_rad-sim
    #/----------------------------------------------------------------------------\#
    # main_sim()
    #\----------------------------------------------------------------------------/#

    # Step 3. Post-process radiance observations and simulations for OCO-2, after run
    #   a. <post-data.h5> will be created under data/01_oco2_rad-sim
    #   b. <01_oco2_rad-sim.png> will be created under current directory
    #/----------------------------------------------------------------------------\#
    # main_post(plot=True)
    #\----------------------------------------------------------------------------/#

    pass
