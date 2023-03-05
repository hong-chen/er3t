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
    1) Linux on 2023-03-04 by Hong Chen
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

import er3t


# global variables
#/--------------------------------------------------------------\#
params = {
         'name_tag' : os.path.relpath(__file__).replace('.py', ''),
             'date' : datetime.datetime(2019, 9, 2),
       'wavelength' : 768.5151,
         'oco_band' : 'o2a',
           'region' : [-109.1, -106.9, 36.9, 39.1],
           'photon' : 1e9,
             'Ncpu' : 12,
       'photon_ipa' : 1e8,
   'wavelength_ipa' : 650.0,
        }
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
        self.fnames['mod_l2'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        filename_tags_03 = er3t.util.get_satfile_tag(self.date, lon, lat, satellite='aqua', instrument='modis')

        for filename_tag in filename_tags_03:
            fnames_03     = er3t.util.download_laads_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_l2     = er3t.util.download_laads_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02     = er3t.util.download_laads_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03

        # MODIS surface product
        self.fnames['mod_09'] = []
        self.fnames['mod_43'] = []
        filename_tags_09 = er3t.util.get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = er3t.util.download_laads_https(self.date, '61/MYD09A1', filename_tag, day_interval=8, fdir_out=self.fdir_out, run=run)
            fnames_43 = er3t.util.download_laads_https(self.date, '61/MCD43A3', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_09'] += fnames_09
            self.fnames['mod_43'] += fnames_43

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

def cal_sat_delta_t(sat):

    # estimate average OCO-2 passing time for the scene
    #/----------------------------------------------------------------------------\#
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
    #\----------------------------------------------------------------------------/#

    # estimate average MODIS passing time for the scene
    #/----------------------------------------------------------------------------\#
    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]
    #\----------------------------------------------------------------------------/#

    return utc_oco.mean()-utc_mod.mean()

def cdata_sat_raw(oco_band=params['oco_band'], plot=True):

    # process wavelength
    #/----------------------------------------------------------------------------\#
    if oco_band.lower() == 'o2a':
        wvl = 650
        index_wvl = 0      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 860
        index_wvl_sfc = 1  # select MODIS 860 nm band surface albedo for scaling
    else:
        msg = '\nError [cdata_sat_raw]: Currently, only <oco_band=\'o2a\'> is supported.>'
        sys.exit(msg)
    #\----------------------------------------------------------------------------/#


    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    fdir_data = os.path.abspath('data/%s/download' % params['name_tag'])
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    #\----------------------------------------------------------------------------/#


    # download satellite data based on given date and region
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=params['date'], fdir_out=fdir_data, extent=params['region'], fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#


    # pre-process downloaded data
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('data/%s/pre-data.h5' % params['name_tag'], 'w')
    f0['extent'] = sat0.extent

    # MODIS data groups in the HDF file
    #/--------------------------------------------------------------\#
    g = f0.create_group('mod')

    g0 = g.create_group('geo')
    g1 = g.create_group('rad')
    g2 = g.create_group('cld')
    g3 = g.create_group('sfc')
    #\--------------------------------------------------------------/#

    # MODIS RGB
    #/--------------------------------------------------------------\#
    mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
    g['rgb'] = mod_rgb

    print('Message [cdata_sat_raw]: the processing of MODIS RGB imagery is complete.')
    #\--------------------------------------------------------------/#


    # MODIS radiance/reflectance at 650 nm
    #/--------------------------------------------------------------\#
    modl1b = er3t.util.modis_l1b(fnames=sat0.fnames['mod_02'], extent=sat0.extent)
    lon0  = modl1b.data['lon']['data']
    lat0  = modl1b.data['lat']['data']
    ref0  = modl1b.data['ref']['data'][index_wvl, ...]
    rad0  = modl1b.data['rad']['data'][index_wvl, ...]
    lon_2d, lat_2d, ref_2d = er3t.util.grid_by_extent(lon0, lat0, ref0, extent=sat0.extent)
    lon_2d, lat_2d, rad_2d = er3t.util.grid_by_extent(lon0, lat0, rad0, extent=sat0.extent)

    g1['rad_%4.4d' % wvl] = rad_2d
    g1['ref_%4.4d' % wvl] = ref_2d

    print('Message [cdata_sat_raw]: the processing of MODIS L1B radiance/reflectance at %d nm is complete.' % wvl)

    f0['lon'] = lon_2d
    f0['lat'] = lat_2d

    lon_1d = lon_2d[:, 0]
    lat_1d = lat_2d[0, :]
    #\--------------------------------------------------------------/#


    # MODIS geo information - sza, saa, vza, vaa
    #/--------------------------------------------------------------\#
    mod03 = er3t.util.modis_03(fnames=sat0.fnames['mod_03'], extent=sat0.extent, vnames=['Height'])
    lon0  = mod03.data['lon']['data']
    lat0  = mod03.data['lat']['data']
    sza0  = mod03.data['sza']['data']
    saa0  = mod03.data['saa']['data']
    vza0  = mod03.data['vza']['data']
    vaa0  = mod03.data['vaa']['data']
    sfh0  = mod03.data['height']['data']/1000.0 # units: km
    sfh0[sfh0<0.0] = np.nan

    lon_2d, lat_2d, sza_2d = er3t.util.grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, saa_2d = er3t.util.grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, vza_2d = er3t.util.grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, vaa_2d = er3t.util.grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, sfh_2d = er3t.util.grid_by_lonlat(lon0, lat0, sfh0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')

    g0['sza'] = sza_2d
    g0['saa'] = saa_2d
    g0['vza'] = vza_2d
    g0['vaa'] = vaa_2d
    g0['sfh'] = sfh_2d

    print('Message [cdata_sat_raw]: the processing of MODIS geo-info is complete.')
    #\--------------------------------------------------------------/#


    # cloud properties
    #/--------------------------------------------------------------\#
    modl2 = er3t.util.modis_l2(fnames=sat0.fnames['mod_l2'], extent=sat0.extent, vnames=['cloud_top_height_1km'])

    lon0  = modl2.data['lon']['data']
    lat0  = modl2.data['lat']['data']
    cer0  = modl2.data['cer']['data']
    cot0  = modl2.data['cot']['data']

    cth0  = modl2.data['cloud_top_height_1km']['data']/1000.0 # units: km
    cth0[cth0<=0.0] = np.nan

    lon_2d, lat_2d, cer_2d_l2 = er3t.util.grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest')
    cer_2d_l2[cer_2d_l2<=1.0] = np.nan

    lon_2d, lat_2d, cot_2d_l2 = er3t.util.grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest')
    cot_2d_l2[cot_2d_l2<=0.0] = np.nan

    lon_2d, lat_2d, cth_2d_l2 = er3t.util.grid_by_lonlat(lon0, lat0, cth0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    cth_2d_l2[cth_2d_l2<=0.0] = np.nan

    g2['cot_l2'] = cot_2d_l2
    g2['cer_l2'] = cer_2d_l2
    g2['cth_l2'] = cth_2d_l2

    print('Message [cdata_sat_raw]: the processing of MODIS cloud properties is complete.')
    #\--------------------------------------------------------------/#


    # surface
    #/--------------------------------------------------------------\#
    # Extract and grid MODIS surface reflectance
    #   band 1: 620  - 670  nm, index 0
    #   band 2: 841  - 876  nm, index 1
    #   band 3: 459  - 479  nm, index 2
    #   band 4: 545  - 565  nm, index 3
    #   band 5: 1230 - 1250 nm, index 4
    #   band 6: 1628 - 1652 nm, index 5
    #   band 7: 2105 - 2155 nm, index 6
    mod09 = er3t.util.modis_09a1(fnames=sat0.fnames['mod_09'], extent=sat0.extent)
    lon_2d_sfc, lat_2d_sfc, sfc_09_0 = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][index_wvl, :], extent=sat0.extent)
    sfc_09_0[sfc_09_0<0.0] = 0.0
    lon_2d_sfc, lat_2d_sfc, sfc_09_1 = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][index_wvl_sfc, :], extent=sat0.extent)
    sfc_09_1[sfc_09_1<0.0] = 0.0

    mod43 = er3t.util.modis_43a3(fnames=sat0.fnames['mod_43'], extent=sat0.extent)
    lon_2d_sfc, lat_2d_sfc, sfc_43_0 = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][index_wvl, :], extent=sat0.extent)
    sfc_43_0[sfc_43_0<0.0] = 0.0
    sfc_43_0[sfc_43_0>1.0] = 1.0
    lon_2d_sfc, lat_2d_sfc, sfc_43_1 = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][index_wvl_sfc, :], extent=sat0.extent)
    sfc_43_1[sfc_43_1<0.0] = 0.0
    sfc_43_1[sfc_43_1>1.0] = 1.0

    g3['lon'] = lon_2d_sfc
    g3['lat'] = lat_2d_sfc

    g3['alb_09_%4.4d' % wvl] = sfc_09_0
    g3['alb_43_%4.4d' % wvl] = sfc_43_0
    g3['alb_09_%4.4d' % wvl_sfc] = sfc_09_1
    g3['alb_43_%4.4d' % wvl_sfc] = sfc_43_1

    print('Message [cdata_sat_raw]: the processing of MODIS surface properties is complete.')
    #\--------------------------------------------------------------/#


    # OCO-2 data groups in the HDF file
    #/--------------------------------------------------------------\#
    gg = f0.create_group('oco')
    gg1 = gg.create_group('o2a')
    gg2 = gg.create_group('geo')
    gg3 = gg.create_group('met')
    gg4 = gg.create_group('sfc')
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
    gg['logic']  = oco.logic_l1b
    gg['snd_id'] = oco.snd_id
    gg1['rad']   = oco.rad_o2_a
    gg1['wvl']   = wvl_o2a
    gg2['sza'] = oco.sza
    gg2['saa'] = oco.saa
    gg2['vza'] = oco.vza
    gg2['vaa'] = oco.vaa
    print('Message [cdata_sat_raw]: the processing of OCO-2 radiance is complete.')
    #\--------------------------------------------------------------/#

    # OCO wind speed
    #/--------------------------------------------------------------\#
    # extract wind speed (10m wind)
    f = h5py.File(sat0.fnames['oco_met'][0], 'r')
    lon_oco_met0 = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_met0 = f['SoundingGeometry/sounding_latitude'][...]
    u_10m0 = f['Meteorology/windspeed_u_met'][...]
    v_10m0 = f['Meteorology/windspeed_v_met'][...]
    logic = (np.abs(u_10m0)<50.0) & (np.abs(v_10m0)<50.0) & \
            (lon_oco_met0>=sat0.extent[0]) & (lon_oco_met0<=sat0.extent[1]) & \
            (lat_oco_met0>=sat0.extent[2]) & (lat_oco_met0<=sat0.extent[3])
    f.close()

    gg3['lon'] = lon_oco_met0[logic]
    gg3['lat'] = lat_oco_met0[logic]
    gg3['u_10m'] = u_10m0[logic]
    gg3['v_10m'] = v_10m0[logic]
    gg3['delta_t'] = cal_sat_delta_t(sat0)
    print('Message [cdata_sat_raw]: the processing of OCO-2 meteorological data is complete.')
    #\--------------------------------------------------------------/#


    # OCO-2 surface reflectance
    #/--------------------------------------------------------------\#
    # process wavelength
    if oco_band.lower() == 'o2a':
        vname = 'brdf_reflectance_o2'
    else:
        msg = '\nError [cdata_sat_raw]: Currently, only <oco_band=\'o2a\'> is supported.>'
        sys.exit(msg)

    oco = er3t.util.oco2_std(fnames=sat0.fnames['oco_std'], vnames=['BRDFResults/%s' % vname], extent=sat0.extent)

    oco_sfc_alb = oco.data[vname]['data']
    oco_sfc_alb[oco_sfc_alb<0.0] = 0.0

    gg4['lon'] = oco.data['lon']['data']
    gg4['lat'] = oco.data['lat']['data']
    gg4['alb_%s' % oco_band.lower()] = oco_sfc_alb

    oco_sfc_alb_2d = cal_sfc_alb_2d(oco.data['lon']['data'], oco.data['lat']['data'], oco_sfc_alb, lon_2d_sfc, lat_2d_sfc, sfc_43_1, scale=True, replace=True)
    gg4['alb_%s_2d' % oco_band.lower()] = oco_sfc_alb_2d
    print('Message [cdata_sat_raw]: the processing of OCO-2 surface reflectance is complete.')
    #\--------------------------------------------------------------/#

    f0.close()
    #/----------------------------------------------------------------------------\#

    if plot:

        f0 = h5py.File('data/%s/pre-data.h5' % params['name_tag'], 'r')
        extent = f0['extent'][...]

        rgb = f0['mod/rgb'][...]
        rad = f0['mod/rad/rad_%4.4d' % wvl][...]
        ref = f0['mod/rad/ref_%4.4d' % wvl][...]

        sza = f0['mod/geo/sza'][...]
        saa = f0['mod/geo/saa'][...]
        vza = f0['mod/geo/vza'][...]
        vaa = f0['mod/geo/vaa'][...]

        cot = f0['mod/cld/cot_l2'][...]
        cer = f0['mod/cld/cer_l2'][...]
        cth = f0['mod/cld/cth_l2'][...]
        sfh = f0['mod/geo/sfh'][...]

        alb09 = f0['mod/sfc/alb_09_%4.4d' % wvl][...]
        alb43 = f0['mod/sfc/alb_43_%4.4d' % wvl][...]

        f0.close()

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig = plt.figure(figsize=(16, 16))

        fig.suptitle('MODIS Products Preview')

        # RGB
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(441)
        cs = ax1.imshow(rgb, zorder=0, extent=extent)
        ax1.set_xlim((extent[:2]))
        ax1.set_ylim((extent[2:]))
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_title('RGB Imagery')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\--------------------------------------------------------------/#

        # L1B radiance
        #/----------------------------------------------------------------------------\#
        ax2 = fig.add_subplot(442)
        cs = ax2.imshow(rad.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.5)
        ax2.set_xlim((extent[:2]))
        ax2.set_ylim((extent[2:]))
        ax2.set_xlabel('Longitude [$^\circ$]')
        ax2.set_ylabel('Latitude [$^\circ$]')
        ax2.set_title('L1B Radiance (%d nm)' % wvl)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # L1B reflectance
        #/----------------------------------------------------------------------------\#
        ax3 = fig.add_subplot(443)
        cs = ax3.imshow(ref.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=1.0)
        ax3.set_xlim((extent[:2]))
        ax3.set_ylim((extent[2:]))
        ax3.set_xlabel('Longitude [$^\circ$]')
        ax3.set_ylabel('Latitude [$^\circ$]')
        ax3.set_title('L1B Reflectance (%d nm)' % wvl)

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # sza
        #/----------------------------------------------------------------------------\#
        ax5 = fig.add_subplot(445)
        cs = ax5.imshow(sza.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax5.set_xlim((extent[:2]))
        ax5.set_ylim((extent[2:]))
        ax5.set_xlabel('Longitude [$^\circ$]')
        ax5.set_ylabel('Latitude [$^\circ$]')
        ax5.set_title('Solar Zenith [$^\circ$]')

        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # saa
        #/----------------------------------------------------------------------------\#
        ax6 = fig.add_subplot(446)
        cs = ax6.imshow(saa.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax6.set_xlim((extent[:2]))
        ax6.set_ylim((extent[2:]))
        ax6.set_xlabel('Longitude [$^\circ$]')
        ax6.set_ylabel('Latitude [$^\circ$]')
        ax6.set_title('Solar Azimuth [$^\circ$]')

        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # vza
        #/----------------------------------------------------------------------------\#
        ax7 = fig.add_subplot(447)
        cs = ax7.imshow(vza.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax7.set_xlim((extent[:2]))
        ax7.set_ylim((extent[2:]))
        ax7.set_xlabel('Longitude [$^\circ$]')
        ax7.set_ylabel('Latitude [$^\circ$]')
        ax7.set_title('Viewing Zenith [$^\circ$]')

        divider = make_axes_locatable(ax7)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # vaa
        #/----------------------------------------------------------------------------\#
        ax8 = fig.add_subplot(448)
        cs = ax8.imshow(vaa.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax8.set_xlim((extent[:2]))
        ax8.set_ylim((extent[2:]))
        ax8.set_xlabel('Longitude [$^\circ$]')
        ax8.set_ylabel('Latitude [$^\circ$]')
        ax8.set_title('Viewing Azimuth [$^\circ$]')

        divider = make_axes_locatable(ax8)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot
        #/----------------------------------------------------------------------------\#
        ax9 = fig.add_subplot(449)
        cs = ax9.imshow(cot.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax9.set_xlim((extent[:2]))
        ax9.set_ylim((extent[2:]))
        ax9.set_xlabel('Longitude [$^\circ$]')
        ax9.set_ylabel('Latitude [$^\circ$]')
        ax9.set_title('L2 COT')

        divider = make_axes_locatable(ax9)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer
        #/----------------------------------------------------------------------------\#
        ax10 = fig.add_subplot(4, 4, 10)
        cs = ax10.imshow(cer.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax10.set_xlim((extent[:2]))
        ax10.set_ylim((extent[2:]))
        ax10.set_xlabel('Longitude [$^\circ$]')
        ax10.set_ylabel('Latitude [$^\circ$]')
        ax10.set_title('L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax10)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth
        #/----------------------------------------------------------------------------\#
        ax11 = fig.add_subplot(4, 4, 11)
        cs = ax11.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax11.set_xlim((extent[:2]))
        ax11.set_ylim((extent[2:]))
        ax11.set_xlabel('Longitude [$^\circ$]')
        ax11.set_ylabel('Latitude [$^\circ$]')
        ax11.set_title('L2 CTH [km]')

        divider = make_axes_locatable(ax11)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # sfh
        #/----------------------------------------------------------------------------\#
        ax12 = fig.add_subplot(4, 4, 12)
        cs = ax12.imshow(sfh.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=5.0)
        ax12.set_xlim((extent[:2]))
        ax12.set_ylim((extent[2:]))
        ax12.set_xlabel('Longitude [$^\circ$]')
        ax12.set_ylabel('Latitude [$^\circ$]')
        ax12.set_title('Surface Height [km]')

        divider = make_axes_locatable(ax12)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD09A1, reflectance)
        #/----------------------------------------------------------------------------\#
        ax13 = fig.add_subplot(4, 4, 13)
        cs = ax13.imshow(alb09.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax13.set_xlim((extent[:2]))
        ax13.set_ylim((extent[2:]))
        ax13.set_xlabel('Longitude [$^\circ$]')
        ax13.set_ylabel('Latitude [$^\circ$]')
        ax13.set_title('09A1 Reflectance at %d nm' % wvl_sfc)

        divider = make_axes_locatable(ax13)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD43A3, white sky albedo)
        #/----------------------------------------------------------------------------\#
        ax14 = fig.add_subplot(4, 4, 14)
        cs = ax14.imshow(alb43.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax14.set_xlim((extent[:2]))
        ax14.set_ylim((extent[2:]))
        ax14.set_xlabel('Longitude [$^\circ$]')
        ax14.set_ylabel('Latitude [$^\circ$]')
        ax14.set_title('43A3 WSA at %d nm' % wvl_sfc)

        divider = make_axes_locatable(ax14)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<%s>.png' % (params['name_tag'], _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#





def cloud_mask_rgb(
        rgb,
        extent,
        lon_2d,
        lat_2d,
        a_r=1.06,
        a_g=1.06,
        a_b=1.06,
        logic_good=None
        ):

    # Find cloudy pixels based on MODIS RGB imagery and upscale/downscale to 250m resolution
    #/----------------------------------------------------------------------------\#
    lon_rgb0 = np.linspace(extent[0], extent[1], rgb.shape[1]+1)
    lat_rgb0 = np.linspace(extent[2], extent[3], rgb.shape[0]+1)
    lon_rgb = (lon_rgb0[1:]+lon_rgb0[:-1])/2.0
    lat_rgb = (lat_rgb0[1:]+lat_rgb0[:-1])/2.0

    _r = rgb[:, :, 0]
    _g = rgb[:, :, 1]
    _b = rgb[:, :, 2]

    logic_rgb_nan0 = (_r<=(np.median(_r)*a_r)) |\
                     (_g<=(np.median(_g)*a_g)) |\
                     (_b<=(np.median(_b)*a_b))
    logic_rgb_nan = np.flipud(logic_rgb_nan0).T

    if logic_good is not None:
        logic_rgb_nan[logic_good] = False

    x0_rgb = lon_rgb[0]
    y0_rgb = lat_rgb[0]
    dx_rgb = lon_rgb[1] - x0_rgb
    dy_rgb = lat_rgb[1] - y0_rgb

    indices_x = np.int_(np.round((lon_2d-x0_rgb)/dx_rgb, decimals=0))
    indices_y = np.int_(np.round((lat_2d-y0_rgb)/dy_rgb, decimals=0))

    logic_ref_nan = logic_rgb_nan[indices_x, indices_y]

    indices    = np.where(logic_ref_nan!=1)
    #\----------------------------------------------------------------------------/#

    return indices[0], indices[1]

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

def cdata_cld_ipa(oco_band=params['oco_band'], plot=True):

    # process wavelength
    #/----------------------------------------------------------------------------\#
    if oco_band.lower() == 'o2a':
        wvl = 650
    else:
        msg = '\nError [cdata_sat_raw]: Currently, only <oco_band=\'o2a\'> is supported.>'
        sys.exit(msg)
    #\----------------------------------------------------------------------------/#

    # read in data
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('data/%s/pre-data.h5' % params['name_tag'], 'r')
    extent = f0['extent'][...]
    ref_2d = f0['mod/rad/ref_%4.4d' % params['wavelength_ipa']][...]
    rad_2d = f0['mod/rad/rad_%4.4d' % params['wavelength_ipa']][...]
    rgb    = f0['mod/rgb'][...]
    cot_l2 = f0['mod/cld/cot_l2'][...]
    cer_l2 = f0['mod/cld/cer_l2'][...]
    lon_2d = f0['lon'][...]
    lat_2d = f0['lat'][...]
    cth = f0['mod/cld/cth_l2'][...]
    sfh = f0['mod/geo/sfh'][...]
    sza = f0['mod/geo/sza'][...]
    saa = f0['mod/geo/saa'][...]
    vza = f0['mod/geo/vza'][...]
    vaa = f0['mod/geo/vaa'][...]
    alb = f0['mod/sfc/alb_43_%4.4d' % params['wavelength_ipa']][...]
    alb_oco = f0['oco/sfc/alb_%s_2d' % oco_band.lower()][...]
    u_10m = f0['oco/met/u_10m'][...]
    v_10m = f0['oco/met/v_10m'][...]
    delta_t = f0['oco/met/delta_t'][...]
    f0.close()
    #\----------------------------------------------------------------------------/#


    # cloud mask method based on rgb image and l2 data
    #/----------------------------------------------------------------------------\#
    # primary selection (over-selection of cloudy pixels is expected)
    #/--------------------------------------------------------------\#
    scale_factor = 1.08
    indices_x0, indices_y0 = cloud_mask_rgb(rgb, extent, lon_2d, lat_2d, a_r=scale_factor, a_g=scale_factor, a_b=scale_factor)

    lon_cld0 = lon_2d[indices_x0, indices_y0]
    lat_cld0 = lat_2d[indices_x0, indices_y0]
    #\--------------------------------------------------------------/#

    # secondary filter (remove incorrect cloudy pixels)
    #/--------------------------------------------------------------\#
    ref_cld0    = ref_2d[indices_x0, indices_y0]

    logic_nan_cth = np.isnan(cth[indices_x0, indices_y0])
    logic_nan_cot = np.isnan(cot_l2[indices_x0, indices_y0])
    logic_nan_cer = np.isnan(cer_l2[indices_x0, indices_y0])

    logic_bad = (ref_cld0<np.median(ref_cld0)) & \
                (logic_nan_cth & \
                 logic_nan_cot & \
                 logic_nan_cer)

    logic = np.logical_not(logic_bad)

    lon_cld = lon_cld0[logic]
    lat_cld = lat_cld0[logic]

    Nx, Ny = ref_2d.shape
    indices_x = indices_x0[logic]
    indices_y = indices_y0[logic]
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # ipa retrievals
    #/----------------------------------------------------------------------------\#
    # cth_ipa0
    # get cth for new cloud field obtained from radiance thresholding
    # [indices_x[logic], indices_y[logic]] from cth from MODIS L2 cloud product
    # this is counter-intuitive but we need to account for the parallax
    # correction (approximately) that has been applied to the MODIS L2 cloud
    # product before assigning CTH to cloudy pixels we selected from reflectance
    # field, where the clouds have not yet been parallax corrected
    #/--------------------------------------------------------------\#
    data0 = np.zeros(ref_2d.shape, dtype=np.int32)
    data0[indices_x, indices_y] = 1

    data = np.zeros(cth.shape, dtype=np.int32)
    data[cth>0.0] = 1

    offset_dx, offset_dy = er3t.util.move_correlate(data0, data)
    dlon = (lon_2d[1, 0]-lon_2d[0, 0]) * offset_dx
    dlat = (lat_2d[0, 1]-lat_2d[0, 0]) * offset_dy

    lon_2d_ = lon_2d + dlon
    lat_2d_ = lat_2d + dlat
    extent_ = [extent[0]+dlon, extent[1]+dlon, extent[2]+dlat, extent[3]+dlat]

    cth_ = cth.copy()
    cth_[cth_==0.0] = np.nan

    cth_ipa0 = np.zeros_like(ref_2d)
    cth_ipa0[indices_x, indices_y] = er3t.util.find_nearest(lon_cld, lat_cld, cth_, lon_2d_, lat_2d_)
    cth_ipa0[np.isnan(cth_ipa0)] = 0.0
    #\--------------------------------------------------------------/#

    # cer_ipa0
    #/--------------------------------------------------------------\#
    cer_ipa0 = np.zeros_like(ref_2d)
    cer_ipa0[indices_x, indices_y] = er3t.util.find_nearest(lon_cld, lat_cld, cer_l2, lon_2d_, lat_2d_)
    #\--------------------------------------------------------------/#

    # cot_ipa0
    # two relationships: one for geometrically thick clouds, one for geometrically thin clouds
    # ipa relationship of reflectance vs cloud optical thickness
    #/--------------------------------------------------------------\#
    cot = np.concatenate((np.arange(0.0, 2.0, 0.5),
                          np.arange(2.0, 30.0, 2.0),
                          np.arange(30.0, 60.0, 5.0),
                          np.arange(60.0, 100.0, 10.0),
                          np.arange(100.0, 201.0, 50.0)))

    dx = np.pi*6378.1*(lon_2d[1, 0]-lon_2d[0, 0])/180.0
    dy = np.pi*6378.1*(lat_2d[0, 1]-lat_2d[0, 0])/180.0

    fdir  = 'tmp-data/ipa-%06.1fnm_thick' % (params['wavelength_ipa'])

    f_mca_thick = er3t.rtm.mca.func_ref_vs_cot(
            cot,
            cer0=25.0,
            dx=dx,
            dy=dy,
            fdir=fdir,
            date=params['date'],
            wavelength=params['wavelength_ipa'],
            surface_albedo=alb.mean(),
            solar_zenith_angle=sza.mean(),
            solar_azimuth_angle=saa.mean(),
            sensor_zenith_angle=vza.mean(),
            sensor_azimuth_angle=vaa.mean(),
            cloud_top_height=10.0,
            cloud_geometrical_thickness=7.0,
            photon_number=params['photon_ipa'],
            solver='3d',
            overwrite=False
            )

    fdir  = 'tmp-data/ipa-%06.1fnm_thin' % (params['wavelength_ipa'])
    f_mca_thin= er3t.rtm.mca.func_ref_vs_cot(
            cot,
            cer0=10.0,
            dx=dx,
            dy=dy,
            fdir=fdir,
            date=params['date'],
            wavelength=params['wavelength_ipa'],
            surface_albedo=alb.mean(),
            solar_zenith_angle=sza.mean(),
            solar_azimuth_angle=saa.mean(),
            sensor_zenith_angle=vza.mean(),
            sensor_azimuth_angle=vaa.mean(),
            cloud_top_height=3.0,
            cloud_geometrical_thickness=1.0,
            photon_number=params['photon_ipa'],
            solver='3d',
            overwrite=False
            )

    ref_cld_norm = ref_2d[indices_x, indices_y]/np.cos(np.deg2rad(sza.mean()))

    logic_thick = (cth_ipa0[indices_x, indices_y] > 4.0)
    logic_thin  = (cth_ipa0[indices_x, indices_y] < 4.0)

    cot_ipa0 = np.zeros_like(ref_2d)

    cot_ipa0[indices_x[logic_thick], indices_y[logic_thick]] = f_mca_thick.get_cot_from_ref(ref_cld_norm[logic_thick])
    cot_ipa0[indices_x[logic_thin] , indices_y[logic_thin]]  = f_mca_thin.get_cot_from_ref(ref_cld_norm[logic_thin])

    logic_out = (cot_ipa0<cot[0]) | (cot_ipa0>cot[-1])
    logic_low = (logic_out) & (ref_2d<np.median(ref_2d[indices_x, indices_y]))
    logic_high = logic_out & np.logical_not(logic_low)
    cot_ipa0[logic_low]  = cot[0]
    cot_ipa0[logic_high] = cot[-1]
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # Parallax and wind correction (for the cloudy pixels detected previously)
    #/----------------------------------------------------------------------------\#

    # parallax correction
    # calculate new lon_corr, lat_corr based on cloud, surface and sensor geometries
    #/--------------------------------------------------------------\#
    vza_cld = vza[indices_x, indices_y]
    vaa_cld = vaa[indices_x, indices_y]
    sfh_cld = sfh[indices_x, indices_y] * 1000.0  # convert to meter from km
    cth_cld = cth_ipa0[indices_x, indices_y] * 1000.0 # convert to meter from km
    lon_corr_p, lat_corr_p = para_corr(lon_cld, lat_cld, vza_cld, vaa_cld, cth_cld, sfh_cld)
    #\--------------------------------------------------------------/#

    # wind correction
    # calculate new lon_corr, lat_corr based on wind speed
    #/--------------------------------------------------------------\#
    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.nanmean(u_10m), np.nanmean(v_10m), delta_t)
    #\--------------------------------------------------------------/#

    # perform parallax correction on cot_ipa0, cer_ipa0, and cot_ipa0
    #/--------------------------------------------------------------\#
    Nx, Ny = ref_2d.shape
    cot_ipa = np.zeros_like(ref_2d)
    cer_ipa = np.zeros_like(ref_2d)
    cth_ipa = np.zeros_like(ref_2d)
    cld_msk  = np.zeros(ref_2d.shape, dtype=np.int32)
    for i in range(indices_x.size):
        ix = indices_x[i]
        iy = indices_y[i]

        lon_corr0 = lon_corr[i]
        lat_corr0 = lat_corr[i]
        ix_corr = int((lon_corr0-lon_2d[0, 0])//(lon_2d[1, 0]-lon_2d[0, 0]))
        iy_corr = int((lat_corr0-lat_2d[0, 0])//(lat_2d[0, 1]-lat_2d[0, 0]))
        if (ix_corr>=0) and (ix_corr<Nx) and (iy_corr>=0) and (iy_corr<Ny):
            cot_ipa[ix_corr, iy_corr] = cot_ipa0[ix, iy]
            cer_ipa[ix_corr, iy_corr] = cer_ipa0[ix, iy]
            cth_ipa[ix_corr, iy_corr] = cth_ipa0[ix, iy]
            cld_msk[ix_corr, iy_corr] = 1
    #\--------------------------------------------------------------/#

    # fill-in the empty cracks due to parallax and wind correction
    #/--------------------------------------------------------------\#
    # cld_msk_ = np.zeros(ref_2d.shape, dtype=np.int32)
    # cld_msk_[indices_x, indices_y] = 1

    # logic_crack = (cld_msk==0) & (cld_msk_==1)
    # logic_cld   = (cld_msk==1)

    # cot_ipa_ = cot_ipa.copy()
    # cot_ipa_[np.logical_not(logic_cld)] = np.nan
    # f_interp = interpolate.interp2d(lon_2d[:, 0], lat_2d[0, :], cot_ipa_, fill_value=np.nan)
    # # f_interp = interpolate.griddata((lon_2d[logic_cld], lat_2d[logic_cld]), cot_ipa[logic_cld])
    # # cot_ipa[logic_crack] = f_interp(lon_2d[logic_crack], lat_2d[logic_crack])

    # f_interp = interpolate.RegularGridInterpolator((lon_2d[logic_cld], lat_2d[logic_cld]), cer_ipa[logic_cld])
    # cer_ipa[logic_crack] = f_interp(lon_2d[logic_crack], lat_2d[logic_crack])

    # f_interp = interpolate.RegularGridInterpolator((lon_2d[logic_cld], lat_2d[logic_cld]), cth_ipa[logic_cld])
    # cth_ipa[logic_crack] = f_interp(lon_2d[logic_crack], lat_2d[logic_crack])

    # cld_msk[logic_crack] = 1
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # write cot_ipa into file
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('data/%s/pre-data.h5' % params['name_tag'], 'r+')
    try:
        f0['mod/cld/cot_ipa'] = cot_ipa
        f0['mod/cld/cer_ipa'] = cer_ipa
        f0['mod/cld/cth_ipa'] = cth_ipa
        f0['mod/cld/cot_ipa0'] = cot_ipa0
        f0['mod/cld/cer_ipa0'] = cer_ipa0
        f0['mod/cld/cth_ipa0'] = cth_ipa0
        f0['mod/cld/logic_cld'] = (cld_msk==1)
    except:
        del(f0['mod/cld/cot_ipa'])
        del(f0['mod/cld/cer_ipa'])
        del(f0['mod/cld/cth_ipa'])
        del(f0['mod/cld/cot_ipa0'])
        del(f0['mod/cld/cer_ipa0'])
        del(f0['mod/cld/cth_ipa0'])
        del(f0['mod/cld/logic_cld'])
        f0['mod/cld/cot_ipa'] = cot_ipa
        f0['mod/cld/cer_ipa'] = cer_ipa
        f0['mod/cld/cth_ipa'] = cth_ipa
        f0['mod/cld/cot_ipa0'] = cot_ipa0
        f0['mod/cld/cer_ipa0'] = cer_ipa0
        f0['mod/cld/cth_ipa0'] = cth_ipa0
        f0['mod/cld/logic_cld'] = (cld_msk==1)
    try:
        g0 = f0.create_group('cld_msk')
        g0['indices_x0'] = indices_x0
        g0['indices_y0'] = indices_y0
        g0['indices_x']  = indices_x
        g0['indices_y']  = indices_y
    except:
        del(f0['cld_msk/indices_x0'])
        del(f0['cld_msk/indices_y0'])
        del(f0['cld_msk/indices_x'])
        del(f0['cld_msk/indices_y'])
        del(f0['cld_msk'])
        g0 = f0.create_group('cld_msk')
        g0['indices_x0'] = indices_x0
        g0['indices_y0'] = indices_y0
        g0['indices_x']  = indices_x
        g0['indices_y']  = indices_y
    try:
        g0 = f0.create_group('mca_ipa_thick')
        g0['cot'] = f_mca_thick.cot
        g0['ref'] = f_mca_thick.ref
        g0['ref_std'] = f_mca_thick.ref_std
        g0 = f0.create_group('mca_ipa_thin')
        g0['cot'] = f_mca_thin.cot
        g0['ref'] = f_mca_thin.ref
        g0['ref_std'] = f_mca_thin.ref_std
    except:
        del(f0['mca_ipa_thick/cot'])
        del(f0['mca_ipa_thick/ref'])
        del(f0['mca_ipa_thick/ref_std'])
        del(f0['mca_ipa_thick'])
        del(f0['mca_ipa_thin/cot'])
        del(f0['mca_ipa_thin/ref'])
        del(f0['mca_ipa_thin/ref_std'])
        del(f0['mca_ipa_thin'])
        g0 = f0.create_group('mca_ipa_thick')
        g0['cot'] = f_mca_thick.cot
        g0['ref'] = f_mca_thick.ref
        g0['ref_std'] = f_mca_thick.ref_std
        g0 = f0.create_group('mca_ipa_thin')
        g0['cot'] = f_mca_thin.cot
        g0['ref'] = f_mca_thin.ref
        g0['ref_std'] = f_mca_thin.ref_std
    f0.close()
    #\----------------------------------------------------------------------------/#

    if plot:

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig = plt.figure(figsize=(16, 16))

        fig.suptitle('MODIS Cloud Re-Processing')

        # RGB
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(441)
        cs = ax1.imshow(rgb, zorder=0, extent=extent)
        ax1.set_xlim((extent[:2]))
        ax1.set_ylim((extent[2:]))
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_title('RGB Imagery')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\--------------------------------------------------------------/#

        # L1B reflectance
        #/----------------------------------------------------------------------------\#
        ax2 = fig.add_subplot(442)
        cs = ax2.imshow(ref_2d.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=1.0)
        ax2.set_xlim((extent[:2]))
        ax2.set_ylim((extent[2:]))
        ax2.set_xlabel('Longitude [$^\circ$]')
        ax2.set_ylabel('Latitude [$^\circ$]')
        ax2.set_title('L1B Reflectance (%d nm)' % wvl)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cloud mask (primary)
        #/----------------------------------------------------------------------------\#
        ax3 = fig.add_subplot(443)
        cs = ax3.imshow(rgb, zorder=0, extent=extent)
        ax3.scatter(lon_2d[indices_x0, indices_y0], lat_2d[indices_x0, indices_y0], s=0.1, c='r', alpha=0.1)
        ax3.set_xlim((extent[:2]))
        ax3.set_ylim((extent[2:]))
        ax3.set_xlabel('Longitude [$^\circ$]')
        ax3.set_ylabel('Latitude [$^\circ$]')
        ax3.set_title('Primary Cloud Mask')

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#

        # cloud mask (final)
        #/----------------------------------------------------------------------------\#
        ax4 = fig.add_subplot(444)
        cs = ax4.imshow(rgb, zorder=0, extent=extent)
        ax4.scatter(lon_2d[indices_x, indices_y], lat_2d[indices_x, indices_y], s=0.1, c='r', alpha=0.1)
        ax4.set_xlim((extent[:2]))
        ax4.set_ylim((extent[2:]))
        ax4.set_xlabel('Longitude [$^\circ$]')
        ax4.set_ylabel('Latitude [$^\circ$]')
        ax4.set_title('Secondary Cloud Mask (Final)')

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#

        # cot l2
        #/----------------------------------------------------------------------------\#
        ax5 = fig.add_subplot(445)
        cs = ax5.imshow(cot_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax5.set_xlim((extent[:2]))
        ax5.set_ylim((extent[2:]))
        ax5.set_xlabel('Longitude [$^\circ$]')
        ax5.set_ylabel('Latitude [$^\circ$]')
        ax5.set_title('L2 COT')

        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer l2
        #/----------------------------------------------------------------------------\#
        ax6 = fig.add_subplot(446)
        cs = ax6.imshow(cer_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax6.set_xlim((extent[:2]))
        ax6.set_ylim((extent[2:]))
        ax6.set_xlabel('Longitude [$^\circ$]')
        ax6.set_ylabel('Latitude [$^\circ$]')
        ax6.set_title('L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth l2
        #/----------------------------------------------------------------------------\#
        ax7 = fig.add_subplot(447)
        cs = ax7.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax7.set_xlim((extent[:2]))
        ax7.set_ylim((extent[2:]))
        ax7.set_xlabel('Longitude [$^\circ$]')
        ax7.set_ylabel('Latitude [$^\circ$]')
        ax7.set_title('L2 CTH [km]')

        divider = make_axes_locatable(ax7)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot ipa0
        #/----------------------------------------------------------------------------\#
        ax9 = fig.add_subplot(449)
        cs = ax9.imshow(cot_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax9.set_xlim((extent[:2]))
        ax9.set_ylim((extent[2:]))
        ax9.set_xlabel('Longitude [$^\circ$]')
        ax9.set_ylabel('Latitude [$^\circ$]')
        ax9.set_title('New IPA COT')

        divider = make_axes_locatable(ax9)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer ipa0
        #/----------------------------------------------------------------------------\#
        ax10 = fig.add_subplot(4, 4, 10)
        cs = ax10.imshow(cer_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax10.set_xlim((extent[:2]))
        ax10.set_ylim((extent[2:]))
        ax10.set_xlabel('Longitude [$^\circ$]')
        ax10.set_ylabel('Latitude [$^\circ$]')
        ax10.set_title('New L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax10)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth ipa0
        #/----------------------------------------------------------------------------\#
        ax11 = fig.add_subplot(4, 4, 11)
        cs = ax11.imshow(cth_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax11.set_xlim((extent[:2]))
        ax11.set_ylim((extent[2:]))
        ax11.set_xlabel('Longitude [$^\circ$]')
        ax11.set_ylabel('Latitude [$^\circ$]')
        ax11.set_title('New L2 CTH [km]')

        divider = make_axes_locatable(ax11)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot_ipa
        #/----------------------------------------------------------------------------\#
        ax13 = fig.add_subplot(4, 4, 13)
        cs = ax13.imshow(cot_ipa.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax13.set_xlim((extent[:2]))
        ax13.set_ylim((extent[2:]))
        ax13.set_xlabel('Longitude [$^\circ$]')
        ax13.set_ylabel('Latitude [$^\circ$]')
        ax13.set_title('New IPA COT (Para. Corr.)')

        divider = make_axes_locatable(ax13)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer_ipa
        #/----------------------------------------------------------------------------\#
        ax14 = fig.add_subplot(4, 4, 14)
        cs = ax14.imshow(cer_ipa.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax14.set_xlim((extent[:2]))
        ax14.set_ylim((extent[2:]))
        ax14.set_xlabel('Longitude [$^\circ$]')
        ax14.set_ylabel('Latitude [$^\circ$]')
        ax14.set_title('New L2 CER [$\mu m$] (Para. Corr.)')

        divider = make_axes_locatable(ax14)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth_ipa
        #/----------------------------------------------------------------------------\#
        ax15 = fig.add_subplot(4, 4, 15)
        cs = ax15.imshow(cth_ipa.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax15.set_xlim((extent[:2]))
        ax15.set_ylim((extent[2:]))
        ax15.set_xlabel('Longitude [$^\circ$]')
        ax15.set_ylabel('Latitude [$^\circ$]')
        ax15.set_title('New L2 CTH [km] (Para. Corr.)')

        divider = make_axes_locatable(ax15)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD43A3, white sky albedo)
        #/----------------------------------------------------------------------------\#
        ax16 = fig.add_subplot(4, 4, 16)
        cs = ax16.imshow(alb_oco.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax16.set_xlim((extent[:2]))
        ax16.set_ylim((extent[2:]))
        ax16.set_xlabel('Longitude [$^\circ$]')
        ax16.set_ylabel('Latitude [$^\circ$]')
        ax16.set_title('43A3 WSA (filled and scaled)')

        divider = make_axes_locatable(ax16)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#


        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<%s>.png' % (params['name_tag'], _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#





def func(x, a):

    return a*x

def cal_sfc_alb_2d(x_ref, y_ref, data_ref, x_bkg_2d, y_bkg_2d, data_bkg_2d, scale=True, replace=True):

    logic = (x_ref>=x_bkg_2d.min()) & (x_ref<=x_bkg_2d.max()) & (y_ref>=y_bkg_2d.min()) & (y_ref<=y_bkg_2d.max())
    x_ref = x_ref[logic]
    y_ref = y_ref[logic]
    data_ref = data_ref[logic]

    points = np.transpose(np.vstack((x_bkg_2d.ravel(), y_bkg_2d.ravel())))
    data_bkg = interpolate.griddata(points, data_bkg_2d.ravel(), (x_ref, y_ref), method='nearest')

    logic_valid = (data_bkg>0.0) & (data_ref>0.0)
    x_ref = x_ref[logic_valid]
    y_ref = y_ref[logic_valid]
    data_bkg = data_bkg[logic_valid]
    data_ref = data_ref[logic_valid]

    if scale:
        popt, pcov = curve_fit(func, data_bkg, data_ref)
        slope = popt[0]
    else:
        slope = 1.0

    print('Message [cal_sfc_alb_2d]: slope:', slope)
    data_2d = data_bkg_2d*slope

    dx = x_bkg_2d[1, 0] - x_bkg_2d[0, 0]
    dy = y_bkg_2d[0, 1] - y_bkg_2d[0, 0]

    if replace:
        indices_x = np.int_(np.round((x_ref-x_bkg_2d[0, 0])/dx, decimals=0))
        indices_y = np.int_(np.round((y_ref-y_bkg_2d[0, 0])/dy, decimals=0))
        data_2d[indices_x, indices_y] = data_ref

    return data_2d





class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad(sat, wavelength, fname_idl, fdir='tmp-data', solver='3D', photon=params['photon'], overwrite=False):

    """
    Simulate OCO-2 radiance
    """

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # atm object
    #/----------------------------------------------------------------------------\#
    # f = readsav(fname_idl)
    # levels = f['atm_zgrd'][...]/1000.0
    levels = np.arange(0.0, 20.1, 0.5)
    fname_atm = '%s/atm.pk' % fdir
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    #\----------------------------------------------------------------------------/#


    # abs object
    # special note: in the future, we will implement OCO2 MET file for this
    #/----------------------------------------------------------------------------\#
    fname_abs = '%s/abs.pk' % fdir
    abs0      = er3t.pre.abs.abs_oco_idl(wavelength=wavelength, fname=fname_abs, fname_idl=fname_idl, atm_obj=atm0, overwrite=overwrite)
    #\----------------------------------------------------------------------------/#


    # sfc object
    #/----------------------------------------------------------------------------\#
    data = {}
    f = h5py.File('data/01_oco2_rad-sim/pre-data.h5', 'r')
    data['alb_2d'] = dict(data=f['oco/sfc/alb_%s_2d' % params['oco_band'].lower()][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')
    f.close()

    fname_sfc = '%s/sfc.pk' % fdir
    mod09 = sat_tmp(data)
    sfc0      = er3t.pre.sfc.sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = er3t.rtm.mca.mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    #\----------------------------------------------------------------------------/#


    # cld object
    #/----------------------------------------------------------------------------\#
    data = {}
    f = h5py.File('data/%s/pre-data.h5' % params['name_tag'], 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['lat'][...])
    if solver.lower() == 'ipa':
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_ipa0'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_ipa0'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_ipa0'][...])
    elif solver.lower() == '3d':
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_ipa'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_ipa'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_ipa'][...])
    f.close()

    modl1b = sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    cth0 = modl1b.data['cth_2d']['data']
    cgt0 = np.zeros_like(cth0)
    cgt0[cth0>0.0] = 1.0                  # all clouds have geometrical thickness of 1 km
    cgt0[cth0>4.0] = cth0[cth0>4.0]-3.0   # high clouds (cth>4km) has cloud base at 3 km
    cld0 = er3t.pre.cld.cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=cgt0, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    #\----------------------------------------------------------------------------/#


    # mca_sca object
    #/----------------------------------------------------------------------------\#
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    #\----------------------------------------------------------------------------/#


    # mca_cld object
    #/----------------------------------------------------------------------------\#
    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    #\----------------------------------------------------------------------------/#


    # solar zenith/azimuth angles and sensor zenith/azimuth angles
    #/----------------------------------------------------------------------------\#
    f = h5py.File('data/01_oco2_rad-sim/pre-data.h5', 'r')
    sza = f['oco/geo/sza'][...][f['oco/logic'][...]].mean()
    saa = f['oco/geo/saa'][...][f['oco/logic'][...]].mean()
    if solver.lower() == '3d':
        vza = f['oco/geo/vza'][...][f['oco/logic'][...]].mean()
        vaa = f['oco/geo/vaa'][...][f['oco/logic'][...]].mean()
    elif solver.lower() == 'ipa':
        vza = 0.0
        vaa = 0.0
    f.close()
    #\----------------------------------------------------------------------------/#


    # run mcarats
    #/----------------------------------------------------------------------------\#
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
            photons=photon,
            solver=solver,
            Ncpu=params['Ncpu'],
            mp_mode='py',
            overwrite=overwrite
            )
    #\----------------------------------------------------------------------------/#


    # mcarats output
    #/----------------------------------------------------------------------------\#
    out0 = er3t.rtm.mca.mca_out_ng(fname='%s/mca-out-rad-oco2-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    #\----------------------------------------------------------------------------/#





def main_pre(oco_band='o2a'):

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
    # cdata_sat_raw(oco_band=oco_band, plot=True)
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
    cdata_cld_ipa(oco_band=oco_band, plot=True)
    #\----------------------------------------------------------------------------/#

def main_sim(oco_band='o2a'):

    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    fdir_data = os.path.abspath('data/%s/download' % params['name_tag'])
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#


    # create tmp-data/01_oco2_rad-sim directory if it does not exist
    #/----------------------------------------------------------------------------\#
    fdir_tmp = os.path.abspath('tmp-data/%s' % (params['name_tag']))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    #\----------------------------------------------------------------------------/#


    # read out wavelength information from absorption file
    #/----------------------------------------------------------------------------\#
    fname_idl = 'data/%s/aux/atm_abs_%s_11.out' % (params['name_tag'], oco_band.lower())
    f = readsav(fname_idl)
    wvls = f.lamx*1000.0
    #\----------------------------------------------------------------------------/#


    # run radiance simulations under both 3D and IPA modes
    #/----------------------------------------------------------------------------\#
    index = np.argmin(np.abs(wvls-params['wavelength']))
    wavelength = wvls[index]
    cal_mca_rad(sat0, wavelength, fname_idl, photon=params['photon'], fdir='%s/3d'  % fdir_tmp, solver='3D', overwrite=True)
    cal_mca_rad(sat0, wavelength, fname_idl, photon=1e10            , fdir='%s/ipa' % fdir_tmp, solver='IPA', overwrite=True)
    #\----------------------------------------------------------------------------/#

def main_post(plot=False):

    wvl0 = params['wavelength']

    # read in OCO-2 measured radiance
    #/----------------------------------------------------------------------------\#
    f = h5py.File('data/%s/pre-data.h5' % name_tag, 'r')
    extent = f['extent'][...]
    lon_2d = f['lon'][...]
    lat_2d = f['lat'][...]
    lon_oco = f['oco/lon'][...]
    lat_oco = f['oco/lat'][...]
    wvl_oco = f['oco/o2a/wvl'][...]
    rad_oco = f['oco/o2a/rad'][...][:, :, np.argmin(np.abs(wvl_oco[0, 0, :]-wvl0))]
    logic_oco = f['oco/logic'][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # read in EaR3T simulations (3D and IPA)
    #/----------------------------------------------------------------------------\#
    fname = 'tmp-data/%s/3d/mca-out-rad-oco2-3d_%.4fnm.h5' % (name_tag, wvl0)
    f = h5py.File(fname, 'r')
    rad_3d     = f['mean/rad'][...]
    rad_3d_std = f['mean/rad_std'][...]
    f.close()

    fname = 'tmp-data/%s/ipa/mca-out-rad-oco2-ipa_%.4fnm.h5' % (name_tag, wvl0)
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
        ax1.set_xlim(extent[2:])
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
    #   a. <mca-out-rad-oco2-3d_768.5151nm.h5>  will be created under tmp-data/01_oco2_rad-sim/3d
    #   b. <mca-out-rad-oco2-ipa_768.5151nm.h5> will be created under tmp-data/01_oco2_rad-sim/ipa
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
