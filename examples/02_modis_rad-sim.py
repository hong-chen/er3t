"""
by Hong Chen (hong.chen.cu@gmail.com)

This code serves as an example code to reproduce 3D MODIS radiance simulation for App. 2 in Chen et al. (2022).

The processes include:
    1) `main_pre()`: automatically download and pre-process satellite data products (~640MB data will be
       downloaded and stored under data/02_modis_rad-sim/download) from NASA data archive
        a) MODIS-Aqua_rgb_2019-09-02_(-109.60,-106.50,35.90,39.00).png
        b) MYD021KM.A2019245.2025.061.2019246161115.hdf
        c) MYD02HKM.A2019245.2025.061.2019246161115.hdf
        d) MYD02QKM.A2019245.2025.061.2019246161115.hdf
        e) MYD03.A2019245.2025.061.2019246155053.hdf
        f) MYD06_L2.A2019245.2025.061.2019246164334.hdf
        g) MYD09A1.A2019241.h09v05.006.2019250044127.hdf

    2) `main_sim()`: run simulation
        a) 3D mode

    3) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2022-08-29 by Hong Chen
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
from matplotlib import rcParams, ticker
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
from matplotlib.ticker import FixedLocator

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_sat
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_09a1, get_sinusoidal_grid_tag
from er3t.util import cal_r_twostream, grid_by_extent, grid_by_lonlat, download_laads_https, download_worldview_rgb, get_satfile_tag

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca




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

        self.fnames['mod_rgb'] = [download_worldview_rgb(self.date, self.extent, fdir_out=self.fdir_out, satellite='aqua', instrument='modis', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02_1km'] = []
        self.fnames['mod_02_hkm'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        filename_tags_03 = get_satfile_tag(self.date, lon, lat, satellite='aqua', instrument='modis')

        for filename_tag in filename_tags_03:
            fnames_03     = download_laads_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_l2     = download_laads_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02_1km = download_laads_https(self.date, '61/MYD021KM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02_hkm = download_laads_https(self.date, '61/MYD02HKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02     = download_laads_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02_1km'] += fnames_02_1km
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03

        # MOD09A1 surface reflectance product
        self.fnames['mod_09'] = []
        filename_tags_09 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = download_laads_https(self.date, '6/MYD09A1', filename_tag, day_interval=8, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_09'] += fnames_09

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

def pre_cld_modis(sat, wvl, scale_factor=1.0, solver='3D'):

    if (wvl>=620) & (wvl<=670):
        wvl = 650
    elif (wvl>=841) & (wvl<=876):
        wvl = 860
    else:
        sys.exit('Error  [pre_cld_modis]: do not support wavelength of %d nm.' % wvl)

    index = {650: 0, 860: 1}

    # Extract
    #   1. cloud top height (cth, 5km resolution);
    #   2. solar zenith and azimuth angles (sza and saa, 1km resolution);
    #   3. sensor zenith and azimuth angles (vza and vaa, 1km resolution);
    #   4. surface height (sfc, 1km resolution)
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
    # ===================================================================================


    # Process MODIS radiance and reflectance at 650 nm (250m resolution)
    # ===================================================================================
    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, ref_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][index[wvl], ...], extent=sat.extent)
    lon_2d, lat_2d, rad_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][index[wvl], ...], extent=sat.extent)
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
    modl2 = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent)
    lon_2d, lat_2d, cer_2d_l2 = grid_by_lonlat(modl2.data['lon']['data'], modl2.data['lat']['data'], modl2.data['cer']['data'], lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='linear')
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
        lon_corr, lat_corr  = para_corr(lon, lat, vza, vaa, cth*1000.0, sfh*1000.0)
    elif solver == 'IPA':
        lon_corr, lat_corr  = para_corr(lon, lat, vza, vaa, sfh, sfh)
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
    modl1b.data['rad_2d'] = dict(name='Gridded radiance'                , units='W/m2/nm/sr' , data=rad_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b*scale_factor)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micron'     , data=cer_2d_l1b)
    modl1b.data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=cth_2d_l1b)
    modl1b.data['wvl']    = dict(name='Wavelength'                      , units='nm'         , data=wvl)
    modl1b.data['sza']    = dict(name='Solar Zenith Angle'              , units='degree'     , data=np.nanmean(sza1))
    modl1b.data['saa']    = dict(name='Solar Azimuth Angle'             , units='degree'     , data=np.nanmean(saa1))
    modl1b.data['vza']    = dict(name='Sensor Zenith Angle'             , units='degree'     , data=np.nanmean(vza1))
    modl1b.data['vaa']    = dict(name='Sensor Azimuth Angle'            , units='degree'     , data=np.nanmean(vaa1))
    # ===================================================================================

    return modl1b




def pre_sfc_modis(sat, wvl):

    if (wvl>=620) & (wvl<=670):
        wvl = 650
    elif (wvl>=841) & (wvl<=876):
        wvl = 860
    else:
        sys.exit('Error  [pre_sfc_modis]: do not support wavelength of %d nm.' % wvl)

    index = {650: 0, 860: 1}

    # Extract and grid MODIS surface reflectance
    #   band 1: 620  - 670  nm, index 0
    #   band 2: 841  - 876  nm, index 1
    #   band 3: 459  - 479  nm, index 2
    #   band 4: 545  - 565  nm, index 3
    #   band 5: 1230 - 1250 nm, index 4
    #   band 6: 1628 - 1652 nm, index 5
    #   band 7: 2105 - 2155 nm, index 6
    mod = modis_09a1(fnames=sat.fnames['mod_09'], extent=sat.extent)
    points = np.transpose(np.vstack((mod.data['lon']['data'], mod.data['lat']['data'])))
    lon_2d, lat_2d, mod_sfc_alb_2d = grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['ref']['data'][index[wvl], :], extent=sat.extent)

    mod.data['alb_2d'] = dict(data=mod_sfc_alb_2d, name='Surface albedo', units='N/A')
    mod.data['lon_2d'] = dict(data=lon_2d        , name='Longitude'     , units='degrees')
    mod.data['lat_2d'] = dict(data=lat_2d        , name='Latitude'      , units='degrees')
    mod.data['wvl']    = dict(data=wvl           , name='Wavelength'    , units='nm')

    return mod




class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad(sat, wavelength, photons=1e7, fdir='tmp-data', solver='3D', overwrite=False):

    """
    Simulate MODIS radiance
    """


    # atm object
    # =================================================================================
    levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================


    # abs object
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================


    # sfc object
    # =================================================================================
    data = {}
    f = h5py.File('data/02_modis_rad-sim/pre-data.h5', 'r')
    data['alb_2d'] = dict(data=f['mod/sfc/alb_%4.4d' % wavelength][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')
    f.close()

    fname_sfc = '%s/sfc.pk' % fdir
    mod09 = sat_tmp(data)
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    data = {}
    f = h5py.File('data/02_modis_rad-sim/pre-data.h5', 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['mod/rad/lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['mod/rad/lat'][...])
    data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_2s'][...])
    data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_l2'][...])
    data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_l2'][...])
    f.close()

    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    cth0 = modl1b.data['cth_2d']['data']
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=1.0, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_sca object
    # =================================================================================
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================


    # solar zenith/azimuth angles and sensor zenith/azimuth angles
    # =================================================================================
    f = h5py.File('data/02_modis_rad-sim/pre-data.h5', 'r')
    sza = f['mod/rad/sza'][...].mean()
    saa = f['mod/rad/saa'][...].mean()
    vza = f['mod/rad/vza'][...].mean()
    vaa = f['mod/rad/vaa'][...].mean()
    f.close()
    # =================================================================================


    # run mcarats
    # =================================================================================
    mca0 = mcarats_ng(
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
            photons=photons,
            solver=solver,
            Ncpu=8,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-modis-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    # =================================================================================




def main_pre(wvl=650):

    # create data directory (for storing data) if the directory does not exist
    # ==================================================================================================
    name_tag = os.path.relpath(__file__).replace('.py', '')

    fdir_data = os.path.abspath('data/%s/download' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ==================================================================================================



    # download satellite data based on given date and region
    # ==================================================================================================
    date   = datetime.datetime(2019, 9, 2)
    extent = [-109.6, -106.5, 35.9, 39.0]

    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ==================================================================================================



    # pre-process downloaded data
    # ==================================================================================================
    f0 = h5py.File('data/%s/pre-data.h5' % name_tag, 'w')
    f0['extent'] = sat0.extent

    # MODIS data groups in the HDF file
    # ==================================================================================================
    g = f0.create_group('mod')
    g1 = g.create_group('rad')
    g2 = g.create_group('cld')
    g3 = g.create_group('sfc')
    # ==================================================================================================

    # MODIS RGB
    # ==================================================================================================
    mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
    g['rgb'] = mod_rgb

    print('Message [pre_data]: the processing of MODIS RGB imagery is complete.')
    # ==================================================================================================

    # cloud optical properties
    # ==================================================================================================
    mod0 = pre_cld_modis(sat0, wvl)
    g1['lon'] = mod0.data['lon_2d']['data']
    g1['lat'] = mod0.data['lat_2d']['data']
    g1['sza'] = mod0.data['sza']['data']
    g1['saa'] = mod0.data['saa']['data']
    g1['vza'] = mod0.data['vza']['data']
    g1['vaa'] = mod0.data['vaa']['data']
    g1['rad_%4.4d' % mod0.data['wvl']['data']] = mod0.data['rad_2d']['data']

    g2['cot_2s'] = mod0.data['cot_2d']['data']
    g2['cer_l2'] = mod0.data['cer_2d']['data']
    g2['cth_l2'] = mod0.data['cth_2d']['data']

    print('Message [pre_data]: the processing of cloud optical properties is complete.')
    # ==================================================================================================

    # surface albedo
    # ==================================================================================================
    mod_sfc = pre_sfc_modis(sat0, wvl)

    g3['lon'] = mod_sfc.data['lon_2d']['data']
    g3['lat'] = mod_sfc.data['lat_2d']['data']
    g3['alb_%4.4d' % mod_sfc.data['wvl']['data']] = mod_sfc.data['alb_2d']['data']

    print('Message [pre_data]: the processing of surface albedo is complete.')
    # ==================================================================================================

    f0.close()

def main_sim(wvl=650):

    # create data directory (for storing data) if the directory does not exist
    # ==================================================================================================
    name_tag = os.path.relpath(__file__).replace('.py', '')
    fdir_data = os.path.abspath('data/%s/download' % name_tag)
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(fname=fname_sat, overwrite=False)
    # ===============================================================


    # create tmp-data/02_modis_rad-sim directory if it does not exist
    # ===============================================================
    fdir_tmp = os.path.abspath('tmp-data/%s' % (name_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    # ===============================================================


    # run radiance simulations under both 3D mode
    # ===============================================================
    cal_mca_rad(sat0, wvl, fdir=fdir_tmp, solver='3D', overwrite=True, photons=1e8)
    # ===============================================================

def main_post(wvl=650, plot=False):

    # create data directory (for storing data) if the directory does not exist
    # ==================================================================================================
    name_tag = os.path.relpath(__file__).replace('.py', '')

    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ==================================================================================================

    # read in MODIS measured radiance
    # ==================================================================================================
    f = h5py.File('data/%s/pre-data.h5' % name_tag, 'r')
    extent = f['extent'][...]
    lon_mod = f['mod/rad/lon'][...]
    lat_mod = f['mod/rad/lat'][...]
    rad_mod = f['mod/rad/rad_%4.4d' % wvl][...]
    f.close()
    # ==================================================================================================


    # read in EaR3T simulations (3D)
    # ==================================================================================================
    fname = 'tmp-data/%s/mca-out-rad-modis-3d_%.4fnm.h5' % (name_tag, wvl)
    f = h5py.File(fname, 'r')
    rad_rtm_3d     = f['mean/rad'][...]
    rad_rtm_3d_std = f['mean/rad_std'][...]
    f.close()
    # ==================================================================================================


    # save data
    # ==================================================================================================
    f = h5py.File('data/%s/post-data.h5' % name_tag, 'w')
    f['wvl'] = wvl
    f['lon'] = lon_mod
    f['lat'] = lat_mod
    f['extent']         = extent
    f['rad_obs']        = rad_mod
    f['rad_sim_3d']     = rad_rtm_3d
    f['rad_sim_3d_std'] = rad_rtm_3d_std
    f.close()
    # ==================================================================================================

    if plot:

        # ==================================================================================================
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221)
        ax1.imshow(rad_mod.T, cmap='Greys_r', extent=extent, origin='lower', vmin=0.0, vmax=0.5)
        ax1.set_xlabel('Longititude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_xlim((-109.0, -107.0))
        ax1.set_ylim((37.0, 39.0))
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax1.set_title('MODIS Measured Radiance')

        logic = (lon_mod>=-109.0) & (lon_mod<=-107.0) & (lat_mod>=37.0) & (lat_mod<=39.0)

        xedges = np.arange(-0.01, 0.61, 0.005)
        yedges = np.arange(-0.01, 0.61, 0.005)
        heatmap, xedges, yedges = np.histogram2d(rad_mod[logic], rad_rtm_3d[logic], bins=(xedges, yedges))
        YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)

        levels = np.concatenate((np.arange(1.0, 10.0, 1.0),
                                 np.arange(10.0, 200.0, 10.0),
                                 np.arange(200.0, 1000.0, 100.0),
                                 np.arange(1000.0, 10001.0, 5000.0)))

        ax2 = fig.add_subplot(223)
        cs = ax2.contourf(XX, YY, heatmap, levels, extend='both', locator=ticker.LogLocator(), cmap='jet')
        ax2.plot([0.0, 1.0], [0.0, 1.0], lw=1.0, ls='--', color='gray', zorder=3)
        ax2.set_xlim(0.0, 0.6)
        ax2.set_ylim(0.0, 0.6)
        ax2.set_xlabel('MODIS Measured Radiance')
        ax2.set_ylabel('Simulated 3D Radiance')

        ax3 = fig.add_subplot(224)
        ax3.imshow(rad_rtm_3d.T, cmap='Greys_r', extent=extent, origin='lower', vmin=0.0, vmax=0.5)
        ax3.set_xlabel('Longititude [$^\circ$]')
        ax3.set_ylabel('Latitude [$^\circ$]')
        ax3.set_xlim((-109.0, -107.0))
        ax3.set_ylim((37.0, 39.0))
        ax3.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax3.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax3.set_title('EaR$^3$T Simulated 3D Radiance')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        plt.savefig('02_modis_rad-sim.png', bbox_inches='tight')
        plt.close(fig)
        # ==================================================================================================




if __name__ == '__main__':

    # Step 1. Download and Pre-process data, after run
    #   a. <pre-data.h5> will be created under data/02_modis_rad-sim
    # =============================================================================
    # main_pre()
    # =============================================================================

    # Step 2. Use EaR3T to run radiance simulations for MODIS, after run
    #   a. <mca-out-rad-modis-3d_650.0000nm.h5> will be created under tmp-data/02_modis_rad-sim
    # =============================================================================
    # main_sim()
    # =============================================================================

    # Step 3. Post-process radiance observations and simulations for MODIS, after run
    #   a. <post-data.h5> will be created under data/02_modis_rad-sim
    #   b. <02_modis_rad-sim.png> will be created under current directory
    # =============================================================================
    # main_post(plot=True)
    # =============================================================================

    pass
