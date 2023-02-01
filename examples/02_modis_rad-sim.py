"""
by Hong Chen (hong.chen.cu@gmail.com)

This code serves as an example code to reproduce 3D MODIS radiance simulation for App. 2 in Chen et al. (2022).

The processes include:
    1) `main_pre()`: automatically download and pre-process satellite data products (~640MB data will be
       downloaded and stored under data/02_modis_rad-sim/download) from NASA data archive
        a) MODIS-Aqua_rgb_2019-09-02_(-109.60,-106.50,35.90,39.00).png
        b) MYD02QKM.A2019245.2025.061.2019246161115.hdf
        c) MYD03.A2019245.2025.061.2019246155053.hdf
        d) MYD06_L2.A2019245.2025.061.2019246164334.hdf
        e) MYD09A1.A2019241.h09v05.006.2019250044127.hdf

    2) `main_sim()`: run simulation
        a) 3D mode

    3) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2022-10-23 by Hong Chen
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

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_sat
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_09a1, modis_43a3, get_sinusoidal_grid_tag
from er3t.util import cal_r_twostream, grid_by_extent, grid_by_lonlat, download_laads_https, download_worldview_rgb, get_satfile_tag

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca



# global variables
#/--------------------------------------------------------------\#
_name_tag = os.path.relpath(__file__).replace('.py', '')

_date   = datetime.datetime(2019, 9, 2)
_region = [-109.6, -106.5, 35.9, 39.0]

_wavelength = 650.0
_photon_sim = 1e8
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

        self.fnames['mod_rgb'] = [download_worldview_rgb(self.date, self.extent, fdir_out=self.fdir_out, satellite='aqua', instrument='modis', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        filename_tags_03 = get_satfile_tag(self.date, lon, lat, satellite='aqua', instrument='modis')

        for filename_tag in filename_tags_03:
            fnames_03     = download_laads_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_l2     = download_laads_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            fnames_02     = download_laads_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03

        # MODIS surface product
        self.fnames['mod_09'] = []
        self.fnames['mod_43'] = []
        filename_tags_09 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = download_laads_https(self.date, '61/MYD09A1', filename_tag, day_interval=8, fdir_out=self.fdir_out, run=run)
            fnames_43 = download_laads_https(self.date, '61/MCD43A3', filename_tag, day_interval=1, fdir_out=self.fdir_out, run=run)
            self.fnames['mod_09'] += fnames_09
            self.fnames['mod_43'] += fnames_43

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



class func_cot_vs_rad:

    def __init__(self,
            fdir,
            wavelength,
            cot=np.concatenate((np.arange(0.0, 1.0, 0.1),
                                np.arange(1.0, 10.0, 1.0),
                                np.arange(10.0, 20.0, 2.0),
                                np.arange(20.0, 50.0, 5.0),
                                np.arange(50.0, 100.0, 10.0),
                                np.arange(100.0, 200.0, 20.0),
                                # np.arange(200.0, 401.0, 50.0),
                                )),
            run=False,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        self.fdir       = fdir
        self.wavelength = wavelength
        self.cot        = cot
        self.rad        = np.array([])
        self.rad_std    = np.array([])

        if run:
            self.run_all()

        for i in range(self.cot.size):
            cot0 = self.cot[i]
            fname = '%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot0)
            f0 = h5py.File(fname, 'r')
            rad0     = f0['mean/rad'][...].mean()
            rad_std0 = f0['mean/rad_std'][...].mean()
            f0.close()
            self.rad     = np.append(self.rad, rad0)
            self.rad_std = np.append(self.rad_std, rad_std0)

    def run_all(self):

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(cot0)

    def run_mca_one(self, cot):

        # atm object
        #/----------------------------------------------------------------------------\#
        levels = np.arange(0.0, 20.1, 1.0)
        fname_atm = '%s/atm.pk' % fdir
        atm0   = atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod = '%s/afglt.dat' % er3t.common.fdir_data_atmmod, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # abs object
        #/----------------------------------------------------------------------------\#
        fname_abs = '%s/abs.pk' % fdir
        abs0      = abs_16g(wavelength=self.wavelength, fname=fname_abs, atm_obj=atm0, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # define cloud
        #/----------------------------------------------------------------------------\#
        # read in modis parameters
        #/----------------------------------------------------------------------------\#
        f = h5py.File('data/20190922/pre-data_1km.h5', 'r')
        sza = f['mod/geo/sza'][...].mean()
        saa = f['mod/geo/saa'][...].mean()
        vza = f['mod/geo/vza'][...].mean()
        vaa = f['mod/geo/vaa'][...].mean()
        logic = (f['mod/cld/cot_l2'][...]>0.00001) & (f['lon'][...]>124.5)
        cer   = f['mod/cld/cer_l2'][...][logic].mean()
        f.close()
        #\----------------------------------------------------------------------------/#

        cot_2d    = np.zeros((2, 2), dtype=np.float64); cot_2d[...] = cot
        cer_2d    = np.zeros((2, 2), dtype=np.float64); cer_2d[...] = cer
        ext_3d    = np.zeros((2, 2, 2), dtype=np.float64)

        fname_les_pk  = '%s/les.pk' % self.fdir
        cld0          = cld_les(fname_nc=_fname_les, fname=fname_les_pk, coarsen=[1, 1, 25, 1], overwrite=False)

        cld0.lev['altitude']['data']    = cld0.lay['altitude']['data'][1:4]

        cld0.lay['x']['data']           = cld0.lay['x']['data'][:2]
        cld0.lay['y']['data']           = cld0.lay['y']['data'][:2]
        cld0.lay['nx']['data']          = 2
        cld0.lay['ny']['data']          = 2
        cld0.lay['altitude']['data']    = cld0.lay['altitude']['data'][1:3]
        cld0.lay['pressure']['data']    = cld0.lay['pressure']['data'][1:3]
        cld0.lay['temperature']['data'] = cld0.lay['temperature']['data'][:2, :2, 1:3]
        cld0.lay['cot']['data']         = cot_2d
        cld0.lay['thickness']['data']   = cld0.lay['thickness']['data'][1:3]

        ext_3d[:, :, 0]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 1]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        cld0.lay['extinction']['data']  = ext_3d
        #\----------------------------------------------------------------------------/#

        # mca_sca object
        #/----------------------------------------------------------------------------\#
        pha0 = pha_mie(wvl0=self.wavelength)
        sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # mca_cld object
        #/----------------------------------------------------------------------------\#
        atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

        fname_atm3d = '%s/mca_atm_3d.bin' % self.fdir
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % self.fdir, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]
        #\----------------------------------------------------------------------------/#



        # run mcarats
        # =================================================================================
        mca0 = mcarats_ng(
                date=_date,
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                surface_albedo=0.03,
                sca=sca,
                Ng=abs0.Ng,
                target='radiance',
                solar_zenith_angle   = sza,
                solar_azimuth_angle  = saa,
                sensor_zenith_angle  = vza,
                sensor_azimuth_angle = vaa,
                fdir='%s/%.2f/rad' % (self.fdir, cot),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=_photon_sim,
                solver='3D',
                Ncpu=12,
                mp_mode='py',
                overwrite=True
                )

        # mcarats output
        out0 = mca_out_ng(fname='%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
        # =================================================================================

    def interp_from_rad(self, rad, method='cubic'):

        f = interp1d(self.rad, self.cot, kind=method, bounds_error=False)

        return f(rad)

    def interp_from_cot(self, cot, method='cubic'):

        f = interp1d(self.cot, self.rad, kind=method, bounds_error=False)

        return f(cot)



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







class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad(sat, wavelength, fdir='tmp-data', solver='3D', overwrite=False):

    """
    Simulate MODIS radiance
    """


    # atm object
    # =================================================================================
    levels = np.arange(0.0, 20.1, 0.5)
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
    f = h5py.File('data/%s/pre-data.h5' % _name_tag, 'r')
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
    f = h5py.File('data/%s/pre-data.h5' % _name_tag, 'r')
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
    f = h5py.File('data/%s/pre-data.h5' % _name_tag, 'r')
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
            photons=_photon_sim,
            solver=solver,
            Ncpu=8,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-modis-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    # =================================================================================



def cdata_modis_raw(wvl=_wavelength, plot=True):

    # process wavelength
    #/----------------------------------------------------------------------------\#
    index = {650: 0, 860: 1}

    if (wvl>=620) & (wvl<=670):
        wvl = 650
    elif (wvl>=841) & (wvl<=876):
        wvl = 860
    else:
        sys.exit('Error [cdata_modis_raw]: do not support wavelength of %d nm.' % wvl)

    index_wvl = index[wvl]
    #\----------------------------------------------------------------------------/#


    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    fdir_data = os.path.abspath('data/%s/download' % _name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    #\----------------------------------------------------------------------------/#


    # download satellite data based on given date and region
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=_date, fdir_out=fdir_data, extent=_region, fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#


    # pre-process downloaded data
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File('data/%s/pre-data.h5' % _name_tag, 'w')
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

    print('Message [pre_data]: the processing of MODIS RGB imagery is complete.')
    #\--------------------------------------------------------------/#


    # MODIS radiance/reflectance at 650 nm
    #/--------------------------------------------------------------\#
    modl1b = modis_l1b(fnames=sat0.fnames['mod_02'], extent=sat0.extent)
    lon0  = modl1b.data['lon']['data']
    lat0  = modl1b.data['lat']['data']
    ref0  = modl1b.data['ref']['data'][index_wvl, ...]
    rad0  = modl1b.data['rad']['data'][index_wvl, ...]
    lon_2d, lat_2d, ref_2d = grid_by_extent(lon0, lat0, ref0, extent=sat0.extent)
    lon_2d, lat_2d, rad_2d = grid_by_extent(lon0, lat0, rad0, extent=sat0.extent)

    g1['rad_%4.4d' % wvl] = rad_2d
    g1['ref_%4.4d' % wvl] = ref_2d

    print('Message [pre_data]: the processing of MODIS L1B radiance/reflectance at %d nm is complete.' % wvl)

    f0['lon'] = lon_2d
    f0['lat'] = lat_2d

    lon_1d = lon_2d[:, 0]
    lat_1d = lat_2d[0, :]
    #\--------------------------------------------------------------/#


    # MODIS geo information - sza, saa, vza, vaa
    #/--------------------------------------------------------------\#
    mod03 = modis_03(fnames=sat0.fnames['mod_03'], extent=sat0.extent, vnames=['Height'])
    lon0  = mod03.data['lon']['data']
    lat0  = mod03.data['lat']['data']
    sza0  = mod03.data['sza']['data']
    saa0  = mod03.data['saa']['data']
    vza0  = mod03.data['vza']['data']
    vaa0  = mod03.data['vaa']['data']
    sfh0  = mod03.data['height']['data']/1000.0 # units: km
    sfh0[sfh0<0.0] = np.nan

    lon_2d, lat_2d, sza_2d = grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, saa_2d = grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, vza_2d = grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, vaa_2d = grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
    lon_2d, lat_2d, sfh_2d = grid_by_lonlat(lon0, lat0, sfh0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')

    g0['sza'] = sza_2d
    g0['saa'] = saa_2d
    g0['vza'] = vza_2d
    g0['vaa'] = vaa_2d
    g0['sfh'] = sfh_2d

    print('Message [pre_data]: the processing of MODIS geo-info is complete.')
    #\--------------------------------------------------------------/#


    # cloud properties
    #/--------------------------------------------------------------\#
    modl2 = modis_l2(fnames=sat0.fnames['mod_l2'], extent=sat0.extent, vnames=['cloud_top_height_1km'])

    lon0  = modl2.data['lon']['data']
    lat0  = modl2.data['lat']['data']
    cer0  = modl2.data['cer']['data']
    cot0  = modl2.data['cot']['data']

    cth0  = modl2.data['cloud_top_height_1km']['data']/1000.0 # units: km
    cth0[cth0<0.0] = np.nan

    lon_2d, lat_2d, cer_2d_l2 = grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest')
    cer_2d_l2[cer_2d_l2<1.0] = 1.0

    lon_2d, lat_2d, cot_2d_l2 = grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest')

    lon_2d, lat_2d, cth_2d_l2 = grid_by_lonlat(lon0, lat0, cth0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')

    g2['cot_l2'] = cot_2d_l2
    g2['cer_l2'] = cer_2d_l2
    g2['cth_l2'] = cth_2d_l2

    print('Message [pre_data]: the processing of MODIS cloud properties is complete.')
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
    mod09 = modis_09a1(fnames=sat0.fnames['mod_09'], extent=sat0.extent)
    lon_2d_sfc, lat_2d_sfc, sfc_09 = grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][index_wvl, :], extent=sat0.extent)

    mod43 = modis_43a3(fnames=sat0.fnames['mod_43'], extent=sat0.extent)
    lon_2d_sfc, lat_2d_sfc, sfc_43 = grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][index_wvl, :], extent=sat0.extent)

    g3['lon'] = lon_2d_sfc
    g3['lat'] = lat_2d_sfc

    g3['alb_09'] = sfc_09
    g3['alb_43'] = sfc_43

    print('Message [pre_data]: the processing of MODIS surface properties is complete.')
    #\--------------------------------------------------------------/#

    f0.close()
    #/----------------------------------------------------------------------------\#

    if plot:

        f0 = h5py.File('data/%s/pre-data.h5' % _name_tag, 'r')
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

        alb09 = f0['mod/sfc/alb_09'][...]
        alb43 = f0['mod/sfc/alb_43'][...]

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
        ax13.set_title('09A1 Reflectance')

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
        ax14.set_title('43A3 WSA')

        divider = make_axes_locatable(ax14)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_<%s>.png' % (_name_tag, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#




class func_cot_vs_rad:

    def __init__(self,
            fdir,
            wavelength,
            cot=np.concatenate((np.arange(0.0, 1.0, 0.1),
                                np.arange(1.0, 10.0, 1.0),
                                np.arange(10.0, 20.0, 2.0),
                                np.arange(20.0, 50.0, 5.0),
                                np.arange(50.0, 100.0, 10.0),
                                np.arange(100.0, 200.0, 20.0),
                                # np.arange(200.0, 401.0, 50.0),
                                )),
            run=False,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        self.fdir       = fdir
        self.wavelength = wavelength
        self.cot        = cot
        self.rad        = np.array([])
        self.rad_std    = np.array([])

        if run:
            self.run_all()

        for i in range(self.cot.size):
            cot0 = self.cot[i]
            fname = '%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot0)
            f0 = h5py.File(fname, 'r')
            rad0     = f0['mean/rad'][...].mean()
            rad_std0 = f0['mean/rad_std'][...].mean()
            f0.close()
            self.rad     = np.append(self.rad, rad0)
            self.rad_std = np.append(self.rad_std, rad_std0)

    def run_all(self):

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(cot0)

    def run_mca_one(self, cot):

        # atm object
        #/----------------------------------------------------------------------------\#
        levels = np.arange(0.0, 20.1, 1.0)
        fname_atm = '%s/atm.pk' % fdir
        atm0   = atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod = '%s/afglt.dat' % er3t.common.fdir_data_atmmod, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # abs object
        #/----------------------------------------------------------------------------\#
        fname_abs = '%s/abs.pk' % fdir
        abs0      = abs_16g(wavelength=self.wavelength, fname=fname_abs, atm_obj=atm0, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # define cloud
        #/----------------------------------------------------------------------------\#
        # read in modis parameters
        #/----------------------------------------------------------------------------\#
        f = h5py.File('data/20190922/pre-data_1km.h5', 'r')
        sza = f['mod/geo/sza'][...].mean()
        saa = f['mod/geo/saa'][...].mean()
        vza = f['mod/geo/vza'][...].mean()
        vaa = f['mod/geo/vaa'][...].mean()
        logic = (f['mod/cld/cot_l2'][...]>0.00001) & (f['lon'][...]>124.5)
        cer   = f['mod/cld/cer_l2'][...][logic].mean()
        f.close()
        #\----------------------------------------------------------------------------/#

        cot_2d    = np.zeros((2, 2), dtype=np.float64); cot_2d[...] = cot
        cer_2d    = np.zeros((2, 2), dtype=np.float64); cer_2d[...] = cer
        ext_3d    = np.zeros((2, 2, 2), dtype=np.float64)

        fname_les_pk  = '%s/les.pk' % self.fdir
        cld0          = cld_les(fname_nc=_fname_les, fname=fname_les_pk, coarsen=[1, 1, 25, 1], overwrite=False)

        cld0.lev['altitude']['data']    = cld0.lay['altitude']['data'][1:4]

        cld0.lay['x']['data']           = cld0.lay['x']['data'][:2]
        cld0.lay['y']['data']           = cld0.lay['y']['data'][:2]
        cld0.lay['nx']['data']          = 2
        cld0.lay['ny']['data']          = 2
        cld0.lay['altitude']['data']    = cld0.lay['altitude']['data'][1:3]
        cld0.lay['pressure']['data']    = cld0.lay['pressure']['data'][1:3]
        cld0.lay['temperature']['data'] = cld0.lay['temperature']['data'][:2, :2, 1:3]
        cld0.lay['cot']['data']         = cot_2d
        cld0.lay['thickness']['data']   = cld0.lay['thickness']['data'][1:3]

        ext_3d[:, :, 0]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 1]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        cld0.lay['extinction']['data']  = ext_3d
        #\----------------------------------------------------------------------------/#

        # mca_sca object
        #/----------------------------------------------------------------------------\#
        pha0 = pha_mie(wvl0=self.wavelength)
        sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # mca_cld object
        #/----------------------------------------------------------------------------\#
        atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

        fname_atm3d = '%s/mca_atm_3d.bin' % self.fdir
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % self.fdir, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]
        #\----------------------------------------------------------------------------/#



        # run mcarats
        # =================================================================================
        mca0 = mcarats_ng(
                date=_date,
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                surface_albedo=0.03,
                sca=sca,
                Ng=abs0.Ng,
                target='radiance',
                solar_zenith_angle   = sza,
                solar_azimuth_angle  = saa,
                sensor_zenith_angle  = vza,
                sensor_azimuth_angle = vaa,
                fdir='%s/%.2f/rad' % (self.fdir, cot),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=_photon_sim,
                solver='3D',
                Ncpu=12,
                mp_mode='py',
                overwrite=True
                )

        # mcarats output
        out0 = mca_out_ng(fname='%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
        # =================================================================================

    def interp_from_rad(self, rad, method='cubic'):

        f = interp1d(self.rad, self.cot, kind=method, bounds_error=False)

        return f(rad)

    def interp_from_cot(self, cot, method='cubic'):

        f = interp1d(self.cot, self.rad, kind=method, bounds_error=False)

        return f(cot)

def cloud_mask_rgb(
        rgb,
        extent,
        lon_2d,
        lat_2d,
        scale_factor=1.06
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

    logic_rgb_nan0 = (_r<=(np.median(_r)*scale_factor)) |\
                     (_g<=(np.median(_g)*scale_factor)) |\
                     (_b<=(np.median(_b)*scale_factor))
    logic_rgb_nan = np.flipud(logic_rgb_nan0).T

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




def cdata_cot_ipa(wvl=_wavelength, plot=True):

    f0 = h5py.File('data/%s/pre-data.h5' % _name_tag, 'r')
    extent = f0['extent'][...]
    ref_2d = f0['mod/rad/ref_%4.4d' % wvl][...]
    rad_2d = f0['mod/rad/rad_%4.4d' % wvl][...]
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
    f0.close()


    # cloud mask method based on rgb image and l2 data
    #/----------------------------------------------------------------------------\#
    # first cloudy pixel selection (over-selection is expected)
    #/--------------------------------------------------------------\#
    indices_x, indices_y = cloud_mask_rgb(rgb, extent, lon_2d, lat_2d, scale_factor=1.08)

    lon_cld0 = lon_2d[indices_x, indices_y]
    lat_cld0 = lat_2d[indices_x, indices_y]
    #\--------------------------------------------------------------/#


    # secondary filter (remove incorrect cloudy pixels)
    #/--------------------------------------------------------------\#
    ref0    = ref_2d[indices_x, indices_y]
    cot_l20 = cot_l2[indices_x, indices_y]

    logic_bad = (ref0<np.median(ref0)) & (cot_l20<0.01)
    logic = np.logical_not(logic_bad)

    lon_cld = lon_cld0[logic]
    lat_cld = lat_cld0[logic]
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # Parallax correction (for the cloudy pixels detected previously)
    #/----------------------------------------------------------------------------\#
    vza_cld = vza[indices_x[logic], indices_y[logic]]
    vaa_cld = vaa[indices_x[logic], indices_y[logic]]
    sfh_cld = sfh[indices_x[logic], indices_y[logic]] * 1000.0  # convert to meter from km

    # get new cth (need more work)
    #/--------------------------------------------------------------\#
    cth_cld = cth[indices_x[logic], indices_y[logic]] * 1000.0  # convert to meter from km
    #\--------------------------------------------------------------/#

    lon_corr, lat_corr  = para_corr(lon_cld, lat_cld, vza_cld, vaa_cld, cth_cld, sfh_cld)
    #\----------------------------------------------------------------------------/#



    # figure
    #/----------------------------------------------------------------------------\#
    plt.close('all')
    fig = plt.figure(figsize=(12, 12))
    # fig.suptitle('Figure')
    # plot
    #/--------------------------------------------------------------\#
    ax1 = fig.add_subplot(111)
    cs = ax1.imshow(rgb, zorder=0, extent=extent)
    # ax1.scatter(lon_cld0, lat_cld0, s=1, c='b', lw=0.0)
    ax1.scatter(lon_cld, lat_cld, s=1, c='r', lw=0.0)
    ax1.scatter(lon_corr, lat_corr, s=1, c='b', lw=0.0)
    # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
    # ax1.set_xlim(())
    # ax1.set_ylim(())
    # ax1.set_xlabel('')
    # ax1.set_ylabel('')
    # ax1.set_title('')
    # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
    # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
    #\--------------------------------------------------------------/#
    # add colorbar
    #/--------------------------------------------------------------\#
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', '5%', pad='3%')
    # cbar = fig.colorbar(cs, cax=cax)
    # cbar.set_label('', rotation=270, labelpad=4.0)
    # cbar.set_ticks([])
    # cax.axis('off')
    #\--------------------------------------------------------------/#
    # save figure
    #/--------------------------------------------------------------\#
    # plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    # plt.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
    #\--------------------------------------------------------------/#
    plt.show()
    sys.exit()
    #\----------------------------------------------------------------------------/#




    # two-stream
    #/----------------------------------------------------------------------------\#
    # theoretical relationship
    #/--------------------------------------------------------------\#
    fdir = 'tmp-data/mca_rad_vs_cot/%3.3d' % (_wavelength)
    f_mca =  func_cot_vs_rad(fdir, _wavelength, run=False)

    mu0        = np.cos(np.deg2rad(sza.mean()))
    xx_mca = f_mca.cot.copy()
    yy_mca = f_mca.rad*np.pi / (1.5718455223851924*mu0)

    f = interp1d(yy_mca, xx_mca, kind='cubic', bounds_error=False)
    #\--------------------------------------------------------------/#

    # assign COT for every cloudy pixel
    #/--------------------------------------------------------------\#
    Nx, Ny = ref_2d.shape
    cot_rt = np.zeros_like(ref_2d)
    for i in range(indices_x.size):
        if 0<=indices_x[i]<Nx and 0<=indices_y[i]<Ny:

            # mca-rt
            cot_rt[indices_x[i], indices_y[i]] = f(ref_2d[indices_x[i], indices_y[i]])
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # write cot_rt into file
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File(fname, 'r+')
    try:
        f0['mod/cld/cot_ipa'] = cot_ipa
    except:
        del(f0['mod/cld/cot_ipa'])
        f0['mod/cld/cot_ipa'] = cot_ipa
    f0.close()
    #\----------------------------------------------------------------------------/#


    if True:
        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111, aspect='equal')

        # heatmap
        #/--------------------------------------------------------------\#
        logic = (lon_2d>124.5)
        data_x = cot_l2[logic]
        data_y = cot_rt[logic]

        xedge = np.linspace(0.0, 30.0, 101)
        yedge = np.linspace(0.0, 30.0, 101)
        heatmap, xedge, yedge = np.histogram2d(data_x.ravel(), data_y.ravel(), bins=(xedge, yedge))
        extent = [xedge[0], xedge[-1], yedge[0], yedge[-1]]
        x = (xedge[:-1]+xedge[1:])/2.0
        y = (yedge[:-1]+yedge[1:])/2.0
        xx, yy = np.meshgrid(x, y, indexing='ij')
        # levels = np.power(10, np.linspace(0.0, np.log10(heatmap.max()/100), 101))
        levels = np.linspace(0, heatmap.max()/100, 101)
        cmap = mpl.cm.get_cmap('jet').copy()
        cmap.set_under('white')
        cs = ax1.contourf(xx, yy, heatmap, levels, extent=extent, extend='both', cmap=cmap, zorder=0)
        ax1.plot(extent[:2], extent[2:], color='gray', ls='--', lw=1.5)
        ax1.set_xlim(extent[:2])
        ax1.set_ylim(extent[2:])
        ax1.set_title('Heatmap')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\--------------------------------------------------------------/#

        ax1.plot([0, 100], [0, 100], ls='--', color='gray')
        ax1.set_xlim((0, 15))
        ax1.set_ylim((0, 15))
        ax1.set_xlabel('MODIS L2 COT')
        ax1.set_ylabel('MCARaTS COT')
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        # plt.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_cot-rt.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
        #\----------------------------------------------------------------------------/#





def main_pre(wvl=_wavelength):


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
    # cdata_modis_raw(wvl=wvl, plot=True)
    #\----------------------------------------------------------------------------/#


    # apply IPA method to retrieve cloud optical thickness (COT) from MODIS radiance
    # so new COT has higher spatial resolution (250m) than COT from MODIS L2 cloud product
    # notes: the IPA method uses radiance vs cot
    #
    #/----------------------------------------------------------------------------\#
    cdata_cot_ipa()
    #\----------------------------------------------------------------------------/#

    sys.exit()
    #/----------------------------------------------------------------------------\#

def main_sim(wvl=_wavelength):

    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    fdir_data = os.path.abspath('data/%s/download' % _name_tag)
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#


    # create tmp-data/02_modis_rad-sim directory if it does not exist
    #/----------------------------------------------------------------------------\#
    fdir_tmp = os.path.abspath('tmp-data/%s' % (_name_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    #\----------------------------------------------------------------------------/#


    # run radiance simulations under both 3D mode
    #/----------------------------------------------------------------------------\#
    cal_mca_rad(sat0, wvl, fdir=fdir_tmp, solver='3D', overwrite=True)
    #\----------------------------------------------------------------------------/#

def main_post(wvl=_wavelength, plot=False):

    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    fdir_data = os.path.abspath('data/%s' % _name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    #\----------------------------------------------------------------------------/#


    # read in MODIS measured radiance
    #/----------------------------------------------------------------------------\#
    f = h5py.File('data/%s/pre-data.h5' % _name_tag, 'r')
    extent = f['extent'][...]
    lon_mod = f['mod/rad/lon'][...]
    lat_mod = f['mod/rad/lat'][...]
    rad_mod = f['mod/rad/rad_%4.4d' % wvl][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # read in EaR3T simulations (3D)
    #/----------------------------------------------------------------------------\#
    fname = 'tmp-data/%s/mca-out-rad-modis-3d_%.4fnm.h5' % (_name_tag, wvl)
    f = h5py.File(fname, 'r')
    rad_rtm_3d     = f['mean/rad'][...]
    rad_rtm_3d_std = f['mean/rad_std'][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # save data
    #/----------------------------------------------------------------------------\#
    f = h5py.File('data/%s/post-data.h5' % _name_tag, 'w')
    f['wvl'] = wvl
    f['lon'] = lon_mod
    f['lat'] = lat_mod
    f['extent']         = extent
    f['rad_obs']        = rad_mod
    f['rad_sim_3d']     = rad_rtm_3d
    f['rad_sim_3d_std'] = rad_rtm_3d_std
    f.close()
    #\----------------------------------------------------------------------------/#

    if plot:

        #/----------------------------------------------------------------------------\#
        fig = plt.figure(figsize=(10, 10))

        # 2D plot: rad_obs
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(221)
        ax1.imshow(rad_mod.T, cmap='viridis', extent=extent, origin='lower', vmin=0.0, vmax=0.5)
        ax1.set_xlabel('Longititude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_xlim(extent[:2])
        ax1.set_ylim(extent[2:])
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax1.set_title('MODIS Measured Radiance')
        #\--------------------------------------------------------------/#

        # heatmap: rad_sim vs rad_obs
        #/--------------------------------------------------------------\#
        logic = (lon_mod>=extent[0]) & (lon_mod<=extent[1]) & (lat_mod>=extent[2]) & (lat_mod<=extent[3])

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
        #\--------------------------------------------------------------/#

        # 2D plot: rad_sim
        #/--------------------------------------------------------------\#
        ax3 = fig.add_subplot(224)
        ax3.imshow(rad_rtm_3d.T, cmap='viridis', extent=extent, origin='lower', vmin=0.0, vmax=0.5)
        ax3.set_xlabel('Longititude [$^\circ$]')
        ax3.set_ylabel('Latitude [$^\circ$]')
        ax3.set_xlim(extent[:2])
        ax3.set_ylim(extent[2:])
        ax3.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax3.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax3.set_title('EaR$^3$T Simulated 3D Radiance')
        #\--------------------------------------------------------------/#

        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        plt.savefig('%s.png' % _name_tag, bbox_inches='tight')
        plt.close(fig)
        #\----------------------------------------------------------------------------/#




if __name__ == '__main__':

    # Step 1. Download and Pre-process data, after run
    #   a. <pre-data.h5> will be created under data/02_modis_rad-sim
    #/----------------------------------------------------------------------------\#
    main_pre()
    #\----------------------------------------------------------------------------/#

    # Step 2. Use EaR3T to run radiance simulations for MODIS, after run
    #   a. <mca-out-rad-modis-3d_650.0000nm.h5> will be created under tmp-data/02_modis_rad-sim
    #/----------------------------------------------------------------------------\#
    # main_sim()
    #\----------------------------------------------------------------------------/#

    # Step 3. Post-process radiance observations and simulations for MODIS, after run
    #   a. <post-data.h5> will be created under data/02_modis_rad-sim
    #   b. <02_modis_rad-sim.png> will be created under current directory
    #/----------------------------------------------------------------------------\#
    # main_post(plot=True)
    #\----------------------------------------------------------------------------/#

    pass
