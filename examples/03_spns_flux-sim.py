"""
by Hong Chen (hong.chen.cu@gmail.com)

This code serves as an example code to reproduce 3D irradiance simulation for App. 3 in Chen et al. (2022).
Special note: due to large data volume, only partial flight track simulation is provided for illustration purpose.

The processes include:
    1) `main_run()`: pre-process aircraft and satellite data and run simulations
        a) partition flight track into mini flight track segments and collocate satellite imagery data
        b) run simulations based on satellite imagery cloud retrievals
            i) 3D mode
            ii) IPA mode

    2) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2022-07-26 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64

"""

import os
import sys
import glob
import h5py
import numpy as np
import datetime
import pickle
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_les, cld_sat
from er3t.pre.pha import pha_mie_wc as pha_mie

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng
from er3t.rtm.mca_v010 import mca_sca
from er3t.util.ahi import ahi_l2
from er3t.util.modis import grid_modis_by_extent



def get_jday_ahi(fnames):

    """
    Get UTC time in hour from the satellite (AHI) file name

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

def check_continuity(data, threshold=0.1):

    data = np.append(data[0], data)

    return (np.abs(data[1:]-data[:-1]) < threshold)

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

        logic      = (flt_trk['jday']>=jday_edges[i]) & (flt_trk['jday']<jday_edges[i+1]) & (np.logical_not(np.isnan(flt_trk['sza'])))
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




def cal_mca_flux(
        index,
        fname_sat,
        extent,
        solar_zenith_angle,
        cloud_top_height=None,
        fdir='tmp-data',
        wavelength=745.0,
        date=datetime.datetime.now(),
        target='flux',
        solver='3D',
        photons=5e6,
        Ncpu=14,
        overwrite=True,
        quiet=False
        ):

    """
    flux simulation using EaR3T for flight track based on AHI cloud retrievals
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
    fname_cld = '%s/cld_ahi_%3.3d.pk' % (fdir, index)

    if overwrite:
        ahi0      = ahi_l2(fnames=[fname_sat], extent=extent, vnames=['cld_height_acha'])
        lon_2d, lat_2d, cot_2d = grid_modis_by_extent(ahi0.data['lon']['data'], ahi0.data['lat']['data'], ahi0.data['cot']['data'], extent=extent)
        lon_2d, lat_2d, cer_2d = grid_modis_by_extent(ahi0.data['lon']['data'], ahi0.data['lat']['data'], ahi0.data['cer']['data'], extent=extent)
        cot_2d[cot_2d>100.0] = 100.0
        cer_2d[cer_2d==0.0] = 1.0
        ahi0.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
        ahi0.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
        ahi0.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d)
        ahi0.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d)

        if cloud_top_height is None:
            lon_2d, lat_2d, cth_2d = grid_modis_by_extent(ahi0.data['lon']['data'], ahi0.data['lat']['data'], ahi0.data['cld_height_acha']['data'], extent=extent)
            cth_2d[cth_2d<0.0]  = 0.0; cth_2d /= 1000.0
            ahi0.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d)
            cloud_top_height = ahi0.data['cth_2d']['data']
        cld0 = cld_sat(sat_obj=ahi0, fname=fname_cld, cth=cloud_top_height, cgt=1.0, dz=(levels[1]-levels[0]), overwrite=overwrite)
    else:
        cld0 = cld_sat(fname=fname_cld, overwrite=overwrite)
    # ----------------------------------------------------------------------------------------------------

    # mie scattering phase function setup
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
            fdir='%s/ahi/%s/%3.3d' % (fdir, solver.lower(), index),
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
    out0 = mca_out_ng(fname='%s/mca-out-%s-%s_ahi_%3.3d.h5' % (fdir, target.lower(), solver.lower(), index), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, quiet=quiet, overwrite=overwrite)
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
            photons=2e6,
            Ncpu=16,
            fdir='tmp-data/03_spns_rad-sim',
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
        self.fdir      = fdir
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

            atm0, cld_ahi0, mca_out_ipa0 = cal_mca_flux(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='IPA', fdir=self.fdir, photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)
            atm0, cld_ahi0, mca_out_3d0  = cal_mca_flux(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='3D' , fdir=self.fdir, photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)

            self.sat_imgs[i]['lon'] = cld_ahi0.lay['lon']['data']
            self.sat_imgs[i]['lat'] = cld_ahi0.lay['lat']['data']
            self.sat_imgs[i]['cot'] = cld_ahi0.lay['cot']['data'] # cloud optical thickness (cot) is 2D (x, y)
            self.sat_imgs[i]['cer'] = cld_ahi0.lay['cer']['data'] # cloud effective radius (cer) is 3D (x, y, z)

            lon_sat = self.sat_imgs[i]['lon'][:, 0]
            lat_sat = self.sat_imgs[i]['lat'][0, :]
            dlon    = lon_sat[1]-lon_sat[0]
            dlat    = lat_sat[1]-lat_sat[0]
            lon_trk = self.flt_trks[i]['lon']
            lat_trk = self.flt_trks[i]['lat']
            indices_lon = np.int_(np.round((lon_trk-lon_sat[0])/dlon, decimals=0))
            indices_lat = np.int_(np.round((lat_trk-lat_sat[0])/dlat, decimals=0))
            self.flt_trks[i]['cot'] = self.sat_imgs[i]['cot'][indices_lon, indices_lat]
            self.flt_trks[i]['cer'] = self.sat_imgs[i]['cer'][indices_lon, indices_lat, 0]

            if 'cth' in cld_ahi0.lay.keys():
                self.sat_imgs[i]['cth'] = cld_ahi0.lay['cth']['data']
                self.flt_trks[i]['cth'] = self.sat_imgs[i]['cth'][indices_lon, indices_lat]

            data_3d_mca = {
                'lon'         : cld_ahi0.lay['lon']['data'][:, 0],
                'lat'         : cld_ahi0.lay['lat']['data'][0, :],
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

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)




def main_run(
        date=datetime.datetime(2019, 9, 20),
        wavelength=745.0,
        spns=True,
        run_rtm=True,
        fdir_sat='data/03_spns_flux-sim/aux/ahi',
        fdir_flt='data/03_spns_flux-sim/aux'):


    # create data directory (for storing data) if the directory does not exist
    # ==================================================================================================
    date_s   = date.strftime('%Y%m%d')
    name_tag = __file__.replace('.py', '')

    fdir = os.path.abspath('tmp-data/%s/%s/%09.4fnm' % (name_tag, date_s, wavelength))
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    # ==================================================================================================


    # pre-process the aircraft and satellite data
    # ==================================================================================================
    # get the avaiable satellite data (AHI) and calculate the time in hour for each file
    fnames_ahi = sorted(glob.glob('%s/*.nc' % (fdir_sat)))
    jday_ahi   = get_jday_ahi(fnames_ahi)

    # read in flight data
    fname_flt = 'data/%s/aux/spns_%s.h5' % (name_tag, date_s)
    if not os.path.exists(fname_flt):
        sys.exit('Error   [main_pre]: cannot locate flight data.')

    # create a filter to remove invalid data, e.g., out of available satellite data time range,
    # invalid solar zenith angles etc.
    f_flt = h5py.File(fname_flt, 'r')
    jday   = f_flt['jday'][...]
    sza    = f_flt['sza'][...]
    lon    = f_flt['lon'][...]
    lat    = f_flt['lat'][...]
    logic = (jday>=jday_ahi[0]) & (jday<=jday_ahi[-1]) & \
            (np.logical_not(np.isnan(jday))) & (np.logical_not(np.isnan(sza))) & \
            check_continuity(lon) & check_continuity(lat)

    # create python dictionary to store qualified flight data
    flt_trk = {}
    flt_trk['jday'] = jday[logic]
    flt_trk['lon']  = lon[logic]
    flt_trk['lat']  = lat[logic]
    flt_trk['sza']  = sza[logic]
    flt_trk['tmhr'] = f_flt['tmhr'][...][logic]
    flt_trk['alt']  = f_flt['alt'][...][logic]/1000.0

    if spns:
        flt_trk['f-down-diffuse_spns']= f_flt['spns_dif'][:, np.argmin(np.abs(f_flt['spns_wvl'][...]-wavelength))][logic]
        flt_trk['f-down_spns']= f_flt['spns_tot'][:, np.argmin(np.abs(f_flt['spns_wvl'][...]-wavelength))][logic]

    f_flt.close()

    # partition the flight track into multiple mini flight track segments
    flt_trks = partition_flight_track(flt_trk, tmhr_interval=0.05, margin_x=1.0, margin_y=1.0)

    # create python dictionary to store corresponding satellite imagery data info
    sat_imgs = []
    for i in range(len(flt_trks)):
        sat_img = {}

        index0  = np.argmin(np.abs(jday_ahi-flt_trks[i]['jday0']))
        sat_img['fname']  = fnames_ahi[index0]
        sat_img['extent'] = flt_trks[i]['extent']

        sat_imgs.append(sat_img)
    # ==================================================================================================


    # EaR3T simulation setup for the flight track
    # ==================================================================================================
    sim0 = flt_sim(
            date=date,
            wavelength=wavelength,
            flt_trks=flt_trks,
            sat_imgs=sat_imgs,
            fname='data/03_spns_flux-sim/flt_sim_%s_%09.4fnm.pk' % (date_s, wavelength),
            fdir=fdir,
            overwrite=True,
            overwrite_rtm=run_rtm
            )
    # ==================================================================================================

def main_post(
        date=datetime.datetime(2019, 9, 20),
        wavelength=745.0,
        vnames=['jday', 'lon', 'lat', 'sza', \
            'tmhr', 'alt', 'f-down-diffuse_spns', 'f-down_spns', \
            'cot', 'cer', 'cth', \
            'f-down_mca-3d', 'f-down-diffuse_mca-3d', 'f-down-direct_mca-3d', 'f-up_mca-3d',\
            'f-down_mca-3d-alt-all', 'f-down-diffuse_mca-3d-alt-all', 'f-down-direct_mca-3d-alt-all', 'f-up_mca-3d-alt-all',\
            'f-down_mca-ipa', 'f-down-diffuse_mca-ipa', 'f-down-direct_mca-ipa', 'f-up_mca-ipa'],
        plot=True
        ):

    # data post-processing
    # ==================================================================================================
    date_s = date.strftime('%Y%m%d')

    # aircraft measurements and simulations
    fname      = 'data/03_spns_flux-sim/flt_sim_%s_%09.4fnm.pk' % (date_s, wavelength)
    flt_sim0   = flt_sim(fname=fname)

    # create hdf5 file to store data
    fname_h5   = 'data/03_spns_flux-sim/post-data.h5'
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
    # ==================================================================================================

    if plot:

        f = h5py.File(fname_h5, 'r')
        tmhr = f['tmhr'][...]
        logic = (tmhr>=4.73611111) & (tmhr<=4.91388889)
        lon  = f['lon'][...][logic]
        f_down_spns   = f['f-down_spns'][...][logic]
        f_down_sim_3d = f['f-down_mca-3d'][...][logic]
        f_down_sim_ipa= f['f-down_mca-ipa'][...][logic]
        f.close()

        # =============================================================================
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.scatter(lon, f_down_spns   , color='k', s=4, lw=0.0)
        ax1.scatter(lon, f_down_sim_3d , color='r', s=4, lw=0.0)
        ax1.scatter(lon, f_down_sim_ipa, color='b', s=4, lw=0.0)

        breaks = [120.999, 121.123, 121.247, 121.377]
        for break0 in breaks:
            ax1.axvline(break0, lw=1.5, alpha=0.7, color='gray', ls='--')

        patches_legend = [
                    mpatches.Patch(color='black' , label='SPN-S'),
                    mpatches.Patch(color='red'   , label='RTM 3D'),
                    mpatches.Patch(color='blue'  , label='RTM IPA')
                    ]
        ax1.legend(handles=patches_legend, loc='upper center', fontsize=12)

        ax1.set_ylim((0.0, 2.25))

        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        plt.savefig('03_spns_flux-sim.png', bbox_inches='tight')
        plt.show()
        # =============================================================================




if __name__ == '__main__':

    # Step 1. Pre-process aircraft and satellite data
    #   a. partition flight track into mini flight track segments: stored in `flt_trks`
    #   b. for each mini flight track segment, crop satellite imageries: stored in `sat_imgs`
    #   c. setup simulation runs for the flight track segments
    # =============================================================================
    # main_run(run_rtm=True)
    # =============================================================================

    # Step 2. Post-process radiance observations and simulations for SPN-S, after run
    #   a. <post-data.h5> will be created under data/03_spns_flux-sim
    #   b. <03_spns_flux-sim.png> will be created under current directory
    # =============================================================================
    # main_post(plot=True)
    # =============================================================================

    pass
