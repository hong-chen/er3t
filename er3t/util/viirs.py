import os
import sys
import datetime
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import shutil
from http.cookiejar import CookieJar
import urllib.request
import requests

import er3t.common
from er3t.util import check_equal



__all__ = ['viirs_03']


# reader for VIIRS (Visible Infrared Imaging Radiometer Suite)
#/---------------------------------------------------------------------------\

class viirs_l1b:

    """
    Read VIIRS Level 1B file into an object `viirs_l1b`

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original netCDF files
        overwrite=  : keyword argument, default=False, whether to overwrite or not
        extent=     : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        resolution= : keyword argument, default=None, data spatial resolution in km, can be detected from filename
        verbose=    : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['rad']
                ['ref']
                ['cnt']
    """


    ID = 'VIIRS Level 1B Calibrated Radiance'


    def __init__(self, \
                 fnames    = None, \
                 extent    = None, \
                 resolution= None, \
                 overwrite = False,\
                 quiet     = True, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the hdf files
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag
        self.quiet      = quiet       # quiet tag


        if resolution is None:
            filename = os.path.basename(fnames[0]).lower()
            if 'qkm' in filename:
                self.resolution = 0.25
            elif 'hkm' in filename:
                self.resolution = 0.5
            elif '1km' in filename:
                self.resolution = 1.0
            else:
                sys.exit('Error   [viirs_l1b]: Resolution (in km) is not defined.')
        else:
            self.resolution = resolution

        for fname in self.fnames:
            self.read(fname)


    def read(self, fname):

        """
        Read radiance/reflectance/corrected counts from the VIIRS L1B data
        self.data
            ['lon']
            ['lat']
            ['rad']
            ['ref']
            ['cnt']
        """

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Warning [viirs_l1b]: To use \'viirs_l1b\', \'netCDF4\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        # when resolution equals to 250 m
        if check_equal(self.resolution, 0.25):
            lon, lat  = upscale_viirs_lonlat(lon0[:], lat0[:], scale=4, extra_grid=False)
            raw0      = f.select('EV_250_RefSB')
            wvl       = np.array([650.0, 860.0])

        # when resolution equals to 500 m
        elif check_equal(self.resolution, 0.5):
            lon, lat  = upscale_viirs_lonlat(lon0[:], lat0[:], scale=2, extra_grid=False)
            raw0      = f.select('EV_500_RefSB')
            wvl       = np.array([470.0, 555.0, 1240.0, 1640.0, 2130.0])

        # when resolution equals to 1000 m
        elif check_equal(self.resolution, 1.0):
            # lon, lat  = upscale_viirs_lonlat(lon0[:], lat0[:], scale=5, extra_grid=False)
            sys.exit('Error   [viirs_l1b]: \'resolution=%.1f\' has not been implemented.' % self.resolution)

        else:
            sys.exit('Error   [viirs_l1b]: \'resolution=%f\' has not been implemented.' % self.resolution)


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.extent is None:

            if 'actual_range' in lon0.attributes().keys():
                lon_range = lon0.attributes()['actual_range']
                lat_range = lat0.attributes()['actual_range']
            elif 'valid_range' in lon0.attributes().keys():
                lon_range = lon0.attributes()['valid_range']
                lat_range = lat0.attributes()['valid_range']
            else:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

        else:

            lon_range = [self.extent[0], self.extent[1]]
            lat_range = [self.extent[2], self.extent[3]]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        # -------------------------------------------------------------------------------------------------


        # Calculate 1. radiance, 2. reflectance, 3. corrected counts from the raw data
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raw = raw0[:][:, logic]
        rad = np.zeros(raw.shape, dtype=np.float64)
        ref = np.zeros(raw.shape, dtype=np.float64)
        cnt = np.zeros(raw.shape, dtype=np.float64)

        for i in range(raw.shape[0]):

            rad[i, ...]  = raw[i, ...]*raw0.attributes()['radiance_scales'][i]         + raw0.attributes()['radiance_offsets'][i]
            rad[i, ...] /= 1000.0 # convert to W/m^2/nm/sr
            ref[i, ...]  = raw[i, ...]*raw0.attributes()['reflectance_scales'][i]      + raw0.attributes()['reflectance_offsets'][i]
            cnt[i, ...]  = raw[i, ...]*raw0.attributes()['corrected_counts_scales'][i] + raw0.attributes()['corrected_counts_offsets'][i]

        f.end()
        # -------------------------------------------------------------------------------------------------



        if hasattr(self, 'data'):

            self.data['lon'] = dict(name='Longitude'        , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'         , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['rad'] = dict(name='Radiance'         , data=np.hstack((self.data['rad']['data'], rad)), units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance'      , data=np.hstack((self.data['ref']['data'], ref)), units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts' , data=np.hstack((self.data['cnt']['data'], cnt)), units='N/A')


        else:

            self.data = {}
            self.data['lon'] = dict(name='Longitude'        , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'         , data=lat, units='degrees')
            self.data['wvl'] = dict(name='Wavelength'       , data=wvl, units='nm')
            self.data['rad'] = dict(name='Radiance'         , data=rad, units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance'      , data=ref, units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts' , data=cnt, units='N/A')


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()

        if not self.quiet:
            print('Message [viirs_l1b]: File \'%s\' is created.' % fname)



class viirs_03:

    """
    Read VIIRS 03 geolocation data

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original netCDF file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['sza']
                ['saa']
                ['vza']
                ['vaa']
    """


    ID = 'VIIRS 03 Geolocation Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 overwrite = False, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)


    def read(self, fname):

        """
        Read solar and sensor angles

        self.data
            ['lon']
            ['lat']
            ['sza']
            ['saa']
            ['vza']
            ['vaa']

        self.logic
        """

        if not er3t.common.has_netcdf4:
            msg = 'Error   [viirs_03]: To use \'viirs_03\', \'netCDF4\' needs to be installed.'

        if er3t.common.has_xarray:

            import xarray as xr

            with xr.open_dataset(fname, group='geolocation_data') as f:

                lon0 = f.longitude
                lat0 = f.latitude

                sza0 = f.solar_zenith
                saa0 = f.solar_azimuth
                vza0 = f.sensor_zenith
                vaa0 = f.sensor_azimuth


            # if self.extent is None:

            #     if 'actual_range' in lon0.attributes().keys():
            #         lon_range = lon0.attributes()['actual_range']
            #         lat_range = lat0.attributes()['actual_range']
            #     elif 'valid_range' in lon0.attributes().keys():
            #         lon_range = lon0.attributes()['valid_range']
            #         lat_range = lat0.attributes()['valid_range']
            #     else:
            #         lon_range = [-180.0, 180.0]
            #         lat_range = [-90.0 , 90.0]

            # else:

            #     lon_range = [self.extent[0], self.extent[1]]
            #     lat_range = [self.extent[2], self.extent[3]]

            # logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
            # lon       = lon[logic]
            # lat       = lat[logic]


        else:
            from netCDF4 import Dataset


        f     = Dataset(fname, 'r')

        # lon lat
        lat0       = f.groups['geolocation_data'].variables['latitude']
        lon0       = f.groups['geolocation_data'].variables['longitude']

        sza0       = f.groups['geolocation_data'].variables['solar_zenith']
        saa0       = f.groups['geolocation_data'].variables['solar_azimuth']
        vza0       = f.groups['geolocation_data'].variables['sensor_zenith']
        vaa0       = f.groups['geolocation_data'].variables['sensor_azimuth']


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #/---------------------------------------------------------------------------\
        lon = lon0[:]
        lat = lat0[:]

        if self.extent is None:

            if 'actual_range' in lon0.attributes().keys():
                lon_range = lon0.attributes()['actual_range']
                lat_range = lat0.attributes()['actual_range']
            elif 'valid_range' in lon0.attributes().keys():
                lon_range = lon0.attributes()['valid_range']
                lat_range = lat0.attributes()['valid_range']
            else:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

        else:

            lon_range = [self.extent[0], self.extent[1]]
            lat_range = [self.extent[2], self.extent[3]]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        #\---------------------------------------------------------------------------/


        # Calculate 1. sza, 2. saa, 3. vza, 4. vaa
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sza0_data = get_data(sza0)
        saa0_data = get_data(saa0)
        vza0_data = get_data(vza0)
        vaa0_data = get_data(vaa0)

        sza = sza0_data[logic]
        saa = saa0_data[logic]
        vza = vza0_data[logic]
        vaa = vaa0_data[logic]

        f.end()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic}

            self.data['lon']   = dict(name='Longitude'                 , data=np.hstack((self.data['lon']['data'], lon    )), units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=np.hstack((self.data['lat']['data'], lat    )), units='degrees')
            self.data['sza']   = dict(name='Solar Zenith Angle'        , data=np.hstack((self.data['sza']['data'], sza    )), units='degrees')
            self.data['saa']   = dict(name='Solar Azimuth Angle'       , data=np.hstack((self.data['saa']['data'], saa    )), units='degrees')
            self.data['vza']   = dict(name='Sensor Zenith Angle'       , data=np.hstack((self.data['vza']['data'], vza    )), units='degrees')
            self.data['vaa']   = dict(name='Sensor Azimuth Angle'      , data=np.hstack((self.data['vaa']['data'], vaa    )), units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic}

            self.data  = {}
            self.data['lon']   = dict(name='Longitude'                 , data=lon    , units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=lat    , units='degrees')
            self.data['sza']   = dict(name='Solar Zenith Angle'        , data=sza    , units='degrees')
            self.data['saa']   = dict(name='Solar Azimuth Angle'       , data=saa    , units='degrees')
            self.data['vza']   = dict(name='Sensor Zenith Angle'       , data=vza    , units='degrees')
            self.data['vaa']   = dict(name='Sensor Azimuth Angle'      , data=vaa    , units='degrees')


    def read_vars(self, fname, vnames=[]):

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Warning [viirs_03]: To use \'viirs_03\', \'netCDF4\' needs to be installed.'
            raise ImportError(msg)

        logic = self.logic[fname]['1km']

        f     = SD(fname, SDC.READ)

        for vname in vnames:

            data0 = f.select(vname)
            data  = get_data(data0)[logic]
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname, data=np.hstack((self.data[vname.lower()]['data'], data)), units=data0.attributes()['units'])
            else:
                self.data[vname.lower()] = dict(name=vname, data=data, units=data0.attributes()['units'])

        f.end()

#\---------------------------------------------------------------------------/





# VIIRS downloader
#/---------------------------------------------------------------------------\

def download_viirs_rgb(
        date,
        extent,
        which='snpp',
        wmts_cgi='https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi',
        fdir='.',
        proj=None,
        coastline=False,
        run=True
        ):

    which  = which.lower()
    date_s = date.strftime('%Y-%m-%d')
    fname  = '%s/%s_rgb_%s_%s.png' % (fdir, which, date_s, '-'.join(['%.2f' % extent0 for extent0 in extent]))

    if run:

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = 'Error   [download_viirs_rgb]: To use \'download_viirs_rgb\', \'matplotlib\' needs to be installed.'
            raise ImportError(msg)

        try:
            from owslib.wmts import WebMapTileService
        except ImportError:
            msg = 'Error   [download_viirs_rgb]: To use \'download_viirs_rgb\', \'owslib\' needs to be installed.'
            raise ImportError(msg)

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error   [download_viirs_rgb]: To use \'download_viirs_rgb\', \'cartopy\' needs to be installed.'
            raise ImportError(msg)

        if which == 'snpp':
            layer_name = 'VIIRS_SNPP_CorrectedReflectance_TrueColor'
        elif which == 'noaa':
            layer_name = 'VIIRS_NOAA20_CorrectedReflectance_TrueColor'
        else:
            sys.exit('Error   [download_viirs_rgb]: Only support \'which="aqua"\' or \'which="terra"\'.')

        if proj is None:
            proj=ccrs.PlateCarree()

        wmts = WebMapTileService(wmts_cgi)

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111, projection=proj)
        ax1.add_wmts(wmts, layer_name, wmts_kwargs={'time': date_s})
        if coastline:
            ax1.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.outline_patch.set_visible(False)
        ax1.axis('off')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    return fname


#\---------------------------------------------------------------------------/


if __name__=='__main__':

    pass