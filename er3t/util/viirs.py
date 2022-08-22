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
from er3t.util import check_equal, get_data_nc



__all__ = ['viirs_03', 'viirs_l1b']


# reader for VIIRS (Visible Infrared Imaging Radiometer Suite)
#/---------------------------------------------------------------------------\

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

        self.fnames     = fnames      # file name of the raw netCDF files
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

        # placeholder for xarray
        #/-----------------------------------------------------------------------------\
        # if er3t.common.has_xarray:
        #     import xarray as xr
        #     with xr.open_dataset(fname, group='geolocation_data') as f:
        #         lon0 = f.longitude
        #         lat0 = f.latitude
        #         sza0 = f.solar_zenith
        #         saa0 = f.solar_azimuth
        #         vza0 = f.sensor_zenith
        #         vaa0 = f.sensor_azimuth
        #\-----------------------------------------------------------------------------/

        if not er3t.common.has_netcdf4:
            msg = 'Error   [viirs_03]: Please install <netCDF4> to proceed.'
            raise OSError(msg)
        else:
            from netCDF4 import Dataset

        f     = Dataset(fname, 'r')

        # geolocation
        #/-----------------------------------------------------------------------------\
        lat0 = f.groups['geolocation_data'].variables['latitude']
        lon0 = f.groups['geolocation_data'].variables['longitude']
        #\-----------------------------------------------------------------------------/

        # only crop necessary data
        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #/-----------------------------------------------------------------------------\
        if self.extent is None:

            if 'valid_min' in lon0.ncattrs():
                lon_range = [lon0.getncattr('valid_min'), lon0.getncattr('valid_max')]
                lat_range = [lon0.getncattr('valid_min'), lon0.getncattr('valid_max')]
            else:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

        else:

            lon_range = [self.extent[0], self.extent[1]]
            lat_range = [self.extent[2], self.extent[3]]

        lon = get_data_nc(lon0)
        lat = get_data_nc(lat0)

        logic = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon   = lon[logic]
        lat   = lat[logic]
        #\-----------------------------------------------------------------------------/

        # solar geometries
        #/-----------------------------------------------------------------------------\
        sza0 = f.groups['geolocation_data'].variables['solar_zenith']
        saa0 = f.groups['geolocation_data'].variables['solar_azimuth']
        #\-----------------------------------------------------------------------------/

        # sensor geometries
        #/-----------------------------------------------------------------------------\
        vza0 = f.groups['geolocation_data'].variables['sensor_zenith']
        vaa0 = f.groups['geolocation_data'].variables['sensor_azimuth']
        #\-----------------------------------------------------------------------------/

        # Calculate 1. sza, 2. saa, 3. vza, 4. vaa
        #/-----------------------------------------------------------------------------\
        sza0_data = get_data_nc(sza0)
        saa0_data = get_data_nc(saa0)
        vza0_data = get_data_nc(vza0)
        vaa0_data = get_data_nc(vaa0)

        sza = sza0_data[logic]
        saa = saa0_data[logic]
        vza = vza0_data[logic]
        vaa = vaa0_data[logic]

        f.close()
        #\-----------------------------------------------------------------------------/

        if hasattr(self, 'data'):

            self.logic[get_fname_pattern(fname)] = {'mask':logic}

            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['sza'] = dict(name='Solar Zenith Angle'  , data=np.hstack((self.data['sza']['data'], sza)), units='degrees')
            self.data['saa'] = dict(name='Solar Azimuth Angle' , data=np.hstack((self.data['saa']['data'], saa)), units='degrees')
            self.data['vza'] = dict(name='Sensor Zenith Angle' , data=np.hstack((self.data['vza']['data'], vza)), units='degrees')
            self.data['vaa'] = dict(name='Sensor Azimuth Angle', data=np.hstack((self.data['vaa']['data'], vaa)), units='degrees')

        else:
            self.logic = {}
            self.logic[get_fname_pattern(fname)] = {'mask':logic}

            self.data  = {}
            self.data['lon'] = dict(name='Longitude'           , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=lat, units='degrees')
            self.data['sza'] = dict(name='Solar Zenith Angle'  , data=sza, units='degrees')
            self.data['saa'] = dict(name='Solar Azimuth Angle' , data=saa, units='degrees')
            self.data['vza'] = dict(name='Sensor Zenith Angle' , data=vza, units='degrees')
            self.data['vaa'] = dict(name='Sensor Azimuth Angle', data=vaa, units='degrees')

    def read_vars(self, fname, vnames=[]):

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_03]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        logic = self.logic[get_fname_pattern(fname)]['mask']

        f = Dataset(fname, 'r')

        for vname in vnames:

            data0 = f.groups['geolocation_data'].variables[vname]
            data  = get_data_nc(data0)
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname.lower().title(), data=np.hstack((self.data[vname.lower()]['data'], data)), units=data0.getncattr('units'))
            else:
                self.data[vname.lower()] = dict(name=vname.lower().title(), data=data, units=data0.getncattr('units'))

        f.close()




class viirs_l1b:

    """
    Read VIIRS Level 1B file into an object <viirs_l1b>

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
                 fnames    = None,  \
                 f03       = None,  \
                 band      = 'M04', \
                 resolution= None,  \
                 overwrite = False, \
                 quiet     = True,  \
                 verbose   = False):

        self.fnames     = fnames      # file name of the netCDF files
        self.f03        = f03         # geolocation file
        self.band       = band.upper()# band
        self.verbose    = verbose     # verbose tag
        self.quiet      = quiet       # quiet tag

        wvls = {
                'I01': 640,
                'I02': 865,
                'I03': 1610,
                'M01': 415,
                'M02': 445,
                'M03': 490,
                'M04': 555,
                'M05': 673,
                'M07': 865,
                'M08': 1240,
                'M10': 1610,
                'M11': 2250,
                }
        self.wvl = wvls[self.band]

        if resolution is None:
            filename = os.path.basename(fnames[0]).lower()
            if '02img' in filename:
                self.resolution = 0.375
            elif ('02mod' in filename) or ('02dnb' in filename):
                self.resolution = 0.75
            else:
                msg = 'Error [viirs_l1b]: Resolution (in km) is not defined.'
                raise ValueError(msg)
        else:

            if resolution not in [0.375, 0.75]:
                msg = 'Error [viirs_l1b]: Resolution of %f km is invalid.' % resolution
                raise ValueError(msg)

            self.resolution = resolution

        for fname in self.fnames:
            self.read(fname, self.band)


    def read(self, fname, band):

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
            msg = 'Error [viirs_l1b]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        f = Dataset(fname, 'r')

        raw0 = f.groups['observation_data'].variables[band]
        raw0.set_auto_scale(False)

        # Calculate 1. radiance, 2. reflectance, 3. corrected counts from the raw data
        #/-----------------------------------------------------------------------------\
        if self.f03 is not None:
            raw = raw0[:][self.f03.logic[get_fname_pattern(fname)]['mask']]
        else:
            raw = raw0[:]

        rad = np.zeros(raw.shape, dtype=np.float64)
        ref = np.zeros(raw.shape, dtype=np.float64)

        rad = raw*raw0.getncattr('radiance_scale_factor') + raw0.getncattr('radiance_add_offset')
        rad /= 1000.0 # from <per micron> to <per nm>
        rad.filled(fill_value=np.nan)
        ref = raw*raw0.getncattr('scale_factor') + raw0.getncattr('add_offset')

        f.close()
        #\-----------------------------------------------------------------------------/

        if hasattr(self, 'data'):

            self.data['rad'] = dict(name='Radiance'   , data=np.hstack((self.data['rad']['data'], rad)), units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance', data=np.hstack((self.data['ref']['data'], ref)), units='N/A')

            if self.f03 is not None:
                for vname in self.f03.data.keys():
                    self.data[vname] = self.f03.data[vname]

        else:

            self.data = {}
            self.data['wvl'] = dict(name='Wavelength' , data=self.wvl, units='nm')
            self.data['rad'] = dict(name='Radiance'   , data=rad     , units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance', data=ref     , units='N/A')

            if self.f03 is not None:
                for vname in self.f03.data.keys():
                    self.data[vname] = self.f03.data[vname]


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()

        if not self.quiet:
            print('Message [viirs_l1b]: File <%s> is created.' % fname)

#\---------------------------------------------------------------------------/





# VIIRS tools
#/---------------------------------------------------------------------------\

def get_fname_pattern(fname, index_s=1, index_e=3):

    filename = os.path.basename(fname)
    pattern  = '.'.join(filename.split('.')[index_s:index_e+1])

    return pattern

#\---------------------------------------------------------------------------/


if __name__=='__main__':

    pass
