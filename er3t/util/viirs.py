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
from er3t.util import check_equal



__all__ = []


# reader for VIIRS (Visible Infrared Imaging Radiometer Suite)
#/---------------------------------------------------------------------------\

class viirs_l1b:

    """
    Read VIIRS Level 1B file into an object `viirs_l1b`

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original HDF4 files
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
                sys.exit('Error   [modis_l1b]: Resolution (in km) is not defined.')
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
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_l1b]: To use \'modis_l1b\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        # when resolution equals to 250 m
        if check_equal(self.resolution, 0.25):
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=4, extra_grid=False)
            raw0      = f.select('EV_250_RefSB')
            wvl       = np.array([650.0, 860.0])

        # when resolution equals to 500 m
        elif check_equal(self.resolution, 0.5):
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=2, extra_grid=False)
            raw0      = f.select('EV_500_RefSB')
            wvl       = np.array([470.0, 555.0, 1240.0, 1640.0, 2130.0])

        # when resolution equals to 1000 m
        elif check_equal(self.resolution, 1.0):
            # lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=False)
            sys.exit('Error   [modis_l1b]: \'resolution=%.1f\' has not been implemented.' % self.resolution)

        else:
            sys.exit('Error   [modis_l1b]: \'resolution=%f\' has not been implemented.' % self.resolution)


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
            print('Message [modis_l1b]: File \'%s\' is created.' % fname)

#\---------------------------------------------------------------------------/





# VIIRS downloader
#/---------------------------------------------------------------------------\


#\---------------------------------------------------------------------------/


if __name__=='__main__':

    pass
