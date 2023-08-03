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
from er3t.util.modis import cal_sinusoidal_grid



__all__ = ['viirs_03', 'viirs_l1b', 'viirs_09a1', 'viirs_43ma3', 'viirs_43ma4']


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

            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

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
                self.data[vname.lower()] = dict(name=vname.lower().title(), data=np.vstack((self.data[vname.lower()]['data'], data)), units=data0.getncattr('units'))
            else:
                self.data[vname.lower()] = dict(name=vname.lower().title(), data=data, units=data0.getncattr('units'))

        f.close()




class viirs_l1b:

    """
    Read VIIRS Level 1B file, e.g., VNP02MOD, into an object <viirs_l1b>

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




class viirs_cldprop_l2:
    """
    Read VIIRS Level 2 cloud properties file, e.g., CLDPROP_L2_VIIRS_SNPP..., into an object <viirs_cldprop>

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original netCDF files
        extent=     : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]

    Output:
        self.data
                ['lon']
                ['lat']
                ['ctp']
                ['cth']
                ['cot']
                ['cer']
                ['cwp']
                ['cot_uct']
                ['cer_uct']
                ['cer_uct']
                ['pcl']
    """


    ID = 'VIIRS Level 2 Cloud Properties'
    
    def __init__(self, fnames=None, extent=None):
        
        self.fnames = fnames
        self.extent = extent
        
        for fname in self.fnames:
            self.read(fname)
        

    def read(self, fname):
        
        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_cldprop_l2]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)
            
        # ------------------------------------------------------------------------------------ #
        f = Dataset(fname, 'r')
        
        #----------------------------------------lat/lon----------------------------------------#
        
        lat = f.groups['geolocation_data'].variables['latitude'][...]
        lon = f.groups['geolocation_data'].variables['longitude'][...]
        
        
        #------------------------------------Cloud variables------------------------------------#
        ctp0 = f.groups['geophysical_data'].variables['Cloud_Top_Pressure']
        cth0 = f.groups['geophysical_data'].variables['Cloud_Top_Height']
        
        
        # TODO
        # Support for cloud phase properties (byte format)
        # Support for cloud mask             (byte format)
        cot0 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness']
        cer0 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius']
        cwp0 = f.groups['geophysical_data'].variables['Cloud_Water_Path']
        
        #-------------------------------------PCL variables-------------------------------------#
        cot1 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL']
        cer1 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL']
        cwp1 = f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL']
        
        #-------------------------------------Uncertainties-------------------------------------#
        ctp_uct0 = f.groups['geophysical_data'].variables['Cloud_Top_Pressure_Uncertainty']
        cth_uct0 = f.groups['geophysical_data'].variables['Cloud_Top_Height_Uncertainty']
        cot_uct0 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_Uncertainty']
        cer_uct0 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_Uncertainty']
        cwp_uct0 = f.groups['geophysical_data'].variables['Cloud_Water_Path_Uncertainty']
        
        
        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]
        
        # Select required region only
        logic_extent  = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
                        (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon           = lon[logic_extent]
        lat           = lat[logic_extent]
        
        # Retrieve 1. ctp, 2. cth, 3. cot, 4. cer, 5. cwp, and select regional extent
        ctp           = get_data_nc(ctp0)[logic_extent]
        cth           = get_data_nc(cth0, nan=False)[logic_extent]
        ctp_uct       = get_data_nc(ctp_uct0)[logic_extent]
        cth_uct       = get_data_nc(cth_uct0)[logic_extent]
        
        cot0_data     = get_data_nc(cot0)[logic_extent]
        cer0_data     = get_data_nc(cer0)[logic_extent]
        cwp0_data     = get_data_nc(cwp0, nan=False)[logic_extent]
        cot1_data     = get_data_nc(cot1)[logic_extent]
        cer1_data     = get_data_nc(cer1)[logic_extent]
        cwp1_data     = get_data_nc(cwp1, nan=False)[logic_extent]
        cot_uct0_data = get_data_nc(cot_uct0)[logic_extent]
        cer_uct0_data = get_data_nc(cer_uct0)[logic_extent]  
        cwp_uct0_data = get_data_nc(cwp_uct0)[logic_extent]  
        
        # Make copies to modify
        cot     = cot0_data.copy()
        cer     = cer0_data.copy()
        cwp     = cwp0_data.copy()
        cot_uct = cot_uct0_data.copy()
        cer_uct = cer_uct0_data.copy()
        cwp_uct = cwp_uct0_data.copy()
        
        # use the partially cloudy data to fill in potential missed clouds
        pcl     = np.zeros_like(cot, dtype=np.uint8)
        logic_pcl = ((cot0_data < 0.0) | (cer0_data <= 0.0) | (cwp0_data <= 0.0)) &\
                    ((cot1_data >= 0.0) & (cer1_data > 0.0) & (cwp1_data > 0.0))
        
        pcl[logic_pcl] = 1
        cot[logic_pcl] = cot1_data[logic_pcl]
        cer[logic_pcl] = cer1_data[logic_pcl]
        cwp[logic_pcl] = cwp1_data[logic_pcl]
        
        # make invalid pixels clear-sky
        logic_invalid = (cot < 0.0) | (cer <= 0.0) | (cwp <= 0.0)
        cot[logic_invalid]     = 0.0
        cer[logic_invalid]     = 1.0
        cwp[logic_invalid]     = 1.0
        cot_uct[logic_invalid] = 0.0
        cer_uct[logic_invalid] = 0.0
        cwp_uct[logic_invalid] = 0.0
        
        f.close()
        # ------------------------------------------------------------------------------------ #
        
        # save the data
        if hasattr(self, 'data'):

            self.logic[fname] = {'0.75km':logic_extent}

            self.data['lon']      = dict(name='Longitude',                           data=np.hstack((self.data['lon']['data'], lon)),                   units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=np.hstack((self.data['lat']['data'], lat)),                   units='degrees')
            self.data['ctp']      = dict(name='Cloud top pressure',                  data=np.hstack((self.data['ctp']['data'], ctp)),                   units='mb')
            self.data['cth']      = dict(name='Cloud top height',                    data=np.hstack((self.data['cth']['data'], cth)),                   units='m')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=np.hstack((self.data['cot']['data'], cot)),                   units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=np.hstack((self.data['cer']['data'], cer)),                   units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=np.hstack((self.data['cwp']['data'], cwp)),                   units='g/m^2')
            self.data['ctp_uct']  = dict(name='Cloud top pressure uncertainty',      data=np.hstack((self.data['ctp_uct']['data'], ctp*ctp_uct/100.0)), units='mb')
            self.data['cth_uct']  = dict(name='Cloud top height uncertainty',        data=np.hstack((self.data['cth_uct']['data'], cth*cth_uct/100.0)), units='m')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=np.hstack((self.data['cot_uct']['data'], cot*cot_uct/100.0)), units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=np.hstack((self.data['cer_uct']['data'], cer*cer_uct/100.0)), units='micron')
            self.data['cer_uct']  = dict(name='Cloud water path uncertainty',        data=np.hstack((self.data['cwp_uct']['data'], cwp*cwp_uct/100.0)), units='g/m^2')
            self.data['pcl']      = dict(name='PCL tag (1:PCL, 0:Cloudy)',           data=np.hstack((self.data['pcl']['data'], pcl)),                   units='N/A')

        else:
            self.logic = {}
            self.logic[fname] = {'0.75km':logic_extent}
            self.data  = {}
            
            self.data['lon']      = dict(name='Longitude',                           data=lon,               units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=lat,               units='degrees')
            self.data['ctp']      = dict(name='Cloud top pressure',                  data=ctp,               units='mb')
            self.data['cth']      = dict(name='Cloud top height',                    data=cth,               units='m')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=cot,               units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=cer,               units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=cwp,               units='g/m^2')
            self.data['ctp_uct']  = dict(name='Cloud top pressure uncertainty',      data=ctp*ctp_uct/100.0, units='mb')
            self.data['cth_uct']  = dict(name='Cloud top height uncertainty',        data=cth*cth_uct/100.0, units='m')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=cot*cot_uct/100.0, units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=cer*cer_uct/100.0, units='micron')
            self.data['cer_uct']  = dict(name='Cloud water path uncertainty',        data=cwp*cwp_uct/100.0, units='g/m^2')
            self.data['pcl']      = dict(name='PCL tag (1:PCL, 0:Cloudy)',           data=pcl,               units='N/A')
            



class viirs_09a1:

    """
    Read VIIRS surface reflectance product (8 day of surface reflectance in sinusoidal projection), e.g., VNP09A1

    Input:
        fnames=   : keyword argument, default=None, a Python list of the file path of the files
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        Nx=       : keyword argument, default=2400, number of points along x direction
        Ny=       : keyword argument, default=2400, number of points along y direction
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['ref']: surface reflectance, all 7 channels
                ['lon']: longitude
                ['lat']: latitude
                ['x']  : sinusoidal x
                ['y']  : sinusoidal y
    """


    ID = 'VIIRS surface reflectance (1 km, 8 day)'


    def __init__(self,
                 fnames=None,
                 extent=None,
                 band='M4',
                 Nx=1200,
                 Ny=1200,
                 verbose=False):

        self.fnames = fnames
        self.extent = extent
        self.band = band.upper().replace('M0', 'M')
        self.Nx = Nx
        self.Ny = Ny

        for fname in self.fnames:
            self.read(fname, self.band)


    def read(self, fname, band):

        filename     = os.path.basename(fname)
        index_str    = filename.split('.')[2]
        index_h = int(index_str[1:3])
        index_v = int(index_str[4:])

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error [viirs_09a1]: Please install <cartopy> to proceed.'
            raise ImportError(msg)

        # grid boxes
        proj_xy     = ccrs.Sinusoidal.MODIS
        proj_lonlat = ccrs.PlateCarree()

        x0, y0 = cal_sinusoidal_grid()

        box = [x0[index_h], x0[index_h+1], y0[index_v], y0[index_v+1]]

        # Lon, Lat for the tile
        x_tmp = np.linspace(box[0], box[1], self.Nx+1)
        y_tmp = np.linspace(box[2], box[3], self.Ny+1)
        x_mid = (x_tmp[1:]+x_tmp[:-1])/2.0
        y_mid = (y_tmp[1:]+y_tmp[:-1])/2.0
        XX, YY = np.meshgrid(x_mid, y_mid)

        LonLat = proj_lonlat.transform_points(proj_xy, XX, YY)

        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        lon   = LonLat[..., 0]
        lat   = LonLat[..., 1]

        logic = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])

        lon = lon[logic]
        lat = lat[logic]
        x   = XX[logic]
        y   = YY[logic]

        if self.extent is None:
            self.extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_09a1]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        f = Dataset(fname, 'r')
        dset = f.groups['HDFEOS'].groups['GRIDS'].groups['VNP_Grid_1km_L3_2d'].groups['Data Fields'].variables['SurfReflect_%s' % band]
        dset.set_auto_maskandscale(True)
        ref = np.ma.getdata(dset[:])[logic]
        f.close()
        #\-----------------------------------------------------------------------------/

        if hasattr(self, 'data'):
            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=np.hstack((self.data['x']['data'], x))    , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=np.hstack((self.data['y']['data'], y))    , units='m')
            self.data['ref'] = dict(name='Surface reflectance' , data=np.hstack((self.data['ref']['data'], ref)), units='N/A')
        else:
            self.data = {}
            self.data['lon'] = dict(name='Longitude'           , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=lat, units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=x  , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=y  , units='m')
            self.data['ref'] = dict(name='Surface reflectance' , data=ref, units='N/A')




class viirs_43ma3:

    """
    Read VNP43MA3 product (surface albedo in sinusoidal projection)

    Input:
        fnames=   : keyword argument, default=None, a Python list of the file path of the files
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        Nx=       : keyword argument, default=1200, number of points along x direction
        Ny=       : keyword argument, default=1200, number of points along y direction
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['bsa']: blue/black-sky surface albedo, all 7 channels
                ['wsa']: white-sky surface albedo, all 7 channels
                ['lon']: longitude
                ['lat']: latitude
                ['x']  : sinusoidal x
                ['y']  : sinusoidal y
    """


    ID = 'VIIRS surface albedo (1 km)'


    def __init__(self,
                 fnames=None,
                 extent=None,
                 channels=['M4'],
                 Nx=1200,
                 Ny=1200,
                 verbose=False):

        self.fnames = fnames
        self.extent = extent
        self.channels = channels
        self.Nx = Nx
        self.Ny = Ny

        for fname in self.fnames:
            self.read(fname, channels=channels)


    def read(self, fname, channels=['M4']):

        filename     = os.path.basename(fname)
        index_str    = filename.split('.')[2]
        index_h = int(index_str[1:3])
        index_v = int(index_str[4:])

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error [viirs_43ma3]: To use <viirs_43ma3>, <cartopy> needs to be installed.'
            raise ImportError(msg)

        # grid boxes
        proj_xy     = ccrs.Sinusoidal.MODIS
        proj_lonlat = ccrs.PlateCarree()

        x0, y0 = cal_sinusoidal_grid()

        box = [x0[index_h], x0[index_h+1], y0[index_v], y0[index_v+1]]

        # Lon, Lat for the tile
        x_tmp = np.linspace(box[0], box[1], self.Nx+1)
        y_tmp = np.linspace(box[2], box[3], self.Ny+1)
        x_mid = (x_tmp[1:]+x_tmp[:-1])/2.0
        y_mid = (y_tmp[1:]+y_tmp[:-1])/2.0
        XX, YY = np.meshgrid(x_mid, y_mid)

        LonLat = proj_lonlat.transform_points(proj_xy, XX, YY)

        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        lon   = LonLat[..., 0]
        lat   = LonLat[..., 1]

        logic = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])

        lon = lon[logic]
        lat = lat[logic]
        x   = XX[logic]
        y   = YY[logic]

        if self.extent is None:
            self.extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_43ma3]: To use <viirs_43ma3>, <netCDF4> needs to be installed.'
            raise ImportError(msg)

        f     = Dataset(fname, 'r')

        Nchan = len(channels)
        bsky_alb = np.zeros((Nchan, logic.sum()), dtype=np.float64)
        wsky_alb = np.zeros((Nchan, logic.sum()), dtype=np.float64)

        for ichan in range(Nchan):
            data0 = f.groups['HDFEOS'].groups['GRIDS'].groups['VIIRS_Grid_BRDF'].groups['Data Fields'].variables['Albedo_BSA_%s' % channels[ichan]]
            data = get_data_nc(data0)
            bsky_alb[ichan, :] = data[logic]

            data0 = f.groups['HDFEOS'].groups['GRIDS'].groups['VIIRS_Grid_BRDF'].groups['Data Fields'].variables['Albedo_WSA_%s' % channels[ichan]]
            data = get_data_nc(data0)
            wsky_alb[ichan, :] = data[logic]

        bsky_alb[bsky_alb>1.0] = -1.0
        bsky_alb[bsky_alb<0.0] = -1.0

        wsky_alb[wsky_alb>1.0] = -1.0
        wsky_alb[wsky_alb<0.0] = -1.0

        f.close()

        if hasattr(self, 'data'):
            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=np.hstack((self.data['x']['data'], x))    , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=np.hstack((self.data['y']['data'], y))    , units='m')
            self.data['bsa'] = dict(name='Blue/Black-Sky surface albedo', data=np.hstack((self.data['bsa']['data'], bsky_alb)), units='N/A')
            self.data['wsa'] = dict(name='White-Sky surface albedo'     , data=np.hstack((self.data['wsa']['data'], wsky_alb)), units='N/A')
        else:
            self.data = {}
            self.data['lon'] = dict(name='Longitude'           , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=lat, units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=x  , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=y  , units='m')
            self.data['bsa'] = dict(name='Blue/Black-Sky surface albedo', data=bsky_alb, units='N/A')
            self.data['wsa'] = dict(name='White-Sky surface albedo'     , data=wsky_alb, units='N/A')




class viirs_43ma4:

    """
    Read VNP43MA4 product (surface reflectance at nadir in sinusoidal projection)

    Input:
        fnames=   : keyword argument, default=None, a Python list of the file path of the files
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        Nx=       : keyword argument, default=1200, number of points along x direction
        Ny=       : keyword argument, default=1200, number of points along y direction
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['ref']: surface reflectance at nadir
                ['lon']: longitude
                ['lat']: latitude
                ['x']  : sinusoidal x
                ['y']  : sinusoidal y
    """


    ID = 'VIIRS surface reflectance (1 km)'


    def __init__(self,
                 fnames=None,
                 extent=None,
                 channels=['M4'],
                 Nx=1200,
                 Ny=1200,
                 verbose=False):

        self.fnames = fnames
        self.extent = extent
        self.channels = channels
        self.Nx = Nx
        self.Ny = Ny

        for fname in self.fnames:
            self.read(fname, channels=channels)


    def read(self, fname, channels=['M4']):

        filename     = os.path.basename(fname)
        index_str    = filename.split('.')[2]
        index_h = int(index_str[1:3])
        index_v = int(index_str[4:])

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error [viirs_43ma4]: To use <viirs_43ma4>, <cartopy> needs to be installed.'
            raise ImportError(msg)

        # grid boxes
        proj_xy     = ccrs.Sinusoidal.MODIS
        proj_lonlat = ccrs.PlateCarree()

        x0, y0 = cal_sinusoidal_grid()

        box = [x0[index_h], x0[index_h+1], y0[index_v], y0[index_v+1]]

        # Lon, Lat for the tile
        x_tmp = np.linspace(box[0], box[1], self.Nx+1)
        y_tmp = np.linspace(box[2], box[3], self.Ny+1)
        x_mid = (x_tmp[1:]+x_tmp[:-1])/2.0
        y_mid = (y_tmp[1:]+y_tmp[:-1])/2.0
        XX, YY = np.meshgrid(x_mid, y_mid)

        LonLat = proj_lonlat.transform_points(proj_xy, XX, YY)

        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        lon   = LonLat[..., 0]
        lat   = LonLat[..., 1]

        logic = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])

        lon = lon[logic]
        lat = lat[logic]
        x   = XX[logic]
        y   = YY[logic]

        if self.extent is None:
            self.extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_43ma4]: To use <viirs_43ma4>, <netCDF4> needs to be installed.'
            raise ImportError(msg)

        f     = Dataset(fname, 'r')

        Nchan = len(channels)
        sfc_ref = np.zeros((Nchan, logic.sum()), dtype=np.float64)

        for ichan in range(Nchan):
            data0 = f.groups['HDFEOS'].groups['GRIDS'].groups['VIIRS_Grid_BRDF'].groups['Data Fields'].variables['Nadir_Reflectance_%s' % channels[ichan]]
            data = get_data_nc(data0)
            sfc_ref[ichan, :] = data[logic]

        sfc_ref[sfc_ref>1.0] = -1.0
        sfc_ref[sfc_ref<0.0] = -1.0

        f.close()

        if hasattr(self, 'data'):
            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=np.hstack((self.data['x']['data'], x))    , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=np.hstack((self.data['y']['data'], y))    , units='m')
            self.data['ref'] = dict(name='Surface reflectance at nadir', data=np.hstack((self.data['ref']['data'], sfc_ref)), units='N/A')
        else:
            self.data = {}
            self.data['lon'] = dict(name='Longitude'           , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=lat, units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=x  , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=y  , units='m')
            self.data['ref'] = dict(name='Surface reflectance at nadir', data=sfc_ref, units='N/A')

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
