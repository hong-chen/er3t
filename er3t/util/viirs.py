import os
import numpy as np
import h5py


import er3t.common
from er3t.util import get_data_nc
from er3t.util.modis import cal_sinusoidal_grid



__all__ = ['viirs_03', 'viirs_l1b', 'viirs_cldprop_l2', 'viirs_09a1', 'viirs_43ma3', 'viirs_43ma4']



VIIRS_ALL_BANDS = {
                    'I01': 640,
                    'I02': 865,
                    'I03': 1610,
                    'I04': 3740,
                    'I05': 11450,
                    'M01': 415,
                    'M02': 445,
                    'M03': 490,
                    'M04': 555,
                    'M05': 673,
                    'M06': 746,
                    'M07': 865,
                    'M08': 1240,
                    'M09': 1378,
                    'M10': 1610,
                    'M11': 2250,
                    'M12': 3700,
                    'M13': 4050,
                    'M14': 8550,
                    'M15': 12013
                    }


VIIRS_L1B_MOD_BANDS = {
                    'M01': 415,
                    'M02': 445,
                    'M03': 490,
                    'M04': 555,
                    'M05': 673,
                    'M06': 746,
                    'M07': 865,
                    'M08': 1240,
                    'M10': 1610,
                    'M11': 2250
                    }

VIIRS_L1B_IMG_BANDS = {
                    'I01': 640,
                    'I02': 865,
                    'I03': 1610
                    }


# reader for VIIRS (Visible Infrared Imaging Radiometer Suite)
#/---------------------------------------------------------------------------\

class viirs_03:

    """
    Read VIIRS 03 geolocation data

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original netCDF file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
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


    def __init__(self,              \
                 fnames,            \
                 extent    = None,  \
                 vnames    = [],    \
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
                lat_range = [lat0.getncattr('valid_min'), lat0.getncattr('valid_max')]
            else:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0, 90.0]

        else:

            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        lon = get_data_nc(lon0)
        lat = get_data_nc(lat0)

        logic = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
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
    Read VIIRS Level 1b file, e.g., VNP02MOD, into an object <viirs_l1b>

    Input:
        fnames=     : list, a Python list of the file paths of the original netCDF files
        f03=        : class instance, default=None, class instance for geolocation (see `class viirs_03`)
        extent=     : list, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        bands=      : list, default=None, a Python list of strings of specific band names. By default, all bands are extracted
        verbose=    : bool, default=False, verbosity tag

    Output:
        self.data
                ['wvl']
                ['rad']
                ['ref']
    """


    ID = 'VIIRS Level 1b Calibrated Radiance'


    def __init__(self,               \
                 fnames,             \
                 f03        = None,  \
                 extent     = None,  \
                 bands      = None,  \
                 verbose    = False):

        self.fnames     = fnames      # Python list of netCDF filenames
        self.f03        = f03         # geolocation class object created using the `viirs_03` reader
        self.bands      = bands       # Python list of bands to extract information


        filename = os.path.basename(fnames[0]).lower()
        if '02img' in filename:
            self.resolution = 0.375
            if bands is None:
                self.bands = list(VIIRS_L1B_IMG_BANDS.keys())
                if verbose:
                    msg = 'Message [viirs_l1b]: Data will be extracted for the following bands %s' % VIIRS_L1B_IMG_BANDS

            elif (bands is not None) and not (set(bands).issubset(set(VIIRS_L1B_IMG_BANDS.keys()))):

                msg = 'Error [viirs_l1b]: Bands must be one or more of %s' % list(VIIRS_L1B_IMG_BANDS.keys())
                raise KeyError(msg)

        elif ('02mod' in filename) or ('02dnb' in filename):
            self.resolution = 0.75
            if bands is None:
                self.bands = list(VIIRS_L1B_MOD_BANDS.keys())
                if verbose:
                    msg = 'Message [viirs_l1b]: Data will be extracted for the following bands %s' % VIIRS_L1B_MOD_BANDS

            elif (bands is not None) and not (set(bands).issubset(set(VIIRS_L1B_MOD_BANDS.keys()))):

                msg = 'Error [viirs_l1b]: Bands must be one or more of %s' % list(VIIRS_L1B_MOD_BANDS.keys())
                raise KeyError(msg)
        else:
            msg = 'Error [viirs_l1b]: Currently, only IMG (0.375km) and MOD (0.75km) products are supported.'
            raise ValueError(msg)


        if extent is not None and verbose:
            msg = '\nMessage [viirs_l1b]: The `extent` argument will be ignored as it is only available for consistency.\n' \
                  'If only region of interest is needed, please use `viirs_03` reader and pass the class object here via `f03=`.\n'
            print(msg)

        if f03 is None and verbose:
            msg = '\nMessage [viirs_l1b]: Geolocation data not provided. File will be read without geolocation.\n'
            print(msg)

        for i in range(len(fnames)):
            self.read(fnames[i])


    def _remove_flags(self, nc_dset, fill_value=np.nan):
        """
        Method to remove all flags without masking.
        This could remove a significant portion of the image.
        """
        nc_dset.set_auto_maskandscale(False)
        data = nc_dset[:]
        data = data.astype('float') # convert to float to use nan
        flags = nc_dset.getncattr('flag_values')

        for flag in flags:
            data[data == flag] = fill_value

        return data


    def _mask_flags(self, nc_dset, fill_value=np.nan):
        """
        Method to keep all flags by masking them with NaN.
        This retains the full image but artifacts exist at extreme swath edges.
        """
        nc_dset.set_auto_scale(False)
        nc_dset.set_auto_mask(True)
        data = nc_dset[:]
        data = np.ma.masked_array(data.data, data.mask, fill_value=fill_value, dtype='float')
        flags = nc_dset.getncattr('flag_values')

        for flag in flags:
            data = np.ma.masked_equal(data.data, flag, copy=False)

        data.filled(fill_value=fill_value)
        return data


    def read(self, fname):

        """
        Read radiance and reflectance from the VIIRS L1b data
        self.data
            ['wvl']
            ['rad']
            ['ref']
        """

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_l1b]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)


        f   = Dataset(fname, 'r')

        # Calculate 1. radiance, 2. reflectance from the raw data
        #/-----------------------------------------------------------------------------\

        if self.f03 is not None:
            mask = self.f03.logic[get_fname_pattern(fname)]['mask']
            rad  = np.zeros((len(self.bands), mask[mask==True].size))
            ref  = np.zeros((len(self.bands), mask[mask==True].size))

        else:
            rad = np.zeros((len(self.bands),
                           f.groups['observation_data'].variables[self.bands[0]].shape[0],
                           f.groups['observation_data'].variables[self.bands[0]].shape[1]))
            ref = np.zeros((len(self.bands),
                           f.groups['observation_data'].variables[self.bands[0]].shape[0],
                           f.groups['observation_data'].variables[self.bands[0]].shape[1]))

        wvl = np.zeros(len(self.bands), dtype='uint16')

        # Calculate 1. radiance, 2. reflectance from the raw data
        #\-----------------------------------------------------------------------------/
        for i in range(len(self.bands)):

            nc_dset = f.groups['observation_data'].variables[self.bands[i]]
            data = self._mask_flags(nc_dset)
            if self.f03 is not None:
                data = data[mask]

            # apply scaling, offset, and unit conversions
            # add_offset is usually 0. for VIIRS solar bands
            rad0 = (data - nc_dset.getncattr('radiance_add_offset')) * nc_dset.getncattr('radiance_scale_factor')

            # if nc_dset.getncattr('radiance_units').endswith('micrometer'):
            rad0 /= 1000. # from <per micron> to <per nm>
            rad[i] = rad0

            ref[i] = (data - nc_dset.getncattr('add_offset')) * nc_dset.getncattr('scale_factor')
            wvl[i] = VIIRS_ALL_BANDS[self.bands[i]]

        f.close()
        #\-----------------------------------------------------------------------------/

        if hasattr(self, 'data'):

            self.data['rad'] = dict(name='Radiance'   , data=np.hstack((self.data['rad']['data'], rad)), units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance', data=np.hstack((self.data['ref']['data'], ref)), units='N/A')

        else:

            self.data = {}
            self.data['wvl'] = dict(name='Wavelengths', data=wvl       , units='nm')
            self.data['rad'] = dict(name='Radiance'   , data=rad       , units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance', data=ref       , units='N/A')


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()





class viirs_cldprop_l2:
    """
    Read VIIRS Level 2 cloud properties/mask file, e.g., CLDPROP_L2_VIIRS_SNPP..., into an object <viirs_cldprop>

    Input:
        fnames=     : list,  a list of the file paths of the original netCDF files
        f03=        : class instance, default=None, class instance for geolocation (see `class viirs_03`)
        extent=     : list, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        maskvars=   : bool, default=False, extracts optical properties by default; set to True to get cloud mask data

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

                or
        self.data
                ['lon']
                ['lat']
                ['ret_std_conf_qa'] => 0: no confidence (do not use), 1: marginal, 2: good, 3: very good
                ['cld_type_qa']     => 0: no cloud mask, 1: no cloud, 2: water cloud, 3: ice cloud, 4: unknown cloud
                ['bowtie_qa']       => 0: normal pixel, 1: bowtie pixel
                ['cloud_mask_flag'] => 0: not determined, 1: determined
                ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
                ['day_night_flag']  => 0: night, 1: day
                ['sunglint_flag']   => 0: in sunglint path, 1: not in sunglint path
                ['snow_ice_flag']   => 0: snow/ice background processing, 1: no snow/ice processing path
                ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
                ['lon_5km']
                ['lat_5km']

    References: (Product Page) https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDPROP_L2_VIIRS_NOAA20
                (User Guide)   https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/Document%20Archive/Science%20Data%20Product%20Documentation/L2_Cloud_Properties_UG_v1.2_March_2021.pdf

    """


    ID = 'VIIRS Level 2 Cloud Properties and Mask'

    def __init__(self, fnames, f03=None, extent=None, maskvars=False):

        self.fnames = fnames
        self.f03    = f03
        self.extent = extent

        for fname in self.fnames:
            if maskvars:
                self.read_mask(fname)
            else:
                self.read_cop(fname)


    def extract_data(self, dbyte, byte=0):
        """
        Extract cloud mask (in byte format) flags and categories
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1) # convert to binary

        if byte == 0:
            # extract flags and categories (*_cat) bit by bit
            land_water_cat  = 2 * data[:, 0] + 1 * data[:, 1] # convert to a value between 0 and 3
            snow_ice_flag   = data[:, 2]
            sunglint_flag   = data[:, 3]
            day_night_flag  = data[:, 4]
            fov_qa_cat      = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3
            cloud_mask_flag = data[:, 7]
            return cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat


    def quality_assurance(self, dbyte, byte=0):
        """
        Extract cloud mask QA data to determine quality

        Reference: VIIRS CLDPROP User Guide, Version 2.1, March 2021
        Filespec:  https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/dump_CLDPROP_L2_V011.txt
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1)

        # process qa flags
        if byte == 0: # byte 0 has spectral retrieval QA

            # 1.6-2.1 retrieval QA
            ret_1621      = data[:, 0]
            ret_1621_conf = 2 * data[:, 1] + 1 * data[:, 2] # convert to a value between 0 and 3 confidence
            ret_1621_data = data[:, 3]

            # VNSWIR-2.1 or Standard (std) retrieval QA
            ret_std      = data[:, 4]
            ret_std_conf = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3 confidence
            ret_std_data = data[:, 7]

            return ret_std, ret_std_conf, ret_std_data, ret_1621, ret_1621_conf, ret_1621_data

        elif byte == 1: # byte 1 has cloud QA

            bowtie            = data[:, 0] # bow tie effect
            cot_oob           = data[:, 1] # cloud optical thickness out of bounds
            cot_bands         = 2 * data[:, 2] + 1 * data[:, 3] # convert to a value between 0 and 3
            rayleigh          = data[:, 4] # whether rayleigh correction was applied
            cld_type_process  = 4 * data[:, 5] + 2 * data[:, 6] + 1 * data[:, 7]

            return cld_type_process, rayleigh, cot_bands, cot_oob, bowtie


    def read_mask(self, fname):
        """
        Extract cloud mask variables from the file
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_cldprop_l2]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        # ------------------------------------------------------------------------------------ #
        f = Dataset(fname, 'r')

        cld_msk0       = f.groups['geophysical_data'].variables['Cloud_Mask']
        qua_assurance0 = f.groups['geophysical_data'].variables['Quality_Assurance']

        #/----------------------------------------------------------------------------\#
        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]

        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        # Select required region only
        if self.f03 is None:

            lat           = f.groups['geolocation_data'].variables['latitude'][...]
            lon           = f.groups['geolocation_data'].variables['longitude'][...]
            logic_extent  = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
                            (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon           = lon[logic_extent]
            lat           = lat[logic_extent]

        else:
            lon          = self.f03.data['lon']['data']
            lat          = self.f03.data['lat']['data']
            logic_extent = self.f03.logic[get_fname_pattern(fname)]['mask']

        # Get cloud mask and flag fields
        #/-----------------------------\#
        cm_data = get_data_nc(cld_msk0)
        qa_data = get_data_nc(qua_assurance0)
        cm = cm_data.copy()
        qa = qa_data.copy()

        cm0 = cm[:, :, 0] # read only the first byte; rest will be supported in the future if needed
        cm0 = np.array(cm0[logic_extent], dtype='uint8')
        cm0 = cm0.reshape((cm0.size, 1))
        cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat = self.extract_data(cm0)

        qa0 = qa[:, :, 0] # read byte 0 for confidence QA (note that indexing is different from MODIS)
        qa0 = np.array(qa0[logic_extent], dtype='uint8')
        qa0 = qa0.reshape((qa0.size, 1))
        _, ret_std_conf_qa, _, _, _, _ = self.quality_assurance(qa0, byte=0) # only get confidence

        qa1 = qa[:, :, 1] # read byte 1 for confidence QA (note that indexing is different from MODIS)
        qa1 = np.array(qa1[logic_extent], dtype='uint8')
        qa1 = qa1.reshape((qa1.size, 1))
        cld_type_qa, _, _, _, bowtie_qa = self.quality_assurance(qa1, byte=1)

        f.close()
        # -------------------------------------------------------------------------------------------------

        # save the data
        if hasattr(self, 'data'):

            self.logic[fname] = {'0.75km':logic_extent}

            self.data['lon']               = dict(name='Longitude',                        data=np.hstack((self.data['lon']['data'], lon)),                                 units='degrees')
            self.data['lat']               = dict(name='Latitude',                         data=np.hstack((self.data['lat']['data'], lat)),                                 units='degrees')
            self.data['ret_std_conf_qa']   = dict(name='QA standard retrieval confidence', data=np.hstack((self.data['ret_std_conf_qa']['data'], ret_std_conf_qa)),         units='N/A')
            self.data['cld_type_qa']       = dict(name='QA cloud type processing path',    data=np.hstack((self.data['cld_type_qa']['data'], cld_type_qa)),                 units='N/A')
            self.data['bowtie_qa']         = dict(name='QA bowtie pixel',                  data=np.hstack((self.data['bowtie_qa']['data'], bowtie_qa)),                     units='N/A')
            self.data['cloud_mask_flag']   = dict(name='Cloud mask flag',                  data=np.hstack((self.data['cloud_mask_flag']['data'], cloud_mask_flag)),         units='N/A')
            self.data['fov_qa_cat']        = dict(name='FOV quality cateogry',             data=np.hstack((self.data['fov_qa_cat']['data'], fov_qa_cat)),                   units='N/A')
            self.data['day_night_flag']    = dict(name='Day/night flag',                   data=np.hstack((self.data['day_night_flag']['data'], day_night_flag)),           units='N/A')
            self.data['sunglint_flag']     = dict(name='Sunglint flag',                    data=np.hstack((self.data['sunglint_flag']['data'], sunglint_flag)),             units='N/A')
            self.data['snow_ice_flag']     = dict(name='Snow/ice flag',                    data=np.hstack((self.data['snow_flag']['data'], snow_ice_flag)),                 units='N/A')
            self.data['land_water_cat']    = dict(name='Land/water flag',                  data=np.hstack((self.data['land_water_cat']['data'], land_water_cat)),           units='N/A')

        else:
            self.logic = {}
            self.logic[fname] = {'0.75km':logic_extent}
            self.data  = {}

            self.data['lon']             = dict(name='Longitude',                        data=lon,             units='degrees')
            self.data['lat']             = dict(name='Latitude',                         data=lat,             units='degrees')
            self.data['ret_std_conf_qa'] = dict(name='QA standard retrieval confidence', data=ret_std_conf_qa, units='N/A')
            self.data['cld_type_qa']     = dict(name='QA cloud type processing path',    data=cld_type_qa,     units='N/A')
            self.data['bowtie_qa']       = dict(name='QA bowtie pixel',                  data=bowtie_qa,       units='N/A')
            self.data['cloud_mask_flag'] = dict(name='Cloud mask flag',                  data=cloud_mask_flag, units='N/A')
            self.data['fov_qa_cat']      = dict(name='FOV quality category',             data=fov_qa_cat,      units='N/A')
            self.data['day_night_flag']  = dict(name='Day/night flag',                   data=day_night_flag,  units='N/A')
            self.data['sunglint_flag']   = dict(name='Sunglint flag',                    data=sunglint_flag,   units='N/A')
            self.data['snow_ice_flag']   = dict(name='Snow/ice flag',                    data=snow_ice_flag,   units='N/A')
            self.data['land_water_cat']  = dict(name='Land/water category',              data=land_water_cat,  units='N/A')


    def read_cop(self, fname):
        """
        Extract cloud optical properties including:
        cloud top height, cloud phase, cloud optical thickness, and cloud effective radius.
        By default, clear-sky restoral parameters are also extracted to fill in the clouds.
        Uncertainties associated with these variables are also included.
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_cldprop_l2]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        # ------------------------------------------------------------------------------------ #
        f = Dataset(fname, 'r')

        #------------------------------------Cloud variables------------------------------------#
        ctp0 = f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties']
        cth0 = f.groups['geophysical_data'].variables['Cloud_Top_Height']

        cot0 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness']
        cer0 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius']
        cwp0 = f.groups['geophysical_data'].variables['Cloud_Water_Path']

        #-------------------------------------PCL variables-------------------------------------#
        cot1 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL']
        cer1 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL']
        cwp1 = f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL']

        #-------------------------------------Uncertainties-------------------------------------#
        cot_uct0 = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_Uncertainty']
        cer_uct0 = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_Uncertainty']
        cwp_uct0 = f.groups['geophysical_data'].variables['Cloud_Water_Path_Uncertainty']


        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0, 90.0]
        else:
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        # Select required region only
        if self.f03 is None:

            lat           = f.groups['geolocation_data'].variables['latitude'][...]
            lon           = f.groups['geolocation_data'].variables['longitude'][...]
            logic_extent  = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
                            (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon           = lon[logic_extent]
            lat           = lat[logic_extent]

        else:
            lon          = self.f03.data['lon']['data']
            lat          = self.f03.data['lat']['data']
            logic_extent = self.f03.logic[get_fname_pattern(fname)]['mask']


        # Retrieve 1. ctp, 2. cth, 3. cot, 4. cer, 5. cwp, and select regional extent
        ctp           = get_data_nc(ctp0, replace_fill_value=None)[logic_extent]
        cth           = get_data_nc(cth0, replace_fill_value=None)[logic_extent]

        cot0_data     = get_data_nc(cot0)[logic_extent]
        cer0_data     = get_data_nc(cer0)[logic_extent]
        cwp0_data     = get_data_nc(cwp0)[logic_extent]

        cot1_data     = get_data_nc(cot1)[logic_extent]
        cer1_data     = get_data_nc(cer1)[logic_extent]
        cwp1_data     = get_data_nc(cwp1)[logic_extent]

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

        pcl = np.zeros_like(cot, dtype=np.uint8)

        # Mark negative (invalid) retrievals with clear-sky values
        logic_invalid          = (cot0_data < 0.0) | (cer0_data < 0.0) | (cwp0_data < 0.0) | (ctp == 0)
        cot[logic_invalid]     = 0.0
        cer[logic_invalid]     = 0.0
        cwp[logic_invalid]     = 0.0
        cot_uct[logic_invalid] = 0.0
        cer_uct[logic_invalid] = 0.0
        cwp_uct[logic_invalid] = 0.0

        # Mark clear-sky pixels using phase as an additional important input
        logic_clear          = ((cot0_data == 0.0) | (cer0_data == 0.0) | (cwp0_data == 0.0)) & (ctp == 1)
        cot[logic_clear]     = 0.0
        cer[logic_clear]     = 0.0
        cwp[logic_clear]     = 0.0

        # Use partially cloudy retrieval to fill in clouds:
        # When the standard retrieval identifies a pixel as being clear-sky AND the corresponding PCL retrieval says it is cloudy,
        # we give credence to the PCL retrieval and mark the pixel with PCL-retrieved values

        logic_pcl      = ((cot0_data == 0.0) | (cer0_data == 0.0) | (cwp0_data == 0.0)) & \
                         ((cot1_data > 0.0)  & (cer1_data > 0.0)  & (cwp1_data > 0.0))

        pcl[logic_pcl] = 1
        cot[logic_pcl] = cot1_data[logic_pcl]
        cer[logic_pcl] = cer1_data[logic_pcl]
        cwp[logic_pcl] = cwp1_data[logic_pcl]

        f.close()
        # ------------------------------------------------------------------------------------ #

        pcl = pcl[logic_extent]

        # save the data
        if hasattr(self, 'data'):

            self.logic[fname] = {'0.75km':logic_extent}

            self.data['lon']      = dict(name='Longitude',                           data=np.hstack((self.data['lon']['data'], lon)),                   units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=np.hstack((self.data['lat']['data'], lat)),                   units='degrees')
            self.data['ctp']      = dict(name='Cloud phase optical proprties',       data=np.hstack((self.data['ctp']['data'], ctp)),                   units='N/A')
            self.data['cth']      = dict(name='Cloud top height',                    data=np.hstack((self.data['cth']['data'], cth)),                   units='m')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=np.hstack((self.data['cot']['data'], cot)),                   units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=np.hstack((self.data['cer']['data'], cer)),                   units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=np.hstack((self.data['cwp']['data'], cwp)),                   units='g/m^2')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=np.hstack((self.data['cot_uct']['data'], cot*cot_uct/100.0)), units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=np.hstack((self.data['cer_uct']['data'], cer*cer_uct/100.0)), units='micron')
            self.data['cwp_uct']  = dict(name='Cloud water path uncertainty',        data=np.hstack((self.data['cwp_uct']['data'], cwp*cwp_uct/100.0)), units='g/m^2')
            self.data['pcl']      = dict(name='PCL tag (1:PCL)',                     data=np.hstack((self.data['pcl']['data'], pcl)),                   units='N/A')

        else:
            self.logic = {}
            self.logic[fname] = {'0.75km':logic_extent}
            self.data  = {}

            self.data['lon']      = dict(name='Longitude',                           data=lon,               units='degrees')
            self.data['lat']      = dict(name='Latitude',                            data=lat,               units='degrees')
            self.data['ctp']      = dict(name='Cloud phase optical properties',      data=ctp,               units='N/A')
            self.data['cth']      = dict(name='Cloud top height',                    data=cth,               units='m')
            self.data['cot']      = dict(name='Cloud optical thickness',             data=cot,               units='N/A')
            self.data['cer']      = dict(name='Cloud effective radius',              data=cer,               units='micron')
            self.data['cwp']      = dict(name='Cloud water path',                    data=cwp,               units='g/m^2')
            self.data['cot_uct']  = dict(name='Cloud optical thickness uncertainty', data=cot*cot_uct/100.0, units='N/A')
            self.data['cer_uct']  = dict(name='Cloud effective radius uncertainty',  data=cer*cer_uct/100.0, units='micron')
            self.data['cwp_uct']  = dict(name='Cloud water path uncertainty',        data=cwp*cwp_uct/100.0, units='g/m^2')
            self.data['pcl']      = dict(name='PCL tag (1:PCL)',                     data=pcl,               units='N/A')




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
                 fnames,
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

def get_fname_pattern(fname, index_s=1, index_e=2):

    filename = os.path.basename(fname)
    pattern  = '.'.join(filename.split('.')[index_s:index_e+1])

    return pattern

#\---------------------------------------------------------------------------/

if __name__=='__main__':

    pass
