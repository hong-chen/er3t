import os
import numpy as np
import h5py


import er3t.common
from er3t.util import get_data_h4, get_data_nc, unpack_uint_to_bits
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
                    'M15': 10763,
                    'M16': 12013
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
                    'M09': 1378,
                    'M10': 1610,
                    'M11': 2250,
                    'M12': 3700,
                    'M13': 4050,
                    'M14': 8550,
                    'M15': 10763,
                    'M16': 12013
                    }

VIIRS_L1B_IMG_BANDS = {
                    'I01': 640,
                    'I02': 865,
                    'I03': 1610,
                    'I04': 3740,
                    'I05': 11450
                    }


VIIRS_L1B_MOD_EMISSIVE_BANDS = {'M12': 3700,
                                'M13': 4050,
                                'M14': 8550,
                                'M15': 10763,
                                'M16': 12013
                                }

VIIRS_L1B_IMG_EMISSIVE_BANDS = {'I04': 3740,
                                'I05': 11450
                                }

VIIRS_L1B_ALL_EMISSIVE_BANDS = {'M12': 3700,
                                'M13': 4050,
                                'M14': 8550,
                                'M15': 10763,
                                'M16': 12013,
                                'I04': 3740,
                                'I05': 11450
                                }

VIIRS_L1B_MOD_DEFAULT_BANDS = {'M05': 673,
                               'M04': 555,
                               'M02':445
                               }

# reader for VIIRS (Visible Infrared Imaging Radiometer Suite)
#/---------------------------------------------------------------------------\

class viirs_03:

    """
    Read VIIRS 03 geolocation data

    Input:
        fnames=   : list, default=None, Python list of the file path of the original netCDF file
        extent=   : list, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : list, default=[], additional variable names to be read in to self.data
        verbose=  : bool, default=False, verbose tag
        keep_dims=: bool, default=False, set to True to get full granule, False to apply geomask of extent

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
                 keep_dims = False, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the raw netCDF files
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag
        self.keep_dims  = keep_dims   # flag; if false -> apply geomask to get 1D; true -> retain 2D granule data

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
        if not self.keep_dims:
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

        sza = get_data_nc(sza0)
        saa = get_data_nc(saa0)
        vza = get_data_nc(vza0)
        vaa = get_data_nc(vaa0)

        if not self.keep_dims:
            sza = sza[logic]
            saa = saa[logic]
            vza = vza[logic]
            vaa = vaa[logic]

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
        keep_dims=  : bool, default=False, set to True to get full granule, False to apply geomask of extent

    Output:
        self.data
                ['wvl']
                ['rad']
                ['ref']
    """


    ID = 'VIIRS Level 1b Calibrated Radiance'


    def __init__(self, \
                 fnames     = None,  \
                 f03        = None,  \
                 extent     = None,  \
                 bands      = None,  \
                 keep_dims  = False, \
                 verbose    = False):

        self.fnames     = fnames      # Python list of netCDF filenames
        self.f03        = f03         # geolocation class object created using the `viirs_03` reader
        self.bands      = bands       # Python list of bands to extract information
        self.keep_dims  = keep_dims   # retain 2D shape; if True -> f03 mask is not applied


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
                self.bands = list(VIIRS_L1B_MOD_DEFAULT_BANDS.keys())
                if verbose:
                    msg = 'Message [viirs_l1b]: Data will be extracted for the following bands %s' % VIIRS_L1B_MOD_DEFAULT_BANDS

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

        if (self.keep_dims) or (self.f03 is None):
            rad = np.zeros((len(self.bands),
                           f.groups['observation_data'].variables[self.bands[0]].shape[0],
                           f.groups['observation_data'].variables[self.bands[0]].shape[1]))
            ref = np.zeros((len(self.bands),
                           f.groups['observation_data'].variables[self.bands[0]].shape[0],
                           f.groups['observation_data'].variables[self.bands[0]].shape[1]))

        else:
            mask = self.f03.logic[get_fname_pattern(fname)]['mask']
            rad  = np.zeros((len(self.bands), mask.sum()))
            ref  = np.zeros((len(self.bands), mask.sum()))


        wvl = np.zeros(len(self.bands), dtype='uint16')

        ## Calculate 1. radiance, 2. reflectance from the raw data
        #\-----------------------------------------------------------------------------/
        for i in range(len(self.bands)):

            nc_dset = f.groups['observation_data'].variables[self.bands[i]]
            data = self._remove_flags(nc_dset)
            if not self.keep_dims:
                data = data[mask]

            # apply scaling, offset, and unit conversions
            # add_offset is usually 0 for VIIRS solar bands
            if hasattr(nc_dset, 'radiance_add_offset'):
                rad0 = (data - nc_dset.getncattr('radiance_add_offset')) * nc_dset.getncattr('radiance_scale_factor') # radiance
                ref[i] = (data - nc_dset.getncattr('add_offset')) * nc_dset.getncattr('scale_factor') # reflectance

            else: # naming convention changes for emissive bands
                rad0 = (data - nc_dset.getncattr('add_offset')) * nc_dset.getncattr('scale_factor') # radiance
                ref[i] = np.full(ref[i].shape, -99, dtype='int8') # make reflectance -99 for emissive bands

            rad0 /= 1000. # from <per micron> to <per nm>
            rad[i] = rad0
            wvl[i] = VIIRS_ALL_BANDS[self.bands[i]]

        f.close()
        #\-----------------------------------------------------------------------------/
        if hasattr(self, 'data'):
            self.data['wvl'] = dict(name='Wavelengths', data=np.hstack((self.data['wvl']['data'], wvl)), units='nm')
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
                ['cloud_mask_flag'] => 0: not determined, 1: determined
                ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
                ['day_night_flag']  => 0: night, 1: day
                ['sunglint_flag']   => 0: in sunglint path, 1: not in sunglint path
                ['snow_ice_flag']   => 0: snow/ice background processing, 1: no snow/ice processing path
                ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
                ['lon_5km']
                ['lat_5km']
        self.qa
                ['ret_std_conf_qa'] => 0: no confidence (do not use), 1: marginal, 2: good, 3: very good
                ['cld_type_qa']     => 0: no cloud mask, 1: no cloud, 2: water cloud, 3: ice cloud, 4: unknown cloud
                ['bowtie_qa']       => 0: normal pixel, 1: bowtie pixel
                ...
                more available. refer documentation below.

    References: (Product Page) https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDPROP_L2_VIIRS_NOAA20
                (User Guide)   https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/Document%20Archive/Science%20Data%20Product%20Documentation/L2_Cloud_Properties_UG_v1.2_March_2021.pdf

    """


    ID = 'VIIRS Level 2 Cloud Properties and Mask'

    def __init__(self,
                 fnames,
                 f03=None,
                 extent=None,
                 maskvars=False,
                 quality_assurance=0,
                 keep_dims=True):

        self.fnames            = fnames    # Python list of the file path of the original HDF4 files
        self.f03               = f03       # geolocation class object created using the `viirs_03` reader
        self.extent            = extent    # specified region [westmost, eastmost, southmost, northmost]
        self.quality_assurance = quality_assurance # None or 0 = no QA; 1 = some QA; 2 = all QA (lot of data)
        self.keep_dims         = keep_dims # flag; if false -> convert to 1D; true -> retain 2D granule data

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
            fov_cat      = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3
            cloud_mask_flag = data[:, 7]

            if self.keep_dims:
                return cloud_mask_flag.reshape(self.data_shape), day_night_flag.reshape(self.data_shape), sunglint_flag.reshape(self.data_shape), snow_ice_flag.reshape(self.data_shape), land_water_cat.reshape(self.data_shape), fov_cat.reshape(self.data_shape)
            return cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_cat


    def quality_assurance_byte0(self, dbyte):
        """
        Extract cloud mask QA data to determine quality. Byte 0 only (spectral retrieval QA)

        Reference: VIIRS CLDPROP User Guide, Version 2.1, March 2021
        Filespec:  https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/dump_CLDPROP_L2_V011.txt
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1)

        # process qa flags
        # 1.6-2.1 retrieval QA
        ret_1621      = data[:, 0] # 1621 Retrieval Outcome
        ret_1621_conf = 2 * data[:, 1] + 1 * data[:, 2] # convert to a value between 0 and 3 confidence
        ret_1621_data = data[:, 3] # 1621 Retrieval Spectral Data Availability QA

        # VNSWIR-2.1 or Standard (std) Retrieval QA
        ret_std      = data[:, 4] # Standard Retrieval Outcome
        ret_std_conf = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3 confidence
        ret_std_data = data[:, 7] # Standard Retrieval Spectral Data Availability QA

        if self.keep_dims:
            return ret_std.reshape(self.data_shape), ret_std_conf.reshape(self.data_shape), ret_std_data.reshape(self.data_shape), ret_1621.reshape(self.data_shape), ret_1621_conf.reshape(self.data_shape), ret_1621_data.reshape(self.data_shape)
        return ret_std, ret_std_conf, ret_std_data, ret_1621, ret_1621_conf, ret_1621_data


    def quality_assurance_byte1(self, dbyte):
        """
        Extract cloud mask QA data to determine quality. Byte 1 only (cloud/sfc QA)

        Reference: VIIRS CLDPROP User Guide, Version 2.1, March 2021
        Filespec:  https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/dump_CLDPROP_L2_V011.txt
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1)

        # process qa flags
        bowtie            = data[:, 0] # bow tie effect
        cot_oob           = data[:, 1] # cloud optical thickness out of bounds
        cot_bands         = 2 * data[:, 2] + 1 * data[:, 3] # convert to a value between 0 and 3
        rayleigh          = data[:, 4] # whether rayleigh correction was applied
        cld_type_process  = 4 * data[:, 5] + 2 * data[:, 6] + 1 * data[:, 7]

        if self.keep_dims:
            return cld_type_process.reshape(self.data_shape), rayleigh.reshape(self.data_shape), cot_bands.reshape(self.data_shape), cot_oob.reshape(self.data_shape), bowtie.reshape(self.data_shape)
        return cld_type_process, rayleigh, cot_bands, cot_oob, bowtie


    def read_mask(self, fname):
        """
        Function to extract cloud mask variables from the file
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [viirs_cldprop_l2]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        # ------------------------------------------------------------------------------------ #
        f = Dataset(fname, 'r')

        cld_msk0       = f.groups['geophysical_data'].variables['Cloud_Mask']

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
            if not self.keep_dims:
                lon           = lon[logic_extent]
                lat           = lat[logic_extent]

        else:
            lon          = self.f03.data['lon']['data']
            lat          = self.f03.data['lat']['data']
            logic_extent = self.f03.logic[get_fname_pattern(fname)]['mask']

        # Get cloud mask and flag fields
        #/-----------------------------\#

        cm_data = get_data_nc(cld_msk0)
        cm = cm_data.copy()
        cm0 = cm[:, :, 0] # read only the first byte; rest will be supported in the future if needed
        self.data_shape = cm0.shape # record shape for future

        if not self.keep_dims:
            cm0 = np.array(cm0[logic_extent], dtype='uint8')

        cm0 = cm0.reshape((cm0.size, 1))
        cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_cat = self.extract_data(cm0)

        if (self.quality_assurance is not None) and (self.quality_assurance > 0):
            qua_assurance0 = f.groups['geophysical_data'].variables['Quality_Assurance']
            qa_data = get_data_nc(qua_assurance0)
            qa = qa_data.copy()
            qa0 = qa[:, :, 0] # read byte 0 for confidence QA (note that indexing is different from MODIS)
            qa1 = qa[:, :, 1] # read byte 1 for confidence QA (note that indexing is different from MODIS)


            if not self.keep_dims:
                qa0 = np.array(qa0[logic_extent], dtype='uint8')
                qa1 = np.array(qa1[logic_extent], dtype='uint8')

            qa0 = qa0.reshape((qa0.size, 1))
            qa1 = qa1.reshape((qa1.size, 1))

            ret_std_qa, ret_std_conf_qa, ret_std_data_qa, ret_1621_qa, ret_1621_conf_qa, ret_1621_data_qa = self.quality_assurance_byte0(qa0)
            cld_type_qa, rayleigh_qa, cot_bands_qa, cot_oob_qa, bowtie_qa = self.quality_assurance_byte1(qa1)

        f.close()
        # -------------------------------------------------------------------------------------------------

        # save the data
        if hasattr(self, 'data'):

            self.logic[fname] = {'0.75km':logic_extent}

            self.data['lon']               = dict(name='Longitude',                        data=np.hstack((self.data['lon']['data'], lon)),                               units='degrees')
            self.data['lat']               = dict(name='Latitude',                         data=np.hstack((self.data['lat']['data'], lat)),                               units='degrees')
            self.data['cloud_mask_flag']   = dict(name='Cloud mask flag',                  data=np.hstack((self.data['cloud_mask_flag']['data'], cloud_mask_flag)),         units='N/A')
            self.data['fov_cat']        = dict(name='FOV quality cateogry',                data=np.hstack((self.data['fov_cat']['data'], fov_cat)),                           units='N/A')
            self.data['day_night_flag']    = dict(name='Day/night flag',                   data=np.hstack((self.data['day_night_flag']['data'], day_night_flag)),          units='N/A')
            self.data['sunglint_flag']     = dict(name='Sunglint flag',                    data=np.hstack((self.data['sunglint_flag']['data'], sunglint_flag)),           units='N/A')
            self.data['snow_ice_flag']     = dict(name='Snow/ice flag',                    data=np.hstack((self.data['snow_flag']['data'], snow_ice_flag)),           units='N/A')
            self.data['land_water_cat']    = dict(name='Land/water category',              data=np.hstack((self.data['land_water_cat']['data'], land_water_cat)),          units='N/A')

        else:
            self.logic = {}
            self.logic[fname] = {'0.75km':logic_extent}
            self.data  = {}

            self.data['lon']              = dict(name='Longitude',                        data=lon,              units='degrees')
            self.data['lat']              = dict(name='Latitude',                         data=lat,              units='degrees')
            self.data['cloud_mask_flag']  = dict(name='Cloud mask flag',                  data=cloud_mask_flag,  units='N/A')
            self.data['fov_cat']          = dict(name='FOV quality category',             data=fov_cat,          units='N/A')
            self.data['day_night_flag']   = dict(name='Day/night flag',                   data=day_night_flag,   units='N/A')
            self.data['sunglint_flag']    = dict(name='Sunglint flag',                    data=sunglint_flag,    units='N/A')
            self.data['snow_ice_flag']    = dict(name='Snow/ice flag',                    data=snow_ice_flag,    units='N/A')
            self.data['land_water_cat']   = dict(name='Land/water category',              data=land_water_cat,   units='N/A')


        # Save QA data if required
        if (self.quality_assurance is not None) and (self.quality_assurance > 0):

            if hasattr(self, 'qa'):
                self.qa['ret_std_conf_qa']   = dict(name='QA standard retrieval confidence', data=np.hstack((self.qa['ret_std_conf_qa']['data'], ret_std_conf_qa)),         units='N/A')
                self.qa['ret_1621_conf_qa']  = dict(name='QA 1.6-2.1 retrieval confidence',  data=np.hstack((self.qa['ret_1621_conf_qa']['data'], ret_1621_conf_qa)),       units='N/A')

                if self.quality_assurance > 1:
                    self.qa['ret_std_qa']        = dict(name='QA standard retrieval outcome', data=np.hstack((self.qa['ret_std_qa']['data'], ret_std_qa)),         units='N/A')
                    self.qa['ret_std_qa_data']   = dict(name='QA standard retrieval spectral data availability', data=np.hstack((self.qa['ret_std_data_qa']['data'], ret_std_data_qa)),         units='N/A')
                    self.qa['ret_1621_qa']        = dict(name='QA 1621 retrieval outcome', data=np.hstack((self.qa['ret_1621_qa']['data'], ret_1621_qa)),         units='N/A')
                    self.qa['ret_1621_data_qa']   = dict(name='QA 1621 retrieval spectral data availability', data=np.hstack((self.qa['ret_1621_data_qa']['data'], ret_1621_data_qa)),         units='N/A')
                    self.qa['cld_type_qa']       = dict(name='QA cloud type processing path',    data=np.hstack((self.qa['cld_type_qa']['data'], cld_type_qa)),                 units='N/A')

                    self.qa['bowtie_qa']         = dict(name='QA bowtie pixel',                  data=np.hstack((self.qa['bowtie_qa']['data'], bowtie_qa)),                     units='N/A')

                    self.qa['rayleigh_qa']       = dict(name='QA Rayleigh correction',           data=np.hstack((self.qa['rayleigh_qa']['data'], rayleigh_qa)),                 units='N/A')

                    self.qa['cot_bands_qa']         = dict(name='QA bands used for COT',                  data=np.hstack((self.qa['cot_bands_qa']['data'], cot_bands_qa)),                     units='N/A')

                    self.qa['cot_oob_qa']       = dict(name='QA COT out of bounds',    data=np.hstack((self.qa['cot_oob_qa']['data'], cot_oob_qa)),                 units='N/A')


            else:

                self.qa  = {}
                self.qa['ret_std_conf_qa']  = dict(name='QA standard retrieval confidence', data=ret_std_conf_qa,  units='N/A')
                self.qa['ret_1621_conf_qa'] = dict(name='QA 1.6-2.1 retrieval confidence',  data=ret_1621_conf_qa, units='N/A')
                self.qa['ret_std_qa']       = dict(name='QA standard retrieval outcome',    data=ret_std_qa,       units='N/A')
                self.qa['ret_std_data_qa']  = dict(name='QA standard retrieval spectral data availability',        data=ret_std_data_qa,      units='N/A')
                self.qa['ret_1621_data_qa'] = dict(name='QA 1621 retrieval spectral data availability',  data=ret_1621_data_qa, units='N/A')

                if self.quality_assurance > 1:
                    self.qa['cld_type_qa']      = dict(name='QA cloud type processing path',    data=cld_type_qa,      units='N/A')
                    self.qa['bowtie_qa']        = dict(name='QA bowtie pixel',                  data=bowtie_qa,        units='N/A')
                    self.qa['rayleigh_qa']      = dict(name='QA Rayleigh correction',           data=rayleigh_qa,      units='N/A')
                    self.qa['cot_bands_qa']     = dict(name='QA bands used for COT',            data=cot_bands_qa,     units='N/A')
                    self.qa['cot_oob_qa']       = dict(name='QA COT out of bounds',             data=cot_oob_qa,       units='N/A')

    #########################################################################################################################
    ############################################### Cloud Optical Properties ################################################
    #########################################################################################################################


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



class viirs_09:
    """
    A class for extracting data from VIIRS Atmospherically Corrected Surface Reflectance 6-Min L2 Swath IP 375m, 750m files.

    Args:
        fname (str): The file name of the MOD09 product.
        resolution (str): The resolution of the product ('375m' or '750m'). Defaults to '750m'.
        quality_assurance (str): The type of quality assurance data to retrieve ('auto', 'ancillary', 'all', or None). Defaults to None.
        bands (list, optional): The list of band names. Defaults to ['M05', 'M04', 'M03'].

    # Note: VIIRS 09 products are only available as HDF files (instead of netCDF like most other VIIRS products)

    References: (User Guide): https://viirsland.gsfc.nasa.gov/PDF/VIIRS_Surf_Refl_UserGuide_v2.0.pdf
    """
    ID = 'VIIRS Atmospherically Corrected Surface Reflectance 6-Min L2 Swath IP 375m, 750m'


    def __init__(self, \
                 fname,  \
                 resolution = '750m', \
                 quality_assurance = None, \
                 bands = ['M05', 'M04', 'M03']):


        self.fname             = fname                       # file name
        self.resolution        = resolution.lower()          # resolution
        self.quality_assurance = quality_assurance           # string to get qa data
        self.bands             = list(map(str.upper, bands)) # list of band names; check VIIRS_ALL_BANDS for convention

        self.available_product_bands = ['I01', 'I02', 'I03', 'M01', 'M02', 'M03', 'M04', 'M05', 'M07', 'M08', 'M10', 'M11']
        self.read(fname)

    # Methods to extract each QQ/QF byte
    ######################### QF1 #########################
    def qa_qf1_viirs_09(self, hdf_obj):
        fdata = hdf_obj.select('QF1 Surface Reflectance')[:]
        bits = unpack_uint_to_bits(fdata, 8, bitorder='little')

        cld_msk_qa = 2 * bits[0] + bits[1]
        cld_msk_confidence_qa = 2 * bits[2] + bits[3]
        day_night_qa = bits[4]
        low_sun_qa   = bits[5]
        sunglint_qa  = 2 * bits[6] + bits[7]
        return cld_msk_qa, cld_msk_confidence_qa, day_night_qa, low_sun_qa, sunglint_qa

    ######################### QF2 #########################
    def qa_qf2_viirs_09(self, hdf_obj):
        fdata = hdf_obj.select('QF2 Surface Reflectance')[:]
        bits = unpack_uint_to_bits(fdata, 8, bitorder='little')

        land_water_qa = 4 * bits[0] + 2 * bits[1] + bits[2]
        cld_shadow_qa = bits[3]
        heavy_aerosol_qa = bits[4]
        snow_ice_qa   = bits[5]
        thin_cirrus_solar_qa  = bits[6]
        thin_cirrus_thermal_qa = bits[7]
        return land_water_qa, cld_shadow_qa, heavy_aerosol_qa, snow_ice_qa, thin_cirrus_solar_qa, thin_cirrus_thermal_qa

    ######################### QF3 #########################
    def qa_qf3_viirs_09(self, hdf_obj):
        fdata = hdf_obj.select('QF3 Surface Reflectance')[:]
        bits = unpack_uint_to_bits(fdata, 8, bitorder='little')

        bad_m1_qa = bits[0]
        bad_m2_qa = bits[1]
        bad_m3_qa = bits[2]
        bad_m4_qa = bits[3]
        bad_m5_qa = bits[4]
        bad_m7_qa = bits[5]
        bad_m8_qa = bits[6]
        bad_m10_qa = bits[7]

        return bad_m1_qa, bad_m2_qa, bad_m3_qa, bad_m4_qa, bad_m5_qa, bad_m7_qa, bad_m8_qa, bad_m10_qa

    ######################### QF4 #########################
    def qa_qf4_viirs_09(self, hdf_obj):
        fdata = hdf_obj.select('QF4 Surface Reflectance')[:]
        bits = unpack_uint_to_bits(fdata, 8, bitorder='little')

        bad_m11_qa = bits[0]
        bad_i1_qa = bits[1]
        bad_i2_qa = bits[2]
        bad_i3_qa = bits[3]
        aot_qa = bits[4]
        missing_aot_input_qa = bits[5]
        invalid_land_qa = bits[6]
        missing_pw_qa = bits[7]
        return bad_m11_qa, bad_i1_qa, bad_i2_qa, bad_i3_qa, aot_qa, missing_aot_input_qa, invalid_land_qa, missing_pw_qa

    ######################### QF5 #########################
    def qa_qf5_viirs_09(self, hdf_obj):
        fdata = hdf_obj.select('QF5 Surface Reflectance')[:]
        bits = unpack_uint_to_bits(fdata, 8, bitorder='little')

        missing_ozone_qa = bits[0]
        missing_sfc_press_qa = bits[1]
        overall_m1_qa = bits[2]
        overall_m2_qa = bits[3]
        overall_m3_qa = bits[4]
        overall_m4_qa = bits[5]
        overall_m5_qa = bits[6]
        overall_m7_qa = bits[7]
        return missing_ozone_qa, missing_sfc_press_qa, overall_m1_qa, overall_m2_qa, overall_m3_qa, overall_m4_qa, overall_m5_qa, overall_m7_qa

    ######################### QF6 #########################
    def qa_qf6_viirs_09(self, hdf_obj):
        fdata = hdf_obj.select('QF6 Surface Reflectance')[:]
        bits = unpack_uint_to_bits(fdata, 8, bitorder='little')

        overall_m8_qa = bits[0]
        overall_m10_qa = bits[1]
        overall_m11_qa = bits[2]
        overall_i1_qa = bits[3]
        overall_i2_qa = bits[4]
        overall_i3_qa = bits[5]
        return overall_m8_qa, overall_m10_qa, overall_m11_qa, overall_i1_qa, overall_i2_qa, overall_i3_qa


    def extract_surface_reflectance(self, hdf_obj):
        """ Extract surface reflectance data """

        # check that if bands are provided that they are valid
        if (self.bands is not None) and (not set(self.bands).issubset(self.available_product_bands)):
            raise AttributeError('Error [viirs_09]: Your input for `bands`={}\n`bands` must be one of {}\n'.format(self.bands, self.available_product_bands))

        # resolution and band settings
        if self.resolution == '375m':
            if self.bands is None:
                self.bands = ['I1', 'I2', 'I3']

            else: # remove leading 0 in string to match product band number convention
                self.bands = [i[0] + i[1:].lstrip('0') for i in self.bands]

        elif self.resolution == '750m':
            if self.bands is None:
                self.bands = ['M1', 'M2', 'M3', 'M4', 'M5', 'M7', 'M8', 'M10', 'M11']

            else: # remove leading 0 in string to match product band number convention
                self.bands = [i[0] + i[1:].lstrip('0') for i in self.bands]

        else:
            raise AttributeError('Error [viirs_09]: `resolution` must be one of `375m` or `750m`')

        # begin extracting data
        # search datasets containing the search term derived from param and resolution
        search_term = self.resolution + ' ' +  'Surface Reflectance'
        search_terms_with_bands = [search_term + ' ' + 'Band {}'.format(str(band)) for band in self.bands]
        params = [i for i in list(hdf_obj.datasets().keys()) if i in search_terms_with_bands] # list of dataset names

        if len(search_terms_with_bands) != len(params):
            print('Warning [viirs_09]: Not all bands were extracted. Check self.bands and self.resolution inputs')

        # use the first param to get shape
        data_shape = tuple(hdf_obj.select(params[0]).dimensions().values())
        surface_reflectance = np.zeros((len(params), data_shape[0], data_shape[1]), dtype=np.float64)
        wvl = np.zeros(len(self.bands), dtype='uint16') # wavelengths

        # loop through bands, scale and offset each param and store in tau
        for idx, band_num in enumerate(self.bands):
            if len(band_num) != 3:
                band_key = band_num[0] + '0{}'.format(band_num[1]) # pad 0 for dictionary indexing
            else:
                band_key = band_num

            surface_reflectance[idx] = get_data_h4(hdf_obj.select(params[idx]))
            wvl[idx] = VIIRS_ALL_BANDS[band_key]


        # save the data
        self.data = {}
        self.data['wvl'] = dict(name='Wavelength'              , data=wvl,     units='nm')
        self.data['surface_reflectance'] = dict(name='Surface Reflectance', data=surface_reflectance,     units='N/A')


        # save qa data if requested
        if self.quality_assurance is not None:
            # save qa in a separate dict
            self.qa = {}

            if (self.quality_assurance.lower() == 'auto') or (self.quality_assurance.lower() == 'all'):
                overall_m8_qa, overall_m10_qa, overall_m11_qa, overall_i1_qa, overall_i2_qa, overall_i3_qa = self.qa_qf6_viirs_09(hdf_obj)
                _, _, overall_m1_qa, overall_m2_qa, overall_m3_qa, overall_m4_qa, overall_m5_qa, overall_m7_qa = self.qa_qf5_viirs_09(hdf_obj)

                overall_ibands_qa = np.stack([overall_i1_qa, overall_i2_qa, overall_i3_qa], axis=0)
                overall_mbands_qa = np.stack([overall_m1_qa, overall_m2_qa, overall_m3_qa,
                                              overall_m4_qa, overall_m5_qa, overall_m7_qa,
                                              overall_m8_qa, overall_m10_qa, overall_m11_qa], axis=0)

                self.qa['overall_ibands_qa']     = dict(name='Overall Quality of Surf. Refl. of Bands: I1, I2, I3', data=overall_ibands_qa, units='N/A')
                self.qa['overall_mbands_qa']     = dict(name='Overall Quality of Surf. Refl. of Bands: M1, M2, M3, M4, M5, M7, M8, M9, M10, M11', data=overall_mbands_qa, units='N/A')


            if (self.quality_assurance.lower() == 'ancillary') or (self.quality_assurance.lower() == 'all'):
                cld_msk_qa, cld_msk_confidence_qa, day_night_qa, low_sun_qa, sunglint_qa = self.qa_qf1_viirs_09(hdf_obj)
                land_water_qa, cld_shadow_qa, heavy_aerosol_qa, snow_ice_qa, thin_cirrus_solar_qa, thin_cirrus_thermal_qa = self.qa_qf2_viirs_09(hdf_obj)
                bad_m1_qa, bad_m2_qa, bad_m3_qa, bad_m4_qa, bad_m5_qa, bad_m7_qa, bad_m8_qa, bad_m10_qa = self.qa_qf3_viirs_09(hdf_obj)
                bad_m11_qa, bad_i1_qa, bad_i2_qa, bad_i3_qa, aot_qa, missing_aot_input_qa, invalid_land_qa, missing_pw_qa = self.qa_qf4_viirs_09(hdf_obj)
                missing_ozone_qa, missing_sfc_press_qa, _, _, _, _, _, _ = self.qa_qf5_viirs_09(hdf_obj)

                bad_ibands_qa = np.stack([bad_i1_qa, bad_i2_qa, bad_i3_qa], axis=0)
                bad_mbands_qa = np.stack([bad_m1_qa, bad_m2_qa, bad_m3_qa,
                                          bad_m4_qa, bad_m5_qa, bad_m7_qa,
                                          bad_m8_qa, bad_m10_qa, bad_m11_qa], axis=0)

                missing_data_qa = np.stack([missing_aot_input_qa, invalid_land_qa, missing_pw_qa, missing_ozone_qa, missing_sfc_press_qa])
                ancillary_data_qa = np.stack([cld_msk_qa, cld_msk_confidence_qa, day_night_qa, low_sun_qa, sunglint_qa,
                                         land_water_qa, cld_shadow_qa, heavy_aerosol_qa, snow_ice_qa, thin_cirrus_solar_qa,
                                         thin_cirrus_thermal_qa, aot_qa], axis=0)

                missing_data_desc = 'index 0 = missing AOT input data\n'\
                                    'index 1 = invalid land AM input over land or ocean\n'\
                                    'index 2 = missing PW input data\n'\
                                    'index 3 = missing ozone input data\n'\
                                    'index 4 = missing surface pressure data\n'

                ancillary_data_desc = 'index 0 = cloud mask (0-3)\n'\
                                      'index 1 = cloud mask confidence (0-3)\n'\
                                      'index 2 = day/night flag (0=day)\n'\
                                      'index 3 = low sun mask (0=high)\n'\
                                      'index 4 = sunglint flag (0-3; 0=no sunglint)\n'\
                                      'index 5 = land/water background (0, 1, 2, 3, 5)\n'\
                                      'index 6 = cloud shadow mask (1=shadow)\n'\
                                      'index 7 = heavy aerosol mask (1=heavy aerosol)\n'\
                                      'index 8 = snow/ice present (1=yes)\n'\
                                      'index 9 = thin cirrus detected by solar/reflective bands (1=yes)\n'\
                                      'index 10 = thin cirrus detected by thermal/emissive bands (1=yes)\n'\
                                      'index 11 = overall quality of aerosol optical thickness (0=good)\n'\


                self.qa['missing_data_qa']       = dict(name='Missing or Invalid Input Data QA', description=missing_data_desc, data=missing_data_qa, units='N/A')
                self.qa['ancillary_data_qa']     = dict(name='Ancillary Data QA', description=ancillary_data_desc, data=ancillary_data_qa, units='N/A')
                self.qa['bad_ibands_qa']         = dict(name='Bad L1b/SDR data of Bands: I1, I2, I3', data=bad_ibands_qa, units='N/A')
                self.qa['bad_mbands_qa']         = dict(name='Bad L1b/SDR data of Bands: M1, M2, M3, M4, M5, M7, M8, M9, M10, M11', data=bad_mbands_qa, units='N/A')


    def read(self, fname):

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [viirs_09]: To use \'viirs_09\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f = SD(fname, SDC.READ)
        self.extract_surface_reflectance(f)
        f.end()
        #------------------------------------------------------------------------------------------------------------------------------#


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
