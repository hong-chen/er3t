import os
import sys
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import shutil
import urllib.request
import requests
from er3t.util import check_equal, get_doy_tag, get_data_h4, get_data_nc, unpack_uint_to_bits



__all__ = [
        'modis_l1b', \
        'modis_l2', \
        'modis_35_l2', \
        'modis_03', \
        'modis_04', \
        'modis_09', \
        'modis_09a1', \
        'modis_43a1', \
        'modis_43a3', \
        'modis_tiff', \
        'upscale_modis_lonlat', \
        'download_modis_rgb', \
        'download_modis_https', \
        'cal_sinusoidal_grid', \
        'get_sinusoidal_grid_tag', \
        ]


MODIS_L1B_QKM_BANDS = {
                        1: 650,
                        2: 860,
                      }


MODIS_L1B_HKM_1KM_BANDS = {
                        1: 650,
                        2: 860,
                        3: 470,
                        4: 555,
                        5: 1240,
                        6: 1640,
                        7: 2130,
                        26: 1380,
                        8: 412,
                        9: 443,
                        10: 488,
                        11: 531,
                        12: 551,
                        13: 667,
                        14: 678,
                        15: 748,
                        16: 869,
                        17: 905,
                        18: 936,
                        19: 940,
                        20: 3750,
                        21: 3964,
                        22: 3964,
                        23: 4050,
                        24: 4465,
                        25: 4515,
                        27: 6715,
                        28: 7235,
                        29: 8550,
                        30: 9730,
                        31: 11030,
                        32: 12020,
                        33: 13335,
                        34: 13635,
                        35: 13935,
                        36: 14235
                      }

MODIS_L1B_HKM_1KM_BANDS_DEFAULT = {1: 650,
                                   4: 555,
                                   3: 470
                                    }

# reader for MODIS (Moderate Resolution Imaging Spectroradiometer)
#╭────────────────────────────────────────────────────────────────────────────╮#

class modis_03:

    """
    Read MODIS 03 geolocation data

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag
        keep_dims=: keyword argument, default=False, set to True to get full granule, False to apply geomask of extent

    Output:
        self.data
                ['lon']
                ['lat']
                ['sza']
                ['saa']
                ['vza']
                ['vaa']
    """


    ID = 'MODIS 03 Geolocation Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 verbose   = False, \
                 keep_dims = False):

        self.fnames     = fnames      # file name of the pickle file
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

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_03]: To use \'modis_03\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        sza0       = f.select('SolarZenith')
        saa0       = f.select('SolarAzimuth')
        vza0       = f.select('SensorZenith')
        vaa0       = f.select('SensorAzimuth')


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        lon = lon0[:]
        lat = lat0[:]

        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0], self.extent[1]]
            lat_range = [self.extent[2], self.extent[3]]

        logic     = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
        if not self.keep_dims:
            lon       = lon[logic]
            lat       = lat[logic]
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # Calculate 1. sza, 2. saa, 3. vza, 4. vaa
        #╭────────────────────────────────────────────────────────────────────────────╮#
        sza = get_data_h4(sza0)
        saa = get_data_h4(saa0)
        vza = get_data_h4(vza0)
        vaa = get_data_h4(vaa0)

        if not self.keep_dims:
            sza = sza[logic]
            saa = saa[logic]
            vza = vza[logic]
            vaa = vaa[logic]

        f.end()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km': logic}

            self.data['lon']   = dict(name='Longitude'                 , data=np.hstack((self.data['lon']['data'], lon    )), units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=np.hstack((self.data['lat']['data'], lat    )), units='degrees')
            self.data['sza']   = dict(name='Solar Zenith Angle'        , data=np.hstack((self.data['sza']['data'], sza    )), units='degrees')
            self.data['saa']   = dict(name='Solar Azimuth Angle'       , data=np.hstack((self.data['saa']['data'], saa    )), units='degrees')
            self.data['vza']   = dict(name='Sensor Zenith Angle'       , data=np.hstack((self.data['vza']['data'], vza    )), units='degrees')
            self.data['vaa']   = dict(name='Sensor Azimuth Angle'      , data=np.hstack((self.data['vaa']['data'], vaa    )), units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km': logic}

            self.data  = {}
            self.data['lon']   = dict(name='Longitude'                 , data=lon    , units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=lat    , units='degrees')
            self.data['sza']   = dict(name='Solar Zenith Angle'        , data=sza    , units='degrees')
            self.data['saa']   = dict(name='Solar Azimuth Angle'       , data=saa    , units='degrees')
            self.data['vza']   = dict(name='Sensor Zenith Angle'       , data=vza    , units='degrees')
            self.data['vaa']   = dict(name='Sensor Azimuth Angle'      , data=vaa    , units='degrees')


    def read_vars(self, fname, vnames=[]):

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_03]: To use \'modis_03\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        logic = self.logic[fname]['1km']

        f     = SD(fname, SDC.READ)

        for vname in vnames:

            data0 = f.select(vname)
            data  = get_data_h4(data0)[logic]
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname, data=np.hstack((self.data[vname.lower()]['data'], data)), units=data0.attributes()['units'])
            else:
                self.data[vname.lower()] = dict(name=vname, data=data, units=data0.attributes()['units'])

        f.end()



class modis_l1b:

    """
    Read MODIS Level 1B file into an object `modis_l1b`

    Input:
        fnames=   : list, default=None, Python list of the file path of the original HDF4 file
        extent=   : list, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        f03=      : class obj, default=None, class object created by `modis_03`
        bands=    : list, default=None, list of band numbers to get.
        verbose=  : bool, default=False, verbose tag
        keep_dims=: bool, default=False, set to True to get full granule, False to apply geomask of extent

    Output:
        self.data
                ['lon']
                ['lat']
                ['wvl']
                ['rad']
                ['ref']
                ['cnt']
                ['uct']
    """


    ID = 'MODIS Level 1b Calibrated Radiance'


    def __init__(self, \
                 fnames    = None, \
                 f03       = None, \
                 extent    = None, \
                 bands     = None, \
                 verbose   = False,\
                 keep_dims = False):

        self.fnames     = fnames      # Python list of the file path of the original HDF4 files
        self.f03        = f03         # geolocation class object created using the `modis_03` reader
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.bands      = bands       # Python list of bands that need to be extracted
        self.verbose    = verbose     # verbose tag
        self.keep_dims  = keep_dims   # flag; if false -> apply geomask to convert to 1D; true -> retain 2D/3D data


        filename = os.path.basename(fnames[0]).lower()
        if 'qkm' in filename:
            self.resolution = 0.25
            if bands is None:
                self.bands = list(MODIS_L1B_QKM_BANDS.keys())

            elif (bands is not None) and not (set(bands).issubset(set(MODIS_L1B_QKM_BANDS.keys()))):

                msg = 'Error [modis_l1b]: Bands must be one or more of %s' % list(MODIS_L1B_QKM_BANDS.keys())
                raise KeyError(msg)

        elif 'hkm' in filename:
            self.resolution = 0.5
            if bands is None:
                self.bands = list(MODIS_L1B_HKM_1KM_BANDS_DEFAULT.keys())

            elif (bands is not None) and not (set(bands).issubset(set(MODIS_L1B_HKM_1KM_BANDS.keys()))):
                msg = 'Error [modis_l1b]: Bands must be one or more of %s' % list(MODIS_L1B_HKM_1KM_BANDS.keys())
                raise KeyError(msg)

        elif '1km' in filename:
            self.resolution = 1.0
            if bands is None:
                self.bands = list(MODIS_L1B_HKM_1KM_BANDS_DEFAULT.keys())

            elif (bands is not None) and not (set(bands).issubset(set(MODIS_L1B_HKM_1KM_BANDS.keys()))):
                msg = 'Error [modis_l1b]: Bands must be one or more of %s' % list(MODIS_L1B_HKM_1KM_BANDS.keys())
                raise KeyError(msg)

        else:
            sys.exit('Error [modis_l1b]: Currently, only QKM (0.25km), HKM (0.5km), and 1KM products are supported.')

        for fname in self.fnames:
            self.read(fname)


    def _get_250_500_attrs(self, hdf_dset_250, hdf_dset_500):
        rad_off = hdf_dset_250.attributes()['radiance_offsets']         + hdf_dset_500.attributes()['radiance_offsets']
        rad_sca = hdf_dset_250.attributes()['radiance_scales']          + hdf_dset_500.attributes()['radiance_scales']
        ref_off = hdf_dset_250.attributes()['reflectance_offsets']      + hdf_dset_500.attributes()['reflectance_offsets']
        ref_sca = hdf_dset_250.attributes()['reflectance_scales']       + hdf_dset_500.attributes()['reflectance_scales']
        cnt_off = hdf_dset_250.attributes()['corrected_counts_offsets'] + hdf_dset_500.attributes()['corrected_counts_offsets']
        cnt_sca = hdf_dset_250.attributes()['corrected_counts_scales']  + hdf_dset_500.attributes()['corrected_counts_scales']
        return rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca


    def _get_250_500_uct(self, hdf_uct_250, hdf_uct_500):
        uct_spc = hdf_uct_250.attributes()['specified_uncertainty'] + hdf_uct_500.attributes()['specified_uncertainty']
        uct_sca = hdf_uct_250.attributes()['scaling_factor']        + hdf_uct_500.attributes()['scaling_factor']
        return uct_spc, uct_sca

    def _get_250_500_1km_attrs(self, hdf_dset_250, hdf_dset_500, hdf_dset_1km_solar, hdf_dset_1km_emissive):
        num_emissive_bands = len(hdf_dset_1km_emissive.attributes()['radiance_scales'])

        rad_off = hdf_dset_250.attributes()['radiance_offsets']    + hdf_dset_500.attributes()['radiance_offsets']    + hdf_dset_1km_solar.attributes()['radiance_offsets'] + hdf_dset_1km_emissive.attributes()['radiance_offsets']
        rad_sca = hdf_dset_250.attributes()['radiance_scales']     + hdf_dset_500.attributes()['radiance_scales']     + hdf_dset_1km_solar.attributes()['radiance_scales'] + hdf_dset_1km_emissive.attributes()['radiance_scales']
        ref_off = hdf_dset_250.attributes()['reflectance_offsets'] + hdf_dset_500.attributes()['reflectance_offsets'] + hdf_dset_1km_solar.attributes()['reflectance_offsets'] + list(np.full(-99, num_emissive_bands))
        ref_sca = hdf_dset_250.attributes()['reflectance_scales']  + hdf_dset_500.attributes()['reflectance_scales']  + hdf_dset_1km_solar.attributes()['reflectance_scales'] + list(np.ones(num_emissive_bands))
        cnt_off = hdf_dset_250.attributes()['corrected_counts_offsets'] + hdf_dset_500.attributes()['corrected_counts_offsets'] + hdf_dset_1km_solar.attributes()['corrected_counts_offsets'] + list(np.full(num_emissive_bands, -99))
        cnt_sca = hdf_dset_250.attributes()['corrected_counts_scales'] + hdf_dset_500.attributes()['corrected_counts_scales'] + hdf_dset_1km_solar.attributes()['corrected_counts_scales'] + list(np.ones(num_emissive_bands))
        return rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca

    def _get_250_500_1km_uct(self, hdf_uct_250, hdf_uct_500, hdf_uct_1km_solar, hdf_uct_1km_emissive):
        uct_spc = hdf_uct_250.attributes()['specified_uncertainty'] + hdf_uct_500.attributes()['specified_uncertainty'] + hdf_uct_1km_solar.attributes()['specified_uncertainty'] + hdf_uct_1km_emissive.attributes()['specified_uncertainty']
        uct_sca = hdf_uct_250.attributes()['scaling_factor']        + hdf_uct_500.attributes()['scaling_factor']        + hdf_uct_1km_solar.attributes()['scaling_factor'] + hdf_uct_1km_emissive.attributes()['scaling_factor']
        return uct_spc, uct_sca


    def read(self, fname):

        """
        Read radiance/reflectance/corrected counts along with their uncertainties from the MODIS L1B data
        Output:
            self.data
                ['lon']
                ['lat']
                ['wvl']
                ['rad']
                ['ref']
                ['cnt']
                ['uct']
    """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_l1b]: To use \'modis_l1b\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # when resolution equals to 250 m
        if check_equal(self.resolution, 0.25):
            if self.f03 is not None:
                lon0  = self.f03.data['lon']['data']
                lat0  = self.f03.data['lat']['data']
            else:
                lat0  = f.select('Latitude')
                lon0  = f.select('Longitude')

            # band info
            band_numbers = list(f.select('Band_250M')[:])
            band_dict = dict(zip(band_numbers, np.arange(0, len(band_numbers))))

            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=4, extra_grid=False)
            raw0      = f.select('EV_250_RefSB')
            uct0      = f.select('EV_250_RefSB_Uncert_Indexes')

            if self.extent is None:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

            else:
                lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
                lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

            logic     = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            if not self.keep_dims:
                lon       = lon[logic]
                lat       = lat[logic]

            # save offsets and scaling factors
            rad_off = raw0.attributes()['radiance_offsets']
            rad_sca = raw0.attributes()['radiance_scales']

            ref_off = raw0.attributes()['reflectance_offsets']
            ref_sca = raw0.attributes()['reflectance_scales']

            cnt_off = raw0.attributes()['corrected_counts_offsets']
            cnt_sca = raw0.attributes()['corrected_counts_scales']

            uct_spc = uct0.attributes()['specified_uncertainty']
            uct_sca = uct0.attributes()['scaling_factor']
            do_region = True

        # when resolution equals to 500 m
        elif check_equal(self.resolution, 0.5):
            if self.f03 is not None:
                lon0  = self.f03.data['lon']['data']
                lat0  = self.f03.data['lat']['data']
            else:
                lat0  = f.select('Latitude')
                lon0  = f.select('Longitude')


            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=2, extra_grid=False)
            raw0_250  = f.select('EV_250_Aggr500_RefSB')
            uct0_250  = f.select('EV_250_Aggr500_RefSB_Uncert_Indexes')
            raw0_500  = f.select('EV_500_RefSB')
            uct0_500  = f.select('EV_500_RefSB_Uncert_Indexes')

            # save offsets and scaling factors (from both QKM and HKM bands)
            rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca = self._get_250_500_attrs(raw0_250, raw0_500)
            uct_spc, uct_sca                                     = self._get_250_500_uct(uct0_250, uct0_500)

            # combine QKM and HKM bands
            raw0      = np.vstack([raw0_250, raw0_500])
            uct0      = np.vstack([uct0_250, uct0_500])

            # band info
            band_numbers = list(f.select('Band_250M')[:]) + list(f.select('Band_500M')[:])
            band_dict = dict(zip(band_numbers, np.arange(0, len(band_numbers))))

            do_region = True

            if self.extent is None:
                lon_range = [-180.0, 180.0]
                lat_range = [-90.0 , 90.0]

            else:
                lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
                lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

            logic     = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            if not self.keep_dims: # converts to 1D
                lon       = lon[logic]
                lat       = lat[logic]

        # when resolution equals to 1000 m
        elif check_equal(self.resolution, 1.0):
            if self.f03 is not None:
                raw0_250           = f.select('EV_250_Aggr1km_RefSB')
                uct0_250           = f.select('EV_250_Aggr1km_RefSB_Uncert_Indexes')
                raw0_500           = f.select('EV_500_Aggr1km_RefSB')
                uct0_500           = f.select('EV_500_Aggr1km_RefSB_Uncert_Indexes')
                raw0_1km_solar     = f.select('EV_1KM_RefSB')
                uct0_1km_solar     = f.select('EV_1KM_RefSB_Uncert_Indexes')
                raw0_1km_emissive  = f.select('EV_1KM_Emissive')
                uct0_1km_emissive  = f.select('EV_1KM_Emissive_Uncert_Indexes')

                # save offsets and scaling factors (from both QKM and HKM and 1KM solar and emissive bands)
                rad_off, rad_sca, ref_off, ref_sca, cnt_off, cnt_sca = self._get_250_500_1km_attrs(raw0_250, raw0_500, raw0_1km_solar, raw0_1km_emissive)
                uct_spc, uct_sca                                     = self._get_250_500_1km_uct(uct0_250, uct0_500, uct0_1km_solar, uct0_1km_emissive)

                # combine QKM and HKM and 1KM bands
                raw0      = np.vstack([raw0_250, raw0_500, raw0_1km_solar, raw0_1km_emissive])
                uct0      = np.vstack([uct0_250, uct0_500, uct0_1km_solar, uct0_1km_emissive])

                # band info
                band_numbers = list(f.select('Band_250M')[:]) + list(f.select('Band_500M')[:]) + list(f.select('Band_1KM_RefSB')[:]) + list(f.select('Band_1KM_Emissive')[:])
                band_dict = dict(zip(band_numbers, np.arange(0, len(band_numbers))))

                do_region = False
                lon       = self.f03.data['lon']['data']
                lat       = self.f03.data['lat']['data']
                logic     = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']
            else:
                sys.exit('Error   [modis_l1b]: 1KM product reader has not been implemented without geolocation file being specified.')

        else:
            sys.exit('Error   [modis_l1b]: \'resolution=%f\' has not been implemented.' % self.resolution)


        # Calculate 1. radiance, 2. reflectance, 3. corrected counts from the raw data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        wvl = np.zeros(len(self.bands), dtype='uint16')

        if self.keep_dims: # don't apply the logic geomask
            raw = raw0[:]
            rad = np.zeros((len(self.bands), raw.shape[1], raw.shape[2]), dtype=np.float32)
            ref = np.zeros((len(self.bands), raw.shape[1], raw.shape[2]), dtype=np.float32)
            cnt = np.zeros((len(self.bands), raw.shape[1], raw.shape[2]), dtype=np.float32)
            # Calculate uncertainty
            uct     = uct0[:]
            uct_pct = np.zeros((len(self.bands), raw.shape[1], raw.shape[2]), dtype=np.float32)

        else:
            raw = raw0[:][:, logic]
            rad = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float32)
            ref = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float32)
            cnt = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float32)
            # Calculate uncertainty
            uct     = uct0[:][:, logic]
            uct_pct = np.zeros((len(self.bands), raw.shape[1]), dtype=np.float32)

        # apply offsets and scales
        for band_counter, i in enumerate(self.bands):
            band_idx                = band_dict[i]
            rad0                    = (raw[band_idx] - rad_off[band_idx]) * rad_sca[band_idx]
            rad[band_counter]       = rad0/1000.0 # convert to W/m^2/nm/sr
            ref[band_counter]       = (raw[band_idx] - ref_off[band_idx]) * ref_sca[band_idx]
            cnt[band_counter]       = (raw[band_idx] - cnt_off[band_idx]) * cnt_sca[band_idx]
            uct_pct[band_counter]   = uct_spc[band_idx] * np.exp(uct[band_idx] / uct_sca[band_idx]) # convert to percentage
            wvl[band_counter]       = MODIS_L1B_HKM_1KM_BANDS[i]

        f.end()
        #╰────────────────────────────────────────────────────────────────────────────╯#



        if hasattr(self, 'data'):
            if do_region:
                self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)),     units='degrees')
                self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)),     units='degrees')

            self.data['rad'] = dict(name='Radiance'                , data=np.hstack((self.data['rad']['data'], rad)),     units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance (x cos(SZA))', data=np.hstack((self.data['ref']['data'], ref)),     units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts'        , data=np.hstack((self.data['cnt']['data'], cnt)),     units='N/A')
            self.data['uct'] = dict(name='Uncertainty Percentage'  , data=np.hstack((self.data['uct']['data'], uct_pct)), units='N/A')

        else:

            self.data = {}
            self.data['lon'] = dict(name='Longitude'               , data=lon,     units='degrees')
            self.data['lat'] = dict(name='Latitude'                , data=lat,     units='degrees')
            self.data['wvl'] = dict(name='Wavelength'              , data=wvl,     units='nm')
            self.data['rad'] = dict(name='Radiance'                , data=rad,     units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance (x cos(SZA))', data=ref,     units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts'        , data=cnt,     units='N/A')
            self.data['uct'] = dict(name='Uncertainty Percentage'  , data=uct_pct, units='N/A')


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()



class modis_l2:

    """
    Read MODIS level 2 cloud product

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        f03=      : keyword argument, default=None, Python list of the corresponding geolocation files to fnames
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['cot']
                ['cer']
                ['cwp']
    """


    ID = 'MODIS Level 2 Cloud Product'


    def __init__(self,              \
                 fnames,            \
                 f03       = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 cop_flag  = '',    \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.f03        = f03         # geolocation class object created using the `modis_03` reader
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname, cop_flag=cop_flag)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)


    def read(self, fname, cop_flag):

        """
        Read cloud optical properties

        self.data
            ['lon']
            ['lat']
            ['cot']
            ['cer']
            ['cwp']
            ['pcl']
            ['lon_5km']
            ['lat_5km']

        self.logic
        self.logic_5km
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_l2]: To use \'modis_l2\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        if len(cop_flag) == 0:
            vname_ctp     = 'Cloud_Phase_Optical_Properties'
            vname_cot     = 'Cloud_Optical_Thickness'
            vname_cer     = 'Cloud_Effective_Radius'
            vname_cwp     = 'Cloud_Water_Path'
            vname_cot_err = 'Cloud_Optical_Thickness_Uncertainty'
            vname_cer_err = 'Cloud_Effective_Radius_Uncertainty'
            vname_cwp_err = 'Cloud_Water_Path_Uncertainty'
        else:
            vname_ctp     = 'Cloud_Phase_Optical_Properties'
            vname_cot     = 'Cloud_Optical_Thickness_%s' % cop_flag
            vname_cer     = 'Cloud_Effective_Radius_%s'  % cop_flag
            vname_cwp     = 'Cloud_Water_Path_%s'  % cop_flag
            vname_cot_err = 'Cloud_Optical_Thickness_Uncertainty_%s' % cop_flag
            vname_cer_err = 'Cloud_Effective_Radius_Uncertainty_%s' % cop_flag
            vname_cwp_err = 'Cloud_Water_Path_Uncertainty_%s'  % cop_flag

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        ctp        = f.select(vname_ctp)

        cot0       = f.select(vname_cot)
        cer0       = f.select(vname_cer)
        cwp0       = f.select(vname_cwp)
        cot1       = f.select('%s_PCL' % vname_cot)
        cer1       = f.select('%s_PCL' % vname_cer)
        cwp1       = f.select('%s_PCL' % vname_cwp)
        cot_err0   = f.select(vname_cot_err)
        cer_err0   = f.select(vname_cer_err)
        cwp_err0   = f.select(vname_cwp_err)


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #╭────────────────────────────────────────────────────────────────────────────╮#
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

            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        if self.f03 is None:
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)
            logic_1km = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon       = lon[logic_1km]
            lat       = lat[logic_1km]
        else:
            lon       = self.f03.data['lon']['data']
            lat       = self.f03.data['lat']['data']
            logic_1km = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']


        lon_5km   = lon0[:]
        lat_5km   = lat0[:]
        logic_5km = (lon_5km >= lon_range[0]) & (lon_5km <= lon_range[1]) & (lat_5km >= lat_range[0]) & (lat_5km <= lat_range[1])
        lon_5km   = lon_5km[logic_5km]
        lat_5km   = lat_5km[logic_5km]
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Calculate 1. cot, 2. cer, 3. ctp
        #╭────────────────────────────────────────────────────────────────────────────╮#
        ctp           = get_data_h4(ctp)[logic_1km]

        cot0_data     = get_data_h4(cot0)[logic_1km]
        cer0_data     = get_data_h4(cer0)[logic_1km]
        cwp0_data     = get_data_h4(cwp0)[logic_1km]

        cot1_data     = get_data_h4(cot1)[logic_1km]
        cer1_data     = get_data_h4(cer1)[logic_1km]
        cwp1_data     = get_data_h4(cwp1)[logic_1km]

        cot_err0_data = get_data_h4(cot_err0)[logic_1km]
        cer_err0_data = get_data_h4(cer_err0)[logic_1km]
        cwp_err0_data = get_data_h4(cwp_err0)[logic_1km]

        # Make copies to modify
        cot     = cot0_data.copy()
        cer     = cer0_data.copy()
        cwp     = cer0_data.copy()
        cot_err = cot_err0_data.copy()
        cer_err = cer_err0_data.copy()
        cwp_err = cwp_err0_data.copy()

        pcl = np.zeros_like(cot, dtype=np.uint8)

        # Mark negative (invalid) retrievals with clear-sky values
        logic_invalid = (cot0_data < 0.0) | (cer0_data < 0.0) | (cwp0_data < 0.0) | (ctp == 0)
        cot[logic_invalid]     = 0.0
        cer[logic_invalid]     = 0.0
        cwp[logic_invalid]     = 0.0
        cot_err[logic_invalid] = 0.0
        cer_err[logic_invalid] = 0.0
        cwp_err[logic_invalid] = 0.0

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

        f.end()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # pcl = pcl[logic_1km]

        if hasattr(self, 'data'):

            self.logic[fname]      = {'1km':logic_1km, '5km':logic_5km}

            self.data['lon']       = dict(name='Longitude',                           data=np.hstack((self.data['lon']['data'], lon)),                   units='degrees')
            self.data['lat']       = dict(name='Latitude',                            data=np.hstack((self.data['lat']['data'], lat)),                   units='degrees')
            self.data['ctp']       = dict(name='Cloud thermodynamic phase',           data=np.hstack((self.data['ctp']['data'], ctp)),                   units='N/A')
            self.data['cot']       = dict(name='Cloud optical thickness',             data=np.hstack((self.data['cot']['data'], cot)),                   units='N/A')
            self.data['cer']       = dict(name='Cloud effective radius',              data=np.hstack((self.data['cer']['data'], cer)),                   units='micron')
            self.data['cot_err']   = dict(name='Cloud optical thickness uncertainty', data=np.hstack((self.data['cot_err']['data'], cot*cot_err/100.0)), units='N/A')
            self.data['cer_err']   = dict(name='Cloud effective radius uncertainty',  data=np.hstack((self.data['cer_err']['data'], cer*cer_err/100.0)), units='micron')
            self.data['pcl']       = dict(name='PCL tag (1:PCL)',                     data=np.hstack((self.data['pcl']['data'], pcl)),                   units='N/A')
            self.data['lon_5km']   = dict(name='Longitude at 5km',                    data=np.hstack((self.data['lon_5km']['data'], lon_5km)),           units='degrees')
            self.data['lat_5km']   = dict(name='Latitude at 5km',                     data=np.hstack((self.data['lat_5km']['data'], lat_5km)),           units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data  = {}
            self.data['lon']       = dict(name='Longitude',                           data=lon,               units='degrees')
            self.data['lat']       = dict(name='Latitude',                            data=lat,               units='degrees')
            self.data['ctp']       = dict(name='Cloud thermodynamic phase',           data=ctp,               units='N/A')
            self.data['cot']       = dict(name='Cloud optical thickness',             data=cot,               units='N/A')
            self.data['cer']       = dict(name='Cloud effective radius',              data=cer,               units='micron')
            self.data['cot_err']   = dict(name='Cloud optical thickness uncertainty', data=cot*cot_err/100.0, units='N/A')
            self.data['cer_err']   = dict(name='Cloud effective radius uncertainty',  data=cer*cer_err/100.0, units='micron')
            self.data['pcl']       = dict(name='PCL tag (1:PCL)',                     data=pcl,               units='N/A')
            self.data['lon_5km']   = dict(name='Longitude at 5km',                    data=lon_5km,           units='degrees')
            self.data['lat_5km']   = dict(name='Latitude at 5km',                     data=lat_5km,           units='degrees')


    def read_vars(self, fname, vnames=[]):

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_l2]: To use \'modis_l2\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        dim_1km = f.select('Cloud_Optical_Thickness').info()[2]
        dim_5km = f.select('Cloud_Top_Height').info()[2]

        for vname in vnames:

            data0 = f.select(vname)
            dim0  = data0.info()[2]
            if dim0 == dim_1km:
                logic = self.logic[fname]['1km']
            elif dim0 == dim_5km:
                logic = self.logic[fname]['5km']
            else:
                msg = 'Error [modis_l2]: Unknow resolution for <%s>.' % vname
                raise ValueError(msg)
            data  = get_data_h4(data0)[logic]
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname, data=np.hstack((self.data[vname.lower()]['data'], data)), units=data0.attributes()['units'])
            else:
                self.data[vname.lower()] = dict(name=vname, data=data, units=data0.attributes()['units'])

        f.end()



class modis_35_l2:

    """
    Read MODIS level 2 cloud mask product

    Note: We currently only support processing of the cloud mask bytes at a 1 km resolution only.

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        f03=      : keyword argument, default=None, Python list of the corresponding geolocation files to fnames
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['use_qa']          => 0: not useful (discard), 1: useful
                ['confidence_qa']   => 0: no confidence (do not use), 1: low confidence, 2, ... 7: very high confidence
                ['cloud_mask_flag'] => 0: not determined, 1: determined
                ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
                ['day_night_flag']  => 0: night, 1: day
                ['sunglint_flag']   => 0: in sunglint path, 1: not in sunglint path
                ['snow_ice_flag']   => 0: snow/ice background processing, 1: no snow/ice processing path
                ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
                ['lon_5km']
                ['lat_5km']

    References: (Product Page) https://atmosphere-imager.gsfc.nasa.gov/products/cloud-mask
                (ATBD)         https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/MOD35_ATBD_Collection6_1.pdf
                (User Guide)   http://cimss.ssec.wisc.edu/modis/CMUSERSGUIDE.PDF
    """


    ID = 'MODIS Level 2 Cloud Mask Product'


    def __init__(self,              \
                 fnames,            \
                 f03       = None,  \
                 extent    = None,  \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.f03        = f03         # geolocation file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:
            self.read(fname)


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
        Extract cloud mask QA data to determine confidence
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = np.unpackbits(dbyte, bitorder='big', axis=1)

        # process qa flags
        if byte == 0:
            # Byte 0 only has 4 bits of useful information, other 4 are always 0
            confidence_qa = 4 * data[:, 4] + 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 7 confidence
            useful_qa = data[:, 7] # usefulness QA flag
            return useful_qa, confidence_qa


    def read(self, fname):

        """
        Read cloud mask flags and tests/categories

        self.data
            ['lon']
            ['lat']
            ['use_qa']          => 0: not useful (discard), 1: useful
            ['confidence_qa']   => 0: no confidence (do not use), 1: low confidence, 2, ... 7: very high confidence
            ['cloud_mask_flag'] => 0: not determined, 1: determined
            ['fov_qa_cat']      => 0: cloudy, 1: uncertain, 2: probably clear, 3: confident clear
            ['day_night_flag']  => 0: night, 1: day
            ['sunglint_flag']   => 0: not in sunglint path, 1: in sunglint path
            ['snow_ice_flag']   => 0: no snow/ice in background, 1: possible snow/ice in background
            ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
            ['lon_5km']
            ['lat_5km']

        self.logic_1km
        self.logic_5km
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_35_l2]: To use \'modis_35_l2\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f          = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')
        cld_msk0   = f.select('Cloud_Mask')
        qa0        = f.select('Quality_Assurance')


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #╭────────────────────────────────────────────────────────────────────────────╮#
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

            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Attempt to get lat/lon from geolocation file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if self.f03 is None:
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)
            logic_1km = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon       = lon[logic_1km]
            lat       = lat[logic_1km]
        else:
            lon       = self.f03.data['lon']['data']
            lat       = self.f03.data['lat']['data']
            logic_1km = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']
        #╰────────────────────────────────────────────────────────────────────────────╯#

        lon_5km   = lon0[:]
        lat_5km   = lat0[:]
        logic_5km = (lon_5km>=lon_range[0]) & (lon_5km<=lon_range[1]) & (lat_5km>=lat_range[0]) & (lat_5km<=lat_range[1])
        lon_5km   = lon_5km[logic_5km]
        lat_5km   = lat_5km[logic_5km]

        # Get cloud mask and flag fields
        #╭────────────────────────────────────────────────────────────────────────────╮#
        cm0_data = get_data_h4(cld_msk0)
        qa0_data = get_data_h4(qa0)
        cm = cm0_data.copy()
        qa = qa0_data.copy()

        cm = cm[0, :, :] # read only the first of 6 bytes; rest will be supported in the future if needed
        cm = np.array(cm[logic_1km], dtype='uint8')
        cm = cm.reshape((cm.size, 1))
        cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat = self.extract_data(cm)


        qa = qa[:, :, 0] # read only the first byte for confidence (indexed differently from cloud mask SDS)
        qa = np.array(qa[logic_1km], dtype='uint8')
        qa = qa.reshape((qa.size, 1))
        use_qa, confidence_qa = self.quality_assurance(qa, byte=0)

        f.end()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data['lon']               = dict(name='Longitude',            data=np.hstack((self.data['lon']['data'], lon)),                         units='degrees')
            self.data['lat']               = dict(name='Latitude',             data=np.hstack((self.data['lat']['data'], lat)),                         units='degrees')
            self.data['use_qa']            = dict(name='QA useful',            data=np.hstack((self.data['use_qa']['data'], use_qa)),                   units='N/A')
            self.data['confidence_qa']     = dict(name='QA Mask confidence',   data=np.hstack((self.data['confidence_qa']['data'], confidence_qa)),     units='N/A')
            self.data['cloud_mask_flag']   = dict(name='Cloud mask flag',      data=np.hstack((self.data['cloud_mask_flag']['data'], cloud_mask_flag)), units='N/A')
            self.data['fov_qa_cat']        = dict(name='FOV quality cateogry', data=np.hstack((self.data['fov_qa_cat']['data'], fov_qa_cat)),           units='N/A')
            self.data['day_night_flag']    = dict(name='Day/night flag',       data=np.hstack((self.data['day_night_flag']['data'], day_night_flag)),   units='N/A')
            self.data['sunglint_flag']     = dict(name='Sunglint flag',        data=np.hstack((self.data['sunglint_flag']['data'], sunglint_flag)),     units='N/A')
            self.data['snow_ice_flag']     = dict(name='Snow/ice flag',        data=np.hstack((self.data['snow_flag']['data'], snow_ice_flag)),         units='N/A')
            self.data['land_water_cat']    = dict(name='Land/water flag',      data=np.hstack((self.data['land_water_cat']['data'], land_water_cat)),   units='N/A')
            self.data['lon_5km']           = dict(name='Longitude at 5km',     data=np.hstack((self.data['lon_5km']['data'], lon_5km)),                 units='degrees')
            self.data['lat_5km']           = dict(name='Latitude at 5km',      data=np.hstack((self.data['lat_5km']['data'], lat_5km)),                 units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic_1km, '5km':logic_5km}

            self.data  = {}
            self.data['lon']             = dict(name='Longitude',            data=lon,             units='degrees')
            self.data['lat']             = dict(name='Latitude',             data=lat,             units='degrees')
            self.data['use_qa']          = dict(name='QA useful',            data=use_qa,          units='N/A')
            self.data['confidence_qa']   = dict(name='QA Mask confidence',   data=confidence_qa,   units='N/A')
            self.data['cloud_mask_flag'] = dict(name='Cloud mask flag',      data=cloud_mask_flag, units='N/A')
            self.data['fov_qa_cat']      = dict(name='FOV quality category', data=fov_qa_cat,      units='N/A')
            self.data['day_night_flag']  = dict(name='Day/night flag',       data=day_night_flag,  units='N/A')
            self.data['sunglint_flag']   = dict(name='Sunglint flag',        data=sunglint_flag,   units='N/A')
            self.data['snow_ice_flag']   = dict(name='Snow/ice flag',        data=snow_ice_flag,   units='N/A')
            self.data['land_water_cat']  = dict(name='Land/water category',  data=land_water_cat,  units='N/A')
            self.data['lon_5km']         = dict(name='Longitude at 5km',     data=lon_5km,         units='degrees')
            self.data['lat_5km']         = dict(name='Latitude at 5km',      data=lat_5km,         units='degrees')


class modis_mvcm_cldmsk_l2:
    """
    A class for extracting data from MODIS/Aqua Cloud Mask 5-Min Swath 1 km files (CLDMSK_L2).
    This is the Continuity MODIS-VIIRS Cloud Mask (MVCM) and is produced slightly differently from the MxD35_L2 cloud mask product.
    Consult the references below for appropriate usage.

    Args:
        fname (str): The file name.
        mode (str, optional): The mode under which to operate and extract data, one of 'auto' (gets some cloud mask data that should be sufficient for most users) or 'all' (gets all geophysical data). Defaults to 'auto'.
        quality_assurance (bool, optional): Flag to get QA data. Defaults to False.

    References: (User Guide) https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/Document%20Archive/Science%20Data%20Product%20Documentation/MODIS_VIIRS_Cloud-Mask_UG_04162020.pdf
                (ATBD) https://modis-atmosphere.gsfc.nasa.gov/sites/default/files/ModAtmo/MOD35_ATBD_Collection6_0.pdf
                (Filespec) https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/1/CLDMSK_L2_MODIS_Aqua
                (Paper) Frey et al. (2020), https://doi.org/10.3390/rs12203334

    Note: MODIS/Terra is not yet supported by MVCM
    """
    ID = 'MODIS MVCM Continuity Cloud Mask 5-Min Swath 1 km'


    def __init__(self, \
                 fname,  \
                 mode = 'auto', \
                 quality_assurance = False):


        self.fname             = fname              # file name
        self.mode              = mode.lower()       # mode under which to operate and extract data
        self.quality_assurance = quality_assurance  # flag to get qa data

        self.read(fname)


    def extract_data_byte0(self, dbyte):
        """
        Extract cloud mask (in byte format) flags and categories
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = unpack_uint_to_bits(dbyte.filled(), 8, bitorder='little')
        # extract flags and categories (*_cat) bit by bit
        cloud_mask_flag = data[0]
        fov_qa_cat      = data[1] + 2 * data[2] # convert to a value between 0 and 3
        day_night_flag  = data[3]
        sunglint_flag   = data[4]
        snow_ice_flag   = data[5]
        land_water_cat  = data[6] + 2 * data[7] # convert to a value between 0 and 3

        return cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat


    def extract_other_data_bytes(self, dbyte1, dbyte2, dbyte3):
        """
        Extract cloud mask (in byte format) flags and categories for bytes 2, 3 and 4 (treated as bytes 1, 2 and 3 here)
        Note that bytes 5 and 6 are always padded spare and not used.
        """

        ################## extract byte 1 ##################
        if dbyte1.dtype != 'uint8':
            dbyte1 = dbyte1.astype('uint8')

        data_dbyte1 = unpack_uint_to_bits(dbyte1.filled(), 8, bitorder='little') # convert to binary

        # extract flags bit by bit
        # bit 0 is spare
        thin_cirrus_flag_solar    = data_dbyte1[1] # thin cirrus detected using solar chanels
        snow_cover_ancillary_map  = data_dbyte1[2] # snow cover map from anicllary sources
        thin_cirrus_flag_ir       = data_dbyte1[3] # thin cirrus detected using IR
        cloud_adjacent_flag       = data_dbyte1[4] # cloud adjacency (cloudy, probably cloudy plus 1-pixel adjacent)
        cloud_flag_ir_thresh      = data_dbyte1[5] # cloud flag ocean IR threshold
        # bits 6 and 7 (co2 high cloud tests) are not used for MVCM

        ################## extract byte 2 ##################
        if dbyte2.dtype != 'uint8':
            dbyte2 = dbyte2.astype('uint8')

        data_dbyte2 = unpack_uint_to_bits(dbyte2.filled(), 8, bitorder='little') # convert to binary

        high_cloud_flag_138       = data_dbyte2[0] # 1.38 micron high cloud test
        high_cloud_flag_ir_night  = data_dbyte2[1] # night only IR high cloud test
        cloud_flag_ir_temp_diff   = data_dbyte2[2] # Cloud Flag - IR Temperature Difference Tests
        cloud_flag_ir_night       = data_dbyte2[3] # Cloud Flag – 3.9-11 μm test
        cloud_flag_vnir_ref       = data_dbyte2[4] # Cloud Flag – VNIR Reflectance Test
        cloud_flag_vnir_ref_ratio = data_dbyte2[5] # Cloud Flag – VNIR Reflectance Ratio Test
        clr_sky_ndvi_coastal      = data_dbyte2[6] # clear-sky restoral test – NDVI in coastal areas
        cloud_flag_water_1621     = data_dbyte2[7] # Cloud Flag – Water 1.6 or 2.1 μm Test

        ################## extract byte 3 ##################
        if dbyte3.dtype != 'uint8':
            dbyte3 = dbyte3.astype('uint8')

        data_dbyte3 = unpack_uint_to_bits(dbyte3.filled(), 8, bitorder='little') # convert to binary

        cloud_flag_water_ir                  = data_dbyte3[0] # Cloud Flag – Water 8.6-11 μm
        clr_sky_ocean_spatial                = data_dbyte3[1] # Clear-sky Restoral Test – Spatial Consistency (ocean)
        clr_sky_polar_night_land_sunglint    = data_dbyte3[2] # Clear-sky Restoral Tests (polar night, land, sun glint)
        cloud_flag_sfc_temp_water_night_land = data_dbyte3[3] # Cloud Flag – Surface Temperature Tests (water, night land)
        # bits 4 and 5 are spare
        cloud_flag_night_ocean_ir_variable   = data_dbyte3[6] # Cloud Flag – Night Ocean 11 μm Variability Test
        cloud_flag_night_ocean_low_emissive  = data_dbyte3[7] # Cloud Flag – Night Ocean “Low-Emissivity” 3.9-11 μm Test

        ################## stacking ##################
        # now stack them by type instead of separate fields
        cloud_flag_tests  = np.stack([cloud_flag_ir_temp_diff, cloud_flag_ir_night, cloud_flag_vnir_ref, cloud_flag_vnir_ref_ratio, cloud_flag_water_1621, cloud_flag_water_ir, cloud_flag_sfc_temp_water_night_land, cloud_flag_night_ocean_ir_variable, cloud_flag_night_ocean_low_emissive, cloud_flag_ir_thresh, thin_cirrus_flag_solar, thin_cirrus_flag_ir, cloud_adjacent_flag], axis=0)

        self.cloud_flag_test_description = 'index 0: Cloud Flag - IR Temperature Difference Tests\n'\
                                           'index 1: Cloud Flag - 3.9-11 μm test\n'\
                                           'index 2: Cloud Flag - VNIR Reflectance Test\n'\
                                           'index 3: Cloud Flag - VNIR Reflectance Ratio Test\n'\
                                           'index 4: Cloud Flag - Water 1.6 or 2.1 μm Test\n'\
                                           'index 5: Cloud Flag - Water 8.6-11 μm\n'\
                                           'index 6: Cloud Flag - Surface Temperature Tests (water, night land)\n'\
                                           'index 7: Cloud Flag - Night Ocean 11 μm Variability Test\n'\
                                           'index 8: Cloud Flag - Night Ocean “Low-Emissivity” 3.9-11 μm Test\n'\
                                           'index 9: Cloud Flag - IR Threshold\n'\
                                           'index 10: Cloud Flag - Thin Cirrus (Solar)\n'\
                                           'index 11: Cloud Flag - Thin Cirrus (IR)\n'\
                                           'index 12: Cloud Flag - Adjacency Test (cloudy, probably cloudy plus 1-pixel adjacent)\n'\

        high_cloud_flag_tests = np.stack([high_cloud_flag_138, high_cloud_flag_ir_night], axis=0)
        self.high_cloud_flag_tests_description = 'index 0: 1.38 μm high cloud test\n'\
                                                 'index 1: night only IR high cloud test\n'\

        clr_sky_restoral_tests = np.stack([clr_sky_ndvi_coastal, clr_sky_ocean_spatial, clr_sky_polar_night_land_sunglint], axis=0)
        self.clr_sky_restoral_tests_description = 'index 0: Clear-sky Restoral Test - NDVI in coastal areas\n'\
                                                  'index 1: Clear-sky Restoral Test - Spatial Consistency (ocean)\n'\
                                                  'index 2: Clear-sky Restoral Tests (polar night, land, sun glint)\n'\

        return cloud_flag_tests, high_cloud_flag_tests, clr_sky_restoral_tests, snow_cover_ancillary_map


    def quality_assurance_byte0(self, dbyte):
        """
        Extract cloud mask QA byte 1 (treated as byte 0 here)
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = unpack_uint_to_bits(dbyte.filled(), 8, bitorder='little')

        # process qa flags
        # Byte 0 only has 4 bits of useful information, other 4 are always 0
        useful_qa = data[0] # usefulness QA flag
        confidence_qa = data[1] + 2 * data[2] + 4 * data[3] # convert to a value between 0 and 7 confidence

        return useful_qa, confidence_qa


    def quality_assurance_byte1(self, dbyte):
        """
        Extract cloud mask QA byte 2 (treated as byte 1 here).
        Note that only some bits are extracted as most other data is already available
        in the main geophysical field data.
        """
        if dbyte.dtype != 'uint8':
            dbyte = dbyte.astype('uint8')

        data = unpack_uint_to_bits(dbyte.filled(), 8, bitorder='little')
        nco_flag = data[0]

        return nco_flag


    def read(self, fname):
        try:
            import netCDF4 as nc
        except ImportError:
            msg = 'Warning [modis_09]: To use \'modis_09\', \'netCDF4\' needs to be installed.'
            raise ImportError(msg)

        f = nc.Dataset(fname, 'r')

        # by default clr sky confidence and cloud mask will be extracted
        clr_sky_confidence = get_data_nc(f['geophysical_data/Clear_Sky_Confidence'], replace_fill_value=np.nan)
        cloud_mask = get_data_nc(f['geophysical_data/Integer_Cloud_Mask'], replace_fill_value=-1).astype('int8')

        # save the data
        self.data = {}
        self.data['clr_sky_confidence'] = dict(name='Clear Sky Confidence', data=clr_sky_confidence, description='The `Clear_Sky_Confidence` is the final numeric value of the confidence of clear sky, or Q value', units='N/A')
        self.data['cloud_mask']         = dict(name='Cloud Mask',           data=cloud_mask,  description='MODIS cloud mask bits 1 & 2 converted to integer\n(0 = cloudy,\n1= probably cloudy,\n2 = probably clear,\n3 = confident clear,\n-1 = no result)\n', units='N/A')


        if (self.mode == 'auto') or (self.mode == 'all'): # extract other data bytes too
            cloud_mask_tests = f['geophysical_data/Cloud_Mask'] # spectral tests
            cloud_mask_tests_dat = get_data_nc(cloud_mask_tests, replace_fill_value=None)
            cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat = self.extract_data_byte0(cloud_mask_tests_dat[0])

            # save the data
            self.data['cloud_mask_flag'] = dict(name='Cloud Mask Flag', data=cloud_mask_flag, units='N/A')
            self.data['day_night_flag']  = dict(name='Day Night Flag',   data=day_night_flag, units='N/A')
            self.data['sunglint_flag']   = dict(name='Sunglint Flag',   data=sunglint_flag, units='N/A')
            self.data['snow_ice_flag']   = dict(name='Snow/ice processing path', data=snow_ice_flag, units='N/A')
            self.data['land_water_flag'] = dict(name='Land/water processing path', data=land_water_cat, units='N/A')
            self.data['fov_flag']        = dict(name='Unobstructed FOV Quality Flag', data=fov_qa_cat, units='N/A')

            if self.mode == 'all':
                cloud_flag_tests, high_cloud_flag_tests, clr_sky_restoral_tests, snow_cover_ancillary_map = self.extract_other_data_bytes(cloud_mask_tests_dat[1], cloud_mask_tests_dat[2], cloud_mask_tests_dat[3])

                # save the data
                self.data['cloud_flag_tests'] = dict(name='Cloud Flags from spectral tests',              data=cloud_flag_tests, description=self.cloud_flag_test_description, units='N/A')
                self.data['high_cloud_flag_tests'] = dict(name='High Cloud Flags from spectral tests',    data=high_cloud_flag_tests, description=self.high_cloud_flag_tests_description, units='N/A')
                self.data['clr_sky_restoral_tests'] = dict(name='Clear sky restoral from spectral tests', data=clr_sky_restoral_tests, description=self.clr_sky_restoral_tests_description, units='N/A')
                self.data['snow_cover_ancillary_map'] = dict(name='Snow cover from ancillary map',        data=snow_cover_ancillary_map, units='N/A')

        # get QA data
        if self.quality_assurance:
            nc_qa = f['geophysical_data/Quality_Assurance']
            qa_dat = get_data_nc(nc_qa, replace_fill_value=None)
            # transpose because for whatever reason this is in reverse order
            qa_dat = np.transpose(qa_dat, axes=(2, 0, 1))

            useful_qa, confidence_qa = self.quality_assurance_byte0(qa_dat[0])
            nco_qa = self.quality_assurance_byte1(qa_dat[1])

            # save the data
            self.qa = {}
            self.qa['useful_qa']     = dict(name='Cloud Mask QA', data=useful_qa, units='N/A')
            self.qa['confidence_qa'] = dict(name='Cloud Mask Confidence QA (8 confidence levels)', data=confidence_qa, units='N/A')
            self.qa['nco_flag']      = dict(name='Non cloud obstruction QA flag', data=nco_qa, units='N/A')



class modis_04:

    """
    Read MODIS 04 deep blue aerosol data

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['AOD_550_land']
                ['Angstrom_Exponent_land']
                ['Aerosol_type_land']
                ['SSA_land']
    """


    ID = 'MODIS 04 Aerosol Product'


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
        Read aerosol properties
        """

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_04]: To use \'modis_04\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        Deep_Blue_AOD_550_land_0            = f.select('Deep_Blue_Aerosol_Optical_Depth_550_Land')
        Deep_Blue_Angstrom_Exponent_land_0  = f.select('Deep_Blue_Angstrom_Exponent_Land')
        Deep_Blue_Aerosol_type_land_0       = f.select('Aerosol_Type_Land')
        Deep_Blue_Aerosol_cloud_frac_land_0 = f.select('Aerosol_Cloud_Fraction_Land')
        Deep_Blue_SSA_land_0                = f.select('Deep_Blue_Spectral_Single_Scattering_Albedo_Land')


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #╭────────────────────────────────────────────────────────────────────────────╮#
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

            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        #╰────────────────────────────────────────────────────────────────────────────╯#


        #
        #╭────────────────────────────────────────────────────────────────────────────╮#
        Deep_Blue_AOD_550_land              = get_data_h4(Deep_Blue_AOD_550_land_0)[logic]
        Deep_Blue_Angstrom_Exponent_land    = get_data_h4(Deep_Blue_Angstrom_Exponent_land_0)[logic]
        Deep_Blue_Aerosol_type_land         = get_data_h4(Deep_Blue_Aerosol_type_land_0)[logic]
        Deep_Blue_Aerosol_cloud_frac_land   = get_data_h4(Deep_Blue_Aerosol_cloud_frac_land_0)[logic]
        Deep_Blue_SSA_land                  = get_data_h4(Deep_Blue_SSA_land_0)[logic]

        f.end()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        if hasattr(self, 'data'):
            self.logic = {}
            self.logic[fname] = {'1km':logic}
            self.data  = {}
            self.data['lon'] = dict(name='Longitude'                                        , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'                                         , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['AOD_550_land'] = dict(name='AOD 550nm (Land)'                        , data=np.hstack((self.data['AOD_550_land']['data'], Deep_Blue_AOD_550_land)), units='None')
            self.data['Angstrom_Exponent_land'] = dict(name='Angstrom Exponent (Land)'      , data=np.hstack((self.data['Angstrom_Exponent_land']['data'], Deep_Blue_Angstrom_Exponent_land)), units='None')
            self.data['Aerosol_type_land'] = dict(name='Aerosol type (Land)'                , data=np.hstack((self.data['aerosol_type_land']['data'], Deep_Blue_Aerosol_type_land)), units='None')
            self.data['Aerosol_cloud_frac_land'] = dict(name='Aerosol Cloud Fraction (Land)', data=np.hstack((self.data['SSA_land']['data'], Deep_Blue_Aerosol_cloud_frac_land)), units='None')
            self.data['SSA_land'] = dict(name='Single Scattering Albedo (Land)'             , data=np.hstack((self.data['SSA_land']['data'], Deep_Blue_SSA_land)), units='None')
        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic}
            self.data  = {}
            self.data['lon'] = dict(name='Longitude'                                        , data=lon                              , units='degrees')
            self.data['lat'] = dict(name='Latitude'                                         , data=lat                              , units='degrees')
            self.data['AOD_550_land'] = dict(name='AOD 550nm (Land)'                        , data=Deep_Blue_AOD_550_land           , units='None')
            self.data['Angstrom_Exponent_land'] = dict(name='Angstrom Exponent (Land)'      , data=Deep_Blue_Angstrom_Exponent_land , units='None')
            self.data['Aerosol_type_land'] = dict(name='Aerosol type (Land)'                , data=Deep_Blue_Aerosol_type_land      , units='None')
            self.data['Aerosol_cloud_frac_land'] = dict(name='Aerosol Cloud Fraction (Land)', data=Deep_Blue_Aerosol_cloud_frac_land, units='None')
            self.data['SSA_land'] = dict(name='Single Scattering Albedo (Land)'             , data=Deep_Blue_SSA_land               , units='None')




    def read_vars(self, fname, vnames=[]):

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_04]: To use \'modis_04\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        logic = self.logic[fname]['1km']
        f     = SD(fname, SDC.READ)

        for vname in vnames:

            data0 = f.select(vname)
            print(get_data_h4(data0).shape)
            data  = get_data_h4(data0)[2,:,:][logic]
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname, data=np.hstack((self.data[vname.lower()]['data'], data)), units=data0.attributes()['units'])
            else:
                self.data[vname.lower()] = dict(name=vname, data=data, units=data0.attributes()['units'])

        f.end()



class modis_09:
    """
    A class for extracting data from MODIS Atmospherically Corrected Surface Reflectance 5-Min L2 Swath 250m, 500m, 1km files.

    Args:
        fname (str): The file name of the MOD09 product.
        resolution (str, optional): The resolution of the data to extract. One of '250m', '500m', or '1km'. Defaults to '1km'.
        param (str, optional): The parameter to extract. One of 'surface_reflectance' or 'tau'.
        ancillary_qa (bool, optional): Flag to get ancillary and qa data. Defaults to False i.e., no ancillary or QA data is extracted
        bands (list, optional): The list of band names. Defaults to extracting bands [1, 4, 3].

    Methods:
        extract_surface_reflectance(hdf_obj): Extracts surface reflectance data for self.bands at self.resolution.
        extract_atmospheric_optical_depth(hdf_obj): Extracts atmospheric optical depth (tau) data.
                                                    Only for 3 "bands" - when using this mode, the data
                                                    can be interpreted as follows:
                                                    index 0 (band 1) = atm. model residual values
                                                    index 1 (band 3) = atm. optical depth
                                                    index 2 (band 8) = angstrom exponent values

    References: (User Guide) https://ladsweb.modaps.eosdis.nasa.gov/archive/Document%20Archive/Science%20Data%20Product%20Documentation/MOD09_C61_UserGuide_v1.7.pdf
                (Filespec) https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD09
    """
    ID = 'MODIS Atmospherically Corrected Surface Reflectance 5-Min L2 Swath 250m, 500m, 1km'


    def __init__(self, \
                 fname,  \
                 resolution = '1km', \
                 param = 'surface_reflectance', \
                 ancillary_qa = False, \
                 bands = [1, 4, 3]):


        self.fname        = fname              # file name
        self.resolution   = resolution.lower() # resolution
        self.param        = param.lower()      # parameter to extract
        self.ancillary_qa = ancillary_qa       # flag to get ancillary and qa data
        self.bands        = bands              # list of band names

        self.read(fname)


    def qa_250m(self, hdf_obj):
        """
        Calculates the quality assurance (QA) values for the 250m Reflectance bands.

        Parameters:
            hdf_obj (object): The HDF object containing the data.

        Returns:
            tuple: A tuple containing the following QA values:
                - modland_qa (int): The MODLAND QA value.
                - all_bands_qa (ndarray): An array containing the QA values for band1 and band2.
                - atm_correction (int): The atmospheric correction QA value.
                - adjacent_correction (int): The adjacent correction QA value.
        """
        band_qa_byte = hdf_obj.select('250m Reflectance Band Quality')
        band_qa_byte = band_qa_byte[:]

        band_qa = unpack_uint_to_bits(band_qa_byte, num_bits=16, bitorder='little')
        modland_qa = 2 * band_qa[0] + band_qa[1]
        band1_qa   = 8 * band_qa[4] + 4 * band_qa[5] + 2 * band_qa[6] + band_qa[7]
        band2_qa   = 8 * band_qa[8] + 4 * band_qa[9] + 2 * band_qa[10] + band_qa[11]
        atm_correction = band_qa[12]
        adjacent_correction = band_qa[13]
        all_bands_qa = np.stack([band1_qa, band2_qa], axis=0)

        return modland_qa, all_bands_qa, atm_correction, adjacent_correction


    def qa_500m_1km(self, hdf_obj):
        """
        Calculates the quality assurance (QA) values for the 500m or 1km Reflectance bands.

        Parameters:
            hdf_obj (object): The HDF object containing the data.

        Returns:
            tuple: A tuple containing the following QA values:
                - modland_qa (int): The MODLAND QA value.
                - all_bands_qa (ndarray): An array containing the QA values for bands 1 through 7.
                - atm_correction (int): The atmospheric correction QA value.
                - adjacent_correction (int): The adjacent correction QA value.
        """
        band_qa_byte = hdf_obj.select('{} Reflectance Band Quality'.format(self.resolution))
        band_qa_byte = band_qa_byte[:]

        band_qa = unpack_uint_to_bits(band_qa_byte, num_bits=32, bitorder='little')
        modland_qa = 2 * band_qa[0] + band_qa[1]
        band1_qa = 8 * band_qa[2] + 4 * band_qa[3] + 2 * band_qa[4] + band_qa[5]
        band2_qa = 8 * band_qa[6] + 4 * band_qa[7] + 2 * band_qa[8] + band_qa[9]
        band3_qa = 8 * band_qa[10] + 4 * band_qa[11] + 2 * band_qa[12] + band_qa[13]
        band4_qa = 8 * band_qa[14] + 4 * band_qa[15] + 2 * band_qa[16] + band_qa[17]
        band5_qa = 8 * band_qa[18] + 4 * band_qa[19] + 2 * band_qa[20] + band_qa[21]
        band6_qa = 8 * band_qa[22] + 4 * band_qa[23] + 2 * band_qa[24] + band_qa[25]
        band7_qa = 8 * band_qa[26] + 4 * band_qa[27] + 2 * band_qa[28] + band_qa[29]
        atm_correction = band_qa[30]
        adjacent_correction = band_qa[31]

        all_bands_qa = np.stack([band1_qa, band2_qa, band3_qa, band4_qa, band5_qa, band6_qa, band7_qa], axis=0)

        return modland_qa, all_bands_qa, atm_correction, adjacent_correction

    def extract_surface_reflectance(self, hdf_obj):
        """ Extract surface reflectance data """
        # get lon/lat
        lon, lat  = hdf_obj.select('Longitude'), hdf_obj.select('Latitude')
        lon, lat = lon[:], lat[:]

        # check that if bands are provided that they are valid
        if (self.bands is not None) and (not set(self.bands).issubset(list(MODIS_L1B_HKM_1KM_BANDS.keys()))):
            raise AttributeError('Error [modis_09]: Your input for `bands`={}\n`bands` must be one of {}\n'.format(self.bands, list(MODIS_L1B_HKM_1KM_BANDS.keys())))

        # resolution and lon/lat interpolation (if needed)
        if self.resolution == '250m':
            lon, lat  = upscale_modis_lonlat(lon, lat, scale=4, extra_grid=False)
            if self.bands is None:
                self.bands = [1, 2]

        elif self.resolution == '500m':
            lon, lat  = upscale_modis_lonlat(lon, lat, scale=2, extra_grid=False)
            if self.bands is None:
                self.bands = [1, 2, 3, 4, 5, 6, 7]

        elif self.resolution == '1km':
            if self.bands is None:
                self.bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26]

        else:
            raise AttributeError('Error [modis_09]: `resolution` must be one of `250m`, `500m`, or `1km`')

        # begin extracting data
        # search datasets containing the search term derived from param and resolution
        search_term = self.resolution + ' ' +  ' '.join(self.param.title().split('_'))
        search_terms_with_bands = [search_term + ' ' + 'Band {}'.format(str(band)) for band in self.bands]
        params = [i for i in list(hdf_obj.datasets().keys()) if i in search_terms_with_bands] # list of dataset names
        if len(search_terms_with_bands) != len(params):
            print('Warning [modis_09]: Not all bands were extracted. Check self.bands and self.resolution inputs')

        # use the first param to get shape
        data_shape = tuple(hdf_obj.select(params[0]).dimensions().values())
        surface_reflectance = np.zeros((len(params), data_shape[0], data_shape[1]), dtype=np.float32)
        wvl = np.zeros(len(self.bands), dtype='uint16') # wavelengths

        # loop through bands, scale and offset each param and store in tau
        for idx, band_num in enumerate(self.bands):
            surface_reflectance[idx] = get_data_h4(hdf_obj.select(params[idx]))
            wvl[idx] = MODIS_L1B_HKM_1KM_BANDS[band_num]

        if self.ancillary_qa:
            if self.resolution == '250m':
                modland_qa, all_bands_qa, atm_correction, adjacent_correction = self.qa_250m(hdf_obj)
            else:
                modland_qa, all_bands_qa, atm_correction, adjacent_correction = self.qa_500m_1km(hdf_obj)


        # save the data
        if hasattr(self, 'data'):

            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)),     units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)),     units='degrees')
            self.data['wvl'] = dict(name='Wavelength'          , data=np.hstack((self.data['wvl']['data'], wvl)),     units='nm')
            self.data['surface_reflectance'] = dict(name='Surface Reflectance', data=np.hstack((self.data['surface_reflectance']['data'], surface_reflectance)),     units='N/A')

            if self.ancillary_qa:
                self.data['modland_qa'] = dict(name='MODLAND QA'  , data=np.hstack((self.data['modland_qa']['data'], modland_qa)),     units='N/A')
                self.data['all_bands_qa'] = dict(name='Band QA for all avaialble bands in increasing order (band 1, band 2, etc.)'  , data=np.hstack((self.data['all_bands_qa']['data'], all_bands_qa)),     units='N/A')
                self.data['atm_correction_qa'] = dict(name='Atmospheric correction QA'  , data=np.hstack((self.data['atm_correction_qa']['data'], atm_correction)),     units='N/A')
                self.data['adjacent_correction_qa'] = dict(name='Adjacency correction QA'  , data=np.hstack((self.data['adjacent_correction_qa']['data'], adjacent_correction)),     units='N/A')

        else:

            self.data = {}
            self.data['lon'] = dict(name='Longitude'               , data=lon,     units='degrees')
            self.data['lat'] = dict(name='Latitude'                , data=lat,     units='degrees')
            self.data['wvl'] = dict(name='Wavelength'              , data=wvl,     units='nm')
            self.data['surface_reflectance'] = dict(name='Surface Reflectance', data=surface_reflectance,     units='N/A')

            if self.ancillary_qa:
                self.qa = {}
                self.qa['modland_qa'] = dict(name='MODLAND QA'  , data=modland_qa,     units='N/A')
                self.qa['all_bands_qa'] = dict(name='Band QA for all available bands in increasing order (band 1, band 2, etc.)'  , data=all_bands_qa,     units='N/A')
                self.qa['atm_correction_qa'] = dict(name='Atmospheric correction QA'  , data=atm_correction,     units='N/A')
                self.qa['adjacent_correction_qa'] = dict(name='Adjacency correction QA'  , data=adjacent_correction,     units='N/A')


    def extract_atmospheric_optical_depth(self, hdf_obj):
        """
        Extract atmospheric optical depth data

        For information on how to interpret the data:
        https://ladsweb.modaps.eosdis.nasa.gov/archive/Document%20Archive/Science%20Data%20Product%20Documentation/MOD09_C61_UserGuide_v1.7.pdf
        """

        # get lon/lat
        lon, lat  = hdf_obj.select('Longitude'), hdf_obj.select('Latitude')
        lon, lat = lon[:], lat[:]

        # allow others for input but change it internally to use for keyword search
        if self.param != 'atmospheric_optical_depth':
            self.param = 'atmospheric_optical_depth'


        # for atm tau, only bands 1 (650nm), 3 (470nm), 8 (412nm) are present at 1km in the MOD09 product
        if (self.bands is not None) or (self.resolution is not None):
            print('Warning [modis_09]: `bands` and `resolution` are ignored when extracting atmospheric optical depth as only preset bands are available in the MOD09 product, all at 1km\n')

        self.bands = [1, 3, 8]
        self.resolution = '1km'

        # begin extracting data
        # search datasets containing the search term derived from param and resolution
        search_term = self.resolution + ' ' +  ' '.join(self.param.title().split('_'))
        search_terms_with_bands = [search_term + ' ' + 'Band {}'.format(str(band)) for band in self.bands]
        params = [i for i in list(hdf_obj.datasets().keys()) if i in search_terms_with_bands] # list of dataset names

        # use the first param to get shape
        data_shape = tuple(hdf_obj.select(params[0]).dimensions().values())
        tau = np.zeros((len(params), data_shape[0], data_shape[1]), dtype=np.float32) # atm optical depth
        wvl = np.zeros(len(self.bands), dtype='uint16') # wavelengths

        # loop through bands, scale and offset each param and store in tau
        for idx, band_num in enumerate(self.bands):
            tau[idx] = get_data_h4(hdf_obj.select(params[idx]), init_dtype='int16') # initially signed int
            wvl[idx] = MODIS_L1B_HKM_1KM_BANDS[band_num]

        if self.ancillary_qa: # get internal cloud mask, qa, and water vapor fields
            tau_model_hdf  = hdf_obj.select('1km Atmospheric Optical Depth Model')
            tau_model      = get_data_h4(tau_model_hdf, init_dtype='uint8', replace_fill_value=0).astype('uint8')
            tau_model_desc = tau_model_hdf.attributes()['long_name'] + '\n' +  tau_model_hdf.attributes()['Model values']

            tau_qa_hdf     = hdf_obj.select('1km Atmospheric Optical Depth Band QA')
            tau_qa         = get_data_h4(tau_qa_hdf, init_dtype='uint16', replace_fill_value=0).astype('uint16')
            # convert bits to 16 bit unsigned ints with the 16, 8, 4, 2, 1, 0 order
            # index 0 = bit 15, index 1 = bit 14, etc.
            tau_qa         = unpack_uint_to_bits(tau_qa, num_bits=16, bitorder='big')
            tau_qa_desc    = tau_qa_hdf.attributes()['long_name'] + '\n' + tau_qa_hdf.attributes()['QA index']

            tau_cloud_mask_hdf = hdf_obj.select('1km Atmospheric Optical Depth Band CM')
            tau_cloud_mask = get_data_h4(tau_cloud_mask_hdf, init_dtype='uint8', replace_fill_value=0).astype('uint8')
            tau_cloud_mask_desc = tau_cloud_mask_hdf.attributes()['long_name'] + '\n' + tau_cloud_mask_hdf.attributes()['QA index']

            water_vapor_hdf = hdf_obj.select('1km water_vapor')
            water_vapor     = get_data_h4(water_vapor_hdf, init_dtype='uint16')


        # save the data
        self.data = {}
        self.data['lon'] = dict(name='Longitude'               , data=lon,     units='degrees')
        self.data['lat'] = dict(name='Latitude'                , data=lat,     units='degrees')
        self.data['wvl'] = dict(name='Wavelength'              , data=wvl,     units='nm')
        self.data['tau'] = dict(name='Atmospheric Optical Depth', data=tau,     units='N/A')

        if self.ancillary_qa:
            self.qa = {}
            self.qa['tau_model'] = dict(name='1km Atmospheric Optical Depth Model',    data=tau_model, description=tau_model_desc, units='N/A')
            self.qa['tau_qa']    = dict(name='1km Atmospheric Optical Depth Band QA',  data=tau_qa,    description=tau_qa_desc,    units='N/A')
            self.qa['tau_cloud_mask']  = dict(name='1km Atmospheric Optical Depth Band CM',  data=tau_cloud_mask,       description=tau_cloud_mask_desc,    units='N/A')
            self.qa['water_vapor']    = dict(name='1km Water Vapor',  data=water_vapor,    units='g/cm^2')


    def read(self, fname):

        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_09]: To use \'modis_09\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f = SD(fname, SDC.READ)
        if self.param == 'surface_reflectance':
            self.extract_surface_reflectance(f)

        elif (self.param == 'atmospheric_optical_depth') or (self.param == 'optical_depth') or (self.param == 'tau'):
            self.extract_atmospheric_optical_depth(f)

        f.end()


class modis_09a1:

    """
    Read MOD09A1 product (8 day of surface reflectance in sinusoidal projection)

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


    ID = 'MODIS surface reflectance (500 m, 8 day)'


    def __init__(self,
                 fnames=None,
                 extent=None,
                 Nx=2400,
                 Ny=2400,
                 verbose=False):

        self.fnames = fnames
        self.extent = extent
        self.Nx = Nx
        self.Ny = Ny

        for fname in self.fnames:
            self.read(fname)


    def read(self, fname):

        filename     = os.path.basename(fname)
        index_str    = filename.split('.')[2]
        index_h = int(index_str[1:3])
        index_v = int(index_str[4:])

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error   [modis_09a1]: To use \'modis_09a1\', \'cartopy\' needs to be installed.'
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
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

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
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Warning [modis_09a1]: To use \'modis_09a1\', \'pyhdf\' needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        Nchan = 7 # wavelengths are 0:620-670nm, 1:841-876nm, 2:459-479nm, 3:545-565nm, 4:1230-1250nm, 5:1628-1652nm, 6:2105-2155nm
        ref = np.zeros((Nchan, logic.sum()), dtype=np.float32)
        for ichan in range(Nchan):
            data0 = f.select('sur_refl_b%2.2d' % (ichan+1))
            data  = get_data_h4(data0)[logic]
            ref[ichan, :] = data

        ref[ref>1.0] = 0.0
        ref[ref<0.0] = 0.0

        f.end()

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



class modis_43a1:

    """
    Read MCD43A1 product (surface BRDF [Ross-Thick-Li-Sparse-Reciprocal model]in sinusoidal projection)

    Input:
        fnames=   : keyword argument, default=None, a Python list of the file path of the files
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        Nx=       : keyword argument, default=2400, number of points along x direction
        Ny=       : keyword argument, default=2400, number of points along y direction
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['f_iso']: Isotropic (iso), all 7 channels
                ['f_vol']: RossThick (vol), all 7 channels
                ['f_geo']: LiSparseR (geo), all 7 channels
                ['lon']: longitude
                ['lat']: latitude
                ['x']  : sinusoidal x
                ['y']  : sinusoidal y
    """


    ID = 'MODIS surface BRDF (500 m)'


    def __init__(self,
                 fnames=None,
                 extent=None,
                 Nx=2400,
                 Ny=2400,
                 verbose=False):

        self.fnames = fnames
        self.extent = extent
        self.Nx = Nx
        self.Ny = Ny

        for fname in self.fnames:
            self.read(fname)


    def read(self, fname):

        import cartopy.crs as ccrs
        from pyhdf.SD import SD, SDC

        filename     = os.path.basename(fname)
        index_str    = filename.split('.')[2]
        index_h = int(index_str[1:3])
        index_v = int(index_str[4:])

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
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        lon   = LonLat[..., 0]
        lat   = LonLat[..., 1]

        logic = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])

        lon = lon[logic]
        lat = lat[logic]
        x   = XX[logic]
        y   = YY[logic]

        if self.extent is None:
            self.extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        f     = SD(fname, SDC.READ)

        Nchan = 7 # wavelengths are 0:620-670nm, 1:841-876nm, 2:459-479nm, 3:545-565nm, 4:1230-1250nm, 5:1628-1652nm, 6:2105-2155nm
        f_iso = np.zeros((Nchan, logic.sum()), dtype=np.float32)
        f_vol = np.zeros((Nchan, logic.sum()), dtype=np.float32)
        f_geo = np.zeros((Nchan, logic.sum()), dtype=np.float32)

        for ichan in range(Nchan):
            data0 = f.select('BRDF_Albedo_Parameters_Band%d' % (ichan+1))
            data  = get_data_h4(data0)[logic, :]
            f_iso[ichan, :] = data[..., 0]
            f_vol[ichan, :] = data[..., 1]
            f_geo[ichan, :] = data[..., 2]

        f.end()

        if hasattr(self, 'data'):
            self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=np.hstack((self.data['x']['data'], x))    , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=np.hstack((self.data['y']['data'], y))    , units='m')
            self.data['f_iso'] = dict(name='Isotropic coefficient'           , data=np.hstack((self.data['f_iso']['data'], f_iso)), units='N/A')
            self.data['f_vol'] = dict(name='Ross-Thick coefficient'          , data=np.hstack((self.data['f_vol']['data'], f_vol)), units='N/A')
            self.data['f_geo'] = dict(name='Li-Sparse-Reciprocal coefficient', data=np.hstack((self.data['f_geo']['data'], f_geo)), units='N/A')
        else:
            self.data = {}
            self.data['lon'] = dict(name='Longitude'           , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'            , data=lat, units='degrees')
            self.data['x']   = dict(name='X of sinusoidal grid', data=x  , units='m')
            self.data['y']   = dict(name='Y of sinusoidal grid', data=y  , units='m')
            self.data['f_iso'] = dict(name='Isotropic coefficient'           , data=f_iso, units='N/A')
            self.data['f_vol'] = dict(name='Ross-Thick coefficient'          , data=f_vol, units='N/A')
            self.data['f_geo'] = dict(name='Li-Sparse-Reciprocal coefficient', data=f_geo, units='N/A')



class modis_43a3:

    """
    Read MCD43A3 product (surface albedo in sinusoidal projection)

    Input:
        fnames=   : keyword argument, default=None, a Python list of the file path of the files
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        Nx=       : keyword argument, default=2400, number of points along x direction
        Ny=       : keyword argument, default=2400, number of points along y direction
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


    ID = 'MODIS surface albedo (500 m)'


    def __init__(self,
                 fnames=None,
                 extent=None,
                 Nx=2400,
                 Ny=2400,
                 verbose=False):

        self.fnames = fnames
        self.extent = extent
        self.Nx = Nx
        self.Ny = Ny

        for fname in self.fnames:
            self.read(fname)


    def read(self, fname):

        filename     = os.path.basename(fname)
        index_str    = filename.split('.')[2]
        index_h = int(index_str[1:3])
        index_v = int(index_str[4:])

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error [modis_43a3]: To use <modis_43a3>, <cartopy> needs to be installed.'
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
            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

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
            from pyhdf.SD import SD, SDC
        except ImportError:
            msg = 'Error [modis_43a3]: To use <modis_43a3>, <pyhdf> needs to be installed.'
            raise ImportError(msg)

        f     = SD(fname, SDC.READ)

        Nchan = 7 # wavelengths are 0:620-670nm, 1:841-876nm, 2:459-479nm, 3:545-565nm, 4:1230-1250nm, 5:1628-1652nm, 6:2105-2155nm
        bsky_alb = np.zeros((Nchan, logic.sum()), dtype=np.float32)
        wsky_alb = np.zeros((Nchan, logic.sum()), dtype=np.float32)

        for ichan in range(Nchan):
            data0 = f.select('Albedo_BSA_Band%d' % (ichan+1))
            data  = get_data_h4(data0)[logic]
            bsky_alb[ichan, :] = data

            data0 = f.select('Albedo_WSA_Band%d' % (ichan+1))
            data  = get_data_h4(data0)[logic]
            wsky_alb[ichan, :] = data

        bsky_alb[bsky_alb>1.0] = -1.0
        bsky_alb[bsky_alb<0.0] = -1.0

        wsky_alb[wsky_alb>1.0] = -1.0
        wsky_alb[wsky_alb<0.0] = -1.0

        f.end()

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



class modis_tiff:

    """
    Read geotiff file downloaded from NASA WorldView

    Input:
        fname: file path of the geotiff file

    Output:
        mod_geotiff object that contains
        self.lon: longitude
        self.lat: latitude
    """

    def __init__(
            self,
            fname,
            extent=None,
            verbose=False,
            quiet=True
            ):

        try:
            import gdal
        except ImportError:
            msg = 'Error   [mod_geotiff]: To use \'mod_geotiff\', \'gdal\' needs to be installed.'
            raise ImportError(msg)

        f       = gdal.Open(fname)
        dataRGB = f.ReadAsArray()
        pixelNx = f.RasterXSize
        pixelNy = f.RasterYSize
        geoInfo = f.GetGeoTransform()
        if verbose:
            print(geoInfo)

        x0 = np.arange(pixelNx)
        y0 = np.arange(pixelNy)
        yy0, xx0 = np.meshgrid(y0, x0)

        # if in normal projection, xx, yy represent longitude and latitude
        # if in  polar projection, a further step is needed to get longitude and latitude
        xx  = np.transpose(geoInfo[0] + geoInfo[1]*xx0 + geoInfo[2]*yy0)
        yy  = np.transpose(geoInfo[3] + geoInfo[4]*xx0 + geoInfo[5]*yy0)
        img = np.transpose(dataRGB,(1,2,0)) # gets first dim for zeroth dim, etc. - gets same format as imgread
        mx  = float(np.max(img))
        img = img/mx
        lon,lat=xx,yy

        nx,ny,nl = img.shape
        self.img = img
        self.lon = lon
        self.lat = lat
        self.row = nx # this is picture-centric (for imshow methods)
        self.col = ny
        self.nx  = ny # this is grid-centric (for processing and other plotting methods)
        self.ny  = nx
        # make "wesn" array
        # tif only reports left/north corner of each pixel, so need to take this into account
        west  = lon[nx-1,0]     # southwest corner
        east  = lon[0,ny-1]     # northeast corner
        south = lat[nx-1,0]     # southwest corner
        north = lat[0,ny-1]     # northeast corner
        nxh   = int(nx/2)       # center of image domain
        nyh   = int(ny/2)
        self.gns   = lat[nxh,nyh]-lat[nxh+1,nyh] # gridding (degrees North/South)
        self.gwe   = lon[nxh,nyh]-lon[nxh,nyh-1] # gridding (degrees West/East)
        wesn=np.array([west,east+self.gwe,south-self.gns,north])  # see comment above: tif [reports corner] --> needed: image extent
        self.wesn=wesn
        # make grid
        # for nx pixels, we have nx+1 boundaries
        self.long = np.linspace(wesn[0],wesn[1],ny+1)
        self.latg = np.linspace(wesn[2],wesn[3],nx+1)
        self.lonm   = 0.5*(self.long[1:]+self.long[:-1])
        self.latm   = 0.5*(self.latg[1:]+self.latg[:-1])

#╰────────────────────────────────────────────────────────────────────────────╯#





# Useful functions
#╭────────────────────────────────────────────────────────────────────────────╮#

def upscale_modis_lonlat(lon_in, lat_in, scale=5, extra_grid=True):

    """
    Upscaling MODIS geolocation from 5km/1km/1km to 1km/250m/500nm.

    Details can be found at
    http://www.icare.univ-lille1.fr/tutorials/MODIS_geolocation

    Input:
        lon_in: numpy array, input longitude
        lat_in: numpy array, input latitude
        scale=: integer, upscaling factor, e.g., 5km to 1km (scale=5), 1km to 250m (scale=4), 1km to 500nm (scale=2)
        extra_grid=: boolen, for MOD/MYD 05, 06 data, extra_grid=True, for other dataset, extra_grid=False

    Output:
        lon_out: numpy array, upscaled longitude
        lat_out: numpy array, upscaled latitude

    How to use:
        # After read in the longitude latitude from MODIS L2 06 data, e.g., lon0, lat0
        lon, lat = upscale_modis_lonlat(lon0, lat0, scale=5, extra_grid=True)
    """

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = 'Error   [upscale_modis_lonlat]: To use \'upscale_modis_lonlat\', \'cartopy\' needs to be installed.'
        raise ImportError(msg)

    offsets_dict = {
            4: {'along_scan':0, 'along_track':1.5},
            5: {'along_scan':2, 'along_track':2},
            2: {'along_scan':0, 'along_track':0.5}
            }
    offsets = offsets_dict[scale]

    lon_in[lon_in>180.0] -= 360.0
    # +
    # find center lon, lat
    proj_lonlat = ccrs.PlateCarree()

    lon0 = np.array([lon_in[0, 0], lon_in[-1, 0], lon_in[-1, -1], lon_in[0, -1], lon_in[0, 0]])
    lat0 = np.array([lat_in[0, 0], lat_in[-1, 0], lat_in[-1, -1], lat_in[0, -1], lat_in[0, 0]])

    if (abs(lon0[0]-lon0[1])>180.0) | (abs(lon0[0]-lon0[2])>180.0) | \
       (abs(lon0[0]-lon0[3])>180.0) | (abs(lon0[1]-lon0[2])>180.0) | \
       (abs(lon0[1]-lon0[3])>180.0) | (abs(lon0[2]-lon0[3])>180.0):

        lon0[lon0<0.0] += 360.0

    center_lon_tmp = lon0[:-1].mean()
    center_lat_tmp = lat0[:-1].mean()

    proj_tmp  = ccrs.Orthographic(central_longitude=center_lon_tmp, central_latitude=center_lat_tmp)
    xy_tmp    = proj_tmp.transform_points(proj_lonlat, lon0, lat0)[:, [0, 1]]
    center_x  = xy_tmp[:, 0].mean()
    center_y  = xy_tmp[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_tmp)
    # -

    proj_xy  = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy_in = proj_xy.transform_points(proj_lonlat, lon_in, lat_in)[:, :, [0, 1]]

    y_in, x_in = lon_in.shape
    XX_in = offsets['along_scan']  + np.arange(x_in) * scale   # along scan
    YY_in = offsets['along_track'] + np.arange(y_in) * scale   # along track

    x  = x_in * scale
    y  = y_in * scale

    if scale==5 and extra_grid:
        XX = np.arange(x+4)
    else:
        XX = np.arange(x)

    YY = np.arange(y)

    # Deprecated since Scipy v1.12
    # f_x = interpolate.interp2d(XX_in, YY_in, xy_in[:, :, 0], kind='linear', fill_value=None)
    # f_y = interpolate.interp2d(XX_in, YY_in, xy_in[:, :, 1], kind='linear', fill_value=None)

    XX_grid, YY_grid = np.meshgrid(XX, YY, indexing='ij')

    # note that RegularGridInterpolator requires transposed data
    f_x = interpolate.RegularGridInterpolator((XX_in, YY_in), xy_in[:, :, 0].T, method='linear', fill_value=None, bounds_error=False)
    f_y = interpolate.RegularGridInterpolator((XX_in, YY_in), xy_in[:, :, 1].T, method='linear', fill_value=None, bounds_error=False)

    # reverse the transpose to get back original form
    lonlat = proj_lonlat.transform_points(proj_xy, f_x((XX_grid, YY_grid)).T, f_y((XX_grid, YY_grid)).T)[:, :, [0, 1]]

    return lonlat[:, :, 0], lonlat[:, :, 1]



def download_modis_rgb(
        date,
        extent,
        which='terra',
        # wmts_cgi='https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
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
            msg = 'Error   [download_modis_rgb]: To use \'download_modis_rgb\', \'matplotlib\' needs to be installed.'
            raise ImportError(msg)

        try:
            from owslib.wmts import WebMapTileService
        except ImportError:
            msg = 'Error   [download_modis_rgb]: To use \'download_modis_rgb\', \'owslib\' needs to be installed.'
            raise ImportError(msg)

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error   [download_modis_rgb]: To use \'download_modis_rgb\', \'cartopy\' needs to be installed.'
            raise ImportError(msg)

        if which == 'terra':
            layer_name = 'MODIS_Terra_CorrectedReflectance_TrueColor'
        elif which == 'aqua':
            layer_name = 'MODIS_Aqua_CorrectedReflectance_TrueColor'
        else:
            sys.exit('Error   [download_modis_rgb]: Only support \'which="aqua"\' or \'which="terra"\'.')

        if proj is None:
            proj=ccrs.PlateCarree()


        import cartopy.io.ogc_clients as ogcc
        ogcc._URN_TO_CRS['urn:ogc:def:crs:EPSG:6.18:3:3857'] = ccrs.GOOGLE_MERCATOR
        ogcc.METERS_PER_UNIT['urn:ogc:def:crs:EPSG:6.18:3:3857'] = 1

        from matplotlib.axes import Axes
        from cartopy.mpl.geoaxes import GeoAxes
        #GeoAxes._pcolormesh_patched = Axes.pcolormesh

        wmts = WebMapTileService(wmts_cgi)

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111, projection=proj)
        ax1.add_wmts(wmts, layer_name, wmts_kwargs={'time': date_s})
        if coastline:
            ax1.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        #ax1.outline_patch.set_visible(False)
        ax1.patch.set_visible(False)
        ax1.axis('off')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    return fname



def download_modis_https(
             date,
             dataset_tag,
             filename_tag,
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             fdir_prefix='/archive/allData',
             day_interval=1,
             fdir_out='data',
             data_format=None,
             run=True,
             quiet=False,
             verbose=False):


    """
    Input:
        date: Python datetime object
        dataset_tag: string, collection + dataset name, e.g. '61/MYD06_L2'
        filename_tag: string, string pattern in the filename, e.g. '.2035.'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        day_interval=: integer, for 8 day data, day_interval=8
        fdir_out=: string, output data directory
        data_format=None: e.g., 'hdf'
        run=: boolen type, if true, the command will only be displayed but not run
        quiet=: Boolen type, quiet tag
        verbose=: Boolen type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded MODIS file paths
    """

    try:
        app_key = os.environ['MODIS_APP_KEY']
    except KeyError:
        app_key = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'
        if verbose:
            print('Warning [download_modis_https]: Please get your app key by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#appkeys\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.bash_profile\'(Mac),\nexport MODIS_APP_KEY="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"\n')

    if shutil.which('curl'):
        command_line_tool = 'curl'
    elif shutil.which('wget'):
        command_line_tool = 'wget'
    else:
        sys.exit('Error [download_modis_https]: \'download_modis_https\' needs \'curl\' or \'wget\' to be installed.')

    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)

    fdir_server = server + fdir_data
    webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
    content  = webpage.read().decode('utf-8')
    lines    = content.split('\n')

    commands = []
    fnames_local = []
    for line in lines:
        filename = line.strip().split(',')[0]
        if filename_tag in filename:
            fname_server = '%s/%s' % (fdir_server, filename)
            fname_local  = '%s/%s' % (fdir_out, filename)
            fnames_local.append(fname_local)
            if command_line_tool == 'curl':
                command = 'mkdir -p %s && curl -H \'Authorization: Bearer %s\' -L -C - \'%s\' -o \'%s\'' % (fdir_out, app_key, fname_server, fname_local)
            elif command_line_tool == 'wget':
                command = 'mkdir -p %s && wget -c "%s" --header "Authorization: Bearer %s" -O %s' % (fdir_out, fname_server, app_key, fname_local)
            commands.append(command)

    if not run:

        if not quiet:
            print('Message [download_modis_https]: The commands to run are:')
            for command in commands:
                print(command)
                print()

    else:

        for i, command in enumerate(commands):

            print('Message [download_modis_https]: Downloading %s ...' % fnames_local[i])
            os.system(command)

            fname_local = fnames_local[i]

            if data_format is None:
                data_format = os.path.basename(fname_local).split('.')[-1]

            if data_format == 'hdf':

                try:
                    from pyhdf.SD import SD, SDC
                except ImportError:
                    msg = 'Warning [download_modis_https]: To use \'download_modis_https\', \'pyhdf\' needs to be installed.'
                    raise ImportError(msg)

                f = SD(fname_local, SDC.READ)
                f.end()
                print('Message [download_modis_https]: \'%s\' has been downloaded.\n' % fname_local)

            else:

                print('Warning [download_modis_https]: Do not support check for \'%s\'. Do not know whether \'%s\' has been successfully downloaded.\n' % (data_format, fname_local))

    return fnames_local



def get_filename_tag(
             date,
             lon,
             lat,
             satID='aqua',
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             fdir_prefix='/archive/geoMeta/61',
             verbose=False):

    """
    Input:
        date: Python datetime.datetime object
        lon : longitude of, e.g. flight track
        lat : latitude of, e.g. flight track
        satID=: default "aqua", can also change to "terra"
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        verbose=: Boolen type, verbose tag

    output:
        filename_tags: Python list of file name tags
    """

    satID     = satID.lower()
    data_tags = dict(terra='MOD03', aqua='MYD03')
    filename  = '%s_%s.txt' % (data_tags[satID], date.strftime('%Y-%m-%d'))

    fname = '%s/%s/%4.4d/%s' % (fdir_prefix, satID.upper(), date.year, filename)

    fname_server = server + fname


    try:
        username = os.environ['EARTHDATA_USERNAME']
        password = os.environ['EARTHDATA_PASSWORD']
    except Exception as err:
        exit('Error   [get_filename_tag]: {}\nCannot find environment variables \'EARTHDATA_USERNAME\' and \'EARTHDATA_PASSWORD\'.'.format(err))

    try:
        with requests.Session() as session:
            session.auth = (username, password)
            r1     = session.request('get', fname_server)
            r      = session.get(r1.url, auth=(username, password))
            if r.ok:
                content = r.content.decode('utf-8')
    except Exception as err:
        exit('Error   [get_filename_tag]: {}\nCannot access {}.'.format(err, fname_server))

    dtype = ['|S41','|S16','<i4','<f8','|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    data  = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype)
    # data['GranuleID'].decode('UTF-8') to get the file name of MODIS granule
    # data['StartDateTime'].decode('UTF-8') to get the time stamp of MODIS granule
    # variable names can be found through
    # print(data.dtype.names)

    Ndata = data.size

    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = 'Error   [get_filename_tag]: To use \'get_filename_tag\', \'cartopy\' needs to be installed.'
        raise ImportError(msg)

    try:
        import matplotlib.path as mpl_path
    except ImportError:
        msg = 'Error   [get_filename_tag]: To use \'get_filename_tag\', \'matplotlib\' needs to be installed.'
        raise ImportError(msg)

    filename_tags = []

    # loop through all the "MODIS granules" constructed through four corner points
    # and find which granules contain the input data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_ori = ccrs.PlateCarree()
    for i in range(Ndata):

        line = data[i]
        xx0  = np.array([line['GRingLongitude1'], line['GRingLongitude2'], line['GRingLongitude3'], line['GRingLongitude4'], line['GRingLongitude1']])
        yy0  = np.array([line['GRingLatitude1'] , line['GRingLatitude2'] , line['GRingLatitude3'] , line['GRingLatitude4'] , line['GRingLatitude1']])

        if (abs(xx0[0]-xx0[1])>180.0) | (abs(xx0[0]-xx0[2])>180.0) | \
           (abs(xx0[0]-xx0[3])>180.0) | (abs(xx0[1]-xx0[2])>180.0) | \
           (abs(xx0[1]-xx0[3])>180.0) | (abs(xx0[2]-xx0[3])>180.0):

            xx0[xx0<0.0] += 360.0

        # roughly determine the center of granule
        #╭──────────────────────────────────────────────────────────────╮#
        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()
        #╰──────────────────────────────────────────────────────────────╯#

        # find the precise center point of MODIS granule
        #╭──────────────────────────────────────────────────────────────╮#
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        #╰──────────────────────────────────────────────────────────────╯#

        proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
        LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

        modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
        pointsIn       = modis_granule.contains_points(LonLat_in)
        percentIn      = float(pointsIn.sum()) / float(pointsIn.size) * 100.0
        if pointsIn.sum()>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
            filename = data[i]['GranuleID'].decode('UTF-8')
            filename_tag = '.'.join(filename.split('.')[1:3])
            filename_tags.append(filename_tag)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return filename_tags



def cal_sinusoidal_grid():

    """
    Calculate MODIS sinusoidal grid bounds in x and y, units: m

    Input:
        No input is required

    Output:
        grid_x: numpy array, boundaries of 36 grid boxes along x
        grid_y: numpy array, boundaries of 18 grid boxes along y
    """

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = 'Error   [cal_sinusoidal_grid]: To use \'cal_sinusoidal_grid\', \'cartopy\' needs to be installed.'
        raise ImportError(msg)

    proj_xy     = ccrs.Sinusoidal.MODIS
    proj_lonlat = ccrs.PlateCarree()

    x0 = proj_xy.transform_points(proj_lonlat, np.array([-180]), np.array(  [0]))[..., 0]
    x1 = proj_xy.transform_points(proj_lonlat, np.array( [180]), np.array(  [0]))[..., 0]
    y0 = proj_xy.transform_points(proj_lonlat, np.array(   [0]), np.array( [90]))[..., 1]
    y1 = proj_xy.transform_points(proj_lonlat, np.array(   [0]), np.array([-90]))[..., 1]

    # grid boxes
    grid_x = np.linspace(x0, x1, 37)
    grid_y = np.linspace(y0, y1, 19)

    return grid_x, grid_y



def get_sinusoidal_grid_tag(lon, lat, verbose=False):

    """
    Get sinusoidal grid tag, e.g., 'h10v17', that contains input track defined by longitude and latitude

    Input:
        lon: numpy array, longitude
        lat: numpy array, latitude
        verbose=: Boolen type, verbose tag

    Output:
        tile_tags: Python list of strings of sinusoidal grid tag
    """

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = 'Error   [get_sinusoidal_grid_tag]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()

    proj_xy     = ccrs.Sinusoidal.MODIS
    proj_lonlat = ccrs.PlateCarree()
    xy_in       = proj_xy.transform_points(proj_lonlat, lon, lat)[..., [0, 1]]

    grid_x, grid_y = cal_sinusoidal_grid()

    tile_tags = []
    for index_h in range(36):
        for index_v in range(18):

            logic = (xy_in[:, 0]>=grid_x[index_h]) & (xy_in[:, 0]<=grid_x[index_h+1]) & (xy_in[:, 1]>=grid_y[index_v+1]) & (xy_in[:, 1]<=grid_y[index_v])
            if logic.sum() > 0:
                tile_tag = 'h%2.2dv%2.2d' % (index_h, index_v)
                if verbose:
                    print('Message [get_sinusoidal_grid_tag]: \'%s\' contains %d/%d.' % (tile_tag, logic.sum(), logic.size))
                tile_tags.append(tile_tag)

    return tile_tags



def find_fname_match(fname0, fnames, index_s=1, index_e=3):

    filename0 = os.path.basename(fname0)
    pattern  = '.'.join(filename0.split('.')[index_s:index_e+1])

    fname_match = None
    for fname in fnames:
        if pattern in fname:
            fname_match = fname

    return fname_match

#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__=='__main__':

    pass
