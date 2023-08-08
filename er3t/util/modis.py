import os
import sys
import datetime
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import shutil
import urllib.request
import requests
from er3t.util import check_equal, get_doy_tag, get_data_h4



__all__ = ['modis_l1b', 'modis_l2', 'modis_35_l2', 'modis_03', 'modis_04', 'modis_09a1', 'modis_43a3', 'modis_tiff', 'upscale_modis_lonlat', \
           'download_modis_rgb', 'download_modis_https', 'cal_sinusoidal_grid', 'get_sinusoidal_grid_tag']


# reader for MODIS (Moderate Resolution Imaging Spectroradiometer)
#/-----------------------------------------------------------------------------\

class modis_l1b:

    """
    Read MODIS Level 1B file into an object `modis_l1b`

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original HDF4 files
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


    ID = 'MODIS Level 1B Calibrated Radiance'


    def __init__(self, \
                 fnames    = None, \
                 f03       = None, \
                 extent    = None, \
                 resolution= None, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the hdf files
        self.f03        = f03         # geolocation class object created using the `modis_03` reader
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag


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
        Read radiance/reflectance/corrected counts along with their uncertainties from the MODIS L1B data
        self.data
            ['lon']
            ['lat']
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
            lat0      = f.select('Latitude')
            lon0      = f.select('Longitude')
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=4, extra_grid=False)
            raw0      = f.select('EV_250_RefSB')
            uct0      = f.select('EV_250_RefSB_Uncert_Indexes')
            wvl       = np.array([650.0, 860.0])
            do_region = True

        # when resolution equals to 500 m
        elif check_equal(self.resolution, 0.5):
            lat0      = f.select('Latitude')
            lon0      = f.select('Longitude')
            lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=2, extra_grid=False)
            raw0      = f.select('EV_500_RefSB')
            uct0      = f.select('EV_500_RefSB_Uncert_Indexes')
            wvl       = np.array([470.0, 555.0, 1240.0, 1640.0, 2130.0])
            do_region = True

        # when resolution equals to 1000 m
        elif check_equal(self.resolution, 1.0):
            if self.f03 is not None:
                raw0_250  = f.select('EV_250_Aggr1km_RefSB')
                uct0_250  = f.select('EV_250_Aggr1km_RefSB_Uncert_Indexes')
                raw0_500  = f.select('EV_500_Aggr1km_RefSB')
                uct0_500  = f.select('EV_500_Aggr1km_RefSB_Uncert_Indexes')
                
                # save offsets and scaling factors (from both visible and near infrared bands)
                rad_off = raw0_250.attributes()['radiance_offsets'] + raw0_500.attributes()['radiance_offsets']
                rad_sca = raw0_250.attributes()['radiance_scales'] + raw0_500.attributes()['radiance_scales']
                ref_off = raw0_250.attributes()['reflectance_offsets'] + raw0_500.attributes()['reflectance_offsets']
                ref_sca = raw0_250.attributes()['reflectance_scales'] + raw0_500.attributes()['reflectance_scales']
                cnt_off = raw0_250.attributes()['corrected_counts_offsets'] + raw0_500.attributes()['corrected_counts_offsets']
                cnt_sca = raw0_250.attributes()['corrected_counts_scales'] + raw0_500.attributes()['corrected_counts_scales']
                uct_spc = uct0_250.attributes()['specified_uncertainty'] + uct0_500.attributes()['specified_uncertainty']
                uct_sca = uct0_250.attributes()['scaling_factor'] + uct0_500.attributes()['scaling_factor']
                
                # combine visible and near infrared bands
                raw0      = np.vstack([raw0_250, raw0_500])
                uct0      = np.vstack([uct0_250, uct0_500])
                wvl       = np.array([650.0, 860.0, 470.0, 555.0, 1240.0, 1640.0, 2130.0])
                do_region = False
                lon       = self.f03.data['lon']['data']
                lat       = self.f03.data['lat']['data']
                logic     = self.f03.logic[find_fname_match(fname, self.f03.logic.keys())]['1km']
            else:
                sys.exit('Error   [modis_l1b]: \'resolution=%f\' has not been implemented without geolocation file being specified.' % self.resolution)

        else:
            sys.exit('Error   [modis_l1b]: \'resolution=%f\' has not been implemented.' % self.resolution)

        
        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #/----------------------------------------------------------------------------\#
        
        if do_region: # applied only for QKM (250m) or HKM (500m) products
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

            logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
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
        # -------------------------------------------------------------------------------------------------


        # Calculate 1. radiance, 2. reflectance, 3. corrected counts from the raw data
        #/----------------------------------------------------------------------------\#
        raw = raw0[:][:, logic]
        rad = np.zeros(raw.shape, dtype=np.float64)
        ref = np.zeros(raw.shape, dtype=np.float64)
        cnt = np.zeros(raw.shape, dtype=np.float64)

        # Calculate uncertainty
        uct     = uct0[:][:, logic]
        uct_pct = np.zeros(uct.shape, dtype=np.float64)

        for i in range(raw.shape[0]):

            rad[i, ...]       = (raw[i, ...] - rad_off[i]) * rad_sca[i]
            rad[i, ...]      /= 1000.0 # convert to W/m^2/nm/sr
            ref[i, ...]       = (raw[i, ...] - ref_off[i]) * ref_sca[i]
            cnt[i, ...]       = (raw[i, ...] - cnt_off[i]) * cnt_sca[i]
            uct_pct[i, ...]   = uct_spc[i] * np.exp(uct[i] / uct_sca[i]) # convert to percentage

        f.end()
        # -------------------------------------------------------------------------------------------------



        if hasattr(self, 'data'):
            if do_region:
                self.data['lon'] = dict(name='Longitude'           , data=np.hstack((self.data['lon']['data'], lon)), units='degrees')
                self.data['lat'] = dict(name='Latitude'            , data=np.hstack((self.data['lat']['data'], lat)), units='degrees')

            self.data['rad'] = dict(name='Radiance'                , data=np.hstack((self.data['rad']['data'], rad)), units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance (x cos(SZA))', data=np.hstack((self.data['ref']['data'], ref)), units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts'        , data=np.hstack((self.data['cnt']['data'], cnt)), units='N/A')
            self.data['uct'] = dict(name='Uncertainty Percentage'  , data=np.hstack((self.data['uct']['data'], uct_pct)), units='N/A')

        else:

            self.data = {}
            self.data['lon'] = dict(name='Longitude'               , data=lon, units='degrees')
            self.data['lat'] = dict(name='Latitude'                , data=lat, units='degrees')
            self.data['wvl'] = dict(name='Wavelength'              , data=wvl, units='nm')
            self.data['rad'] = dict(name='Radiance'                , data=rad, units='W/m^2/nm/sr')
            self.data['ref'] = dict(name='Reflectance (x cos(SZA))', data=ref, units='N/A')
            self.data['cnt'] = dict(name='Corrected Counts'        , data=cnt, units='N/A')
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
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['cot']
                ['cer']
    """


    ID = 'MODIS Level 2 Cloud Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 cop_flag  = '',    \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname, cop_flag=cop_flag)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)


    def read(self, fname, cop_flag=''):

        """
        Read cloud optical properties

        self.data
            ['lon']
            ['lat']
            ['cot']
            ['cer']
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
            vname_cot_err = 'Cloud_Optical_Thickness_Uncertainty'
            vname_cer_err = 'Cloud_Effective_Radius_Uncertainty'
        else:
            vname_ctp     = 'Cloud_Phase_Optical_Properties'
            vname_cot     = 'Cloud_Optical_Thickness_%s' % cop_flag
            vname_cer     = 'Cloud_Effective_Radius_%s'  % cop_flag
            vname_cot_err = 'Cloud_Optical_Thickness_Uncertainty_%s' % cop_flag
            vname_cer_err = 'Cloud_Effective_Radius_Uncertainty_%s' % cop_flag

        f     = SD(fname, SDC.READ)

        # lon lat
        lat0       = f.select('Latitude')
        lon0       = f.select('Longitude')

        ctp0       = f.select(vname_ctp)
        cot0       = f.select(vname_cot)
        cer0       = f.select(vname_cer)
        cot1       = f.select('%s_PCL' % vname_cot)
        cer1       = f.select('%s_PCL' % vname_cer)
        cot_err0   = f.select(vname_cot_err)
        cer_err0   = f.select(vname_cer_err)


        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        #/----------------------------------------------------------------------------\#
        lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)

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

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]

        lon_5km   = lon0[:]
        lat_5km   = lat0[:]
        logic_5km = (lon_5km>=lon_range[0]) & (lon_5km<=lon_range[1]) & (lat_5km>=lat_range[0]) & (lat_5km<=lat_range[1])
        lon_5km   = lon_5km[logic_5km]
        lat_5km   = lat_5km[logic_5km]
        # -------------------------------------------------------------------------------------------------


        # Calculate 1. cot, 2. cer, 3. ctp
        #/--------------------------------\#
        ctp0_data     = get_data_h4(ctp0)
        cot0_data     = get_data_h4(cot0)
        cer0_data     = get_data_h4(cer0)
        cot1_data     = get_data_h4(cot1)
        cer1_data     = get_data_h4(cer1)
        cot_err0_data = get_data_h4(cot_err0)
        cer_err0_data = get_data_h4(cer_err0)

        ctp     = ctp0_data.copy()
        cot     = cot0_data.copy()
        cer     = cer0_data.copy()
        cot_err = cot_err0_data.copy()
        cer_err = cer_err0_data.copy()

        pcl_tag = np.zeros_like(cot, dtype=np.int32)

        logic_pcl = ((cot0_data<0.0) | (cer0_data<=0.0)) & ((cot1_data>=0.0) & (cer1_data>0.0))
        pcl_tag[logic_pcl] = 1
        cot[logic_pcl] = cot1_data[logic_pcl]
        cer[logic_pcl] = cer1_data[logic_pcl]

        logic_invalid = (cot<0.0) | (cer<=0.0)
        cot[logic_invalid]     = 0.0
        cer[logic_invalid]     = 1.0
        cot_err[logic_invalid] = 0.0
        cer_err[logic_invalid] = 0.0

        f.end()
        # -------------------------------------------------------------------------------------------------

        ctp = ctp[logic]
        cot = cot[logic]
        cer = cer[logic]
        cot_err = cot_err[logic]
        cer_err = cer_err[logic]
        pcl_tag = pcl_tag[logic]

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic, '5km':logic_5km}

            self.data['lon']   = dict(name='Longitude'                 , data=np.hstack((self.data['lon']['data'], lon    )), units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=np.hstack((self.data['lat']['data'], lat    )), units='degrees')
            self.data['ctp']   = dict(name='Cloud thermodynamic phase' , data=np.hstack((self.data['ctp']['data'], ctp    )), units='N/A')
            self.data['cot']   = dict(name='Cloud optical thickness'   , data=np.hstack((self.data['cot']['data'], cot    )), units='N/A')
            self.data['cer']   = dict(name='Cloud effective radius'    , data=np.hstack((self.data['cer']['data'], cer    )), units='micron')
            self.data['cot_err']   = dict(name='Cloud optical thickness uncertainty', data=np.hstack((self.data['cot_err']['data'], cot*cot_err/100.0)), units='N/A')
            self.data['cer_err']   = dict(name='Cloud effective radius uncertainty' , data=np.hstack((self.data['cer_err']['data'], cer*cer_err/100.0)), units='micron')
            self.data['pcl']       = dict(name='PCL tag (1:PCL, 0:Cloudy)' , data=np.hstack((self.data['pcl']['data'], pcl_tag)), units='N/A')
            self.data['lon_5km']   = dict(name='Longitude at 5km', data=np.hstack((self.data['lon_5km']['data'], lon_5km)), units='degrees')
            self.data['lat_5km']   = dict(name='Latitude at 5km' , data=np.hstack((self.data['lat_5km']['data'], lat_5km)), units='degrees')

        else:
            self.logic = {}
            self.logic[fname] = {'1km':logic, '5km':logic_5km}

            self.data  = {}
            self.data['lon']   = dict(name='Longitude'                 , data=lon    , units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=lat    , units='degrees')
            self.data['ctp']   = dict(name='Cloud thermodynamic phase' , data=ctp    , units='N/A')
            self.data['cot']   = dict(name='Cloud optical thickness'   , data=cot    , units='N/A')
            self.data['cer']   = dict(name='Cloud effective radius'    , data=cer    , units='micron')
            self.data['cot_err']   = dict(name='Cloud optical thickness uncertainty' , data=cot*cot_err/100.0, units='N/A')
            self.data['cer_err']   = dict(name='Cloud effective radius uncertainty'  , data=cer*cer_err/100.0, units='micron')
            self.data['pcl']       = dict(name='PCL tag (1:PCL, 0:Cloudy)' , data=pcl_tag, units='N/A')
            self.data['lon_5km']   = dict(name='Longitude at 5km', data=lon_5km, units='degrees')
            self.data['lat_5km']   = dict(name='Latitude at 5km' , data=lat_5km, units='degrees')


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
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        overwrite=: keyword argument, default=False, whether to overwrite or not
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
                ['sunglint_flag']   => 0: not in sunglint path, 1: in sunglint path
                ['snow_ice_flag']   => 0: no snow/ice in background, 1: possible snow/ice in background
                ['land_water_cat']  => 0: water, 1: coastal, 2: desert, 3: land
                ['lon_5km']
                ['lat_5km']

    References: (Product Page) https://atmosphere-imager.gsfc.nasa.gov/products/cloud-mask
                (ATBD)         https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/MOD35_ATBD_Collection6_1.pdf
                (User Guide)   http://cimss.ssec.wisc.edu/modis/CMUSERSGUIDE.PDF
    """


    ID = 'MODIS Level 2 Cloud Mask Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:
            self.read(fname)


    def extract_data(self, data):
        """
        Extract cloud mask (in byte format) flags and categories
        """
        if data.dtype != 'uint8':
            data = data.astype('uint8')

        data = np.unpackbits(data, bitorder='big', axis=1) # convert to binary

        # extract flags and categories (*_cat) bit by bit
        land_water_cat  = 2 * data[:, 0] + 1 * data[:, 1] # convert to a value between 0 and 3
        snow_ice_flag   = data[:, 2]
        sunglint_flag   = data[:, 3]
        day_night_flag  = data[:, 4]
        fov_qa_cat      = 2 * data[:, 5] + 1 * data[:, 6] # convert to a value between 0 and 3
        cloud_mask_flag = data[:, 7]
        return cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat


    def quality_assurance(self, data):
        """
        Extract cloud mask QA data to determine confidence
        """
        if data.dtype != 'uint8':
            data = data.astype('uint8')

        # process qa flags
        data = np.unpackbits(data, bitorder='big', axis=1)
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

        self.logic
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

        #/----------------------------------------------------------------------------\#
        lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)

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

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]

        lon_5km   = lon0[:]
        lat_5km   = lat0[:]
        logic_5km = (lon_5km>=lon_range[0]) & (lon_5km<=lon_range[1]) & (lat_5km>=lat_range[0]) & (lat_5km<=lat_range[1])
        lon_5km   = lon_5km[logic_5km]
        lat_5km   = lat_5km[logic_5km]


        # -------------------------------------------------------------------------------------------------

        # Get cloud mask and flag fields
        #/-----------------------------\#
        cm0_data = get_data_h4(cld_msk0)
        qa0_data = get_data_h4(qa0)
        cm = cm0_data.copy()
        qa = qa0_data.copy()

        cm = cm[0, :, :] # read only the first of 6 bytes; rest will be supported in the future if needed
        cm = np.array(cm[logic], dtype='uint8')
        cm = cm.reshape((cm.size, 1))
        cloud_mask_flag, day_night_flag, sunglint_flag, snow_ice_flag, land_water_cat, fov_qa_cat = self.extract_data(cm)


        qa = qa[:, :, 0] # read only the first byte for confidence (indexed differently from cloud mask SDS)
        qa = np.array(qa[logic], dtype='uint8')
        qa = qa.reshape((qa.size, 1))
        use_qa, confidence_qa = self.quality_assurance(qa)

        f.end()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.logic[fname] = {'1km':logic, '5km':logic_5km}

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
            self.logic[fname] = {'1km':logic, '5km':logic_5km}

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




class modis_03:

    """
    Read MODIS 03 geolocation data

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
        #/----------------------------------------------------------------------------\#
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

            lon_range = [self.extent[0] - 0.01, self.extent[1] + 0.01]
            lat_range = [self.extent[2] - 0.01, self.extent[3] + 0.01]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        # -------------------------------------------------------------------------------------------------


        # Calculate 1. sza, 2. saa, 3. vza, 4. vaa
        #/---------------------------------------\#
        sza0_data = get_data_h4(sza0)
        saa0_data = get_data_h4(saa0)
        vza0_data = get_data_h4(vza0)
        vaa0_data = get_data_h4(vaa0)

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


    ID = 'MODIS 04 Geolocation Product'


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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        # -------------------------------------------------------------------------------------------------


        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Deep_Blue_AOD_550_land              = get_data_h4(Deep_Blue_AOD_550_land_0)[logic]
        Deep_Blue_Angstrom_Exponent_land    = get_data_h4(Deep_Blue_Angstrom_Exponent_land_0)[logic]
        Deep_Blue_Aerosol_type_land         = get_data_h4(Deep_Blue_Aerosol_type_land_0)[logic]
        Deep_Blue_Aerosol_cloud_frac_land   = get_data_h4(Deep_Blue_Aerosol_cloud_frac_land_0)[logic]
        Deep_Blue_SSA_land                  = get_data_h4(Deep_Blue_SSA_land_0)[logic]

        f.end()
        # -------------------------------------------------------------------------------------------------

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
        ref = np.zeros((Nchan, logic.sum()), dtype=np.float64)
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
        bsky_alb = np.zeros((Nchan, logic.sum()), dtype=np.float64)
        wsky_alb = np.zeros((Nchan, logic.sum()), dtype=np.float64)

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
        if verb: print(xx.shape,yy.shape,img.shape)

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

#\-----------------------------------------------------------------------------/





# Useful functions
#/-----------------------------------------------------------------------------\

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

    f_x = interpolate.interp2d(XX_in, YY_in, xy_in[:, :, 0], kind='linear', fill_value=None)
    f_y = interpolate.interp2d(XX_in, YY_in, xy_in[:, :, 1], kind='linear', fill_value=None)

    lonlat = proj_lonlat.transform_points(proj_xy, f_x(XX, YY), f_y(XX, YY))[:, :, [0, 1]]

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
    except:
        exit('Error   [get_filename_tag]: cannot find environment variables \'EARTHDATA_USERNAME\' and \'EARTHDATA_PASSWORD\'.')

    try:
        with requests.Session() as session:
            session.auth = (username, password)
            r1     = session.request('get', fname_server)
            r      = session.get(r1.url, auth=(username, password))
            if r.ok:
                content = r.content.decode('utf-8')
    except:
        exit('Error   [get_filename_tag]: cannot access \'%s\'.' % fname_server)

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
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()
        # ---------------------------------------------------------------------

        # find the precise center point of MODIS granule
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        # ---------------------------------------------------------------------

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
    # ---------------------------------------------------------------------

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

#\-----------------------------------------------------------------------------/


if __name__=='__main__':

    pass
