import os
import sys
import datetime
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from er3t.util import check_equal



__all__ = ['oco2_rad_nadir', 'oco2_std', 'oco2_met', 'get_fnames_from_web', 'get_dtime_from_xml', 'download_oco2_https']



def convert_photon_unit(data_photon, wavelength, scale_factor=2.0):

    c = 299792458.0
    h = 6.62607015e-34
    wavelength = wavelength * 1e-9
    data = data_photon/1000.0*c*h/wavelength*scale_factor

    return data



class oco2_rad_nadir:

    def __init__(self, sat):

        self.fname_l1b = sat.fnames['oco_l1b'][0]
        self.fname_std = sat.fnames['oco_std'][0]

        self.extent = sat.extent

        # =================================================================================
        self.cal_wvl()
        # after this, the following three functions will be created
        # Input: index, range from 0 to 7, e.g., 0, 1, 2, ..., 7
        # self.get_wvl_o2_a(index)
        # self.get_wvl_co2_weak(index)
        # self.get_wvl_co2_strong(index)
        # =================================================================================

        # =================================================================================
        self.get_index(self.extent)
        # after this, the following attributes will be created
        # self.index_s: starting index
        # self.index_e: ending index
        # =================================================================================

        # =================================================================================
        self.overlap(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.logic_l1b
        # self.lon_l1b
        # self.lat_l1b
        # =================================================================================

        # =================================================================================
        self.get_data(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.rad_o2_a
        # self.rad_co2_weak
        # self.rad_co2_strong
        # =================================================================================

    def cal_wvl(self, Nchan=1016):

        """
        Oxygen A band: centered at 765 nm
        Weak CO2 band: centered at 1610 nm
        Strong CO2 band: centered at 2060 nm
        """

        f = h5py.File(self.fname_l1b, 'r')
        wvl_coef = f['InstrumentHeader/dispersion_coef_samp'][...]
        f.close()

        Nspec, Nfoot, Ncoef = wvl_coef.shape

        wvl_o2_a       = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_weak   = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_strong = np.zeros((Nfoot, Nchan), dtype=np.float64)

        chan = np.arange(1, Nchan+1)
        for i in range(Nfoot):
            for j in range(Ncoef):
                wvl_o2_a[i, :]       += wvl_coef[0, i, j]*chan**j
                wvl_co2_weak[i, :]   += wvl_coef[1, i, j]*chan**j
                wvl_co2_strong[i, :] += wvl_coef[2, i, j]*chan**j

        wvl_o2_a       *= 1000.0
        wvl_co2_weak   *= 1000.0
        wvl_co2_strong *= 1000.0

        self.get_wvl_o2_a       = lambda index: wvl_o2_a[index, :]
        self.get_wvl_co2_weak   = lambda index: wvl_co2_weak[index, :]
        self.get_wvl_co2_strong = lambda index: wvl_co2_strong[index, :]

    def get_index(self, extent):

        if extent is None:
            self.index_s = 0
            self.index_e = None
        else:
            f = h5py.File(self.fname_l1b, 'r')
            lon_l1b     = f['SoundingGeometry/sounding_longitude'][...]
            lat_l1b     = f['SoundingGeometry/sounding_latitude'][...]

            logic = (lon_l1b>=extent[0]) & (lon_l1b<=extent[1]) & (lat_l1b>=extent[2]) & (lat_l1b<=extent[3])
            indices = np.where(np.sum(logic, axis=1)>0)[0]
            self.index_s = indices[0]
            self.index_e = indices[-1]

    def overlap(self, index_s=0, index_e=None, lat0=0.0, lon0=0.0):

        f       = h5py.File(self.fname_l1b, 'r')
        if index_e is None:
            lon_l1b = f['SoundingGeometry/sounding_longitude'][...][index_s:, ...]
            lat_l1b = f['SoundingGeometry/sounding_latitude'][...][index_s:, ...]
            snd_id  = f['SoundingGeometry/sounding_id'][...][index_s:, ...]
        else:
            lon_l1b     = f['SoundingGeometry/sounding_longitude'][...][index_s:index_e, ...]
            lat_l1b     = f['SoundingGeometry/sounding_latitude'][...][index_s:index_e, ...]
            snd_id_l1b  = f['SoundingGeometry/sounding_id'][...][index_s:index_e, ...]
        f.close()

        shape    = lon_l1b.shape
        lon_l1b  = lon_l1b
        lat_l1b  = lat_l1b

        f       = h5py.File(self.fname_std, 'r')
        lon_std = f['RetrievalGeometry/retrieval_longitude'][...]
        lat_std = f['RetrievalGeometry/retrieval_latitude'][...]
        xco2_std= f['RetrievalResults/xco2'][...]
        snd_id_std = f['RetrievalHeader/sounding_id'][...]
        sfc_pres_std = f['RetrievalResults/surface_pressure_fph'][...]
        f.close()

        self.logic_l1b = np.in1d(snd_id_l1b, snd_id_std).reshape(shape)

        self.lon_l1b   = lon_l1b
        self.lat_l1b   = lat_l1b
        self.snd_id    = snd_id_l1b

        xco2      = np.zeros_like(self.lon_l1b); xco2[...] = np.nan
        sfc_pres  = np.zeros_like(self.lon_l1b); sfc_pres[...] = np.nan

        for i in range(xco2.shape[0]):
            for j in range(xco2.shape[1]):
                logic = (snd_id_std==snd_id_l1b[i, j])
                if logic.sum() == 1:
                    xco2[i, j] = xco2_std[logic]
                    sfc_pres[i, j] = sfc_pres_std[logic]
                elif logic.sum() > 1:
                    sys.exit('Error   [oco_rad_nadir]: More than one point is found.')

        self.xco2      = xco2
        self.sfc_pres  = sfc_pres

    def get_data(self, index_s=0, index_e=None):

        f       = h5py.File(self.fname_l1b, 'r')
        if index_e is None:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:, ...]
            self.vza            = f['SoundingGeometry/sounding_zenith'][...][index_s:, ...]
            self.vaa            = f['SoundingGeometry/sounding_azimuth'][...][index_s:, ...]
        else:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:index_e, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:index_e, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:index_e, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:index_e, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:index_e, ...]
            self.vza            = f['SoundingGeometry/sounding_zenith'][...][index_s:index_e, ...]
            self.vaa            = f['SoundingGeometry/sounding_azimuth'][...][index_s:index_e, ...]

        for i in range(8):
            self.rad_o2_a[:, i, :]       = convert_photon_unit(self.rad_o2_a[:, i, :]      , self.get_wvl_o2_a(i))
            self.rad_co2_weak[:, i, :]   = convert_photon_unit(self.rad_co2_weak[:, i, :]  , self.get_wvl_co2_weak(i))
            self.rad_co2_strong[:, i, :] = convert_photon_unit(self.rad_co2_strong[:, i, :], self.get_wvl_co2_strong(i))
        f.close()



class oco2_std:

    """
    Read OCO2 standard file into an object `oco2_std`

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original HDF4 files
        overwrite=  : keyword argument, default=False, whether to overwrite or not
        extent=     : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        verbose=    : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['xco2']
    """


    ID = 'OCO2 Standard CO2 retrievals'


    def __init__(self, \
                 fnames    = None, \
                 vnames    = [],
                 extent    = None, \
                 overwrite = False,\
                 quiet     = True, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the hdf files
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag
        self.quiet      = quiet       # quiet tag

        for fname in self.fnames:

            self.read(fname)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)


    def read(self, fname):

        """
        Read CO2 retrievals from the OCO2 standard file
        self.data
            ['lon']
            ['lat']
            ['xco2']
        """

        try:
            import h5py
        except ImportError:
            msg = 'Warning [oco2_std]: To use \'oco2_std\', \'h5py\' needs to be installed.'
            raise ImportError(msg)

        f     = h5py.File(fname, 'r')

        # L1bScSoundingReference/retrieval_index
        # AlbedoResults/albedo_o2_fph
        # RetrievalGeometry/retrieval_latitude
        # RetrievalGeometry/retrieval_longitude
        # RetrievalResults/xco2

        # lon lat
        lat       = f['RetrievalGeometry/retrieval_latitude'][...]
        lon       = f['RetrievalGeometry/retrieval_longitude'][...]
        xco2      = f['RetrievalResults/xco2'][...]

        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        xco2      = xco2[logic]

        f.close()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.logic[fname] = {'all':logic}

            self.data['lon']  = dict(name='Longitude'        , data=np.hstack((self.data['lon']['data'] , lon)), units='degrees')
            self.data['lat']  = dict(name='Latitude'         , data=np.hstack((self.data['lat']['data'] , lat)), units='degrees')
            self.data['xco2'] = dict(name='CO2 concentration', data=np.hstack((self.data['xco2']['data'], xco2)), units='N/A')

        else:

            self.logic = {}
            self.logic[fname] = {'all':logic}

            self.data = {}
            self.data['lon']  = dict(name='Longitude'        , data=lon , units='degrees')
            self.data['lat']  = dict(name='Latitude'         , data=lat , units='degrees')
            self.data['xco2'] = dict(name='CO2 concentration', data=xco2, units='N/A')


    def read_vars(self, fname, vnames=[], resolution='5km'):

        try:
            import h5py
        except ImportError:
            msg = 'Warning [oco2_std]: To use \'oco2_std\', \'h5py\' needs to be installed.'
            raise ImportError(msg)

        f     = h5py.File(fname, 'r')

        logic = self.logic[fname]['all']

        for vname in vnames:

            data  = f[vname][...][logic]
            vname_new = vname.lower().split('/')[-1]
            if vname_new in self.data.keys():
                self.data[vname_new] = dict(name=vname, data=np.hstack((self.data[vname_new]['data'], data)))
            else:
                self.data[vname_new] = dict(name=vname, data=data)

        f.close()


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()

        if not self.quiet:
            print('Message [oco2_std]: File \'%s\' is created.' % fname)



class oco2_met:

    """
    Read OCO2 meteorological file into an object `oco2_met`

    Input:
        fnames=     : keyword argument, default=None, Python list of the file path of the original HDF4 files
        overwrite=  : keyword argument, default=False, whether to overwrite or not
        extent=     : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        verbose=    : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['xco2']
    """


    ID = 'OCO2 Meteorological Data'


    def __init__(self, \
                 fnames    = None, \
                 extent    = None, \
                 overwrite = False,\
                 quiet     = True, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the hdf files
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag
        self.quiet      = quiet       # quiet tag

        for fname in self.fnames:
            self.read(fname)


    def read(self, fname):

        """
        Read CO2 retrievals from the OCO2 standard file
        self.data
            ['lon']
            ['lat']
            ['xco2']
        """

        try:
            import h5py
        except ImportError:
            msg = 'Warning [oco2_std]: To use \'oco2_std\', \'h5py\' needs to be installed.'
            raise ImportError(msg)

        f     = h5py.File(fname, 'r')

        # L1bScSoundingReference/retrieval_index
        # AlbedoResults/albedo_o2_fph
        # RetrievalGeometry/retrieval_latitude
        # RetrievalGeometry/retrieval_longitude
        # RetrievalResults/xco2

        # lon lat
        lat       = f['RetrievalGeometry/retrieval_latitude'][...]
        lon       = f['RetrievalGeometry/retrieval_longitude'][...]
        xco2      = f['RetrievalResults/xco2'][...]

        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        xco2      = xco2[logic]

        f.close()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.data['lon']  = dict(name='Longitude'        , data=np.hstack((self.data['lon']['data'] , lon)), units='degrees')
            self.data['lat']  = dict(name='Latitude'         , data=np.hstack((self.data['lat']['data'] , lat)), units='degrees')
            self.data['xco2'] = dict(name='CO2 concentration', data=np.hstack((self.data['xco2']['data'], xco2)), units='N/A')

        else:

            self.data = {}
            self.data['lon']  = dict(name='Longitude'        , data=lon , units='degrees')
            self.data['lat']  = dict(name='Latitude'         , data=lat , units='degrees')
            self.data['xco2'] = dict(name='CO2 concentration', data=xco2, units='N/A')


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')
        for key in self.data.keys():
            f[key] = self.data[key]['data']
        f.close()

        if not self.quiet:
            print('Message [oco2_std]: File \'%s\' is created.' % fname)



def get_fnames_from_web(website, extension):

    """
    Read the file names of the OCO2 granules from the web link

    Input:
        website: string, web link, e.g., https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Met.8r/2015/010
        extension: string, file extension, e.g., 'h5'

    Output:
        fnames: Python list of file names
    """

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        msg = 'Warning [get_fnames]: To use \'get_fnames\', \'beautifulsoup4\' needs to be installed.'
        raise ImportError(msg)

    try:
        web = urllib.request.urlopen(website)
    except urllib.error.HTTPError:
        sys.exit('Error   [get_fnames_from_web]: \'%s\' does not exist.' % website)
    content = web.read()
    bs = BeautifulSoup(content, 'html.parser')

    fnames_web = bs.findAll('a')
    fnames     = []
    for fname_web in fnames_web:
        if (fname_web['href'] not in fnames) and (fname_web['href'][-len(extension):]==extension):
            fnames.append(fname_web['href'])

    return fnames



def get_dtime_from_xml(fname_link):

    """
    Read the beginning and ending date time of the OCO2 granule from the data description file (.xml)

    Input:
        fname_link: link address of the xml file

    Output:
        dtime_begin: Python datetime.datetime object, beginning time of the granule
        dtime_end: Python datetime.datetime object, ending time of the granule
    """

    f = urllib.request.urlopen(fname_link)
    content = f.read()

    root = ET.fromstring(content)

    if ('date' in root[2][3].tag.lower()) and ('time' in root[2][2].tag.lower()) and \
       ('date' in root[2][1].tag.lower()) and ('time' in root[2][0].tag.lower()):
        dtime_begin_str = root[2][3].text + ' ' + root[2][2].text[:-1]
        dtime_end_str   = root[2][1].text + ' ' + root[2][0].text[:-1]
    else:
        sys.exit('Error   [get_dtime_from_xml]: Cannot locate the time information.')


    seconds_begin_ori = dtime_begin_str.split(':')[-1]
    seconds_begin     = float(seconds_begin_ori)
    seconds_end_ori   = dtime_end_str.split(':')[-1]
    seconds_end       = float(seconds_end_ori)

    dtime_begin_str = dtime_begin_str.replace(':'+seconds_begin_ori, '')
    dtime_end_str   = dtime_end_str.replace(':'+seconds_end_ori, '')

    dtime_begin = datetime.datetime.strptime(dtime_begin_str, '%Y-%m-%d %H:%M') + datetime.timedelta(seconds=seconds_begin)
    dtime_end   = datetime.datetime.strptime(dtime_end_str  , '%Y-%m-%d %H:%M') + datetime.timedelta(seconds=seconds_end)

    return dtime_begin, dtime_end



def download_oco2_https(
             dtime,
             dataset_tag,
             fnames=None,
             server='https://oco2.gesdisc.eosdis.nasa.gov',
             fdir_prefix='/data/OCO2_DATA',
             fdir_out='data',
             data_format=None,
             run=True,
             quiet=False,
             verbose=False):

    """
    Input:
        dtime: Python datetime object
        dataset_tag: string, e.g. 'OCO2_L2_Standard.8r'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        fdir_out=: string, output data directory
        data_format=None: e.g., 'h5'
        run=: boolen type, if true, the command will only be displayed but not run
        quiet=: Boolen type, quiet tag
        verbose=: Boolen type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded OCO2 file paths
    """

    fname_login = '~/.netrc'
    if not os.path.exists(os.path.expanduser(fname_login)):
        sys.exit('Error [download_oco2_https]: Please follow the instructions at \nhttps://disc.gsfc.nasa.gov/data-access\nto register a login account and create a \'~/.netrc\' file.')

    fname_cookies = '~/.urs_cookies'
    if not os.path.exists(os.path.expanduser(fname_cookies)):
        print('Message [download_modis_https]: Creating ~/.urs_cookies ...')
        os.system('touch ~/.urs_cookies')

    if shutil.which('curl'):
        command_line_tool = 'curl'
    elif shutil.which('wget'):
        command_line_tool = 'wget'
    else:
        sys.exit('Error [download_oco2_https]: \'download_oco2_https\' needs \'curl\' or \'wget\' to be installed.')

    year_str = str(dtime.timetuple().tm_year).zfill(4)
    doy_str  = str(dtime.timetuple().tm_yday).zfill(3)

    if dataset_tag in ['OCO2_L2_Met.10', 'OCO2_L2_Met.10r', 'OCO2_L2_Standard.10', 'OCO2_L2_Standard.10r',
                       'OCO2_L1B_Science.10', 'OCO2_L1B_Science.10r', 'OCO2_L1B_Calibration.10', 'OCO2_L1B_Calibration.10r', 
                       'OCO2_L2_CO2Prior.10r', 'OCO2_L2_CO2Prior.10', 'OCO2_L2_IMAPDOAS.10r', 'OCO2_L2_IMAPDOAS.10',
                       'OCO2_L2_Diagnostic.10r', 'OCO2_L2_Diagnostic.10']:
        fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)
    elif dataset_tag in ['OCO2_L2_Lite_FP.9r', 'OCO2_L2_Lite_FP.10r', 'OCO2_L2_Lite_SIF.10r']:
        fdir_data = '%s/%s/%s' % (fdir_prefix, dataset_tag, year_str)
    else:
        sys.exit('Error   [download_oco2_https]: Do not support downloading \'%s\'.' % dataset_tag)

    fdir_server = server + fdir_data

    fnames_xml = get_fnames_from_web(fdir_server, 'xml')
    if len(fnames_xml) > 0:
        data_format = fnames_xml[0].split('.')[-2]
    else:
        sys.exit('Error   [download_oco2_https]: XML files are not available at %s.' % fdir_server)


    fnames_server = []

    if fnames is not None:

        for fname in fnames:
            fname_server = '%s/%s' % (fdir_server, fname)
            fnames_server.append(fname_server)

    else:

        fnames_dat  = get_fnames_from_web(fdir_server, data_format)
        Nfile      = len(fnames_dat)

        if not all([fnames_dat[i] in fnames_xml[i] for i in range(Nfile)]):
            sys.exit('Error   [download_oco2_https]: The description files [xml] do not match with data files.')

        for i in range(Nfile):
            dtime_s, dtime_e = get_dtime_from_xml('%s/%s' % (fdir_server, fnames_xml[i]))
            if (dtime>=dtime_s) & (dtime<=dtime_e):
                fname_server = '%s/%s' % (fdir_server, fnames_dat[i])
                fnames_server.append(fname_server)

    commands = []
    fnames_local = []
    for fname_server in fnames_server:
        filename     = os.path.basename(fname_server)
        fname_local  = '%s/%s' % (fdir_out, filename)
        fnames_local.append(fname_local)

        if command_line_tool == 'curl':
            command = 'mkdir -p %s && curl -n -c ~/.urs_cookies -b ~/.urs_cookies -L -C - \'%s\' -o \'%s\'' % (fdir_out, fname_server, fname_local)
        elif command_line_tool == 'wget':
            command = 'mkdir -p %s && wget -c "%s" --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -O %s' % (fdir_out, fname_server, fname_local)
        commands.append(command)

    if not run and len(commands)>0:

        if not quiet:
            print('Message [download_oco2_https]: The commands to run are:')
            for command in commands:
                print(command)
                print()

    else:

        for i, command in enumerate(commands):

            print('Message [download_oco2_https]: Downloading %s ...' % fnames_local[i])
            os.system(command)

            fname_local = fnames_local[i]

            if data_format == 'h5':

                try:
                    import h5py
                except ImportError:
                    msg = 'Warning [downlad_oco2_https]: To use \'download_oco2_https\', \'h5py\' needs to be installed.'
                    raise ImportError(msg)

                f = h5py.File(fname_local, 'r')
                f.close()
                print('Message [download_oco2_https]: \'%s\' has been downloaded.\n' % fname_local)

            elif data_format == 'nc':

                try:
                    import netCDF4 as nc4
                except ImportError:
                    msg = 'Warning [downlad_oco2_https]: To use \'download_oco2_https\', \'netCDF4\' needs to be installed.'
                    raise ImportError(msg)

                f = nc4.Dataset(fname_local, 'r')
                f.close()
                print('Message [download_oco2_https]: \'%s\' has been downloaded.\n' % fname_local)

            else:

                print('Warning [download_oco2_https]: Do not support check for \'%s\'. Do not know whether \'%s\' has been successfully downloaded.\n' % (data_format, fname_local))

    return fnames_local



if __name__ == '__main__':

    pass
