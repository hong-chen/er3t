import os
import sys
import shutil
import datetime
import time
import requests
import urllib.request
from io import StringIO
from er3t.util.util import get_doy_tag
import numpy as np
import warnings

import er3t


__all__ = ['get_satfile_tag', 'get_nrt_satfile_tag', \
           'download_laads_https', 'download_lance_https',\
           'download_worldview_rgb', 'download_oco2_https']


def get_satfile_tag(
             date,
             lon,
             lat,
             satellite='aqua',
             instrument='modis',
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             local='./',
             verbose=False):

    """
    Get filename tag/overpass information for standard products.
    Currently supported satellites/instruments are:
    Aqua/MODIS, Terra/MODIS, SNPP/VIIRS, NOAA-20/VIIRS.

    Input:
        date: Python datetime.datetime object
        lon : longitude of, e.g. flight track
        lat : latitude of, e.g. flight track
        satellite=: default "aqua", can also change to "terra", 'snpp', 'noaa20'
        instrument=: default "modis", can also change to "viirs"
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        verbose=: Boolen type, verbose tag
    output:
        filename_tags: Python list of file name tags
    """

    # check cartopy and matplotlib
    #/----------------------------------------------------------------------------\#
    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = '\nError [get_satfile_tag]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    try:
        import matplotlib.path as mpl_path
    except ImportError:
        msg = '\nError [get_satfile_tag]: Please install <matplotlib> to proceed.'
        raise ImportError(msg)
    #\----------------------------------------------------------------------------/#

    from er3t.common import fdir_data_tmp

    # check satellite and instrument
    #/----------------------------------------------------------------------------\#
    if instrument.lower() == 'modis' and (satellite.lower() in ['aqua', 'terra']):
        instrument = instrument.upper()
        satellite  = satellite.lower().title()
    elif instrument.lower() == 'viirs' and (satellite.lower() in ['noaa20', 'snpp']):
        instrument = instrument.upper()
        satellite  = satellite.upper()
    else:
        msg = 'Error [get_satfile_tag]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)
    #\----------------------------------------------------------------------------/#


    # check login
    #/----------------------------------------------------------------------------\#
    try:
        username = os.environ['EARTHDATA_USERNAME']
        password = os.environ['EARTHDATA_PASSWORD']
    except:
        msg = '\nError [get_satfile_tag]: cannot find environment variables \'EARTHDATA_USERNAME\' and \'EARTHDATA_PASSWORD\'.'
        raise OSError(msg)
    #\----------------------------------------------------------------------------/#


    # generate satellite filename on LAADS DAAC server
    #/----------------------------------------------------------------------------\#
    vname  = '%s|%s' % (satellite, instrument)
    date_s = date.strftime('%Y-%m-%d')
    fnames_server = {
        'Aqua|MODIS'  : '%s/archive/geoMeta/61/AQUA/%4.4d/MYD03_%s.txt'           % (server, date.year, date_s),
        'Terra|MODIS' : '%s/archive/geoMeta/61/TERRA/%4.4d/MOD03_%s.txt'          % (server, date.year, date_s),
        'NOAA20|VIIRS': '%s/archive/geoMetaVIIRS/5200/NOAA-20/%4.4d/VJ103MOD_%s.txt' % (server, date.year, date_s),
        'SNPP|VIIRS'  : '%s/archive/geoMetaVIIRS/5110/NPP/%4.4d/VNP03MOD_%s.txt'   % (server, date.year, date_s),
        }
    fname_server = fnames_server[vname]
    #\----------------------------------------------------------------------------/#


    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    #/----------------------------------------------------------------------------\#
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]
    #\----------------------------------------------------------------------------/#


    # try to access the server
    #/----------------------------------------------------------------------------\#

    # try to get information from local
    # check two locations:
    #   1) <tmp-data/satfile> directory under er3t main directory
    #   2) current directory;
    #/--------------------------------------------------------------\#
    fdir_satfile_tmp = '%s/satfile' % fdir_data_tmp
    if not os.path.exists(fdir_satfile_tmp):
        os.makedirs(fdir_satfile_tmp)

    fname_local1 = os.path.abspath('%s/%s' % (fdir_satfile_tmp, os.path.basename(fname_server)))
    fname_local2 = os.path.abspath('%s/%s' % (local           , os.path.basename(fname_server)))

    if os.path.exists(fname_local1):
        with open(fname_local1, 'r') as f_:
            content = f_.read()

    elif os.path.exists(fname_local2):
        os.system('cp %s %s' % (fname_local2, fname_local1))
        with open(fname_local2, 'r') as f_:
            content = f_.read()
    #\--------------------------------------------------------------/#

    else:

        # get information from server
        #/--------------------------------------------------------------\#
        try:
            with requests.Session() as session:
                session.auth = (username, password)
                r1     = session.request('get', fname_server)
                r      = session.get(r1.url, auth=(username, password))
        except:
            msg = '\nError [get_satfile_tag]: cannot access <%s>.' % fname_server
            raise OSError(msg)

        if r.ok:
            content = r.content.decode('utf-8')
        else:
            msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.' % fname_server
            raise OSError(msg)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    # extract granule information from <content>
    # after the following session, granule information will be stored under <data>
    # data['GranuleID'].decode('UTF-8') to get the file name of MODIS granule
    # data['StartDateTime'].decode('UTF-8') to get the time stamp of MODIS granule
    # variable names can be found through
    # print(data.dtype.names)
    #/----------------------------------------------------------------------------\#

    if vname in ['Aqua|MODIS', 'Terra|MODIS']:
        dtype = ['|S41', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    elif vname in ['NOAA20|VIIRS', 'SNPP|VIIRS']:
        dtype = ['|S43', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    usecols = (0, 4, 9, 10, 11, 12, 13, 14, 15, 16)

    #\----------------------------------------------------------------------------/#
    # LAADS DAAC servers are known to cause some issues occasionally while
    # accessing the metadata. We will attempt to read the txt file online directly
    # on the server but as a backup, we will download the txt file locally and
    # access the data there
    #/----------------------------------------------------------------------------\#
    try:
        data  = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols)
    except ValueError:

        msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.\nAttempting to download the file to access the data...\n' % fname_server
        print(msg)

        try:
            token = os.environ['EARTHDATA_TOKEN']
        except KeyError:
            token = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'

            msg = '\nWarning [download_laads_https]: Please get a token by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.zshrc\'(Mac),\nexport EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\n'
            warnings.warn(msg)

        try:
            command = "wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 {} --header \"Authorization: Bearer {}\" -O {}".format(fname_server, token, fname_local1)
            os.system(command)
            with open(fname_local1, 'r') as f_:
                content = f_.read()
            data = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols)
        except ValueError:
            msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.\nThis is likely an issue with LAADS DAAC servers, please try downloading the files manually or try again later.\n' % fname_server
            raise OSError(msg)

    #\----------------------------------------------------------------------------/#
    # loop through all the "MODIS granules" constructed through four corner points
    # and find which granules contain the input data
    #/----------------------------------------------------------------------------\#
    Ndata = data.size
    filename_tags = []
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
        #/----------------------------------------------------------------------------\#
        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()
        #\----------------------------------------------------------------------------/#

        # find the precise center point of MODIS granule
        #/----------------------------------------------------------------------------\#
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        #\----------------------------------------------------------------------------/#

        proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
        LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

        modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
        pointsIn       = modis_granule.contains_points(LonLat_in)
        percentIn      = float(pointsIn.sum()) * 100.0 / float(pointsIn.size)
        # if pointsIn.sum()>0 and percentIn>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
        if pointsIn.sum()>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
            filename = data[i]['GranuleID'].decode('UTF-8')
            filename_tag = '.'.join(filename.split('.')[1:3])
            filename_tags.append(filename_tag)

    #\----------------------------------------------------------------------------/#
    return filename_tags



def get_nrt_satfile_tag(
             date,
             lon,
             lat,
             satellite='aqua',
             instrument='modis',
             server='https://nrt3.modaps.eosdis.nasa.gov/api/v2/content',
             local='./',
             verbose=False):

    """
    Get filename tag/overpass information for Near Real Time (NRT) products.
    Currently supported satellites/instruments are:
    Aqua/MODIS, Terra/MODIS, SNPP/VIIRS, NOAA-20/VIIRS.

    Input:
        date: Python datetime.datetime object
        lon : longitude of, e.g. flight track
        lat : latitude of, e.g. flight track
        satellite=: default "aqua", can also change to "terra", 'snpp', 'noaa20'
        instrument=: default "modis", can also change to "viirs"
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        verbose=: Boolen type, verbose tag
    output:
        filename_tags: Python list of file name tags
    """

    # check cartopy and matplotlib
    #/----------------------------------------------------------------------------\#
    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = '\nError [get_satfile_tag]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    try:
        import matplotlib.path as mpl_path
    except ImportError:
        msg = '\nError [get_satfile_tag]: Please install <matplotlib> to proceed.'
        raise ImportError(msg)
    #\----------------------------------------------------------------------------/#

    from er3t.common import fdir_data_tmp

    # check satellite and instrument
    #/----------------------------------------------------------------------------\#
    if instrument.lower() == 'modis' and (satellite.lower() in ['aqua', 'terra']):
        instrument = instrument.upper()
        satellite  = satellite.lower().title()
    elif instrument.lower() == 'viirs' and (satellite.lower() in ['noaa20', 'snpp']):
        instrument = instrument.upper()
        satellite  = satellite.upper()
    else:
        msg = 'Error [get_satfile_tag]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)
    #\----------------------------------------------------------------------------/#


    # check login
    #/----------------------------------------------------------------------------\#
    try:
        username = os.environ['EARTHDATA_USERNAME']
        password = os.environ['EARTHDATA_PASSWORD']
    except:
        msg = '\nError [get_satfile_tag]: cannot find environment variables \'EARTHDATA_USERNAME\' and \'EARTHDATA_PASSWORD\'.'
        raise OSError(msg)
    #\----------------------------------------------------------------------------/#


    # generate satellite filename on LAADS DAAC server
    #/----------------------------------------------------------------------------\#
    vname  = '%s|%s' % (satellite, instrument)
    date_s = date.strftime('%Y-%m-%d')
    fnames_server = {
        'Aqua|MODIS'  : '%s/archives/geoMetaMODIS/61/AQUA/%4.4d/MYD03_%s.txt'             % (server, date.year, date_s),
        'Terra|MODIS' : '%s/archives/geoMetaMODIS/61/TERRA/%4.4d/MOD03_%s.txt'            % (server, date.year, date_s),
        'NOAA20|VIIRS': '%s/archives/geoMetaVIIRS/5201/NOAA-20/%4.4d/VJ103MOD_NRT_%s.txt' % (server, date.year, date_s),
        'SNPP|VIIRS'  : '%s/archives/geoMetaVIIRS/5200/NPP/%4.4d/VNP03MOD_NRT_%s.txt'     % (server, date.year, date_s),
        }
    fname_server = fnames_server[vname]
    #\----------------------------------------------------------------------------/#


    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    #/----------------------------------------------------------------------------\#
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]
    #\----------------------------------------------------------------------------/#


    # try to access the server
    #/----------------------------------------------------------------------------\#

    # try to get information from local
    # check two locations:
    #   1) <tmp-data/satfile> directory under er3t main directory
    #   2) current directory;
    #/--------------------------------------------------------------\#
    fdir_satfile_tmp = '%s/satfile' % fdir_data_tmp
    if not os.path.exists(fdir_satfile_tmp):
        os.makedirs(fdir_satfile_tmp)

    fname_local1 = os.path.abspath('%s/%s' % (fdir_satfile_tmp, os.path.basename(fname_server)))
    fname_local2 = os.path.abspath('%s/%s' % (local           , os.path.basename(fname_server)))

    if os.path.exists(fname_local1):
        with open(fname_local1, 'r') as f_:
            content = f_.read()

    elif os.path.exists(fname_local2):
        os.system('cp %s %s' % (fname_local2, fname_local1))
        with open(fname_local2, 'r') as f_:
            content = f_.read()
    #\--------------------------------------------------------------/#

    else:

        # get information from server
        #/--------------------------------------------------------------\#
        try:
            with requests.Session() as session:
                session.auth = (username, password)
                r1     = session.request('get', fname_server)
                r      = session.get(r1.url, auth=(username, password))
        except:
            msg = '\nError [get_satfile_tag]: cannot access <%s>.' % fname_server
            raise OSError(msg)

        if r.ok:
            content = r.content.decode('utf-8')
        else:
            msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.' % fname_server
            raise OSError(msg)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    # extract granule information from <content>
    # after the following session, granule information will be stored under <data>
    # data['GranuleID'].decode('UTF-8') to get the file name of MODIS granule
    # data['StartDateTime'].decode('UTF-8') to get the time stamp of MODIS granule
    # variable names can be found through
    # print(data.dtype.names)
    #/----------------------------------------------------------------------------\#

    if vname in ['Aqua|MODIS', 'Terra|MODIS']:
        dtype = ['|S41', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    elif vname in ['NOAA20|VIIRS', 'SNPP|VIIRS']:
        dtype = ['|S43', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    usecols = (0, 4, 9, 10, 11, 12, 13, 14, 15, 16)

    #\----------------------------------------------------------------------------/#
    # LAADS DAAC servers are known to cause some issues occasionally while
    # accessing the metadata. We will attempt to read the txt file online directly
    # on the server but as a backup, we will download the txt file locally and
    # access the data there
    #/----------------------------------------------------------------------------\#
    try:
        data  = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols)
    except ValueError:

        msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.\nAttempting to download the file to access the data...\n' % fname_server
        print(msg)

        try:
            token = os.environ['EARTHDATA_TOKEN']
        except KeyError:
            token = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'

            msg = '\nWarning [download_laads_https]: Please get a token by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.zshrc\'(Mac),\nexport EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\n'
            warnings.warn(msg)

        try:
            command = "wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 {} --header \"Authorization: Bearer {}\" -O {}".format(fname_server, token, fname_local1)
            os.system(command)
            with open(fname_local1, 'r') as f_:
                content = f_.read()
            data = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols)
        except ValueError:
            msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.\nThis is likely an issue with LAADS DAAC servers, please try downloading the files manually or try again later.\n' % fname_server
            raise OSError(msg)

    #\----------------------------------------------------------------------------/#
    # loop through all the "MODIS granules" constructed through four corner points
    # and find which granules contain the input data
    #/----------------------------------------------------------------------------\#
    Ndata = data.size
    filename_tags = []
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
        #/----------------------------------------------------------------------------\#
        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()
        #\----------------------------------------------------------------------------/#

        # find the precise center point of MODIS granule
        #/----------------------------------------------------------------------------\#
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        #\----------------------------------------------------------------------------/#

        proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
        LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

        modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
        pointsIn       = modis_granule.contains_points(LonLat_in)
        percentIn      = float(pointsIn.sum()) * 100.0 / float(pointsIn.size)
        # if pointsIn.sum()>0 and percentIn>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
        if pointsIn.sum()>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
            filename = data[i]['GranuleID'].decode('UTF-8')
            filename_tag = '.'.join(filename.split('.')[1:3])
            filename_tags.append(filename_tag)

    #\----------------------------------------------------------------------------/#
    return filename_tags



def download_laads_https(
             date,
             dataset_tag,
             filename_tag,
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             fdir_prefix='/archive/allData',
             day_interval=1,
             fdir_out='tmp-data',
             data_format=None,
             run=True,
             verbose=True):


    """
    Downloads products from the LAADS Data Archive (DAAC).

    Input:
        date: Python datetime object
        dataset_tag: string, collection + dataset name, e.g. '61/MYD06_L2'
        filename_tag: string, string pattern in the filename, e.g. '.2035.'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        day_interval=: integer, for 8 day data, day_interval=8
        fdir_out=: string, output data directory
        data_format=None: e.g., 'hdf'
        run=: boolean type, if False, the command will only be displayed but not run
        verbose=: boolean type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded satellite data file paths
    """

    try:
        token = os.environ['EARTHDATA_TOKEN']
    except KeyError:
        token = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'
        if verbose:
            msg = '\nWarning [download_laads_https]: Please get a token by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.zshrc\'(Mac),\nexport EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\n'
            warnings.warn(msg)

    if shutil.which('curl'):
        command_line_tool = 'curl'
    elif shutil.which('wget'):
        command_line_tool = 'wget'
    else:
        msg = '\nError [download_laads_https]: <download_laads_https> needs <curl> or <wget> to be installed.'
        raise OSError(msg)

    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)

    fdir_server = server + fdir_data

    #\----------------------------------------------------------------------------/#
    # Use error handling to overcome occasional issues with LAADS DAAC servers
    #/----------------------------------------------------------------------------\#
    try:
        webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
    except urllib.error.HTTPError:
        msg = "The LAADS DAAC servers appear to be down. Attempting again in 10 seconds..."
        print(msg)
        time.sleep(10)
        try:
            webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
        except urllib.error.HTTPError:
            msg = '\nError [download_laads_https]: cannot access <%s>.' % fdir_server
            raise OSError(msg)
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
                command = 'mkdir -p %s && curl -H \'Authorization: Bearer %s\' -L -C - \'%s\' -o \'%s\' --max-time 300' % (fdir_out, token, fname_server, fname_local)
            elif command_line_tool == 'wget':
                command = 'mkdir -p %s && wget -c "%s" --header "Authorization: Bearer %s" -O %s' % (fdir_out, fname_server, token, fname_local)
            else:
                msg = '\nError [download_laads_https]: command line tool %s is not currently supported. Please use one of `curl` or `wget`.' % command_line_tool
                raise OSError(msg)
            commands.append(command)

    if not run:
        print('Message [download_laads_https]: The commands to run are:')
        for command in commands:
            print(command)

    else:

        for i, command in enumerate(commands):

            if verbose:
                print('Message [download_laads_https]: Downloading %s ...' % fnames_local[i])
            os.system(command)

            fname_local = fnames_local[i]

            if data_format is None:
                data_format = os.path.basename(fname_local).split('.')[-1]

            if data_format == 'hdf':

                try:
                    from pyhdf.SD import SD, SDC
                    import pyhdf
                except ImportError:
                    msg = '\nError [download_laads_https]: To use \'download_laads_https\', \'pyhdf\' needs to be installed.'
                    raise ImportError(msg)

                #\----------------------------------------------------------------------------/#
                # Attempt to download files. In case of an HDF4Error, attempt to re-download
                # afer a time period as this could be caused by an internal timeout at
                # the server side
                #/----------------------------------------------------------------------------\#
                try:
                    if verbose:
                        print('Message [download_laads_https]: Reading \'%s\' ...\n' % fname_local)
                    f = SD(fname_local, SDC.READ)
                    f.end()
                    if verbose:
                        print('Message [download_laads_https]: \'%s\' has been downloaded.\n' % fname_local)

                except pyhdf.error.HDF4Error:
                    print('Message [download_laads_https]: Encountered an error with \'%s\', trying again ...\n' % fname_local)
                    try:
                        os.remove(fname_local)
                        time.sleep(10) # wait 10 seconds
                        os.system(command) # re-download
                        f = SD(fname_local, SDC.READ)
                        f.end()
                        if verbose:
                            print('Message [download_laads_https]: \'%s\' has been downloaded.\n' % fname_local)
                    except pyhdf.error.HDF4Error:
                        print('Message [download_laads_https]: WARNING: Failed to read \'%s\'. File will be deleted as it might not be downloaded correctly. \n' % fname_local)
                        fnames_local.remove(fname_local)
                        os.remove(fname_local)
                        continue


            elif data_format == 'nc':

                try:
                    from netCDF4 import Dataset
                    f = Dataset(fname_local, 'r')
                    f.close()
                    if verbose:
                        print('Message [download_laads_https]: <%s> has been downloaded.\n' % fname_local)
                except:
                    msg = '\nWarning [download_laads_https]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
                    warnings.warn(msg)


            elif data_format == 'h5':

                try:
                    import h5py
                    f = h5py.File(fname_local, 'r')
                    f.close()
                    if verbose:
                        print('Message [download_laads_https]: <%s> has been downloaded.\n' % fname_local)
                except:
                    msg = '\nWarning [download_laads_https]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
                    warnings.warn(msg)

            else:

                msg = '\nWarning [download_laads_https]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
                warnings.warn(msg)

    return fnames_local



def download_lance_https(
             date,
             dataset_tag,
             filename_tag,
             server='https://nrt3.modaps.eosdis.nasa.gov/api/v2/content',
             fdir_prefix='/archives/allData',
             day_interval=1,
             fdir_out='tmp-data',
             data_format=None,
             run=True,
             verbose=True):


    """
    Downloads products from the LANCE Near Real Time (NRT) Data Archive (DAAC).

    Input:
        date: Python datetime object
        dataset_tag: string, collection + dataset name, e.g. '61/MYD06_L2'
        filename_tag: string, string pattern in the filename, e.g. '.2035.'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        day_interval=: integer, for 8 day data, day_interval=8
        fdir_out=: string, output data directory
        data_format=None: e.g., 'hdf'
        run=: boolean type, if False, the command will only be displayed but not run
        verbose=: boolean type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded satellite data file paths
    """

    try:
        token = os.environ['EARTHDATA_TOKEN']
    except KeyError:
        token = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'
        if verbose:
            msg = '\nWarning [download_lance_https]: Please get a token by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.zshrc\'(Mac),\nexport EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\n'
            warnings.warn(msg)

    if shutil.which('curl'):
        command_line_tool = 'curl'
    elif shutil.which('wget'):
        command_line_tool = 'wget'
    else:
        msg = '\nError [download_lance_https]: <download_lance_https> needs <curl> or <wget> to be installed.'
        raise OSError(msg)

    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    #\----------------------------------------------------------------------------/#
    # VIIRS NRT is labeled differently from the standard product.
    # Therefore, the dataset_tag needs to be updated only for VIIRS NRT products.
    #/----------------------------------------------------------------------------\#
    if dataset_tag.split('/')[-1].upper().startswith(('VNP', 'VJ1', 'VJ2')):
        dataset_tag = dataset_tag + '_NRT'

    fdir_csv_prefix = '/details/allData'
    fdir_csv_format = '?fields=all&formats=csv'
    fdir_csv_data   = '%s/%s/%s/%s%s' % (fdir_csv_prefix, dataset_tag, year_str, doy_str, fdir_csv_format)

    fdir_data       = '%s/%s/%s/%s.csv' % (fdir_prefix, dataset_tag, year_str, doy_str)
    fdir_server     = server + fdir_data
    fdir_csv_server = server + fdir_csv_data

    #\----------------------------------------------------------------------------/#
    # Use error handling to overcome occasional issues with LANCE DAAC servers
    #/----------------------------------------------------------------------------\#
    try:
        webpage = urllib.request.urlopen(fdir_csv_server)
    except urllib.error.HTTPError:
        msg = "The LANCE DAAC servers appear to be down or there could be an error with the fetch request. Attempting again in 10 seconds..."
        print(msg)
        time.sleep(10)
        try:
            webpage = urllib.request.urlopen(fdir_csv_server)
        except urllib.error.HTTPError:
            msg = '\nError [download_lance_https]: cannot access <%s>.' % fdir_csv_server
            raise OSError(msg)

    content  = webpage.read().decode('utf-8')
    lines    = content.split('\n')

    commands = []
    fnames_local = []
    for line in lines:
        filename = line.strip().split(',')[0]
        if filename_tag in filename and (filename.endswith('.hdf') or filename.endswith('.nc')):
            fname_server = '%s/%s' % (fdir_server, filename)
            fname_local  = '%s/%s' % (fdir_out, filename)
            fnames_local.append(fname_local)
            if command_line_tool == 'curl':
                command = 'mkdir -p %s && curl -H \'Authorization: Bearer %s\' -L -C - \'%s\' -o \'%s\' --max-time 300' % (fdir_out, token, fname_server, fname_local)
            elif command_line_tool == 'wget':
                command = 'mkdir -p %s && wget -c "%s" --header "Authorization: Bearer %s" -O %s' % (fdir_out, fname_server, token, fname_local)
            else:
                msg = '\nError [download_lance_https]: command line tool %s is not currently supported. Please use one of `curl` or `wget`.' % command_line_tool
                raise OSError(msg)
            commands.append(command)

    if not run:
        print('Message [download_lance_https]: The commands to run are:')
        for command in commands:
            print(command)

    else:

        for i, command in enumerate(commands):

            if verbose:
                print('Message [download_lance_https]: Downloading %s ...' % fnames_local[i])
            os.system(command)

            fname_local = fnames_local[i]

            if data_format is None:
                data_format = os.path.basename(fname_local).split('.')[-1]

            if data_format == 'hdf':

                try:
                    from pyhdf.SD import SD, SDC
                    import pyhdf
                except ImportError:
                    msg = '\nError [download_lance_https]: To use \'download_lance_https\', \'pyhdf\' needs to be installed.'
                    raise ImportError(msg)

                #\----------------------------------------------------------------------------/#
                # Attempt to download files. In case of an HDF4Error, attempt to re-download
                # afer a time period as this could be caused by an internal timeout at
                # the server side
                #/----------------------------------------------------------------------------\#
                try:
                    if verbose:
                        print('Message [download_lance_https]: Reading \'%s\' ...\n' % fname_local)
                    f = SD(fname_local, SDC.READ)
                    f.end()
                    if verbose:
                        print('Message [download_lance_https]: \'%s\' has been downloaded.\n' % fname_local)

                except pyhdf.error.HDF4Error:
                    print('Message [download_lance_https]: Encountered an error with \'%s\', trying again ...\n' % fname_local)
                    try:
                        os.remove(fname_local)
                        time.sleep(10) # wait 10 seconds
                        os.system(command) # re-download
                        f = SD(fname_local, SDC.READ)
                        f.end()
                        if verbose:
                            print('Message [download_lance_https]: \'%s\' has been downloaded.\n' % fname_local)
                    except pyhdf.error.HDF4Error:
                        msg = 'Warning [download_lance_https]: Failed to read \'%s\'. File will be deleted as it might not be downloaded correctly. \n' % fname_local
                        warnings.warn(msg)
                        fnames_local.remove(fname_local)
                        os.remove(fname_local)
                        continue


            elif data_format == 'nc':

                try:
                    from netCDF4 import Dataset
                    f = Dataset(fname_local, 'r')
                    f.close()
                    if verbose:
                        print('Message [download_lance_https]: <%s> has been downloaded.\n' % fname_local)
                except:
                    msg = '\nWarning [download_lance_https]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
                    warnings.warn(msg)


            elif data_format == 'h5':

                try:
                    import h5py
                    f = h5py.File(fname_local, 'r')
                    f.close()
                    if verbose:
                        print('Message [download_lance_https]: <%s> has been downloaded.\n' % fname_local)
                except:
                    msg = '\nWarning [download_lance_https]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
                    warnings.warn(msg)

            else:

                msg = '\nWarning [download_lance_https]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
                warnings.warn(msg)

    return fnames_local



def download_worldview_rgb(
        date,
        extent,
        fdir_out='tmp-data',
        instrument='modis',
        satellite='aqua',
        wmts_cgi='https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
        layer_name0=None,
        proj=None,
        coastline=False,
        fmt='png',
        run=True
        ):

    """
    Purpose: download satellite RGB imagery from NASA Worldview for a user-specified date and region
    Inputs:
        date: date object of <datetime.datetime>
        extent: rectangular region, Python list of [west_most_longitude, east_most_longitude, south_most_latitude, north_most_latitude]
        fdir_out=: directory to store RGB imagery from NASA Worldview
        instrument=: satellite instrument, currently only supports 'modis' and 'viirs'
        satellite=: satellite, currently only supports 'aqua' and 'terra' for 'modis', and 'snpp' and 'noaa20' for 'viirs'
        wmts_cgi=: cgi link to NASA Worldview GIBS (Global Imagery Browse Services)
        proj=: map projection for plotting the RGB imagery
        coastline=: boolen type, whether to plot coastline
        fmt=: can be either 'png' or 'h5'
        run=: boolen type, whether to plot
    Output:
        fname: file name of the saved RGB file (png format)
    Usage example:
        import datetime
        fname = download_wordview_rgb(datetime.datetime(2022, 5, 18), [-94.26,-87.21,31.86,38.91], instrument='modis', satellite='aqua')
    """

    if instrument.lower() == 'modis' and (satellite.lower() in ['aqua', 'terra']):
        instrument = instrument.upper()
        satellite  = satellite.lower().title()
        sat_kind = 'polar-orbiting'
    elif instrument.lower() == 'viirs' and (satellite.lower() in ['noaa20', 'snpp']):
        instrument = instrument.upper()
        satellite  = satellite.upper()
        sat_kind = 'polar-orbiting'
    elif instrument.lower() == 'abi' and (satellite.lower() in ['goes-east', 'goes-west']):
        instrument = instrument.upper()
        satellite  = satellite.upper().replace('WEST', 'West').replace('EAST', 'East')
        sat_kind = 'geostationary'
    else:
        msg = 'Error [download_worldview_rgb]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)

    if sat_kind == 'polar-orbiting':
        date_s = date.strftime('%Y-%m-%d')
        if layer_name0 is None:
            layer_name0='CorrectedReflectance_TrueColor',
        layer_name = '%s_%s_%s' % (instrument, satellite, layer_name0)
    elif sat_kind == 'geostationary':
        date += datetime.timedelta(minutes=5)
        date -= datetime.timedelta(minutes=date.minute % 10,
                                   seconds=0)
        date_s = date.strftime('%Y-%m-%dT%H:%M:%SZ')
        if layer_name0 is None:
            layer_name0='GeoColor'
        layer_name = '%s_%s_%s' % (satellite, instrument, layer_name0)

    fname  = '%s/%s-%s_rgb_%s_(%s).png' % (fdir_out, instrument, satellite, date_s, ','.join(['%.2f' % extent0 for extent0 in extent]))
    fname  = os.path.abspath(fname)

    if run:

        try:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.image as mpl_img
        except ImportError:
            msg = 'Error [download_worldview_rgb]: Please install <matplotlib> to proceed.'
            raise ImportError(msg)

        try:
            import cartopy.crs as ccrs
        except ImportError:
            msg = 'Error [download_worldview_rgb]: Please install <cartopy> to proceed.'
            raise ImportError(msg)

        if not os.path.exists(fdir_out):
            os.makedirs(fdir_out)

        if proj is None:
            proj=ccrs.PlateCarree()

        try:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(111, projection=proj)
            ax1.add_wmts(wmts_cgi, layer_name, wmts_kwargs={'time': date_s})
            if coastline:
                ax1.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            # ax1.outline_patch.set_visible(False) # changed according to DeprecationWarning
            ax1.spines['geo'].set_visible(False)
            ax1.axis('off')
            plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close(fig)
        except:
            msg = '\nError [download_wordview_rgb]: Unable to download imagery for <%s> onboard <%s> at <%s>.' % (instrument, satellite, date_s)
            warnings.warn(msg)

    if fmt == 'png':

        pass

    elif fmt == 'h5':

        try:
            import h5py
        except ImportError:
            msg = 'Error [download_worldview_rgb]: Please install <h5py> to proceed.'
            raise ImportError(msg)

        data = mpl_img.imread(fname)

        lon  = np.linspace(extent[0], extent[1], data.shape[1])
        lat  = np.linspace(extent[2], extent[3], data.shape[0])

        fname = fname.replace('.png', '.h5')

        f = h5py.File(fname, 'w')

        f['extent'] = extent

        f['lon'] = lon
        f['lon'].make_scale('Longitude')

        f['lat'] = lat
        f['lat'].make_scale('Latitude')

        f['rgb'] = np.swapaxes(data[::-1, :, :3], 0, 1)
        f['rgb'].dims[0].label = 'Longitude'
        f['rgb'].dims[0].attach_scale(f['lon'])
        f['rgb'].dims[1].label = 'Latitude'
        f['rgb'].dims[1].attach_scale(f['lat'])
        f['rgb'].dims[2].label = 'RGB'

        f.close()

    return fname



def download_oco2_https(
             dtime,
             dataset_tag,
             fnames=None,
             server='https://oco2.gesdisc.eosdis.nasa.gov',
             fdir_prefix='/data/OCO2_DATA',
             fdir_out='data',
             data_format=None,
             run=True,
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
        verbose=: Boolen type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded OCO2 file paths
    """
    from er3t.util.oco2 import get_fnames_from_web, get_dtime_from_xml

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
            if (dtime >= dtime_s) & (dtime <= dtime_e):
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

        print('Message [download_oco2_https]: The commands to run are:')
        for command in commands:
            print(command)

    else:

        for i, command in enumerate(commands):

            if verbose:
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
                if verbose:
                    print('Message [download_oco2_https]: \'%s\' has been downloaded.\n' % fname_local)

            elif data_format == 'nc':

                try:
                    import netCDF4 as nc4
                except ImportError:
                    msg = 'Warning [downlad_oco2_https]: To use \'download_oco2_https\', \'netCDF4\' needs to be installed.'
                    raise ImportError(msg)

                f = nc4.Dataset(fname_local, 'r')
                f.close()
                if verbose:
                    print('Message [download_oco2_https]: \'%s\' has been downloaded.\n' % fname_local)

            else:

                print('Warning [download_oco2_https]: Do not support check for \'%s\'. Do not know whether \'%s\' has been successfully downloaded.\n' % (data_format, fname_local))

    return fnames_local



if __name__ == '__main__':

    pass
