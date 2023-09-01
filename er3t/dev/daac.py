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


__all__ = [
        'get_satname', \
        'get_token_earthdata', \
        'get_login_earthdata', \
        'get_fname_geometa', \
        'get_local_file', \
        'get_online_file', \
        'read_geometa', \
        'cal_proj_xy_geometa', \
        'cal_lon_lat_utc_geometa', \
        'get_satfile_tag', \
        'download_laads_https', \
        'get_nrt_satfile_tag', \
        'download_lance_https',\
        'download_oco2_https', \
        'download_worldview_image', \
        ]


def get_satname(satellite, instrument):

    # check satellite and instrument
    #/----------------------------------------------------------------------------\#
    if instrument.lower() == 'modis' and (satellite.lower() in ['aqua', 'terra']):
        instrument = instrument.upper()
        satellite  = satellite.lower().title()
    elif instrument.lower() == 'viirs' and (satellite.lower() in ['noaa20', 'snpp', 'noaa-20', 's-npp']):
        instrument = instrument.upper()
        satellite  = satellite.replace('-', '').upper()
    elif instrument.lower() == 'abi' and (satellite.lower() in ['goes-east', 'goes-west']):
        instrument = instrument.upper()
        satellite  = satellite.upper().replace('WEST', 'West').replace('EAST', 'East')
    else:
        msg = '\nError [get_satname]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)
    #\----------------------------------------------------------------------------/#

    satname = '%s|%s' % (satellite, instrument)

    return satname



def get_token_earthdata():

    try:
        token = os.environ['EARTHDATA_TOKEN']
    except KeyError:
        token = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'

        msg = '\nWarning [get_earthdata_token]: Please get a token by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.zshrc\'(Mac),\nexport EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\n'
        warnings.warn(msg)

    return token



def get_login_earthdata():

    try:
        username = os.environ['EARTHDATA_USERNAME']
        password = os.environ['EARTHDATA_PASSWORD']

        return username, password

    except:
        msg = '\nError [get_earthdata_login]: cannot find environment variables \'EARTHDATA_USERNAME\' and \'EARTHDATA_PASSWORD\'.'
        raise OSError(msg)

        return



def get_command_earthdata(
        fname_target,
        filename=None,
        tools=['curl', 'wget'],
        token=get_token_earthdata(),
        fdir_save='%s/satfile' % er3t.common.fdir_data_tmp,
        ):

    if filename is None:
        filename = os.path.basename(fname_target)

    fname_save = '%s/%s' % (fdir_save, filename)

    header = '"Authorization: Bearer %s"' % token

    options = {
            'curl': '--header %s --connect-timeout 120.0 --retry 3 --location --continue-at - --output %s %s' % (header, fname_save, fname_target),
            'wget': '--header=%s --continue --timeout=120 --tries=3 --show-progress --output-document=%s --quiet %s' % (header, fname_save, fname_target),
            }

    command = None

    for command_line_tool in tools:

        if shutil.which(command_line_tool):

            command = 'mkdir -p %s && %s %s' % (fdir_save, command_line_tool, options[command_line_tool])

            if command is not None:

                return command

    return command



def get_fname_geometa(
        date,
        satname='Aqua|MODIS',
        server='https://ladsweb.modaps.eosdis.nasa.gov',
        ):

    # generate satellite filename on LAADS DAAC server
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y-%m-%d')
    fnames_geometa = {
           'Aqua|MODIS': '%s/archive/geoMeta/61/AQUA/%4.4d/MYD03_%s.txt'              % (server, date.year, date_s),
          'Terra|MODIS': '%s/archive/geoMeta/61/TERRA/%4.4d/MOD03_%s.txt'             % (server, date.year, date_s),
         'NOAA20|VIIRS': '%s/archive/geoMetaVIIRS/5200/NOAA-20/%4.4d/VJ103MOD_%s.txt' % (server, date.year, date_s),
           'SNPP|VIIRS': '%s/archive/geoMetaVIIRS/5110/NPP/%4.4d/VNP03MOD_%s.txt'     % (server, date.year, date_s),
        'NOAA-20|VIIRS': '%s/archive/geoMetaVIIRS/5200/NOAA-20/%4.4d/VJ103MOD_%s.txt' % (server, date.year, date_s),
          'S-NPP|VIIRS': '%s/archive/geoMetaVIIRS/5110/NPP/%4.4d/VNP03MOD_%s.txt'     % (server, date.year, date_s),
        }
    fname_geometa = fnames_geometa[satname]
    #\----------------------------------------------------------------------------/#

    return fname_geometa



def get_local_file(
        fname_file,
        filename=None,
        fdir_local='./',
        fdir_save='%s/satfile' % er3t.common.fdir_data_tmp,
        ):

    if filename is None:
        filename = os.path.basename(fname_file)

    # try to get information from local
    # check two locations:
    #   1) <tmp-data/satfile> directory under er3t main directory
    #   2) current directory;
    #/--------------------------------------------------------------\#
    if not os.path.exists(fdir_save):
        os.makedirs(fdir_save)

    fname_local1 = os.path.abspath('%s/%s' % (fdir_save, filename))
    fname_local2 = os.path.abspath('%s/%s' % (fdir_local, filename))

    if os.path.exists(fname_local1):

        with open(fname_local1, 'r') as f_:
            content = f_.read()

    elif os.path.exists(fname_local2):

        os.system('cp %s %s' % (fname_local2, fname_local1))
        with open(fname_local2, 'r') as f_:
            content = f_.read()

    else:

        content = None
    #\--------------------------------------------------------------/#

    return content



def get_online_file(
        fname_file,
        filename=None,
        download=True,
        fdir_save='%s/satfile' % er3t.common.fdir_data_tmp,
        ):

    if filename is None:
        filename = os.path.basename(fname_file)

    if download:

        fname_save = '%s/%s' % (fdir_save, filename)
        command = get_command_earthdata(fname_file, filename=filename, fdir_save=fdir_save)
        os.system(command)

        content = get_local_file(fname_file, filename=filename, fdir_save=fdir_save)

    else:

        # this can be revisited, disabling it for now
        #/--------------------------------------------------------------\#
        # try:
        #     with requests.Session() as session:
        #         session.auth = (username, password)
        #         r1     = session.request('get', fname_server)
        #         r      = session.get(r1.url, auth=(username, password))
        # except:
        #     msg = '\nError [get_online_file]: cannot access <%s>.' % fname_server
        #     raise OSError(msg)

        # if r.ok:
        #     content = r.content.decode('utf-8')
        # else:
        #     msg = '\nError [get_online_file]: failed to retrieve information from <%s>.' % fname_server
        #     warnings.warn(msg)
        #\--------------------------------------------------------------/#

        # this can be revisited, disabling it for now
        # Use error handling to overcome occasional issues with LAADS DAAC servers
        #/----------------------------------------------------------------------------\#
        # try:
        #     webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
        # except urllib.error.HTTPError:
        #     msg = "The LAADS DAAC servers appear to be down. Attempting again in 10 seconds..."
        #     print(msg)
        #     time.sleep(10)
        #     try:
        #         webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
        #     except urllib.error.HTTPError:
        #         msg = '\nError [get_online_file]: cannot access <%s>.' % fdir_server
        #         raise OSError(msg)
        # content  = webpage.read().decode('utf-8')
        #\----------------------------------------------------------------------------/#

        content = None

    return content



def final_file_check(fname_local, data_format=None, verbose=False):

    if data_format is None:
        data_format = os.path.basename(fname_local).split('.')[-1].lower()

    checked = False

    if data_format in ['hdf', 'hdf4', 'h4']:

        try:
            import pyhdf
            from pyhdf.SD import SD, SDC
            f = SD(fname_local, SDC.READ)
            f.end()

            checked = True
        except:
            pass

    elif data_format in ['nc', 'nc4', 'netcdf', 'netcdf4']:
        try:
            from netCDF4 import Dataset
            f = Dataset(fname_local, 'r')
            f.close()

            checked = True
        except:
            pass

    elif data_format in ['h5', 'hdf5']:

        try:
            import h5py
            f = h5py.File(fname_local, 'r')
            f.close()

            checked = True
        except:
            pass

    else:

        msg = '\nWarning [final_file_check]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
        warnings.warn(msg)

    if checked:
        if verbose:
            msg = '\nMessage [final_file_check]: <%s> has been successfully downloaded.\n' % fname_local
            print(msg)

    else:
        msg = '\nWarning [final_file_check]: Do not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
        warnings.warn(msg)



def read_geometa(content):

    """
    Parse geometa data in a list of Python dictionaries that contain information such as "GranuleID", "GRingLongitude1" etc.

    Input:
        content: a long string that contains the whole content of the geometa txt file

    Output:
        data: a list of Python dictionaries.

              An example,
              {'GranuleID': 'MYD03.A2019245.0000.061.2019245151541.hdf',
               'StartDateTime': '2019-09-02 00:00',
               'ArchiveSet': '61',
               'OrbitNumber': '92177',
               'DayNightFlag': 'N',
               'EastBoundingCoord': '40.0756634630551',
               'NorthBoundingCoord': '-57.0356008659127',
               'SouthBoundingCoord': '-81.1407080607332',
               'WestBoundingCoord': '-42.1164863420172',
               'GRingLongitude1': '-10.7904979067681',
               'GRingLongitude2': '32.4557535169638',
               'GRingLongitude3': '40.6409172253246',
               'GRingLongitude4': '-42.4223934777823',
               'GRingLatitude1': '-57.0310011746246',
               'GRingLatitude2': '-63.1593513428869',
               'GRingLatitude3': '-81.224336567583',
               'GRingLatitude4': '-68.7511407602523',
               'Satellite': 'Aqua',
               'Instrument': 'MODIS',
               'Orbit': 'Ascending'}
    """

    lines = content.split('\n')

    index_header = 0
    while (len(lines[index_header]) > 0) and lines[index_header][0] == '#':
        index_header += 1

    index_header -= 1

    if index_header == -1:
        msg = '\nError [read_geometa]: Cannot locate header in the provided content.'
        raise OSError(msg)

    header_line = lines[index_header]
    vnames = [word.strip() for word in header_line[1:].split(',')]

    Nvar = len(vnames)

    data = []
    for line_data in lines[index_header+1:]:
        if len(line_data) > 0:
            data0_ = [word.strip() for word in line_data.split(',')]
            data0 = {vnames[i]:data0_[i] for i in range(Nvar)}

            if 'MYD03' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'Aqua'
                data0['Instrument'] = 'MODIS'
                data0['Orbit']      = 'Ascending'
            elif 'MOD03' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'Terra'
                data0['Instrument'] = 'MODIS'
                data0['Orbit']      = 'Descending'
            elif 'VJ103' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'NOAA-20'
                data0['Instrument'] = 'VIIRS'
                data0['Orbit']      = 'Ascending'
            elif 'VNP03' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'S-NPP'
                data0['Instrument'] = 'VIIRS'
                data0['Orbit']      = 'Ascending'
            else:
                data0['Satellite']  = 'Unknown'
                data0['Instrument'] = 'Unknown'
                data0['Orbit']      = 'Unknown'

            data.append(data0)

    return data



def cal_proj_xy_geometa(line_data, closed=True):

    """
    Calculate globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points

    Input:
        line_data: Python dictionary (details see <read_geo_meta>) that contains basic information of a satellite granule
        closed=True: if True, return five corner points with the last point repeating the first point;
                     if False, return four corner points

    Output:
        proj_xy: globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points
        xy: dimension of (5, 2) if <closed=True> and (4, 2) if <closed=False>
    """

    import cartopy.crs as ccrs

    # get corner points
    #/----------------------------------------------------------------------------\#
    lon_  = np.array([
        float(line_data['GRingLongitude1']),
        float(line_data['GRingLongitude2']),
        float(line_data['GRingLongitude3']),
        float(line_data['GRingLongitude4']),
        float(line_data['GRingLongitude1'])
        ])

    lat_  = np.array([
        float(line_data['GRingLatitude1']),
        float(line_data['GRingLatitude2']),
        float(line_data['GRingLatitude3']),
        float(line_data['GRingLatitude4']),
        float(line_data['GRingLatitude1'])
        ])

    if (abs(lon_[0]-lon_[1])>180.0) | (abs(lon_[0]-lon_[2])>180.0) | \
       (abs(lon_[0]-lon_[3])>180.0) | (abs(lon_[1]-lon_[2])>180.0) | \
       (abs(lon_[1]-lon_[3])>180.0) | (abs(lon_[2]-lon_[3])>180.0):

        lon_[lon_<0.0] += 360.0
    #\----------------------------------------------------------------------------/#


    # roughly determine the center of granule
    #/----------------------------------------------------------------------------\#
    lon = lon_[:-1]
    lat = lat_[:-1]
    center_lon_ = lon.mean()
    center_lat_ = lat.mean()
    #\----------------------------------------------------------------------------/#


    # find the true center
    #/----------------------------------------------------------------------------\#
    proj_lonlat = ccrs.PlateCarree()

    proj_xy_ = ccrs.Orthographic(central_longitude=center_lon_, central_latitude=center_lat_)
    xy_ = proj_xy_.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]

    center_x  = xy_[:, 0].mean()
    center_y  = xy_[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_xy_)
    #\----------------------------------------------------------------------------/#


    # convert lon/lat corner points into xy
    #/----------------------------------------------------------------------------\#
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy_  = proj_xy.transform_points(proj_lonlat, lon_, lat_)[:, [0, 1]]
    #\----------------------------------------------------------------------------/#


    if closed:
        return proj_xy, xy_
    else:
        return proj_xy, xy_[:-1, :]



def cal_lon_lat_utc_geometa(
        line_data,
        delta_t=300.0,
        N_along=2030,
        N_cross=1354,
        N_scan=203,
        scan='cw',
        testing=False,
        ):

    """
    Calculate (more of an estimation) longitude, latitude, and utc time (julian day) from corner points provided by geometa

    Input:
        line_data: Python dictionary (details see <read_geo_meta>) that contains basic information of a satellite granule
        delta_t=300.0: time span of the given granule in seconds, e.g., 5 minutes for MODIS granule
        N_along=2030: number of pixels along the satellite track
        N_cross=1354: number of pixels across the satellite track
        N_scan=203: number of rotatory scans, MODIS and VIIRS both have N_scan=203
        scan='cw': direction of the rotatory scan (viewing along the satellite travel direction), MODIS and VIIRS both rotate clockwise
        testing=False: testing mode, if True, a figure will be generated

    Output:
        lon_out: longitude, dimension of (N_along, N_cross)
        lat_out: latitude, dimension of (N_along, N_cross)
        jday_out: julian day, dimension of (N_along, N_cross)

    Notes:
    Aqua    (delta_t=300.0, N_scan=203, N_along=2030, N_cross=1354, scan='cw')
    Terra   (delta_t=300.0, N_scan=203, N_along=2030, N_cross=1354, scan='cw')
    NOAA-20 (delta_t=360.0, N_scan=203, N_along=3248, N_cross=3200, scan='cw')
    S-NPP   (delta_t=360.0, N_scan=203, N_along=3248, N_cross=3200, scan='cw')
    """

    import cartopy.crs as ccrs

    # check if delta_t is correct
    #/----------------------------------------------------------------------------\#
    if line_data['Instrument'].lower() == 'modis' and delta_t != 300.0:
        msg = '\nWarning [cal_lon_lat_utc_geometa]: MODIS should have <delta_t=300.0> but given <delta_t=%.1f>, please double-check.' % delta_t
        warning.warn(msg)
    elif line_data['Instrument'].lower() == 'viirs' and delta_t != 360.0:
        msg = '\nWarning [cal_lon_lat_utc_geometa]: VIIRS should have <delta_t=360.0> but given <delta_t=%.1f>, please double-check.' % delta_t
        warning.warn(msg)
    #\----------------------------------------------------------------------------/#


    # get lon/lat corner points into xy
    #/----------------------------------------------------------------------------\#
    proj_xy, xy_ = cal_proj_xy_geometa(line_data, closed=True)
    xy  = xy_[:-1, :]
    x = xy[:, 0]
    y = xy[:, 1]
    #\----------------------------------------------------------------------------/#


    # reconstruct x y grids
    # c: cross track
    # a: along track
    #/----------------------------------------------------------------------------\#
    x_01_c = (x[0]+x[1])/2.0
    y_01_c = (y[0]+y[1])/2.0

    x_23_c = (x[2]+x[3])/2.0
    y_23_c = (y[2]+y[3])/2.0

    dist_a = np.sqrt((x_23_c-x_01_c)**2+(y_23_c-y_01_c)**2)
    slope_a = (y_23_c-y_01_c) / (x_23_c-x_01_c)

    x_03_c = (x[0]+x[3])/2.0
    y_03_c = (y[0]+y[3])/2.0

    x_12_c = (x[1]+x[2])/2.0
    y_12_c = (y[1]+y[2])/2.0

    dist_c = np.sqrt((x_03_c-x_12_c)**2+(y_03_c-y_12_c)**2)
    slope_c = (y_12_c-y_03_c) / (x_12_c-x_03_c)

    N_a = N_along
    N_c = N_cross

    i_a = np.arange(N_a, dtype=np.float64)
    i_c = np.arange(N_c, dtype=np.float64)
    ii_a, ii_c = np.meshgrid(i_a, i_c, indexing='ij')

    res_a = dist_a/N_a
    res_c = dist_c/N_c

    ang_a = np.arctan(slope_a)
    ang_c = np.arctan(slope_c)

    # this is experimental, might cause some problem in the future
    #/--------------------------------------------------------------\#
    if   ((x[0]>x[1]) or (x[3]>x[2])) or ((y[2]>y[1]) or (y[3]>y[0])):
        if ((x[0]>x[3]) or (x[1]>x[2])):
            index0 = 0
        else:
            index0 = 3
    else:
        if ((x[0]>x[3]) or (x[1]>x[2])):
            index0 = 1
        else:
            index0 = 2
    #\--------------------------------------------------------------/#

    xx = x[index0]-res_c*ii_c*np.cos(ang_c)-res_a*ii_a*np.cos(ang_a)
    yy = y[index0]-res_c*ii_c*np.sin(ang_c)-res_a*ii_a*np.sin(ang_a)
    #\----------------------------------------------------------------------------/#


    # calculate lon lat
    #/----------------------------------------------------------------------------\#
    proj_lonlat = ccrs.PlateCarree()
    lonlat_out = proj_lonlat.transform_points(proj_xy, xx, yy)[..., [0, 1]]
    lon_out = lonlat_out[..., 0]
    lat_out = lonlat_out[..., 1]
    #\----------------------------------------------------------------------------/#


    # calculate utc (jday)
    #/----------------------------------------------------------------------------\#
    filename = line_data['GranuleID']
    dtime0_s = '.'.join(filename.split('.')[1:3])
    dtime0 = datetime.datetime.strptime(dtime0_s, 'A%Y%j.%H%M')
    jday0 = er3t.util.dtime_to_jday(dtime0)

    jday_out = np.zeros(lon_out.shape, dtype=np.float64)
    delta_t0 = delta_t / N_scan

    delta_t0_c = delta_t0/3.0/N_c*i_c  # 120 degree coverage thus </3.0>
    if scan == 'ccw':
        delta_t0_c = delta_t0_c[::-1]

    # this is experimental, might cause some problem in the future
    #/--------------------------------------------------------------\#
    if index0 in [1, 3]:
        lon_out = lon_out[:, ::-1]
        lat_out = lat_out[:, ::-1]
        delta_t0_c = delta_t0_c[::-1]
    #\--------------------------------------------------------------/#

    N_a0 = int(N_a//N_scan)

    for i in range(N_scan):
        index_s = N_a0*i
        index_e = N_a0*(i+1)
        jday_out0_ = jday0+(delta_t0*i+delta_t0_c)/86400.0

        if (i == N_scan-1):
            index_e = N_a
            N_a0 = N_a - index_s

        jday_out0 = np.tile(jday_out0_, N_a0).reshape((N_a0, N_c))
        jday_out[index_s:index_e, :] = jday_out0
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if testing:
        utc_sec_out = (jday_out-jday_out.min())*86400.0

        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111, projection=proj_xy)
        cs = ax1.scatter(lon_out[::5, ::5], lat_out[::5, ::5], c=utc_sec_out[::5, ::5], transform=ccrs.PlateCarree(), vmin=0.0, vmax=delta_t, cmap='jet', s=1, lw=0.0)
        ax1.text(x[0], y[0], '0-LR', color='black')
        ax1.text(x[1], y[1], '1-LL', color='black')
        ax1.text(x[2], y[2], '2-UL', color='black')
        ax1.text(x[3], y[3], '3-UR', color='black')

        granule  = mpl_path.Path(xy_, closed=True)
        patch = mpatches.PathPatch(granule, facecolor='none', edgecolor='black', lw=1.0)
        ax1.add_patch(patch)
        if (utc_sec_out[:, -1]-utc_sec_out[:, 0]).sum()>0.0:
            ax1.scatter(xx[:, -1], yy[:, -1], c='red' , lw=0.0, s=3.0)
            ax1.scatter(xx[:, 0], yy[:, 0], c='blue', lw=0.0, s=3.0)
        else:
            ax1.scatter(xx[:, -1], yy[:, -1], c='blue' , lw=0.0, s=3.0)
            ax1.scatter(xx[:, 0], yy[:, 0], c='red', lw=0.0, s=3.0)

        ax1.set_title(filename)
        ax1.set_global()
        ax1.add_feature(cartopy.feature.OCEAN, zorder=0)
        ax1.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='none')
        ax1.coastlines(color='gray', lw=0.5)
        g3 = ax1.gridlines()
        g3.xlocator = FixedLocator(np.arange(-180, 181, 60))
        g3.ylocator = FixedLocator(np.arange(-80, 81, 20))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fname_png = filename.replace('.hdf', '.png').replace('.nc', '.png')
        fig.savefig('globe-view_%s' % fname_png, bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    return lon_out, lat_out, jday_out



def get_satfile_tag(
             date,
             lon,
             lat,
             satellite='aqua',
             instrument='modis',
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             fdir_local='./',
             fdir_save='%s/satfile' % er3t.common.fdir_data_tmp,
             geometa=False,
             verbose=False
             ):

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
    import cartopy.crs as ccrs
    import matplotlib.path as mpl_path
    #\----------------------------------------------------------------------------/#


    # get formatted satellite tag
    #/----------------------------------------------------------------------------\#
    satname = get_satname(satellite, instrument)
    #\----------------------------------------------------------------------------/#


    # get satellite geometa filename on LAADS DAAC server
    #/----------------------------------------------------------------------------\#
    fname_geometa = get_fname_geometa(date, satname, server=server)
    #\----------------------------------------------------------------------------/#


    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    #/----------------------------------------------------------------------------\#
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]
    #\----------------------------------------------------------------------------/#


    # get geometa info
    #/----------------------------------------------------------------------------\#
    # try to get geometa information from local
    content = get_local_file(fname_geometa, fdir_local=fdir_local, fdir_save=fdir_save)

    # try to get geometa information online
    if content is None:
        content = get_online_file(fname_geometa, fdir_save=fdir_save)
    #\----------------------------------------------------------------------------/#


    # read in geometa info
    #/----------------------------------------------------------------------------\#
    data = read_geometa(content)
    #\----------------------------------------------------------------------------/#


    # loop through all the satellite "granules" constructed through four corner points
    # and find which granules contain the input data
    #/----------------------------------------------------------------------------\#
    proj_lonlat = ccrs.PlateCarree()

    Ndata = len(data)
    filename_tags = []
    for i in range(Ndata):

        line = data[i]

        proj_xy, xy_granule = cal_proj_xy_geometa(line, closed=True)
        sat_granule  = mpl_path.Path(xy_granule, closed=True)

        xy_in      = proj_xy.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]
        points_in  = sat_granule.contains_points(xy_in)

        Npoint_in  = points_in.sum()
        Npoint_tot = points_in.size

        # placeholder for percentage threshold
        percent_in = float(Npoint_in) * 100.0 / float(Npoint_tot)

        if (Npoint_in>0) and (data[i]['DayNightFlag']=='D'):
            if geometa:
                filename_tags.append(line)
            else:
                filename = data[i]['GranuleID']
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
             fdir_save='%s/satfile' % er3t.common.fdir_data_tmp,
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

    # retrieve the directory where satellite data is stored for picked date
    #/----------------------------------------------------------------------------\#
    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)
    fdir_server = server + fdir_data
    #\----------------------------------------------------------------------------/#


    # get csv info
    #/----------------------------------------------------------------------------\#
    fname_csv = '%s.csv' % fdir_server
    filename_csv = '_'.join(fname_csv.replace('%s/' % server, '').split('/'))

    # try to get geometa information from local
    content = get_local_file(fname_csv, filename=filename_csv, fdir_save=fdir_save)

    # try to get geometa information online
    if content is None:
        content = get_online_file(fname_csv, filename=filename_csv, fdir_save=fdir_save)
    #\----------------------------------------------------------------------------/#


    # get download commands
    #/----------------------------------------------------------------------------\#
    lines    = content.split('\n')

    commands = []
    fnames_local = []
    for line in lines:
        filename = line.strip().split(',')[0]

        if filename_tag in filename:
            fname_server = '%s/%s' % (fdir_server, filename)
            fname_local  = '%s/%s' % (fdir_out, filename)
            fnames_local.append(fname_local)

            command = get_command_earthdata(fname_server, filename=filename, fdir_save=fdir_out)
            commands.append(command)
    #\----------------------------------------------------------------------------/#


    # run/print command
    #/----------------------------------------------------------------------------\#
    if run:

        for i, command in enumerate(commands):

            fname_local = fnames_local[i]

            if verbose:
                print('Message [download_laads_https]: Downloading %s ...' % fname_local)
            os.system(command)

            final_file_check(fname_local, data_format=data_format, verbose=verbose)

    else:

        print('Message [download_laads_https]: The commands to run are:')
        for command in commands:
            print(command)
    #\----------------------------------------------------------------------------/#


    return fnames_local



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
    elif instrument.lower() == 'viirs' and (satellite.lower() in ['noaa20', 'snpp', 'noaa-20', 's-npp']):
        instrument = instrument.upper()
        satellite  = satellite.upper()
    else:
        msg = 'Error [get_satfile_tag]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)
    #\----------------------------------------------------------------------------/#


    # check login
    #/----------------------------------------------------------------------------\#
    username, password = get_login_earthdata()
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
            warnings.warn(msg)
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
        data  = read_geometa(content)
    # except ValueError:
    except:

        msg = '\nError [get_satfile_tag]: failed to retrieve information from <%s>.\nAttempting to download the file to access the data...\n' % fname_server
        print(msg)

        token = get_token_earthdata()

        try:
            command = "wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 {} --header \"Authorization: Bearer {}\" -O {}".format(fname_server, token, fname_local1)
            os.system(command)
            with open(fname_local1, 'r') as f_:
                content = f_.read()
            data  = read_geometa(content)
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

    token = get_token_earthdata()

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



def download_worldview_image(
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
        dpi=300,
        run=True,
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
        fname = download_wordview_image(datetime.datetime(2022, 5, 18), [-94.26,-87.21,31.86,38.91], instrument='modis', satellite='aqua')
    """

    satname = get_satname(satellite, instrument)
    satellite, instrument = satname.split('|')

    if satellite in ['Aqua', 'Terra', 'NOAA20', 'SNPP']:
        if layer_name0 is None:
            layer_name0='CorrectedReflectance_TrueColor'
        layer_name = '%s_%s_%s' % (instrument, satellite, layer_name0)

        date_s = date.strftime('%Y-%m-%d')

        try:
            lon__ = np.arange(extent[0], extent[1], 500.0/111000.0)
            lat__ = np.arange(extent[2], extent[3], 500.0/111000.0)
            lon_, lat_ = np.meshgrid(lon__, lat__, indexing='ij')

            line_data = get_satfile_tag(date, lon_, lat_, satellite=satellite, instrument=instrument, geometa=True)[-1]
            if satellite in ['Aqua', 'Terra']:
                lon0_, lat0_, jday0_ = cal_lon_lat_utc_geometa(line_data, delta_t=300.0, N_along=2030, N_cross=1354, scan='cw', testing=False)
            else:
                lon0_, lat0_, jday0_ = cal_lon_lat_utc_geometa(line_data, delta_t=360.0, N_along=3248, N_cross=3200, scan='cw', testing=False)

            logic_in = (lon0_>=extent[0]) & (lon0_<=extent[1]) & (lat0_>=extent[2]) & (lat0_<=extent[3])
            jday0 = np.nanmean(jday0_[logic_in])
            date0 = er3t.util.jday_to_dtime(jday0)
            date_s0 = date0.strftime('%Y-%m-%dT%H:%M:%SZ')

            fname  = '%s/%s-%s_%s_%s_(%s).png' % (fdir_out, instrument, satellite, layer_name0.split('_')[-1], date_s0, ','.join(['%.2f' % extent0 for extent0 in extent]))

        except:
            fname  = '%s/%s-%s_%s_%s_(%s).png' % (fdir_out, instrument, satellite, layer_name0.split('_')[-1], date_s, ','.join(['%.2f' % extent0 for extent0 in extent]))

    elif satellite in ['GOES-West', 'GOES-East']:
        date += datetime.timedelta(minutes=5)
        date -= datetime.timedelta(minutes=date.minute % 10,
                                   seconds=date.second)
        date_s = date.strftime('%Y-%m-%dT%H:%M:%SZ')
        if layer_name0 is None:
            layer_name0='GeoColor'
        layer_name = '%s_%s_%s' % (satellite, instrument, layer_name0)

        fname  = '%s/%s-%s_%s_%s_(%s).png' % (fdir_out, instrument, satellite, layer_name0.split('_')[-1], date_s, ','.join(['%.2f' % extent0 for extent0 in extent]))

    fname  = os.path.abspath(fname)

    if run:

        import h5py
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpl_img
        import cartopy.crs as ccrs

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
            ax1.spines['geo'].set_visible(False)
            ax1.axis('off')
            plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig)

        except:

            msg = '\nError [download_wordview_image]: Unable to download imagery for <%s> onboard <%s> at <%s>.' % (instrument, satellite, date_s)
            warnings.warn(msg)

        if fmt == 'h5':

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



if __name__ == '__main__':

    pass
