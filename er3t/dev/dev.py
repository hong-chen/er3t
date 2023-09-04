import os
import sys
import glob
import copy
import shutil
import datetime
import time
import requests
import warnings
import urllib.request
from io import StringIO
import numpy as np
from scipy import interpolate, stats
from scipy.spatial import KDTree
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.path as mpl_path
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy
mpl.use('Agg')


import er3t


# __all__ = ['cal_dtime_fast']


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

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = '\nError [cal_proj_xy_geometa]: Needs <cartopy> to be installed before proceeding.'
        raise ImportError(msg)

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


def cal_sec_offset_abi(extent, satname='GEOS-East|ABI', sec_per_scan=30.0):

    """
    Details see https://www.goes-r.gov/users/abiScanModeInfo.html
    """

    import cartopy.crs as ccrs

    satellite, instrument = satname.split('|')

    if satname == 'GEOS-East|ABI':

        full_disk_time_span = {
                0: [[19.8, 26.6]],
                1: [[3.7, 12.4], [12.9, 23.0]],
                2: [[4.5, 15.7]],
                3: [[3.0, 15.1]],
                4: [[2.7, 15.4]],
                5: [[2.5, 15.6]],
                6: [[2.3, 15.8]],
                7: [[2.1, 16.0]],
                8: [[2.0, 16.0]],
                9: [[2.0, 16.1]],
               10: [[2.0, 16.1]],
               11: [[2.0, 16.0]],
               12: [[2.1, 16.0]],
               13: [[2.3, 15.8]],
               14: [[2.4, 15.6]],
               15: [[2.7, 15.4]],
               16: [[3.0, 15.0]],
               17: [[3.5, 14.6]],
               18: [[4.0, 14.1]],
               19: [[4.7, 13.4], [13.9, 20.7]],
                }

        center_lon = -75.0
        center_lat = 0.0

    elif satname == 'GEOS-West|ABI':

        full_disk_time_span = {
                0: None,
                1: [[1.4,  8.2], [ 8.7, 17.4], [18.0, 28.1]],
                2: [[5.1, 16.3]],
                3: [[2.3, 14.4]],
                4: [[2.0, 14.7]],
                5: [[1.8, 14.9]],
                6: [[1.6, 15.1]],
                7: [[1.4, 15.3]],
                8: [[1.3, 15.3]],
                9: [[1.3, 15.4]],
               10: [[1.3, 15.4]],
               11: [[1.3, 15.3]],
               12: [[1.4, 15.3]],
               13: [[1.6, 15.1]],
               14: [[1.8, 14.9]],
               15: [[2.0, 14.7]],
               16: [[2.3, 14.4]],
               17: [[2.8, 13.9]],
               18: [[3.3, 13.4], [14.0, 22.7]],
               19: [[1.4,  8.2]],
                }

        center_lon = -137.0
        center_lat = 0.0

    else:

        msg = '\nError [cal_utc_abi]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)


    # define projections
    #/----------------------------------------------------------------------------\#
    proj_lonlat = ccrs.PlateCarree()
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    #\----------------------------------------------------------------------------/#

    # get scan stripe edges in y
    #/----------------------------------------------------------------------------\#
    Nscan = len(full_disk_time_span.keys())

    lat_scan_edges = np.linspace(90.0, -90.0, Nscan+1)
    lon_scan_edges = np.repeat(center_lon, Nscan+1)
    xy = proj_xy.transform_points(proj_lonlat, lon_scan_edges, lat_scan_edges)[:, [0, 1]]

    y_scan_edges = xy[:, 1]
    #\----------------------------------------------------------------------------/#


    # calculate corner points from <extent>
    #/----------------------------------------------------------------------------\#
    lon_in_ = np.arange(extent[0], extent[1], 0.001)
    lat_in_ = np.arange(extent[2], extent[3], 0.001)
    lon_in, lat_in = np.meshgrid(lon_in_, lat_in_, indexing='ij')

    xy_in = proj_xy.transform_points(proj_lonlat, lon_in.ravel(), lat_in.ravel())[:, [0, 1]]
    x_in = xy_in[:, 0]
    y_in = xy_in[:, 1]

    y_min = np.nanmin(y_in)
    y_max = np.nanmax(y_in)
    #\----------------------------------------------------------------------------/#

    Ns = max(np.argmin(np.abs(y_scan_edges-y_max))-1, 0)
    Ne = min(np.argmin(np.abs(y_scan_edges-y_min))+1, Nscan)

    sec_offset = np.zeros_like(x_in)
    sec_offset[...] = np.nan

    for i in range(Ns, Ne):
        span_time0 = full_disk_time_span[i]
        if span_time0 is not None:
            span_time0 = span_time0[-1]
            time_s = span_time0[0] + i * sec_per_scan
            time_e = span_time0[1] + i * sec_per_scan
            y_edge_min = y_scan_edges[i+1]
            y_edge_max = y_scan_edges[i]
            logic_in = (y_in>=y_edge_min) & (y_in<y_edge_max)

            if logic_in.sum() > 0:

                # calculate time offset
                #/----------------------------------------------------------------------------\#
                R_earth = np.nanmax(y_scan_edges)
                delta_scan_x_half = R_earth * np.sin(np.arccos(((y_edge_min+y_edge_max)/2.0)/R_earth))
                delta_x = delta_scan_x_half*2.0
                x_s = -delta_scan_x_half

                delta_t = time_e-time_s
                slope = delta_t/delta_x

                x_min = np.nanmin(x_in[logic_in])

                time0 = time_s + ((x_min-x_s)/delta_x)*delta_t
                sec_offset[logic_in] = time0 + slope*(x_in[logic_in]-x_min)
                #\----------------------------------------------------------------------------/#


    return sec_offset




def test_aqua_modis():

    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/MYD03_2019-09-02.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa(content)
    #\----------------------------------------------------------------------------/#

    Ndata = len(data)
    for i in range(Ndata):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa(line, delta_t=300.0, N_cross=1354, N_along=2030, scan='cw', testing=True)

def test_terra_modis():

    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/MOD03_2023-08-18.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa(content)
    #\----------------------------------------------------------------------------/#

    Ndata = len(data)
    for i in range(Ndata):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa(line, delta_t=300.0, N_cross=1354, N_along=2030, scan='cw', testing=True)

def test_snpp_viirs():

    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/VNP03MOD_2023-08-05.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa(content)
    #\----------------------------------------------------------------------------/#

    Ndata = len(data)
    for i in range(Ndata):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa(line, delta_t=360.0, N_along=3248, N_cross=3200, scan='cw', testing=True)

def test_noaa20_viirs_extra():

    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '/data/hong/2023/work/01_libera/03_demo/data_l1b/VJ103MOD_2021-05-18.txt'
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa(content)
    #\----------------------------------------------------------------------------/#

    Ndata = len(data)
    for i in range(Ndata):

        line = data[i]

        pattern = '.'.join(line['GranuleID'].split('.')[:3])
        fnames = glob.glob('/data/hong/2023/work/01_libera/03_demo/data_l1b/VJ103MOD/2021/138/*%s*.nc' % pattern)
        if len(fnames) == 1:
            fname = fnames[0]
            f = Dataset(fname, 'r')
            lon0 = f.groups['geolocation_data'].variables['longitude'][...]
            lat0 = f.groups['geolocation_data'].variables['latitude'][...]
            f.close()

            N_along, N_cross = lon0.shape

            lon1, lat1, jday1 = cal_lon_lat_utc_geometa(line, delta_t=360.0, N_along=N_along, N_cross=N_cross, scan='cw', testing=True)

            filename = os.path.basename(fname)

            # figure
            #/----------------------------------------------------------------------------\#
            if True:
                plt.close('all')
                fig = plt.figure(figsize=(12, 12))
                fig.suptitle(filename)
                # plot
                #/--------------------------------------------------------------\#
                ax1 = fig.add_subplot(221)
                cs = ax1.imshow(lon0, origin='lower', cmap='jet', zorder=0)
                ax1.set_title('Longitude (Original)')
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)
                ax1.set_xlabel('N (along track)')
                ax1.set_ylabel('N (cross track)')

                ax2 = fig.add_subplot(222)
                cs = ax2.imshow(lat0, origin='lower', cmap='jet', zorder=0)
                ax2.set_title('Latitude (Original)')
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)
                ax2.set_xlabel('N (along track)')
                ax2.set_ylabel('N (cross track)')

                ax3 = fig.add_subplot(223)
                cs = ax3.imshow(lon1, origin='lower', cmap='jet', zorder=0)
                ax3.set_title('Longitude (Estimated)')
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)
                ax3.set_xlabel('N (along track)')
                ax3.set_ylabel('N (cross track)')

                ax4 = fig.add_subplot(224)
                cs = ax4.imshow(lat1, origin='lower', cmap='jet', zorder=0)
                ax4.set_title('Latitude (Estimated)')
                divider = make_axes_locatable(ax4)
                cax = divider.append_axes('right', '5%', pad='3%')
                cbar = fig.colorbar(cs, cax=cax)
                ax4.set_xlabel('N (along track)')
                ax4.set_ylabel('N (cross track)')
                #\--------------------------------------------------------------/#
                # save figure
                #/--------------------------------------------------------------\#
                fig.subplots_adjust(hspace=0.3, wspace=0.3)
                _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                fname_png = filename.replace('.nc', '.png')
                fig.savefig(fname_png, bbox_inches='tight', metadata=_metadata)
                #\--------------------------------------------------------------/#
            #\----------------------------------------------------------------------------/#

            print(i)
            print(line)
            print()


if __name__ == '__main__':

    # test_aqua_modis()
    # test_terra_modis()
    # test_snpp_viirs()
    # test_noaa20_viirs_extra()
    # extent = [-60.0, -59.0, 12.5, 13.5]
    # extent = [-110.0, -109.0, 31.0, 32.0]
    # extent = [-60.0, -59.0, -53.5, -52.5]
    sec_offset = cal_sec_offset_abi(extent, satname='GEOS-East|ABI')
    pass
