import os
import sys
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
mpl.use('Agg')


import er3t


# __all__ = ['cal_dtime_fast']


def read_geometa_txt(content, sname='Aqua|MODIS'):

    if sname in ['Aqua|MODIS', 'Terra|MODIS']:
        dtype = ['|S41', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    elif sname in ['NOAA20|VIIRS', 'SNPP|VIIRS']:
        dtype = ['|S43', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    else:
        msg = '\nError [read_geometra_txt]: Cannot recognize <%s>.' % sname
        raise OSError(msg)

    usecols = (0, 4, 9, 10, 11, 12, 13, 14, 15, 16)

    data = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols).reshape((-1, 1))

    return data


def cal_lon_lat_utc_geometa_line(
        geometa_line,
        delta_t=300.0,
        scan='cw',
        N_scan=203,
        N_cross=1354,
        N_along=2030,
        ):

    """
    Aqua    (delta_t=300.0, N_scan=203, ascending=True , N_along=2030, N_cross=1354, scan='cw')
    Terra   (delta_t=300.0, N_scan=203, ascending=False, N_along=2030, N_cross=1354, scan='cw')
    NOAA-20 (delta_t=360.0, N_scan=203, ascending=True , N_along=3248, N_cross=3200, scan='cw')
    S-NPP   (delta_t=360.0, N_scan=203, ascending=True , N_along=3248, N_cross=3200, scan='cw')
    """

    try:
        import matplotlib.path as mpl_path
    except ImportError:
        msg = '\nError [cal_lonlat_geometa]: Needs <matplotlib> to be installed before proceeding.'
        raise ImportError(msg)

    try:
        import cartopy.crs as ccrs
    except ImportError:
        msg = '\nError [cal_lonlat_geometa]: Needs <cartopy> to be installed before proceeding.'
        raise ImportError(msg)

    # prep
    #/----------------------------------------------------------------------------\#
    line = np.squeeze(geometa_line)
    proj_lonlat = ccrs.PlateCarree()
    #\----------------------------------------------------------------------------/#


    # get corner points
    #/----------------------------------------------------------------------------\#
    lon_  = np.array([line['GRingLongitude1'], line['GRingLongitude2'], line['GRingLongitude3'], line['GRingLongitude4'], line['GRingLongitude1']])
    lat_  = np.array([line['GRingLatitude1'] , line['GRingLatitude2'] , line['GRingLatitude3'] , line['GRingLatitude4'] , line['GRingLatitude1']])

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
    proj_xy_ = ccrs.Orthographic(central_longitude=center_lon_, central_latitude=center_lat_)
    xy_ = proj_xy_.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]

    center_x  = xy_[:, 0].mean()
    center_y  = xy_[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_xy_)
    #\----------------------------------------------------------------------------/#


    # get lon/lat corner points into xy and get orientation
    #/----------------------------------------------------------------------------\#
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy  = proj_xy.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]
    xy_ = proj_xy.transform_points(proj_lonlat, lon_, lat_)[:, [0, 1]]
    x = xy[:, 0]
    y = xy[:, 1]

    if (x[0] >= x[1]) or (x[3] >= x[2]):
        inorder_x = True
    else:
        inorder_x = False

    if (y[2] >= y[1]) or (y[3] >= y[0]):
        inorder_y = True
    else:
        inorder_y = False
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

    index0 = np.argmax(x)
    xx = x[index0]-res_c*ii_c*np.cos(ang_c)-res_a*ii_a*np.cos(ang_a)
    yy = y[index0]-res_c*ii_c*np.sin(ang_c)-res_a*ii_a*np.sin(ang_a)
    #\----------------------------------------------------------------------------/#


    # calculate lon lat
    #/----------------------------------------------------------------------------\#
    lonlat_out = proj_lonlat.transform_points(proj_xy, xx, yy)[..., [0, 1]]
    lon_out = lonlat_out[..., 0]
    lat_out = lonlat_out[..., 1]
    #\----------------------------------------------------------------------------/#



    # calculate utc (jday)
    #/----------------------------------------------------------------------------\#
    filename = np.str_(line['GranuleID'])
    if filename[0] == 'b':
        filename = filename[1:]
    dtime0_s = '.'.join(filename.split('.')[1:3])
    dtime0 = datetime.datetime.strptime(dtime0_s, 'A%Y%j.%H%M')
    jday0 = er3t.util.dtime_to_jday(dtime0)

    jday_out = np.zeros(lon_out.shape, dtype=np.float64)
    delta_t0 = delta_t / N_scan

    if index0 in [0, 2]:
        delta_t0_c = delta_t0/3.0/N_c*i_c       # 120 degree coverage thus </3.0>
    else:
        delta_t0_c = delta_t0/3.0/N_c*i_c[::-1] # 120 degree coverage thus </3.0>

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

    if scan == 'ccw':
        jday_out = jday_out[:, ::-1]
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        utc_sec_out = (jday_out-jday_out.min())*86400.0

        plt.close('all')

        proj = ccrs.NearsidePerspective(central_longitude=center_lon, central_latitude=center_lat)

        fig = plt.figure(figsize=(8, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111, projection=proj)
        cs = ax1.scatter(lon_out[::5, ::5], lat_out[::5, ::5], c=utc_sec_out[::5, ::5], transform=ccrs.PlateCarree(), vmin=0.0, vmax=delta_t, cmap='jet', s=1, lw=0.0)
        cs = ax1.scatter(center_lon, center_lat, s=200, marker='*', lw=0.5, edgecolor='white', facecolor='black', transform=ccrs.PlateCarree())
        ax1.text(lon[0], lat[0], '0-LR', color='black', transform=ccrs.PlateCarree())
        ax1.text(lon[1], lat[1], '1-LL', color='black', transform=ccrs.PlateCarree())
        ax1.text(lon[2], lat[2], '2-UL', color='black', transform=ccrs.PlateCarree())
        ax1.text(lon[3], lat[3], '3-UR', color='black', transform=ccrs.PlateCarree())

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
        fname_png = '%s.png' % dtime0_s
        fig.savefig(fname_png, bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    return lon_out, lat_out, jday_out


def test():


    # target region
    #/----------------------------------------------------------------------------\#
    extent = [-109.1, -106.9, 36.9, 39.1]
    #\----------------------------------------------------------------------------/#


    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = 'MYD03_2019-09-02.txt'
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa_txt(content, sname='Aqua|MODIS')
    #\----------------------------------------------------------------------------/#

    Ndata = data.size

    for i in range(Ndata):

        line = data[i]

        lon1, lat1, jday1 = cal_lon_lat_utc_geometa_line(line, scan='cw')

    # actual 03 file
    #/----------------------------------------------------------------------------\#
    fname = '%s/data/02_modis_rad-sim/download/MYD03.A2019245.2025.061.2019246155053.hdf' % er3t.common.fdir_examples
    f = SD(fname, SDC.READ)
    lat0 = f.select('Latitude')[:]
    lon0 = f.select('Longitude')[:]
    f.end()
    #\----------------------------------------------------------------------------/#

    print(lon0.shape)
    print(lat0.shape)
    print(lon1.shape)
    print(lat1.shape)

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 12))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(lon0, lat0, s=20, c='gray', lw=0.0)

        utc_sec1 = (jday1-jday1.min())*86400.0
        cs = ax1.scatter(lon1, lat1, s=0.1, c=utc_sec1, lw=0.0, vmin=0.0, vmax=300.0, cmap='jet', alpha=1.0)
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_title('Lon/Lat/UTC')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


def test_aqua_modis():


    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/MYD03_2019-09-02.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa_txt(content, sname='Aqua|MODIS')
    #\----------------------------------------------------------------------------/#

    Ndata = data.size

    for i in range(20):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa_line(line, scan='cw')


def test_terra_modis():


    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/MOD03_2023-08-18.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa_txt(content, sname='Aqua|MODIS')
    #\----------------------------------------------------------------------------/#

    Ndata = data.size

    for i in range(20):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa_line(line, scan='cw')


def test_noaa20_viirs():


    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/VJ103MOD_2023-08-27.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa_txt(content, sname='NOAA20|VIIRS')
    #\----------------------------------------------------------------------------/#

    Ndata = data.size

    for i in range(20):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa_line(line, N_along=3248, N_cross=3200, scan='cw')


def test_snpp_viirs():


    # deal with geoMeta data
    #/----------------------------------------------------------------------------\#
    fname_txt = '%s/satfile/VNP03MOD_2023-08-05.txt' % er3t.common.fdir_data_tmp
    with open(fname_txt, 'r') as f_:
        content = f_.read()
    data = read_geometa_txt(content, sname='SNPP|VIIRS')
    #\----------------------------------------------------------------------------/#

    Ndata = data.size

    for i in range(20):

        line = data[i]

        print(i)
        print(line)
        print()

        lon, lat, jday = cal_lon_lat_utc_geometa_line(line, N_along=3248, N_cross=3200, scan='cw')


if __name__ == '__main__':

    # test()
    # test_aqua_modis()
    # test_terra_modis()
    # test_noaa20_viirs()
    # test_snpp_viirs()
    pass
