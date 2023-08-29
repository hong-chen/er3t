import os
import sys
import copy
import shutil
import datetime
import time
import requests
import urllib.request
from io import StringIO
import numpy as np
from scipy import interpolate, stats
from scipy.spatial import KDTree
import warnings

from pyhdf.SD import SD, SDC



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

    data = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols).reshape((1, -1))

    return data


def cal_lon_lat_utc_geometa_line(line, res=1000.0, delta_t=300.0, N_scan=203, ascending=True, rotation='ccw'):

    """
    Aqua    (delta_t=300.0, N_scan=203, ascending=True): ascending orbit
    Terra   (delta_t=300.0, N_scan=203, ascending=False): descending orbit
    NOAA-20 (delta_t=360.0, N_scan=203, ascending=True): ascending orbit
    S-NPP   (delta_t=360.0, N_scan=203, ascending=True): ascending orbit
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
    line = np.squeeze(line)
    proj_lonlat = ccrs.PlateCarree()
    #\----------------------------------------------------------------------------/#


    # lon_ = [lower-right, lower-left, upper-left, upper-right, lower-right]
    # lat_ = [lower-right, lower-left, upper-left, upper-right, lower-right]
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


    # reconstruct x y grids
    # c: cross track
    # a: along track
    #/----------------------------------------------------------------------------\#
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy = proj_xy.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]

    x = xy[:, 0]
    y = xy[:, 1]

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

    N_a = int(dist_a//res)
    N_c = int(dist_c//res)
    i_a = np.arange(N_a, dtype=np.float64)
    i_c = np.arange(N_c, dtype=np.float64)
    ii_a, ii_c = np.meshgrid(i_a, i_c, indexing='ij')

    ang_a = np.arctan(slope_a)
    ang_c = np.arctan(slope_c)

    xx = x[0]-res*(ii_c*np.cos(ang_c)+ii_a*np.cos(ang_a))
    yy = y[0]-res*(ii_c*np.sin(ang_c)+ii_a*np.sin(ang_a))
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
    delta_t0_c = delta_t0/N_c*i_c

    N_a0 = int(N_a//N_scan)

    for i in range(N_scan):
        index_s = N_a0*i
        index_e = N_a0*(i+1)
        jday_out0_ = jday0+delta_t0*i+delta_t0_c

        if (i == N_scan-1):
            index_e = N_a
            N_a0 = N_a - index_s

        jday_out0 = np.tile(jday_out0_, N_a0).reshape((N_a0, N_c))
        jday_out[index_s:index_e, :] = jday_out0

    if rotation == 'cw':
        jday_out = jday_out[:, ::-1]

    if not ascending:
        jday_out = jday_out[::-1, :]
    #\----------------------------------------------------------------------------/#

    return lon_out, lat_out, jday_out


def test():

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.image as mpl_img
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams, ticker
    from matplotlib.ticker import FixedLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # mpl.use('Agg')


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

        lon1, lat1, jday1 = cal_lon_lat_utc_geometa_line(line, ascending=False, rotation='cw')

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
        fig = plt.figure(figsize=(12, 5))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(121)
        ax1.scatter(lon0, lat0, s=10, c='k', lw=0.0)
        ax1.scatter(lon1, lat1, s=0.1, c='r', lw=0.0)
        # ax1.imshow(lon0, cmap='jet', origin='lower', aspect='auto', vmin=lon0.min(), vmax=lon0.max())
        # ax1.scatter(lon1, lat1, s=1, c='r', lw=0.0)
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_title('Original')
        #\--------------------------------------------------------------/#

        #/--------------------------------------------------------------\#
        ax2 = fig.add_subplot(122)
        ax2.imshow(jday1, cmap='jet', origin='lower', aspect='auto')
        # ax2.scatter(lon0, lat0, s=10, c='gray', lw=0.0)
        # ax2.scatter(lon1, lat1, s=0.1, c='r', lw=0.0)
        ax2.set_xlabel('Longitude [$^\circ$]')
        ax2.set_ylabel('Latitude [$^\circ$]')
        ax2.set_title('New')
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



if __name__ == '__main__':

    test()
    pass
