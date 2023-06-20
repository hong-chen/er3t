import os
import sys
import shutil
import datetime
import time
import requests
import urllib.request
from io import StringIO
import numpy as np
from scipy import interpolate, stats
import warnings

import er3t




__all__ = ['grid_by_dxdy']


def grid_by_dxdy(lon, lat, data, extent=None, dx=None, dy=None, method='nearest'):

    """
    !!!!!!!!! under dev !!!!!!!!!
    Grid irregular data into a regular grid by input 'extent' (westmost, eastmost, southmost, northmost)
    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        extent=: Python list, [westmost, eastmost, southmost, northmost]
        NxNy=: Python list, [Nx, Ny], lon_2d = np.linspace(westmost, eastmost, Nx)
                                      lat_2d = np.linspace(southmost, northmost, Ny)
    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data
    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat data = grid_by_extent(lon0, lat0, data0, extent=[10, 15, 10, 20])
    """

    # get central lon/lat
    #/----------------------------------------------------------------------------\#
    lon0 = np.nanmean(lon)
    lat0 = np.nanmean(lat)
    #\----------------------------------------------------------------------------/#

    # get extent
    #/----------------------------------------------------------------------------\#
    if extent is None:
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    #\----------------------------------------------------------------------------/#

    # Nx and Ny
    #/----------------------------------------------------------------------------\#
    xy = (extent[1]-extent[0])*(extent[3]-extent[2])
    N0 = np.sqrt(lon.size/xy)
    Nx = int(N0*(extent[1]-extent[0]))
    Ny = int(N0*(extent[3]-extent[2]))
    #\----------------------------------------------------------------------------/#

    lon1 = lon.copy()
    lon1[...] = extent[0]
    lat1 = lat.copy()
    dist_x = er3t.util.cal_geodesic_dist(lon, lat, lon1, lat1).reshape(lon.shape)

    lon1 = lon.copy()
    lat1 = lat.copy()
    lat1[...] = extent[2]
    dist_y = er3t.util.cal_geodesic_dist(lon, lat, lon1, lat1).reshape(lat.shape)


    # point1_x, point1_y = er3t.util.cal_geodesic_lonlat(extent[0], extent[2], 222000.0, 90.0)
    # point2_x, point2_y = er3t.util.cal_geodesic_lonlat(extent[0], extent[2], 222000.0, 0.0)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.path as mpl_path
    import matplotlib.image as mpl_img
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams, ticker
    from matplotlib.ticker import FixedLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # import cartopy.crs as ccrs
    # mpl.use('Agg')
    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(dist_y.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.scatter(dist_x, dist_y, s=2, c='k', lw=0.0)
        # ax1.scatter(lon, lat, s=2, c='k', lw=0.0)
        # ax1.scatter(point1_x, point1_y, s=20, c='r', lw=0.0)
        # ax1.scatter(point2_x, point2_y, s=20, c='r', lw=0.0)
        # ax1.scatter(point1_x, point2_y, s=20, c='r', lw=0.0)
        ax1.grid()
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
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


    sys.exit()


    # get dx and dy
    #/----------------------------------------------------------------------------\#
    #\----------------------------------------------------------------------------/#






    if (dx is None) or (dy is None):

        # calculate dx and dy
        #/----------------------------------------------------------------------------\#
        lon0 = extent[0]
        lat0 = (extent[2]+extent[3])/2.0
        lon1 = extent[1]
        lat1 = (extent[2]+extent[3])/2.0
        dx = cal_geodesic_dist(lon0, lat0, lon1, lat1)/Nx

        lon0 = (extent[0]+extent[1])/2.0
        lat0 = extent[2]
        lon1 = (extent[0]+extent[1])/2.0
        lat1 = extent[3]
        dy = cal_geodesic_dist(lon0, lat0, lon1, lat1)/Ny
        #\----------------------------------------------------------------------------/#

    lon_1d0 = np.linspace(extent[0], extent[1], Nx+1)
    lat_1d0 = np.linspace(extent[2], extent[3], Ny+1)

    lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
    lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)

    points   = np.transpose(np.vstack((lon, lat)))
    data_2d0 = interpolate.griddata(points, data, (lon_2d, lat_2d), method='linear', fill_value=np.nan)

    if method == 'nearest':
        data_2d  = interpolate.griddata(points, data, (lon_2d, lat_2d), method='nearest')
        logic = np.isnan(data_2d0) | np.isnan(data_2d)
        data_2d[logic] = 0.0
        return lon_2d, lat_2d, data_2d
    else:
        logic = np.isnan(data_2d0)
        data_2d0[logic] = 0.0
        return lon_2d, lat_2d, data_2d0


if __name__ == '__main__':

    pass
