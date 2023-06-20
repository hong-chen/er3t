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

    # flatten lon/lat/data
    #/----------------------------------------------------------------------------\#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#


    # get extent
    #/----------------------------------------------------------------------------\#
    if extent is None:
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    #\----------------------------------------------------------------------------/#


    # dist_x and dist_y
    #/----------------------------------------------------------------------------\#
    lon0 = [extent[0], extent[0]]
    lat0 = [extent[2], extent[3]]
    lon1 = [extent[1], extent[1]]
    lat1 = [extent[2], extent[3]]
    dist_x = er3t.util.cal_geodesic_dist(lon0, lat0, lon1, lat1).min()

    lon0 = [extent[0], extent[1]]
    lat0 = [extent[2], extent[2]]
    lon1 = [extent[0], extent[1]]
    lat1 = [extent[3], extent[3]]
    dist_y = er3t.util.cal_geodesic_dist(lon0, lat0, lon1, lat1).min()
    #\----------------------------------------------------------------------------/#


    # get Nx/Ny and dx/dy
    #/----------------------------------------------------------------------------\#
    if dx is None or dy is None:

        # Nx and Ny
        #/----------------------------------------------------------------------------\#
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)
        Nx = int(N0*(extent[1]-extent[0]))
        Ny = int(N0*(extent[3]-extent[2]))
        #\----------------------------------------------------------------------------/#

        # dx and dy
        #/----------------------------------------------------------------------------\#
        dx = dist_x / Nx
        dy = dist_y / Ny
        #\----------------------------------------------------------------------------/#

    else:

        Nx = dist_x // dx
        Ny = dist_y // dy
    #\----------------------------------------------------------------------------/#


    # get west-most lon_1d/lat_1d
    #/----------------------------------------------------------------------------\#
    lon_1d = np.repeat(extent[0], Ny)
    lat_1d = np.repeat(extent[2], Ny)
    for i in range(1, Ny):
        lon_1d[i], lat_1d[i] = er3t.util.cal_geodesic_lonlat(lon_1d[i-1], lat_1d[i-1], dy, 0.0)
    #\----------------------------------------------------------------------------/#


    # get lon_2d/lat_2d
    #/----------------------------------------------------------------------------\#
    lon_2d = np.zeros((Nx, Ny), dtype=np.float64)
    lat_2d = np.zeros((Nx, Ny), dtype=np.float64)
    lon_2d[0, :] = lon_1d
    lat_2d[0, :] = lat_1d
    for i in range(1, Nx):
        lon_2d[i, :], lat_2d[i, :] = er3t.util.cal_geodesic_lonlat(lon_2d[i-1, :], lat_2d[i-1, :], dx, 90.0)
    #\----------------------------------------------------------------------------/#


    # gridding
    #/----------------------------------------------------------------------------\#
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
    #\----------------------------------------------------------------------------/#




if __name__ == '__main__':

    pass
