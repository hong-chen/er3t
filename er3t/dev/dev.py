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

import er3t




__all__ = ['grid_nearest_fast']




def grid_nearest_fast(x, y, data, x_2d, y_2d, Ngrid_limit=1, fill_value=np.nan):

    """
    Use scipy.spatial.KDTree to perform fast nearest gridding

    References:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html

    Inputs:
        x: x position of raw data
        y: y position of raw data
        data: value of raw data
        x_2d: x position of the data (to be gridded)
        y_2d: y position of the data (to be gridded)
        Ngrid_limit=<1>=: number of grids for defining "too far"
        fill_value=<np.nan>: fill-in value for the data that is "too far" away from raw data

    Output:
        data_2d: gridded data
    """

    # check equidistant
    #/----------------------------------------------------------------------------\#
    if er3t.util.check_equidistant(x_2d[:, 0]) and er3t.util.check_equidistant(x_2d[:, -1]) and \
       er3t.util.check_equidistant(y_2d[0, :]) and er3t.util.check_equidistant(y_2d[-1, :]):

        dx = x_2d[1, 0] - x_2d[0, 0]
        dy = y_2d[0, 1] - y_2d[0, 0]

    else:

        msg = '\nError [grid_nearest_fast]: Do not support non-equidistant gridding.'
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # preprocess raw data
    #/----------------------------------------------------------------------------\#
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#


    # check whether raw data is contained within the gridded region
    #/----------------------------------------------------------------------------\#
    logic_in  = (x>=x_2d[0, 0]) & (x<=x_2d[-1, 0]) & (y>=y_2d[0, 0]) & (y<=y_2d[0, -1])

    x = x[logic_in]
    y = y[logic_in]
    data = data[logic_in]
    #\----------------------------------------------------------------------------/#


    # create KDTree
    #/----------------------------------------------------------------------------\#
    points = np.transpose(np.vstack((x, y)))
    tree_xy = KDTree(points, leafsize=50)
    #\----------------------------------------------------------------------------/#


    # search KDTree for the nearest neighbor
    #/----------------------------------------------------------------------------\#
    points_query = np.transpose(np.vstack((x_2d.ravel(), y_2d.ravel())))
    dist_xy, indices_xy = tree_xy.query(points_query, workers=-1)

    dist_2d = dist_xy.reshape(x_2d.shape)
    data_2d = data[indices_xy].reshape(x_2d.shape)
    #\----------------------------------------------------------------------------/#


    # use fill value to fill in grids that are "two far"* away from raw data
    #   * by default 1 grid away is defined as "too far"
    #/----------------------------------------------------------------------------\#
    dist_limit = np.sqrt((dx*Ngrid_limit)**2+(dy*Ngrid_limit)**2)
    logic_out = (dist_2d>dist_limit)

    data_2d[logic_out] = fill_value
    #\----------------------------------------------------------------------------/#

    return data_2d




if __name__ == '__main__':

    pass
