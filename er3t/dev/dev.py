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
import warnings

import er3t




__all__ = ['grid_nearest_fast']




def grid_nearest_fast(x, y, data, x_2d, y_2d, fill_nan=True):

    # check equidistant
    #/----------------------------------------------------------------------------\#
    if check_equidistant(x_2d[:, 0]) and check_equidistant(x_2d[:, -1]) and
       check_equidistant(y_2d[0, :]) and check_equidistant(y_2d[-1, :]):

        dx = x_2d[1, 0] - x_2d[0, 0]
        dy = y_2d[0, 1] - y_2d[0, 0]

    else:

        msg = '\nError [grid_nearest_fast]: Do not support non-equidistant gridding.'
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # check whether raw data is contained within the gridded region
    #/----------------------------------------------------------------------------\#
    logic_in  = (x>=x_2d[0, 0]) & (x<=x_2d[-1, 0]) & (y>=y_2d[0, 0]) & (y<=y_2d[0, -1])
    logic_out = np.logical_not(logic_in)
    #\----------------------------------------------------------------------------/#


    # preprocess raw data
    #/----------------------------------------------------------------------------\#
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#


    # define data_2d for storing gridded data
    #/----------------------------------------------------------------------------\#
    data_2d = np.zeros(x_2d.shape, dtype=np.float64)
    if fill_nan:
        data_2d[...] = np.nan
    #\----------------------------------------------------------------------------/#


    indices_x = np.int_((x-x_2d[0, 0])//dx)
    indices_y = np.int_((y-y_2d[0, 0])//dy)

    nearest = np.zeros(x.size, dtype=np.float64)
    nearest[logic_in] = data_2d[indices_x[logic_in], indices_y[logic_in]]

    # deal with nan data
    #/----------------------------------------------------------------------------\#
    logic_nan  = np.isnan(data_2d)
    logic_good = np.logical_not(logic_nan)
    if np.isnan(nearest).sum()>0 and fill_nan:
        data_2d_ = data_2d[logic_good]
        x_2d_    = x_2d[logic_good]
        y_2d_    = y_2d[logic_good]

        indices_nan = np.where(np.isnan(nearest))[0]
        for index in indices_nan:
            x_ = x[index]
            y_ = y[index]
            index_closest = np.argmin(np.abs((x_2d_-x_)**2+(y_2d_-y_)**2))
            nearest[index] = data_2d_[index_closest]

    nearest[logic_out] = np.nan
    #\----------------------------------------------------------------------------/#

    return nearest




if __name__ == '__main__':

    pass
