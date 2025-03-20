import os
import sys
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import numpy as np
from scipy import interpolate

__all__ = ['cal_shd_saa', 'cal_shd_vaa']


def cal_shd_saa(normal_azimuth_angle):

    """
    Convert normal azimuth angle (0 pointing north, positive when clockwise) to viewing azimuth in SHDOM

    Input:
        normal_azimuth_angle: float/integer, normal azimuth angle (0 pointing north, positive when clockwise)

    Output:
        SHDOM solar azimuth angle (0 sun shining from west, positive when counterclockwise)
    """

    while normal_azimuth_angle < 0.0:
        normal_azimuth_angle += 360.0

    while normal_azimuth_angle > 360.0:
        normal_azimuth_angle -= 360.0

    shd_saa = 270.0 - normal_azimuth_angle
    if shd_saa < 0.0:
        shd_saa += 360.0

    return shd_saa


def cal_shd_vaa(normal_azimuth_angle):

    """
    Convert normal azimuth angle (0 pointing north, positive when clockwise) to viewing azimuth in SHDOM

    Input:
        normal_azimuth_angle: float/integer, normal azimuth angle (0 pointing north, positive when clockwise)

    Output:
        SHDOM sensor azimuth angle (0 sensor looking from east, positive when counterclockwise)
    """

    while normal_azimuth_angle < 0.0:
        normal_azimuth_angle += 360.0

    while normal_azimuth_angle > 360.0:
        normal_azimuth_angle -= 360.0

    shd_vaa = 90.0 - normal_azimuth_angle
    if shd_vaa < 0.0:
        shd_vaa += 360.0

    return shd_vaa


if __name__ == '__main__':

    date = datetime.datetime(2019, 9, 2)
    sza = 34.93346840064262
    saa = -144.90807471473198
    vza = 14.44008093613803
    vaa = -99.99851773953723


    print('SOLARFLUX: %.6f' % er3t.util.cal_sol_fac(date))
    print('SOLARMU: %.6f' % np.cos(np.deg2rad(sza)))
    print('SOLARAZ: %.6f' % cal_shd_saa(saa))

    print('SENSORMU: %.6f' % np.cos(np.deg2rad(vza)))
    print('SENSORAZ: %.6f' % cal_shd_vaa(vaa))
    pass
