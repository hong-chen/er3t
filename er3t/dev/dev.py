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




__all__ = ['cal_dtime_fast']



def cal_dtime_fast(corners, extent):

    pass


def read_geometa_txt(content, sname='Aqua|MODIS'):

    if sname in ['Aqua|MODIS', 'Terra|MODIS']:
        dtype = ['|S41', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    elif sname in ['NOAA20|VIIRS', 'SNPP|VIIRS']:
        dtype = ['|S43', '|S1','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8']
    usecols = (0, 4, 9, 10, 11, 12, 13, 14, 15, 16)

    data = np.genfromtxt(StringIO(content), delimiter=',', skip_header=2, names=True, dtype=dtype, invalid_raise=False, loose=True, usecols=usecols)

    return data

def test():

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

    # proj_ori = ccrs.PlateCarree()
    # for i in range(Ndata):

    #     line = data[i]
    #     xx0  = np.array([line['GRingLongitude1'], line['GRingLongitude2'], line['GRingLongitude3'], line['GRingLongitude4'], line['GRingLongitude1']])
    #     yy0  = np.array([line['GRingLatitude1'] , line['GRingLatitude2'] , line['GRingLatitude3'] , line['GRingLatitude4'] , line['GRingLatitude1']])

    #     if (abs(xx0[0]-xx0[1])>180.0) | (abs(xx0[0]-xx0[2])>180.0) | \
    #        (abs(xx0[0]-xx0[3])>180.0) | (abs(xx0[1]-xx0[2])>180.0) | \
    #        (abs(xx0[1]-xx0[3])>180.0) | (abs(xx0[2]-xx0[3])>180.0):

    #         xx0[xx0<0.0] += 360.0

    #     # roughly determine the center of granule
    #     #/----------------------------------------------------------------------------\#
    #     xx = xx0[:-1]
    #     yy = yy0[:-1]
    #     center_lon = xx.mean()
    #     center_lat = yy.mean()
    #     #\----------------------------------------------------------------------------/#

    #     # find the precise center point of MODIS granule
    #     #/----------------------------------------------------------------------------\#
    #     proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    #     LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
    #     center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
    #     center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
    #     #\----------------------------------------------------------------------------/#

    #     proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    #     LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
    #     LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

    #     modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
    #     pointsIn       = modis_granule.contains_points(LonLat_in)
    #     percentIn      = float(pointsIn.sum()) * 100.0 / float(pointsIn.size)
    #     # if pointsIn.sum()>0 and percentIn>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
    #     if pointsIn.sum()>0 and data[i]['DayNightFlag'].decode('UTF-8')=='D':
    #         filename = data[i]['GranuleID'].decode('UTF-8')
    #         filename_tag = '.'.join(filename.split('.')[1:3])
    #         filename_tags.append(filename_tag)


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

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.scatter(lon0, lat0, s=1, c='k', lw=0.0)
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



if __name__ == '__main__':

    test()
    pass
