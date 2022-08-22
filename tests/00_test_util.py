"""
Purpose:
    testing code under er3t/util
"""

import os
import sys
import glob
import datetime
import multiprocessing as mp

import numpy as np
from scipy import interpolate


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpl_img
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import cartopy.crs as ccrs

import er3t
from er3t.util import grid_by_extent


def test_download_worldview():

    from er3t.util import download_worldview_rgb

    date = datetime.datetime(2022, 5, 18)
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]

    download_worldview_rgb(date, extent, fdir_out='tmp-data', instrument='modis', satellite='aqua'  , fmt='png')
    download_worldview_rgb(date, extent, fdir_out='tmp-data', instrument='modis', satellite='terra' , fmt='png')
    download_worldview_rgb(date, extent, fdir_out='tmp-data', instrument='viirs', satellite='snpp'  , fmt='h5')
    download_worldview_rgb(date, extent, fdir_out='tmp-data', instrument='viirs', satellite='noaa20', fmt='h5')



def test_download_laads():

    from er3t.util import download_laads_https
    from er3t.util import get_doy_tag

    date = datetime.datetime(2022, 5, 18)
    doy_tag = get_doy_tag(date, day_interval=1)
    print(doy_tag)


    pass



def test_modis():

    download_modis_rgb(datetime.datetime(2015, 9, 6), [-2.0077, 2.9159, 48.5883, 53.4864], fdir='.', which='aqua', coastline=True)
    download_modis_rgb(datetime.datetime(2015, 9, 6), [-2.0077, 2.9159, 48.5883, 53.4864], fdir='.', which='terra', coastline=True)
    sys.exit()


    lon = np.arange(10.0, 15.0)
    lat = np.arange(10.0, 15.0)
    # tile_tags = get_sinusoidal_grid_tag(lon, lat)
    filename_tags = get_filename_tag(datetime.datetime(2016, 3, 10), lon, lat)
    print(filename_tags)
    # print(tile_tags)

    # tag = get_doy_tag(datetime.datetime(2016, 3, 10), day_interval=8)
    # print(tag)
    # dtime = datetime.datetime(2017, 8, 13)
    # download_modis_https(dtime, '6/MOD09A1','h01v10', day_interval=8, run=False)

def test_viirs():

    import er3t.util.viirs

    fname_03  = 'tmp-data/VNP03IMG.A2022138.1912.002.2022139022209.nc'
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]
    f03 = er3t.util.viirs.viirs_03(fnames=[fname_03], extent=extent, vnames=['height'])

    fname_l1b = 'tmp-data/VNP02IMG.A2022138.1912.002.2022139023833.nc'
    f02 = er3t.util.viirs.viirs_l1b(fnames=[fname_l1b], f03=f03)

    lon_2d, lat_2d, rad_2d = grid_by_extent(f02.data['lon']['data'], f02.data['lat']['data'], f02.data['rad']['data'].filled(fill_value=np.nan), extent=extent)
    lon_2d, lat_2d, ref_2d = grid_by_extent(f02.data['lon']['data'], f02.data['lat']['data'], f02.data['ref']['data'].filled(fill_value=np.nan), extent=extent)

    #/---------------------------------------------------------------------------\
    fig = plt.figure(figsize=(16, 5.5))

    ax1 = fig.add_subplot(131)

    img = mpl_img.imread('tmp-data/VIIRS-SNPP_rgb_2022-05-18_(-94.26,-87.21,31.86,38.91).png')
    ax1.imshow(img, extent=extent)
    ax1.set_xlabel('Longitude [$^\circ$]')
    ax1.set_ylabel('Latitude [$^\circ$]')
    ax1.set_title('VIIRS (Suomi NPP) RGB')

    ax2 = fig.add_subplot(132)
    cs  = ax2.imshow(rad_2d.T, origin='lower', extent=extent, cmap='jet', vmin=0.0, vmax=0.4)
    ax2.set_xlabel('Longitude [$^\circ$]')
    ax2.set_ylabel('Latitude [$^\circ$]')
    ax2.set_title('VIIRS Radiance (Band I01, 650 nm)')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(cs, cax=cax)

    ax3 = fig.add_subplot(133)
    cs = ax3.imshow(ref_2d.T, origin='lower', extent=extent, cmap='jet', vmin=0.0, vmax=1.0)
    ax3.set_xlabel('Longitude [$^\circ$]')
    ax3.set_ylabel('Latitude [$^\circ$]')
    ax3.set_title('VIIRS Reflectance (Band I01, 650 nm)')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(cs, cax=cax)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    plt.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)

    plt.show()
    exit()
    #\---------------------------------------------------------------------------/

    pass




def main():
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    url = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer = 'VIIRS_CityLights_2012'

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_wmts(url, layer)
    ax.set_extent((-15, 25, 35, 60))

    plt.title('Suomi NPP Earth at night April/October 2012')
    plt.show()

if __name__ == '__main__':

    # test_modis()

    # test_download_worldview() # passed test on 2022-08-19

    # test_download_laads()

    test_viirs()

    # main()
