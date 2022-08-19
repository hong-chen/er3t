"""
Purpose:
    testing code under er3t/util
"""

import er3t
import datetime
import xarray as xr
from er3t.util.viirs import *

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import cartopy.crs as ccrs



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

    data_tag = '5200/VNP02IMG'

    fname_03 = 'data/VNP03IMG.A2020140.1400.002.2021125145414.nc'
    with xr.open_dataset(fname_03, group='geolocation_data') as f_03:
        lon = f_03.longitude
        lat = f_03.latitude

        if lon.valid_min == -180.0:
            lon[lon<0.0] += 360.0

        print(lon)
    exit()

    fname_l1b = 'data/VNP02IMG.A2020140.1400.002.2021127041009.nc'
    with xr.open_dataset(fname_l1b, group='observation_data') as f_l1b:
        rad = f_l1b.I01


    #/---------------------------------------------------------------------------\
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(lon, lat, c=rad, lw=0.0, cmap='Greys_r', vmin=0.0, vmax=1.0, s=3)
    # ax1.imshow(.T, extent=extent, origin='lower', cmap='jet', zorder=0)
    # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
    # ax1.set_xlim(())
    # ax1.set_ylim(())
    # ax1.set_xlabel('')
    # ax1.set_ylabel('')
    # ax1.set_title('')
    # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
    # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
    #
    # _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    # plt.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
    #
    plt.show()
    exit()
    #\---------------------------------------------------------------------------/

    pass



def test_download_laads():

    from er3t.util import download_laads_https

    pass

def test_download_worldview():

    from er3t.util import download_worldview_rgb

    date = datetime.datetime(2022, 5, 18)
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]
    download_worldview_rgb(date, extent, instrument='modis', satellite='aqua')
    download_worldview_rgb(date, extent, instrument='modis', satellite='terra')
    download_worldview_rgb(date, extent, instrument='viirs', satellite='snpp')
    download_worldview_rgb(date, extent, instrument='viirs', satellite='noaa20')


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

    # test_viirs()

    # test_download_laads()

    test_download_worldview()
    # main()
