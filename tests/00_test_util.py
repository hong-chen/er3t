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

    # data_tag = '5200/VNP02IMG'
    # data_tag = '5200/VNP03IMG'

    fname_03  = 'tmp-data/VNP03IMG.A2022138.1912.002.2022139022209.nc'
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]
    f03 = er3t.util.viirs.viirs_03(fnames=[fname_03], extent=extent, vnames=['height'])

    # fname_l1b = 'tmp-data/VNP02IMG.A2022138.1912.002.2022139023833.nc'

    #/---------------------------------------------------------------------------\
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(f03.data['lon']['data'], f03.data['lat']['data'], c=f03.data['vaa']['data'], lw=0.0, cmap='jet', s=3)
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
