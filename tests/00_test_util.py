"""
Purpose:
    testing code under er3t/util
"""

from er3t.util import download_laads_https
from er3t.util.viirs import *

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


    pass


if __name__ == '__main__':

    # test_modis()

    test_viirs()
