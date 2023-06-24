"""
Purpose:
    testing code under er3t/util
"""

import os
import sys
import time
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


def test_solar_spectra(fdir, date=datetime.datetime.now()):

    levels = np.linspace(0.0, 20.0, 41)
    atm0   = er3t.pre.atm.atm_atmmod(levels=levels)

    wvls   = np.arange(300.0, 2301.0, 1.0)
    sols   = np.zeros_like(wvls)

    sol_fac = er3t.util.cal_sol_fac(date)

    for i, wvl in enumerate(wvls):

        abs0 = er3t.pre.abs.abs_16g(wavelength=wvl, atm_obj=atm0)
        norm = sol_fac/(abs0.coef['weight']['data']*abs0.coef['slit_func']['data'][-1, :]).sum()
        sols[i] = norm*(abs0.coef['solar']['data']*abs0.coef['weight']['data']*abs0.coef['slit_func']['data'][-1, :]).sum()

    # figure
    #/----------------------------------------------------------------------------\#
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(wvls, sols, s=2)
    ax1.set_xlim((200, 2400))
    ax1.set_ylim((0.0, 2.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('Solar Spectra')
    plt.savefig('solar_spectra.png')
    plt.show()
    #\----------------------------------------------------------------------------/#


def test_download_laads():

    from er3t.util import download_laads_https
    from er3t.util import get_doy_tag

    date = datetime.datetime(2022, 5, 18)
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]

    doy_tag = get_doy_tag(date, day_interval=1)

    """
    https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5111/AERDB_L2_VIIRS_SNPP/2020/022/
    """

    print(doy_tag)


    pass



def test_modis():

    download_modis_rgb(datetime.datetime(2015, 9, 6), [-2.0077, 2.9159, 48.5883, 53.4864], fdir='.', which='aqua', coastline=True)
    download_modis_rgb(datetime.datetime(2015, 9, 6), [-2.0077, 2.9159, 48.5883, 53.4864], fdir='.', which='terra', coastline=True)


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
    import er3t.util.modis

    fname_03  = 'tmp-data/VNP03MOD.A2022138.1912.002.2022139022209.nc'
    extent = [-94.2607, -87.2079, 31.8594, 38.9122]

    f03 = er3t.util.viirs.viirs_03(fnames=[fname_03], extent=extent, vnames=['height'])

    fname_l1b = 'tmp-data/VNP02MOD.A2022138.1912.002.2022139023833.nc'
    f02 = er3t.util.viirs.viirs_l1b(fnames=[fname_l1b], f03=f03, band='M04')

    # VIIRS 09A1
    #/-----------------------------------------------------------------------------/
    # tags = er3t.util.modis.get_sinusoidal_grid_tag(f02.data['lon']['data'], f02.data['lat']['data'])
    fnames_09 = [
            'tmp-data/VNP09A1.A2022137.h09v05.001.2022145073923.h5',
            'tmp-data/VNP09A1.A2022137.h10v05.001.2022145074734.h5',
            'tmp-data/VNP09A1.A2022137.h11v05.001.2022145074322.h5'
            ]
    f09 = er3t.util.viirs.viirs_09a1(fnames=fnames_09, extent=extent)
    #\-----------------------------------------------------------------------------\

    lon_2d, lat_2d, rad_2d = grid_by_extent(f02.data['lon']['data'], f02.data['lat']['data'], f02.data['rad']['data'].filled(fill_value=np.nan), extent=extent)
    # lon_2d, lat_2d, ref_2d = grid_by_extent(f02.data['lon']['data'], f02.data['lat']['data'], f02.data['ref']['data'].filled(fill_value=np.nan), extent=extent, method='cubic')
    logic = np.logical_not(np.isnan(f02.data['ref']['data'].filled(fill_value=np.nan)))
    lon_2d, lat_2d, ref_2d = grid_by_extent(f02.data['lon']['data'][logic], f02.data['lat']['data'][logic], f02.data['ref']['data'][logic], extent=extent, method='linear')
    lon_2d, lat_2d, sfc_2d = grid_by_extent(f09.data['lon']['data'], f09.data['lat']['data'], f09.data['ref']['data'].filled(fill_value=np.nan), extent=extent, NxNy=lon_2d.shape)

    #/---------------------------------------------------------------------------\
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(221)

    img = mpl_img.imread('tmp-data/VIIRS-SNPP_rgb_2022-05-18_(-94.26,-87.21,31.86,38.91).png')
    ax1.imshow(img, extent=extent)
    ax1.set_xlabel('Longitude [$^\circ$]')
    ax1.set_ylabel('Latitude [$^\circ$]')
    ax1.set_title('VIIRS (Suomi NPP) RGB')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', '5%', pad='3%')
    cax.axis('off')

    ax2 = fig.add_subplot(222)
    cs  = ax2.imshow(sfc_2d.T, origin='lower', extent=extent, cmap='jet', vmin=0.0, vmax=0.5)
    ax2.set_xlabel('Longitude [$^\circ$]')
    ax2.set_ylabel('Latitude [$^\circ$]')
    ax2.set_title('09A1 Surface Ref. (Band M4, 555 nm)')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(cs, cax=cax)

    ax3 = fig.add_subplot(223)
    cs = ax3.imshow(ref_2d.T, origin='lower', extent=extent, cmap='jet', vmin=0.0, vmax=1.0)
    ax3.set_xlabel('Longitude [$^\circ$]')
    ax3.set_ylabel('Latitude [$^\circ$]')
    ax3.set_title('L1B Ref. (Band M4, 555 nm)')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(cs, cax=cax)

    ax4 = fig.add_subplot(224)
    cs = ax4.imshow((ref_2d-sfc_2d).T, origin='lower', extent=extent, cmap='bwr', vmin=-0.3, vmax=0.3)
    ax4.set_xlabel('Longitude [$^\circ$]')
    ax4.set_ylabel('Latitude [$^\circ$]')
    ax4.set_title('Difference (L1B - 09A1)')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(cs, cax=cax)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    plt.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)

    plt.show()
    exit()
    #\---------------------------------------------------------------------------/

    pass

def test_grid_by_dxdy():

    extent_lonlat = [125.0, 127.0, 35.0, 37.0] # china
    # extent_lonlat = [125.0, 127.0, 15.0, 17.0] # philippine sea
    # extent_lonlat = [125.0, 127.0, -1.0, 1.0]  # equator
    # extent_lonlat = [125.0, 127.0, 85.0, 87.0] # arctic region

    lon_1d = np.linspace(extent_lonlat[0], extent_lonlat[1], 201)
    lat_1d = np.linspace(extent_lonlat[2], extent_lonlat[3], 201)

    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')

    data_2d = lon_2d**2 + lat_2d**2

    lon_2d, lat_2d, data_2d0 = er3t.util.grid_by_dxdy(lon_2d, lat_2d, data_2d)

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
        ax1.scatter(lon_2d, lat_2d, s=6, c='k', lw=0.0)
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

    print(lon_2d[:, 0])
    print(lat_2d[0, :])

def test_allocate_jobs():

    import psutil
    import multiprocessing as mp

    weights0 = np.array([
        14824075, 14483931, 13811633, 12822922, 11540979, \
        9995858,  8223786,   6266341, 4349287,   752063,
        676158,   599970,    523397,   446830,   363135,   319635])

    weights_in = np.tile(weights0, 3)
    print(weights_in)

    indices_out = er3t.dev.rearrange_jobs(5, weights_in)
    print(indices_out)

    pass


if __name__ == '__main__':

    # test_modis()

    # test_download_worldview() # passed test on 2022-08-19

    # test_download_laads()

    # test_solar_spectra('tmp-data/abs_16g')

    # test_grid_by_dxdy()

    test_allocate_jobs()
