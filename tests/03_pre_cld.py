import os
import datetime
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from er3t.util import cal_r_twostream, cal_cth_hist
from er3t.util.modis import modis_l1b, modis_l2, modis_09a1, download_modis_https, grid_modis_by_extent
from er3t.pre.cld import cld_les, cld_sev, cld_mod, cld_sat
from er3t.pre.atm import atm_atmmod



def test_cld_les(fdir):

    fname_nc  = 'data/les.nc'
    fname_les = '%s/les.pk' % fdir

    cld0 = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=True)



def test_cld_seviri(fdir):

    atm0 = atm_atmmod(levels=np.arange(21))

    fname_h4     = 'data/seviri.hdf'
    fname_seviri = '%s/seviri.pk' % fdir


    cld0 = cld_sev(fname_h4=fname_h4, fname=fname_seviri, extent=np.array([8.0, 10.0, -18.5, -13.5]), coarsing=[2, 2, 1], overwrite=True)
    # cld0 = cld_sev(fname_h4=fname_h4, fname=fname_seviri, atm_obj=atm0, extent=np.array([8.0, 10.0, -18.5, -13.5]), coarsing=[1, 1, 1], overwrite=True)

    for key in cld0.lay.keys():
        print(key, cld0.lay[key])
        print()

    ext_2d = np.sum(cld0.lay['extinction']['data'], axis=-1)
    print(ext_2d.min())
    print(ext_2d.max())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.imshow(np.transpose(ext_2d))
    # ax1.set_xlim(())
    # ax1.set_ylim(())
    # ax1.set_xlabel('')
    # ax1.set_ylabel('')
    # ax1.set_title('')
    # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    # plt.savefig('test.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------



def test_download_modis(fdir):

    date = datetime.datetime(2017, 8, 25)

    dataset_tags = ['61/MYD02QKM', '61/MYD03', '61/MYD06_L2']

    filename_tag = '.2035.'

    for dataset_tag in dataset_tags:
        download_modis_https(date, dataset_tag, filename_tag, day_interval=1, run=False)



def test_cld_modis_l1b(fdir):

    atm0 = atm_atmmod(levels=np.arange(21))

    extent      = [-112.0, -111.0, 29.4, 30.4]
    fname_l1b   = 'data/MYD02QKM.A2017237.2035.061.2018034164525.hdf'
    fname_l2    = 'data/MYD06_L2.A2017237.2035.061.2018038220417.hdf'
    fname_cld   = '%s/modis.pk' % fdir

    l1b = modis_l1b(fname=fname_l1b, extent=extent)
    l2  = modis_l2(fname=fname_l2, extent=extent, vnames=['Solar_Zenith', 'Solar_Azimuth', 'Sensor_Zenith', 'Sensor_Azimuth'])
    print(l2.data['solar_zenith']['data'].mean(), l2.data['solar_azimuth']['data'].mean())
    exit()


    lon_2d, lat_2d, ref_2d = grid_modis_by_extent(l1b.data['lon']['data'], l1b.data['lat']['data'], l1b.data['ref']['data'][0, ...], extent=extent)

    a0        = np.median(ref_2d)
    threshold = a0 * 2.0

    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=np.cos(np.deg2rad(l2.data['solar_zenith']['data'].mean())))

    indices   = np.where(ref_2d>threshold)
    indices_x = indices[0]
    indices_y = indices[1]

    cot_2d = np.zeros_like(ref_2d)
    cer_2d = np.zeros_like(ref_2d); cer_2d[...] = 1.0
    for i in range(indices_x.size):
        cot_2d[indices_x[i], indices_y[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
        cer_2d[indices_x[i], indices_y[i]] = 12.0

    l1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees', data=lon_2d)
    l1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees', data=lat_2d)

    l1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'    , data=cot_2d)
    l1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'  , data=cer_2d)

    cld0 = cld_sat(sat_obj=l1b, fname=fname_cld, overwrite=True)



def test_cld_modis_l2(fdir):

    atm0 = atm_atmmod(levels=np.arange(21))

    extent      = [-112.0, -111.0, 29.4, 30.4]
    fname_l1b   = 'data/MYD02QKM.A2017237.2035.061.2018034164525.hdf'
    fname_l2    = 'data/MYD06_L2.A2017237.2035.061.2018038220417.hdf'
    fname_cld   = '%s/modis.pk' % fdir

    # l1b = modis_l1b(fname=fname_l1b, extent=extent)
    l2 = modis_l2(fname=fname_l2, vnames=['Solar_Zenith', 'Solar_Azimuth', 'Sensor_Zenith', 'Sensor_Azimuth', 'Cloud_Top_Height'])



    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # fig = plt.figure(figsize=(8, 6))
    # ax1 = fig.add_subplot(111)
    # ax1.hist(l2.data['cloud_top_height']['data']/1000.0, 100)
    # ax1.set_xlim((0.0, 12.0))
    # # ax1.set_ylim(())
    # # ax1.set_xlabel('')
    # # ax1.set_ylabel('')
    # # ax1.set_title('')
    # # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    # # plt.savefig('test.png')
    # plt.show()
    # exit()
    # # ---------------------------------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 10))
    # logic = l2.data['pcl']['data']==1
    # ax1.scatter(l2.data['lon']['data'][logic], l2.data['lat']['data'][logic], c=l2.data['cot']['data'][logic], s=0.1)
    # ax1 = fig.add_subplot(122)
    # logic = l2.data['pcl']['data']==0
    ax1 = fig.add_subplot(221)
    cs1 = ax1.scatter(l2.data['lon_5km']['data'], l2.data['lat_5km']['data'], c=l2.data['solar_zenith']['data'], s=0.1)
    plt.colorbar(cs1)
    ax1.set_title('Solar Zenith Angle')

    ax2 = fig.add_subplot(222)
    angle = l2.data['solar_azimuth']['data']
    angle[angle<0.0] += 360.0
    cs2 = ax2.scatter(l2.data['lon_5km']['data'], l2.data['lat_5km']['data'], c=angle, s=0.1)
    plt.colorbar(cs2)
    ax2.set_title('Solar Azimuth Angle')

    ax3 = fig.add_subplot(223)
    cs3 = ax3.scatter(l2.data['lon_5km']['data'], l2.data['lat_5km']['data'], c=l2.data['sensor_zenith']['data'], s=0.1)
    plt.colorbar(cs3)
    ax3.set_title('Sensor Zenith Angle')

    ax4 = fig.add_subplot(224)
    angle = l2.data['sensor_azimuth']['data']
    # angle[angle<0.0] += 360.0
    # ax4.hist(angle.ravel(), 100)
    cs4 = ax4.scatter(l2.data['lon_5km']['data'], l2.data['lat_5km']['data'], c=angle, s=0.1)
    plt.colorbar(cs4)
    ax4.set_title('Sensor Azimuth Angle')
    plt.savefig('geometry.png', bbox_inches='tight')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

    print(l2.data)
    exit()

    # cld0 = cld_mod(fname_h4=fname_h4, fname=fname_modis, extent=np.array([8.0, 10.0, -18.5, -13.5]), coarsing=[1, 1, 1], overwrite=True)
    # cld0 = cld_mod(fname_h4=fname_h4, fname=fname_modis, extent=None, coarsing=[1, 1, 1], overwrite=True)

    # for key in cld0.lay.keys():
    #     print(key, cld0.lay[key])
    #     print()

    ext_2d = np.sum(cld0.lay['extinction']['data'], axis=-1)
    print(ext_2d.shape)
    # print(ext_2d.min())
    # print(ext_2d.max())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.imshow(np.transpose(ext_2d))
    # ax1.set_xlim(())
    # ax1.set_ylim(())
    # ax1.set_xlabel('')
    # ax1.set_ylabel('')
    # ax1.set_title('')
    # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    # plt.savefig('test.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------



def test_sfc_modis_09a1(fdir):

    date = datetime.datetime(2017, 8, 13)
    filename_tag = 'h11v05'
    fnames = download_modis_https(date, '6/MOD09A1', filename_tag, day_interval=8, fdir_out='data', run=True)

    extent = [-80.0, -77.5, 32.5, 35.0]
    mod09 = modis_09a1(fnames=fnames, extent=extent)
    lon_2d, lat_2d, surf_ref_2d = grid_modis_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][0, ...], extent=extent)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax1.coastlines(color='r', linewidth=1.0, resolution='10m')
    ax1.set_extent(extent)
    ax1.scatter(mod09.data['lon']['data'], mod09.data['lat']['data'], c=mod09.data['ref']['data'][0, ...], cmap='Greys_r', s=4, vmin=0.0, vmax=0.1, transform=ccrs.PlateCarree())
    # ax1.imshow(surf_ref_2d.T, extent=extent, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.1, transform=ccrs.PlateCarree())

    ax1.set_xticks(np.arange(-80.0, -77.0, 0.5), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(32.5, 35.1, 0.5), crs=ccrs.PlateCarree())

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.grid(linewidth=1.0, color='blue', alpha=0.8, linestyle='--')

    plt.savefig('mod09a1_sfc.png', bbox_inches='tight')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # lon_2d, lat_2d, surf_ref_2d = grid_modis_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][0, ...], extent=extent)
    # mod09.data['alb_2d'] = dict(data=surf_ref_2d, name='Surface albedo', units='N/A')
    # mod09.data['lon_2d'] = dict(data=lon_2d, name='Longitude', units='degrees')
    # mod09.data['lat_2d'] = dict(data=lat_2d, name='Latitude' , units='degrees')

    fname_sfc   = '%s/sfc.pk' % fdir
    sfc0 = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=extent, verbose=True)

    sfc_2d = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/sfc_2d.bin' % fdir)

    tags = ['']



def main():

    # create tmp-data/03 directory if it does not exist
    fdir = os.path.abspath('tmp-data/03')
    if not os.path.exists(fdir):
        os.makedirs(fdir)


    # test LES cloud
    # test_cld_les(fdir)


    # test SEVIRI cloud
    # test_cld_seviri(fdir)


    # test MODIS Level 1B
    # test_download_modis(fdir)
    # test_cld_modis_l1b(fdir)

    # test MODIS Level 2
    # test_cld_modis_l2(fdir)

    test_sfc_modis_09a1(fdir)



if __name__ == '__main__':

    main()
