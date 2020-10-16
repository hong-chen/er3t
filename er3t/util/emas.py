import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from pyhdf.SD import SD, SDC
import netCDF4 as nc4
import scipy
from scipy import interpolate
from scipy.io import readsav
from pyhdf.SD import SD, SDC
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.patches as patches



def CDATA_eMAS_RGB_20160920_1130_1202(RGB_channels=[9, 5, 2]):

    fname = '/data/hong/work/04_oracles/2016/emas/data/eMASL1B/eMASL1B_16958_10_20160920_1130_1202_V01.hdf'
    f0    = SD(fname, SDC.READ)
    lon   = f0.select('PixelLongitude')[:].ravel()
    lat   = f0.select('PixelLatitude')[:].ravel()
    data  = f0.select('CalibratedData')

    index = RGB_channels[0]
    scale = data.attributes()['scale_factor'][index]
    data0 = (data[:][:, index, :] * scale).ravel()
    R0    = data0/np.nanmax(data0)
    R0[np.isnan(R0)] = -1.0

    index = RGB_channels[1]
    scale = data.attributes()['scale_factor'][index]
    data0 = (data[:][:, index, :] * scale).ravel()
    G0    = data0/np.nanmax(data0)
    G0[np.isnan(G0)] = -1.0

    index = RGB_channels[2]
    scale = data.attributes()['scale_factor'][index]
    data0 = (data[:][:, index, :] * scale).ravel()
    B0    = data0/np.nanmax(data0)
    B0[np.isnan(B0)] = -1.0
    f0.end()

    logic = (lon>=-180.0)&(lon<=180.0)&(lat>=-90.0)&(lat<=90.0)&(R0>=0.0)&(R0<=1.0)&(G0>=0.0)&(G0<=1.0)&(B0>=0.0)&(B0<=1.0)

    lon = lon[logic]
    lat = lat[logic]
    R0  = R0[logic]
    G0  = G0[logic]
    B0  = B0[logic]

    extent = np.array([lon.min(), lon.max(), lat.min(), lat.max()])

    Nx = (extent[1]-extent[0])//0.001 + 1
    Ny = (extent[3]-extent[2])//0.001 + 1
    xx = np.linspace(extent[0], extent[1], Nx)
    yy = np.linspace(extent[2], extent[3], Ny)

    points = np.transpose(np.vstack((lon, lat)))

    XX, YY = np.meshgrid(xx, yy)

    R = scipy.interpolate.griddata(points, R0.ravel(), (XX, YY), method='cubic')
    G = scipy.interpolate.griddata(points, G0.ravel(), (XX, YY), method='cubic')
    B = scipy.interpolate.griddata(points, B0.ravel(), (XX, YY), method='cubic')

    RGB = np.hstack((np.hstack((R, G)), B))
    RGB = RGB.reshape((XX.shape[0], 3, XX.shape[1]))
    RGB = np.swapaxes(RGB, 1, 2)
    RGB[np.isnan(RGB)] = 0.0
    RGB[(RGB<0.0)|(RGB>1.0)] = 0.0

    f = h5py.File('eMAS_20160920_1130_1202_10.h5', 'w')
    f['extent'] = extent
    f['RGB'] = RGB
    f['lon'] = XX
    f['lat'] = YY
    f.close()

def CDATA_eMAS_COP_20160920_1130_1202():

    f = h5py.File('eMAS_20160920_1130_1202_10.h5', 'r+')
    XX = f['lon'][...]
    YY = f['lat'][...]

    fname = '/data/hong/work/04_oracles/2016/emas/data/eMASL2CLD/eMASL2CLD_16958_10_20160920_1130_1202_20180925_0921.nc4'
    f0    = nc4.Dataset(fname, 'r')
    lon   = f0.groups['geolocation_data'].variables['PixelLongitude'][...].ravel()
    lat   = f0.groups['geolocation_data'].variables['PixelLatitude'][...].ravel()
    cot   = f0.groups['geophysical_data'].variables['Cloud_Optical_Thickness_16'][...].ravel()
    cer   = f0.groups['geophysical_data'].variables['Cloud_Effective_Radius_16'][...].ravel()
    f0.close()

    logic = (lon>=-180.0)&(lon<=180.0)&(lat>=-90.0)&(lat<=90.0)&(cot>=0.03)&(cot<=150.0)&(cer>=4.0)&(cer<=60.0)
    lon   = lon[logic]
    lat   = lat[logic]
    cot   = cot[logic]
    cer   = cer[logic]

    points = np.transpose(np.vstack((lon, lat)))

    cot = scipy.interpolate.griddata(points, cot, (XX, YY), method='cubic')
    cer = scipy.interpolate.griddata(points, cer, (XX, YY), method='cubic')

    f['cot_16'] = cot
    f['cer_16'] = cer
    f.close()



class eMAS_L2:

    """
    Read L2 cloud product of eMAS downloaded from
    https://ladsweb.modaps.eosdis.nasa.gov/archive/MAS_eMAS/ORACLES
    """

    def __init__(self, fname, res=0.001):

        self.fname = fname
        self.res   = res

        # self.GRID()

        self.LOAD()

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.imshow(self.cot, cmap='jet', vmin=0.0, vmax=30.0, extent=self.extent, origin='lower')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
        # plt.savefig('test.png')
        plt.show()
        sys.exit()
        # ---------------------------------------------------------------------

    def GRID(self):

        f0    = nc4.Dataset(self.fname, 'r')
        lon  = f0.groups['geolocation_data'].variables['PixelLongitude'][...].ravel()
        lat  = f0.groups['geolocation_data'].variables['PixelLatitude'][...].ravel()
        cot  = f0.groups['geophysical_data'].variables['Cloud_Optical_Thickness_16'][...].ravel()
        cer  = f0.groups['geophysical_data'].variables['Cloud_Effective_Radius_16'][...].ravel()
        cwp  = f0.groups['geophysical_data'].variables['Cloud_Water_Path_16'][...].ravel()
        cth  = f0.groups['geophysical_data'].variables['Cloud_Top_Height'][...].ravel()
        f0.close()

        logic = (lon>=-180.0) & (lon<=180.0) & (lat>=-90.0) & (lat<=90.0) & \
                (cot>=0.03)   & (cot<=150.0) & (cer>=4.0)   & (cer<=60.0) & \
                (cwp>=0.0)    & (cth>=0.0)

        lon = lon[logic]
        lat = lat[logic]
        cot = cot[logic]
        cer = cer[logic]
        cwp = cwp[logic]
        cth = cth[logic]

        extent = (lon.min(), lon.max(), lat.min(), lat.max())

        Nx = (extent[1]-extent[0])//self.res + 1
        Ny = (extent[3]-extent[2])//self.res + 1
        xx = np.linspace(extent[0], extent[1], Nx)
        yy = np.linspace(extent[2], extent[3], Ny)
        XX, YY = np.meshgrid(xx, yy)

        points = np.transpose(np.vstack((lon, lat)))

        cot = scipy.interpolate.griddata(points, cot.ravel(), (XX, YY), method='cubic')
        cer = scipy.interpolate.griddata(points, cer.ravel(), (XX, YY), method='cubic')
        cwp = scipy.interpolate.griddata(points, cwp.ravel(), (XX, YY), method='cubic')
        cth = scipy.interpolate.griddata(points, cth.ravel(), (XX, YY), method='cubic')

        f = h5py.File('emas_20160920_10_grid.h5', 'w')
        f['extent'] = extent
        f['lon']    = XX
        f['lat']    = YY
        f['cot']    = cot
        f['cer']    = cer
        f['cwp']    = cwp
        f['cth']    = cth
        f.close()

    def LOAD(self):

        f = h5py.File('emas_20160920_10_grid.h5', 'r')
        self.lon = f['lon'][...]
        self.lat = f['lat'][...]
        self.cot = f['cot'][...]
        self.cer = f['cer'][...]
        self.cth = f['cth'][...]
        self.cwp = f['cwp'][...]
        self.extent = f['extent'][...]
        f.close()




if __name__ == "__main__":

    fname = '/data/hong/work/04_oracles/2016/emas/data/eMASL2CLD/eMASL2CLD_16958_10_20160920_1130_1202_20180925_0921.nc4'
    cld = eMAS_L2(fname)
