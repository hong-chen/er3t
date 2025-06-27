import os
import sys
import warnings
import datetime
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import shutil
import urllib.request
from er3t.util import check_equal



__all__ = ['abi_l2']


# reader for ABI (Advanced Baseline Imager)
#╭────────────────────────────────────────────────────────────────────────────╮#

class abi_l2:

    """
    Read ABI level 2 cloud product

    Input:
        fnames=   : keyword argument, default=None, Python list of the file path of the original HDF4 file
        extent=   : keyword argument, default=None, region to be cropped, defined by [westmost, eastmost, southmost, northmost]
        vnames=   : keyword argument, default=[], additional variable names to be read in to self.data
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['lon']
                ['lat']
                ['cot']
                ['cer']
    """


    ID = 'ABI Level 2 Cloud Product'


    def __init__(self, \
                 fnames    = None,  \
                 extent    = None,  \
                 vnames    = [],    \
                 cop_flag  = '',    \
                 overwrite = False, \
                 verbose   = False):

        self.fnames     = fnames      # file name of the pickle file
        self.extent     = extent      # specified region [westmost, eastmost, southmost, northmost]
        self.verbose    = verbose     # verbose tag

        for fname in self.fnames:

            self.read(fname, cop_flag=cop_flag)

            if len(vnames) > 0:
                self.read_vars(fname, vnames=vnames)


    def read(self, fname, cop_flag=''):

        """
        Read cloud optical properties

        self.data
            ['lon']
            ['lat']
            ['cot']
            ['cer']
            ['pcl']
            ['lon_5km']
            ['lat_5km']

        self.logic
        """

        try:
            import netCDF4 as nc4
        except ImportError:
            msg = 'Error [abi_l2]: To use <abi_l2>, <netCDF4> needs to be installed.'
            raise ImportError(msg)

        if len(cop_flag) == 0:
            vname_cot = 'cld_opd_dcomp'
            vname_cer = 'cld_reff_dcomp'
        else:
            vname_cot = 'cld_opd_%s'  % cop_flag
            vname_cer = 'cld_reff_%s' % cop_flag

        f = nc4.Dataset(fname, 'r')

        # lon lat
        lat       = f.variables['latitude'][:]
        lon       = f.variables['longitude'][:]

        cot       = f.variables[vname_cot][:]
        cer       = f.variables[vname_cer][:]

        # 1. If region (extent=) is specified, filter data within the specified region
        # 2. If region (extent=) is not specified, filter invalid data
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.extent is None:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]
        else:
            lon_range = [self.extent[0]-0.01, self.extent[1]+0.01]
            lat_range = [self.extent[2]-0.01, self.extent[3]+0.01]

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & \
                    (lat>=lat_range[0]) & (lat<=lat_range[1]) & \
                    (np.logical_not(np.ma.getmaskarray(lon))) & (np.logical_not(np.ma.getmaskarray(lon))) & \
                    (np.logical_not(np.ma.getmaskarray(cot))) & (np.logical_not(np.ma.getmaskarray(cer)))

        lon       = lon[logic]
        lat       = lat[logic]
        cot       = cot[logic]
        cer       = cer[logic]

        logic_invalid = (cot<0.0) | (cer<=0.0)
        cot[logic_invalid] = 0.0
        cer[logic_invalid] = 1.0

        f.close()
        # -------------------------------------------------------------------------------------------------

        if hasattr(self, 'data'):

            self.logic[fname] = logic

            self.data['lon']   = dict(name='Longitude'                 , data=np.hstack((self.data['lon']['data'], lon    )), units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=np.hstack((self.data['lat']['data'], lat    )), units='degrees')
            self.data['cot']   = dict(name='Cloud optical thickness'   , data=np.hstack((self.data['cot']['data'], cot    )), units='N/A')
            self.data['cer']   = dict(name='Cloud effective radius'    , data=np.hstack((self.data['cer']['data'], cer    )), units='micron')

        else:
            self.logic = {}
            self.logic[fname] = logic

            self.data  = {}
            self.data['lon']   = dict(name='Longitude'                 , data=lon    , units='degrees')
            self.data['lat']   = dict(name='Latitude'                  , data=lat    , units='degrees')
            self.data['cot']   = dict(name='Cloud optical thickness'   , data=cot    , units='N/A')
            self.data['cer']   = dict(name='Cloud effective radius'    , data=cer    , units='micron')


    def read_vars(self, fname, vnames=[], resolution='3km'):

        try:
            import netCDF4 as nc4
        except ImportError:
            msg = 'Error [abi_l2]: To use <abi_l2>, <netCDF4> needs to be installed.'
            raise ImportError(msg)

        logic = self.logic[fname]

        f = nc4.Dataset(fname, 'r')

        for vname in vnames:

            data = f.variables[vname][...][logic]
            if vname.lower() in self.data.keys():
                self.data[vname.lower()] = dict(name=vname, data=np.hstack((self.data[vname.lower()]['data'], data)))
            else:
                self.data[vname.lower()] = dict(name=vname, data=data)

        f.close()

#╰────────────────────────────────────────────────────────────────────────────╯#

if __name__=='__main__':

    pass
