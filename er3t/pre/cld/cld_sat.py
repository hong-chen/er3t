import os
import sys
import pickle
import numpy as np
import copy
from scipy import interpolate

from er3t.util import mmr2vmr, cal_rho_air, downscale, cal_ext, cal_geodesic_dist
from er3t.pre.atm import atm_atmmod



__all__ = ['cld_sat']




class cld_sat:

    """
    Input:
        sat_obj=  : keyword argument, default=None, the satellite object created from modis_l1b, modis_l2, seviri_l2 etc
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        coarsen=  : keyword argument, default=[1, 1, 1], the parameter to downscale the data in [x, y, z]
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.lay
                ['x']
                ['y']
                ['dx']
                ['dy']
                ['nx']
                ['ny']
                ['cot']
                ['cer']
                ['lon']
                ['lat']
                ['altitude']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)

        self.lev
                ['altitude']
    """


    ID = 'Satellite Cloud 3D'


    def __init__(self, \
                 sat_obj   = None, \
                 fname     = None, \
                 extent    = None, \
                 cth       = None, \
                 cgt       = None, \
                 dz        = None, \
                 coarsen   = [1, 1, 1], \
                 overwrite = False, \
                 verbose   = False):

        self.verbose    = verbose     # verbose tag
        self.coarsen    = coarsen     # (dn_x, dn_y, dn_z)

        self.fname      = fname       # file name of the pickle file
        self.extent     = extent
        self.sat        = sat_obj

        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((self.sat is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.sat is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run(cth, cgt, dz)
            self.dump(self.fname)

        elif ((self.sat is not None) and (self.fname is None)):

            self.run(self.sat, cth, cgt, dz)

        else:

            sys.exit('Error   [cld_sat]: Please check if \'%s\' exists or provide \'sat_obj\' to proceed.' % self.fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [cld_sat]: loading %s ...' % fname)
                self.fname  = obj.fname
                self.extent = obj.extent
                self.lay    = obj.lay
                self.lev    = obj.lev
            else:
                sys.exit('Error   [cld_sat]: %s is not the correct \'pickle\' file to load.' % fname)


    def run(self, cth, cgt, dz):

        if cth is None:
            cth = 3.0
            print("Warning [cld_sat]: \'cth\' is not specified, setting \'cth\' to 3km ...")

        if cgt is None:
            cgt = 1.0
            print("Warning [cld_sat]: \'cgt\' is not specified, setting \'cgt\' to 1km ...")

        if dz is None:
            dz = 1.0
            print("Warning [cld_sat]: \'dz\' is not specified, setting \'dz\' to 1km ...")

        # process
        self.process(self.sat, cloud_top_height=cth, cloud_geometrical_thickness=cgt, layer_thickness=dz)

        # downscale data if needed
        if any([i!=1 for i in self.coarsen]):
            self.downscale(self.coarsen)


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_sat]: saving object into %s ...' % fname)
            pickle.dump(self, f)


    def process(self, sat_obj, cloud_geometrical_thickness=1.0, cloud_top_height=3.0, layer_thickness=1.0):

        self.lay = {}
        self.lev = {}

        keys = self.sat.data.keys()
        if ('lon_2d' not in keys) or ('lat_2d' not in keys) or ('cot_2d' not in keys) or ('cer_2d' not in keys):
            sys.exit('Error   [cld_sat]: Please make sure \'sat_obj.data\' contains \'lon_2d\', \'lat_2d\', \'cot_2d\' and \'cer_2d\'.')

        cloud_bottom_height = cloud_top_height-cloud_geometrical_thickness

        if isinstance(cloud_top_height, np.ndarray):
            if cloud_top_height.shape != self.sat.data['cot_2d']['data'].shape:
                sys.exit('Error   [cld_sat]: The dimension of \'cloud_top_height\' does not match \'lon_2d\', \'lat_2d\', \'cot_2d\' and \'cer_2d\'.')

            cloud_bottom_height[cloud_bottom_height<layer_thickness] = layer_thickness
            h_bottom = max([np.nanmin(cloud_bottom_height), layer_thickness])
            h_top    = min([np.nanmax(cloud_top_height), 30.0])

        else:

            if cloud_bottom_height < layer_thickness:
                cloud_bottom_height = layer_thickness
            h_bottom = max([cloud_bottom_height, layer_thickness])
            h_top    = min([cloud_top_height, 30.0])

        if h_bottom >= h_top:
            sys.exit('Error   [cld_sat]: Cloud bottom height is greater than cloud top height, check whether the input cloud top height \'cth\' is in the units of \'km\'.')

        levels   = np.arange(h_bottom, h_top+0.1*layer_thickness, layer_thickness)
        self.atm = atm_atmmod(levels=levels)

        lon_1d = self.sat.data['lon_2d']['data'][:, 0]
        lat_1d = self.sat.data['lat_2d']['data'][0, :]

        if 'dx' not in keys:
            dx = cal_geodesic_dist(
                    self.sat.data['lon_2d']['data'][:-1, :], self.sat.data['lat_2d']['data'][:-1, :], \
                    self.sat.data['lon_2d']['data'][1:, :] , self.sat.data['lat_2d']['data'][1:, :]   \
                    ).mean()
        else:
            dx = self.sat.data['dx']['data']

        if 'dy' not in keys:
            dy = cal_geodesic_dist(
                    self.sat.data['lon_2d']['data'][:, :-1], self.sat.data['lat_2d']['data'][:, :-1], \
                    self.sat.data['lon_2d']['data'][:, 1:] , self.sat.data['lat_2d']['data'][:, 1:]   \
                    ).mean()
        else:
            dy = self.sat.data['dy']['data']

        x_1d = (lon_1d-lon_1d[0])*dx
        y_1d = (lat_1d-lat_1d[0])*dy

        Nx   = x_1d.size
        Ny   = y_1d.size

        self.lay['x']  = {'data':x_1d     , 'name':'X'          , 'units':'km'}
        self.lay['y']  = {'data':y_1d     , 'name':'Y'          , 'units':'km'}
        self.lay['nx'] = {'data':Nx       , 'name':'Nx'         , 'units':'N/A'}
        self.lay['ny'] = {'data':Ny       , 'name':'Ny'         , 'units':'N/A'}
        self.lay['dx'] = {'data':dx       , 'name':'dx'         , 'units':'km'}
        self.lay['dy'] = {'data':dy       , 'name':'dy'         , 'units':'km'}
        self.lay['altitude'] = copy.deepcopy(self.atm.lay['altitude'])
        self.lay['thickness']= copy.deepcopy(self.atm.lay['thickness'])
        self.lay['lon']      = copy.deepcopy(self.sat.data['lon_2d'])
        self.lay['lat']      = copy.deepcopy(self.sat.data['lat_2d'])
        self.lay['cot']      = copy.deepcopy(self.sat.data['cot_2d'])

        self.lev['altitude'] = copy.deepcopy(self.atm.lev['altitude'])

        # temperature 3d
        t_1d = self.atm.lay['temperature']['data']
        Nz   = t_1d.size
        t_3d      = np.empty((Nx, Ny, Nz), dtype=t_1d.dtype)
        t_3d[...] = t_1d[None, None, :]

        cer_2d = self.sat.data['cer_2d']['data']
        cer_3d = np.empty((Nx, Ny, Nz), dtype=cer_2d.dtype)
        cer_3d[...] = cer_2d[:, :, None]

        self.lay['temperature'] = {'data':t_3d, 'name':'Temperature', 'units':'K'}

        # extinction 3d
        ext_3d      = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        alt = self.atm.lay['altitude']['data']

        for i in range(Nx):
            for j in range(Ny):
                if isinstance(cloud_top_height, np.ndarray):
                    cbh0 = cloud_bottom_height[i, j]
                    cth0 = cloud_top_height[i, j]
                else:
                    cbh0 = cloud_bottom_height
                    cth0 = cloud_top_height

                cot0  = self.lay['cot']['data'][i, j]
                cer0  = cer_2d[i, j]
                indices =  np.where((alt>=cbh0) & (alt<=cth0))[0]
                if indices.size == 0:
                    indices = np.array([-1])


                dz    = self.atm.lay['thickness']['data'][indices].sum() * 1000.0
                ext_3d[i, j, indices] = cot0/dz

        ext_3d[np.isnan(ext_3d)] = 0.0
        cer_3d[np.isnan(ext_3d)] = 0.0
        self.lay['extinction']   = {'data':ext_3d, 'name':'Extinction coefficients', 'units':'m^-1'}
        self.lay['cer']          = {'data':cer_3d, 'name':'Effective radius'       , 'units':'mm'}

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz


    def downscale(self, coarsen):

        dnx, dny, dnz = coarsen

        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or \
           (self.Nz%dnz != 0):
            sys.exit('Error   [cld_mod]: the original dimension %s is not divisible with %s, please check input (dnx, dny, dnz).' % (str(self.lay['temperature']['data'].shape), str(coarsen)))
        else:
            new_shape = (self.Nx//dnx, self.Ny//dny, self.Nz//dnz)

            if self.verbose:
                print('Message [cld_mod]: Downscaling data from dimension %s to %s ...' % (str(self.lay['temperature']['data'].shape), str(new_shape)))

            self.lay['x']['data']         = downscale(self.lay['x']['data']       , (self.Nx//dnx,), operation='mean')
            self.lay['y']['data']         = downscale(self.lay['y']['data']       , (self.Ny//dny,), operation='mean')
            self.lay['altitude']['data']  = downscale(self.lay['altitude']['data'], (self.Nz//dnz,), operation='mean')
            self.lay['thickness']['data'] = downscale(self.lay['thickness']['data'], (self.Nz//dnz,), operation='sum')

            self.lay['dx']['data'] *= dnx
            self.lay['dy']['data'] *= dny

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsen):
                        self.lay[key]['data']  = downscale(self.lay[key]['data'], new_shape, operation='mean')





if __name__ == '__main__':

    pass
