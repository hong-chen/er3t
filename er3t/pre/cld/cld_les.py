import os
import sys
import pickle
import warnings
import numpy as np

from er3t.util import mmr2vmr, cal_rho_air, downscale



__all__ = ['cld_les']



class cld_les:

    """
    Input:
        fname_nc= : keyword argument, default=None, the file path of the original netCDF4 file
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        coarsen=  : keyword argument, default=[1, 1, 1], the parameter to downscale the data in [x, y, z]
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.lay
                ['x']
                ['y']
                ['altitude']
                ['pressure']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)
                ['cot']           (x, y, z)
                ['cer']           (x, y, z)

        self.lev
                ['altitude']
    """


    ID = 'LES Cloud 3D'


    def __init__(self, \
                 fname_nc  = None, \
                 fname     = None, \
                 altitude  = None, \
                 coarsen   = [1, 1, 1], \
                 overwrite = False, \
                 verbose   = True):

        self.verbose = verbose     # verbose tag
        self.fname   = fname       # file name of the pickle file
        self.coarsen  = coarsen    # (dn_x, dn_y, dn_z, dn_t)
        self.altitude = altitude   # in km

        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((fname_nc is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((fname_nc is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run(fname_nc)
            self.dump(self.fname)

        elif ((fname_nc is not None) and (self.fname is None)):

            self.run(fname_nc)

        else:

            msg = 'Error [cld_les]: Please check if <%s> exists or provide <fname_nc> to proceed.' % self.fname
            raise OSError(msg)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            try:
                file_correct = (obj.ID == 'LES Cloud 3D')
            except:
                file_correct = False

            if file_correct:
                if self.verbose:
                    print('Message [cld_les]: Loading <%s> ...' % fname)
                self.fname = obj.fname
                self.lay   = obj.lay
                self.lev   = obj.lev
            else:
                msg = 'Error [cld_les]: <%s> is not the correct pickle file to load.' % fname
                raise OSError(msg)


    def run(self, fname_nc):

        if self.verbose:
            print('Message [cld_les]: Processing <%s> ...' % fname_nc)

        # pre process
        self.pre_les(fname_nc)

        # post process
        self.post_les(self.altitude)


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_les]: Saving object into <%s> ...' % fname)
            pickle.dump(self, f)


    def pre_les(self, fname_nc, q_factor=2, index_t=0):

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Error [cld_les]: Please install <netCDF4> to proceed.'
            raise ImportError(msg)

        # read data
        #/----------------------------------------------------------------------------\#
        f = Dataset(fname_nc, 'r')

        x      = f.variables['x'][:]/1000.0        # x direction (in km)
        y      = f.variables['y'][:]/1000.0        # y direction (in km)
        z0     = f.variables['z'][:]/1000.0        # z direction, altitude (in km)
        qc_3d  = f.variables['QC'][index_t, ...]   # cloud water mixing ratio

        # in vertical dimension, only select data where clouds exist to shrink data size
        # and accelerate calculation
        #/--------------------------------------------------------------\#
        Nz0 = z0.size

        qc_z = np.sum(qc_3d, axis=(1, 2))
        index_e = -1
        while (qc_z[index_e-2] < 1e-10) and (Nz0+index_e>1):
            index_e -= 1

        if self.coarsen[2] > 1:
            index_e = min(self.coarsen[2]*((Nz0+index_e)//self.coarsen[2]+1), Nz0)

        z      = z0[:index_e]   # z direction, altitude
        qc_3d  = qc_3d[:index_e, :, :]

        p      = f.variables['p'][:index_e]                   # pressure
        qr_3d  = f.variables['QR'][index_t, :index_e, :, :]   # rain water mixing ratio
        qv_3d  = f.variables['QV'][index_t, :index_e, :, :]   # water vapor
        cer_3d = f.variables['REL'][index_t, :index_e, :, :]  # cloud effective radius
        Nc_3d  = f.variables['NC'][index_t, :index_e, :, :]   # cloud droplet number concentration
        t_3d   = f.variables['TABS'][index_t, :index_e, :, :] # absolute temperature
        #\--------------------------------------------------------------/#

        f.close()
        #\----------------------------------------------------------------------------/#

        # check whether the data is equidistant or non-equidistant
        #/----------------------------------------------------------------------------\#
        dz  = z[1:]-z[:-1]
        diff = np.abs(dz-dz[0])
        if any([i>1e-3 for i in diff]):
            msg = '\nWarning [cld_les]: Altitude is non-equidistant.'
            warnings.warn(msg)
            self.logic_equidist = False
        else:
            self.logic_equidist = True
        #\----------------------------------------------------------------------------/#

        Nz, Ny, Nx = qc_3d.shape

        # 3d pressure field
        #/--------------------------------------------------------------\#
        p_3d      = np.empty((Nz, Ny, Nx), dtype=p.dtype)
        p_3d[...] = p[:, None, None]
        #\--------------------------------------------------------------/#

        # calculate cloud extinction
        #/--------------------------------------------------------------\#
        # water vapor volume mixing ratio (from mass mixing ratio, kg/kg)
        vmr_3d = mmr2vmr(qv_3d * 0.001)

        # air density (humid air, kg/m3)
        rho_3d = cal_rho_air(p_3d, t_3d, vmr_3d)

        # liquid water content (kg/m3)
        lwc_3d = qc_3d * 0.001 * rho_3d

        # grid cells that are cloudy
        logic = (Nc_3d>=1) & (cer_3d>0.0)
        cer_3d[np.logical_not(logic)] = 0.0

        # extinction coefficients (m^-1)
        const0        = 0.75*q_factor/(1000.0*1e-6)
        ext_3d        = np.zeros_like(t_3d)
        ext_3d[logic] = const0 / cer_3d[logic] * lwc_3d[logic]
        cer_3d[np.logical_not(logic)] = 0.0
        #\--------------------------------------------------------------/#

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        # layer property
        #/--------------------------------------------------------------\#
        self.lay = {}

        self.lay['x']           = {'data':x             , 'name':'X' , 'units':'km'}
        self.lay['y']           = {'data':y             , 'name':'Y' , 'units':'km'}
        self.lay['nx']          = {'data':x.size        , 'name':'Nx', 'units':'N/A'}
        self.lay['ny']          = {'data':y.size        , 'name':'Ny', 'units':'N/A'}
        self.lay['dx']          = {'data':abs(x[1]-x[0]), 'name':'dx', 'units':'km'}
        self.lay['dy']          = {'data':abs(y[1]-y[0]), 'name':'dy', 'units':'km'}

        self.lay['altitude']    = {'data':z, 'name':'Altitude'   , 'units':'km'}
        self.lay['pressure']    = {'data':p, 'name':'Pressure'   , 'units':'mb'}

        dz = np.append(dz, dz[-1])
        self.lay['thickness']   = {'data':dz, 'name':'Layer thickness', 'units':'km'}

        self.lay['temperature'] = {'data':t_3d  , 'name':'Temperature (3D)'            , 'units':'K'}
        self.lay['extinction']  = {'data':ext_3d, 'name':'Extinction coefficients (3D)', 'units':'m^-1'}
        self.lay['cer']         = {'data':cer_3d, 'name':'Cloud effective radius (3D)' , 'units':'mm'}
        #\--------------------------------------------------------------/#


        # level property
        #/--------------------------------------------------------------\#
        self.lev = {}

        # in km
        zs_ = max(self.lay['altitude']['data'][0]-self.lay['thickness']['data'][0]/2.0, 0.0)
        zm_ = self.lay['altitude']['data'][1:-1] - self.lay['thickness']['data'][1:-1]/2.0
        ze_ = self.lay['altitude']['data'][-1] + self.lay['thickness']['data'][-1]/2.0
        z_ = np.append(np.append(zs_, zm_), ze_)

        dz_ = z_[1:] - z_[:-1]
        dz_ = np.append(dz_, dz_[-1])
        self.lev['altitude']  = {'data':z_ , 'name':'Altitude'       , 'units':'km'}
        self.lev['thickness'] = {'data':dz_, 'name':'Layer thickness', 'units':'km'}
        #\--------------------------------------------------------------/#


    def post_les(self, altitude):

        """
        altitude: re-mapping cloud properties at given level-altitudes (under development, unavailable yet)
        """

        # transpose netCDF data from (Nz, Ny, Nx) to (Nx, Ny, Nz)
        #/----------------------------------------------------------------------------\#
        for key in self.lay.keys():
            if isinstance(self.lay[key]['data'], np.ndarray):
                if self.lay[key]['data'].ndim == 3:
                    self.lay[key]['data'] = np.transpose(self.lay[key]['data'])
        #\----------------------------------------------------------------------------/#

        # downscale (usually in vertical z dimension)
        #/----------------------------------------------------------------------------\#
        if any([i!=1 for i in self.coarsen]):
            self.downscale(self.coarsen)
        #\----------------------------------------------------------------------------/#

        # cloud optical thickness
        #/----------------------------------------------------------------------------\#
        cot_3d = np.zeros_like(self.lay['extinction']['data'])
        dz_    = self.lay['thickness']['data']
        for i, dz0 in enumerate(dz_):
            cot_3d[:, :, i] = self.lay['extinction']['data'][:, :, i] * dz0 * 1000.0
        self.lay['cot'] = {'data':cot_3d, 'name':'Cloud optical thickness (3D)'}

        self.lev['cot_2d'] = {'data': np.sum(cot_3d, axis=-1), 'name': 'Cloud optical thickness (2D)'}
        #\----------------------------------------------------------------------------/#

        # placeholder for <altitude> implementation
        #/----------------------------------------------------------------------------\#
        #\----------------------------------------------------------------------------/#


    def downscale(self, coarsen):

        # extract downscaling factors of dnx, dny, dnz from <coarsen>
        #/----------------------------------------------------------------------------\#
        if len(coarsen) == 3:
            dnx, dny, dnz = coarsen
        elif len(coarsen) == 4:
            dnx, dny, dnz, dnt = coarsen
            msg = '\nWarning [cld_les]: <coarsen=[dnx, dny, dnz, dnt]> will be deprecated in future release (will only support <coarsen=[dnx, dny, dnz]>)'
            warnings.warn(msg)
        else:
            msg = 'Error [cld_les]: Cannot interpret <coarsen> factors.'
            raise OSError(msg)
        #\----------------------------------------------------------------------------/#

        # downscale in process
        #/----------------------------------------------------------------------------\#
        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or (self.Nz%dnz != 0):
            msg = 'Error [cld_les]: The original dimension %s is not divisible with %s, please check input (dnx, dny, dnz, dnt).' % (str(self.lay['temperature']['data'].shape), str(coarsen))
            raise ValueError(msg)

        else:
            new_shape = (self.Nx//dnx, self.Ny//dny, self.Nz//dnz)

            if self.verbose:
                print('Message [cld_les]: Downscaling data from dimension %s to %s ...' % (str(self.lay['temperature']['data'].shape), str(new_shape)))

            self.lay['x']['data']         = downscale(self.lay['x']['data']        , (self.Nx//dnx,), operation='mean')
            self.lay['y']['data']         = downscale(self.lay['y']['data']        , (self.Ny//dny,), operation='mean')
            self.lay['altitude']['data']  = downscale(self.lay['altitude']['data'] , (self.Nz//dnz,), operation='mean')
            self.lay['pressure']['data']  = downscale(self.lay['pressure']['data'] , (self.Nz//dnz,), operation='mean')
            self.lay['thickness']['data'] = downscale(self.lay['thickness']['data'], (self.Nz//dnz,), operation='sum')

            self.lay['dx']['data'] *= dnx
            self.lay['dy']['data'] *= dny
            self.lay['nx']['data'] = self.Nx//dnx
            self.lay['ny']['data'] = self.Ny//dny

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim >= 3:
                        self.lay[key]['data']  = downscale(self.lay[key]['data'], new_shape, operation='mean')

            self.Nx = self.lay['nx']['data']
            self.Ny = self.lay['ny']['data']
        #\----------------------------------------------------------------------------/#


    def get_cloud_mask(self):

        """
        cloud mask
        """

        # cloud mask based on cer
        #/----------------------------------------------------------------------------\#
        cld_msk_3d = np.zeros(self.lay['cer']['data'].shape, dtype=np.int32)
        cld_msk_3d[self.lay['cer']['data']>0] = 1

        cld_msk_2d = np.sum(cld_msk_3d, axis=-1)
        cld_msk_2d[cld_msk_2d>0] = 1
        #\----------------------------------------------------------------------------/#





if __name__ == '__main__':

    pass
