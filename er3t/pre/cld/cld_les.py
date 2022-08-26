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
        coarsen=  : keyword argument, default=[1, 1, 1, 1], the parameter to downscale the data in [x, y, z, t]
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
                 coarsen   = [1, 1, 1, 1], \
                 overwrite = False, \
                 verbose   = False):

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

        # downscale data if needed
        if any([i!=1 for i in self.coarsen]):
            self.downscale(self.coarsen)

        # post process
        self.post_les()


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

        f = Dataset(fname_nc, 'r')

        x      = f.variables['x'][:]               # x direction
        y      = f.variables['y'][:]               # y direction
        z0     = f.variables['z'][:]               # z direction, altitude
        qc_3d  = f.variables['QC'][index_t, ...]   # cloud water mixing ratio

        # in vertical dimension, only select data where has clouds to shrink data size
        # and accelerate calculation
        #/-----------------------------------------------------------------------------/
        Nz0 = z0.size

        qc_z = np.sum(qc_3d, axis=(1, 2))
        index_e = -1
        while (qc_z[index_e-2] < 1e-8) and (Nz0+index_e>1):
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
        #\-----------------------------------------------------------------------------\

        f.close()

        Nz, Ny, Nx = qc_3d.shape

        # 3d pressure field
        #/-----------------------------------------------------------------------------/
        p_3d      = np.empty((Nz, Ny, Nx), dtype=p.dtype)
        p_3d[...] = p[:, None, None]
        #\-----------------------------------------------------------------------------\


        # calculate cloud extinction
        #/-----------------------------------------------------------------------------/
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
        #\-----------------------------------------------------------------------------\


        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        # layer property
        #/-----------------------------------------------------------------------------/
        self.lay = {}

        self.lay['x']           = {'data':x/1000.0             , 'name':'X'          , 'units':'km'}
        self.lay['y']           = {'data':y/1000.0             , 'name':'Y'          , 'units':'km'}
        self.lay['nx']          = {'data':x.size               , 'name':'Nx'         , 'units':'N/A'}
        self.lay['ny']          = {'data':y.size               , 'name':'Ny'         , 'units':'N/A'}
        self.lay['dx']          = {'data':abs(x[1]-x[0])/1000.0, 'name':'dx'         , 'units':'km'}
        self.lay['dy']          = {'data':abs(y[1]-y[0])/1000.0, 'name':'dy'         , 'units':'km'}

        self.lay['altitude']    = {'data':z/1000.0, 'name':'Altitude'   , 'units':'km'}
        self.lay['pressure']    = {'data':p       , 'name':'Pressure'   , 'units':'mb'}

        dz  = self.lay['altitude']['data'][1:]-self.lay['altitude']['data'][:-1]
        diff = np.abs(dz-dz[0])
        if any([i>1e-3 for i in diff]):
            msg = 'Warning [cld_les]: Altitude is non-equidistant.'
            warnings.warn(msg)
        self.lay['thickness']   = {'data':dz , 'name':'Layer thickness', 'units':'km'}

        self.lay['temperature'] = {'data':t_3d  , 'name':'Temperature', 'units':'K'}
        self.lay['extinction']  = {'data':ext_3d, 'name':'Extinction coefficients', 'units':'m^-1'}
        self.lay['cer']         = {'data':cer_3d, 'name':'Cloud effective radius', 'units':'mm'}
        #\-----------------------------------------------------------------------------\


        # level property
        #/-----------------------------------------------------------------------------/
        self.lev = {}

        z_ = np.append(self.lay['altitude']['data']-dz/2.0, self.lay['altitude']['data'][-1]+dz[-1]/2.0)

        if z_[0] < 0.0:
            msg = 'Error [cld_les]: Surface below 0.'
            raise ValueError(msg)

        dz_ = z_[1:] - z_[:-1]
        self.lev['altitude']  = {'data':z_ , 'name':'Altitude'       , 'units':'km'}
        self.lay['thickness'] = {'data':dz_, 'name':'Layer thickness', 'units':'km'}
        #\-----------------------------------------------------------------------------\


        if False:
            z_km = z/1000.0

            while altitude.min() > z_km.min():
                altitude = np.append(altitude[0]-(altitude[1]-altitude[0]), altitude)
            altitude[altitude<0.0] = 0.0

            if altitude.max() < z_km.max():
                msg = 'Error [cld_les]: The highest level of LES is higher than the highest level of given altitude.'
                raise ValueError(msg)

            alt_min0 = altitude[altitude <= z_km.min()].max()
            alt_max0 = altitude[altitude >= z_km.max()].min()
            altitude = altitude[(altitude>=alt_min0)&(altitude<=alt_max0)]

            self.lay['altitude']    = {'data':altitude             , 'name':'Altitude'   , 'units':'km'}

            self.Nx = Nx
            self.Ny = Ny
            self.Nz = self.lay['altitude']['data'].size
            self.Nt = Nt

            # for given altitude
            # ==================================================================================
            dz  = self.lay['altitude']['data'][1:]-self.lay['altitude']['data'][:-1]
            dz0 = dz[0]
            diff = np.abs(dz-dz0)
            if any([i>0.001 for i in diff]):
                msg = 'Warning [cld_les]: Non-equidistant intervals found in <dz>.'
                warnings.warn(msg)
            dz  = np.append(dz, dz0)
            alt = np.append(self.lay['altitude']['data']-dz0/2.0, self.lay['altitude']['data'][-1]+dz0/2.0)
            alt[alt<0.0] = 0.0
            dz  = alt[1:] - alt[:-1]
            self.lev['altitude']    = {'data':alt, 'name':'Altitude'       , 'units':'km'}
            self.lay['thickness']   = {'data':dz , 'name':'Layer thickness', 'units':'km'}
            # ==================================================================================

            # for LES altitude
            # ==================================================================================
            dz_  = z_km[1:]-z_km[:-1]
            dz0_ = dz_[0]
            diff_ = np.abs(dz_-dz0_)
            if any([i>0.001 for i in diff_]):
                msg = 'Warning [cld_les]: Non-equidistant intervals found in <dz_>.'
                warnings.warn(msg)
            dz_  = np.append(dz_, dz0_)
            alt_ = np.append(z_km-dz0_/2.0, z_km[-1]+dz0_/2.0)
            alt_[alt_<0.0] = 0.0
            dz_  = alt_[1:] - alt_[:-1]
            # ==================================================================================

            p_new      = np.zeros(self.Nz, dtype=p.dtype)
            t_3d_new   = np.zeros((self.Nt, self.Nz, self.Ny, self.Nx), dtype=t_3d.dtype)
            ext_3d_new = np.zeros((self.Nt, self.Nz, self.Ny, self.Nx), dtype=t_3d.dtype)
            cer_3d_new = np.zeros((self.Nt, self.Nz, self.Ny, self.Nx), dtype=t_3d.dtype)

            for i in range(self.Nz):
                indices = np.where((z_km>=alt[i]) & (z_km<alt[i+1]))[0]
                logic   = (z_km>=alt[i]) & (z_km<alt[i+1])
                if logic.sum() > 0:
                    p_new[i]               = p[logic].mean()
                    t_3d_new[:, i, :, :]   = np.mean(t_3d[:, logic, :, :], axis=1)
                    cer_3d_new[:, i, :, :] = np.mean(cer_3d[:, logic, :, :], axis=1)

                    ext0 = np.zeros((self.Nt, self.Ny, self.Nx), dtype=ext_3d.dtype)
                    for index in indices:
                        ext0 += ext_3d[:, index, :, :] * dz_[index]
                    ext_3d_new[:, i, :, :] = ext0 / dz[i]

            self.lay['pressure']    = {'data':p_new       , 'name':'Pressure'   , 'units':'mb'}
            self.lay['temperature'] = {'data':t_3d_new    , 'name':'Temperature', 'units':'K'}
            self.lay['extinction']  = {'data':ext_3d_new  , 'name':'Extinction coefficients', 'units':'m^-1'}
            self.lay['cer']         = {'data':cer_3d_new  , 'name':'Cloud effective radius' , 'units':'mm'}


    def post_les(self):

        for key in self.lay.keys():

            if isinstance(self.lay[key]['data'], np.ndarray):
                if self.lay[key]['data'].ndim == 4:
                    self.lay[key]['data']  = np.transpose(self.lay[key]['data'])[:, :, :, 0]

        cot_3d = np.zeros_like(self.lay['extinction']['data'])
        for i, dz in enumerate(self.lay['thickness']['data']):
            cot_3d[:, :, i] = self.lay['extinction']['data'][:, :, i] * dz * 1000.0

        self.lay['cot'] = {'data':cot_3d, 'name':'Cloud optical thickness'}


    def downscale(self, coarsen):

        dnx, dny, dnz, dnt = coarsen

        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or \
           (self.Nz%dnz != 0) or (self.Nt%dnt != 0):
            msg = 'Error [cld_les]: The original dimension %s is not divisible with %s, please check input (dnx, dny, dnz, dnt).' % (str(self.lay['temperature']['data'].shape), str(coarsen))
            raise ValueError(msg)
        else:
            new_shape = (self.Nt//dnt, self.Nz//dnz, self.Ny//dny, self.Nx//dnx)

            if self.verbose:
                print('Message [cld_les]: Downscaling data from dimension %s to %s ...' % (str(self.lay['temperature']['data'].shape), str(new_shape)))

            self.lay['x']['data']         = downscale(self.lay['x']['data']        , (self.Nx//dnx,), operation='mean')
            self.lay['y']['data']         = downscale(self.lay['y']['data']        , (self.Ny//dny,), operation='mean')
            self.lay['altitude']['data']  = downscale(self.lay['altitude']['data'] , (self.Nz//dnz,), operation='mean')
            self.lay['pressure']['data']  = downscale(self.lay['pressure']['data'] , (self.Nz//dnz,), operation='mean')
            self.lay['thickness']['data'] = downscale(self.lay['thickness']['data'], (self.Nz//dnz,), operation='sum')

            self.lay['dx']['data'] *= dnx
            self.lay['dy']['data'] *= dny

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsen):
                        self.lay[key]['data']  = downscale(self.lay[key]['data'], new_shape, operation='mean')



if __name__ == '__main__':

    import er3t.common
    fname_nc = os.path.abspath('%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples)
    cld0 = cld_les(fname_nc=fname_nc, coarsen=[1, 1, 10, 1])
    pass
