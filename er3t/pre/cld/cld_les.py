import os
import sys
import pickle
import numpy as np

from er3t.util import mmr2vmr, cal_rho_air, downgrading



__all__ = ['cld_les']



class cld_les:

    """
    Input:
        fname_nc= : keyword argument, default=None, the file path of the original netCDF4 file
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        coarsing= : keyword argument, default=[1, 1, 1, 1], the parameter to downgrade the data in [x, y, z, t]
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
                 altitude  = None,
                 coarsing  = [1, 1, 1, 1], \
                 overwrite = False, \
                 verbose   = False):

        self.verbose = verbose     # verbose tag
        self.fname   = fname       # file name of the pickle file
        self.coarsing = coarsing   # (dn_x, dn_y, dn_z, dn_t)
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

            sys.exit('Error   [cld_les]: Please check if \'%s\' exists or provide \'fname_nc\' to proceed.' % self.fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [cld_les]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.lay   = obj.lay
                self.lev   = obj.lev
            else:
                sys.exit('Error   [cld_les]: %s is not the correct \'pickle\' file to load.' % fname)


    def run(self, fname_nc):

        if self.verbose:
            print("Message [cld_les]: Processing %s ..." % fname_nc)

        # pre process
        self.pre_les(fname_nc, altitude=self.altitude)

        # downgrade data if needed
        if any([i!=1 for i in self.coarsing]):
            self.downgrade(self.coarsing)

        # post process
        self.post_les()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_les]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def pre_les(self, fname_nc, q_factor=2, altitude=None):

        self.lay = {}
        self.lev = {}

        try:
            from netCDF4 import Dataset
        except ImportError:
            msg = 'Warning [cld_les.py]: To use \'cld_les.py\', \'netCDF4\' needs to be installed.'
            raise ImportError(msg)

        f = Dataset(fname_nc, 'r')
        time   = f.variables['time'][...]
        x      = f.variables['x'][...]
        y      = f.variables['y'][...]
        z      = f.variables['z'][...]
        p      = f.variables['p'][...]
        t_3d   = f.variables['TABS'][...]
        qc_3d  = f.variables['QC'][...]
        qr_3d  = f.variables['QR'][...]
        qv_3d  = f.variables['QV'][...]
        cer_3d = f.variables['REL'][...]   # cloud effective radius
        Nc_3d  = f.variables['NC'][...]    # cloud droplet number concentration
        f.close()

        self.lay['x']           = {'data':x/1000.0             , 'name':'X'          , 'units':'km'}
        self.lay['y']           = {'data':y/1000.0             , 'name':'Y'          , 'units':'km'}
        self.lay['nx']          = {'data':x.size               , 'name':'Nx'         , 'units':'N/A'}
        self.lay['ny']          = {'data':y.size               , 'name':'Ny'         , 'units':'N/A'}
        self.lay['dx']          = {'data':abs(x[1]-x[0])/1000.0, 'name':'dx'         , 'units':'km'}
        self.lay['dy']          = {'data':abs(y[1]-y[0])/1000.0, 'name':'dy'         , 'units':'km'}

        Nx = x.size
        Ny = y.size
        Nz = z.size
        Nt = time.size

        # 3d pressure field
        p_3d      = np.empty((Nt, Nz, Ny, Nx), dtype=p.dtype)
        p_3d[...] = p[None, :, None, None]

        # water vapor volume mixing ratio (from mass mixing ratio, kg/kg)
        vmr_3d = mmr2vmr(qv_3d * 0.001)

        # air density (humid air, kg/m3)
        rho_3d = cal_rho_air(p_3d, t_3d, vmr_3d)

        # liquid water content (kg/m3)
        lwc_3d = qc_3d * 0.001 * rho_3d

        # extinction coefficients
        logic         = (Nc_3d>=1) & (cer_3d>0.0)
        ext_3d        = np.zeros_like(t_3d)
        ext_3d[logic] = 0.75 * q_factor / (1000.0*cer_3d[logic]*1e-6) * lwc_3d[logic]
        cer_3d[np.logical_not(logic)] = 0.0

        if altitude is None:
            self.lay['altitude']    = {'data':z/1000.0, 'name':'Altitude'   , 'units':'km'}
            self.lay['pressure']    = {'data':p       , 'name':'Pressure'   , 'units':'mb'}
            self.lay['temperature'] = {'data':t_3d    , 'name':'Temperature', 'units':'K'}
            self.lay['extinction']  = {'data':ext_3d  , 'name':'Extinction coefficients', 'units':'m^-1'}

            self.lay['cer']          = {'data':cer_3d  , 'name':'Effective radius', 'units':'mm'}

            self.Nx = Nx
            self.Ny = Ny
            self.Nz = Nz
            self.Nt = Nt

            dz  = self.lay['altitude']['data'][1:]-self.lay['altitude']['data'][:-1]
            dz0 = dz[0]
            diff = np.abs(dz-dz0)
            if any([i>0.001 for i in diff]):
                print('Warning [cld_les]: Non-equidistant intervals found in \'dz\'.')
            dz  = np.append(dz, dz0)
            alt = np.append(self.lay['altitude']['data']-dz0/2.0, self.lay['altitude']['data'][-1]+dz0/2.0)
            self.lev['altitude']    = {'data':alt, 'name':'Altitude'       , 'units':'km'}
            self.lay['thickness']   = {'data':dz , 'name':'Layer thickness', 'units':'km'}

        else:
            z_km = z/1000.0

            while altitude.min() > z_km.min():
                altitude = np.append(altitude[0]-(altitude[1]-altitude[0]), altitude)
            altitude[altitude<0.0] = 0.0

            if altitude.max() < z_km.max():
                sys.exit('Error [cld_les]: The highest level of LES is higher than the highest level of given altitude.')

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
                print('Warning [cld_les]: Non-equidistant intervals found in \'dz\'.')
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
                print('Warning [cld_les]: Non-equidistant intervals found in \'dz_\'.')
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


    def downgrade(self, coarsing):

        dnx, dny, dnz, dnt = coarsing

        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or \
           (self.Nz%dnz != 0) or (self.Nt%dnt != 0):
            sys.exit('Error   [cld_les]: The original dimension %s is not divisible with %s, please check input (dnx, dny, dnz, dnt).' % (str(self.lay['Temperature'].shape), str(coarsing)))
        else:
            new_shape = (self.Nt//dnt, self.Nz//dnz, self.Ny//dny, self.Nx//dnx)

            if self.verbose:
                print('Message [cld_les]: Downgrading data from dimension %s to %s ...' % (str(self.P.shape), str(new_shape)))

            self.lay['x']['data']         = downgrading(self.lay['x']['data']        , (self.Nx//dnx,))
            self.lay['y']['data']         = downgrading(self.lay['y']['data']        , (self.Ny//dny,))
            self.lay['altitude']['data']  = downgrading(self.lay['altitude']['data'] , (self.Nz//dnz,))
            self.lay['pressure']['data']  = downgrading(self.lay['pressure']['data'] , (self.Nz//dnz,))
            self.lay['thickness']['data'] = downgrading(self.lay['thickness']['data'], (self.Nz//dnz,))

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsing):
                        self.lay[key]['data']  = downgrading(self.lay[key]['data'], new_shape)


    def post_les(self):

        for key in self.lay.keys():

            if isinstance(self.lay[key]['data'], np.ndarray):
                if self.lay[key]['data'].ndim == 4:
                    self.lay[key]['data']  = np.transpose(self.lay[key]['data'])[:, :, :, 0]

        cot_3d = np.zeros_like(self.lay['extinction']['data'])
        for i, dz in enumerate(self.lay['thickness']['data']):
            cot_3d[:, :, i] = self.lay['extinction']['data'][:, :, i] * dz * 1000.0

        self.lay['cot'] = {'data':cot_3d, 'name':'Cloud optical thickness'}


if __name__ == '__main__':

    pass
