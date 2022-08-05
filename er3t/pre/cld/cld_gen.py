import os
import sys
import pickle
import numpy as np


__all__ = ['cld_gen_hem']



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
                print('Message [cld_les]: Downgrading data from dimension %s to %s ...' % (str(self.lay['temperature']['data'].shape), str(new_shape)))

            self.lay['x']['data']         = downgrading(self.lay['x']['data']        , (self.Nx//dnx,), operation='mean')
            self.lay['y']['data']         = downgrading(self.lay['y']['data']        , (self.Ny//dny,), operation='mean')
            self.lay['altitude']['data']  = downgrading(self.lay['altitude']['data'] , (self.Nz//dnz,), operation='mean')
            self.lay['pressure']['data']  = downgrading(self.lay['pressure']['data'] , (self.Nz//dnz,), operation='mean')
            self.lay['thickness']['data'] = downgrading(self.lay['thickness']['data'], (self.Nz//dnz,), operation='sum')

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsing):
                        self.lay[key]['data']  = downgrading(self.lay[key]['data'], new_shape, operation='mean')


    def post_les(self):

        for key in self.lay.keys():

            if isinstance(self.lay[key]['data'], np.ndarray):
                if self.lay[key]['data'].ndim == 4:
                    self.lay[key]['data']  = np.transpose(self.lay[key]['data'])[:, :, :, 0]

        cot_3d = np.zeros_like(self.lay['extinction']['data'])
        for i, dz in enumerate(self.lay['thickness']['data']):
            cot_3d[:, :, i] = self.lay['extinction']['data'][:, :, i] * dz * 1000.0

        self.lay['cot'] = {'data':cot_3d, 'name':'Cloud optical thickness'}



class cld_gen_hem:

    """
    Purpose: generate 3D cloud field (hemispherical clouds)

    Input:
        Nx=: keyword argument, default=400, number of pixel in x of 3D space
        Ny=: keyword argument, default=400, number of pixel in y of 3D space
        dx=: keyword argument, default=100, delta length in x per pixel (units: km)
        dy=: keyword argument, default=100, delta length in y per pixel (units: km)
        radii=      : keyword argument, default=[5000], a pool of radius of clouds that will be radomly picked from (units: meter)
        weights=    : keyword argument, default=None (evenly pick from radii), possibilities for picking the size of clouds specified in radii
        w2h_ratio=  : keyword argmument, default=1.0, width (x) to height (z) ratio, smaller the number, taller the clouds
        min_dist=   : keyword argument, default=0, minimum distance between each two clouds, the larger the number, the more sparse of the cloud fields
        cloud_frac= : keyword argument, default=0.2, target cloud fraction for the generated cloud field
        trial_limit=: keyword argument, default=100, number of trials if the cloud scene is too full to add new clouds
        overlap=    : keyword argument, default=False, whether different clouds can overlap

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

    ID = 'Hemispherical Cloud 3D'

    def __init__(
            self,
            fname=None,
            altitude=np.arange(1.0, 6.1, 0.5),
            Nx=400,
            Ny=400,
            dx=0.1,
            dy=0.1,
            radii=[5.0],
            weights=None,
            w2h_ratio=1.0,
            min_dist=0,
            cloud_frac_tgt=0.2,
            trial_limit=100,
            overlap=False,
            overwrite=False,
            verbose=True
            ):

        self.fname    = fname      # file name of the pickle file
        self.altitude = altitude   # in km

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy

        self.radii     = radii
        self.weights   = weights
        self.w2h_ratio = w2h_ratio
        self.min_dist  = min_dist

        self.trial = 0
        self.trial_limit    = trial_limit
        self.cloud_frac_tgt = cloud_frac_tgt

        self.overlap = overlap
        self.verbose = verbose     # verbose tag

        # check for pickle file
        # =============================================================================
        # if pickle file exists - load the data directly from pickle file
        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        # if pickle file does not exist or overwrite is specified - run the program and save
        # data into pickle file
        elif ((self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.fname is not None) and (not os.path.exists(self.fname))):

            self.run()
            self.dump(self.fname)

        # if pickle file doesn't get specified
        else:

            sys.exit('Error   [cld_gen_hem]: Please specify file name for pickle file. For example,\ncld0 = cld_gen_hem(fname=\'tmp-data/cloud.pk\')')
        # =============================================================================

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [pre.cld.cld_gen_hem]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.lay   = obj.lay
                self.lev   = obj.lev
            else:
                sys.exit('Error   [pre.cld.cld_gen_hem]: %s is not the correct \'pickle\' file to load.' % fname)

    def run(self):

        if self.verbose:
            print('Message [cld_gen_hem]: Generating an artificial 3D cloud field filled with hemispherical clouds...')

        dz = np.unique(self.altitude[1:]-self.altitude[:-1])
        if dz.size > 1:
            sys.exit('Error   [cld_gen_hem]: Only support equidistant altitude (z), as well as equidistant x and y.')

        self.x = np.arange(self.Nx) * self.dx
        self.y = np.arange(self.Ny) * self.dy

        self.z  = self.altitude-self.altitude[0]
        self.dz = dz[0]
        self.Nz = self.z.size

        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y, indexing='ij')
        self.x_3d, self.y_3d, self.z_3d= np.meshgrid(self.x, self.y, self.z, indexing='ij')

        self.clouds   = []
        self.space_3d = np.zeros_like(self.x_3d)
        self.where_2d = np.ones((self.Nx, self.Ny))
        self.can_add_more = True
        self.cloud_frac   = 0.0

        radii = np.array(self.radii)
        if self.weights is not None:
            self.weights = np.array(self.weights)
        while (self.cloud_frac<self.cloud_frac_tgt) and (self.can_add_more):
            self.add_a_cloud(np.random.choice(self.radii, p=self.weights), min_dist=self.min_dist, w2h_ratio=self.w2h_ratio, limit=1)

        self.lay = {}
        self.lev = {}

        self.lay['x']  = {'data':self.x , 'name':'X' , 'units':'km'}
        self.lay['y']  = {'data':self.y , 'name':'Y' , 'units':'km'}
        self.lay['z']  = {'data':self.z , 'name':'Z' , 'units':'km'}
        self.lay['nx'] = {'data':self.Nx, 'name':'Nx', 'units':'N/A'}
        self.lay['ny'] = {'data':self.Ny, 'name':'Ny', 'units':'N/A'}
        self.lay['nz'] = {'data':self.Nz, 'name':'Nz', 'units':'N/A'}
        self.lay['dx'] = {'data':self.dx, 'name':'dx', 'units':'km'}
        self.lay['dy'] = {'data':self.dy, 'name':'dy', 'units':'km'}
        self.lay['dz'] = {'data':self.dz, 'name':'dz', 'units':'km'}

        self.pre_cld_opt_prop()

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_gen_hem]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

    def add_a_cloud(self, radius, min_dist=0, w2h_ratio=1.0, limit=1):

        """
        Purpose: add a cloud into the 3D space

        Input:
            radius   : position argument, radius of the hemispherical cloud (units: km)
            min_dist=: keyword argument, minimum distance between clouds - the larger the value, the more distant away from cloud to cloud (units: km)
            limit=   : keyword argument, when to stop adding more clouds to avoid overlap

        Output:
            1) if a cloud is successfully added,
                i) self.clouds gets updated
               ii) self.cloud_frac gets updated
            2) if not
                i) self.trial gets updated
               ii) self.can_add_more gets updated
        """

        if not self.overlap:
            self.where_2d = np.ones_like(self.where_2d)
            for cloud0 in self.clouds:
                logic_no = ((self.x_2d-cloud0['x'])**2 + (self.y_2d-cloud0['y'])**2) <= (cloud0['radius']+radius+min_dist)**2
                self.where_2d[logic_no] = 0

        indices = np.where(self.where_2d==1)
        N_avail = indices[0].size

        if N_avail > limit:

            index = np.random.randint(0, N_avail-1)

            index_x0 = indices[0][index]
            index_y0 = indices[1][index]

            loc_x = self.x[index_x0]
            loc_y = self.y[index_y0]

            ndx = int(radius//self.dx)
            index_x_s = max((0, index_x0-ndx-1))
            index_x_e = min((self.x.size-1, index_x0+ndx+1))

            ndy = int(radius//self.dy)
            index_y_s = max((0, index_y0-ndy-1))
            index_y_e = min((self.y.size-1, index_y0+ndy+1))

            logic_cloud0 = ((self.x_3d[index_x_s:index_x_e, index_y_s:index_y_e, :]-loc_x)**2 + \
                            (self.y_3d[index_x_s:index_x_e, index_y_s:index_y_e, :]-loc_y)**2 + \
                            (self.z_3d[index_x_s:index_x_e, index_y_s:index_y_e, :]*w2h_ratio)**2) <= radius**2
            self.space_3d[index_x_s:index_x_e, index_y_s:index_y_e, :][logic_cloud0] = 1

            # add this newly created cloud into self.clouds
            # =============================================================================
            cloud0 = {
                    'ID': len(self.clouds),
                    'x' : loc_x,
                    'y' : loc_y,
                    'index_x'  : index_x0,
                    'index_y'  : index_y0,
                    'radius'   : radius,
                    'w2h_ratio': w2h_ratio,
                    'min_dist' : min_dist
                    }

            self.clouds.append(cloud0)
            # =============================================================================

            self.cloud_frac = self.space_3d[:, :, 0].sum()/(self.Nx*self.Ny)

            self.trial = 0

        else:

            self.trial += 1
            if self.trial >= self.trial_limit:
                self.can_add_more = False

    def pre_cld_opt_prop(self, extinction0=1.5e-6):

        """
        Assign cloud optical properties, e.g., cloud optical thickness, cloud effective radius

        extinction0: volume extinction per m^3
        """

        extinction_per_grid = extinction0 * self.dx*self.dy * 1000.0**2

        self.ext_3d = extinction_per_grid * self.space_3d

        self.cot_2d = np.sum(self.ext_3d*self.dz*1000.0, axis=-1)

    def test(self, q_factor=2, altitude=None):


        # 3d temperature
        # =============================================================================
        # t_3d      = np.empty((Nt, Nz, Ny, Nx), dtype=p.dtype)
        # t_3d[...] = t[None, :, None, None]
        # =============================================================================

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



if __name__ == '__main__':

    pass
