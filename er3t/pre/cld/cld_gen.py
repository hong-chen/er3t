import os
import sys
import copy
import pickle
import warnings
import numpy as np

import er3t.common
from er3t.pre.atm import atm_atmmod
from er3t.util import downscale, check_equidistant



__all__ = ['cld_gen_hem', 'cld_gen_hom', 'cld_gen_cop']




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
        overwrite=  : keyword argument, default=False, whether to overite
        verbose=    : keyword argument, default=True, verbose tag

    Output:

        self.fname     : absolute file path to the pickle file
        self.clouds    : a list of Python dictionaries that represent all the hemispherical clouds
        self.cloud_frac: cloud fraction of the created scene
        self.x_3d
        self.y_3d
        self.z_3d      : x, y, z in 3D
        self.space_3d  : 0 and 1 values in (x, y, z), 1 indicates cloudy
        self.x_2d
        self.y_2d      : x, y in 2D
        self.min_dist  : minimum distance between clouds
        self.w2h_ratio : width to height ration of the clouds

        self.lay
                ['x']
                ['y']
                ['z']
                ['nx']
                ['ny']
                ['nz']
                ['dx']
                ['dy']
                ['dz']
                ['altitude']
                ['thickness']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)
                ['cot']           (x, y, z)
                ['cer']           (x, y, z)

        self.lev
                ['altitude']
                ['cot_2d']        (x, y)
                ['cth_2d']        (x, y)

    """

    ID = 'Hemispherical Cloud 3D'

    def __init__(
            self,
            fname=None,
            altitude=np.arange(1.5, 6.6, 0.5),
            Nx=400,
            Ny=400,
            dx=0.1,
            dy=0.1,
            radii=[5.0],
            weights=None,
            w2h_ratio=2.0,
            min_dist=0,
            cloud_frac_tgt=0.2,
            trial_limit=100,
            overlap=False,
            overwrite=False,
            verbose=True
            ):

        self.fname    = os.path.abspath(fname) # file name of the pickle file
        self.altitude = altitude               # in km

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

            msg = 'Error [cld_gen_hem]: Please specify file name of pickle file at <fname=>. For example,\ncld0 = cld_gen_hem(fname=\'tmp-data/cloud.pk\')'
            raise OSError(msg)
        # =============================================================================

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)

            try:
                file_correct = (obj.ID == 'Hemispherical Cloud 3D')
            except:
                file_correct = False

            if file_correct:
                if self.verbose:
                    print('Message [cld_gen_hem]: Loading <%s> ...' % fname)
                self.fname      = obj.fname
                self.verbose    = obj.verbose
                self.lay        = obj.lay
                self.lev        = obj.lev
                self.clouds     = obj.clouds
                self.cloud_frac = obj.cloud_frac
                self.space_3d   = obj.space_3d
                self.x          = obj.x
                self.y          = obj.y
                self.z          = obj.z
                self.Nx         = obj.Nx
                self.Ny         = obj.Ny
                self.Nz         = obj.Nz
                self.dx         = obj.dx
                self.dy         = obj.dy
                self.dz         = obj.dz
                self.x_3d       = obj.x_3d
                self.y_3d       = obj.y_3d
                self.z_3d       = obj.z_3d
                self.x_2d       = obj.x_2d
                self.y_2d       = obj.y_2d
                self.min_dist   = obj.min_dist
                self.w2h_ratio  = obj.w2h_ratio
            else:
                msg = 'Error [cld_gen_hem]: <%s> is not the correct pickle file to load.' % fname
                raise OSError(msg)

    def run(self):

        if self.verbose:
            print('Message [cld_gen_hem]: Generating an artificial 3D cloud field filled with hemispherical clouds...')

        if not check_equidistant(self.altitude):
            msg = '\nWarning [cld_gen_hem]: Only support equidistant altitude (z), as well as equidistant x and y.'
            warnings.warn(msg)

        dz = self.altitude[1:]-self.altitude[:-1]

        self.x = np.arange(self.Nx) * self.dx
        self.y = np.arange(self.Ny) * self.dy

        self.dz = dz[0]
        altitude_new  = np.arange(self.altitude[0], min([self.altitude[-1], max(self.radii)/self.w2h_ratio+self.altitude[0]])+self.dz, self.dz)
        self.altitude = altitude_new
        self.z  = self.altitude-self.altitude[0]
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
            self.add_hem_cloud(np.random.choice(self.radii, p=self.weights), min_dist=self.min_dist, w2h_ratio=self.w2h_ratio, limit=1)

        self.lev = {}
        alt_lev = np.append(self.altitude-self.dz/2.0, self.altitude[-1]+self.dz/2.0)
        self.lev['altitude'] = {'data':alt_lev, 'name':'Altitude', 'units':'km'}

        self.lay = {}
        self.lay['x']  = {'data':self.x , 'name':'X' , 'units':'km'}
        self.lay['y']  = {'data':self.y , 'name':'Y' , 'units':'km'}
        self.lay['z']  = {'data':self.z , 'name':'Z' , 'units':'km'}
        self.lay['nx'] = {'data':self.Nx, 'name':'Nx', 'units':'N/A'}
        self.lay['ny'] = {'data':self.Ny, 'name':'Ny', 'units':'N/A'}
        self.lay['nz'] = {'data':self.Nz, 'name':'Nz', 'units':'N/A'}
        self.lay['dx'] = {'data':self.dx, 'name':'dx', 'units':'km'}
        self.lay['dy'] = {'data':self.dy, 'name':'dy', 'units':'km'}
        self.lay['dz'] = {'data':self.dz, 'name':'dz', 'units':'km'}
        self.lay['altitude']  = {'data':self.altitude, 'name':'Altitude', 'units':'km'}

        thickness = self.lev['altitude']['data'][1:] - self.lev['altitude']['data'][:-1]
        self.lay['thickness'] = {'data':thickness, 'name':'Layer thickness', 'units':'km'}

        atm  = atm_atmmod(levels=alt_lev)
        t_1d = atm.lay['temperature']['data']
        t_3d = np.empty((self.Nx, self.Ny, self.Nz), dtype=t_1d.dtype)
        t_3d[...] = t_1d[None, None, :]
        self.lay['temperature'] = {'data':t_3d, 'name':'Temperature', 'units':'K'}

        self.cal_cld_opt_prop()

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_gen_hem]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

    def add_hem_cloud(self, radius, min_dist=0, w2h_ratio=1.0, limit=1):

        """
        Purpose: add a hemispherical cloud into the 3D space

        Input:
            radius   : position argument, radius of the hemispherical cloud (units: km)
            min_dist=: keyword argument, minimum distance between clouds - the larger the value, the more distant away from cloud to cloud (units: km)
            limit=   : keyword argument, when to stop adding more clouds to avoid overlap

        Output:
            1) if a cloud is successfully added,
                i) self.clouds gets updated
               ii) self.space_3d gets updated
              iii) self.cloud_frac gets updated

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

    def cal_cld_opt_prop(self, ext0=0.03, cer0=12.0, cot_scale=1.0):

        """
        Assign cloud optical properties, e.g., cloud optical thickness, cloud effective radius

        ext0=: keyword argument, default=0.03, volume extinction coefficient, reference see https://doi.org/10.5194/acp-11-2903-2011 (units: m^-1)
        cer0=: keyword argument, default=12.0, cloud effective radius (units: micron)
        cot_scale=: keyword argument, default=1.0, scale factor for cloud optical thickness
        """

        # cloud effective radius (3D)
        data = self.space_3d.copy()
        data[data>0] = cer0
        self.lay['cer'] = {'data':data, 'name':'Cloud effective radius', 'units':'micron'}

        # extinction coefficients (3D)
        data = ext0*cot_scale*self.space_3d
        self.lay['extinction'] = {'data':data, 'name':'Extinction coefficients', 'units':'m^-1'}

        # cloud optical thickness (3D)
        data = data*self.dz*1000.0
        self.lay['cot'] = {'data':data, 'name':'Cloud optical thickness', 'units':'N/A'}

        # column integrated cloud optical thickness
        self.lev['cot_2d'] = {'data':np.sum(data, axis=-1), 'name':'Cloud optical thickness', 'units':'N/A'}

        # cloud top height
        data = self.space_3d*self.dz
        self.lev['cth_2d'] = {'data':np.sum(data, axis=-1)+self.lev['altitude']['data'][0], 'name':'Cloud top height', 'units':'km'}

    def update_clouds(
            self,
            w2h_ratio=2.0,
            cot_scale=1.0,
            coarsen=[1, 1, 1]
            ):

        """
        Purpose: update existing cloud field with new
                 1) width-to-height ratio (w2h_ratio)
                 2) scale factor for cloud optical thickness (cot_scale)
                 3) coarsening factors in x, y, and z directions (coarsen)
        """


        self.space_3d = np.zeros_like(self.x_3d)

        for cloud0 in self.clouds:

            radius   = cloud0['radius']
            index_x0 = cloud0['index_x']
            index_y0 = cloud0['index_y']

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

            if w2h_ratio < cloud0['w2h_ratio'] and self.verbose:
                msg = 'Warning [cld_gen_hem]: Cloud %2.2d is taller than before, cloud top might be cutted.' % cloud0['ID']
                warnings.warn(msg)

            cloud0['w2h_ratio'] = w2h_ratio

        self.cal_cld_opt_prop(cot_scale=cot_scale)

        # downscale (coarsen) data if needed
        if any([i!=1 for i in coarsen]):
            self.downscale(coarsen)

    def downscale(self, coarsen):

        dnx, dny, dnz = coarsen

        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or (self.Nz%dnz != 0):
            msg = 'Error [cld_gen_hem]: The original dimension %s is not divisible with %s, please check input (dnx, dny, dnz).' % (str(self.lay['temperature']['data'].shape), str(coarsen))
            raise ValueError(msg)
        else:
            new_shape = (self.Nx//dnx, self.Ny//dny, self.Nz//dnz)

            if self.verbose:
                print('Message [cld_gen_hem]: Downscaling data from dimension %s to %s ...' % (str(self.lay['temperature']['data'].shape), str(new_shape)))

            # self.lay
            # =============================================================================
            self.lay['x']['data']         = downscale(self.lay['x']['data']        , (new_shape[0], ), operation='mean')
            self.lay['y']['data']         = downscale(self.lay['y']['data']        , (new_shape[1], ), operation='mean')
            self.lay['z']['data']         = downscale(self.lay['z']['data']        , (new_shape[2], ), operation='mean')

            self.lay['altitude']['data']  = downscale(self.lay['altitude']['data'] , (new_shape[2], ), operation='mean')
            self.lay['thickness']['data'] = downscale(self.lay['thickness']['data'], (new_shape[2], ), operation='sum')

            self.lay['dx']['data'] *= dnx
            self.lay['dy']['data'] *= dny
            self.lay['dz']['data'] *= dnz

            self.lay['nx']['data'] = new_shape[0]
            self.lay['ny']['data'] = new_shape[1]
            self.lay['nz']['data'] = new_shape[2]

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsen):
                        if key in ['extinction', 'cot']:
                            operation = 'mean'
                        else:
                            operation = 'max'
                        self.lay[key]['data']  = downscale(self.lay[key]['data'], new_shape, operation=operation)
            # =============================================================================

            # self.lev
            # =============================================================================
            alt_lev = np.append(self.lay['altitude']['data']-self.lay['dz']['data']/2.0, self.lay['altitude']['data'][-1]+self.lay['dz']['data']/2.0)
            self.lev['altitude']['data'] = alt_lev

            for key in self.lev.keys():
                if isinstance(self.lev[key]['data'], np.ndarray):
                    if self.lev[key]['data'].ndim == 2:
                        if key in ['cot_2d']:
                            operation = 'mean'
                        else:
                            operation = 'max'
                        self.lev[key]['data']  = downscale(self.lev[key]['data'], (new_shape[0], new_shape[1]), operation=operation)
            # =============================================================================





class cld_gen_hom:

    """
    Purpose: generate 3D homogeneous cloud field

    Input:
        Nx=: keyword argument, default=400, number of pixel in x of 3D space
        Ny=: keyword argument, default=400, number of pixel in y of 3D space
        dx=: keyword argument, default=100, delta length in x per pixel (units: km)
        dy=: keyword argument, default=100, delta length in y per pixel (units: km)
        overwrite=  : keyword argument, default=False, whether to overite
        verbose=    : keyword argument, default=True, verbose tag

    Output:

        self.fname     : absolute file path to the pickle file

        self.lay
                ['x']
                ['y']
                ['z']
                ['nx']
                ['ny']
                ['nz']
                ['dx']
                ['dy']
                ['dz']
                ['altitude']
                ['thickness']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)
                ['cot']           (x, y, z)
                ['cer']           (x, y, z)

        self.lev
                ['altitude']
                ['cot_2d']        (x, y)

    """

    ID = 'Homogeneous Cloud 3D'

    def __init__(
            self,
            fname=None,
            altitude=np.arange(1.5, 2.5, 0.5),
            Nx=10,
            Ny=10,
            dx=0.1,
            dy=0.1,
            cot0=10.0,
            cer0=10.0,
            atm_obj=None,
            overwrite=False,
            verbose=True
            ):

        if fname is not None:
            self.fname = os.path.abspath(fname) # file name of the pickle file
        else:
            self.fname = fname

        self.altitude = altitude               # in km

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy

        self.verbose = verbose     # verbose tag

        # check for pickle file
        #╭────────────────────────────────────────────────────────────────────────────╮#

        # if pickle file exists - load the data directly from pickle file
        #╭──────────────────────────────────────────────────────────────╮#
        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)
        #╰──────────────────────────────────────────────────────────────╯#

        # if pickle file does not exist or overwrite is specified - run the program and save
        # data into pickle file
        #╭──────────────────────────────────────────────────────────────╮#
        elif ((self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.fname is not None) and (not os.path.exists(self.fname))):

            self.run(cot0, cer0, atm_obj=atm_obj)
            self.dump(self.fname)
        #╰──────────────────────────────────────────────────────────────╯#

        # if pickle file doesn't get specified
        #╭──────────────────────────────────────────────────────────────╮#
        else:

            self.run(cot0, cer0, atm_obj=atm_obj)
        #╰──────────────────────────────────────────────────────────────╯#

        #╰────────────────────────────────────────────────────────────────────────────╯#

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)

            try:
                file_correct = (obj.ID == 'Homogeneous Cloud 3D')
            except:
                file_correct = False

            if file_correct:
                if self.verbose:
                    print('Message [cld_gen_hom]: Loading <%s> ...' % fname)
                self.fname      = obj.fname
                self.verbose    = obj.verbose
                self.lay        = obj.lay
                self.lev        = obj.lev
                self.x          = obj.x
                self.y          = obj.y
                self.z          = obj.z
                self.Nx         = obj.Nx
                self.Ny         = obj.Ny
                self.Nz         = obj.Nz
                self.dx         = obj.dx
                self.dy         = obj.dy
                self.dz         = obj.dz
            else:
                msg = 'Error [cld_gen_hom]: <%s> is not the correct pickle file to load.' % fname
                raise OSError(msg)

    def run(self, cot0, cer0, atm_obj=None):

        if self.verbose:
            print('Message [cld_gen_hom]: Generating an artificial homogeneous 3D cloud field with <COT=%.1f> and <CER=%.1f μm> ...' % (cot0, cer0))

        if not check_equidistant(self.altitude):
            msg = 'Warning [cld_gen_hom]: Only support equidistant altitude (z), as well as equidistant x and y.'
            warnings.warn(msg)

        self.x = np.arange(self.Nx) * self.dx
        self.y = np.arange(self.Ny) * self.dy

        dz = self.altitude[1:]-self.altitude[:-1]
        self.dz = dz[0]
        self.z  = self.altitude-self.altitude[0]
        self.Nz = self.z.size

        self.lev = {}
        alt_lev = np.append(self.altitude-self.dz/2.0, self.altitude[-1]+self.dz/2.0)
        self.lev['altitude'] = {'data':alt_lev, 'name':'Altitude', 'units':'km'}

        self.lay = {}
        self.lay['x']  = {'data':self.x , 'name':'X' , 'units':'km'}
        self.lay['y']  = {'data':self.y , 'name':'Y' , 'units':'km'}
        self.lay['z']  = {'data':self.z , 'name':'Z' , 'units':'km'}
        self.lay['nx'] = {'data':self.Nx, 'name':'Nx', 'units':'N/A'}
        self.lay['ny'] = {'data':self.Ny, 'name':'Ny', 'units':'N/A'}
        self.lay['nz'] = {'data':self.Nz, 'name':'Nz', 'units':'N/A'}
        self.lay['dx'] = {'data':self.dx, 'name':'dx', 'units':'km'}
        self.lay['dy'] = {'data':self.dy, 'name':'dy', 'units':'km'}
        self.lay['dz'] = {'data':self.dz, 'name':'dz', 'units':'km'}
        self.lay['altitude']  = {'data':self.altitude, 'name':'Altitude', 'units':'km'}

        thickness = self.lev['altitude']['data'][1:] - self.lev['altitude']['data'][:-1]
        self.lay['thickness'] = {'data':thickness, 'name':'Layer thickness', 'units':'km'}

        # get temperature profile
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if atm_obj is None:
            atm_obj = atm_atmmod(levels=alt_lev)
            t_1d = atm_obj.lay['temperature']['data']
        else:
            t_1d = np.interp(self.lay['altitude']['data'], atm_obj.lay['altitude']['data'], atm_obj.lay['temperature']['data'])
        #╰────────────────────────────────────────────────────────────────────────────╯#

        t_3d = np.empty((self.Nx, self.Ny, self.Nz), dtype=t_1d.dtype)
        t_3d[...] = t_1d[None, None, :]
        self.lay['temperature'] = {'data':t_3d, 'name':'Temperature', 'units':'K'}

        self.cal_cld_opt_prop(cot0=cot0, cer0=cer0)

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_gen_hom]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

    def cal_cld_opt_prop(self, cot0=10.0, cer0=10.0, cot_scale=1.0):

        """
        Assign cloud optical properties, e.g., cloud optical thickness, cloud effective radius

        ext0=: keyword argument, default=0.03, volume extinction coefficient, reference see https://doi.org/10.5194/acp-11-2903-2011 (units: m^-1)
        cer0=: keyword argument, default=12.0, cloud effective radius (units: micron)
        cot_scale=: keyword argument, default=1.0, scale factor for cloud optical thickness
        """

        data0 = np.zeros((self.Nx, self.Ny, self.Nz), dtype=er3t.common.f_dtype)

        # cloud effective radius (3D)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data = data0.copy()
        data[...] = cer0
        self.lay['cer'] = {'data':data, 'name':'Cloud effective radius', 'units':'micron'}
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # extinction coefficients (3D)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        cot0_ = cot0*cot_scale/self.Nz
        ext0 = cot0_/self.dz/1000.0
        data = data0.copy()
        data[...] = ext0
        self.lay['extinction'] = {'data':data, 'name':'Extinction coefficients', 'units':'m^-1'}
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # cloud optical thickness (3D)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data = data0.copy()
        data[...] = cot0_
        self.lay['cot'] = {'data':data, 'name':'Cloud optical thickness', 'units':'N/A'}
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # column integrated cloud optical thickness
        #╭────────────────────────────────────────────────────────────────────────────╮#
        self.lev['cot_2d'] = {'data':np.sum(data, axis=-1), 'name':'Cloud optical thickness', 'units':'N/A'}
        #╰────────────────────────────────────────────────────────────────────────────╯#





class cld_gen_cop:

    """
    Input:
        cot=      : keyword argument, default=None, cloud optical thickness retrieval
        cer=      : keyword argument, default=None, cloud effective radius retrieval
        cth=      : keyword argument, default=None, cloud top height retrieval in [km]
        cgt=      : keyword argument, default=None, cloud geometrical thickness retrieval in [km]
        fname=    : keyword argument, default=None, the file path of the Python pickle file
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
                ['altitude']
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)

        self.lev
                ['altitude']
    """


    ID = 'Cloud Product 3D (Satellite)'


    def __init__(self,              \
                 fname     = None,  \
                 cot       = None,  \
                 cer       = None,  \
                 cth       = None,  \
                 cgt       = None,  \
                 dz        = None,  \
                 atm_obj   = None,  \
                 extent_xy = None,  \
                 overwrite = False, \
                 verbose   = True):


        self.fname      = fname       # file name of the pickle file
        self.atm        = atm_obj
        self.extent     = extent_xy
        self.verbose    = verbose     # verbose tag

        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((cot is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((cot is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run(cot, cer, cth, cgt, dz)
            self.dump(self.fname)

        elif ((cot is not None) and (self.fname is None)):

            self.run(cot, cer, cth, cgt, dz)

        else:

            msg = '\nError [cld_gen_cop]: Please check if <%s> exists or provide <cot> to proceed.' % self.fname
            raise OSError(msg)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [cld_gen_cop]: Loading <%s> ...' % fname)
                self.fname  = obj.fname
                self.extent = obj.extent
                self.lay    = obj.lay
                self.lev    = obj.lev
            else:
                msg = '\nError [cld_gen_cop]: <%s> is not the correct pickle file to load.' % fname
                raise OSError(msg)


    def run(self, cot, cer, cth, cgt, dz):

        if self.extent is None:
            msg = '\nError [cld_gen_cop]: Please provide <extent_xy> to proceed.'
            raise OSError(msg)

        if cot is None:
            msg = '\nError [cld_gen_cop]: Please provide <cot> to proceed.'
            raise OSError(msg)

        if cer is None:
            msg = '\nWarning [cld_gen_cop]: <cer> is not specified, setting <cer> to 10 micron ...'
            warnings.warn(msg)

        if cth is None:
            cth = 3.0
            msg = '\nWarning [cld_gen_cop]: <cth> is not specified, setting <cth> to 3km ...'
            warnings.warn(msg)

        if cgt is None:
            cgt = 1.0
            msg = '\nWarning [cld_gen_cop]: <cgt> is not specified, setting <cgt> to 1km ...'
            warnings.warn(msg)

        if dz is None:
            dz  = 1.0
            msg = '\nWarning [cld_gen_cop]: <dz> is not specified, setting <dz> to 1km ...'
            warnings.warn(msg)

        # process
        self.process(
                cot,
                cloud_effective_radius=cer,
                cloud_top_height=cth,
                cloud_geometrical_thickness=cgt,
                layer_thickness=dz
                )


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_gen_cop]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def process(
            self,
            cloud_optical_thickness,
            cloud_effective_radius=10.0,
            cloud_top_height=3.0,
            cloud_geometrical_thickness=1.0,
            layer_thickness=1.0
            ):

        self.lay = {}
        self.lev = {}

        Nx, Ny = cloud_optical_thickness.shape

        logic_clear = np.isnan(cloud_optical_thickness) | (cloud_optical_thickness <= 0.0)
        if not isinstance(cloud_effective_radius, np.ndarray):
            cloud_effective_radius = np.full(cloud_optical_thickness.shape, cloud_effective_radius)
            cloud_effective_radius[logic_clear] = 0.0

        if not isinstance(cloud_top_height, np.ndarray):
            cloud_top_height = np.full(cloud_optical_thickness.shape, cloud_top_height)
            cloud_top_height[logic_clear] = 0.0

        if not isinstance(cloud_geometrical_thickness, np.ndarray):
            cloud_geometrical_thickness = np.full(cloud_optical_thickness.shape, cloud_geometrical_thickness)
            cloud_geometrical_thickness[logic_clear] = 0.0

        if cloud_top_height.shape != cloud_optical_thickness.shape:
            msg = '\nError [cld_gen_cop]: The dimension of <cloud_top_height> does not match <cloud_optical_thickness>.'
            raise OSError(msg)

        cloud_bottom_height = cloud_top_height-cloud_geometrical_thickness
        cloud_bottom_height[cloud_bottom_height<0.0] = 0.0

        h_bottom = max([np.nanmin(cloud_bottom_height), 0.0])
        h_top    = min([np.nanmax(cloud_top_height), 20.0])

        if h_bottom >= h_top:
            msg = '\nError [cld_gen_cop]: Cloud bottom height is greater than cloud top height, check whether the input cloud top height <cth> is in the units of <km>.'
            raise ValueError(msg)

        levels   = np.arange(h_bottom, h_top+0.1*layer_thickness, layer_thickness)
        if self.atm is None:
            self.atm = atm_atmmod(levels=levels)
        else:
            self.atm = atm_atmmod(levels=levels, fname_atmmod=self.atm.fname_atmmod)

        x_1d_ = np.linspace(self.extent[0], self.extent[1], Nx+1)
        y_1d_ = np.linspace(self.extent[2], self.extent[3], Ny+1)

        dx = x_1d_[1]-x_1d_[0]
        dy = y_1d_[1]-y_1d_[0]
        x_1d = x_1d_[:-1] + dx/2.0
        y_1d = y_1d_[:-1] + dy/2.0

        self.lay['x']  = {'data':x_1d     , 'name':'X'          , 'units':'km'}
        self.lay['y']  = {'data':y_1d     , 'name':'Y'          , 'units':'km'}
        self.lay['nx'] = {'data':Nx       , 'name':'Nx'         , 'units':'N/A'}
        self.lay['ny'] = {'data':Ny       , 'name':'Ny'         , 'units':'N/A'}
        self.lay['dx'] = {'data':dx       , 'name':'dx'         , 'units':'km'}
        self.lay['dy'] = {'data':dy       , 'name':'dy'         , 'units':'km'}
        self.lay['altitude'] = copy.deepcopy(self.atm.lay['altitude'])
        self.lay['thickness']= copy.deepcopy(self.atm.lay['thickness'])

        self.lev['altitude'] = copy.deepcopy(self.atm.lev['altitude'])

        # temperature 3d
        #╭──────────────────────────────────────────────────────────────╮#
        t_1d = self.atm.lay['temperature']['data']
        Nz   = t_1d.size
        t_3d      = np.empty((Nx, Ny, Nz), dtype=t_1d.dtype)
        t_3d[...] = t_1d[None, None, :]

        self.lay['temperature'] = {'data':t_3d, 'name':'Temperature', 'units':'K'}
        #╰──────────────────────────────────────────────────────────────╯#


        # extinction 3d
        #╭──────────────────────────────────────────────────────────────╮#
        ext_3d      = np.zeros((Nx, Ny, Nz), dtype=er3t.common.f_dtype)

        alt = self.atm.lay['altitude']['data']

        for i in range(Nx):
            for j in range(Ny):
                cbh0 = cloud_bottom_height[i, j]
                cth0 = cloud_top_height[i, j]

                cot0  = cloud_optical_thickness[i, j]
                indices =  np.where((alt>=cbh0) & (alt<=cth0))[0]
                if indices.size == 0:
                    indices = np.array([-1])

                dz    = self.atm.lay['thickness']['data'][indices].sum() * 1000.0
                ext_3d[i, j, indices] = cot0/dz

        ext_3d[np.isnan(ext_3d)] = 0.0
        self.lay['extinction']   = {'data':ext_3d, 'name':'Extinction coefficients', 'units':'m^-1'}
        #╰──────────────────────────────────────────────────────────────╯#

        # cer 3d
        #╭──────────────────────────────────────────────────────────────╮#
        cer_3d = np.empty((Nx, Ny, Nz), dtype=cloud_effective_radius.dtype)
        cer_3d[...] = cloud_effective_radius[:, :, None]

        cer_3d[ext_3d<=0.0] = 0.0
        self.lay['cer'] = {'data':cer_3d, 'name':'Cloud Effective radius', 'units':'micron'}
        #╰──────────────────────────────────────────────────────────────╯#


    def get_extra_cloud_property(self, cer_mode='top', fill_clear=0.0, overwrite=False):

        """
        cloud mask
        cloud effective radius
        """

        # cloud mask based on cer
        #╭────────────────────────────────────────────────────────────────────────────╮#
        cer_3d = self.lay['cer']['data'].copy()
        if ('cld_msk' not in self.lay.keys()) or ('cld_msk_2d' not in self.lev.keys()) or (overwrite):
            cld_msk_3d = np.zeros(cer_3d.shape, dtype=np.int32)
            cld_msk_3d[cer_3d>0.0] = 1

            cld_msk_2d = np.sum(cld_msk_3d, axis=-1)
            cld_msk_2d[cld_msk_2d>0] = 1

            self.lay['cld_msk'] = {'data':cld_msk_3d, 'name':'Cloud Mask (1: Cloud)', 'units':'N/A'}
            self.lev['cld_msk_2d'] = {'data': cld_msk_2d, 'name': 'Cloud Mask (1: Cloud)', 'units':'N/A'}
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # cloud optical thickness
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if ('cot_2d' not in self.lev.keys()) or (overwrite):
            cot_2d = np.nansum(self.lay['extinction']['data']*self.lay['thickness']['data'][0]*1000.0, axis=-1)
            cot_2d[cld_msk_2d==0] = fill_clear
            self.lev['cot_2d'] = {'data': cot_2d, 'name': 'Cloud Optical Thickness', 'units': 'N/A'}
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # cloud effective radius <cer_2d>
        # cloud top height <cth_2d>
        # cloud base height <cbh_2d>
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if ('cer_2d' not in self.lev.keys()) or ('cth_2d' not in self.lev.keys()) or ('cbh_2d' not in self.lev.keys()) or (overwrite):
            cer_3d[cld_msk_3d==0] = np.nan

            Nx, Ny, Nz = cer_3d.shape

            z  = self.lay['altitude']['data']
            dz = self.lay['thickness']['data'][0]
            z_indice = np.arange(Nz)

            cer_2d = np.zeros((Nx, Ny), dtype=er3t.common.f_dtype); cer_2d[...] = fill_clear

            cth_2d = np.zeros((Nx, Ny), dtype=er3t.common.f_dtype); cth_2d[...] = fill_clear
            cbh_2d = np.zeros((Nx, Ny), dtype=er3t.common.f_dtype); cbh_2d[...] = fill_clear

            for i in range(Nx):
                for j in range(Ny):
                    cer0 = cer_3d[i, j, :]
                    logic_good = np.logical_not(np.isnan(cer0)) & (cer0>0)
                    if logic_good.sum() > 0:
                        if cer_mode.lower() == 'top':
                            cer_2d[i, j]  = cer0[np.max(z_indice[logic_good])]
                        elif cer_mode.lower() == 'mean':
                            cer_2d[i, j] = np.mean(cer0[logic_good])

                        cth_2d[i, j] = z[np.max(z_indice[logic_good])] + dz/2.0
                        cbh_2d[i, j] = z[np.min(z_indice[logic_good])] - dz/2.0

            self.lev['cer_2d'] = {'data': cer_2d, 'name': 'Cloud Effective Radius', 'units': 'micron'}
            self.lev['cth_2d'] = {'data': cth_2d, 'name': 'Cloud Top Height'      , 'units': 'km'}
            self.lev['cbh_2d'] = {'data': cbh_2d, 'name': 'Cloud Base Height'     , 'units': 'km'}
        #╰────────────────────────────────────────────────────────────────────────────╯#





if __name__ == '__main__':

    pass
