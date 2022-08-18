import os
import sys
import pickle
import numpy as np
import copy
from scipy import interpolate

from er3t.util import mmr2vmr, cal_rho_air, downscale, cal_ext
from er3t.pre.atm import atm_atmmod

try:
    from pyhdf.SD import SD, SDC
except ImportError:
    msg = 'Warning [cld_sev.py]: To use \'cld_sev.py\', \'pyhdf\' needs to be installed.'
    # raise ImportError(msg)
    print(msg)



__all__ = ['cld_sev']



class cld_sev:

    """
    Input:
        fname_h4= : keyword argument, default=None, the file path of the original HDF4 file
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        coarsing= : keyword argument, default=[1, 1, 1], the parameter to downgrade the data in [x, y, z]
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
                ['altitude']
                ['pressure']
                ['lon']           (x, y)
                ['lat']           (x, y)
                ['cot']           (x, y)
                ['cer']           (x, y)
                ['temperature']   (x, y, z)
                ['extinction']    (x, y, z)

        self.lev
                ['altitude']
    """


    ID = 'SEVIRI Cloud 3D'


    def __init__(self, \
                 fname_h4  = None, \
                 fname     = None, \
                 extent    = None, \
                 coarsing  = [1, 1, 1], \
                 overwrite = False, \
                 verbose   = False):

        self.verbose  = verbose     # verbose tag
        self.coarsing = coarsing    # (dn_x, dn_y, dn_z, dn_t)

        self.fname    = fname       # file name of the pickle file
        self.fname_h4 = fname_h4    # file name of the HDF4 file
        self.extent   = extent

        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((self.fname_h4 is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.fname_h4 is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run(self.fname_h4)
            self.dump(self.fname)

        elif ((self.fname_h4 is not None) and (self.fname is None)):

            self.run(self.fname_h4)

        else:

            sys.exit('Error   [cld_sev]: Please check if \'%s\' exists or provide \'fname_h4\' to proceed.' % self.fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [cld_sev]: loading %s ...' % fname)
                self.fname = obj.fname
                self.lay   = obj.lay
                self.lev   = obj.lev
            else:
                sys.exit('Error   [cld_sev]: %s is not the correct \'pickle\' file to load.' % fname)


    def run(self, fname_h4):

        if self.verbose:
            print("Message [cld_sev]: Processing %s ..." % fname_h4)

        # pre process
        self.pre_sev()

        # downgrade data if needed
        if any([i!=1 for i in self.coarsing]):
            self.downgrade(self.coarsing)

        # post process
        self.post_sev()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [cld_sev]: saving object into %s ...' % fname)
            pickle.dump(self, f)


    def pre_sev(self, earth_radius=6378.0, vertical_resolution=0.1, cloud_top_height=1.0, cloud_bottom_height=0.5):

        self.lay = {}
        self.lev = {}

        f     = SD(self.fname_h4, SDC.READ)

        # lon lat
        lon0       = f.select('Longitude')
        lat0       = f.select('Latitude')
        cot0       = f.select('Cloud_Optical_Thickness_16')
        cer0       = f.select('Cloud_Effective_Radius_16')
        cot_pcl0   = f.select('Cloud_Optical_Thickness_16_PCL')
        cer_pcl0   = f.select('Cloud_Effective_Radius_16_PCL')
        cth0       = f.select('Cloud_Top_Height')


        if 'actual_range' in lon0.attributes().keys():
            lon_range = lon0.attributes()['actual_range']
            lat_range = lat0.attributes()['actual_range']
        else:
            lon_range = [-180.0, 180.0]
            lat_range = [-90.0 , 90.0]

        lon       = lon0[:]
        lat       = lat0[:]
        cot       = np.float_(cot0[:])
        cer       = np.float_(cer0[:])
        cot_pcl   = np.float_(cot_pcl0[:])
        cer_pcl   = np.float_(cer_pcl0[:])
        cth       = np.float_(cth0[:])

        logic     = (lon>=lon_range[0]) & (lon<=lon_range[1]) & (lat>=lat_range[0]) & (lat<=lat_range[1])
        lon       = lon[logic]
        lat       = lat[logic]
        cot       = cot[logic]
        cer       = cer[logic]
        cot_pcl   = cot_pcl[logic]
        cer_pcl   = cer_pcl[logic]
        cth       = cth[logic]

        if self.extent is not None:
            logic     = (lon>=self.extent[0])&(lon<=self.extent[1])&(lat>=self.extent[2])&(lat<=self.extent[3])
            lon       = lon[logic]
            lat       = lat[logic]
            cot       = cot[logic]
            cer       = cer[logic]
            cot_pcl   = cot_pcl[logic]
            cer_pcl   = cer_pcl[logic]
            cth       = cth[logic]

        xy = (self.extent[1]-self.extent[0])*(self.extent[3]-self.extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(self.extent[1]-self.extent[0]))
        if Nx%2 == 1:
            Nx += 1

        Ny = int(N0*(self.extent[3]-self.extent[2]))
        if Ny%2 == 1:
            Ny += 1

        lon_1d0 = np.linspace(self.extent[0], self.extent[1], Nx+1)
        lat_1d0 = np.linspace(self.extent[2], self.extent[3], Ny+1)

        lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
        lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

        dx = (lon_1d[1]-lon_1d[0])/180.0 * (np.pi*earth_radius)
        dy = (lat_1d[1]-lat_1d[0])/180.0 * (np.pi*earth_radius)

        x_1d = (lon_1d-lon_1d[0])*dx
        y_1d = (lat_1d-lat_1d[0])*dy


        # lon, lat
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)
        # lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        # cot
        cot_range     = cot0.attributes()['valid_range']
        cer_range     = cer0.attributes()['valid_range']
        cot_pcl_range = cot_pcl0.attributes()['valid_range']
        cer_pcl_range = cer_pcl0.attributes()['valid_range']
        cth_range     = cth0.attributes()['valid_range']

        # +
        # create cot_all/cer_all that contains both cot/cer and cot_pcl/cer_pcl
        cot_all = np.zeros(cot.size, dtype=np.float64)
        cer_all = np.zeros(cer.size, dtype=np.float64)
        cth_all = np.zeros(cth.size, dtype=np.float64); cth_all[...] = np.nan

        logic = (cot>=cot_range[0]) & (cot<=cot_range[1]) & (cer>=cer_range[0]) & (cer<=cer_range[1]) & (cth>=cth_range[0]) & (cth<=cth_range[1])
        cot_all[logic] = cot[logic]*cot0.attributes()['scale_factor'] + cot0.attributes()['add_offset']
        cer_all[logic] = cer[logic]*cer0.attributes()['scale_factor'] + cer0.attributes()['add_offset']
        cth_all[logic] = cth[logic]*cth0.attributes()['scale_factor'] + cth0.attributes()['add_offset']

        logic_pcl = np.logical_not(logic) & (cot_pcl>=cot_pcl_range[0]) & (cot_pcl<=cot_pcl_range[1]) & (cer_pcl>=cer_pcl_range[0]) & (cer_pcl<=cer_pcl_range[1]) & (cth>=cth_range[0]) & (cth<=cth_range[1])
        cot_all[logic_pcl] = cot_pcl[logic_pcl]*cot_pcl0.attributes()['scale_factor'] + cot_pcl0.attributes()['add_offset']
        cer_all[logic_pcl] = cer_pcl[logic_pcl]*cer_pcl0.attributes()['scale_factor'] + cer_pcl0.attributes()['add_offset']
        cth_all[logic_pcl] = cth[logic_pcl]*cth0.attributes()['scale_factor'] + cth0.attributes()['add_offset']
        cth_all /= 1000.0

        logic_all = logic | logic_pcl
        # -

        cot = cot_all
        cer = cer_all
        cth = cth_all

        cot[np.logical_not(logic_all)] = 0.0
        cer[np.logical_not(logic_all)] = 1.0
        cth[np.logical_not(logic_all)] = np.nan

        points = np.transpose(np.vstack((lon, lat)))

        cot_2d = interpolate.griddata(points, cot, (lon_2d, lat_2d), method='nearest')
        cer_2d = interpolate.griddata(points, cer, (lon_2d, lat_2d), method='nearest')
        cth_2d = interpolate.griddata(points, cth, (lon_2d, lat_2d), method='nearest')

        f.end()

        # self.atm = atm_atmmod(np.arange(int(np.nanmax(cth_2d))+2))
        self.atm = atm_atmmod(levels=np.arange(cloud_bottom_height, cloud_top_height+vertical_resolution, vertical_resolution))
        self.lay['x']  = {'data':x_1d     , 'name':'X'          , 'units':'km'}
        self.lay['y']  = {'data':y_1d     , 'name':'Y'          , 'units':'km'}
        self.lay['nx'] = {'data':Nx       , 'name':'Nx'         , 'units':'N/A'}
        self.lay['ny'] = {'data':Ny       , 'name':'Ny'         , 'units':'N/A'}
        self.lay['dx'] = {'data':dx       , 'name':'dx'         , 'units':'km'}
        self.lay['dy'] = {'data':dy       , 'name':'dy'         , 'units':'km'}
        self.lay['altitude'] = copy.deepcopy(self.atm.lay['altitude'])
        self.lay['cot']= {'data':cot_2d   , 'name':'Cloud optical thickness', 'units':'N/A'}
        self.lay['cer']= {'data':cer_2d   , 'name':'Cloud effective radius' , 'units':'micron'}
        self.lay['cth']= {'data':cth_2d   , 'name':'Cloud top height'       , 'units':'km'}
        self.lay['lon']= {'data':lon_2d   , 'name':'Longitude'              , 'units':'degree'}
        self.lay['lat']= {'data':lat_2d   , 'name':'Latitude'               , 'units':'degree'}


        # temperature 3d
        t_1d = self.atm.lay['temperature']['data']
        Nz   = t_1d.size
        t_3d      = np.empty((Nx, Ny, Nz), dtype=t_1d.dtype)
        t_3d[...] = t_1d[None, None, :]

        self.lay['temperature'] = {'data':t_3d, 'name':'Temperature', 'units':'K'}


        # extinction 3d
        ext_3d      = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        # alt = self.atm.lay['altitude']['data']
        # for i in range(Nx):
        #     for j in range(Ny):
        #         cld_top  = cth_2d[i, j]
        #         if not np.isnan(cld_top):
        #             lwp  = 5.0/9.0 * 1.0 * cot_2d[i, j] * cer_2d[i, j] / 10.0
        #             ext0 = 0.75 * 2.0 * lwp / cer_2d[i, j] / 100.0
        #             index = np.argmin(np.abs(cld_top-alt))
        #             ext_3d[i, j, index] = ext0

        for i in range(Nx):
            for j in range(Ny):

                ext0 = cal_ext(cot_2d[i, j], cer_2d[i, j])
                ext_3d[i, j, :] = ext0/(self.atm.lay['thickness']['data'].sum()*1000.0)

        self.lay['extinction'] = {'data':ext_3d, 'name':'Extinction coefficients'}

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz


    def downgrade(self, coarsing):

        dnx, dny, dnz = coarsing

        if (self.Nx%dnx != 0) or (self.Ny%dny != 0) or \
           (self.Nz%dnz != 0):
            sys.exit('Error   [cld_sev]: the original dimension %s is not divisible with %s, please check input (dnx, dny, dnz).' % (str(self.lay['Temperature']['data'].shape), str(coarsing)))
        else:
            new_shape = (self.Nx//dnx, self.Ny//dny, self.Nz//dnz)

            if self.verbose:
                print('Message [cld_sev]: Downgrading data from dimension %s to %s ...' % (str(self.lay['Temperature']['data'].shape), str(new_shape)))

            self.lay['x']['data']        = downgrading(self.lay['x']['data']       , (self.Nx//dnx,))
            self.lay['y']['data']        = downgrading(self.lay['y']['data']       , (self.Ny//dny,))
            self.lay['altitude']['data'] = downgrading(self.lay['altitude']['data'], (self.Nz//dnz,))

            for key in self.lay.keys():
                if isinstance(self.lay[key]['data'], np.ndarray):
                    if self.lay[key]['data'].ndim == len(coarsing):
                        self.lay[key]['data']  = downgrading(self.lay[key]['data'], new_shape)


    def post_sev(self):

        dz  = self.lay['altitude']['data'][1:]-self.lay['altitude']['data'][:-1]
        dz0 = dz[0]
        diff = np.abs(dz-dz0)
        if any([i>0.001 for i in diff]):
            print(dz0, dz)
            sys.exit('Error   [cld_sev]: Non-equidistant intervals found in \'dz\'.')
        else:
            dz  = np.append(dz, dz0)
            alt = np.append(self.lay['altitude']['data']-dz0/2.0, self.lay['altitude']['data'][-1]+dz0/2.0)

        self.lev['altitude']    = {'data':alt, 'name':'Altitude'       , 'units':'km'}

        self.lay['thickness']   = {'data':dz , 'name':'Layer thickness', 'units':'km'}



if __name__ == '__main__':

    pass
