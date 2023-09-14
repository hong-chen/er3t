import os
import sys
import copy
import pickle
import numpy as np

import er3t



__all__ = ['cal_ocean_brdf', 'sfc_2d_gen']


def cal_ocean_brdf(
        wvl=500.0,
        u10=1.0,
        sal=34.3,
        pcl=0.01,
        whitecaps=True,
        ):

    """
    This code is adapted from <libRadtran/libsrc_f/oceabrdf.f>

    Input parameters:
        u10: 10m wind speed, units: m/s, default=1.0, can be either value or 2D array for a domain
        sal: salinity, units: per mille [0.1% or ppt or psu], default=34.3
        pcl: pigment concentration, units: mg/m^3, default=0.01
    """

    # check data dimension
    #/----------------------------------------------------------------------------\#
    try:
        Nx, Ny = u10.shape
        u10 = np.float_(u10)
        ndim = u10.ndim
    except Exception as error:
        # print(error)
        u10 = float(u10)
        ndim = 0

    if ndim == 2:
        wvl_ = np.zeros_like(u10)
        wvl_[...] = wvl
        wvl = wvl_

        sal_ = np.zeros_like(u10)
        sal_[...] = sal
        sal = sal_
    #\----------------------------------------------------------------------------/#


    # refractive index of water as a function of wavelength and salinity
    # Hale and Querry, 1973: Optical Constants of Water in the 200-nm to 200-μm Wavelength Region
    #/----------------------------------------------------------------------------\#
    refractive_index_water = {
            'wvl': np.array([ \
                   0.250,0.275,0.300,0.325,0.345,0.375,0.400,0.425,0.445,0.475,\
                   0.500,0.525,0.550,0.575,0.600,0.625,0.650,0.675,0.700,0.725,\
                   0.750,0.775,0.800,0.825,0.850,0.875,0.900,0.925,0.950,0.975,\
                   1.000,1.200,1.400,1.600,1.800,2.000,2.200,2.400,2.600,2.650,\
                   2.700,2.750,2.800,2.850,2.900,2.950,3.000,3.050,3.100,3.150,\
                   3.200,3.250,3.300,3.350,3.400,3.450,3.500,3.600,3.700,3.800,\
                   3.900,4.000], dtype=np.float64) * 1000.0,

           'real': np.array([ \
                   1.362,1.354,1.349,1.346,1.343,1.341,1.339,1.338,1.337,1.336,\
                   1.335,1.334,1.333,1.333,1.332,1.332,1.331,1.331,1.331,1.330,\
                   1.330,1.330,1.329,1.329,1.329,1.328,1.328,1.328,1.327,1.327,\
                   1.327,1.324,1.321,1.317,1.312,1.306,1.296,1.279,1.242,1.219,\
                   1.188,1.157,1.142,1.149,1.201,1.292,1.371,1.426,1.467,1.483,\
                   1.478,1.467,1.450,1.432,1.420,1.410,1.400,1.385,1.374,1.364,\
                   1.357,1.351], dtype=np.float64),

      'imaginary': np.array([ \
                   3.35E-08,2.35E-08,1.60E-08,1.08E-08,6.50E-09,\
                   3.50E-09,1.86E-09,1.30E-09,1.02E-09,9.35E-10,\
                   1.00E-09,1.32E-09,1.96E-09,3.60E-09,1.09E-08,\
                   1.39E-08,1.64E-08,2.23E-08,3.35E-08,9.15E-08,\
                   1.56E-07,1.48E-07,1.25E-07,1.82E-07,2.93E-07,\
                   3.91E-07,4.86E-07,1.06E-06,2.93E-06,3.48E-06,\
                   2.89E-06,9.89E-06,1.38E-04,8.55E-05,1.15E-04,\
                   1.10E-03,2.89E-04,9.56E-04,3.17E-03,6.70E-03,\
                   1.90E-02,5.90E-02,1.15E-01,1.85E-01,2.68E-01,\
                   2.98E-01,2.72E-01,2.40E-01,1.92E-01,1.35E-01,\
                   9.24E-02,6.10E-02,3.68E-02,2.61E-02,1.95E-02,\
                   1.32E-02,9.40E-03,5.15E-03,3.60E-03,3.40E-03,\
                   3.80E-03,4.60E-03], dtype=np.float64)
                 }

    refrac_r = np.interp(wvl, refractive_index_water['wvl'], refractive_index_water['real'])
    refrac_i = np.interp(wvl, refractive_index_water['wvl'], refractive_index_water['imaginary'])

    # salinity corrections
    #   - Friedman, 1969: Infrared Characteristics of Ocean Water (1.5-15μ)
    #   - McLellan, 1965: Elements of Physical Oceanography
    #   - Sverdrup et al., 1942 - The Oceans
    #/--------------------------------------------------------------\#
    refrac_r += 0.006*(sal/34.3)
    # refrac_i += 0.000*(sal/34.3)
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # variance of micro-scopic surface slope
    #   - Cox and Munk, 1954: Measurement of the roughness of the sea surface from photographs of the sun’s glitter
    #/----------------------------------------------------------------------------\#
    slope = 0.00512*u10 + 0.003
    #\----------------------------------------------------------------------------/#


    # whitecaps treatment
    #   - Koepke, 1984 - Effective reflectance of oceanic whitecaps
    #/----------------------------------------------------------------------------\#
    if whitecaps:
        reflectance_whitecaps = {
                'wvl': np.arange(200.0, 4001.0, 100.0),
                'ref': np.array([
                       0.220,0.220,0.220,0.220,0.220,0.220,0.215,0.210,0.200,0.190,
                       0.175,0.155,0.130,0.080,0.100,0.105,0.100,0.080,0.045,0.055,
                       0.065,0.060,0.055,0.040,0.000,0.000,0.000,0.000,0.000,0.000,
                       0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
                       ]),
                }

        diffuse_frac = 2.95e-06 * (u10**3.52)
        diffuse_alb  = np.interp(wvl, reflectance_whitecaps['wvl'], reflectance_whitecaps['ref'])
    else:
        diffuse_frac = 0.0*u10
        diffuse_alb  = 0.0*u10
    #\----------------------------------------------------------------------------/#

    params = {
          'diffuse_alb': diffuse_alb,
         'diffuse_frac': diffuse_frac,
             'refrac_r': refrac_r,
             'refrac_i': refrac_i,
                'slope': slope,
            }

    return params


class sfc_2d_gen:

    """
    Input:
        alb_2d=   : keyword argument, default=None, 2D array of surface albedo
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.sfc
                ['nx']
                ['ny']
                ['alb']
    """


    ID = 'Surface 2D'


    def __init__(self, \
                 alb_2d    = None, \
                 fname     = None, \
                 overwrite = False, \
                 verbose   = False):


        self.alb        = alb_2d
        self.fname      = fname       # file name of the pickle file
        self.verbose    = verbose     # verbose tag


        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((self.alb is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.alb is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run()
            self.dump(self.fname)

        elif ((self.alb is not None) and (self.fname is None)):

            self.run()

        else:

            msg = 'Error [sfc_2d_gen]: Please check if <%s> exists or provide <alb_2d> to proceed.' % self.fname
            raise OSError(msg)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'data'):
                if self.verbose:
                    print('Message [sfc_2d_gen]: Loading <%s> ...' % fname)
                self.fname  = obj.fname
                self.data   = obj.data
                self.Nx     = obj.Nx
                self.Ny     = obj.Ny
            else:
                msg = 'Error [sfc_2d_gen]: <%s> is not the correct <pickle> file to load.' % fname
                raise OSError(msg)


    def run(self):

        self.pre_alb()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [sfc_2d_gen]: Saving object into <%s> ...' % fname)
            pickle.dump(self, f)


    def pre_alb(self):

        self.data = {}

        if isinstance(self.alb, np.ndarray):

            Nx, Ny = self.alb.shape
            alb = np.zeros((Nx, Ny, 1), dtype=np.float64)
            alb[:, :, 0] = self.alb[:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['alb']  = {'data':alb, 'name':'Surface albedo (Lambertian)', 'units':'N/A'}

        elif isinstance(self.alb, dict):

            keys = {key.lower().replace('_', ''):key for key in self.alb.keys()}
            keys_check = [key for key in keys.keys()]

            if ('fiso' in keys_check) and ('fvol' in keys_check) and ('fgeo' in keys_check):

                Nx, Ny = self.alb[keys['fiso']].shape
                alb = np.zeros((Nx, Ny, 3), dtype=np.float64)
                alb[:, :, 0] = self.alb[keys['fiso']][:, :]
                alb[:, :, 1] = self.alb[keys['fgeo']][:, :]
                alb[:, :, 2] = self.alb[keys['fvol']][:, :]

                self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
                self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
                self.data['alb']  = {'data':alb, 'name':'Surface BRDF-LSRT (Isotropic, LiSparseR, RossThick)', 'units':'N/A'}

            if ('diffusealb' in keys_check) and ('diffusefrac' in keys_check) and \
               ('refracr'    in keys_check) and ('refraci'     in keys_check) and \
               ('slope'      in keys_check):

                Nx, Ny = self.alb[keys['slope']].shape
                alb = np.zeros((Nx, Ny, 5), dtype=np.float64)
                alb[:, :, 0] = self.alb[keys['diffusealb']][:, :]
                alb[:, :, 1] = self.alb[keys['diffusefrac']][:, :]
                alb[:, :, 2] = self.alb[keys['refracr']][:, :]
                alb[:, :, 3] = self.alb[keys['refraci']][:, :]
                alb[:, :, 4] = self.alb[keys['slope']][:, :]

                self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
                self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
                self.data['alb']  = {'data':alb, 'name':'Surface BRDF-DSM (Diffuse-Specular Mixture)', 'units':'N/A'}

            else:

                msg = '\nError [sfc_2d_gen]: Currently we only support 2D surface albedo or BRDF [RossThickLiSparseReciprocal].'
                raise OSError(msg)

        else:

            msg = '\nError [sfc_2d_gen]: Currently we only support 2D surface albedo or BRDF [RossThickLiSparseReciprocal].'
            raise OSError(msg)


        self.Nx = Nx
        self.Ny = Ny


if __name__ == '__main__':

    pass
