import os
import sys
import copy
import pickle
import numpy as np


import er3t.common
import er3t.util
# from .util import *



__all__ = ['sfc_2d_gen']



class sfc_2d_gen:
    """
    A class for generating 2D surface properties for radiative transfer calculations.

    This class handles various surface types including Lambertian albedo, BRDF models
    (LSRT, LSRT-Jiao, DSM), and ocean surfaces. It can load existing surface data
    from pickle files or generate new surface data from input dictionaries.

    Attributes:
        ID (str): Class identifier 'Surface 2D'
        sfc (dict): Dictionary containing surface property data
        fname (str): File name for pickle file storage
        verbose (bool): Flag for verbose output
        data (dict): Processed surface data with metadata
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        dx (float): Grid spacing in x-direction (km)
        dy (float): Grid spacing in y-direction (km)

    Parameters:
        sfc_dict (dict, optional): Dictionary containing surface properties.
            For Lambertian: {'alb': albedo_array, 'dx': spacing, 'dy': spacing}
            For BRDF-LSRT: {'fiso': iso_array, 'fgeo': geo_array, 'fvol': vol_array, 'dx': spacing, 'dy': spacing}
            For BRDF-LSRT-Jiao: {'fiso': iso_array, 'fgeo': geo_array, 'fvol': vol_array, 'fj': jiao_array, 'alpha': alpha_array, 'dx': spacing, 'dy': spacing}
            For BRDF-DSM: {'diffusealb': diffuse_albedo, 'diffusefrac': diffuse_fraction, 'refracr': real_refrac, 'refraci': imag_refrac, 'slope': slope_array, 'dx': spacing, 'dy': spacing}
            For Ocean: {'windspeed': wind_array, 'pigment': pigment_array, 'dx': spacing, 'dy': spacing}
        fname (str, optional): Path to pickle file for loading/saving surface data
        overwrite (bool, optional): Whether to overwrite existing pickle file. Default is False
        verbose (bool, optional): Enable verbose output. Default is False

    Raises:
        OSError: If file doesn't exist and no surface dictionary is provided,
                or if unsupported surface type is specified

    Methods:
        load(fname): Load surface data from pickle file
        run(): Process surface data
        dump(fname): Save surface data to pickle file
        pre_sfc(): Process raw surface data into standardized format

    Examples:
        # Create Lambertian surface
        sfc_dict = {'alb': albedo_2d, 'dx': 1.0, 'dy': 1.0}
        sfc = sfc_2d_gen(sfc_dict=sfc_dict, fname='surface.pkl')

        # Load existing surface data
        sfc = sfc_2d_gen(fname='surface.pkl')

        # Create BRDF surface
        sfc_dict = {'fiso': fiso_2d, 'fgeo': fgeo_2d, 'fvol': fvol_2d, 'dx': 1.0, 'dy': 1.0}
        sfc = sfc_2d_gen(sfc_dict=sfc_dict, fname='brdf_surface.pkl', verbose=True)
    """


    ID = 'Surface 2D'


    def __init__(self, \
                 sfc_dict  = None, \
                 fname     = None, \
                 overwrite = False, \
                 verbose   = False):


        self.sfc        = sfc_dict
        self.fname      = fname       # file name of the pickle file
        self.verbose    = verbose     # verbose tag


        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((self.sfc is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.sfc is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run()
            self.dump(self.fname)

        elif ((self.sfc is not None) and (self.fname is None)):

            self.run()

        else:

            msg = 'Error [sfc_2d_gen]: Please check if <%s> exists or provide <sfc_dict> to proceed.' % self.fname
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
                self.dx     = obj.dx
                self.dy     = obj.dy
            else:
                msg = 'Error [sfc_2d_gen]: <%s> is not the correct <pickle> file to load.' % fname
                raise OSError(msg)


    def run(self):

        self.pre_sfc()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [sfc_2d_gen]: Saving object into <%s> ...' % fname)
            pickle.dump(self, f)


    def pre_sfc(self):

        self.data = {}

        keys = {key.lower().replace('_', ''):key for key in self.sfc.keys()}
        keys_check = [key for key in keys.keys()]

        if ('alb' in keys_check):

            Nx, Ny = self.sfc[keys['alb']].shape
            sfc = np.zeros((Nx, Ny, 1), dtype=er3t.common.f_dtype)
            sfc[:, :, 0] = self.sfc[keys['alb']][:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['dx']   = {'data':self.sfc[keys['dx']], 'name':'dx', 'units':'km'}
            self.data['dy']   = {'data':self.sfc[keys['dy']], 'name':'dy', 'units':'km'}
            self.data['sfc']  = {'data':sfc, 'name':'Surface albedo (Lambertian)', 'units':'N/A'}

        elif ('fiso' in keys_check) and ('fvol' in keys_check) and ('fgeo' in keys_check) and ('fj' in keys_check) and ('alpha' in keys_check):

            Nx, Ny = self.sfc[keys['fiso']].shape
            sfc = np.zeros((Nx, Ny, 5), dtype=er3t.common.f_dtype)
            sfc[:, :, 0] = self.sfc[keys['fiso']][:, :]
            sfc[:, :, 1] = self.sfc[keys['fgeo']][:, :]
            sfc[:, :, 2] = self.sfc[keys['fvol']][:, :]
            sfc[:, :, 3] = self.sfc[keys['fj']][:, :]
            sfc[:, :, 4] = self.sfc[keys['alpha']][:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['dx']   = {'data':self.sfc[keys['dx']], 'name':'dx', 'units':'km'}
            self.data['dy']   = {'data':self.sfc[keys['dy']], 'name':'dy', 'units':'km'}
            self.data['sfc']  = {'data':sfc, 'name':'Surface BRDF-LSRT-Jiao (Snow BRDF - Jiao)', 'units':'N/A'}

        elif ('fiso' in keys_check) and ('fvol' in keys_check) and ('fgeo' in keys_check):

            Nx, Ny = self.sfc[keys['fiso']].shape
            sfc = np.zeros((Nx, Ny, 3), dtype=er3t.common.f_dtype)
            sfc[:, :, 0] = self.sfc[keys['fiso']][:, :]
            sfc[:, :, 1] = self.sfc[keys['fgeo']][:, :]
            sfc[:, :, 2] = self.sfc[keys['fvol']][:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['dx']   = {'data':self.sfc[keys['dx']], 'name':'dx', 'units':'km'}
            self.data['dy']   = {'data':self.sfc[keys['dy']], 'name':'dy', 'units':'km'}
            self.data['sfc']  = {'data':sfc, 'name':'Surface BRDF-LSRT (Isotropic, LiSparseR, RossThick)', 'units':'N/A'}

        elif ('diffusealb' in keys_check) and ('diffusefrac' in keys_check) and \
           ('refracr' in keys_check) and ('refraci' in keys_check) and \
           ('slope' in keys_check):

            Nx, Ny = self.sfc[keys['slope']].shape
            sfc = np.zeros((Nx, Ny, 5), dtype=er3t.common.f_dtype)
            sfc[:, :, 0] = self.sfc[keys['diffusealb']][:, :]
            sfc[:, :, 1] = self.sfc[keys['diffusefrac']][:, :]
            sfc[:, :, 2] = self.sfc[keys['refracr']][:, :]
            sfc[:, :, 3] = self.sfc[keys['refraci']][:, :]
            sfc[:, :, 4] = self.sfc[keys['slope']][:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['dx']   = {'data':self.sfc[keys['dx']], 'name':'dx', 'units':'km'}
            self.data['dy']   = {'data':self.sfc[keys['dy']], 'name':'dy', 'units':'km'}
            self.data['sfc']  = {'data':sfc, 'name':'Surface BRDF-DSM (Diffuse-Specular Mixture)', 'units':'N/A'}

        elif ('windspeed' in keys_check) and ('pigment' in keys_check):

            Nx, Ny = self.sfc[keys['windspeed']].shape
            sfc = np.zeros((Nx, Ny, 2), dtype=er3t.common.f_dtype)
            sfc[:, :, 0] = self.sfc[keys['windspeed']][:, :]
            sfc[:, :, 1] = self.sfc[keys['pigment']][:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['dx']   = {'data':self.sfc[keys['dx']], 'name':'dx', 'units':'km'}
            self.data['dy']   = {'data':self.sfc[keys['dy']], 'name':'dy', 'units':'km'}
            self.data['sfc']  = {'data':sfc, 'name':'Surface BRDF-Ocean', 'units':'N/A'}

        else:

            msg = '\nError [sfc_2d_gen]: Currently we only support 2D surface albedo or BRDF.'
            raise OSError(msg)

        self.Nx = self.data['nx']['data']
        self.Ny = self.data['ny']['data']
        self.dx = self.data['dx']['data']
        self.dy = self.data['dy']['data']



if __name__ == '__main__':

    pass
