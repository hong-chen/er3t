import os
import sys
import copy
import pickle
import numpy as np


import er3t.common
import er3t.util
from .util import *



__all__ = ['sfc_2d_gen']



class sfc_2d_gen:

    """
    Input:
        sfc_2d=   : keyword argument, default=None, 2D array of surface albedo
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.data
                ['nx']
                ['ny']
                ['sfc']
    """


    ID = 'Surface 2D'


    def __init__(self, \
                 sfc_2d    = None, \
                 fname     = None, \
                 overwrite = False, \
                 verbose   = False):


        self.sfc        = sfc_2d
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

            msg = 'Error [sfc_2d_gen]: Please check if <%s> exists or provide <sfc_2d> to proceed.' % self.fname
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

        self.pre_sfc()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [sfc_2d_gen]: Saving object into <%s> ...' % fname)
            pickle.dump(self, f)


    def pre_sfc(self):

        self.data = {}

        if isinstance(self.sfc, np.ndarray):

            Nx, Ny = self.sfc.shape
            sfc = np.zeros((Nx, Ny, 1), dtype=np.float64)
            sfc[:, :, 0] = self.sfc[:, :]

            self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
            self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
            self.data['sfc']  = {'data':sfc, 'name':'Surface albedo (Lambertian)', 'units':'N/A'}

        elif isinstance(self.sfc, dict):

            keys = {key.lower().replace('_', ''):key for key in self.sfc.keys()}
            keys_check = [key for key in keys.keys()]

            if ('fiso' in keys_check) and ('fvol' in keys_check) and ('fgeo' in keys_check):

                Nx, Ny = self.sfc[keys['fiso']].shape
                sfc = np.zeros((Nx, Ny, 3), dtype=np.float64)
                sfc[:, :, 0] = self.sfc[keys['fiso']][:, :]
                sfc[:, :, 1] = self.sfc[keys['fgeo']][:, :]
                sfc[:, :, 2] = self.sfc[keys['fvol']][:, :]

                self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
                self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
                self.data['sfc']  = {'data':sfc, 'name':'Surface BRDF-LSRT (Isotropic, LiSparseR, RossThick)', 'units':'N/A'}

            if ('diffusealb' in keys_check) and ('diffusefrac' in keys_check) and \
               ('refracr'    in keys_check) and ('refraci'     in keys_check) and \
               ('slope'      in keys_check):

                Nx, Ny = self.sfc[keys['slope']].shape
                sfc = np.zeros((Nx, Ny, 5), dtype=np.float64)
                sfc[:, :, 0] = self.sfc[keys['diffusealb']][:, :]
                sfc[:, :, 1] = self.sfc[keys['diffusefrac']][:, :]
                sfc[:, :, 2] = self.sfc[keys['refracr']][:, :]
                sfc[:, :, 3] = self.sfc[keys['refraci']][:, :]
                sfc[:, :, 4] = self.sfc[keys['slope']][:, :]

                self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
                self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
                self.data['sfc']  = {'data':sfc, 'name':'Surface BRDF-DSM (Diffuse-Specular Mixture)', 'units':'N/A'}

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
