import os
import sys
import copy
import pickle
import numpy as np

import er3t



__all__ = ['sfc_2d_gen']


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

            keys = [key.lower().replace('_', '') for key in self.alb.keys()]
            if ('fiso' in keys) and ('fvol' in keys) and ('fgeo' in keys):

                Nx, Ny = self.alb['fiso'].shape
                alb = np.zeros((Nx, Ny, 3), dtype=np.float64)
                alb[:, :, 0] = self.alb['fiso'][:, :]
                alb[:, :, 1] = self.alb['fgeo'][:, :]
                alb[:, :, 2] = self.alb['fvol'][:, :]

                self.data['nx']   = {'data':Nx , 'name':'Nx', 'units':'N/A'}
                self.data['ny']   = {'data':Ny , 'name':'Ny', 'units':'N/A'}
                self.data['alb']  = {'data':alb, 'name':'Surface BRDF-LSRT (Isotropic, LiSparseR, RossThick)', 'units':'N/A'}

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
