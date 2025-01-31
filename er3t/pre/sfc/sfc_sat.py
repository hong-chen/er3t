import os
import sys
import pickle
import numpy as np
import copy


__all__ = ['sfc_sat']



class sfc_sat:

    """
    Input:
        sat_obj=  : keyword argument, default=None, the satellite object created from modis_l1b, modis_l2, seviri_l2 etc
        fname=    : keyword argument, default=None, the file path of the Python pickle file
        overwrite=: keyword argument, default=False, whether to overwrite or not
        verbose=  : keyword argument, default=False, verbose tag

    Output:
        self.sfc
                ['nx']
                ['ny']
                ['alb']
    """


    ID = 'Satellite Surface 2D'


    def __init__(self, \
                 sat_obj   = None, \
                 fname     = None, \
                 extent    = None, \
                 overwrite = False, \
                 verbose   = False):


        self.sat        = sat_obj
        self.fname      = fname       # file name of the pickle file
        self.extent     = extent
        self.verbose    = verbose     # verbose tag


        if ((self.fname is not None) and (os.path.exists(self.fname)) and (not overwrite)):

            self.load(self.fname)

        elif ((self.sat is not None) and (self.fname is not None) and (os.path.exists(self.fname)) and (overwrite)) or \
             ((self.sat is not None) and (self.fname is not None) and (not os.path.exists(self.fname))):

            self.run()
            self.dump(self.fname)

        elif ((self.sat is not None) and (self.fname is None)):

            self.run()

        else:

            sys.exit('Error   [sfc_sat]: Please check if \'%s\' exists or provide \'sat_obj\' to proceed.' % self.fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'data'):
                if self.verbose:
                    print('Message [sfc_sat]: Loading \'%s\' ...' % fname)
                self.fname  = obj.fname
                self.extent = obj.extent
                self.data   = obj.data
                self.Nx     = obj.Nx
                self.Ny     = obj.Ny
            else:
                sys.exit('Error   [sfc_sat]: \'%s\' is not the correct \'pickle\' file to load.' % fname)


    def run(self):

        self.pre_sat()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [sfc_sat]: Saving object into \'%s\' ...' % fname)
            pickle.dump(self, f)


    def pre_sat(self):

        self.data = {}

        keys = self.sat.data.keys()
        if ('alb_2d' not in keys):
            sys.exit('Error   [sfc_sat]: Please make sure \'sat_obj.data\' contains \'alb_2d\'.')

        Nx, Ny = self.sat.data['alb_2d']['data'].shape

        self.data['nx']   = {'data':Nx       , 'name':'Nx'         , 'units':'N/A'}
        self.data['ny']   = {'data':Ny       , 'name':'Ny'         , 'units':'N/A'}
        self.data['sfc']  = copy.deepcopy(self.sat.data['alb_2d'])
        self.data['lon']  = copy.deepcopy(self.sat.data['lon_2d'])
        self.data['lat']  = copy.deepcopy(self.sat.data['lat_2d'])

        self.Nx = Nx
        self.Ny = Ny



if __name__ == '__main__':

    pass
