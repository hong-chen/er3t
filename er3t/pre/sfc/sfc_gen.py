import os
import sys
import copy
import pickle
import numpy as np

import er3t



__all__ = ['sfc_gen']



# under development

class sfc_gen:

    """
    Input:

        fname=       : keyword argument, string, default=None, the atmoshpere file user wants to name
        fname_atmmod=: keyword argument, string, defult='mca-data/atmmod/afglus.dat', the base atmosphere file to interpolate to levels and layers
        overwrite=   : keyword argument, boolen, default=False, whether or not user wants to overwrite the atmosphere file
        verbose=     : keyword argument, boolen, default=False, whether or not print detailed messages

    Note:
    If levels is provided but fname does not exisit:
        calculate atmospheric gases profile and save data into fname

    if levels is not provided but fname is provided (also exists):
        read out the data from fname

    if levels and fname are neither provided:
        exit with error message

    Output:
        self.lev['pressure']
        self.lev['temperature']
        self.lev['altitude']
        self.lev['h2o']
        self.lev['o2']
        self.lev['o3']
        self.lev['co2']
        self.lev['no2']
        self.lev['ch4']
        self.lev['factor']

        self.lay['pressure']
        self.lay['temperature']
        self.lay['altitude']
        self.lay['thickness']
        self.lay['h2o']
        self.lay['o2']
        self.lay['o3']
        self.lay['co2']
        self.lay['no2']
        self.lay['ch4']
        self.lay['factor']
    """


    ID     = 'Surface 2D'


    def __init__(self,                \
                 levels       = None, \
                 fname        = None, \
                 fname_atmmod = '%s/atmmod/afglus.dat' % er3t.common.fdir_data, \
                 overwrite    = False, \
                 verbose      = False):

        self.verbose      = verbose
        self.fname_atmmod = fname_atmmod

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((levels is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((levels is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(levels)
            self.dump(fname)

        elif ((levels is not None) and (fname is None)):

            self.run(levels)

        else:

            exit('Error   [atm_atmmod]: Please check if \'%s\' exists or provide \'levels\' to proceed.' % fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [atm_atmmod]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.lev   = obj.lev
                self.lay   = obj.lay
            else:
                exit('Error   [atm_atmmod]: \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, levels):

        self.levels = levels
        self.layers = 0.5 * (levels[1:]+levels[:-1])

        # self.atm0: Python dictionary
        #   self.atm0['altitude']
        #   self.atm0['pressure']
        #   self.atm0['temperature']
        #   self.atm0['co2']
        #   self.atm0['no2']
        #   self.atm0['h2o']
        #   self.atm0['o3']
        #   self.atm0['o2']
        self.atmmod()

        # self.lev, self.lay: Python dictionary
        #   self.lev['altitude']    | self.lay['altitude']
        #   self.lev['pressure']    | self.lay['pressure']
        #   self.lev['temperature'] | self.lay['temperature']
        #   self.lev['co2']         | self.lay['co2']
        #   self.lev['no2']         | self.lay['no2']
        #   self.lev['h2o']         | self.lay['h2o']
        #   self.lev['o3']          | self.lay['o3']
        #   self.lev['o2']          | self.lay['o2']
        self.interp()

        # add self.lev['ch4'] and self.lay['ch4']
        self.add_ch4()

        # covert mixing ratio [unitless] to number density [cm-3]
        self.cal_num_den()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [atm_atmmod]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



if __name__ == '__main__':

    pass
