import os
import sys
import copy
import pickle
import numpy as np


import er3t.common
from .util import *



__all__ = ['atm_atmmod']



class atm_atmmod:

    """
    Input:

        levels=      : keyword argument, numpy array, height in km
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


    ID     = 'Atmosphere 1D'

    gases  = ['o3', 'o2', 'h2o', 'co2', 'no2', 'ch4']

    reference = '\nAFGL Atmospheric Profile (Anderson et al., 1986):\n- Anderson, G. P., Clough, S. A., Kneizys, F. X., Chetwynd, J. H., and Shettle, E. P.: AFGL atmospheric constituent profiles (0–120 km), Tech. Rep. AFGL-TR-86–0110, Air Force Geophys. Lab., Hanscom Air Force Base, Bedford, Massachusetts, USA, 1986.'

    def __init__(self,                \
                 levels       = None, \
                 fname        = None, \
                 fname_atmmod = '%s/afglus.dat' % er3t.common.fdir_data_atmmod, \
                 overwrite    = False, \
                 verbose      = False):

        er3t.util.add_reference(self.reference)

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

            sys.exit('Error   [atm_atmmod]: Please check if \'%s\' exists or provide \'levels\' to proceed.' % fname)


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
                sys.exit('Error   [atm_atmmod]: File \'%s\' is not the correct pickle file to load.' % fname)


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


    def atmmod(self):

        vnames = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
        units  = ['km', 'mb', 'K', 'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3']
        data   = np.genfromtxt(self.fname_atmmod)

        # read original data from *.dat file into Python dictionary that contains 'data', 'name', and 'units'
        self.atm0 = {}
        for i, vname in enumerate(vnames):
            self.atm0[vname] = {'data':data[:, i], 'name':vname, 'units':units[i]}

        # 1. change the values in array from descending order to ascending order
        indices = np.argsort(self.atm0['altitude']['data'])
        for key in self.atm0.keys():
            self.atm0[key]['data'] = self.atm0[key]['data'][indices]

        # 2. calculate the mixing ratio from volume number density for each gas
        for key in self.atm0.keys():
            if key in self.gases:
                self.atm0[key]['data']  = self.atm0[key]['data']/self.atm0['air']['data']
                self.atm0[key]['units'] = 'N/A'


    def interp(self):

        # check whether the input height is within the atmosphere height range
        if self.levels.min() < self.atm0['altitude']['data'].min():
            sys.exit('Error   [atm_atmmod]: Input levels too low.')
        if self.levels.max() > self.atm0['altitude']['data'].max():
            sys.exit('Error   [atm_atmmod]: Input levels too high.')

        self.lev = {}
        self.lev = copy.deepcopy(self.atm0)
        self.lev['altitude']['data']  = self.levels

        self.lay = {}
        self.lay = copy.deepcopy(self.atm0)
        self.lay['altitude']['data']  = self.layers
        self.lay['thickness'] = { \
                 'name' : 'Thickness', \
                 'units':'km', \
                 'data':self.levels[1:]-self.levels[:-1]}

        # Linear interpolate to input levels and layers
        for key in self.atm0.keys():
            if key not in ['altitude', 'pressure']:
                self.lev[key]['data'] = np.interp(self.lev['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])
                self.lay[key]['data'] = np.interp(self.lay['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])

        # Use Barometric formula to interpolate pressure
        self.lev['pressure']['data'] = interp_pres_from_alt_temp(self.atm0['pressure']['data'], self.atm0['altitude']['data'], self.atm0['temperature']['data'], \
                self.lev['altitude']['data'], self.lev['temperature']['data'])
        self.lay['pressure']['data'] = interp_pres_from_alt_temp(self.atm0['pressure']['data'], self.atm0['altitude']['data'], self.atm0['temperature']['data'], \
                self.lay['altitude']['data'], self.lay['temperature']['data'])


    def add_ch4(self):

        ch4 = {'name':'ch4', 'units':'cm-3', 'data':interp_ch4(self.levels)}
        self.lev['ch4'] = ch4

        ch4 = {'name':'ch4', 'units':'cm-3', 'data':interp_ch4(self.layers)}
        self.lay['ch4'] = ch4


    def cal_num_den(self):

        self.lev['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':6.02214179e23/8.314472*self.lev['pressure']['data']/self.lev['temperature']['data']*1.0e-4}

        self.lay['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':6.02214179e23/8.314472*self.lay['pressure']['data']/self.lay['temperature']['data']*1.0e-4}

        for key in self.lev.keys():
            if key in self.gases:
                self.lev[key]['data']  = self.lev[key]['data'] * self.lev['factor']['data']
                self.lev[key]['units'] = 'cm-3'
                self.lay[key]['data']  = self.lay[key]['data'] * self.lay['factor']['data']
                self.lay[key]['units'] = 'cm-3'



if __name__ == '__main__':

    pass
