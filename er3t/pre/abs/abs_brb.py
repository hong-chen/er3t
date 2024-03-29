"""
This code contains functions/classes for providing gas absorption for broadband.

Contains:
    abs_rrtmg_sw: absorption coefficients for RRTMG shortwave bands
    abs_rrtmg_lw: absorption coefficients for RRTMG longwave bands
"""

# import os
# import sys
# import pickle
# import multiprocessing as mp
# import h5py
# import copy
# import numpy as np

import er3t

import os
import sys
import glob
import datetime
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


__all__ = ['abs_rrtmg_sw']



class abs_rrtmg_sw:

    """
    This module is based on the RRTMG shortwave database
    publicly avaiable at
    https://github.com/AER-RC/RRTMG_SW

    Output:
        self.coef['wavelength']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """

    fname = '%s/rrtmg/rrtmg_sw.nc' % er3t.common.fdir_data_abs
    reference = 'Iacono, M.J., Delamere, J.S., Mlawer, E.J., Shephard, M.W., Clough, S.A., and Collins, W.D.: Radiative forcing by long-lived greenhouse gases: Calculations with the AER radiative transfer models, J. Geophys. Res., 113, D13103, https://doi.org/10.1029/2008JD009944, 2008.'

    def __init__(self, \
                 iband  = 0,
                 g_mode = 0,
                 wavelength = None,  \
                 fname      = None,  \
                 atm_obj    = None,  \
                 overwrite  = False, \
                 verbose    = False):

        if g_mode != 0:
            msg = '\nError [abs_rrtmg_sw]: currently only <g_mode=0> is supported.'
            raise OSError(msg)

        self.iband   = iband
        self.g_mode  = g_mode
        self.atm_obj = atm_obj

        self.get_coef(iband, g_mode, atm_obj)

    def get_coef(self, iband, g_mode, atm_obj):

        self.coef = {}

        f0 = Dataset(self.fname, 'r')

        # band info
        #   <self.wavelength>
        #   <self.band_range>
        #/----------------------------------------------------------------------------\#
        wvln_min = f0.variables['BandWavenumberLowerLimit'][:][iband]
        wvln_max = f0.variables['BandWavenumberUpperLimit'][:][iband]
        wvl_min = 1.0e7/wvln_max
        wvl_max = 1.0e7/wvln_min

        band_range = (wvl_min, wvl_max)

        self.wavelength = 2.0e7/(wvln_min+wvln_max)
        self.band_range = band_range
        #\----------------------------------------------------------------------------/#


        # read out gas names
        #   <gases>
        #/----------------------------------------------------------------------------\#
        gas_bytes = f0.variables['AbsorberNames'][:]
        Ngas, Nchar = gas_bytes.shape
        gases = []
        for i in range(Ngas):
            gas_name0 = ''.join([j.decode('utf-8') for j in gas_bytes[i, :]]).strip().lower()
            if len(gas_name0) > 0:
                gases.append(gas_name0)
        #\----------------------------------------------------------------------------/#


        # read out key gas names
        #   <key_gas_low>
        #   <key_gas_upp>
        #/----------------------------------------------------------------------------\#
        key_gas_low_bytes = f0.variables['KeySpeciesNamesLowerAtmos'][:][:, iband, :]
        key_gas_upp_bytes = f0.variables['KeySpeciesNamesUpperAtmos'][:][:, iband, :]
        Nkey, Nchar = key_gas_upp_bytes.shape
        ikey_gas_low = []
        ikey_gas_upp = []
        for i in range(Nkey):
            key_gas_low_name0 = ''.join([j.decode('utf-8') for j in key_gas_low_bytes[i, :]]).strip().lower()
            key_gas_upp_name0 = ''.join([j.decode('utf-8') for j in key_gas_upp_bytes[i, :]]).strip().lower()
            if len(key_gas_low_name0) > 0:
                ikey_gas_low.append(gases.index(key_gas_low_name0))
            if len(key_gas_upp_name0) > 0:
                ikey_gas_upp.append(gases.index(key_gas_upp_name0))

        key_gas_low = [gases[i] for i in ikey_gas_low]
        key_gas_upp = [gases[i] for i in ikey_gas_upp]
        #\----------------------------------------------------------------------------/#


        # Gs, Nz
        #/----------------------------------------------------------------------------\#
        Ng = f0.variables['NumGPoints'][:][g_mode, iband]
        Nz = atm_obj.lay['pressure']['data'].size
        #\----------------------------------------------------------------------------/#


        # solar
        #/----------------------------------------------------------------------------\#
        sol_upp = f0['SolarSourceFunctionUpperAtmos'][:][g_mode, iband, :, :Ng]
        sol_low = f0['SolarSourceFunctionLowerAtmos'][:][g_mode, iband, :, :Ng]

        self.coef['solar'] = {
                'name': 'Solar Factor (Ng)',
                'data': sol_upp[0, :],
                }
        #\----------------------------------------------------------------------------/#


        # slit function
        #/----------------------------------------------------------------------------\#
        self.coef['slit_func'] = {
                'name': 'Slit Function (Nz, Ng)',
                'data': np.ones((Nz, Ng), dtype=np.float64),
                }
        #\----------------------------------------------------------------------------/#


        # weights
        #/----------------------------------------------------------------------------\#
        weight =  np.array([ \
              0.1527534276, 0.1491729617, 0.1420961469, \
              0.1316886544, 0.1181945205, 0.1019300893, \
              0.0832767040, 0.0626720116, 0.0424925000, \
              0.0046269894, 0.0038279891, 0.0030260086, \
              0.0022199750, 0.0014140010, 0.0005330000, \
              0.0000750000 \
              ])

        weight *= 1.0/weight.sum() # make sure weights can add up to 1.0

        self.coef['weight'] = {
                'name': 'Weight (Ng)',
                'data': weight,
                }
        #\----------------------------------------------------------------------------/#


        # coef
        #/----------------------------------------------------------------------------\#
        abso_coef = np.zeros((Nz, Ng), dtype=np.float64)

        # read data from RRTMG SW
        #/--------------------------------------------------------------\#
        # coef
        coef_low = f0.variables['AbsorptionCoefficientsLowerAtmos'][:][g_mode, iband, :, :Ng, :, :]
        coef_upp = f0.variables['AbsorptionCoefficientsUpperAtmos'][:][g_mode, iband, :, :Ng, :, :]
        coef_key_low = f0.variables['KeySpeciesAbsorptionCoefficientsLowerAtmos'][:][g_mode, iband, :Ng, :, :, :]
        coef_key_upp = f0.variables['KeySpeciesAbsorptionCoefficientsUpperAtmos'][:][g_mode, iband, :Ng, :, :, :]
        coef_h2o_fore_low = f0.variables['H2OForeignAbsorptionCoefficientsLowerAtmos'][:][g_mode, iband, :Ng, :]
        coef_h2o_fore_upp = f0.variables['H2OForeignAbsorptionCoefficientsUpperAtmos'][:][g_mode, iband, :Ng, :]
        coef_h2o_self     = f0.variables['H2OSelfAbsorptionCoefficients'][:][g_mode, iband, :Ng, :]

        # axes
        # mr_low  = f0.variables['KeySpeciesRatioLowerAtmos'][:] # for some reason, Python netCDF4 library cannot read the value for this variable correctly
        # mr_upp  = f0.variables['KeySpeciesRatioUpperAtmos'][:] # for some reason, Python netCDF4 library cannot read the value for this variable correctly
        mr_low  = np.linspace(0.0, 1.0, coef_key_low.shape[-1])
        mr_upp  = np.linspace(0.0, 1.0, coef_key_upp.shape[-1])
        p_upp   = f0.variables['PressureUpperAtmos'][:]
        p_low   = f0.variables['PressureLowerAtmos'][:]
        t       = f0.variables['Temperature'][:]
        dt      = f0.variables['TemperatureDiffFromMLS'][:]
        #\--------------------------------------------------------------/#

        for ig in range(Ng):
            for gas0 in atm_obj.gases:
                if gas0 != 'no2':
                    igas = gases.index(gas0)
                    # coef0_low = coef_low[igas, ig, :, :]
                    # coef0_upp = coef_upp[igas, ig, :, :]

                    coef0_key_low = coef_key_low[ig, :, :, :]
                    coef0_key_upp = coef_key_upp[ig, :, :, :]

                    print(gas0, ig)
                    print(coef0_low)



        self.coef['abso_coef'] = {
                'name': 'Absorption Coefficient (Nz, Ng)',
                'data': abso_coef,
                }
        #\----------------------------------------------------------------------------/#
        sys.exit()



        # profile
        #/----------------------------------------------------------------------------\#
        t_ref = f0.variables['ReferenceTemperature'][:]
        p     = f0.variables['Pressure'][:]
        #\----------------------------------------------------------------------------/#



        # Coef    :  ('Absorber', 'GPoint', 'Temperature', 'KeySpeciesRatioLowerAtmos')
        # Coef Key:  ('GPoint', 'PressureLowerAtmos', 'TemperatureDiffFromMLS', 'KeySpeciesRatioLowerAtmos')

        print('-'*80)
        print('Band #%d' % (iband+1))
        print('Center wavelength: %.4fnm' % self.wavelength)
        print('Wavelength range: %.4f - %.4fnm' % self.band_range)
        print('Number of Gs: ', Ng)
        print()
        print('Lower Atmosphere:')
        print('Key species: ', key_gas_low)
        print('Pressure: %s\n%s' % (p_low.shape, er3t.util.nice_array_str(p_low)))
        print('Mixing Ratio: %s\n%s' % (mr_low.shape, er3t.util.nice_array_str(mr_low)))
        print('Temperature Diff.: %s\n%s' % (dt.shape, er3t.util.nice_array_str(dt)))
        print('Coef.: %s\n' % str(coef_low.shape))
        print('Coef. Key: %s\n' % str(coef_key_low.shape))
        print()
        print('Upper Atmosphere:')
        print('Key species: ', key_gas_upp)
        print('Pressure: %s\n%s' % (p_upp.shape, er3t.util.nice_array_str(p_upp)))
        print('Mixing Ratio: %s\n%s' % (mr_upp.shape, er3t.util.nice_array_str(mr_upp)))
        print('Temperature Diff.: %s\n%s' % (dt.shape, er3t.util.nice_array_str(dt)))
        print('Coef.: %s\n' % str(coef_upp.shape))
        print('Coef. Key: %s\n' % str(coef_key_upp.shape))
        print('-'*80)


        # + netCDF
        # variables['BandWavenumberLowerLimit'] -------------------- : Dataset  (14,)
        # variables['BandWavenumberUpperLimit'] -------------------- : Dataset  (14,)
        # variables['RRTMBandNumber'] ------------------------------ : Dataset  (14,)

        # variables['NumGPoints'] ---------------------------------- : Dataset  (2, 14)
        # variables['AbsorberNames'] ------------------------------- : Dataset  (12, 5)
        # variables['Temperature'] --------------------------------- : Dataset  (19,)
        # variables['TemperatureDiffFromMLS'] ---------------------- : Dataset  (5,)
        # variables['PressureH2OForeign'] -------------------------- : Dataset  (4,)



        # variables['ReferenceTemperature'] ------------------------ : Dataset  (59,)
        # variables['Pressure'] ------------------------------------ : Dataset  (59,)
        # variables['LogPressure'] --------------------------------- : Dataset  (59,)

        # variables['SolarSourceFunctionUpperAtmos'] --------------- : Dataset  (2, 14, 5, 16)
        # variables['NRLSSI2SSFFacularUpperAtmos'] ----------------- : Dataset  (2, 14, 5, 16)
        # variables['NRLSSI2SSFQuietSunUpperAtmos'] ---------------- : Dataset  (2, 14, 5, 16)
        # variables['NRLSSI2SSFSunspotUpperAtmos'] ----------------- : Dataset  (2, 14, 5, 16)

        # variables['SolarSourceFunctionLowerAtmos'] --------------- : Dataset  (2, 14, 9, 16)
        # variables['NRLSSI2SSFFacularLowerAtmos'] ----------------- : Dataset  (2, 14, 9, 16)
        # variables['NRLSSI2SSFQuietSunLowerAtmos'] ---------------- : Dataset  (2, 14, 9, 16)
        # variables['NRLSSI2SSFSunspotLowerAtmos'] ----------------- : Dataset  (2, 14, 9, 16)

        # variables['KeySpeciesRatioUpperAtmos'] ------------------- : Dataset  (5,)
        # variables['KeySpeciesAbsorptionCoefficientsUpperAtmos'] -- : Dataset  (2, 14, 16, 47, 5, 5)
        # variables['AbsorptionCoefficientsUpperAtmos'] ------------ : Dataset  (2, 14, 12, 16, 19, 5)
        # variables['KeySpeciesNamesUpperAtmos'] ------------------- : Dataset  (2, 14, 3)
        # variables['TemperatureH2OForeignUpperAtmos'] ------------- : Dataset  (2,)
        # variables['PressureUpperAtmos'] -------------------------- : Dataset  (47,)
        # variables['H2OForeignAbsorptionCoefficientsUpperAtmos'] -- : Dataset  (2, 14, 16, 2)
        # variables['RayleighExtinctionCoefficientsUpperAtmos'] ---- : Dataset  (2, 14, 5, 16)

        # varibles['KeySpeciesRatioLowerAtmos'] ------------------- : Dataset  (9,)
        # variables['KeySpeciesAbsorptionCoefficientsLowerAtmos'] -- : Dataset  (2, 14, 16, 13, 5, 9)
        # variables['AbsorptionCoefficientsLowerAtmos'] ------------ : Dataset  (2, 14, 12, 16, 19, 9)
        # variables['KeySpeciesNamesLowerAtmos'] ------------------- : Dataset  (2, 14, 3)
        # variables['TemperatureH2OForeignLowerAtmos'] ------------- : Dataset  (3,)
        # variables['PressureLowerAtmos'] -------------------------- : Dataset  (13,)
        # variables['H2OForeignAbsorptionCoefficientsLowerAtmos'] -- : Dataset  (2, 14, 16, 3)
        # variables['RayleighExtinctionCoefficientsLowerAtmos'] ---- : Dataset  (2, 14, 9, 16)

        # variables['TemperatureH2OSelf'] -------------------------- : Dataset  (10,)
        # variables['H2OSelfAbsorptionCoefficients'] --------------- : Dataset  (2, 14, 16, 10)
        # -

        f0.close()



if __name__ == '__main__':

    pass
