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
                 iband = 0,
                 ig    = 0,
                 wavelength = None,  \
                 fname      = None,  \
                 atm_obj    = None,  \
                 overwrite  = False, \
                 verbose    = False):

        if ig != 0:
            msg = '\nError [abs_rrtmg_sw]: currently only <ig=0> is supported.'
            raise OSError(msg)

        self.iband   = iband
        self.ig      = ig
        self.atm_obj = atm_obj

        self.load_data(iband, ig)

    def load_data(self, iband, ig):

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
            gas_name0 = ''.join([j.decode('utf-8') for j in gas_bytes[i, :]]).strip()
            if len(gas_name0) > 0:
                gases.append(gas_name0)
        #\----------------------------------------------------------------------------/#
        print(gases)


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
            key_gas_low_name0 = ''.join([j.decode('utf-8') for j in key_gas_low_bytes[i, :]]).strip()
            key_gas_upp_name0 = ''.join([j.decode('utf-8') for j in key_gas_upp_bytes[i, :]]).strip()
            if len(key_gas_low_name0) > 0:
                ikey_gas_low.append(gases.index(key_gas_low_name0))
            if len(key_gas_upp_name0) > 0:
                ikey_gas_upp.append(gases.index(key_gas_upp_name0))

        key_gas_low = [gases[i] for i in ikey_gas_low]
        key_gas_upp = [gases[i] for i in ikey_gas_upp]
        #\----------------------------------------------------------------------------/#
        print(key_gas_low)
        print(key_gas_upp)


        # Gs
        #/----------------------------------------------------------------------------\#
        Ng = f0.variables['NumGPoints'][:][ig, iband]
        #\----------------------------------------------------------------------------/#

        # solar
        #/----------------------------------------------------------------------------\#
        sol_upp = f0['SolarSourceFunctionUpperAtmos'][:][ig, iband, :, :Ng]
        sol_low = f0['SolarSourceFunctionLowerAtmos'][:][ig, iband, :, :Ng]
        #\----------------------------------------------------------------------------/#

        # weights
        #/----------------------------------------------------------------------------\#
        wgt =  np.array([ \
              0.1527534276, 0.1491729617, 0.1420961469, \
              0.1316886544, 0.1181945205, 0.1019300893, \
              0.0832767040, 0.0626720116, 0.0424925000, \
              0.0046269894, 0.0038279891, 0.0030260086, \
              0.0022199750, 0.0014140010, 0.0005330000, \
              0.0000750000 \
              ])

        """
          igcsm = 0
          do ibnd = 1,nbndsw
             iprsm = 0
             if (ngc(ibnd).lt.mg) then
                do igc = 1,ngc(ibnd)
                   igcsm = igcsm + 1
                   wtsum = 0.
                   do ipr = 1, ngn(igcsm)
                      iprsm = iprsm + 1
                      wtsum = wtsum + wt(iprsm)
                   enddo
                   wtsm(igc) = wtsum
                enddo
                do ig = 1, ng(ibnd+15)
                   ind = (ibnd-1)*mg + ig
                   rwgt(ind) = wt(ig)/wtsm(ngm(ind))
                enddo
             else
                do ig = 1, ng(ibnd+15)
                   igcsm = igcsm + 1
                   ind = (ibnd-1)*mg + ig
                   rwgt(ind) = 1.0_rb
                enddo
             endif
          enddo
        """

        #\----------------------------------------------------------------------------/#


        # coef
        #/----------------------------------------------------------------------------\#
        dt = f0.variables['TemperatureDiffFromMLS'][:]

        # upper atm
        #/--------------------------------------------------------------\#
        p_upp   = f0.variables['PressureUpperAtmos'][:]
        mr_upp0 = f0.variables['KeySpeciesRatioUpperAtmos'][:] # for some reason, Python netCDF4 library cannot read the value for this variable correctly
        mr_upp  = np.linspace(0.0, 1.0, mr_upp0.size)

        coef_upp = f0.variables['AbsorptionCoefficientsUpperAtmos'][:][ig, iband, ikey_gas_upp, :Ng, :, :]
        coef_key_upp = f0.variables['KeySpeciesAbsorptionCoefficientsUpperAtmos'][:][ig, iband, :Ng, :, :, :]
        #\--------------------------------------------------------------/#

        # lower atm
        #/--------------------------------------------------------------\#
        p_low   = f0.variables['PressureLowerAtmos'][:]
        mr_low0 = f0.variables['KeySpeciesRatioLowerAtmos'][:] # for some reason, Python netCDF4 library cannot read the value for this variable correctly
        mr_low  = np.linspace(0.0, 1.0, mr_low0.size)

        coef_low = f0.variables['AbsorptionCoefficientsLowerAtmos'][:][ig, iband, ikey_gas_low, :Ng, :, :]
        coef_key_low = f0.variables['KeySpeciesAbsorptionCoefficientsLowerAtmos'][:][ig, iband, :Ng, :, :, :]
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#

        print('Delta Temperature')
        print(dt.shape)
        print('Pressure [Upper]')
        print(p_upp.shape)
        print('Ratio [Upper]')
        print(mr_upp.shape)
        print(coef_upp.shape)
        print(coef_key_upp.shape)
        print()

        print('Pressure [Lower]')
        print(p_low.shape)
        print('Ratio [Lower]')
        print(mr_low.shape)
        print(coef_low.shape)
        print(coef_key_low.shape)
        print()


        print('Band #%d' % (iband+1))
        print('Center wavelength: %.4fnm' % self.wavelength)
        print('Wavelength range: %.4f - %.4fnm' % self.band_range)
        print('Number of Gs: ', Ng)
        print('Lower atm species: ', key_gas_low)
        print('Upper atm species: ', key_gas_upp)
        print('-'*40)

        # Ngas  = 12
        # Nband = 14
        # variables['AbsorberNames'] ------------------------------- : Dataset  (12, 5)




        # profile
        #/----------------------------------------------------------------------------\#
        t_ref = f0.variables['ReferenceTemperature'][:]
        p     = f0.variables['Pressure'][:]
        #\----------------------------------------------------------------------------/#

        # # figure
        # #/----------------------------------------------------------------------------\#
        # plt.close('all')
        # fig = plt.figure(figsize=(8, 6))
        # #/--------------------------------------------------------------\#
        # ax1 = fig.add_subplot(111)
        # ax1.scatter(t_ref, p, s=6, c='k', lw=0.0)
        # #\--------------------------------------------------------------/#
        # plt.show()
        # sys.exit()
        # #\----------------------------------------------------------------------------/#
        # #\----------------------------------------------------------------------------/#

        f0.close()



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
        pass




if __name__ == '__main__':


    pass
