# import os
# import sys
# import pickle
# import multiprocessing as mp
# import h5py
# import copy
# import numpy as np

import er3t.common
from er3t.pre.atm import atm_atmmod
from er3t.util import all_files

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
                 wavelength = None,  \
                 fname      = None,  \
                 atm_obj    = None,  \
                 overwrite  = False, \
                 verbose    = False):

        self.iband   = iband
        self.atm_obj = atm_obj

        self.load_data(iband)

    def load_data(self, iband):

        f0 = Dataset(self.fname, 'r')

        # Ngas  = 12
        # Nband = 14
        # variables['AbsorberNames'] ------------------------------- : Dataset  (12, 5)

        # read out gas names <gases>
        #/----------------------------------------------------------------------------\#
        gas_bytes = f0.variables['AbsorberNames'][:]
        Ngas, Nchar = gas_bytes.shape
        gases = []
        for i in range(Ngas):
            gas_name0 = ''.join([j.decode('utf-8') for j in gas_bytes[i, :]]).strip()
            if len(gas_name0) > 0:
                gases.append(gas_name0)
        #\----------------------------------------------------------------------------/#

        # read out gas names <gases>
        #/----------------------------------------------------------------------------\#
        key_spec_low_bytes = f0.variables['KeySpeciesNamesLowerAtmos'][:][:, iband, :]
        key_spec_upp_bytes = f0.variables['KeySpeciesNamesUpperAtmos'][:][:, iband, :]
        Nkey, Nchar = key_spec_upp_bytes.shape
        key_spec_low = []
        key_spec_upp = []
        for i in range(Nkey):
            key_spec_low_name0 = ''.join([j.decode('utf-8') for j in key_spec_low_bytes[i, :]]).strip()
            key_spec_upp_name0 = ''.join([j.decode('utf-8') for j in key_spec_upp_bytes[i, :]]).strip()
            if len(key_spec_low_name0) > 0:
                key_spec_low.append(key_spec_low_name0)
            if len(key_spec_upp_name0) > 0:
                key_spec_upp.append(key_spec_upp_name0)
        #\----------------------------------------------------------------------------/#

        print(key_spec_low)
        print(key_spec_upp)
        sys.exit()


        # band
        #/----------------------------------------------------------------------------\#
        wvln_min = f0.variables['BandWavenumberLowerLimit'][:][iband]
        wvln_max = f0.variables['BandWavenumberUpperLimit'][:][iband]
        wvl_min = 1.0e7/wvln_max
        wvl_max = 1.0e7/wvln_min

        band_range = (wvl_min, wvl_max)

        self.wavelength = 2.0e7/(wvln_min+wvln_max)
        self.band_range = band_range
        #\----------------------------------------------------------------------------/#


        # Gs
        #/----------------------------------------------------------------------------\#
        Ng = f0.variables['NumGPoints'][:][:, iband]
        #\----------------------------------------------------------------------------/#


        # profile
        #/----------------------------------------------------------------------------\#
        t_ref = f0.variables['ReferenceTemperature'][:]
        p     = f0.variables['Pressure'][:]
        #\----------------------------------------------------------------------------/#

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.scatter(t_ref, p, s=6, c='k', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # add colorbar
        #/--------------------------------------------------------------\#
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('right', '5%', pad='3%')
        # cbar = fig.colorbar(cs, cax=cax)
        # cbar.set_label('', rotation=270, labelpad=4.0)
        # cbar.set_ticks([])
        # cax.axis('off')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # plt.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # plt.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
        #\----------------------------------------------------------------------------/#
        print(p)
        print(t_ref)
        #\----------------------------------------------------------------------------/#

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

        # variables['KeySpeciesAbsorptionCoefficientsLowerAtmos'] -- : Dataset  (2, 14, 16, 13, 5, 9)
        # variables['KeySpeciesAbsorptionCoefficientsUpperAtmos'] -- : Dataset  (2, 14, 16, 47, 5, 5)

        # variables['AbsorptionCoefficientsLowerAtmos'] ------------ : Dataset  (2, 14, 12, 16, 19, 9)
        # variables['AbsorptionCoefficientsUpperAtmos'] ------------ : Dataset  (2, 14, 12, 16, 19, 5)

        # variables['ReferenceTemperature'] ------------------------ : Dataset  (59,)
        # variables['Pressure'] ------------------------------------ : Dataset  (59,)
        # variables['LogPressure'] --------------------------------- : Dataset  (59,)

        # variables['KeySpeciesNamesLowerAtmos'] ------------------- : Dataset  (2, 14, 3)
        # variables['KeySpeciesNamesUpperAtmos'] ------------------- : Dataset  (2, 14, 3)

        # variables['KeySpeciesRatioUpperAtmos'] ------------------- : Dataset  (5,)
        # variables['TemperatureH2OForeignUpperAtmos'] ------------- : Dataset  (2,)
        # variables['PressureUpperAtmos'] -------------------------- : Dataset  (47,)
        # variables['H2OForeignAbsorptionCoefficientsUpperAtmos'] -- : Dataset  (2, 14, 16, 2)
        # variables['NRLSSI2SSFFacularUpperAtmos'] ----------------- : Dataset  (2, 14, 5, 16)
        # variables['NRLSSI2SSFQuietSunUpperAtmos'] ---------------- : Dataset  (2, 14, 5, 16)
        # variables['NRLSSI2SSFSunspotUpperAtmos'] ----------------- : Dataset  (2, 14, 5, 16)
        # variables['SolarSourceFunctionUpperAtmos'] --------------- : Dataset  (2, 14, 5, 16)
        # variables['RayleighExtinctionCoefficientsUpperAtmos'] ---- : Dataset  (2, 14, 5, 16)

        # variables['KeySpeciesRatioLowerAtmos'] ------------------- : Dataset  (9,)
        # variables['TemperatureH2OForeignLowerAtmos'] ------------- : Dataset  (3,)
        # variables['PressureLowerAtmos'] -------------------------- : Dataset  (13,)
        # variables['H2OForeignAbsorptionCoefficientsLowerAtmos'] -- : Dataset  (2, 14, 16, 3)
        # variables['NRLSSI2SSFFacularLowerAtmos'] ----------------- : Dataset  (2, 14, 9, 16)
        # variables['NRLSSI2SSFQuietSunLowerAtmos'] ---------------- : Dataset  (2, 14, 9, 16)
        # variables['NRLSSI2SSFSunspotLowerAtmos'] ----------------- : Dataset  (2, 14, 9, 16)
        # variables['SolarSourceFunctionLowerAtmos'] --------------- : Dataset  (2, 14, 9, 16)
        # variables['RayleighExtinctionCoefficientsLowerAtmos'] ---- : Dataset  (2, 14, 9, 16)

        # variables['TemperatureH2OSelf'] -------------------------- : Dataset  (10,)
        # variables['H2OSelfAbsorptionCoefficients'] --------------- : Dataset  (2, 14, 16, 10)
        # -
        pass




if __name__ == '__main__':

    from er3t.pre.atm import atm_atmmod

    levels = np.arange(0.0, 20.1, 0.5)
    atm0 = atm_atmmod(levels=levels, overwrite=True)

    abs0 = abs_rrtmg_sw(atm_obj=atm0)

    pass
