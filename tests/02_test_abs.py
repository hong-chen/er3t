import os
import sys
import time
import numpy as np
import datetime

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


import er3t



def test_abs_16g(fdir):

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # create atm file
    levels = np.linspace(0.0, 20.0, 41)
    fname_atm  = '%s/atm.pk' % fdir
    atm0 = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)

    # create abs file, but we will need to input the created atm file
    fname_abs  = '%s/abs.pk' % fdir
    wavelength = 500.0

    print('Case 1: run with \'wavelength\' only ...')
    """
    Calculation with 'wavelength' only without writing file.
    """
    abs_obj = er3t.pre.abs.abs_16g(wavelength=wavelength, atm_obj=atm0, verbose=True)
    print()


    print('Case 2: run with \'wavelength\' and \'fname\' but overwrite if exists ...')
    """
    Run calculation with 'wavelength' and overwrite data into 'fname'
    """
    abs_obj = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=True, verbose=True)
    print()


    print('Case 3: run with \'wavelength\' and \'fname\' ...')
    """
    If 'wavelength' and 'fname' are both provided, the module will first check whether
    the 'fname' exists or not, if
    1. 'fname' not exsit, run with 'levels' and store data into 'fname'
    2. 'fname' exsit, restore data from 'fname'
    """
    abs_obj = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, verbose=True)
    print()


    print('Case 4: run with \'fname\' only ...')
    """
    Restore data from 'fname'
    """
    abs_obj = er3t.pre.abs.abs_16g(fname=fname_abs, atm_obj=atm0, verbose=True)


def test_abs_rrtmg(fdir='tmp-data'):

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # create atm file
    #/----------------------------------------------------------------------------\#
    levels = np.arange(0.5, 20.6, 1.0) # to match layer altitude with <alt_ref>
    fname_atm  = '%s/atm.pk' % fdir
    atm0 = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)
    #\----------------------------------------------------------------------------/#

    # read out abs coef for USSA (US standard atmosphere)
    # provided by Xiuhong Chen, gives <alt_ref> and <coef_ref>
    #/----------------------------------------------------------------------------\#
    Nz = 39; Nb = 14; Ng = 16
    fname    = 'data/coef_sw_ussa.txt'
    data     = np.genfromtxt(fname)
    alt_ref0  = data[:, 0]
    coef_ref0 = data[:, 1:].reshape((Nz, Nb, Ng))

    logic = (alt_ref0>=atm0.lay['altitude']['data'][0]) & (alt_ref0<=atm0.lay['altitude']['data'][-1])
    alt_ref  = alt_ref0[logic]
    coef_ref = coef_ref0[logic, ...]
    #\----------------------------------------------------------------------------/#


    # compare
    #/----------------------------------------------------------------------------\#
    # coef_ref0 = coef_ref[:, iband, :]

    iband = 0
    abs0 = er3t.pre.abs.abs_rrtmg_sw(iband=iband, atm_obj=atm0)
    print(abs0.coef['weight']['data'].sum())

    # for iband in range(14):
    #     abs0 = er3t.pre.abs.abs_rrtmg_sw(iband=iband, atm_obj=atm0)
    #\----------------------------------------------------------------------------/#
    sys.exit()



    # figures
    #/----------------------------------------------------------------------------\#
    if False:
        for i in range(Nb):

            colors = mpl.cm.jet(np.linspace(0.0, 1.0, Ng))

            # figure
            #/----------------------------------------------------------------------------\#
            plt.close('all')
            fig = plt.figure(figsize=(6, 12))
            # plot
            #/--------------------------------------------------------------\#
            ax1 = fig.add_subplot(111)

            for j in range(Ng):
                ax1.plot(coef_ref[:, i, j], alt_ref, color=colors[j, ...], lw=1.5, label='g %d' % (j+1))

            # ax1.set_xlim((0, 0.08))
            ax1.set_ylim((0, 20))
            ax1.set_xlabel('Absorption Coefficient [m$^{-1}$]')
            ax1.set_ylabel('Altitude [km]')
            ax1.set_title('Band %d' % (i+1))

            plt.legend(fontsize=16)
            #\--------------------------------------------------------------/#
            # save figure
            #/--------------------------------------------------------------\#
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            plt.savefig('%s_band-%2.2d.png' % (_metadata['Function'], i+1), bbox_inches='tight', metadata=_metadata)
            #\--------------------------------------------------------------/#
            #\----------------------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

    # create abs file, but we will need to input the created atm file
    #/----------------------------------------------------------------------------\#
    # fname_abs  = '%s/abs.pk' % fdir
    # wavelength = 500.0
    # abs_obj = abs_16g(wavelength=wavelength, atm_obj=atm0, verbose=True)
    #\----------------------------------------------------------------------------/#
    sys.exit()


if __name__ == '__main__':

    # test_abs_16g('tmp-data/abs_16g')

    test_abs_rrtmg()
