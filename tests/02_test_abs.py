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


def test_abs_reptran(fdir='tmp-data'):

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    wavelength = 770.0

    # create atm file
    #/----------------------------------------------------------------------------\#
    levels = np.arange(0.5, 20.6, 1.0) # to match layer altitude with <alt_ref>
    fname_atm  = '%s/atm.pk' % fdir
    atm0 = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)
    alt0 = atm0.lay['altitude']['data']
    #\----------------------------------------------------------------------------/#

    # create abs file
    #/----------------------------------------------------------------------------\#
    # fname_abs  = '%s/abs.pk' % fdir
    # abs0 = er3t.pre.abs.abs_16g(wavelength=wavelength, atm_obj=atm0, overwrite=True)
    # coef0 = np.zeros_like(atm0.lay['altitude']['data'])
    # for i in range(coef0.size):
    #     coef0[i] = (abs0.coef['abso_coef']['data'][i, :] * abs0.coef['weight']['data'] * abs0.coef['slit_func']['data'][i, :]).sum()
    # print(coef0)
    #\----------------------------------------------------------------------------/#

    # reptran
    #/----------------------------------------------------------------------------\#
    wvls  = np.arange(300.0, 2401.0, 2.0)
    coef1 = np.zeros((wvls.size, alt0.size), dtype=np.float64)
    for i, wavelength in enumerate(wvls):
        try:
            print(i, wavelength)
            # abs1 = er3t.pre.abs.abs_rep(wavelength=wavelength, target='modis', atm_obj=atm0, band_name='modis_aqua_b01')
            abs1 = er3t.pre.abs.abs_rep(wavelength=wavelength, target='coarse', atm_obj=atm0)
            for j in range(alt0.size):
                coef1[i, j] = (abs1.coef['abso_coef']['data'][j, :] * abs1.coef['weight']['data']).sum()
        except Exception as error:
            print(error)
    #\----------------------------------------------------------------------------/#

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvls, coef1[:, 0], color='r')
        ax1.plot(wvls, coef1[:, 10], color='g')
        ax1.plot(wvls, coef1[:, -1], color='b')
        # ax1.plot(coef0, alt0, c='k', lw=1.0)
        # ax1.plot(coef1, alt0, c='r', lw=1.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#

    # print(coef0)
    # print(abs0.coef['solar']['data'])
    # print(coef1)
    # print(abs1.coef['solar']['data'])

    # abs0 = er3t.pre.abs.abs_rep(wavelength=650.0, target='modis')
    # abs0 = er3t.pre.abs.abs_rep(wavelength=650.0, target='modis', atm_obj=atm0)
    # abs0 = er3t.pre.abs.abs_rep(wavelength=650.0, target='modis', atm_obj=atm0, band_name='modis_terra_b15')
    # abs0 = er3t.pre.abs.abs_rep(wavelength=760.0, target='coarse', atm_obj=atm0)


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



if __name__ == '__main__':

    # test_abs_16g('tmp-data/abs_16g')

    # test_abs_rrtmg()

    test_abs_reptran()
