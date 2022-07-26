import os
import time
import numpy as np
import datetime

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g, abs_oco_idl
from er3t.util import cal_sol_fac



def test_abs_16g(fdir):

    # create atm file
    levels = np.linspace(0.0, 20.0, 41)
    fname_atm  = '%s/atm.pk' % fdir
    atm0 = atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)

    # create abs file, but we will need to input the created atm file
    fname_abs  = '%s/abs.pk' % fdir
    wavelength = 500.0

    print('Case 1: run with \'wavelength\' only ...')
    """
    Calculation with 'wavelength' only without writing file.
    """
    abs_obj = abs_16g(wavelength=wavelength, atm_obj=atm0, verbose=True)
    print()


    print('Case 2: run with \'wavelength\' and \'fname\' but overwrite if exists ...')
    """
    Run calculation with 'wavelength' and overwrite data into 'fname'
    """
    abs_obj = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=True, verbose=True)
    print()


    print('Case 3: run with \'wavelength\' and \'fname\' ...')
    """
    If 'wavelength' and 'fname' are both provided, the module will first check whether
    the 'fname' exists or not, if
    1. 'fname' not exsit, run with 'levels' and store data into 'fname'
    2. 'fname' exsit, restore data from 'fname'
    """
    abs_obj = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, verbose=True)
    print()


    print('Case 4: run with \'fname\' only ...')
    """
    Restore data from 'fname'
    """
    abs_obj = abs_16g(fname=fname_abs, atm_obj=atm0, verbose=True)



def test_abs_oco(fdir):

    # create atm file
    levels = np.linspace(0.0, 20.0, 41)
    fname_atm  = '%s/atm.pk' % fdir
    atm0 = atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)

    # create abs file, but we will need to input the created atm file
    fname_abs  = '%s/abs.pk' % fdir
    wavelength = 760.0

    print('Case 1: run with \'wavelength\' only ...')
    """
    Calculation with 'wavelength' only without writing file.
    """
    abs_obj = abs_oco_idl(wavelength=wavelength, atm_obj=atm0, verbose=True)
    print()


    print('Case 2: run with \'wavelength\' and \'fname\' but overwrite if exists ...')
    """
    Run calculation with 'wavelength' and overwrite data into 'fname'
    """
    abs_obj = abs_oco_idl(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=True, verbose=True)
    print()


    print('Case 3: run with \'wavelength\' and \'fname\' ...')
    """
    If 'wavelength' and 'fname' are both provided, the module will first check whether
    the 'fname' exists or not, if
    1. 'fname' not exsit, run with 'levels' and store data into 'fname'
    2. 'fname' exsit, restore data from 'fname'
    """
    abs_obj = abs_oco_idl(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, verbose=True)
    print()


    print('Case 4: run with \'fname\' only ...')
    """
    Restore data from 'fname'
    """
    abs_obj = abs_oco_idl(fname=fname_abs, atm_obj=atm0, verbose=True)



def test_solar_spectra(fdir, date=datetime.datetime.now()):

    levels = np.linspace(0.0, 20.0, 41)
    atm0   = atm_atmmod(levels=levels)

    wvls   = np.arange(300.0, 2301.0, 1.0)
    sols   = np.zeros_like(wvls)

    sol_fac = cal_sol_fac(date)

    for i, wvl in enumerate(wvls):

        abs0 = abs_16g(wavelength=wvl, atm_obj=atm0)
        norm = sol_fac/(abs0.coef['weight']['data']*abs0.coef['slit_func']['data'][-1, :]).sum()
        sols[i] = norm*(abs0.coef['solar']['data']*abs0.coef['weight']['data']*abs0.coef['slit_func']['data'][-1, :]).sum()


    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from matplotlib import rcParams
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(wvls, sols, s=2)
    ax1.set_xlim((200, 2400))
    ax1.set_ylim((0.0, 2.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('Solar Spectra')
    plt.savefig('solar_spectra.png')
    plt.show()
    # ---------------------------------------------------------------------



def main():

    # create tmp-data/02 directory if it does not exist
    fdir = os.path.abspath('tmp-data/02')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # test_abs_16g(fdir)

    # test_abs_oco(fdir)

    test_solar_spectra(fdir)



if __name__ == '__main__':


    main()
