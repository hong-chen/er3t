import os
import numpy as np

from er3t.pre.atm import atm_atmmod



def test_atm_atmmod(fdir):

    """
    Test for module er3t.pre.atm.atm_atmmod
    """

    levels = np.linspace(0.0, 20.0, 41)
    fname_atm  = '%s/atm.pk' % fdir

    print('Case 1: run with \'levels\' only ...')
    """
    Calculation with 'levels' only without writing file.
    """
    atm_obj = atm_atmmod(levels=levels, verbose=True)
    print()


    print('Case 2: run with \'levels\' and \'fname\' but overwrite if exists ...')
    """
    Run calculation with 'levels' and overwrite data into 'fname'
    """
    atm_obj = atm_atmmod(levels=levels, fname=fname_atm, overwrite=True, verbose=True)
    print()


    print('Case 3: run with \'levels\' and \'fname\' ...')
    """
    If 'levels' and 'fname' are both provided, the module will first check whether
    the 'fname' exists or not, if
    1. 'fname' not exsit, run with 'levels' and store data into 'fname'
    2. 'fname' exsit, restore data from 'fname'
    """
    atm_obj = atm_atmmod(levels=levels, fname=fname_atm, verbose=True)
    print()


    print('Case 4: run with \'fname\' only ...')
    """
    Restore data from 'fname'
    """
    atm_obj = atm_atmmod(fname=fname_atm, verbose=True)



def test_high_res_atm(fdir):

    """
    Test for module er3t.pre.atm.atm_atmmod
    """

    levels = np.linspace(0.0, 20.0, 1001)
    fname_atm  = '%s/atm.pk' % fdir

    atm_obj = atm_atmmod(levels=levels, verbose=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from matplotlib import rcParams

    fig = plt.figure(figsize=(3, 7))
    ax1 = fig.add_subplot(111)
    ax1.scatter(atm_obj.lev['pressure']['data'], atm_obj.lev['altitude']['data'])
    ax1.set_xlabel('Pressure [hPa]')
    ax1.set_ylabel('Altitude [km]')
    plt.show()
    # ---------------------------------------------------------------------



def main():

    # create tmp-data/01 directory if it does not exist
    fdir = os.path.abspath('tmp-data/01')
    if not os.path.exists(fdir):
        os.makedirs(fdir)


    test_atm_atmmod(fdir)

    test_high_res_atm(fdir)



if __name__ == '__main__':

    main()
