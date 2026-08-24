import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

from pre_atm import ATMOSPHERE


class AEROSOL:

    """
    Input:
        aod: aerosol optical thickness
        ssa: single scattering albedo
        asy: asymmetry parameter

    Output:

        aer_ext

    """

    ID = 'Aerosol'

    def __init__(
            self            , \
            aod       = None, \
            ssa       = None, \
            asy       = None, \
            fname     = None, \
            fname_atm = None, \
            overwrite = False,\
            verbose   = True
            ):

        self.verbose = verbose

        atm0 = ATMOSPHERE(fname=fname_atm, verbose=self.verbose)
        self.coef = {}


if __name__ == '__main__':

    aer = AEROSOL()
