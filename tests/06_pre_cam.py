#!/bin/env python

#SBATCH --partition=shas
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hong.chen-1@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=les_04

import os
import h5py
import glob
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
import datetime
import time
from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams, ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from er3t.util.cloud import clouds

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g, abs_oco_idl
from er3t.pre.cld import cld_sat, cld_les
from er3t.pre.sfc import sfc_sat
from er3t.util.modis import modis_l1b, modis_l2, modis_09a1, grid_modis_by_extent, grid_modis_by_lonlat, download_modis_https, get_sinusoidal_grid_tag
from er3t.util import cal_r_twostream, cal_ext

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng



def run_rad_sim(f_mca, wavelength=600, fname_nc, fdir0, coarsen_factor=2, overwrite=True):

    """
    core function to run radiance simulation
    """

    fdir = '%s/%dnm' % (fdir0, wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    levels = np.arange(0.0, 20.1, 1.0)

    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)

    fname_les = '%s/les.pk' % fdir
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, altitude=atm0.lay['altitude']['data'], coarsing=[1, 1, 1, 1], overwrite=overwrite)

    # radiance 3d
    # =======================================================================================================
    target    = 'radiance'
    solver    = '3D'
    atm1d0    = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm3d0    = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)

    # coarsen
    atm3d0.nml['Atm_dx']['data'] *= coarsen_factor
    atm3d0.nml['Atm_dy']['data'] *= coarsen_factor

    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]

    mca0 = mcarats_ng(
            date=datetime.datetime(2016, 8, 29),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            target=target,
            surface_albedo=0.0,
            solar_zenith_angle=29.162360459281544,
            solar_azimuth_angle=-63.16777636586792,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            # photons=1e8*coarsen_factor,
            # photons=2e9,
            photons=1e6,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=24,
            mp_mode='py',
            overwrite=overwrite)

    out0 = mca_out_ng(fname='%s/mca-out-rad-%s_%.2fnm.h5' % (fdir0, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    rad_3d      = out0.data['rad']['data']
    rad_3d[np.isnan(rad_3d)] = 0.0
    rad_3d[rad_3d<0.0] = 0.0

    cot_true      = np.sum(cld0.lay['cot']['data'], axis=-1)
    cot_true[np.isnan(cot_true)] = 0.0
    cot_true[cot_true<0.0] = 0.0

    cot_1d      = f_mca.interp_from_rad(rad_3d)
    cot_1d[np.isnan(cot_1d)] = 0.0
    cot_1d[cot_1d<0.0] = 0.0
    # =======================================================================================================

    fname_new = 'data/data_%s_coa-fac-%d_%dnm.h5' % (os.path.basename(fdir0), coarsen_factor, wavelength)
    f = h5py.File(fname_new, 'w')

    f['cot_true'] = cot_true
    f['rad_3d']   = rad_3d
    f['cot_1d']   = cot_1d

    f.close()


class aircraft:

    """
    create an aircraft object
    """

    def __init__(self):

        pass



if __name__ == '__main__':

    # step 1
    # create an aircraft object
    # =============================================================================
    aircraft0 = aircraft()
    # =============================================================================

    fname = 'data/les.nc'

    # step 2
    # run ERT for LES scenes at specified coarsening factor
    # (spatial resolution depends on coarsening factor)
    # =============================================================================
    # for coarsen_factor in [1, 2, 4]:
    #     main_les(coarsen_factor=coarsen_factor)
    # =============================================================================

    pass
