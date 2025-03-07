import os
import sys
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
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

import er3t

# global variables
#╭────────────────────────────────────────────────────────────────────────────╮#
name_tag = '00_er3t_shd'
fdir0    = er3t.common.fdir_examples
Ncpu    = 12
rcParams['font.size'] = 14
#╰────────────────────────────────────────────────────────────────────────────╯#


def gen_two_params_lwc_file(
        wavelength=650.0,
        solver='3D',
        overwrite=False,
        plot=True
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 1.0, 2.0, ...., 20.0
    levels    = np.linspace(0.0, 20.0, 21)

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % fdir

    # absorption object
    abs0      = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the netcdf file
    fname_nc  = '%s/data/00_er3t_mca/aux/les.nc' % (er3t.common.fdir_examples)

    # file name of the pickle file for cloud
    fname_les = '%s/les.pk' % fdir

    # cloud object
    cld0      = er3t.pre.cld.cld_les(fname_nc=fname_nc, fname=fname_les, coarsen=[2, 2, 25], overwrite=overwrite)

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate sfc_lanb.txt
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File('/Users/hchen/Work/01_libera/06_mod-lrt-bmk/aux/surface_albedo_land.h5', 'r')
    sfc_alb = f['alb_659'][...]
    f.close()

    sfc_alb[sfc_alb<0.0] = 0.0
    sfc_alb[sfc_alb>1.0] = 1.0
    Nx, Ny = sfc_alb.shape

    with open('sfc_land.txt', 'w') as f:
        f.write('L\n')
        f.write('%d %d %.4f %.4f\n' % (Nx, Ny, (48.0/Nx), (48.0/Ny)))
        for ix in np.arange(Nx):
            for iy in np.arange(Ny):
                f.write('%d %d %.2f %.6f\n' % ((ix+1), (iy+1), 285.0, sfc_alb[ix, iy]))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate les_lwc.txt
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lwc = cld0.lay['lwc']['data']*1000.0
    cer = cld0.lay['cer']['data']
    cer[cer>=25.0] = 25.0
    cer[cer<=0.5] = 0.5
    temp = cld0.lay['temperature']['data']

    Nx, Ny, Nz = lwc.shape

    with open('les_lwc.txt', 'w') as f:
        f.write('2  parameter LWC file\n')
        f.write('%d %d %d\n' % lwc.shape)
        f.write('%.4f %.4f\n' % (cld0.lay['dx']['data'], cld0.lay['dy']['data']))
        f.write('%s\n' % ' '.join([str('%.4f' % alt0) for alt0 in cld0.lay['altitude']['data']]))
        f.write('%s\n' % ' '.join([str('%.4f' % np.mean(temp[:, :, iz])) for iz in range(Nz)]))
        for ix in np.arange(Nx):
            for iy in np.arange(Ny):
                for iz in np.arange(Nz):
                    f.write('%d %d %d %.6f %.2f\n' % ((ix+1), (iy+1), (iz+1), lwc[ix, iy, iz], cer[ix, iy, iz]))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return


if __name__ == '__main__':

    gen_two_params_lwc_file()

    pass
