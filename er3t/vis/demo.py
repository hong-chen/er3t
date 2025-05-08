import datetime
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import cartopy.crs as ccrs
# mpl.use('Agg')

import er3t



def anim_phase_mie():

    cer0 = 10.0
    angles = np.linspace(0.0, 180.0, 1800)
    pha_r_ = er3t.pre.pha.pha_mie_wc_shd(wavelength=650.0, Npmom_max=1000, angles=angles, overwrite=True)
    pha_g_ = er3t.pre.pha.pha_mie_wc_shd(wavelength=550.0, Npmom_max=1000, angles=angles, overwrite=True)
    pha_b_ = er3t.pre.pha.pha_mie_wc_shd(wavelength=450.0, Npmom_max=1000, angles=angles, overwrite=True)
    index_cer = np.argmin(np.abs(pha_r_.data['ref']['data']-cer0))

    for Npmom_max in np.arange(5, 751, 5):

        pmom_r = pha_r_.data['pmom']['data'][index_cer, :Npmom_max]
        logic_r = ~np.isnan(pmom_r)
        pha_r = er3t.pre.pha.legendre2phase(pmom_r[logic_r], angle=angles, lrt=False, normalize=True, deltascaling=False)

        pmom_g = pha_g_.data['pmom']['data'][index_cer, :Npmom_max]
        logic_g = ~np.isnan(pmom_g)
        pha_g = er3t.pre.pha.legendre2phase(pmom_g[logic_g], angle=angles, lrt=False, normalize=True, deltascaling=False)

        pmom_b = pha_b_.data['pmom']['data'][index_cer, :Npmom_max]
        logic_b = ~np.isnan(pmom_b)
        pha_b = er3t.pre.pha.legendre2phase(pmom_b[logic_b], angle=angles, lrt=False, normalize=True, deltascaling=False)

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(8, 4))
            # plot1
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(111)
            ax1.plot(pha_r_.data['ang']['data'], pha_r, color='r', lw=3.0)
            ax1.plot(pha_g_.data['ang']['data'], pha_g, color='g', lw=1.8)
            ax1.plot(pha_b_.data['ang']['data'], pha_b, color='b', lw=0.9)

            ax1.plot(pha_r_.data['ang']['data'], pha_r_.data['pha']['data'][:, index_cer], color='r', lw=3.0, alpha=0.2, ls='-')
            ax1.plot(pha_g_.data['ang']['data'], pha_g_.data['pha']['data'][:, index_cer], color='g', lw=1.8, alpha=0.2, ls='-')
            ax1.plot(pha_b_.data['ang']['data'], pha_b_.data['pha']['data'][:, index_cer], color='b', lw=0.9, alpha=0.2, ls='-')

            ax1.set_xlim((-10, 190))
            ax1.set_ylim((1.0e-2, 1.0e5))
            ax1.set_yscale('log')

            ax1.xaxis.set_major_locator(FixedLocator(np.arange(0.0, 180.1, 30.0)))
            ax1.set_xlabel('Angle [$^\\circ$]')
            ax1.set_ylabel('Phase Function')
            ax1.set_title('Using %d Legendre Coefficients ...' % Npmom_max)
            #╰──────────────────────────────────────────────────────────────╯#

            # polar plot
            #╭──────────────────────────────────────────────────────────────╮#
            ax_polar = ax1.inset_axes([0.25, 0.43, 0.5, 0.5], polar=True)
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)

            ang_full = np.append(pha_r_.data['ang']['data'], 180.0+pha_r_.data['ang']['data'][1:]) % 360.0
            pha_r_full = np.append(pha_r, pha_r[:-1][::-1])
            pha_g_full = np.append(pha_g, pha_g[:-1][::-1])
            pha_b_full = np.append(pha_b, pha_b[:-1][::-1])

            ax_polar.plot(np.deg2rad(ang_full), pha_r_full, color='r', lw=0.5)
            ax_polar.plot(np.deg2rad(ang_full), pha_g_full, color='g', lw=0.5)
            ax_polar.plot(np.deg2rad(ang_full), pha_b_full, color='b', lw=0.5)
            ax_polar.set_rscale('log')
            ax_polar.set_rlim((1.0e-2, 1.0e5))
            ax_polar.tick_params(axis='x', labelsize=8, pad=-2)
            ax_polar.tick_params(axis='y', labelsize=8)
            ax_polar.set_yticks([])
            #╰──────────────────────────────────────────────────────────────╯#

            patches_legend = [
                             mpatches.Patch(color='red'   , label='650 nm'), \
                             mpatches.Patch(color='green' , label='550 nm'), \
                             mpatches.Patch(color='blue'  , label='450 nm'), \
                             ]
            ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.35, wspace=0.35)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
            fname_fig = '%3.3d_%s.png' % (Npmom_max, 'phase',)
            plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
            #╰──────────────────────────────────────────────────────────────╯#
            # plt.show()
            # sys.exit()
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#



if __name__ == '__main__':

    anim_phase_mie()

    pass
