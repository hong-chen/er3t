import os
import sys
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
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

__all__ = ['add_er3t_logo']

def add_er3t_logo(
        ax,
        loc=[0.05, 0.45, 0.7, 0.35],
        fname=f"{er3t.common.fdir_er3t}/docs/assets/er3t-logo.png",
        ):

    data_logo = mpl_img.imread(fname)

    ax_inset = ax.inset_axes(loc)
    ax_inset.imshow(data_logo)
    ax_inset.axis('off')

    return ax

if __name__ == '__main__':

    pass
