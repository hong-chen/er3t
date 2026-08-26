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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


def plot_flux3d(i, iz):
    # read h5 file
    # ╭────────────────────────────────────────────────────────────────────────────╮#
    fname = "/Users/hchen/Work/mygit/er3t/examples/tmp-data/00_er3t_mca/example_02_flux_les_cloud_3d/mca-out-flux-3d_example_02_flux_les_cloud_3d.h5"
    h5f = h5py.File(fname, "r")
    f_down = h5f["mean/f_down"][...]
    f_up = h5f["mean/f_up"][...]
    h5f.close()
    # ╰────────────────────────────────────────────────────────────────────────────╯#

    f_net = f_down

    # figure
    # ╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close("all")
        fig = plt.figure(figsize=(12, 12))
        # plot1
        # ╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111, projection="3d")

        # plot surface
        # ╭────────────────────────────────────────────────────────────────────────────╮#
        x = np.linspace(0.0, 48.0, 480)
        y = np.linspace(0.0, 48.0, 480)
        z = np.linspace(0.0, 20.0, 21)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.zeros_like(xx)

        flux = f_net[:, :, iz]
        zz[...] = z[iz]

        cmap = mpl.colormaps["jet"].copy()
        cmap.set_under("white")
        norm = mcolors.Normalize(vmin=0.0, vmax=0.8)
        colors = cmap(norm(flux), alpha=0.1)
        ax1.plot_surface(
            xx, yy, zz, facecolors=colors, shade=False, rstride=1, cstride=1, zorder=0
        )
        # ╰────────────────────────────────────────────────────────────────────────────╯#

        # plot 4km
        # ╭────────────────────────────────────────────────────────────────────────────╮#
        # zz[...] = 4.0

        # flux = f_net[:, :, 4]

        # cmap = mpl.colormaps['jet'].copy()
        # cmap.set_under('white')
        # norm = mcolors.Normalize(vmin=0.0, vmax=2.0)
        # colors = cmap(norm(flux))
        # ax1.plot_surface(xx, yy, zz, facecolors=colors, shade=False, rstride=1, cstride=1)
        # ╰────────────────────────────────────────────────────────────────────────────╯#

        # plot X (fixed-y)
        # ╭────────────────────────────────────────────────────────────────────────────╮#
        x = np.linspace(0.0, 48.0, 480)
        z = np.linspace(0.0, 20.0, 21)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        yy = np.zeros_like(zz)

        flux = f_net[:, 240, :]
        yy[...] = 24.0

        cmap = mpl.colormaps["jet"].copy()
        cmap.set_under("white")
        norm = mcolors.Normalize(vmin=0.0, vmax=0.8)
        colors = cmap(norm(flux), alpha=0.1)
        ax1.plot_surface(
            xx, yy, zz, facecolors=colors, shade=False, rstride=1, cstride=1, zorder=1
        )
        # ╰────────────────────────────────────────────────────────────────────────────╯#

        # plot Y (fixed-x)
        # ╭────────────────────────────────────────────────────────────────────────────╮#
        y = np.linspace(0.0, 48.0, 480)
        z = np.linspace(0.0, 20.0, 21)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        xx = np.zeros_like(zz)

        flux = f_net[100, :, :]
        xx[...] = 10.0

        cmap = mpl.colormaps["jet"].copy()
        cmap.set_under("white")
        norm = mcolors.Normalize(vmin=0.0, vmax=0.8)
        colors = cmap(norm(flux), alpha=0.1)
        ax1.plot_surface(
            xx, yy, zz, facecolors=colors, shade=False, rstride=1, cstride=1, zorder=2
        )
        # ╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.set_xlim((0, 48))
        ax1.set_ylim((0, 48))
        ax1.set_zlim((0, 20))
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=30.0, azim=-35.0, roll=0)
        # ╰──────────────────────────────────────────────────────────────╯#

        # save figure
        # ╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {
            "Computer": os.uname()[1],
            "Script": os.path.abspath(__file__),
            "Function": sys._getframe().f_code.co_name,
            "Date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
        fname_fig = f"{i:02d}_flux3d.png"
        plt.savefig(
            fname_fig, bbox_inches="tight", metadata=_metadata_, transparent=False
        )
        # ╰──────────────────────────────────────────────────────────────╯#
        plt.close(fig)
        plt.clf()
    # ╰────────────────────────────────────────────────────────────────────────────╯#


def plot_grid(i, iz):
    # figure
    # ╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close("all")
        fig = plt.figure(figsize=(12, 12))

        # plot1
        # ╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111, projection="3d")

        for i in range(3):
            for j in range(3):
                if (i == 0) or (j == 2):
                    ax1.plot([i, i], [0, 2], [j, j], color="k", ls="-")
                else:
                    ax1.plot([i, i], [0, 2], [j, j], color="gray", ls=":")

        for i in range(3):
            for j in range(3):
                if (i == 0) or (j == 0):
                    ax1.plot([i, i], [j, j], [0, 2], color="k", ls="-")
                else:
                    ax1.plot([i, i], [j, j], [0, 2], color="gray", ls=":")

        for i in range(3):
            for j in range(3):
                if (i == 0) or (j == 2):
                    ax1.plot([0, 2], [i, i], [j, j], color="k", ls="-")
                else:
                    ax1.plot([0, 2], [i, i], [j, j], color="gray", ls=":")

        ax1.scatter(1, 1, 1, marker="o", s=100, color="red")

        ax1.grid(False)
        ax1.axis("off")

        ax1.set_xlim((0, 2))
        ax1.set_ylim((0, 2))
        ax1.set_zlim((0, 2))
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=10.0, azim=250.0, roll=0)
        # ╰──────────────────────────────────────────────────────────────╯#

        ax1.xaxis.set_major_locator(FixedLocator(np.arange(2)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(2)))
        ax1.zaxis.set_major_locator(FixedLocator(np.arange(2)))

        # save figure
        # ╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {
            "Computer": os.uname()[1],
            "Script": os.path.abspath(__file__),
            "Function": sys._getframe().f_code.co_name,
            "Date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(
            fname_fig, bbox_inches="tight", metadata=_metadata_, transparent=False
        )
        # ╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        # plt.close(fig)
        # plt.clf()
    # ╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == "__main__":
    # z = np.arange(5)
    # z = np.concatenate((z, z[::-1][1:]))
    # for i, iz in enumerate(z):
    #     plot_flux3d(i, iz)
    # pass

    plot_grid(1, 1)
