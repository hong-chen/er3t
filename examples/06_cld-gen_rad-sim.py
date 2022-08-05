import os
import sys
import pickle
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import cartopy.crs as ccrs


from er3t.pre.cld import cld_gen_hem as cld_gen



if __name__ == '__main__':

    cld0 = cld_gen(fname='test.pk', radii=[1.0, 2.0, 8.0], weights=[0.6, 0.3, 0.1], cloud_frac_tgt=0.7, w2h_ratio=3.0, min_dist=1.5, overwrite=True)

    # =============================================================================
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(121, projection='3d')
    cmap = mpl.cm.get_cmap('jet').copy()
    cs = ax1.plot_surface(cld0.x_3d[:, :, 0], cld0.y_3d[:, :, 0], np.sum(cld0.space_3d, axis=-1), cmap=cmap, alpha=0.8, antialiased=False)
    # cs = ax1.plot_trisurf(cld0.x_3d[:, :, 0], cld0.y_3d[:, :, 0], np.sum(cld0.space_3d, axis=-1), cmap=cmap, alpha=0.8, antialiased=False)
    ax1.set_zlim((0, 10))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')

    ax2 = fig.add_subplot(122)
    cs = ax2.imshow(cld0.cot_2d.T, cmap=cmap, origin='lower')
    plt.colorbar(cs, shrink=0.75)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Cloud Optical Thickness')

    plt.subplots_adjust(wspace=0.3)
    plt.show()
    # =============================================================================


    pass
