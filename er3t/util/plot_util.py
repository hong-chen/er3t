"""
plot_util.py

This module contains utility functions and colormaps for plotting data related to the SIF SAT project.
It includes functions for setting plot fonts and predefined colormaps.

"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as mplt
import matplotlib.font_manager as font_manager
import cartopy
import cartopy.crs as ccrs

import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# parent directory
er3t_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

# set matplotlib style
MPL_STYLE_PATH = os.path.join(er3t_dir, 'util/plotting_utils/er3t_plots.mplstyle')


def set_plot_fonts(plt, serif_style='sans-serif', font='Helvetica Neue'):
    """
    Set the fonts for matplotlib plots.
    This function configures the font settings for matplotlib plots by adding
    custom fonts from a specified directory and setting the global font family.

    Args:
    ----
        plt : module
            The matplotlib.pyplot module.
        serif_style : str, optional
            The style of the font family to use, one of 'serif' or 'sans-serif'.
        font : str, optional
            The name of the font to use (default is 'Helvetica Neue').
    """

    # look for fonts in the fonts directory
    all_fonts_dir = os.path.join(er3t_dir, 'util/plotting_utils/fonts/')
    font_dir = [os.path.join(all_fonts_dir, f) for f in os.listdir(all_fonts_dir) if os.path.isdir(os.path.join(all_fonts_dir, f))]
    # add font if available from user's environment
    if 'MPL_FONT_DIR' in os.environ.keys():
        font_dir.append(os.environ['MPL_FONT_DIR'])

    # add all the fonts to matplotlib's library
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)

    # set font family globally (in-place for plt)
    plt.rc('font', **{'family':'{}'.format(serif_style),
                      '{}'.format(serif_style):['{}'.format(font)]})


def add_ancillary(ax, title=None, scale=1, dx=20, dy=5, cartopy_black=False, ccrs_data=None, coastline=True, ocean=True, gridlines=True, land=True, x_fontcolor='black', y_fontcolor='black', y_inline=True, x_inline=False, zorders={'land': 0, 'ocean': 1, 'coastline': 2, 'gridlines': 2}, colors=None):
    """
    Add cartopy features and styling elements (title, ocean/land color, coastlines, gridlines) to a cartopy map plot.

    Args:
    ----
        ax: A matplotlib or cartopy axes object where the features will be drawn.
        title (str, optional): The title of the plot. Defaults to None.
        scale (float, optional): A scaling factor for text size. Defaults to 1.
        dx (int, optional): Longitude spacing in degrees. Defaults to 20.
        dy (int, optional): Latitude spacing in degrees. Defaults to 5.
        cartopy_black (bool, optional): Whether to use a black color scheme for background
            and cartographic features. Defaults to False.
        ccrs_data (cartopy.crs, optional): Coordinate reference system to use
            for the plot. Defaults to ccrs.PlateCarree().
        coastline (bool, optional): Whether to draw coastlines. Defaults to True.
        ocean (bool, optional): Whether to fill ocean areas. Defaults to True.
        gridlines (bool, optional): Whether to draw gridlines. Defaults to True.
        land (bool, optional): Whether to fill land areas. Defaults to True.
        x_fontcolor (str, optional): Font color for x-axis gridline labels. Defaults to 'black'.
        y_fontcolor (str, optional): Font color for y-axis gridline labels. Defaults to 'black'.
        y_inline (bool, optional): Flag to decide whether latitude labels are drawn inline or not. Defaults to True.
        zorders (dict, optional): Z-order values for different features (land, ocean, coastline,
            gridlines). Defaults to {'land': 0, 'ocean': 1, 'coastline': 2, 'gridlines': 2}.
        colors (dict, optional): Color mappings for features like ocean, land, coastline,
            title, and background. If None, defaults are used.

    Returns:
    -------
        None, modifies axis in-place
    """

    if ccrs_data is None:
        ccrs_data = ccrs.PlateCarree()

    # set title
    if title is not None:
        title_fontsize = int(18 * scale)
        ax.set_title(title, pad=7.5, fontsize=title_fontsize, fontweight="bold")

    if colors is None:
        if cartopy_black:
            colors = {'ocean':'black', 'land':'black', 'coastline':'black', 'title':'white', 'background':'black'}

        else:
            colors = {'ocean':'aliceblue', 'land':'#fcf4e8', 'coastline':'black', 'title':'black', 'background':'white'}

    if ocean:
        ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), zorder=zorders['ocean'], facecolor=colors['ocean'], edgecolor='none')

    if land:
        ax.add_feature(cartopy.feature.LAND.with_scale('10m'), zorder=zorders['land'], facecolor=colors['land'], edgecolor='none')

    if coastline:
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), zorder=zorders['coastline'], edgecolor=colors['coastline'], linewidth=1, alpha=1)

    if gridlines:
        gl = ax.gridlines(linewidth=1.5, color='darkgray',
                    draw_labels=True, zorder=zorders['gridlines'], alpha=0.75, linestyle=(0, (1, 1)),
                    x_inline=x_inline, y_inline=y_inline, crs=ccrs_data)

        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, dx))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 90, dy))
        gl.xlabel_style = {'size': int(12 * scale), 'color': x_fontcolor}
        gl.ylabel_style = {'size': int(12 * scale), 'color': y_fontcolor}
        gl.rotate_labels = False
        gl.top_labels    = False
        gl.right_labels  = False
        gl.xpadding = 7.5
        gl.ypadding = 7.5

    # for spine in ax.spines.values():
    #     if cartopy_black:
    #         spine.set_edgecolor('white')
    #     else:
    #         spine.set_edgecolor('black')

    #     spine.set_linewidth(1)


ccrs_geog = ccrs.PlateCarree()
ccrs_ortho = ccrs.Orthographic(central_latitude=84, central_longitude=-50)
ccrs_nearside = ccrs.NearsidePerspective(central_latitude=84, central_longitude=-50, satellite_height=500e3)

############################## Cloud phase IR colormap ##############################
ctp_ir_cmap_arr = np.array([
                            [0.5, 0.5, 0.5, 1.], # clear
                            [0., 0., 0.55, 1.], # liquid
                            [0.75, 0.85, 0.95, 1.], # ice
                            [0.55, 0.55, 0.95, 1.], # mixed
                            [0., 0.95, 0.95, 1.]])# undet. phase
ctp_ir_cmap_ticklabels = np.array(["clear", "liquid", "ice", "mixed phase", "uncertain"])
ctp_ir_tick_locs = (np.arange(len(ctp_ir_cmap_ticklabels)) + 0.5)*(len(ctp_ir_cmap_ticklabels) - 1)/len(ctp_ir_cmap_ticklabels)
ctp_ir_cmap = matplotlib.colors.ListedColormap(ctp_ir_cmap_arr)
ctp_ir_cmap.set_bad("black", 1)


############################## Cloud phase SWIR/COP colormap ##############################
ctp_swir_cmap_arr = np.array([
                            #   [0, 0, 0, 1], # undet. mask
                              [0.5, 0.5, 0.5, 1.], # clear
                              [0., 0., 0.55, 1.], # liquid
                              [0.75, 0.85, 0.95, 1.], # ice
                              [0., 0.95, 0.95, 1.]])# no phase (liquid)
ctp_swir_cmap_ticklabels = np.array(["clear", "liquid", "ice", "uncertain"])
ctp_swir_tick_locs = (np.arange(len(ctp_swir_cmap_ticklabels)) + 0.5)*(len(ctp_swir_cmap_ticklabels) - 1)/len(ctp_swir_cmap_ticklabels)
ctp_swir_cmap = matplotlib.colors.ListedColormap(ctp_swir_cmap_arr)
# ctp_swir_cmap.set_bad("black", 1)


############################## Cloud top height colormap ##############################
cth_cmap_arr = np.array([[0., 0., 0., 1], # no retrieval
                        [0.5, 0.5, 0.5, 1], # clear
                        [0.05, 0.7, 0.95, 1], # low clouds
                        [0.65, 0.05, 0.3, 1.],  # mid clouds
                        [0.95, 0.95, 0.95, 1.]])    # high clouds
cth_cmap_ticklabels = ["undet.", "clear", "low\n0.1 - 2 km", "mid\n2 - 6 km", "high\n>=6 km"]
cth_tick_locs = (np.arange(len(cth_cmap_ticklabels)) + 0.5)*(len(cth_cmap_ticklabels) - 1)/len(cth_cmap_ticklabels)
cth_cmap = matplotlib.colors.ListedColormap(cth_cmap_arr)

############################## Cloud top temperature colormap ##############################
ctt_cmap_arr = np.array(list(mplt.get_cmap('Blues_r')(np.linspace(0, 0.8, 4))) + list(mplt.get_cmap('Reds')(np.linspace(0, 1, 4))))
ctt_cmap = matplotlib.colors.ListedColormap(ctt_cmap_arr)


arctic_cloud_cmap = 'RdBu_r'
arctic_cloud_alt_cmap = 'RdBu_r'

cfs_alert = (-62.3167, 82.5) # Station Alert
stn_nord  = (-16.6667, 81.6) # Station Nord
thule_pituffik = (-68.703056, 76.531111) # Pituffik Space Base
