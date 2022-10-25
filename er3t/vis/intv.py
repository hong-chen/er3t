import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav


__all__ = ['compare_data_2d']



def compare_data_2d(
        data_x,
        data_y,
        wvl0=None,
        tmhr0=None,
        tmhr_range=None,
        wvl_range=[300.0, 2200.0],
        tmhr_step=10,
        wvl_step=2,
        description=None,
        fname_html=None
        ):


    from bokeh.layouts import layout, gridplot
    from bokeh.models import ColumnDataSource, ColorBar
    from bokeh.models.widgets import Select, Slider, CheckboxGroup
    from bokeh.models import Toggle, CustomJS, Legend, Span, HoverTool
    from bokeh.plotting import figure, output_file, save
    from bokeh.transform import linear_cmap
    from bokeh.palettes import RdYlBu6, Spectral6
    from bokeh.tile_providers import get_provider, Vendors


    # obtain basic information of the script, function, system etc.
    #/----------------------------------------------------------------------------\#
    _metadata = {
            'Computer': os.uname()[1],
            'Script': os.path.abspath(__file__),
            'Function':sys._getframe().f_code.co_name,
            'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    #\----------------------------------------------------------------------------/#


    # set title
    #/----------------------------------------------------------------------------\#
    if description is not None:
        title = 'Compare Data-2D (%s)' % description
    else:
        title = 'Compare Data-2D'
    #\----------------------------------------------------------------------------/#


    # set html file name
    #/----------------------------------------------------------------------------\#
    if fname_html is None:
        fname_html = '%s_%s-x_%s-y.html' % (_metadata['Function'], data_x['name'], data_y['name'])
    #\----------------------------------------------------------------------------/#


    output_file(fname_html, title=title, mode='inline')

    layout0 = layout(
              [[plt_spec, plt_geo],
               [slider_spec],
               [plt_time],
               [slider_time]], sizing_mode='fixed')

    save(layout0)



if __name__ == '__main__':

    pass
