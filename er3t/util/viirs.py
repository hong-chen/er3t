import os
import sys
import datetime
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import shutil
from http.cookiejar import CookieJar
import urllib.request
import requests
from er3t.util import check_equal



__all__ = []


# VIIRS reader
#/---------------------------------------------------------------------------\
#\---------------------------------------------------------------------------/





# VIIRS downloader
#/---------------------------------------------------------------------------\

def download_viirs_https(
             date,
             dataset_tag,
             filename_tag,
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             fdir_prefix='/archive/allData',
             day_interval=1,
             fdir_out='data',
             data_format=None,
             run=True,
             quiet=False,
             verbose=False):


    """
    Input:
        date: Python datetime object
        dataset_tag: string, collection + dataset name, e.g. '5200/VNP02IMG'
        filename_tag: string, string pattern in the filename, e.g. '.2035.'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        day_interval=: integer, for 8 day data, day_interval=8
        fdir_out=: string, output data directory
        data_format=None: e.g., 'hdf'
        run=: boolen type, if true, the command will only be displayed but not run
        quiet=: Boolen type, quiet tag
        verbose=: Boolen type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded VIIRS file paths
    """

    try:
        app_key = os.environ['MODIS_APP_KEY']
    except KeyError:
        app_key = 'aG9jaDQyNDA6YUc5dVp5NWphR1Z1TFRGQVkyOXNiM0poWkc4dVpXUjE6MTYzMzcyNTY5OTplNjJlODUyYzFiOGI3N2M0NzNhZDUxYjhiNzE1ZjUyNmI1ZDAyNTlk'
        if verbose:
            print('Warning [download_viirs_https]: Please get your app key by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#appkeys\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.bash_profile\'(Mac),\nexport MODIS_APP_KEY="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"\n')

    if shutil.which('curl'):
        command_line_tool = 'curl'
    elif shutil.which('wget'):
        command_line_tool = 'wget'
    else:
        sys.exit('Error [download_viirs_https]: \'download_viirs_https\' needs \'curl\' or \'wget\' to be installed.')

    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)

    fdir_server = server + fdir_data
    webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
    content  = webpage.read().decode('utf-8')
    lines    = content.split('\n')

    commands = []
    fnames_local = []
    for line in lines:
        filename = line.strip().split(',')[0]
        if filename_tag in filename:
            fname_server = '%s/%s' % (fdir_server, filename)
            fname_local  = '%s/%s' % (fdir_out, filename)
            fnames_local.append(fname_local)
            if command_line_tool == 'curl':
                command = 'mkdir -p %s && curl -H \'Authorization: Bearer %s\' -L -C - \'%s\' -o \'%s\'' % (fdir_out, app_key, fname_server, fname_local)
            elif command_line_tool == 'wget':
                command = 'mkdir -p %s && wget -c "%s" --header "Authorization: Bearer %s" -O %s' % (fdir_out, fname_server, app_key, fname_local)
            commands.append(command)

    if not run:

        if not quiet:
            print('Message [download_viirs_https]: The commands to run are:')
            for command in commands:
                print(command)
                print()

    else:

        for i, command in enumerate(commands):

            print('Message [download_viirs_https]: Downloading %s ...' % fnames_local[i])
            os.system(command)

            fname_local = fnames_local[i]

            if data_format is None:
                data_format = os.path.basename(fname_local).split('.')[-1]

            if data_format == 'hdf':

                try:
                    from pyhdf.SD import SD, SDC
                except ImportError:
                    msg = 'Warning [download_viirs_https]: To use \'download_viirs_https\', \'pyhdf\' needs to be installed.'
                    raise ImportError(msg)

                f = SD(fname_local, SDC.READ)
                f.end()
                print('Message [download_viirs_https]: \'%s\' has been downloaded.\n' % fname_local)

            else:

                print('Warning [download_viirs_https]: Do not support check for \'%s\'. Do not know whether \'%s\' has been successfully downloaded.\n' % (data_format, fname_local))

    return fnames_local

#\---------------------------------------------------------------------------/


if __name__=='__main__':

    pass
