import os
import sys
import requests
import datetime
import numpy as np
import warnings


from er3t.util.util import get_doy_tag, dtime_to_jday, jday_to_dtime, has_common_substring
import er3t.common
from er3t.common import fdir_data_tmp


__all__ = [
    'format_satname', \
    'get_token_earthdata', \
    'gen_file_earthdata', \
    'get_command_earthdata', \
    'get_fname_geometa', \
    'delete_file', \
    'get_local_file', \
    'get_online_file', \
    'get_nsidc_file_list', \
    'final_file_check', \
    'read_geometa', \
    'cal_proj_xy_geometa', \
    'cal_lon_lat_utc_geometa', \
    'cal_sec_offset_abi', \
    'get_satfile_tag', \
    'download_laads_https', \
    'download_lance_https', \
    'download_nsidc_https', \
    'download_oco2_https', \
    'download_worldview_image', \
    ]

def format_satname(satellite, instrument):
    """ Format satellite and instrument name """

    # check satellite and instrument
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if instrument.lower() == 'modis' and (satellite.lower() in ['aqua', 'terra']):
        instrument = instrument.upper()
        satellite  = satellite.lower().title()
    elif instrument.lower() == 'viirs' and (satellite.lower() in ['noaa20', 'snpp', 'noaa-20', 's-npp', 'noaa21', 'noaa-21']):
        instrument = instrument.upper()
        satellite  = satellite.replace('-', '').upper()
    elif instrument.lower() == 'abi' and (satellite.lower() in ['goes-east', 'goes-west']):
        instrument = instrument.upper()
        satellite  = satellite.upper().replace('WEST', 'West').replace('EAST', 'East')
    else:
        msg = '\nError [format_satname]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    satname = '%s|%s' % (satellite, instrument)

    return satname



def get_token_earthdata():
    """ Get Earthdata token to access data on NASA servers """

    try:
        token = os.environ['EARTHDATA_TOKEN']
    except KeyError:
        token = 'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImhvY2g0MjQwIiwiZXhwIjoxNzM5NTYyODkzLCJpYXQiOjE3MzQzNzg4OTMsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFzc3VyYW5jZV9sZXZlbCI6Mn0.4Kl8rPSIex6ib0aue0qQkhafOvnJZETO9fwxr5cXYNauQ1cxP40jbgypV5R2BRCFlxEsJHT1G-9S6ipnZ-O2FQyJGFF1Tu8oe4HSz-yFET2waS3OZ0pp9ca3jwPz-1byatscxoVneys7CgDPxuTq5XHIo6ooDxEK5k_LfDH6qti8NpMWITNHd11t96H0C6AuDBfVDU5CYENpvg1YnD7_nasi2H4o78cEiypmapG86vQYvN7dV-idrs3BJ4sk7lgnIXYlU3rPYGLTJCXgIEMjandZpzoEBos58Er59Bwft-BCepmNbYO4xRh-4yyzNGOd06SIhbWk7i_pIO_X7JRx8g'


        msg = '\nWarning [get_earthdata_token]: Please get a token by following the instructions at\nhttps://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens\nThen add the following to the source file of your shell, e.g. \'~/.bashrc\'(Unix) or \'~/.zshrc\'(Mac),\nexport EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\n'
        warnings.warn(msg)

    return token



def gen_file_earthdata(
        fname_login = '~/.netrc',
        fname_cookies = '~/.urs_cookies',
        ):

    """ Handle cookies and save meta information for Earthdata websites """

    secret = {}

    fname_login = os.path.abspath(os.path.expanduser(fname_login))
    if not os.path.exists(fname_login):

        try:
            username = os.environ['EARTHDATA_USERNAME']
            password = os.environ['EARTHDATA_PASSWORD']
        except Exception as error:
            print(error)
            msg = '\nError [gen_file_earthdata]: Please follow the instructions at \nhttps://disc.gsfc.nasa.gov/data-access\nto register a login account and create a <~/.netrc> file.'
            raise OSError(msg)

        content = 'machine urs.earthdata.nasa.gov login %s password %s' % (username, password)
        print('Message [gen_file_earthdata]: Creating <~/.netrc> ...')
        with open(fname_login, 'w') as f:
            f.write(content)

    secret['netrc'] = fname_login

    fname_cookies = os.path.abspath(os.path.expanduser(fname_cookies))
    if not os.path.exists(fname_cookies):
        print('Message [gen_file_earthdata]: Creating <~/.urs_cookies> ...')
        os.system('touch ~/.urs_cookies')

    secret['cookies'] = fname_cookies

    return secret



def get_command_earthdata(
        fname_target,
        filename=None,
        token_mode=True,
        primary_tool='curl',
        backup_tool='wget',
        fdir_save='%s/satfile' % fdir_data_tmp,
        verbose=False):

    """
    Get the LINUX/UNIX download command using curl or wget as the download tool.
    Also returns a backup command.
    """

    if filename is None:
        filename = os.path.basename(fname_target)

    fname_save = '%s/%s' % (fdir_save, filename)

    if token_mode: # recommended

        token = get_token_earthdata()
        header = '"Authorization: Bearer %s"' % token

        if verbose:
            options = {
                    'curl': '--header %s --connect-timeout 60.0 --retry 1 --max-time 60.0 --location --continue-at - --output "%s" "%s"' % (header, fname_save, fname_target),
                    'wget': '--header=%s --continue --timeout=60 --tries=2 --show-progress --output-document="%s" "%s"' % (header, fname_save, fname_target),
                    }
        else:
            options = {
                    'curl': '-sS --no-progress-bar --header %s --connect-timeout 60.0 --max-time 60.0 --retry 1 --location --continue-at - --output "%s" "%s"' % (header, fname_save, fname_target),
                    'wget': '--header=%s --continue --timeout=60 --tries=2  --quiet --output-document="%s" "%s"' % (header, fname_save, fname_target),
                    }


    else:

        secret = gen_file_earthdata()

        if verbose:
            options = {
                    'curl': '--netrc --cookie-jar %s --cookie %s --connect-timeout 60.0 --max-time 60.0 --retry 1 --location --continue-at - --output "%s" "%s"' % (secret['cookies'], secret['cookies'], fname_save, fname_target),
                    'wget': '--continue --load-cookies=%s --save-cookies=%s --auth-no-challenge --keep-session-cookies --content-disposition --timeout=60 --tries=2 --show-progress --output-document="%s" "%s"' % (secret['cookies'], secret['cookies'], fname_save, fname_target),
                    }
        else:
            options = {
                'curl': '-s --netrc --cookie-jar %s --cookie %s --connect-timeout 60.0 --max-time 60.0 --retry 1 --location --continue-at - --output "%s" "%s"' % (secret['cookies'], secret['cookies'], fname_save, fname_target),
                'wget': '--continue --load-cookies=%s --save-cookies=%s --auth-no-challenge --quiet --keep-session-cookies --content-disposition --timeout=60 --tries=2 --output-document="%s" "%s"' % (secret['cookies'], secret['cookies'], fname_save, fname_target),
                }

    if not os.path.exists(fdir_save):
        os.makedirs(fdir_save)

    primary_command = '%s %s' % (primary_tool, options[primary_tool])
    backup_command  = '%s %s' % (backup_tool,  options[backup_tool])

    return primary_command, backup_command



def get_fname_geometa(
        date,
        satname,
        server,
        ):


    date_s = date.strftime('%Y-%m-%d')

    # generate satellite filename on LAADS DAAC server
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if server == 'https://ladsweb.modaps.eosdis.nasa.gov':
        fnames_geometa = {
               'Aqua|MODIS': '%s/archive/geoMeta/61/AQUA/%4.4d/MYD03_%s.txt'                % (server, date.year, date_s),
               'Terra|MODIS': '%s/archive/geoMeta/61/TERRA/%4.4d/MOD03_%s.txt'              % (server, date.year, date_s),
               'SNPP|VIIRS': '%s/archive/geoMetaVIIRS/5200/NPP/%4.4d/VNP03MOD_%s.txt'       % (server, date.year, date_s),
               'NOAA20|VIIRS': '%s/archive/geoMetaVIIRS/5201/NOAA-20/%4.4d/VJ103MOD_%s.txt' % (server, date.year, date_s),
               'NOAA21|VIIRS': '%s/archive/geoMetaVIIRS/5200/NOAA-21/%4.4d/VJ203MOD_%s.txt' % (server, date.year, date_s),
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate satellite filename on LANCE DAAC server (near real time)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    elif server == 'https://nrt3.modaps.eosdis.nasa.gov':
        fnames_geometa = {
               'Aqua|MODIS': '%s/api/v2/content/archives/geoMetaMODIS/61/AQUA/%4.4d/MYD03_%s.txt'               % (server, date.year, date_s),
               'Terra|MODIS': '%s/api/v2/content/archives/geoMetaMODIS/61/TERRA/%4.4d/MOD03_%s.txt'             % (server, date.year, date_s),
               'SNPP|VIIRS': '%s/api/v2/content/archives/geoMetaVIIRS/5200/NPP/%4.4d/VNP03MOD_NRT_%s.txt'       % (server, date.year, date_s),
               'NOAA20|VIIRS': '%s/api/v2/content/archives/geoMetaVIIRS/5201/NOAA-20/%4.4d/VJ103MOD_NRT_%s.txt' % (server, date.year, date_s),
               'NOAA21|VIIRS': '%s/api/v2/content/archives/geoMetaVIIRS/5200/NOAA-21/%4.4d/VJ203MOD_NRT_%s.txt' % (server, date.year, date_s),
            }
    else:
        msg = '\nError [get_fname_geometa]: Currently do not support accessing geometa data from <%s>.' % server
        raise OSError(msg)

    fname_geometa = fnames_geometa[satname]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_geometa



def delete_file(
        fname_file,
        filename=None,
        fdir_local='./',
        fdir_save='%s/satfile' % fdir_data_tmp,
        ):

    """ Delete file (used in case file could not be downloaded correctly)"""

    if filename is None:
        filename = os.path.basename(fname_file)

    fname_local1 = os.path.abspath('%s/%s' % (fdir_save,  filename))
    fname_local2 = os.path.abspath('%s/%s' % (fdir_local, filename))

    if os.path.exists(fname_local1):
        os.remove(fname_local1)

    if os.path.exists(fname_local2):
        os.remove(fname_local2)



def get_local_file(
        fname_file,
        filename=None,
        fdir_local='./',
        fdir_save='%s/satfile' % fdir_data_tmp,
        ):

    if filename is None:
        filename = os.path.basename(fname_file)

    # try to get information from local
    # check two locations:
    #   1) current directory;
    #   2) <tmp-data/satfile> directory under er3t main directory
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if not os.path.exists(fdir_save):
        os.makedirs(fdir_save)

    fname_local1 = os.path.abspath('%s/%s' % (fdir_local, filename))
    fname_local2 = os.path.abspath('%s/%s' % (fdir_save,  filename))

    if os.path.exists(fname_local1):

        os.system('cp %s %s' % (fname_local1, fname_local2))
        with open(fname_local1, 'r') as f_:
            content = f_.read()

    elif os.path.exists(fname_local2):

        with open(fname_local2, 'r') as f_:
            content = f_.read()

    else:

        content = None
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return content



def get_online_file(
        fname_file,
        geometa=True,
        csv=False,
        filename=None,
        download=True,
        primary_tool='curl',
        backup_tool='wget',
        fdir_save='%s/satfile' % fdir_data_tmp,
        verbose=False):

    if filename is None:
        filename = os.path.basename(fname_file)

    if download:

        # fname_save = '%s/%s' % (fdir_save, filename)
        primary_command, backup_command = get_command_earthdata(fname_file,
                                                                filename=filename,
                                                                fdir_save=fdir_save,
                                                                primary_tool=primary_tool,
                                                                backup_tool=backup_tool,
                                                                verbose=verbose)
        # attempt to download using primary tool first.
        # if that does not work, try again with primary tool.
        # as a last resort, attempt with backup tool.
        try:
            # delete local version of the geometa first as this seems to cause issues downstream
            if (geometa is True) or (csv is True): # they can be None so make True explicit
                delete_file(fname_file, filename=filename, fdir_save=fdir_save)
            # if primary_tool == 'wget': # force wget to timeout
            #     primary_command = "timeout 60 " + primary_command

            os.system(primary_command)
            content = get_local_file(fname_file, filename=filename, fdir_save=fdir_save)
        except Exception as message:
            print(message, "\n")
            print("Message [get_online_file]: Failed to download/read {},\nAttempting again...".format(fname_file))

            try:
                os.system(primary_command)
                content = get_local_file(fname_file, filename=filename, fdir_save=fdir_save)
            except Exception as message:
                print(message, "\n")
                print("Message [get_online_file]: Failed to download/read {},\nAttempting with backup tool...".format(fname_file))
                delete_file(fname_file, filename=filename, fdir_save=fdir_save)

                try:
                    # if backup_tool == 'wget':
                    #     backup_command = "timeout 60 " + backup_command
                    # print("Executing following operation as a backup...\n{}".format(backup_command))

                    os.system(backup_command)
                    content = get_local_file(fname_file, filename=filename, fdir_save=fdir_save)
                except Exception as message:
                    print(message, "\n")
                    msg = "Message [get_online_file]: Failed to download/read {},\nTry again later.".format(fname_file)
                    delete_file(fname_file, filename=filename, fdir_save=fdir_save)
                    return None

    else:

        # this can be revisited, disabling it for now
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # try:
        #     with requests.Session() as session:
        #         session.auth = (username, password)
        #         r1     = session.request('get', fname_server)
        #         r      = session.get(r1.url, auth=(username, password))
        # except:
        #     msg = '\nError [get_online_file]: cannot access <%s>.' % fname_server
        #     raise OSError(msg)

        # if r.ok:
        #     content = r.content.decode('utf-8')
        # else:
        #     msg = '\nError [get_online_file]: failed to retrieve information from <%s>.' % fname_server
        #     warnings.warn(msg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # this can be revisited, disabling it for now
        # Use error handling to overcome occasional issues with LAADS DAAC servers
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # try:
        #     webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
        # except urllib.error.HTTPError:
        #     msg = "The LAADS DAAC servers appear to be down. Attempting again in 10 seconds..."
        #     print(msg)
        #     time.sleep(10)
        #     try:
        #         webpage  = urllib.request.urlopen('%s.csv' % fdir_server)
        #     except urllib.error.HTTPError:
        #         msg = '\nError [get_online_file]: cannot access <%s>.' % fdir_server
        #         raise OSError(msg)
        # content  = webpage.read().decode('utf-8')
        #╰────────────────────────────────────────────────────────────────────────────╯#

        content = None

    return content



def get_nsidc_file_list(
        product_id,
        version,
        instrument,
        extent,
        start_dt_hhmm,
        end_dt_hhmm,
        ):
    """
    Get a list of files available from an NSIDC URL directory.
    Modeled after Jupyter Notebook from the official NSIDC Github repo:
    https://github.com/nsidc/NSIDC-Data-Access-Notebook/blob/master/notebooks/Customize%20and%20Access%20NSIDC%20Data.ipynb

    Args:
    ----
        url : str
            The URL of the NSIDC directory to list files from.
        fdir_save : str, optional
            Directory to save temporary files.
        verbose : bool, optional
            Whether to print verbose messages.

    Returns:
    --------
        list
            A list of filenames available in the directory.
    """

    #\----------------------------------------------------------------------------------------------------------------------/#
    # first we need to check that collections for this product exist given the time and region. An example URL:
    # https://cmr.earthdata.nasa.gov/search/collections.json?instrument=MODIS&short_name=MOD29&temporal=2024-05-31T00:00:00Z,2024-05-31T23:00:00Z&bounding_box=-100,80,-50,85&has_granules=true

    # build search params
    temporal = start_dt_hhmm.strftime('%Y-%m-%dT%H:%M:%SZ') + ',' + end_dt_hhmm.strftime('%Y-%m-%dT%H:%M:%SZ')
    bbox     = f'{extent[0]},{extent[2]},{extent[1]},{extent[3]}' # bottom lonlat, top lonlat
    search_params = dict(instrument=instrument,
                         short_name=product_id,
                         version=version, # collection version like "61" for MODIS collection 6.1
                         temporal=temporal,
                         bounding_box=bbox)

    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json?has_granules=True'
    search_params_string = '&'.join('{}={}'.format(k, v) for (k, v) in search_params.items())
    cmr_collections_url = f'{cmr_collections_url}&{search_params_string}'
    session = requests.session()
    response = session.get(cmr_collections_url, timeout=10)
    if response.status_code != 200:
        print(f'Message [get_nsidc_file_list]: Could not submit query to {cmr_collections_url} as it resulted in a status code {response.status_code}')
        return []

    json_results = response.json()
    results = json_results['feed']['entry']
    if len(results) == 0:
        print(f'Message [get_nsidc_file_list]: Could not find any results on {cmr_collections_url}')
        return []
    # print('Message [get_nsidc_file_list]: Found dataset {}'.format(results[0]['dataset_id'])) # debug statement

    # loop through results just to make sure data is on the NSIDC servers
    nsidc_counts = 0
    for result in results:
        if 'NSIDC' in result['archive_center'].upper():
            nsidc_counts += 1

    if nsidc_counts == 0:
        print('Message [get_nsidc_file_list]: Could not find any data on NSIDC data centers')
        return []

    #\----------------------------------------------------------------------------------------------------------------------/#

    # now that we have found and verified that the collection and product exists, we need to query granules.
    # granules are indexed by page numbers and page size (defaults to 10) so we need to go through each page.
    # example url: # https://cmr.earthdata.nasa.gov/search/granules.json?downloadable=true&instrument=MODIS&short_name=MOD29&temporal=2024-05-31T00:00:00Z,2024-05-31T23:00:00Z&bounding_box=-100,80,-50,85

    cmr_granules_base_url = 'https://cmr.earthdata.nasa.gov/search/granules.json?downloadable=True'
    # update search params with page info
    search_params['page_num'] = 1
    search_params['page_size'] = 10 # do not make this too big as query will take too long

    # create list for links
    nsidc_download_links = []

    # set up input parameters
    headers = {'Accept': 'application/json'}
    application_formats = ('application/x-hdfeos', 'application/x-netcdf')
    data_formats        = ('hdf', 'nc', 'hdf5', 'hdf4')

    # send queries until we run out of pages
    while True:
        # update search params and url
        search_params_string = '&'.join('{}={}'.format(k, v) for (k, v) in search_params.items())
        cmr_granules_url = f'{cmr_granules_base_url}&{search_params_string}'

        response = requests.get(cmr_granules_url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f'Message [get_nsidc_file_list]: Could not submit query to {cmr_granules_url} as it resulted in a status code {response.status_code}')
            return []

        json_results = response.json()
        results = json_results['feed']['entry']
        if len(results) == 0: # out of results, so break out of loop
            break

        # loop through to get downloadable links
        for granule in results:
            links = granule['links']
            for link in links:
                if link['href'].lower().endswith(data_formats):
                    nsidc_download_links.append(link['href']) # add link to list

        # update page number for next request query
        search_params['page_num'] += 1
    #\----------------------------------------------------------------------------------------------------------------------/#

    print(f'Message [get_nsidc_file_list]: Found {len(nsidc_download_links)} granules')
    return nsidc_download_links



def final_file_check(fname_local, data_format, verbose):
    """
    Check if the file has been successfully downloaded.

    This function currently supports checking for HDF, NetCDF, and HDF5 files.
    If the file format is not supported, a warning is issued.

    Params:
        fname_local (str): The local filename to check.
        data_format (str): The format of the file. If None, the format is inferred from the file extension.
        verbose (bool): If True, prints a message when the file has been successfully downloaded.

    Returns:
        Returns 1 if the file has been successfully downloaded and 0 otherwise.
    """


    if data_format is None:
        data_format = os.path.basename(fname_local).split('.')[-1].lower()

    checked = False

    if data_format in ['hdf', 'hdf4', 'h4']:

        try:
            from pyhdf.SD import SD, SDC
            f = SD(fname_local, SDC.READ)
            f.end()
            checked = True

        except Exception as error:
            print(error)
            pass

    elif data_format in ['nc', 'nc4', 'netcdf', 'netcdf4']:
        try:
            from netCDF4 import Dataset
            f = Dataset(fname_local, 'r')
            f.close()
            checked = True

        except Exception as error:
            print(error)
            pass

    elif data_format in ['h5', 'hdf5']:

        try:
            import h5py
            f = h5py.File(fname_local, 'r')
            f.close()
            checked = True

        except Exception as error:
            print(error)
            pass

    else:

        msg = '\nWarning [final_file_check]: Do not support check for <.%s> file.\nDo not know whether <%s> has been successfully downloaded.\n' % (data_format, fname_local)
        warnings.warn(msg)

    if checked:
        if verbose:
            msg = '\nMessage [final_file_check]: <%s> has been successfully downloaded.\n' % fname_local
            print(msg)
        return 1

    else:
        msg = '\nWarning [final_file_check]: Do not know whether <%s> has been successfully downloaded.\n' % (fname_local)
        warnings.warn(msg)
        return 0




def read_geometa(content):

    """
    Parse geometa data in a list of Python dictionaries that contain information such as "GranuleID", "GRingLongitude1" etc.

    Input:
        content: a long string that contains the whole content of the geometa txt file

    Output:
        data: a list of Python dictionaries.

              An example,
              {'GranuleID': 'MYD03.A2019245.0000.061.2019245151541.hdf',
               'StartDateTime': '2019-09-02 00:00',
               'ArchiveSet': '61',
               'OrbitNumber': '92177',
               'DayNightFlag': 'N',
               'EastBoundingCoord': '40.0756634630551',
               'NorthBoundingCoord': '-57.0356008659127',
               'SouthBoundingCoord': '-81.1407080607332',
               'WestBoundingCoord': '-42.1164863420172',
               'GRingLongitude1': '-10.7904979067681',
               'GRingLongitude2': '32.4557535169638',
               'GRingLongitude3': '40.6409172253246',
               'GRingLongitude4': '-42.4223934777823',
               'GRingLatitude1': '-57.0310011746246',
               'GRingLatitude2': '-63.1593513428869',
               'GRingLatitude3': '-81.224336567583',
               'GRingLatitude4': '-68.7511407602523',
               'Satellite': 'Aqua',
               'Instrument': 'MODIS',
               'Orbit': 'Ascending'}
    """

    lines = content.split('\n')

    if len(lines) == 1:
        msg = 'Error [read_geometa]: Could not download the geoMeta text file. This could be an issue with either the download tool or the Earthdata token'
        if (lines[0][0] == '{') and (lines[0][-1] == '}'):
            msg = msg + ' or the date for which you are looking to download does not have data.\n'

        print(msg)
        return None

    if lines[0] == '<!DOCTYPE html>' or lines[1] == '<!DOCTYPE html>':
        msg = 'Error [read_geometa]: Could not download the geoMeta text file. This could be an issue with either the download tool or the Earthdata token.\n'
        print(msg)
        return None

    index_header = 0
    while (len(lines[index_header]) > 0) and lines[index_header][0] == '#':
        index_header += 1

    index_header -= 1

    if index_header == -1:
        msg = 'Error [read_geometa]: Cannot locate header in the provided content.\n'
        print(msg)
        return None

    header_line = lines[index_header]
    vnames = [word.strip() for word in header_line[1:].split(',')]

    Nvar = len(vnames)

    data = []
    for line_data in lines[index_header+1:]:
        if len(line_data) > 0:
            data0_ = [word.strip() for word in line_data.split(',')]
            data0 = {vnames[i]:data0_[i] for i in range(Nvar)}

            if 'MYD03' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'Aqua'
                data0['Instrument'] = 'MODIS'
                data0['Orbit']      = 'Ascending'
            elif 'MOD03' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'Terra'
                data0['Instrument'] = 'MODIS'
                data0['Orbit']      = 'Descending'
            elif 'VJ103' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'NOAA-20'
                data0['Instrument'] = 'VIIRS'
                data0['Orbit']      = 'Ascending'
            elif 'VJ203' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'NOAA-21'
                data0['Instrument'] = 'VIIRS'
                data0['Orbit']      = 'Ascending'
            elif 'VNP03' in data0_[0].split('.')[0]:
                data0['Satellite']  = 'S-NPP'
                data0['Instrument'] = 'VIIRS'
                data0['Orbit']      = 'Ascending'
            else:
                data0['Satellite']  = 'Unknown'
                data0['Instrument'] = 'Unknown'
                data0['Orbit']      = 'Unknown'

            data.append(data0)

    return data



def cal_proj_xy_geometa(line_data, closed=True):

    """
    Calculate globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points

    Input:
        line_data: Python dictionary (details see <read_geo_meta>) that contains basic information of a satellite granule
        closed=True: if True, return five corner points with the last point repeating the first point;
                     if False, return four corner points

    Output:
        proj_xy: globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points
        xy: dimension of (5, 2) if <closed=True> and (4, 2) if <closed=False>
    """

    import cartopy.crs as ccrs

    # get corner points
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon_  = np.array([
        float(line_data['GRingLongitude1']),
        float(line_data['GRingLongitude2']),
        float(line_data['GRingLongitude3']),
        float(line_data['GRingLongitude4']),
        float(line_data['GRingLongitude1'])
        ])

    lat_  = np.array([
        float(line_data['GRingLatitude1']),
        float(line_data['GRingLatitude2']),
        float(line_data['GRingLatitude3']),
        float(line_data['GRingLatitude4']),
        float(line_data['GRingLatitude1'])
        ])

    if (abs(lon_[0]-lon_[1])>180.0) | (abs(lon_[0]-lon_[2])>180.0) | \
       (abs(lon_[0]-lon_[3])>180.0) | (abs(lon_[1]-lon_[2])>180.0) | \
       (abs(lon_[1]-lon_[3])>180.0) | (abs(lon_[2]-lon_[3])>180.0):

        lon_[lon_<0.0] += 360.0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # roughly determine the center of granule
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon = lon_[:-1]
    lat = lat_[:-1]
    center_lon_ = lon.mean()
    center_lat_ = lat.mean()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # find the true center
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_lonlat = ccrs.PlateCarree()

    proj_xy_ = ccrs.Orthographic(central_longitude=center_lon_, central_latitude=center_lat_)
    xy_ = proj_xy_.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]

    center_x  = xy_[:, 0].mean()
    center_y  = xy_[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_xy_)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # convert lon/lat corner points into xy
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy_  = proj_xy.transform_points(proj_lonlat, lon_, lat_)[:, [0, 1]]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    if closed:
        return proj_xy, xy_
    else:
        return proj_xy, xy_[:-1, :]



def cal_lon_lat_utc_geometa(
        line_data,
        delta_t=300.0,
        N_along=2030,
        N_cross=1354,
        N_scan=203,
        scan='cw',
        testing=False,
        ):

    """
    Calculate (more of an estimation) longitude, latitude, and utc time (julian day) from corner points provided by geometa

    Input:
        line_data: Python dictionary (details see <read_geo_meta>) that contains basic information of a satellite granule
        delta_t=300.0: time span of the given granule in seconds, e.g., 5 minutes for MODIS granule
        N_along=2030: number of pixels along the satellite track
        N_cross=1354: number of pixels across the satellite track
        N_scan=203: number of rotatory scans, MODIS and VIIRS both have N_scan=203
        scan='cw': direction of the rotatory scan (viewing along the satellite travel direction), MODIS and VIIRS both rotate clockwise
        testing=False: testing mode, if True, a figure will be generated

    Output:
        lon_out: longitude, dimension of (N_along, N_cross)
        lat_out: latitude, dimension of (N_along, N_cross)
        jday_out: julian day, dimension of (N_along, N_cross)

    Notes:
    Aqua    (delta_t=300.0, N_scan=203, N_along=2030, N_cross=1354, scan='cw')
    Terra   (delta_t=300.0, N_scan=203, N_along=2030, N_cross=1354, scan='cw')
    NOAA-20 (delta_t=360.0, N_scan=203, N_along=3248, N_cross=3200, scan='cw')
    S-NPP   (delta_t=360.0, N_scan=203, N_along=3248, N_cross=3200, scan='cw')
    """

    import cartopy
    import cartopy.crs as ccrs


    # check if delta_t is correct
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if line_data['Instrument'].lower() == 'modis' and delta_t != 300.0:
        msg = '\nWarning [cal_lon_lat_utc_geometa]: MODIS should have <delta_t=300.0> but given <delta_t=%.1f>, please double-check.' % delta_t
        warnings.warn(msg)
    elif line_data['Instrument'].lower() == 'viirs' and delta_t != 360.0:
        msg = '\nWarning [cal_lon_lat_utc_geometa]: VIIRS should have <delta_t=360.0> but given <delta_t=%.1f>, please double-check.' % delta_t
        warnings.warn(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get lon/lat corner points into xy
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_xy, xy_ = cal_proj_xy_geometa(line_data, closed=True)
    xy  = xy_[:-1, :]
    x = xy[:, 0]
    y = xy[:, 1]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # reconstruct x y grids
    # c: cross track
    # a: along track
    #╭────────────────────────────────────────────────────────────────────────────╮#
    x_01_c = (x[0]+x[1])/2.0
    y_01_c = (y[0]+y[1])/2.0

    x_23_c = (x[2]+x[3])/2.0
    y_23_c = (y[2]+y[3])/2.0

    dist_a = np.sqrt((x_23_c-x_01_c)**2+(y_23_c-y_01_c)**2)
    slope_a = (y_23_c-y_01_c) / (x_23_c-x_01_c)

    x_03_c = (x[0]+x[3])/2.0
    y_03_c = (y[0]+y[3])/2.0

    x_12_c = (x[1]+x[2])/2.0
    y_12_c = (y[1]+y[2])/2.0

    dist_c = np.sqrt((x_03_c-x_12_c)**2+(y_03_c-y_12_c)**2)
    slope_c = (y_12_c-y_03_c) / (x_12_c-x_03_c)

    N_a = N_along
    N_c = N_cross

    i_a = np.arange(N_a, dtype=np.float64)
    i_c = np.arange(N_c, dtype=np.float64)

    ii_a, ii_c = np.meshgrid(i_a, i_c, indexing='ij')

    res_a = dist_a/N_a
    res_c = dist_c/N_c

    ang_a = np.arctan(slope_a)
    ang_c = np.arctan(slope_c)

    # TODO: this is experimental, might cause some problem in the future
    #╭──────────────────────────────────────────────────────────────╮#
    if ((x[0]>x[1]) or (x[3]>x[2])) or ((y[2]>y[1]) or (y[3]>y[0])):
        if ((x[0]>x[3]) or (x[1]>x[2])):
            index0 = 0
        else:
            index0 = 3
    else:
        if ((x[0]>x[3]) or (x[1]>x[2])):
            index0 = 1
        else:
            index0 = 2
    #╰──────────────────────────────────────────────────────────────╯#

    xx = x[index0]-res_c*ii_c*np.cos(ang_c)-res_a*ii_a*np.cos(ang_a)
    yy = y[index0]-res_c*ii_c*np.sin(ang_c)-res_a*ii_a*np.sin(ang_a)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # calculate lon lat
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_lonlat = ccrs.PlateCarree()
    lonlat_out = proj_lonlat.transform_points(proj_xy, xx, yy)[..., [0, 1]]
    lon_out = lonlat_out[..., 0]
    lat_out = lonlat_out[..., 1]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # calculate utc (jday)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    filename = line_data['GranuleID']
    dtime0_s = '.'.join(filename.split('.')[1:3])
    dtime0 = datetime.datetime.strptime(dtime0_s, 'A%Y%j.%H%M')
    jday0 = dtime_to_jday(dtime0)

    jday_out = np.zeros(lon_out.shape, dtype=np.float64)

    delta_t0 = delta_t / N_scan

    delta_t0_c = delta_t0/3.0/N_c*i_c  # 120 degree coverage thus </3.0>
    if scan == 'ccw':
        delta_t0_c = delta_t0_c[::-1]

    # TODO: this is experimental, might cause some problem in the future
    #╭──────────────────────────────────────────────────────────────╮#
    if index0 in [1, 3]:
        lon_out = lon_out[:, ::-1]
        lat_out = lat_out[:, ::-1]
        delta_t0_c = delta_t0_c[::-1]
    #╰──────────────────────────────────────────────────────────────╯#

    N_a0 = int(N_a//N_scan)

    for i in range(N_scan):
        index_s = N_a0*i
        index_e = N_a0*(i+1)
        jday_out0_ = jday0+(delta_t0*i+delta_t0_c)/86400.0

        if (i == N_scan-1):
            index_e = N_a
            N_a0 = N_a - index_s

        jday_out0 = np.tile(jday_out0_, N_a0).reshape((N_a0, N_c))
        jday_out[index_s:index_e, :] = jday_out0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if testing:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.path as mpl_path
        import matplotlib.patches as mpatches
        from matplotlib.ticker import FixedLocator

        utc_sec_out = (jday_out-jday_out.min())*86400.0

        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111, projection=proj_xy)
        cs = ax1.scatter(lon_out[::5, ::5], lat_out[::5, ::5], c=utc_sec_out[::5, ::5], transform=ccrs.PlateCarree(), vmin=0.0, vmax=delta_t, cmap='jet', s=1, lw=0.0)
        ax1.text(x[0], y[0], '0-LR', color='black')
        ax1.text(x[1], y[1], '1-LL', color='black')
        ax1.text(x[2], y[2], '2-UL', color='black')
        ax1.text(x[3], y[3], '3-UR', color='black')

        granule  = mpl_path.Path(xy_, closed=True)
        patch = mpatches.PathPatch(granule, facecolor='none', edgecolor='black', lw=1.0)
        ax1.add_patch(patch)
        if (utc_sec_out[:, -1]-utc_sec_out[:, 0]).sum()>0.0:
            ax1.scatter(xx[:, -1], yy[:, -1], c='red' , lw=0.0, s=3.0)
            ax1.scatter(xx[:, 0], yy[:, 0], c='blue', lw=0.0, s=3.0)
        else:
            ax1.scatter(xx[:, -1], yy[:, -1], c='blue' , lw=0.0, s=3.0)
            ax1.scatter(xx[:, 0], yy[:, 0], c='red', lw=0.0, s=3.0)

        ax1.set_title(filename)
        ax1.set_global()
        ax1.add_feature(cartopy.feature.OCEAN, zorder=0)
        ax1.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='none')
        ax1.coastlines(color='gray', lw=0.5)
        g3 = ax1.gridlines()
        g3.xlocator = FixedLocator(np.arange(-180, 181, 60))
        g3.ylocator = FixedLocator(np.arange(-80, 81, 20))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fname_png = filename.replace('.hdf', '.png').replace('.nc', '.png')
        fig.savefig('globe-view_%s' % fname_png, bbox_inches='tight', metadata=_metadata)
        print("\nMessage [cal_lon_lat_utc_geometa]: Figure saved as 'globe-view_%s'" % fname_png)
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return lon_out, lat_out, jday_out



def cal_sec_offset_abi(extent, satname='GOES-East|ABI', sec_per_scan=30.0):

    """
    Details see https://www.goes-r.gov/users/abiScanModeInfo.html
    """

    import cartopy.crs as ccrs

    satellite, instrument = satname.split('|')

    if satname == 'GOES-East|ABI':

        full_disk_time_span = {
                0: [[19.8, 26.6]],
                1: [[3.7, 12.4], [12.9, 23.0]],
                2: [[4.5, 15.7]],
                3: [[3.0, 15.1]],
                4: [[2.7, 15.4]],
                5: [[2.5, 15.6]],
                6: [[2.3, 15.8]],
                7: [[2.1, 16.0]],
                8: [[2.0, 16.0]],
                9: [[2.0, 16.1]],
               10: [[2.0, 16.1]],
               11: [[2.0, 16.0]],
               12: [[2.1, 16.0]],
               13: [[2.3, 15.8]],
               14: [[2.4, 15.6]],
               15: [[2.7, 15.4]],
               16: [[3.0, 15.0]],
               17: [[3.5, 14.6]],
               18: [[4.0, 14.1]],
               19: [[4.7, 13.4], [13.9, 20.7]],
                }

        center_lon = -75.0
        center_lat = 0.0

    elif satname == 'GOES-West|ABI':

        full_disk_time_span = {
                0: None,
                1: [[1.4,  8.2], [ 8.7, 17.4], [18.0, 28.1]],
                2: [[5.1, 16.3]],
                3: [[2.3, 14.4]],
                4: [[2.0, 14.7]],
                5: [[1.8, 14.9]],
                6: [[1.6, 15.1]],
                7: [[1.4, 15.3]],
                8: [[1.3, 15.3]],
                9: [[1.3, 15.4]],
               10: [[1.3, 15.4]],
               11: [[1.3, 15.3]],
               12: [[1.4, 15.3]],
               13: [[1.6, 15.1]],
               14: [[1.8, 14.9]],
               15: [[2.0, 14.7]],
               16: [[2.3, 14.4]],
               17: [[2.8, 13.9]],
               18: [[3.3, 13.4], [14.0, 22.7]],
               19: [[1.4,  8.2]],
                }

        center_lon = -137.0
        center_lat = 0.0

    else:

        msg = '\nError [cal_utc_abi]: Currently do not support <%s> onboard <%s>.' % (instrument, satellite)
        raise NameError(msg)


    # define projections
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj_lonlat = ccrs.PlateCarree()
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # get scan stripe edges in y
    #╭────────────────────────────────────────────────────────────────────────────╮#
    Nscan = len(full_disk_time_span.keys())

    lat_scan_edges = np.linspace(90.0, -90.0, Nscan+1)
    lon_scan_edges = np.repeat(center_lon, Nscan+1)
    xy = proj_xy.transform_points(proj_lonlat, lon_scan_edges, lat_scan_edges)[:, [0, 1]]

    y_scan_edges = xy[:, 1]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # calculate corner points from <extent>
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon_in_ = np.arange(extent[0], extent[1], 0.001)
    lat_in_ = np.arange(extent[2], extent[3], 0.001)
    lon_in, lat_in = np.meshgrid(lon_in_, lat_in_, indexing='ij')

    xy_in = proj_xy.transform_points(proj_lonlat, lon_in.ravel(), lat_in.ravel())[:, [0, 1]]
    x_in = xy_in[:, 0]
    y_in = xy_in[:, 1]

    y_min = np.nanmin(y_in)
    y_max = np.nanmax(y_in)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    Ns = max(np.argmin(np.abs(y_scan_edges-y_max))-1, 0)
    Ne = min(np.argmin(np.abs(y_scan_edges-y_min))+1, Nscan)

    sec_offset = np.zeros_like(x_in)
    sec_offset[...] = np.nan

    for i in range(Ns, Ne):
        span_time0 = full_disk_time_span[i]
        if span_time0 is not None:
            span_time0 = span_time0[-1]
            time_s = span_time0[0] + i * sec_per_scan
            time_e = span_time0[1] + i * sec_per_scan
            y_edge_min = y_scan_edges[i+1]
            y_edge_max = y_scan_edges[i]
            logic_in = (y_in>=y_edge_min) & (y_in<y_edge_max)

            if logic_in.sum() > 0:

                # calculate time offset
                #╭────────────────────────────────────────────────────────────────────────────╮#
                R_earth = np.nanmax(y_scan_edges)
                delta_scan_x_half = R_earth * np.sin(np.arccos(((y_edge_min+y_edge_max)/2.0)/R_earth))
                delta_x = delta_scan_x_half*2.0
                x_s = -delta_scan_x_half

                delta_t = time_e-time_s
                slope = delta_t/delta_x

                x_min = np.nanmin(x_in[logic_in])

                time0 = time_s + ((x_min-x_s)/delta_x)*delta_t
                sec_offset[logic_in] = time0 + slope*(x_in[logic_in]-x_min)
                #╰────────────────────────────────────────────────────────────────────────────╯#


    return sec_offset



def get_satfile_tag(
             date,
             lon,
             lat,
             satellite,
             instrument,
             nrt=False,
             fdir_local='./',
             fdir_save='%s/satfile' % fdir_data_tmp,
             geometa=False,
             percent0=0.0,
             worldview=False,
             start_dt_hhmm=None,
             end_dt_hhmm=None
             ):

    """
    Get filename tag/overpass information for standard products.
    Currently supported satellites/instruments are:
    Aqua/MODIS, Terra/MODIS, SNPP/VIIRS, NOAA-20/VIIRS, NOAA-21/VIIRS.

    Input:
        date: Python datetime.datetime object
        lon : longitude of, e.g. flight track
        lat : latitude of, e.g. flight track
        satellite=: one of "aqua", "terra", 'snpp', 'noaa20', 'noaa21'
        instrument=: "modis" or "viirs" as appropriate
        nrt=: bool, near real time. if True, will access https://nrt3.modaps.eosdis.nasa.gov,
                                    if False (default) will access https://ladsweb.modaps.eosdis.nasa.gov,
        fdir_prefix=: string, data directory on NASA server
        verbose=: Boolen type, verbose tag
    output:
        filename_tags: Python list of file name tags
    """

    # check cartopy and matplotlib
    #╭────────────────────────────────────────────────────────────────────────────╮#
    import cartopy.crs as ccrs
    import matplotlib.path as mpl_path
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # get formatted satellite tag
    #╭────────────────────────────────────────────────────────────────────────────╮#
    satname = format_satname(satellite, instrument)
    satellite, instrument = satname.split('|')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # get satellite geometa filename on the appropriate DAAC server
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if nrt:
        server = 'https://nrt3.modaps.eosdis.nasa.gov'
    else:
        server = 'https://ladsweb.modaps.eosdis.nasa.gov'

    fname_geometa = get_fname_geometa(date, satname, server=server)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    # or check overlap within region of interest
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # get geometa info
    filename_geometa = '%s_%s' % (server.replace('https://', '').split('.')[0], os.path.basename(fname_geometa))

    # try to get geometa information from local
    content = get_local_file(fname_geometa, filename=filename_geometa, fdir_local=fdir_local, fdir_save=fdir_save)

    # try to get geometa information online
    if (content is None) or ('<!DOCTYPE html>' in content):
        content = get_online_file(fname_geometa, geometa=True, csv=None, filename=filename_geometa, fdir_save=fdir_save)

    # for now, always use online file since local seems to cause downstream issues
    # content = get_online_file(fname_geometa, geometa=True, csv=None, filename=filename_geometa, fdir_save=fdir_save)

    # read in geometa info
    data = read_geometa(content)

    if data is None:
        return []

    # loop through all the satellite "granules" constructed through four corner points
    # and find which granules contain the input data
    #╭────────────────────────────────────────────────────────────────────────────╮#

    # by default if no start and end times are given, use 0000 and 2359
    if start_dt_hhmm is None:
        start_dt_hhmm = datetime.datetime(date.year, date.month, date.day, 0, 0)

    if end_dt_hhmm is None:
        end_dt_hhmm = datetime.datetime(date.year, date.month, date.day, 23, 59)

    proj_lonlat = ccrs.PlateCarree()

    Ndata = len(data)
    filename_tags = []

    percent_all   = np.array([], dtype=er3t.common.f_dtype)
    i_all         = []
    for i in range(Ndata):

        line = data[i]

        # get bounds of the satellite overpass/granule
        proj_xy, xy_granule = cal_proj_xy_geometa(line, closed=True)
        sat_granule  = mpl_path.Path(xy_granule, closed=True)

        # check if the overpass/granule overlaps with region of interest
        xy_in      = proj_xy.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]
        points_in  = sat_granule.contains_points(xy_in)

        Npoint_in  = points_in.sum()
        Npoint_tot = points_in.size

        percent_in = float(Npoint_in) * 100.0 / float(Npoint_tot)

        if (Npoint_in>0) and (line['DayNightFlag']=='D') and (percent_in>=percent0):

            if geometa:
                filename_tags.append(line)
            else:
                granule_dt = line['StartDateTime']
                granule_dt = datetime.datetime.strptime(granule_dt, '%Y-%m-%d %H:%M') # format it for processing

                if start_dt_hhmm <= granule_dt <= end_dt_hhmm: # get the filename only if within time bounds
                    filename = line['GranuleID']
                    filename_tag = '.'.join(filename.split('.')[1:3])
                    filename_tags.append(filename_tag)

            percent_all = np.append(percent_all, percent_in)
            i_all.append(i)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sort by percentage-in and time if <percent0> is specified or <wordview=True>
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if (percent0 > 0.0 ) or worldview:
        indices_sort_p = np.argsort(percent_all)
        if satellite != 'Terra':
            indices_sort_i = i_all[::-1]
        else:
            indices_sort_i = i_all

        if all(percent_i>97.0 for percent_i in percent_all):
            indices_sort = np.lexsort((indices_sort_p, indices_sort_i))[::-1]
        else:
            indices_sort = np.lexsort((indices_sort_i, indices_sort_p))[::-1]

        filename_tags = [filename_tags[i] for i in indices_sort]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return filename_tags



def download_laads_https(
             date,
             dataset_tag,
             filename_tag,
             server='https://ladsweb.modaps.eosdis.nasa.gov',
             fdir_prefix='/archive/allData',
             day_interval=1,
             fdir_out='tmp-data',
             fdir_save='%s/satfile' % fdir_data_tmp,
             data_format=None,
             run=True,
             verbose=True):

    """
    Downloads products from the LAADS Data Archive (DAAC).

    Input:
        date: Python datetime object
        dataset_tag: string, collection + dataset name, e.g. '61/MYD06_L2'
        filename_tag: string, string pattern in the filename, e.g. '.2035.'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        day_interval=: integer, for 8 day data, day_interval=8
        fdir_out=: string, output data directory
        data_format=None: e.g., 'hdf'
        run=: boolean type, if False, the command will only be displayed but not run
        verbose=: boolean type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded satellite data file paths
    """

    # retrieve the directory where satellite data is stored for picked date
    #╭────────────────────────────────────────────────────────────────────────────╮#
    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)
    fdir_server = server + fdir_data
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get csv info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_csv = '%s.csv' % fdir_server
    filename_csv = server.replace('https://', '').split('.')[0] + '_'.join(('%s.csv' % fdir_data).split('/'))

    # try to get geometa information from local
    content = get_local_file(fname_csv, filename=filename_csv, fdir_save=fdir_save)

    # try to get geometa information online
    if content is None:
        content = get_online_file(fname_csv, geometa=False, csv=True, filename=filename_csv, fdir_save=fdir_save)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get download commands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    exist_count = 0 # to prevent re-downloading. TODO: Add `overwrite` option instead for user
    lines = content.split('\n')
    primary_commands = []
    backup_commands  = []
    fnames_local = []
    for line in lines:
        filename = line.strip().split(',')[0]

        if filename_tag in filename:
            fname_server = '%s/%s' % (fdir_server, filename)
            fname_local  = '%s/%s' % (fdir_out, filename)
            if os.path.isfile(fname_local) and final_file_check(fname_local, data_format=data_format, verbose=verbose):
                print("Message [download_laads_https]: File {} already exists and looks good. Will not re-download this file.".format(fname_local))
                exist_count += 1
            else:
                fnames_local.append(fname_local)
                primary_command, backup_command = get_command_earthdata(fname_server, filename=filename, fdir_save=fdir_out, verbose=verbose)
                primary_commands.append(primary_command)
                backup_commands.append(backup_command)

    print("Message [download_laads_https]: Total of {} will be downloaded. {} will be skipped as they already exist and work as advertised.".format(len(fnames_local), exist_count))
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # run/print command
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if run:

        for i in range(len(primary_commands)):

            fname_local = fnames_local[i]

            if verbose:
                print('Message [download_laads_https]: Downloading %s ...' % fname_local)
            os.system(primary_commands[i])

            # if primary command fails, execute backup command.
            # if that fails again, then delete the file and remove from list
            if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                os.system(backup_commands[i])

                if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                    print("Message [download_laads_https]: Could not complete the download of or something is wrong with {}...deleting...".format(fname_local))
                    os.remove(fname_local)
                    fnames_local.remove(fname_local) #remove from list
    else:

        print('Message [download_laads_https]: The commands to run are:')
        for command in primary_commands:
            print(command)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return fnames_local



def download_lance_https(
             date,
             dataset_tag,
             filename_tag,
             server='https://nrt3.modaps.eosdis.nasa.gov',
             fdir_prefix='/archives/allData',
             day_interval=1,
             fdir_out='tmp-data',
             fdir_save='%s/satfile' % fdir_data_tmp,
             data_format=None,
             run=True,
             verbose=True):

    """
    Downloads products from the LANCE Data Archive (DAAC).

    Input:
        date: Python datetime object
        dataset_tag: string, collection + dataset name, e.g. '61/MYD06_L2'
        filename_tag: string, string pattern in the filename, e.g. '.2035.'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        day_interval=: integer, for 8 day data, day_interval=8
        fdir_out=: string, output data directory
        data_format=None: e.g., 'hdf'
        run=: boolean type, if False, the command will only be displayed but not run
        verbose=: boolean type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded satellite data file paths
    """

    # VIIRS NRT is labeled differently from the standard product.
    # Therefore, the dataset_tag needs to be updated only for VIIRS NRT products.
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if dataset_tag.split('/')[-1].upper().startswith(('VNP', 'VJ1', 'VJ2')):
        dataset_tag = dataset_tag + '_NRT'
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # retrieve the directory where satellite data is stored for picked date
    #╭────────────────────────────────────────────────────────────────────────────╮#
    year_str = str(date.timetuple().tm_year).zfill(4)
    if day_interval == 1:
        doy_str  = str(date.timetuple().tm_yday).zfill(3)
    else:
        doy_str = get_doy_tag(date, day_interval=day_interval)

    fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get csv info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_csv = '%s/api/v2/content/details/allData/%s/%s/%s?fields=all&formats=csv' % (server, dataset_tag, year_str, doy_str)
    filename_csv = server.replace('https://', '').split('.')[0] + '_'.join(('%s.csv' % fdir_data).split('/'))

    # try to get geometa information from local
    content = get_local_file(fname_csv, filename=filename_csv, fdir_save=fdir_save)

    # try to get geometa information online
    if content is None:
        content = get_online_file(fname_csv, geometa=False, csv=True, filename=filename_csv, fdir_save=fdir_save)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get download commands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    exist_count = 0
    lines = content.split('\n')
    primary_commands = []
    backup_commands  = []
    fnames_local = []
    for line in lines:
        filename = line.strip().split(',')[0]

        if (filename_tag in filename) and ('.met' not in filename):
            fname_server = '%s/api/v2/content%s/%s' % (server, fdir_data, filename)
            fname_local  = '%s/%s' % (fdir_out, filename)


            if os.path.isfile(fname_local) and final_file_check(fname_local, data_format=data_format, verbose=verbose):
                print("Message [download_lance_https]: File {} already exists and looks good. Will not re-download this file.".format(fname_local))
                exist_count += 1
            else:
                fnames_local.append(fname_local)
                primary_command, backup_command = get_command_earthdata(fname_server, filename=filename, fdir_save=fdir_out, primary_tool='curl', backup_tool='wget', verbose=verbose)
                primary_commands.append(primary_command)
                backup_commands.append('timeout 60 ' + backup_command) # force timeout for wget

    print("Message [download_lance_https]: Total of {} will be downloaded. {} will be skipped as they already exist and work as advertised.".format(len(fnames_local), exist_count))
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # run/print command
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if run:

        for i in range(len(primary_commands)):

            fname_local = fnames_local[i]

            if verbose:
                print('Message [download_lance_https]: Downloading %s ...' % fname_local)
            os.system(primary_commands[i])

            # if primary command fails, execute backup command.
            # if that fails again, then delete the file and remove from list
            if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                os.system(backup_commands[i])

                if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                    print("Message [download_lance_https]: Could not complete the download of or something is wrong with {}...deleting...".format(fname_local))
                    os.remove(fname_local)
                    fnames_local.remove(fname_local) #remove from list
    else:

        print('Message [download_lance_https]: The commands to run are:')
        for command in primary_commands:
            print(command)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fnames_local



def download_oco2_https(
             dtime,
             dataset_tag,
             fnames=None,
             server='https://oco2.gesdisc.eosdis.nasa.gov',
             fdir_prefix='/data/OCO2_DATA',
             fdir_out='tmp-data',
             data_format=None,
             run=True,
             verbose=True):

    """
    Input:
        dtime: Python datetime object
        dataset_tag: string, e.g. 'OCO2_L2_Standard.8r'
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        fdir_out=: string, output data directory
        data_format=None: e.g., 'h5'
        run=: boolen type, if true, the command will only be displayed but not run
        verbose=: Boolen type, verbose tag

    Output:
        fnames_local: Python list that contains downloaded OCO2 file paths
    """

    from er3t.util.oco2 import get_fnames_from_web, get_dtime_from_xml

    year_str = str(dtime.timetuple().tm_year).zfill(4)
    doy_str  = str(dtime.timetuple().tm_yday).zfill(3)

    if dataset_tag in [
            'OCO2_L2_Met.10',
            'OCO2_L2_Met.10r',
            'OCO2_L2_Met.11r',
            'OCO2_L2_Standard.10',
            'OCO2_L2_Standard.10r',
            'OCO2_L2_Standard.11r',
            'OCO2_L1B_Science.10',
            'OCO2_L1B_Science.10r',
            'OCO2_L1B_Science.11r',
            'OCO2_L1B_Calibration.10',
            'OCO2_L1B_Calibration.10r',
            'OCO2_L1B_Calibration.11r',
            'OCO2_L2_CO2Prior.10',
            'OCO2_L2_CO2Prior.10r',
            'OCO2_L2_CO2Prior.11r',
            'OCO2_L2_IMAPDOAS.10',
            'OCO2_L2_IMAPDOAS.10r',
            'OCO2_L2_IMAPDOAS.11r',
            'OCO2_L2_Diagnostic.10',
            'OCO2_L2_Diagnostic.10r',
            'OCO2_L2_Diagnostic.11r'
            ]:

        fdir_data = '%s/%s/%s/%s' % (fdir_prefix, dataset_tag, year_str, doy_str)

    elif dataset_tag in [
            'OCO2_L2_Lite_FP.9r',
            'OCO2_L2_Lite_FP.10r',
            'OCO2_L2_Lite_FP.11.1r',
            'OCO2_L2_Lite_SIF.10r'
            ]:

        fdir_data = '%s/%s/%s' % (fdir_prefix, dataset_tag, year_str)

    else:

        msg = '\nError [download_oco2_https]: Currently do not support downloading <%s>.' % dataset_tag
        raise OSError(msg)

    fdir_server = server + fdir_data

    fnames_xml = get_fnames_from_web(fdir_server, 'xml')
    if len(fnames_xml) > 0:
        data_format = fnames_xml[0].split('.')[-2]
    else:
        msg = '\nError [download_oco2_https]: XML files are not available at <%s>.' % fdir_server
        raise OSError(msg)

    fnames_server = []

    if fnames is not None:

        for fname in fnames:
            fname_server = '%s/%s' % (fdir_server, fname)
            fnames_server.append(fname_server)

    else:

        fnames_dat  = get_fnames_from_web(fdir_server, data_format)
        Nfile = len(fnames_dat)

        if not all([fnames_dat[i] in fnames_xml[i] for i in range(Nfile)]):
            msg = '\nError [download_oco2_https]: The description files [xml] do not match with data files.'
            raise OSError(msg)

        for i in range(Nfile):
            dtime_s, dtime_e = get_dtime_from_xml('%s/%s' % (fdir_server, fnames_xml[i]))
            if (dtime >= dtime_s) & (dtime <= dtime_e):
                fname_server = '%s/%s' % (fdir_server, fnames_dat[i])
                fnames_server.append(fname_server)

    primary_commands = []
    backup_commands  = []
    fnames_local = []
    for fname_server in fnames_server:
        filename     = os.path.basename(fname_server)
        fname_local  = '%s/%s' % (fdir_out, filename)
        fnames_local.append(fname_local)

        primary_command, backup_command = get_command_earthdata(fname_server, filename=filename, fdir_save=fdir_out, token_mode=False, verbose=verbose)
        primary_commands.append(primary_command)
        backup_commands.append(backup_command)

    if run:
        for i in range(len(primary_commands)):

            fname_local = fnames_local[i]

            if verbose:
                print('Message [download_oco2_https]: Downloading %s ...' % fname_local)

            os.system(primary_commands[i])

            if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                os.system(backup_commands[i])

    else:
        print('Message [download_oco2_https]: The commands to run are:')
        for command in primary_commands:
            print(command)


    return fnames_local



def download_worldview_image(
        date,
        extent,
        fdir_out='tmp-data',
        instrument='modis',
        satellite='aqua',
        wmts_cgi='https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
        layer_name0=None,
        proj=None,
        coastline=False,
        fmt='png',
        dpi=300,
        run=True,
        ):

    """
    Purpose: download satellite RGB imagery from NASA Worldview for a user-specified date and region
    Inputs:
        date: date object of <datetime.datetime>
        extent: rectangular region, Python list of [west_most_longitude, east_most_longitude, south_most_latitude, north_most_latitude]
        fdir_out=: directory to store RGB imagery from NASA Worldview
        instrument=: satellite instrument, currently only supports 'modis' and 'viirs'
        satellite=: satellite, currently only supports 'aqua' and 'terra' for 'modis';  'snpp', 'noaa20', 'noaa21' for 'viirs'
        wmts_cgi=: cgi link to NASA Worldview GIBS (Global Imagery Browse Services)
        proj=: map projection for plotting the RGB imagery
        coastline=: boolen type, whether to plot coastline
        fmt=: can be either 'png' or 'h5'
        run=: boolen type, whether to plot
    Output:
        fname: file name of the saved RGB file (png format)
    Usage example:
        import datetime
        fname = download_wordview_image(datetime.datetime(2022, 5, 18), [-94.26,-87.21,31.86,38.91], instrument='modis', satellite='aqua')
    """

    # get formatted satellite and instrument name
    #╭────────────────────────────────────────────────────────────────────────────╮#
    satname = format_satname(satellite, instrument)
    satellite, instrument = satname.split('|')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # time stamping the satellite imagery (contained in file name)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if satellite in ['Aqua', 'Terra', 'NOAA20', 'SNPP', 'NOAA21']:

        # pick layer
        #╭──────────────────────────────────────────────────────────────╮#
        if layer_name0 is None:
            layer_name0='CorrectedReflectance_TrueColor'
        layer_name = '%s_%s_%s' % (instrument, satellite, layer_name0)
        #╰──────────────────────────────────────────────────────────────╯#

        # calculate time based on the relative location of
        # selected region to satellite granule
        #╭──────────────────────────────────────────────────────────────╮#
        date_s = date.strftime('%Y-%m-%d')

        try:
            lon__ = np.arange(extent[0], extent[1], 500.0/111000.0)
            lat__ = np.arange(extent[2], extent[3], 500.0/111000.0)

            if (lon__.size>800) or (lat__.size>800):
                lon__ = np.linspace(extent[0], extent[1], 800)
                lat__ = np.linspace(extent[2], extent[3], 800)

            lon_, lat_ = np.meshgrid(lon__, lat__, indexing='ij')

            line_data = get_satfile_tag(date, lon_, lat_, satellite=satellite, instrument=instrument, nrt=False, geometa=True, percent0=25.0, worldview=True)[0]

            if satellite in ['Aqua', 'Terra']:
                lon0_, lat0_, jday0_ = cal_lon_lat_utc_geometa(line_data, delta_t=300.0, N_along=1015, N_cross=677, scan='cw', testing=False)
            else:
                lon0_, lat0_, jday0_ = cal_lon_lat_utc_geometa(line_data, delta_t=360.0, N_along=1624, N_cross=1600, scan='cw', testing=False)

            logic_in = (lon0_>=extent[0]) & (lon0_<=extent[1]) & (lat0_>=extent[2]) & (lat0_<=extent[3])
            jday0 = np.nanmean(jday0_[logic_in])
            date0 = jday_to_dtime(jday0)
            date_s0 = date0.strftime('%Y-%m-%dT%H:%M:%SZ')

            fname  = '%s/%s-%s_%s_%s_(%s).png' % (fdir_out, instrument, satellite, layer_name0.split('_')[-1], date_s0, ','.join(['%.4f' % extent0 for extent0 in extent]))

        except Exception as error:
            print(error)
            fname  = '%s/%s-%s_%s_%s_(%s).png' % (fdir_out, instrument, satellite, layer_name0.split('_')[-1], date_s, ','.join(['%.4f' % extent0 for extent0 in extent]))
        #╰──────────────────────────────────────────────────────────────╯#

    elif satellite in ['GOES-West', 'GOES-East']:

        # pick layer
        #╭──────────────────────────────────────────────────────────────╮#
        if layer_name0 is None:
            layer_name0='GeoColor'
        layer_name = '%s_%s_%s' % (satellite, instrument, layer_name0)
        #╰──────────────────────────────────────────────────────────────╯#

        # every 10 minutes, e.g., 10:10, 10:20, 10:30 ...
        #╭──────────────────────────────────────────────────────────────╮#
        delta = datetime.timedelta(minutes=10)
        date = datetime.datetime.min + round((date-datetime.datetime.min)/delta) * delta
        date_s = date.strftime('%Y-%m-%dT%H:%M:%SZ')

        sec_offset = np.nanmean(cal_sec_offset_abi(extent, satname=satname))
        date0 = date + datetime.timedelta(seconds=sec_offset)
        date_s0 = date0.strftime('%Y-%m-%dT%H:%M:%SZ')

        fname  = '%s/%s-%s_%s_%s_(%s).png' % (fdir_out, instrument, satellite, layer_name0.split('_')[-1], date_s0, ','.join(['%.4f' % extent0 for extent0 in extent]))
        #╰──────────────────────────────────────────────────────────────╯#

    fname  = os.path.abspath(fname)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if run:

        import h5py
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpl_img
        import cartopy.crs as ccrs

        if not os.path.exists(fdir_out):
            os.makedirs(fdir_out)

        if proj is None:
            proj=ccrs.PlateCarree()

        try:

            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(111, projection=proj)
            ax1.add_wmts(wmts_cgi, layer_name, wmts_kwargs={'time': date_s})
            if coastline:
                ax1.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            ax1.spines['geo'].set_visible(False)
            ax1.axis('off')
            plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig)

        except Exception as error:

            msg = '\nError [download_wordview_image]: Unable to download imagery for <%s> onboard <%s> at <%s>.' % (instrument, satellite, date_s)
            warnings.warn(msg)
            print('!!!\n%s\n!!!\n' % error)

        if fmt == 'h5':

            data = mpl_img.imread(fname)

            lon  = np.linspace(extent[0], extent[1], data.shape[1])
            lat  = np.linspace(extent[2], extent[3], data.shape[0])

            fname = fname.replace('.png', '.h5')

            f = h5py.File(fname, 'w')

            f['extent'] = extent

            f['lon'] = lon
            f['lon'].make_scale('Longitude')

            f['lat'] = lat
            f['lat'].make_scale('Latitude')

            f['rgb'] = np.swapaxes(data[::-1, :, :3], 0, 1)
            f['rgb'].dims[0].label = 'Longitude'
            f['rgb'].dims[0].attach_scale(f['lon'])
            f['rgb'].dims[1].label = 'Latitude'
            f['rgb'].dims[1].attach_scale(f['lat'])
            f['rgb'].dims[2].label = 'RGB'

            f.close()

    return fname



if __name__ == '__main__':

    pass
