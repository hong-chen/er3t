import os
import sys
from io import StringIO
import numpy as np
import h5py
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import shutil
import urllib.request
import requests
from er3t import common
from er3t.util import check_equal, get_doy_tag, get_data_h4, get_data_nc, unpack_uint_to_bits
from er3t.util.daac import final_file_check, get_command_earthdata
# import h5py

__all__ = [
        'get_calipso_vfm_rel', \
        'read_calipso_vfm', \
        ]

def get_calipso_vfm_rel(
             date,
             extent, # wesn
             version='v4-51',
             verbose=False):

    """
    Input:
        date: Python datetime.datetime object
        lon : longitude of, e.g. flight track
        lat : latitude of, e.g. flight track
        satID=: default "aqua", can also change to "terra"
        server=: string, data server
        fdir_prefix=: string, data directory on NASA server
        verbose=: Boolen type, verbose tag

    output:
        filename_tags: Python list of file name tags
    """

    search_server = 'https://cmr.earthdata.nasa.gov/opensearch/granules?utf8=✓&'
    if version == 'v4-51':
        concept_id = 'C2667982867-LARC_ASDC'  # for CALIPSO VFM V4-51 product
    elif version == 'v4-20':
        concept_id = 'C1556717900-LARC_ASDC'  # for CALIPSO VFM V4-20 product
    elif version == 'v4-21':
        concept_id = 'C1978624326-LARC_ASDC'  # for CALIPSO VFM V4-21 product
    else:
        print('Error [get_calipso_vfm_rel]: Version not supported.')
        sys.exit()
    lon_w, lon_e, lat_s, lat_n = extent
    yyyy = date.year
    mm = date.month
    dd = date.day
    if lon_w > 180.0:
        lon_w -= 360.0
    if lon_w < -180.0:
        lon_w += 360.0
    if lon_e > 180.0:
        lon_e -= 360.0
    if lon_e < -180.0:
        lon_e += 360.0
    
    print('domain: ', lon_w, lon_e, lat_s, lat_n)

    search_option_id = f'parentIdentifier={concept_id}&'
    search_option_time = f'startTime={yyyy}-{mm:02d}-{dd:02d}T00%3A00%3A00Z&endTime={yyyy}-{mm:02d}-{dd:02d}T23%3A59%3A59Z&'
    search_option_domain = f'spatial_type=bbox&boundingBox={lon_w:.2f}%2C{lat_s:.2f}%2C{lon_e:.2f}%2C{lat_n:.2f}&'
    search_act = 'numberOfResults=49&commit=Search'
    fname_server = search_server + search_option_id + search_option_time + search_option_domain + search_act

    print('fname_server: ', fname_server)
    

    try:
        username = os.environ['EARTHDATA_USERNAME']
        password = os.environ['EARTHDATA_PASSWORD']
    except Exception as err:
        exit('Error   [get_filename_tag]: {}\nCannot find environment variables \'EARTHDATA_USERNAME\' and \'EARTHDATA_PASSWORD\'.'.format(err))

    try:
        with requests.Session() as session:
            session.auth = (username, password)
            r1     = session.request('get', fname_server)
            r      = session.get(r1.url, auth=(username, password))
            if r.ok:
                content = r.content.decode('utf-8')
    except Exception as err:
        exit('Error   [get_filename_tag]: {}\nCannot access {}.'.format(err, fname_server))

    print('content: ', content)
    print('hdf"' in content)

    start_index = 0
    search_content = content
    rel_result = []
    while start_index >= 0:
        start_index = search_content.find('https://asdc.larc.nasa.gov/data/CALIPSO/LID_L2_VFM-Standard-V4-51')
        end_index = search_content.find('.hdf"')
        if start_index > 0:
            rel_result.append(search_content[start_index:end_index+4])
            search_content = search_content[end_index+4:]

    return rel_result

def download_calipso_vfm_http(
                date,
                extent, # wesn
                fdir_out='tmp-data',
                run=True,
                data_format=None,
                verbose=False):
    
    rel_result = get_calipso_vfm_rel(date, extent)

    print('rel_result: ', rel_result)

    # get download commands
    #╭────────────────────────────────────────────────────────────────────────────╮#
    exist_count = 0 # to prevent re-downloading. TODO: Add `overwrite` option instead for user
    primary_commands = []
    backup_commands  = []
    fnames_local = []
    for rel in rel_result:
        filename = rel.split('/')[-1]
        fname_server = rel
        fname_local  = '%s/%s' % (fdir_out, filename)
        if os.path.isfile(fname_local) and final_file_check(fname_local, data_format=data_format, verbose=verbose):
            fnames_local.append(fname_local)
            print("Message [download_calipso_vfm_http]: File {} already exists and looks good. Will not re-download this file.".format(fname_local))
            exist_count += 1
        else:
            fnames_local.append(fname_local)
            primary_command, backup_command = get_command_earthdata(fname_server, filename=filename, fdir_save=fdir_out, verbose=verbose)
            primary_commands.append(primary_command)
            backup_commands.append(backup_command)

    print("Message [download_calipso_vfm_http]: Total of {} will be downloaded. {} will be skipped as they already exist and work as advertised.".format(len(fnames_local), exist_count))
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # run/print command
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if run:

        for i in range(len(primary_commands)):

            fname_local = fnames_local[i]

            if verbose:
                print('Message [download_calipso_vfm_http]: Downloading %s ...' % fname_local)
            os.system(primary_commands[i])

            # if primary command fails, execute backup command.
            # if that fails again, then delete the file and remove from list
            if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                os.system(backup_commands[i])

                if not final_file_check(fname_local, data_format=data_format, verbose=verbose):
                    print("Message [download_calipso_vfm_http]: Could not complete the download of or something is wrong with {}...deleting...".format(fname_local))
                    os.remove(fname_local)
                    fnames_local.remove(fname_local) #remove from list
    else:

        print('Message [download_calipso_vfm_http]: The commands to run are:')
        for command in primary_commands:
            print(command)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return fnames_local

def vfm_row2block(vfm_row):
    """
    # This function is modified from NASA VOCAL calipso code on github:
        https://github.com/NASA-DEVELOP/VOCAL/blob/master/calipso/plot/vfm_row2block.py
    #
    Description: Rearanges a vfm row to a 2d grid
    Inputs: vfm_row - an array 1 x 5515

    Outputs: block - 2d array of vfm data, see function vfm_altitude for
            altitude array information. Altitude array is in similar format as
            CALIPSO L1 profile data (i.e. it isn't uniform).

    Translated from Matlab:
    Brian Magill
    10/16/2013

    Written by Ralph Kuehn
    3/24/2005

    The layout of a row of VFM data is described at
    http://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/vfm/index.php#feature_classification_flags
    under the heading 'Layout of the Feature Classification Flag data block.'
    """

    #   For higher altitude data, info will be over-sampled in horizontal dimension
    #   for 8-20km block it will be 200x15 = 3000 rather than 200x5 = 1000
    #   for 20-30 km block it will be 55x15 = 825, rather than 55x3 = 165

    #   Resolutions defined here are defined in terms of lengths of index numbers
    #
    HIGH_ALT_RES = 55
    MID_ALT_RES = 200
    LOW_ALT_RES = 290
    ALT_DIM = HIGH_ALT_RES + MID_ALT_RES + LOW_ALT_RES

    block = np.ones((ALT_DIM, 15), dtype=np.uint8)
    offset = 0
    step = HIGH_ALT_RES
    indA = 0
    indB = HIGH_ALT_RES

    for i in range(3):

        iLow = offset + step * i
        iHi = iLow + step

        n = i * 5
        #        block[indA:indB, n:n+4] = vfm_row[iLow:iHi]

        for k in range(5):
            block[indA:indB, n + k] = vfm_row[iLow:iHi]

    offset = 3 * HIGH_ALT_RES
    step = MID_ALT_RES
    indA = HIGH_ALT_RES
    indB = HIGH_ALT_RES + MID_ALT_RES

    for i in range(5):

        iLow = offset + step * i
        iHi = iLow + step

        n = i * 3
        #        block[indA:indB, n:n+2] = vfm_row[iLow:iHi]

        for k in range(3):
            block[indA:indB, n + k] = vfm_row[iLow:iHi]

            # element 1,1 correspond to Alt -0.5km, position -2.5 km from center lat lon.

    offset = 3 * HIGH_ALT_RES + 5 * MID_ALT_RES
    step = LOW_ALT_RES
    indA = HIGH_ALT_RES + MID_ALT_RES
    indB = ALT_DIM

    for i in range(15):
        iLow = offset + step * i
        iHi = iLow + step
        block[indA:indB, i] = vfm_row[iLow:iHi]

    return block

def extract_type(vfm_array):
    """
    # This function is modified from NASA VOCAL calipso code on github:
        https://github.com/NASA-DEVELOP/VOCAL/blob/master/calipso/plot/interpret_vfm_type.py
    #
    Extracts feature type for each element in a vertical feature mask array:

        0 = invalid (bad or missing data)
        1 = 'clear air'
        2 = cloud
        3 = aerosol
        4 = stratospheric feature
        5 = surface
        6 = subsurface
        7 = no signal (totally attenuated)

    """
    mask_3bits = np.uint16(7)
    return np.bitwise_and(mask_3bits, vfm_array)

def uniform_alt_2(max_altitude, old_altitude_array):
    """
    
    # This function is modified from NASA VOCAL calipso code on github:
        https://github.com/NASA-DEVELOP/VOCAL/blob/master/calipso/plot/uniform_alt_2.py
    #
    # uniform_alt_2.py
    # Translated by Brian Magill
    # 12/31/2013
    #
    #
    # Description (8/2013):  Builds a uniformly spaced altitude grid above region 2
    # of the CALIOP lidar data.  The spacing is 30 km.  From what I have been told
    # the idea here is to have a 30 km spacing instead of 60 km.  Note that the altitude
    # is stored in descending order.
    # 
    # Parameters:
    #   max_altitude        - [in] maximum altitude for grid (should be above region 2)
    #   old_altitude_array  - [in] region 2 altitudes
    #
    #   new_alt             - [out] output array with region 2 and above
    #
    """

    D_ALT = 0.03 # spacing is 30 km
    MID_RES_TOP = 288
    MID_RES_BOT = 576

    # Altitude indices for high res altitude region (region 2):
    # 288:576
    
    alt2 = old_altitude_array[MID_RES_TOP:MID_RES_BOT]

    new_num_bins = int(np.ceil((max_altitude-alt2[0])/D_ALT))

    new_length = int(new_num_bins + len(alt2))

    new_alt = np.zeros(int(new_length))
    new_alt[int(new_num_bins):int(new_length)] = alt2

    upper_altitudes =  (np.arange(new_num_bins) + 1.)*D_ALT
    new_alt[:int(new_num_bins)] = new_alt[int(new_num_bins)] + upper_altitudes[::-1]

    return new_alt

def regrid_lidar(alt, inMatrix, new_alt, method = 'linear'):
    """
    # This function is modified from NASA VOCAL calipso code on github:
        https://github.com/NASA-DEVELOP/VOCAL/blob/master/calipso/plot/regrid_lidar.py
    #
    # regrid_lidar.py
    # translated by Brian Magill
    # 12/31/2013
    #
    #
    # This function will regrid the matrix inMatrix defined by the (Nx1) vector 'alt'
    # onto the new grid (Jx1) 'new_alt'.
    # The assumption is that the horizontal dimension changes column by
    # column, and altitude is stored row by row (e.g. row x col == alt x (dist
    # or time).
    #
    # Note that all values outside of bounds are returned as NaN's
    # For interp1d to work, the ordinate array has to be monotonically increasing
    # This is why the altitude and inMatrix arrays have been reversed in their
    # common dimension.  
    #
    """
    

    interpFunc = interpolate.interp1d(alt[::-1], inMatrix[::-1,:], kind=method, 
                                      axis=0, bounds_error=False)

    return interpFunc(new_alt)

def read_calipso_vfm(filename, extent, x_range=(0, 1000), y_range=(0, 20), fig_output='./tmp', plot=False):
    """
    # This function is modified from NASA VOCAL calipso code on github:
        https://github.com/NASA-DEVELOP/VOCAL/blob/master/calipso

    _summary_
    """

    import ccplot.utils
    from ccplot.hdf import HDF

    # 15 profiles are in 1 record of VFM data. At the highest altitudes 5 profiles are averaged
    # together. In the mid altitudes 3 are averaged and at roughly 8 km or less, there are
    # separate profiles.
    prof_per_row = 15

    # constant variables
    alt_len = 545
    first_alt = y_range[0]
    last_alt = y_range[1]
    first_lat = int(x_range[0]/prof_per_row)
    last_lat = int(x_range[1]/prof_per_row)
    colormap = f'{common.fdir_er3t}/er3t/dev/calipso-vfm.cmap'
    
    lon_w, lon_e, lat_s, lat_n = extent
    
    # naming products within the HDF file
    with HDF(filename) as product:
        latitude = product['Latitude'][:, 0]
        first_lat = np.argmin(np.abs(latitude-lat_s))
        last_lat = np.argmin(np.abs(latitude-lat_n))
        
        time = product['Profile_UTC_Time'][first_lat:last_lat, 0]
        minimum = min(product['Profile_UTC_Time'][::])[0]
        maximum = max(product['Profile_UTC_Time'][::])[0]

        # Determine how far the file can be viewed
        if time[-1] >= maximum and len(time) < 950:
            raise IndexError
        if time[0] < minimum:
            raise IndexError

        height = product['metadata']['Lidar_Data_Altitudes'][33:-5:]
        dataset = product['Feature_Classification_Flags'][first_lat:last_lat, :]
        latitude = product['Latitude'][first_lat:last_lat, 0]
        print('latitude: ', latitude)
        #latitude = latitude[::prof_per_row]
        #print('latitude: ', latitude)
        time = np.array([ccplot.utils.calipso_time2dt(t) for t in time])

        # Mask all unknown values
        dataset = np.ma.masked_equal(dataset, -999)

        # Give the number of rows in the dataset
        num_rows = dataset.shape[0]

        # Create an empty array the size of of L1 array so they match on the plot
        unpacked_vfm = np.zeros((alt_len, prof_per_row * num_rows), np.uint8)

        # Assign the values from 0-7 to subtype
        vfm = extract_type(dataset)

        # Place 15-wide, alt_len-tall blocks of data into the
        for i in range(num_rows):
            unpacked_vfm[:, prof_per_row * i:prof_per_row * (i + 1)] = vfm_row2block(vfm[i, :])
        vfm = unpacked_vfm
        
        max_alt = 20
        unif_alt = uniform_alt_2(max_alt, height)
        regrid_vfm = regrid_lidar(height, vfm, unif_alt)
        unif_alt_expanded = np.tile(unif_alt, (regrid_vfm.shape[1], 1)).T
        
        if plot:
            # Format color map
            cmap = ccplot.utils.cmap(colormap)
            cm = matplotlib.colors.ListedColormap(cmap['colors'] / 255.0)
            cm.set_under(cmap['under'] / 255.0)
            cm.set_over(cmap['over'] / 255.0)
            cm.set_bad(cmap['bad'] / 255.0)
            norm = matplotlib.colors.BoundaryNorm(cmap['bounds'], cm.N)

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(
                regrid_vfm,
                extent=(latitude[0], latitude[-1], first_alt, last_alt),
                cmap=cm,
                aspect='auto',
                norm=norm,
                interpolation='nearest',
            )

            ax.set_ylabel('Altitude (km)')
            ax.set_xlabel('Latitude')
            ax.set_title("Vertical Feature Mask")

            cbar_label = 'Vertical Feature Mask Flags'
            cbar = fig.colorbar(im)
            cbar.set_label(cbar_label)
            # Set labels using dict in interpret_vfm_type
            cbar.ax.set_yticks(np.arange(1.5, 8))
            cbar.ax.set_yticklabels(['Clear\nAir','Cloud','Aerosol','Stratospheric\nAerosol',
                            'Surface','Subsurface','Totally\nAttenuated'])

            ax2 = ax.twiny()
            ax2.set_xlabel('Time')
            ax2.set_xlim(time[0], time[-1])
            ax2.get_xaxis().set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))

            ax2.set_zorder(0)
            ax2.set_zorder(1)

            title = ax.set_title('Vertical Feature Mask')
            title_xy = title.get_position()
            title.set_position([title_xy[0], title_xy[1] * 1.07])
        
            fig.savefig(f'{fig_output}/calipso_lidr_flag.png', bbox_inches='tight', dpi=300)
    return regrid_vfm, unif_alt_expanded 
