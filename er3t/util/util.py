import os
import sys
import fnmatch
import datetime
import numpy as np
import warnings

import er3t.common


EARTH_RADIUS = er3t.common.params['earth_radius']

__all__ = ['get_all_files', 'get_all_folders', 'load_h5', \
           'check_equal', 'check_equidistant', 'send_email', \
           'nice_array_str', 'h5dset_to_pydict', 'dtime_to_jday', 'jday_to_dtime', \
           'get_data_nc', 'get_data_h4', \
           'find_nearest', 'move_correlate', \
           'grid_by_extent', 'grid_by_lonlat', 'grid_by_dxdy', \
           'get_doy_tag', 'add_reference', 'print_reference', \
           'combine_alt', 'get_lay_index', 'downscale', 'upscale_2d', 'mmr2vmr', \
           'cal_rho_air', 'cal_sol_fac', 'cal_mol_ext_atm', 'mol_ext_wvl', 'cal_mol_ext', 'cal_ext', \
           'cal_r_twostream', 'cal_t_twostream', 'cal_geodesic_dist', 'cal_geodesic_lonlat', \
           'format_time', 'region_parser', 'parse_geojson', 'unpack_uint_to_bits']


def get_all_files(fdir, pattern='*'):

    fnames = []
    for fdir_root, fdir_sub, fnames_tmp in os.walk(fdir):
        for fname_tmp in fnames_tmp:
            if fnmatch.fnmatch(fname_tmp, pattern):
                fnames.append(os.path.join(fdir_root, fname_tmp))
    return sorted(fnames)



def get_all_folders(fdir, pattern='*'):

    fnames = get_all_files(fdir)

    folders = []
    for fname in fnames:
        folder_tmp = os.path.abspath(os.path.dirname(os.path.relpath(fname)))
        if (folder_tmp not in folders) and fnmatch.fnmatch(folder_tmp, pattern):
                folders.append(folder_tmp)

    return folders



def load_h5(fname):

    import h5py

    def get_variable_names(obj, prefix=''):

        """
        Purpose: Walk through the file and extract information of data groups and data variables

        Input: h5py file object <f>, e.g., f = h5py.File('file.h5', 'r')

        Outputs:
            data variable path in the format of <['group1/variable1']> to
            mimic the style of accessing HDF5 data variables using h5py, e.g.,
            <f['group1/variable1']>
        """

        for key in obj.keys():

            item = obj[key]
            path = '{prefix}/{key}'.format(prefix=prefix, key=key)
            if isinstance(item, h5py.Dataset):
                yield path
            elif isinstance(item, h5py.Group):
                yield from get_variable_names(item, prefix=path)

    data = {}
    f = h5py.File(fname, 'r')
    keys = get_variable_names(f)
    for key in keys:
        data[key[1:]] = f[key[1:]][...]
    f.close()
    return data



def check_equal(a, b, threshold=1.0e-6):

    """
    Check if two values are equal (or close to each other)
    Input:
        a: integer or float, value of a
        b: integer or float, value of b
    Output:
        boolen, true or false
    """

    if abs(a-b) >= threshold:
        return False
    else:
        return True



def check_equidistant(z, threshold=1.0e-6):

    """
    Check if an array is equidistant (or close to each other)
    Input:
        z: numpy array
    Output:
        boolen, true or false
    """

    if not isinstance(z, np.ndarray):
        msg = '\nError [check_equidistant]: Only support numpy.ndarray.'
        raise ValueError(msg)

    if z.size < 2:
        msg = '\nError [check_equidistant]: Too few data for checking.'
        raise ValueError(msg)
    else:
        dz = z[1:] - z[:-1]

    fac = dz/dz[0]

    if np.abs(fac-1.0).sum() >= (threshold*z.size):
        return False
    else:
        return True



def send_email(
        content=None,             \
        files=None,               \
        receiver='me@hongchen.cz' \
        ):


    """
    Send email using default account er3t@hongchen.cz
    Input:
        content= : string, text content of the email
        files=   : Python list, contains file paths of the email attachments
        receiver=: string, reveiver's email address
    Output:
        None
    """

    import socket
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication
    import datetime

    sender_email    = 'er3t@hongchen.cz'
    sender_password = 'er3t@cuboulder'

    msg = MIMEMultipart()
    msg['Subject'] = '%s@%s: %s' % (os.getlogin(), socket.gethostname(), sys.argv[0])
    msg['From']    = 'er3t'
    msg['To']      = receiver

    if content is None:
        content = 'No message.'
    msg_detl = 'Details:\nName: %s/%s\nPID: %d\nTime: %s' % (os.getcwd(), sys.argv[0], os.getpid(), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    msg_body = '%s\n\n%s\n' % (content, msg_detl)
    msg.attach(MIMEText(msg_body))

    for fname in files or []:
        with open(fname, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(fname))
        part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(fname)
        msg.attach(part)

    try:
        server = smtplib.SMTP('mail.hongchen.cz', port=587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [receiver], msg.as_string())
        server.quit()
    except Exception as err:
        raise OSError(err, "Error [send_email]: Failed to send the email.")



def nice_array_str(array1d, numPerLine=6):

    """
    Covert 1d array to string
    Input:
        array1d: numpy array, 1d array to be converted to string
    Output:
        converted string
    """

    if array1d.ndim > 1:
        raise ValueError('Error [nice_array_str]: Only support 1-D array.')

    niceString = ''
    numLine    = array1d.size // numPerLine
    numRest    = array1d.size  % numPerLine

    for iLine in range(numLine):
        lineS = ''
        for iNum in range(numPerLine):
            lineS += '  %12g' % array1d[iLine*numPerLine + iNum]
        lineS += '\n'
        niceString += lineS
    if numRest != 0:
        lineS = ''
        for iNum in range(numRest):
            lineS += '  %12g' % array1d[numLine*numPerLine + iNum]
        lineS += '\n'
        niceString += lineS

    return niceString



def h5dset_to_pydict(dset):

    """
    Retreive information about the H5 dataset and
    store them into a Python dictionary
    e.g.,
    The dataset dset = f['mean/f_down'] can be converted into
    variable = {
                'data' : f_down    ,                # numpy.array
                'units': 'W/m^2/nm',                # string
                'name' : 'Global downwelling flux'  # string
    }
    """

    data = {}

    for var in dset.attrs.keys():
        data[var]  = dset.attrs[var]

    data['data']  = dset[...]

    return data



def dtime_to_jday(dtime):

    """
    Purpose: convert regular date and time (Python <datetime.datetime> object) to julian day (referenced to 0001-01-01)
    Input:
        dtime: Python <datetime.datetime> object
    Output:
        jday: julian day (float number)
    """

    jday = (dtime - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0

    return jday



def jday_to_dtime(jday):

    """
    Purpose: convert julian day (referenced to 0001-01-01) to regular date and time (Python <datetime.datetime> object)
    Input:
        jday: julian day (float number)
    Output:
        dtime: Python <datetime.datetime> object
    """

    dtime = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=np.round(((jday-1)*86400.0), decimals=0))

    return dtime


def get_data_h4(hdf_dset, init_dtype=None, replace_fill_value=np.nan):
    """
    Retrieves data from an HDF dataset and performs optional data type conversion and fill value replacement.

    Args:
    ----
        hdf_dset (h5py.Dataset): The HDF dataset to retrieve data from.
        init_dtype (dtype, optional): The desired data type for the retrieved data. Defaults to None.
        replace_fill_value (float or int, optional): The value to replace the fill value with. Defaults to np.nan.

    Returns:
        numpy.ndarray: The retrieved data with optional data type conversion and fill value replacement.
    """

    attrs = hdf_dset.attributes()
    data  = hdf_dset[:]
    if init_dtype is not None:
        data = np.array(data, dtype=init_dtype)

    # Check if the dataset has a fill value attribute and if fill value replacement is requested
    if '_FillValue' in attrs and replace_fill_value is not None:
        # If the replacement fill value is NaN, convert the fill value attribute to float64
        if np.isnan(replace_fill_value):
            _FillValue = np.array(attrs['_FillValue'], dtype='float64')
            data = data.astype('float64')

        else: # otherwise let the fill value be the same data type as the dataset
            _FillValue = np.array(attrs['_FillValue'], dtype=data.dtype)

        # Replace the fill values in the dataset with the replacement fill value
        data[data == _FillValue] = replace_fill_value

    # If the dataset has an add_offset attribute, subtract it from the data
    if 'add_offset' in attrs:
        data = data - attrs['add_offset']

    # If the dataset has a scale_factor attribute, multiply it with the data
    if 'scale_factor' in attrs:
        data = data * attrs['scale_factor']

    # Return the processed data
    return data



def get_data_nc(nc_dset, replace_fill_value=np.nan):

    nc_dset.set_auto_maskandscale(True)
    data  = nc_dset[:]

    if replace_fill_value is not None:
        data = data.astype('float32')
        data.filled(fill_value=replace_fill_value)

    return data



def move_correlate(data0, data, Ndx=5, Ndy=5):

    try:
        from scipy import stats
    except ImportError:
        msg = '\nError [move_correlate]: `scipy` installation is required.'
        raise ImportError(msg)

    Nx, Ny = data.shape
    x = np.arange(Nx, dtype=np.int32)
    y = np.arange(Ny, dtype=np.int32)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    xx0 = xx.copy()
    yy0 = yy.copy()
    valid = np.ones((Nx, Ny), dtype=np.int32)

    corr_coef = np.zeros((2*Ndx+1, 2*Ndy+1), dtype=np.float32)
    dxx = np.arange(-Ndx, Ndx+1)
    dyy = np.arange(-Ndy, Ndy+1)

    for idx, dx in enumerate(dxx):
        xx_ = xx + dx
        valid[xx_< 0 ] = 0
        valid[xx_>=Nx] = 0
        for idy, dy in enumerate(dyy):
            yy_ = yy + dy
            valid[yy_< 0 ] = 0
            valid[yy_>=Ny] = 0

            logic = (valid == 1)
            data0_ = data0[xx0[logic], yy0[logic]]
            data_  = data[xx_[logic], yy_[logic]]

            corr_coef[idx, idy] = stats.pearsonr(data0_, data_)[0]

    indices = np.unravel_index(np.argmax(corr_coef, axis=None), corr_coef.shape)

    offset_dx = -dxx[indices[0]]
    offset_dy = -dyy[indices[1]]

    return offset_dx, offset_dy



def find_nearest(x_raw, y_raw, data_raw, x_out, y_out, Ngrid_limit=1, fill_value=np.nan):

    """
    Use scipy.spatial.KDTree to perform fast nearest gridding

    References:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html

    Inputs:
        x_raw: x position of raw data
        y_raw: y position of raw data
        data_raw: value of raw data
        x_out: x position of the data (e.g., x of data to be gridded)
        y_out: y position of the data (e.g., y of data to be gridded)
        Ngrid_limit=<1>=: number of grids for defining "too far"
        fill_value=<np.nan>: fill-in value for the data that is "too far" away from raw data

    Output:
        data_out: gridded data
    """

    try:
        from scipy.spatial import KDTree
    except ImportError:
        msg = 'Error [find_nearest]: `scipy` installation is required.'
        raise ImportError(msg)

    # only support output at maximum dimension of 2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if x_out.ndim > 2:
        msg = '\nError [find_nearest]: Only supports <x_out.ndim<=2> and <y_out.ndim<=2>.'
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # preprocess raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    x = np.array(x_raw).ravel()
    y = np.array(y_raw).ravel()
    data = np.array(data_raw).ravel()

    logic_valid = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(data))
    x = x[logic_valid]
    y = y[logic_valid]
    data = data[logic_valid]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # create KDTree
    #╭────────────────────────────────────────────────────────────────────────────╮#
    points = np.transpose(np.vstack((x, y)))
    tree_xy = KDTree(points)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # search KDTree for the nearest neighbor
    #╭────────────────────────────────────────────────────────────────────────────╮#
    points_query = np.transpose(np.vstack((x_out.ravel(), y_out.ravel())))
    dist_xy, indices_xy = tree_xy.query(points_query, workers=-1)
    indices_xy[indices_xy>=data.size] = -1

    dist_out = dist_xy.reshape(x_out.shape)
    data_out = data[indices_xy].reshape(x_out.shape)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # use fill value to fill in grids that are "two far"* away from raw data
    #   * by default 1 grid away is defined as "too far"
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if Ngrid_limit is None:

        logic_out = np.repeat(False, data_out.size).reshape(x_out.shape)

    else:

        dx = np.zeros_like(x_out, dtype=np.float32)
        dy = np.zeros_like(y_out, dtype=np.float32)

        dx[1:, ...] = x_out[1:, ...] - x_out[:-1, ...]
        dx[0, ...]  = dx[1, ...]

        dy[..., 1:] = y_out[..., 1:] - y_out[..., :-1]
        dy[..., 0]  = dy[..., 1]

        dist_limit = np.sqrt((dx*Ngrid_limit)**2+(dy*Ngrid_limit)**2)
        logic_out = (dist_out>dist_limit)

    logic_out = logic_out | (indices_xy.reshape(data_out.shape)==indices_xy.size) | (indices_xy.reshape(data_out.shape)==-1)
    data_out[logic_out] = fill_value
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return data_out



def grid_by_extent(lon, lat, data, extent=None, NxNy=None, method='nearest', fill_value=0.0, Ngrid_limit=1):

    """
    Grid irregular data into a regular grid by input 'extent' (westmost, eastmost, southmost, northmost)
    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        extent=: Python list, [westmost, eastmost, southmost, northmost]
        NxNy=: Python list, [Nx, Ny], lon_2d = np.linspace(westmost, eastmost, Nx)
                                      lat_2d = np.linspace(southmost, northmost, Ny)
    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data
    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat, data = grid_by_extent(lon0, lat0, data0, extent=[10, 15, 10, 20])
    """

    try:
        from scipy import interpolate
    except ImportError:
        msg = '\nError [grid_by_extent]: `scipy` installation is required.'
        raise ImportError(msg)

    # flatten lon/lat/data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if extent is None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    else:
        extent = np.float_(np.array(extent))

    if NxNy is None:
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(extent[1]-extent[0]))
        if Nx % 2 == 1:
            Nx += 1

        Ny = int(N0*(extent[3]-extent[2]))
        if Ny % 2 == 1:
            Ny += 1
    else:
        Nx, Ny = NxNy

    lon_1d0 = np.linspace(extent[0], extent[1], Nx+1)
    lat_1d0 = np.linspace(extent[2], extent[3], Ny+1)

    lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
    lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)

    points   = np.transpose(np.vstack((lon, lat)))

    if method == 'nearest':
        data_2d = find_nearest(lon, lat, data, lon_2d, lat_2d, fill_value=np.nan, Ngrid_limit=Ngrid_limit)
    else:
        data_2d = interpolate.griddata(points, data, (lon_2d, lat_2d), method=method, fill_value=np.nan)

    logic = np.isnan(data_2d)
    data_2d[logic] = fill_value

    return lon_2d, lat_2d, data_2d



def grid_by_lonlat(lon, lat, data, lon_1d=None, lat_1d=None, method='nearest', fill_value=0.0, Ngrid_limit=1):

    """
    Grid irregular data into a regular grid by input longitude and latitude
    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        lon_1d=: numpy array, the longitude of the grids
        lat_1d=: numpy array, the latitude of the grids
    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data
    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat, data = grid_by_lonlat(lon0, lat0, data0, lon_1d=np.linspace(10.0, 15.0, 100), lat_1d=np.linspace(10.0, 20.0, 100))
    """

    try:
        from scipy import interpolate
    except ImportError:
        msg = '\nError [grid_by_lonlat]: `scipy` installation is required.'
        raise ImportError(msg)

    # flatten lon/lat/data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if lon_1d is None or lat_1d is None:

        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(extent[1]-extent[0]))
        if Nx % 2 == 1:
            Nx += 1

        Ny = int(N0*(extent[3]-extent[2]))
        if Ny % 2 == 1:
            Ny += 1

        lon_1d0 = np.linspace(extent[0], extent[1], Nx+1)
        lat_1d0 = np.linspace(extent[2], extent[3], Ny+1)

        lon_1d = (lon_1d0[1:]+lon_1d0[:-1])/2.0
        lat_1d = (lat_1d0[1:]+lat_1d0[:-1])/2.0

    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d)

    points   = np.transpose(np.vstack((lon, lat)))

    if method == 'nearest':
        data_2d = find_nearest(lon, lat, data, lon_2d, lat_2d, fill_value=np.nan, Ngrid_limit=Ngrid_limit)
    else:
        data_2d = interpolate.griddata(points, data, (lon_2d, lat_2d), method=method, fill_value=np.nan)

    logic = np.isnan(data_2d)
    data_2d[logic] = fill_value

    return lon_2d, lat_2d, data_2d



def grid_by_dxdy(lon, lat, data, extent=None, dx=None, dy=None, method='nearest', mode='min', fill_value=0.0, Ngrid_limit=1, R_earth=EARTH_RADIUS):

    """
    Grid irregular data into a regular xy grid by input 'extent' (westmost, eastmost, southmost, northmost)
    Input:
        lon: numpy array, input longitude to be gridded
        lat: numpy array, input latitude to be gridded
        data: numpy array, input data to be gridded
        extent=: Python list, [westmost, eastmost, southmost, northmost]
        dx=: float, zonal spatial resolution in meter
        dy=: float, meridional spatial resolution in meter
    Output:
        lon_2d : numpy array, gridded longitude
        lat_2d : numpy array, gridded latitude
        data_2d: numpy array, gridded data
    How to use:
        After read in the longitude latitude and data into lon0, lat0, data0
        lon, lat, data = grid_by_dxdy(lon0, lat0, data0, dx=250.0, dy=250.0)
    """

    try:
        from scipy import interpolate
    except ImportError:
        msg = '\nError [grid_by_dxdy]: `scipy` installation is required.'
        raise ImportError(msg)

    # flatten lon/lat/data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()*1.0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get extent
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if extent is None:
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    else:
        extent = np.float_(np.array(extent))
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # dist_x and dist_y
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if mode == 'min':
        dist_x = np.abs(extent[1]-extent[0])/180.0*np.pi*R_earth*np.cos(np.deg2rad(np.abs(extent[2:]).max()))*1000.0
    elif mode == 'max':
        dist_x = np.abs(extent[1]-extent[0])/180.0*np.pi*R_earth*np.cos(np.deg2rad(np.abs(extent[2:]).min()))*1000.0

    lon0 = [extent[0], extent[1]]
    lat0 = [extent[2], extent[2]]
    lon1 = [extent[0], extent[1]]
    lat1 = [extent[3], extent[3]]
    dist_y = cal_geodesic_dist(lon0, lat0, lon1, lat1).max()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get Nx/Ny and dx/dy
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if dx is None or dy is None:

        # Nx and Ny
        #╭──────────────────────────────────────────────────────────────╮#
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)
        Nx = int(N0*(extent[1]-extent[0]))
        Ny = int(N0*(extent[3]-extent[2]))
        #╰──────────────────────────────────────────────────────────────╯#

        # dx and dy
        #╭──────────────────────────────────────────────────────────────╮#
        dx = dist_x / Nx
        dy = dist_y / Ny
        #╰──────────────────────────────────────────────────────────────╯#

    else:

        Nx = int(dist_x // dx)
        Ny = int(dist_y // dy)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get west-most lon_1d/lat_1d
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon_1d = np.repeat(extent[0], Ny)
    lat_1d = np.repeat(extent[2], Ny)
    for i in range(1, Ny):
        lon_1d[i], lat_1d[i] = cal_geodesic_lonlat(lon_1d[i-1], lat_1d[i-1], dy, 0.0)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get lon_2d/lat_2d
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon_2d = np.zeros((Nx, Ny), dtype=np.float32)
    lat_2d = np.zeros((Nx, Ny), dtype=np.float32)
    lon_2d[0, :] = lon_1d
    lat_2d[0, :] = lat_1d
    for i in range(1, Nx):
        lon_2d[i, :], lat_2d[i, :] = cal_geodesic_lonlat(lon_2d[i-1, :], lat_2d[i-1, :], dx, 90.0)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # gridding
    #╭────────────────────────────────────────────────────────────────────────────╮#
    points   = np.transpose(np.vstack((lon, lat)))

    if method == 'nearest':
        data_2d = find_nearest(lon, lat, data, lon_2d, lat_2d, fill_value=np.nan, Ngrid_limit=Ngrid_limit)
    else:
        data_2d = interpolate.griddata(points, data, (lon_2d, lat_2d), method=method, fill_value=np.nan)

    logic = np.isnan(data_2d)
    data_2d[logic] = fill_value

    return lon_2d, lat_2d, data_2d
    #╰────────────────────────────────────────────────────────────────────────────╯#



def get_doy_tag(date, day_interval=8):

    """
    Get day of year tag, e.g., 078, for a given day
    Input:
        date: datetime/date object, e.g., datetime.datetime(2000, 1, 1)
    Output:
        doy_tag: string, closest day of the year, e.g. '097'
    """

    doy = date.timetuple().tm_yday

    day_total = datetime.datetime(date.year, 12, 31).timetuple().tm_yday

    doys = np.arange(1, day_total+1, day_interval)

    doy_tag = '%3.3d' % doys[np.argmin(np.abs(doys-doy))]

    return doy_tag



def add_reference(reference, reference_list=er3t.common.references):

    if reference not in reference_list:

        reference_list.append(reference)



def print_reference():

    print('\nReferences:')
    print('╭────────────────────────────────────────────────────────────────────────────╮')
    for reference in er3t.common.references:
        print(reference)
    print('╰────────────────────────────────────────────────────────────────────────────╯')
    print()

    return



def combine_alt(atm_z, cld_z):

    z1 = atm_z[atm_z < cld_z.min()]
    if z1.size == 0:
        msg = 'Warning [combine_alt]: cloud locates below the bottom of the atmosphere.'
        warnings.warn(msg)

    z2 = atm_z[atm_z > cld_z.max()]
    if z2.size == 0:
        msg = 'Warning [combine_alt]: cloud locates above the top of the atmosphere.'
        warnings.warn(msg)

    z = np.concatenate(z1, cld_z, z2)

    return z



def get_lay_index(lay, lay_ref):

    """
    Check where the input 'lay' locates in input 'lay_ref'.
    Input:
        lay    : numpy array, layer height
        lay_ref: numpy array, reference layer height
        threshold=: float, threshold of the largest difference between 'lay' and 'lay_ref'
    Output:
        layer_index: numpy array, indices for where 'lay' locates in 'lay_ref'
    """

    threshold = (lay_ref[1:]-lay_ref[:-1]).max()/2.0

    layer_index = np.array([], dtype=np.int32)

    for i, z in enumerate(lay):

        index = np.argmin(np.abs(z-lay_ref))

        dd = np.abs(z-lay_ref[index])
        if dd > threshold:
            print(z, lay_ref[index])
            raise ValueError("Error [get_layer_index]: Mismatch between layer and reference layer: "+str(dd))

        layer_index = np.append(layer_index, index)

    return layer_index



def downscale(ndarray, new_shape, operation='mean'):

    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.
    Input:
        ndarray: numpy array, any dimension of array to be downscaled
        new_shape: Python tuple or list, new dimension/shape of the array
        operation=: string, can be 'mean', 'sum', or 'max', default='mean'
    Output:
        ndarray: numpy array, downscaled array
    """
    operation = operation.lower()
    if operation not in ['sum', 'mean', 'max', 'median']:
        raise ValueError('Error [downscale]: Operation of \'%s\' not supported.' % operation)
    if ndarray.ndim != len(new_shape):
        raise ValueError("Error [downscale]: Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))

    compression_pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    if operation == 'median':
        ndarray = np.median(ndarray, axis=1)
    else:
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
    return ndarray



def upscale_2d(ndarray, scale_factors=(1, 1)):

    Nx, Ny = ndarray.shape
    scale_factor_x, scale_factor_y = scale_factors

    data = np.zeros((Nx*scale_factor_x, Ny*scale_factor_y), dtype=ndarray.dtype)
    for i in range(scale_factor_x):
        for j in range(scale_factor_y):
            data[i::scale_factor_x, j::scale_factor_y] = ndarray

    return data



def mmr2vmr(mmr):

    """
    Convert water vapor mass mixing ratio to volume mixing ratio (=partial pressure ratio)
    Input:
        mmr: numpy array, mass mixing ratio
    Output:
        vmr: numpy array, volume mixing ratio
    """

    Md  = 0.0289644   # molar mass of dry air  [kg/mol]
    Mv  = 0.0180160   # model mass of water    [kg/mol]
    q   = mmr/(1-mmr)
    vmr = q/(q+Mv/Md)

    return vmr



def cal_rho_air(p, T, vmr):

    """
    Calculate the density of humid air [kg/m3]
    Input:
        p: numpy array, pressure in hPa
        T: numpy array, temperature in K
        vmr: numpy array, water vapor mixing ratio
    Output:
        rho: numpy array, density of humid air in kg/m3
    """

    # pressure [hPa], temperature [K], vapor volume mixing ratio (=partial pressure ratio)
    p   = np.array(p)*100.
    T   = np.array(T)
    vmr = np.array(vmr)

    # check that dimensions are the same (1d,2d,3d)
    pd = p.shape
    Td = T.shape
    vd = vmr.shape
    if ((pd != Td) | (vd != Td)):
        raise ValueError("Error [cal_rho_air]: input variables have different dimensions.")

    R   = 8.31447     # ideal gas constant     [J /(mol K)]
    Md  = 0.0289644   # molar mass of dry air  [kg/mol]
    Mv  = 0.0180160   # model mass of water    [kg/mol]
    rho = p*Md/(R*T)*(1-vmr*(1-Mv/Md)) # [kg/m3]

    return rho



def cal_sol_fac(dtime):

    """
    Calculate solar factor that accounts for Sun-Earth distance
    Input:
        dtime: datetime.datetime object
    Output:
        solfac: solar factor
    """

    doy = dtime.timetuple().tm_yday
    eps = 0.0167086
    perh= 4.0
    rsun = (1.0 - eps*np.cos(0.017202124161707175*(doy-perh)))
    solfac = 1.0/(rsun**2)

    return solfac



def cal_sol_ang(julian_day, longitude, latitude, altitude):

    """
    Calculate solar angles - solar zenith angle and solar azimuth angle
    Input:
        julian_day: julian data (day count starting from 0001-01-01)
        longitude: longitude in degree
        latitude: latitude in degree
        altitude: altitude in meter
    Output:
        sza: solar zenith angle
        saa: solar azimuth angle
    """

    dateRef = datetime.datetime(1, 1, 1)
    jdayRef = 1.0

    sza = np.zeros_like(julian_day)
    saa = np.zeros_like(julian_day)

    for i in range(julian_day.size):

        jday = julian_day[i]

        dtime_i = (dateRef + datetime.timedelta(days=jday-jdayRef)).replace(tzinfo=datetime.timezone.utc)

        sza_i = 90.0 - pysolar.solar.get_altitude(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
        if sza_i < 0.0 or sza_i > 90.0:
            sza_i = np.nan
        sza[i] = sza_i

        saa_i = pysolar.solar.get_azimuth(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
        # if saa_i >= 0.0:
        #     if 0.0<=saa_i<=180.0:
        #         saa_i = 180.0 - saa_i
        #     elif 180.0<saa_i<=360.0:
        #         saa_i = 540.0 - saa_i
        #     else:
        #         saa_i = np.nan
        # elif saa_i < 0.0:
        #     if -180.0<=saa_i<0.0:
        #         saa_i = -saa_i + 180.0
        #     elif -360.0<=saa_i<-180.0:
        #         saa_i = -saa_i - 180.0
        #     else:
        #         saa_i = np.nan
        saa[i] = saa_i

    return sza, saa



def g0_calc(lat):

    """
    Calculate the surface gravity acceleration.

    according to Eq. 11 of Bodhaine et al, `On Rayleigh optical depth calculations', J. Atm. Ocean Technol., 16, 1854-1861, 1999.
    """

    lat_rad = lat * np.pi / 180

    return 9.806160 * (1 - 0.0026373 * np.cos(2*lat_rad) + 0.0000059 * np.cos(2*lat_rad)**2) # in m/s^2



def g_alt_calc(g0, lat, z):

    """
    Calculate the gravity acceleration at z.

    according to Eq. 10 of Bodhaine et al, `On Rayleigh optical depth calculations', J. Atm. Ocean Technol., 16, 1854-1861, 1999.

    Input:
        g0: gravity acceleration at the surface (m/s^2)
        lat: latitude (degrees)
        z: height (m)
    """

    lat_rad = lat * np.pi / 180
    g = g0*100 - (3.085462e-4 + 2.27e-7 * np.cos(2 * lat_rad)) * z \
           + (7.254e-11 + 1.0e-13 * np.cos(2 * lat_rad)) * z**2 \
           - (1.517e-17 + 6.0e-20 * np.cos(2 * lat_rad)) * z**3
    return g/100



def cal_mol_ext_atm(wv0, atm0, method='atm'):

    """
    Input:
        wv0    : wavelength (in microns) --- can be an array
        atm0   : er3t atmosphere object
        method=: string, 'sfc', 'atm', or 'lay'
    Output:
        tauray: extinction

    in Python program:
        result=cal_mol_ext_atm(0.5, atm0)
    Note: If you input an array of wavelengths, the result will also be an
          array corresponding to the Rayleigh optical depth at these wavelengths.
    """

    reference = 'Rayleigh Extinction (Bodhaine et al., 1999):\n- Bodhaine, B. A., Wood, N. B., Dutton, E. G., and Slusser, J. R.: On Rayleigh Optical Depth Calculations, J. Atmos. Ocean. Tech., 16, 1854–1861, 1999.'

    # avogadro's number
    A_ = 6.02214179e23
    if hasattr(atm0, 'lat'):
        lat = atm0.lat
    else:
        lat = 0.0 # default latitude is 0 degree

    g0 = g0_calc(lat) # m/s^2
    z = atm0.lay['altitude']['data']
    g = g_alt_calc(g0, lat, z*1000.0) * 100.0 # convert to cm/s^2

    g0 = g0 * 100.0 # convert to cm/s^2
    ma = 28.9595 + (15.0556 * atm0.lay['co2']['data']/atm0.lay['air']['data'])

    p_lev = atm0.lev['pressure']['data'] * 1000.0 # convert to dyne/cm^2
    dp_lev = (p_lev[:-1]-p_lev[1:]) # convert to dyne/cm^2
    crs = mol_ext_wvl(wv0)

    # original calculation
    # tauray = 0.00210966*(crs)*(p_lev[:-1]-p_lev[1:])/1013.25

    if method == 'sfc':
        const_sfc = p_lev[0] * A_ / (g0 * ma[0]) * 1.0e-28
        tauray = const_sfc*(crs)*(p_lev[:-1]-p_lev[1:])/p_lev[0]
    elif method == 'lay':
        const_lay = dp_lev * A_ / (g * ma) * 1.0e-28
        tauray = const_lay*(crs)
    elif method == 'atm':
        tauray = (crs) * 1.0e-28 * atm0.lay['air']['data'] * atm0.lay['thickness']['data'] * 1000.0 * 100.0
    else:
        msg = 'Error [cal_mol_ext_atm]: method not supported.'
        raise ValueError(msg)

    add_reference(reference)

    return tauray



def mol_ext_wvl(wv0):

    """
    Calculate the rayleigh scattering cross-section for given wavelength.

    according to Eq. 29 of Bodhaine et al, `On Rayleigh optical depth calculations', J. Atm. Ocean Technol., 16, 1854-1861, 1999.

    Input:
        wv0: wavelength (in microns)
    """

    num = 1.0455996 - 341.29061*wv0**(-2.0) - 0.90230850*wv0**2.0
    den = 1.0 + 0.0027059889*wv0**(-2.0) - 85.968563*wv0**2.0
    crs = num/den

    return crs   # in 10^-28 cm^2/molecule



def cal_mol_ext(wv0, pz1, pz2):

    """
    Input:
        wv0: wavelength (in microns) --- can be an array
        pz1: numpy array, Pressure of lower layer (hPa)
        pz2: numpy array, Pressure of upper layer (hPa; pz1 > pz2)
    Output:
        tauray: extinction
    Example: calculate Rayleigh optical depth between 37 km (~4 hPa) and sea level (1000 hPa) at 0.5 microns:
    in Python program:
        result = cal_mol_ext(0.5,1000,4)
    Note: If you input an array of wavelengths, the result will also be an
          array corresponding to the Rayleigh optical depth at these wavelengths.
    """

    reference = 'Rayleigh Extinction (Bodhaine et al., 1999):\n- Bodhaine, B. A., Wood, N. B., Dutton, E. G., and Slusser, J. R.: On Rayleigh Optical Depth Calculations, J. Atmos. Ocean. Tech., 16, 1854–1861, 1999.'

    tauray = 0.00210966 * mol_ext_wvl(wv0) * (pz1-pz2) / 1013.25

    add_reference(reference)

    return tauray



def cal_ext(cot, cer, dz=1.0, Qe=2.0):

    """
    Calculate extinction (m^-1) from cloud optical thickness and cloud effective radius
    Input:
        cot: float or array, cloud optical thickness
        cer: float or array, cloud effective radius in micro meter (10^-6 m)
    """

    # liquid water path [g/m^2]
    # from equation 7.86 in Petty's book
    #           3*lwp
    # cot = ---------------, where rho is the density of water
    #         2*rho*cer
    lwp  = 2.0/3000.0 * cot * cer

    # liquid water content [g/m^3]
    # assume vertically homogeneous distribution of cloud water
    lwc  = lwp / dz

    # Extinction
    # from equation 7.70 in Petty's book
    #            3*Qe
    # ext = ---------------, where rho is the density of water
    #         4*rho*cer
    ext = 0.75 * Qe * lwc / cer * 1.0e3

    return ext



def cal_r_twostream(tau, a=0.0, g=0.85, mu=1.0):

    """
    Two-stream approximation of reflectance (no absorption)
    Input:
        tau: optical thickness
        a: surface albedo
        g: asymmetry parameter
        mu: cosine of solar zenith angle
    Output:
        Reflectance
    """

    x = 2.0 * mu / (1.0-g) / (1.0-a)
    r = (tau + a*x) / (tau + x)

    return r



def cal_t_twostream(tau, a=0.0, g=0.85, mu=1.0):

    """
    Two-stream approximation of transmittance (no absorption)
    Input:
        a: surface albedo
        g: asymmetry parameter
        mu: cosine of solar zenith angle
    Output:
        Transmittance
    """

    x = 2.0 * mu / (1.0-g) / (1.0-a)
    t = x*(1.0-a) / (tau + x)

    return t



def cal_geodesic_dist(lon0, lat0, lon1, lat1):

    try:
        import cartopy.geodesic as cg
    except ImportError:
        msg = '\nError [cal_geodesic_dist]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon0 = np.array(lon0).ravel()
    lat0 = np.array(lat0).ravel()
    lon1 = np.array(lon1).ravel()
    lat1 = np.array(lat1).ravel()

    geo0 = cg.Geodesic()

    points0 = np.transpose(np.vstack((lon0, lat0)))

    points1 = np.transpose(np.vstack((lon1, lat1)))

    output = np.squeeze(np.asarray(geo0.inverse(points0, points1)))

    dist = output[..., 0]

    return dist



def cal_geodesic_lonlat(lon0, lat0, dist, azimuth):

    try:
        import cartopy.geodesic as cg
    except ImportError:
        msg = '\nError [cal_geodesic_lonlat]: Please install <cartopy> to proceed.'
        raise ImportError(msg)

    lon0 = np.array(lon0).ravel()
    lat0 = np.array(lat0).ravel()
    dist = np.array(dist).ravel()
    azimuth = np.array(azimuth).ravel()

    points = np.transpose(np.vstack((lon0, lat0)))

    geo0 = cg.Geodesic()

    output = np.squeeze(np.asarray(geo0.direct(points, azimuth, dist)))

    lon1 = output[..., 0]
    lat1 = output[..., 1]

    return lon1, lat1



def parse_geojson(geojson_fpath):

    import json
    with open(geojson_fpath, 'r') as f:
        data = json.load(f)
        # n_coords = len(data['features'][0]['geometry']['coordinates'][0])

    coords = data['features'][0]['geometry']['coordinates']

    lons = np.array(coords[0])[:, 0]
    lats = np.array(coords[0])[:, 1]
    return lons, lats



def region_parser(extent, lons, lats, geojson_fpath):
    """
    Parse region specifications and return longitude and latitude arrays.
    This function processes different forms of region specifications: extent, lon/lat coordinates, or a geoJSON file.
    It validates inputs and returns arrays of longitudes and latitudes that define the region.
    Args:
    ----
        extent (list or None): Region extent as [lon_min, lon_max, lat_min, lat_max]
            (i.e., West, East, South, North).
        lons (list or None): Longitude bounds as [lon_min, lon_max].
        lats (list or None): Latitude bounds as [lat_min, lat_max].
        geojson_fpath (str or None): File path to a geoJSON file containing region information.

    Returns:
    -------
        tuple: A tuple containing:
            - llons (numpy.ndarray): Array of longitudes linearly spaced across the region.
            - llats (numpy.ndarray): Array of latitudes linearly spaced across the region.
    Raises:
        SystemExit: If inputs are invalid or insufficient to define a region.
    """

    if (extent is None) and ((lats is None) or (lons is None)) and (geojson_fpath is None):
        print('Error [region_parser]: Must provide either extent or lon/lat coordinates or a geoJSON file')
        sys.exit()

    if (extent is not None) and ((lats is not None) or (lons is not None)) and (geojson_fpath is not None):
        print('Warning [region_parser]: Received multiple regions of interest. Only `extent` will be used.')
        llons = np.linspace(extent[0], extent[1], 100)
        llats = np.linspace(extent[2], extent[3], 100)
        return llons, llats


    if (extent is not None):
        if (len(extent) != 4) and ((lats is None) or (lons is None) or (len(lats) == 0) or (len(lons) == 0)):
            print('Error [region_parser]: Must provide either extent with [lon1 lon2 lat1 lat2] or lon/lat coordinates via --lons and --lats')
            sys.exit()

        # check to make sure extent is correct
        if (extent[0] >= extent[1]) or (extent[2] >= extent[3]):
            msg = 'Error [region_parser]: The given extents of lon/lat are incorrect: %s.\nPlease check to make sure extent is passed as `lon1 lon2 lat1 lat2` format i.e. West, East, South, North.' % extent
            print(msg)
            sys.exit()

        llons = np.linspace(extent[0], extent[1], 100)
        llats = np.linspace(extent[2], extent[3], 100)
        return llons, llats

    elif (lats is not None) and (lons is not None):
        if ((len(lats) == 2) and (len(lons) == 2)) and (lons[0] < lons[1]) and (lats[0] < lats[1]):
            llons = np.linspace(lons[0], lons[1], 100)
            llats = np.linspace(lats[0], lats[1], 100)
            return llons, llats
        else:
            print('Error [region_parser]: Must provide two coorect bounds each for `--lons` and `--lats`')
            sys.exit()


    elif (geojson_fpath is not None):
        llons, llats = parse_geojson(geojson_fpath)
        return llons, llats



def format_time(total_seconds):
    """
    Convert seconds to hours, minutes, seconds, and milliseconds.

    Parameters:
    - total_seconds: The total number of seconds to convert.

    Returns:
    - A tuple containing hours, minutes, seconds, and milliseconds.
    """
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = (total_seconds - int(total_seconds)) * 1000

    return (int(hours), int(minutes), int(seconds), int(milliseconds))



def unpack_uint_to_bits(uint_array, num_bits, bitorder='big'):
    """
    Unpack a uint16 or 32 or 64 array into binary bits.
    """

    # convert to right dtype
    uint_array = uint_array.astype('uint{}'.format(num_bits))

    if num_bits == 8: # just use numpy
        bits = np.unpackbits(uint_array.flatten(), bitorder=bitorder)
        # num_bits has to be the last dimensions to get the right array
        bits = bits.reshape(list(uint_array.shape) + [num_bits])
        # now we can transpose
        return np.transpose(bits, axes=(2, 0, 1))

    elif (num_bits == 16) or (num_bits == 32) or (num_bits == 64):

        # Convert uintxx array to uint8 array
        uint8_array = uint_array.view(np.uint8).reshape(-1, int(num_bits/8))

        # Unpack bits from uint8 array
        # force little endian since big endian seems to pad an extra 0
        # and then reverse it if needed
        bits = np.unpackbits(uint8_array, bitorder='little', axis=1)

        # Reshape to match original uint16 array shape with an additional dimension for bits
        # note that num_bits must be the last dimension here to get the right reshaped array
        bits = bits.reshape(list(uint_array.shape) + [num_bits])

    else:
        raise ValueError("Only uint8, uint16, uint32, and uint64 dtypes are supported. `num_bits` must be >=8 ")

    if bitorder == 'big': # reverse the order
        return np.transpose(bits[:, :, ::-1], axes=(2, 0, 1))

    return np.transpose(bits, axes=(2, 0, 1))



def has_common_substring(input_str, substring_list):
    """
    Check if the input string contains any of the substrings from the list without using for loops.

    Args:
    ----
        input_str (str): The string to check against.
        substring_list (list): List of substrings to look for in the input string.

    Returns:
    -------
        bool: True if input_str contains any substring from substring_list, False otherwise.
    """
    return any(substring in input_str for substring in substring_list)



if __name__ == '__main__':

    pass
