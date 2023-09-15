import os
import sys
import datetime
import numpy as np
import warnings

import er3t.common

import er3t.common
EARTH_RADIUS = er3t.common.params['earth_radius']


__all__ = ['all_files', 'check_equal', 'check_equidistant', 'send_email', \
           'nice_array_str', 'h5dset_to_pydict', 'dtime_to_jday', 'jday_to_dtime', \
           'get_data_nc', 'get_data_h4', \
           'find_nearest', 'move_correlate', \
           'grid_by_extent', 'grid_by_lonlat', 'grid_by_dxdy', \
           'get_doy_tag'] + \
          ['combine_alt', 'get_lay_index', 'downscale', 'upscale_2d', 'mmr2vmr', \
           'cal_rho_air', 'cal_sol_fac', 'cal_mol_ext', 'cal_ext', \
           'cal_r_twostream', 'cal_t_twostream', 'cal_geodesic_dist', 'cal_geodesic_lonlat']


# tools
#/---------------------------------------------------------------------------\
def all_files(root_dir):

    """
    Go through all the subdirectories of the input directory and return all the file paths
    Input:
        root_dir: string, the directory to walk through
    Output:
        allfiles: Python list, all the file paths under the 'root_dir'
    """

    allfiles = []
    for root_dir, dirs, files in os.walk(root_dir):
        for f in files:
            allfiles.append(os.path.join(root_dir, f))

    return sorted(allfiles)



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
    except:
        raise OSError("Error [send_email]: Failed to send the email.")



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



def get_data_nc(nc_dset, nan=True):

    nc_dset.set_auto_maskandscale(True)
    data  = nc_dset[:]

    if nan:
        data.filled(fill_value=np.nan)

    return data



def get_data_h4(hdf_dset, nan=True):

    attrs = hdf_dset.attributes()
    data  = hdf_dset[:]

    if 'scale_factor' in attrs:
        data = data * attrs['scale_factor']

    if 'add_offset' in attrs:
        data = data + attrs['add_offset']

    if '_FillValue' in attrs and nan:
        _FillValue = attrs['_FillValue']
        logic_fill = (data==_FillValue)
        data.astype(np.float64)[logic_fill] = np.nan

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

    corr_coef = np.zeros((2*Ndx+1, 2*Ndy+1), dtype=np.float64)
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
    #/----------------------------------------------------------------------------\#
    if x_out.ndim > 2:
        msg = '\nError [find_nearest]: Only supports <x_out.ndim<=2> and <y_out.ndim<=2>.'
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # preprocess raw data
    #/----------------------------------------------------------------------------\#
    x = np.array(x_raw).ravel()
    y = np.array(y_raw).ravel()
    data = np.array(data_raw).ravel()

    logic_valid = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(data))
    x = x[logic_valid]
    y = y[logic_valid]
    data = data[logic_valid]
    #\----------------------------------------------------------------------------/#


    # create KDTree
    #/----------------------------------------------------------------------------\#
    points = np.transpose(np.vstack((x, y)))
    tree_xy = KDTree(points)
    #\----------------------------------------------------------------------------/#


    # search KDTree for the nearest neighbor
    #/----------------------------------------------------------------------------\#
    points_query = np.transpose(np.vstack((x_out.ravel(), y_out.ravel())))
    dist_xy, indices_xy = tree_xy.query(points_query, workers=-1)
    indices_xy[indices_xy>=data.size] = -1

    dist_out = dist_xy.reshape(x_out.shape)
    data_out = data[indices_xy].reshape(x_out.shape)
    #\----------------------------------------------------------------------------/#


    # use fill value to fill in grids that are "two far"* away from raw data
    #   * by default 1 grid away is defined as "too far"
    #/----------------------------------------------------------------------------\#
    if Ngrid_limit is None:

        logic_out = np.repeat(False, data_out.size).reshape(x_out.shape)

    else:

        dx = np.zeros_like(x_out, dtype=np.float64)
        dy = np.zeros_like(y_out, dtype=np.float64)

        dx[1:, ...] = x_out[1:, ...] - x_out[:-1, ...]
        dx[0, ...]  = dx[1, ...]

        dy[..., 1:] = y_out[..., 1:] - y_out[..., :-1]
        dy[..., 0]  = dy[..., 1]

        dist_limit = np.sqrt((dx*Ngrid_limit)**2+(dy*Ngrid_limit)**2)
        logic_out = (dist_out>dist_limit)

    logic_out = logic_out | (indices_xy.reshape(data_out.shape)==indices_xy.size) | (indices_xy.reshape(data_out.shape)==-1)
    data_out[logic_out] = fill_value
    #\----------------------------------------------------------------------------/#

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
    #/----------------------------------------------------------------------------\#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#

    if extent is None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    else:
        extent = np.float_(np.array(extent))

    if NxNy is None:
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(extent[1]-extent[0]))
        if Nx%2 == 1:
            Nx += 1

        Ny = int(N0*(extent[3]-extent[2]))
        if Ny%2 == 1:
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
    #/----------------------------------------------------------------------------\#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#

    if lon_1d is None or lat_1d is None:

        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)

        Nx = int(N0*(extent[1]-extent[0]))
        if Nx%2 == 1:
            Nx += 1

        Ny = int(N0*(extent[3]-extent[2]))
        if Ny%2 == 1:
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
    #/----------------------------------------------------------------------------\#
    lon = np.array(lon).ravel()
    lat = np.array(lat).ravel()
    data = np.array(data).ravel()
    #\----------------------------------------------------------------------------/#


    # get extent
    #/----------------------------------------------------------------------------\#
    if extent is None:
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    else:
        extent = np.float_(np.array(extent))
    #\----------------------------------------------------------------------------/#


    # dist_x and dist_y
    #/----------------------------------------------------------------------------\#
    if mode == 'min':
        dist_x = np.abs(extent[1]-extent[0])/180.0*np.pi*R_earth*np.cos(np.deg2rad(np.abs(extent[2:]).max()))*1000.0
    elif mode == 'max':
        dist_x = np.abs(extent[1]-extent[0])/180.0*np.pi*R_earth*np.cos(np.deg2rad(np.abs(extent[2:]).min()))*1000.0

    lon0 = [extent[0], extent[1]]
    lat0 = [extent[2], extent[2]]
    lon1 = [extent[0], extent[1]]
    lat1 = [extent[3], extent[3]]
    dist_y = er3t.util.cal_geodesic_dist(lon0, lat0, lon1, lat1).max()
    #\----------------------------------------------------------------------------/#


    # get Nx/Ny and dx/dy
    #/----------------------------------------------------------------------------\#
    if dx is None or dy is None:

        # Nx and Ny
        #/----------------------------------------------------------------------------\#
        xy = (extent[1]-extent[0])*(extent[3]-extent[2])
        N0 = np.sqrt(lon.size/xy)
        Nx = int(N0*(extent[1]-extent[0]))
        Ny = int(N0*(extent[3]-extent[2]))
        #\----------------------------------------------------------------------------/#

        # dx and dy
        #/----------------------------------------------------------------------------\#
        dx = dist_x / Nx
        dy = dist_y / Ny
        #\----------------------------------------------------------------------------/#

    else:

        Nx = int(dist_x // dx)
        Ny = int(dist_y // dy)
    #\----------------------------------------------------------------------------/#


    # get west-most lon_1d/lat_1d
    #/----------------------------------------------------------------------------\#
    lon_1d = np.repeat(extent[0], Ny)
    lat_1d = np.repeat(extent[2], Ny)
    for i in range(1, Ny):
        lon_1d[i], lat_1d[i] = cal_geodesic_lonlat(lon_1d[i-1], lat_1d[i-1], dy, 0.0)
    #\----------------------------------------------------------------------------/#


    # get lon_2d/lat_2d
    #/----------------------------------------------------------------------------\#
    lon_2d = np.zeros((Nx, Ny), dtype=np.float64)
    lat_2d = np.zeros((Nx, Ny), dtype=np.float64)
    lon_2d[0, :] = lon_1d
    lat_2d[0, :] = lat_1d
    for i in range(1, Nx):
        lon_2d[i, :], lat_2d[i, :] = cal_geodesic_lonlat(lon_2d[i-1, :], lat_2d[i-1, :], dx, 90.0)
    #\----------------------------------------------------------------------------/#


    # gridding
    #/----------------------------------------------------------------------------\#
    points   = np.transpose(np.vstack((lon, lat)))

    if method == 'nearest':
        data_2d = find_nearest(lon, lat, data, lon_2d, lat_2d, fill_value=np.nan, Ngrid_limit=Ngrid_limit)
    else:
        data_2d = interpolate.griddata(points, data, (lon_2d, lat_2d), method=method, fill_value=np.nan)

    logic = np.isnan(data_2d)
    data_2d[logic] = fill_value

    return lon_2d, lat_2d, data_2d
    #\----------------------------------------------------------------------------/#



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



# physics
#/---------------------------------------------------------------------------\

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
    if not operation in ['sum', 'mean', 'max']:
        raise ValueError('Error [downscale]: Operation of \'%s\' not supported.' % operation)
    if ndarray.ndim != len(new_shape):
        raise ValueError("Error [downscale]: Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))

    compression_pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
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
    eps = 0.01673
    perh= 2.0
    rsun = (1.0 - eps*np.cos(2.0*np.pi*(doy-perh)/365.0))
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
        if saa_i >= 0.0:
            if 0.0<=saa_i<=180.0:
                saa_i = 180.0 - saa_i
            elif 180.0<saa_i<=360.0:
                saa_i = 540.0 - saa_i
            else:
                saa_i = np.nan
        elif saa_i < 0.0:
            if -180.0<=saa_i<0.0:
                saa_i = -saa_i + 180.0
            elif -360.0<=saa_i<-180.0:
                saa_i = -saa_i - 180.0
            else:
                saa_i = np.nan
        saa[i] = saa_i

    return sza, saa



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
        result=bodhaine(0.5,1000,4)
    Note: If you input an array of wavelengths, the result will also be an
          array corresponding to the Rayleigh optical depth at these wavelengths.
    """

    num = 1.0455996 - 341.29061*wv0**(-2.0) - 0.90230850*wv0**2.0
    den = 1.0 + 0.0027059889*wv0**(-2.0) - 85.968563*wv0**2.0
    tauray = 0.00210966*(num/den)*(pz1-pz2)/1013.25
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

#\---------------------------------------------------------------------------/

if __name__ == '__main__':

    pass
