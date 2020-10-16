import os
import sys
import numpy as np



__all__ = ['all_files', 'check_equal', 'send_email', 'nice_array_str', 'h5dset_to_pydict'] + \
          ['combine_alt', 'get_lay_index', 'downgrading', 'mmr2vmr', 'cal_rho_air', 'cal_sol_fac', \
           'cal_mol_ext', 'cal_ext', 'cal_r_twostream', 'cal_dist', 'cal_cth_hist']


# tools
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        exit("Error   [send_email]: Failed to send the email.")



def nice_array_str(array1d, numPerLine=6):

    """
    Covert 1d array to string

    Input:
        array1d: numpy array, 1d array to be converted to string

    Output:
        converted string
    """

    if array1d.ndim > 1:
        sys.exit('Error   [nice_array_str]: Only support 1-D array.')

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

# -----------------------------------------------------------------------------------------------------------------


# physics
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def combine_alt(atm_z, cld_z):

    z1 = atm_z[atm_z < cld_z.min()]
    if z1.size == 0:
        print('Warning [combine_alt]: cloud locates below the bottom of the atmosphere.')

    z2 = atm_z[atm_z > cld_z.max()]
    if z2.size == 0:
        print('Warning [combine_alt]: cloud locates above the top of the atmosphere.')

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
            sys.exit("Error   [get_layer_index]: Mismatch between layer and reference layer: "+str(dd))

        layer_index = np.append(layer_index, index)

    return layer_index



def downgrading(ndarray, new_shape, operation='mean'):

    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Input:
        ndarray: numpy array, any dimension of array to be downgraded
        new_shape: Python tuple or list, new dimension/shape of the array
        operation=: string, can be 'mean' or 'sum', default='mean'

    Output:
        ndarray: numpy array, downgraded array
    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError('Error   [downgrading]: Operation of \'%s\' not supported.' % operation)
    if ndarray.ndim != len(new_shape):
        raise ValueError("Error   [downgrading]: Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))

    compression_pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



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
        sys.exit("Error [cal_rho_air]: input variables have different dimensions.")

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



def cal_ext(cot, cer, Qe=2.0):

    """
    Calculate extinction (m^-1) from cloud optical thickness and cloud effective radius
    Input:
        cot: float or array, cloud optical thickness
        cer: float or array, cloud effective radius in micro meter (10^-6 m)
    """

    # liquid water path
    # from equation 7.86 in Petty's book
    #           3*lwp
    # cot = ---------------, where rho is the density of water
    #         2*rho*cer
    lwp  = 2.0/3000.0 * cot * cer


    # Extinction
    # from equation 7.70 in Petty's book
    #            3*Qe
    # ext = ---------------, where rho is the density of water
    #         4*rho*cer
    ext = 0.75 * Qe * lwp / cer * 1.0e3

    return ext



def cal_r_twostream(tau, a=0.0, g=0.85, mu=1.0):

    """
    Two-stream approximation no absorption

    Input:
        a: surface albedo
        g: asymmetry parameter
        mu: cosine of solar zenith angle

    Output:
        Reflectance
    """

    x = 2.0 * mu / (1.0-g) / (1.0-a)
    r = (tau + a*x) / (tau + x)

    return r



def cal_dist(delta_degree, earth_radius=6378.0):

    """
    Calculate distance from longitude/latitude difference

    Input:
        delta_degree: float or numpy array

    Output:
        dist: float or numpy array, distance in km
    """

    dist = delta_degree/180.0 * (np.pi*earth_radius)

    return dist



def cal_cth_hist(cth):

    """
    Calculate the cloud top height based on the peak of histograms

    Input:
        cth: cloud top height

    Output:
        cth_peak: cloud top height of the peak of histogram
    """

    hist = np.histogram(cth)


    pass

# -----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    pass
