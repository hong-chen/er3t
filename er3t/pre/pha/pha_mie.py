import os
import sys
import copy
import pickle
import numpy as np
import warnings
from scipy import interpolate

import er3t.common
import er3t.util



__all__ = ['pha_mie_wc']




def read_mie(fname):

    """
    Read phase function file (netCDF) from libRadtran

    Input:
        fname: file path of the file

    Output:
        wvl, ref, ssa, pmom = read_pmom(fname)

        wvl: wavelength in nm
        ref: effective radius in mm
        ssa: single scattering albedo
        pmom: pmom coefficients
    """

    if er3t.common.has_netcdf4:
        from netCDF4 import Dataset
    else:
        msg = 'Error [read_pmom]: Please install <netCDF4> to proceed.'
        raise ImportError(msg)

    f = Dataset(fname, 'r')

    wvl  = f.variables['wavelen'][...].data

    # effective radius
    ref  = f.variables['reff'][...].data

    # single scattering albedo
    ssa  = f.variables['ssa'][...].data

    # angle
    ang  = f.variables['theta'][...].data

    # phase function
    pha  = f.variables['phase'][...].data
    
    # Asymmetry factor
    if 'gg' in f.variables.keys():
        gg  = f.variables['gg'][...].data
    else:
        gg  = None

    f.close()

    wvl = wvl * 1000.0       # from micron to nm
    ang  = ang[:, :, 0, :]   # pick the first of 4 stokes
    pha  = pha[:, :, 0, :]   # pick the first of 4 stokes

    return wvl, ref, ssa, ang, pha, gg

class pha_mie_wc:

    """
    Calculate Mie phase functions (water clouds) for a given wavelength and angles

    Input:
        wavelength: wavelength in nm, float value, default is 500.0
        angles: numpy array, angles to extract mie phase functions at
        *interpolate: boolen, whether to interpolate phase functions based on the wavelength, default is False
        overwrite: boolen, whether to overwrite the pre-existing phase functions stored at er3t/data/pha/mie, default is True
        verbose: boolen, whether to print all the messages, default is False

    Output:
        phase object, e.g.,
        pha0 = pha_mie_wc(wvl0=500.0)

        pha0.data['id']: identification
        pha0.data['wvl0']: given wavelength
        pha0.data['wvl']: actual wavelength
        pha0.data['ang']: angles
        pha0.data['pha']: phase functions
        pha0.data['ssa']: single scattering albedo
        pha0.data['asy']: asymmetry parameter
        pha0.data['ref']: effective radius
    """

    fname_coef = '%s/wc.sol.mie.cdf' % er3t.common.fdir_data_pha

    reference = '\nMie Scattering (Wiscombe, 1980):\n- Wiscombe, W.: Improved Mie scattering algorithms, Applied Optics, 19, 1505–1509, https://doi.org/10.1364/AO.19.001505, 1980.'

    ID = 'Mie (Water Clouds)'

    def __init__(self,
                 wavelength=555.0,
                 angles=np.concatenate((
                    np.arange(  0.0,   2.0, 0.01),
                    np.arange(  2.0,   5.0, 0.05),
                    np.arange(  5.0,  10.0, 0.1),
                    np.arange( 10.0,  15.0, 0.5),
                    np.arange( 15.0, 176.0, 1.0),
                    np.arange(176.0, 180.1, 0.25),
                 )),
                 fdir_pha_mie = '%s/pha/mie' % er3t.common.fdir_data_tmp,
                 interpolate=False,
                 angles_fine=False,
                 overwrite=True,
                 verbose=False):

        er3t.util.add_reference(self.reference)

        self.interpolate = interpolate
        self.angles_fine = angles_fine
        self.overwrite   = overwrite
        self.verbose     = verbose

        self.get_data(wavelength, angles, fdir=fdir_pha_mie)

    def get_data(self,
            wvl0,
            angles,
            fdir='%s/pha/mie' % er3t.common.fdir_data_tmp,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = '%s/pha_mie_wc_%09.4fnm.pk' % (fdir, wvl0)

        if not self.overwrite:
            if os.path.exists(fname):
                with open(fname, 'rb') as f0:
                    data0 = pickle.load(f0)
                if np.abs(angles-data0['ang']['data']).sum() < 0.00000001:
                    print('Message [pha_mie_wc]: Re-using phase function from <%s> ...' % fname)
                    self.data = copy.deepcopy(data0)
                else:
                    self.run(fname, wvl0, angles)
            else:
                self.run(fname, wvl0, angles)
        else:
            self.run(fname, wvl0, angles)

    def run(self,
            fname,
            wvl0,
            angles
            ):

        Na = angles.size
        wvl, ref, ssa, ang_all, pha_all, asy_all = read_mie(self.fname_coef)

        Nwvl, Nreff, Nang = pha_all.shape

        if not self.interpolate:
            iwvl = np.argmin(np.abs(wvl-wvl0))
        else:
            msg = 'Error [pha_mie_wc]: Interpolation has not been implemented.'
            raise ValueError(msg)
        
        if self.angles_fine:
            # use the fine angles in the pre-calculated table
            ang_iwvl = ang_all[iwvl, :, :]
            angles = np.array(sorted(set(ang_iwvl[ang_iwvl>=0])))
            Na = angles.size

        pha  = np.zeros((Na, Nreff), dtype=np.float64)
        if asy_all is None:
            asy_ = np.zeros(Nreff, dtype=np.float64)
            asy  = np.zeros(Nreff, dtype=np.float64)
            mus  = np.cos(np.deg2rad(angles))


        
        for ireff in range(Nreff):

            ang0_ = ang_all[iwvl, ireff, :]
            logic0 = (ang0_>=0) & (ang0_<=180)
            ang0 = ang_all[iwvl, ireff, logic0]
            mu0  = np.cos(np.deg2rad(ang0))
            pha0 = pha_all[iwvl, ireff, logic0]

            f_pha0 = interpolate.interp1d(ang0, pha0, kind='linear')

            pha[:, ireff] = f_pha0(angles)
            
            if asy_all is None:
                asy[ireff]  = np.trapz(pha0*mu0, x=mu0)/2.0
                asy_[ireff] = np.trapz(pha[::-1, ireff]*mus[::-1], x=mus[::-1])/2.0
            
        if asy_all is not None:
            # use the pre-calculated asymmetry parameter from the file
            asy = asy_all[iwvl, :]
            asy_ = asy_all[iwvl, :]

        data = {
                'id'   : {'data':'Mie'       , 'name':'Mie'                , 'unit':'N/A'},
                'wvl0' : {'data':wvl0        , 'name':'Given wavelength'   , 'unit':'nm'},
                'wvl'  : {'data':wvl[iwvl]   , 'name':'Actual wavelength'  , 'unit':'nm'},
                'ang'  : {'data':angles      , 'name':'Angle'              , 'unit':'degree'},
                'pha'  : {'data':pha         , 'name':'Phase function'     , 'unit':'N/A'},
                'ssa'  : {'data':ssa[iwvl, :], 'name':'Single scattering albedo', 'unit':'N/A'},
                'asy'  : {'data':asy         , 'name':'Asymmetry parameter'     , 'unit':'N/A'},
                'asy_' : {'data':asy_        , 'name':'Asymmetry parameter_'    , 'unit':'N/A'},
                'ref'  : {'data':ref         , 'name':'Effective radius'        , 'unit':'mm'}
                }

        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        print('Message [pha_mie_wc]: Phase function for %.2fnm has been stored at <%s>.' % (wvl0, fname))

        self.data = data





def legendre2phase(
        poly_coef,
        angle=None,
        deltascaling=True,
        normalize=False,
        step=0.01,
        lrt=False
        ):

    Npoly = poly_coef.size
    if deltascaling:
        poly_coef = (poly_coef-poly_coef[-1])/(1.0-poly_coef[-1])

    poly_coef *= (2.0*np.arange(Npoly)+1.0)

    if normalize:
        factors = 1.0/poly_coef[0]
        poly_coef *= factors

    if angle is None:
        angle = np.arange(0.0, 180.0+step, step)

    mu    = np.cos(np.deg2rad(angle))

    if lrt:
        phase     = np.zeros_like(mu)
        for i, mu0 in enumerate(mu):
            phase[i]     = mom2phase(poly_coef, mu0)
    else:
        phase = np.polynomial.legendre.legval(mu, poly_coef)

    return phase

def mom2phase(polys, mu):

    """
    Purpose: calculate phase function from phase function moments.
             Adapted from libRadtran/libsrc_c/miecalc.c:<mom2phase>
             by Bernhard Mayer

    Inputs:
        polys: Legendre moment vector
        mu: cosine angles

    Output:
        phase function

    by Hong Chen (hong.chen.cu@gmail.com)
    """

    plm1 = mu
    plm2 = 1.0

    pha = plm2*polys[0] + plm1*polys[1]

    Npoly = polys.size
    for i in range(2, Npoly):
        plm0 = ((2.0*i - 1.0)*mu*plm1 - (i-1)*plm2) / i

        pha += polys[i] * plm0

        plm2 = plm1
        plm1 = plm0

    return pha

def mom2phaseint(polys, mu):

    """
    Purpose: Calculate integral of the phase function from -1 to x from the phase function moments
             Adapted from libRadtran/src/phase.c:<mom2phaseint>

    Inputs:
        polys: Legendre moment vector
        mu: cosine angles

    Output:
        pha_int: Integral of phase function

    by Hong Chen (hong.chen.cu@gmail.com)
    """

    plm1 = mu
    plm2 = 1.0

    pldashm1 = 1.0
    pldashm2 = 0.0

    pha_int = (1.0 - mu)*polys[0] + 0.5*(1.0 - mu**2) * polys[1]

    Npoly = polys.size
    for i in range(2, Npoly):
        plm0     = ((2.0*i - 1.0)*mu*plm1 - (i-1)*plm2) / i
        pldashm0 = ((2.0*i - 1.0)*(plm1 + mu*pldashm1) - (i-1)*pldashm2) / i

        pha_int += polys[i] * (1.0 - mu**2) / (i * (i+1))*pldashm0

        plm2 = plm1
        plm1 = plm0

        pldashm2 = pldashm1
        pldashm1 = pldashm0

    return pha_int

def read_pmom(fname):

    """
    Read phase function file (netCDF) from libRadtran

    Input:
        fname: file path of the file

    Output:
        wvl, ref, ssa, pmom = read_pmom(fname)

        wvl: wavelength in nm
        ref: effective radius in mm
        ssa: single scattering albedo
        pmom: pmom coefficients
    """

    if er3t.common.has_netcdf4:
        from netCDF4 import Dataset
    else:
        msg = 'Error [read_pmom]: Please install <netCDF4> to proceed.'
        raise ImportError(msg)

    f = Dataset(fname, 'r')

    wvl  = f.variables['wavelen'][...].data

    # effective radius
    ref  = f.variables['reff'][...].data

    # single scattering albedo
    ssa  = f.variables['ssa'][...].data

    # legendre polynomial coefficients
    pmom = f.variables['pmom'][...].data

    f.close()

    wvl = wvl * 1000.0       # from micron to nm
    pmom = pmom[:, :, 0, :]  # pick the first of 4 stokes

    return wvl, ref, ssa, pmom

class pha_mie_wc_pmom:

    """
    Calculate Mie phase functions (water clouds) for a given wavelength and angles

    Input:
        wavelength: wavelength in nm, float value, default is 500.0
        angles: numpy array, angles to extract mie phase functions at
        *interpolate: boolen, whether to interpolate phase functions based on the wavelength, default is False
        overwrite: boolen, whether to overwrite the pre-existing phase functions stored at er3t/data/pha/mie, default is True
        verbose: boolen, whether to print all the messages, default is False

    Output:
        phase object, e.g.,
        pha0 = pha_mie_wc(wvl0=500.0)

        pha0.data['id']: identification
        pha0.data['wvl0']: given wavelength
        pha0.data['wvl']: actual wavelength
        pha0.data['ang']: angles
        pha0.data['pha']: phase functions
        pha0.data['ssa']: single scattering albedo
        pha0.data['asy']: asymmetry parameter
        pha0.data['ref']: effective radius
    """

    fname_coef = '%s/wc.sol.mie.cdf' % er3t.common.fdir_data_pha

    reference = 'Wiscombe, W.: Improved Mie scattering algorithms, Applied Optics, 19, 1505–1509, 1980.'

    def __init__(self,
                 wavelength=555.0,
                 angles=np.concatenate((
                    np.arange(  0.0,   2.0, 0.01),
                    np.arange(  2.0,   5.0, 0.05),
                    np.arange(  5.0,  10.0, 0.1),
                    np.arange( 10.0,  15.0, 0.5),
                    np.arange( 15.0, 176.0, 1.0),
                    np.arange(176.0, 180.1, 0.25),
                 )),
                 fdir_pha_mie = '%s/pha/mie' % er3t.common.fdir_data_tmp,
                 interpolate=False,
                 overwrite=True,
                 verbose=False):


        self.interpolate = interpolate
        self.overwrite= overwrite
        self.verbose = verbose

        if self.reference not in er3t.common.references:
            er3t.common.references.append(self.reference)

        self.get_data(wavelength, angles, fdir=fdir_pha_mie)

    def get_data(self,
            wvl0,
            angles,
            fdir='%s/pha/mie' % er3t.common.fdir_data_tmp,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = '%s/pha_mie_wc_%09.4fnm.pk' % (fdir, wvl0)

        if not self.overwrite:
            if os.path.exists(fname):
                with open(fname, 'rb') as f0:
                    data0 = pickle.load(f0)
                if np.abs(angles-data0['ang']['data']).sum() < 0.00000001:
                    print('Message [pha_mie_wc]: Re-using phase function from <%s> ...' % fname)
                    self.data = copy.deepcopy(data0)
                else:
                    self.run(fname, wvl0, angles)
            else:
                self.run(fname, wvl0, angles)
        else:
            self.run(fname, wvl0, angles)

    def run(self,
            fname,
            wvl0,
            angles
            ):

        Na = angles.size
        wvl, ref, ssa, pmom = read_pmom(self.fname_coef)
        Nwvl, Nreff, Npoly = pmom.shape

        if not self.interpolate:
            iwvl = np.argmin(np.abs(wvl-wvl0))
        else:
            msg = '\nError [pha_mie_wc]: Interpolation has not been implemented.'
            raise ValueError(msg)

        pha = np.zeros((Na, Nreff), dtype=np.float64)
        mus = np.cos(np.deg2rad(angles))
        asy = np.zeros(Nreff, dtype=np.float64)

        for ireff in range(Nreff):

            pmom0 = pmom[iwvl, ireff, :]

            if pmom0[-1] > 0.001:
                msg = '\nWarning [pha_mie]: Ref=%.2f Legendre series did not converge.' % ref[ireff]
                warnings.warn(msg)

            pmom0 = pmom0/(2.0*np.arange(Npoly)+1.0)

            pha0 = legendre2phase(pmom0, angle=angles)
            pha[:, ireff] = pha0

            # asymmetry parameter
            # half of the integral of: from cos(ang)=-1 to cos(ang)=1 for function pha(ang)*cos(ang)
            asy[ireff] = np.trapz(pha0[::-1]*mus[::-1], x=mus[::-1])/2.0

        data = {
                'id'  : {'data':'Mie'       , 'name':'Mie'                , 'unit':'N/A'},
                'wvl0': {'data':wvl0        , 'name':'Given wavelength'   , 'unit':'nm'},
                'wvl' : {'data':wvl[iwvl]   , 'name':'Actual wavelength'  , 'unit':'nm'},
                'ang' : {'data':angles      , 'name':'Angle'              , 'unit':'degree'},
                'pha' : {'data':pha         , 'name':'Phase function'     , 'unit':'N/A'},
                'ssa' : {'data':ssa[iwvl, :], 'name':'Single scattering albedo', 'unit':'N/A'},
                'asy' : {'data':asy         , 'name':'Asymmetry parameter'     , 'unit':'N/A'},
                'ref' : {'data':ref         , 'name':'Effective radius'        , 'unit':'mm'}
                }

        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        print('Message [pha_mie_wc]: Phase function for %.2fnm has been stored at <%s>.' % (wvl0, fname))

        self.data = data




if __name__ == '__main__':

    pass
