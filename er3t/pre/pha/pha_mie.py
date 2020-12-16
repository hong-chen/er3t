import h5py
import os
import numpy as np
from netCDF4 import Dataset

import er3t



__all__ = ['pha_mie_wc']



def read_pmom(fname):

    f = Dataset(fname, 'r')

    wvl  = f.variables['wavelen'][...]
    ref  = f.variables['reff'][...]
    ssa  = f.variables['ssa'][...].T
    pmom = f.variables['pmom'][...].T

    f.close()

    wvl = wvl * 1000.0
    pmom = pmom[:, 0, :, :]
    asy  = pmom[1, :, :]/3.0

    return wvl, ref, ssa, asy, pmom



class pha_mie_wc:

    fname_coef = '%s/pha/wc.sol.mie.cdf' % er3t.common.fdir_data

    def __init__(self,
                 wvl0=500.0,
                 verbose=False):

        fdir_lrt = os.environ['LIBRADTRAN_PY']

        angles = np.concatenate((
            np.arange(  0.0,   2.0, 0.01),
            np.arange(  2.0,   5.0, 0.05),
            np.arange(  5.0,  10.0, 0.1),
            np.arange( 10.0,  15.0, 0.5),
            np.arange( 15.0, 176.0, 1.0),
            np.arange(176.0, 180.1, 0.25),
            ))
        Na = angles.size

        wvl, ref, ssa, asy, pmom = read_pmom(self.fname_coef)
        Np, Nr, Nl = pmom.shape

        index = np.argmin(np.abs(wvl-wvl0))

        if verbose:
            print('Extracting wavelength=%.2fnm' % wvl[index])

        pha = np.zeros((Na, Nr), dtype=np.float64)

        for ir in range(Nr):

            pmom0 = pmom[:, ir, index]

            if pmom0[-1] > 0.1:
                if verbose:
                    print('Warning [pha_mie]: Ref=%.2f Legendre series did not converge.' % ref[ir])

            fname_tmp_inp = os.path.abspath('tmp_pmom_inp.txt')
            fname_tmp_out = os.path.abspath('tmp_pmom_out.txt')

            np.savetxt(fname_tmp_inp, pmom0/(2.0*np.arange(Np)+1.0))

            command = '%s/bin/phase -d -c -f %s > %s' % (fdir_lrt, fname_tmp_inp, fname_tmp_out)

            os.system(command)
            data0 = np.genfromtxt(fname_tmp_out)

            pha[:, ir] = np.interp(angles, data0[:, 0], data0[:, 1])

        self.data = {
                'wvl': {'data':wvl[index], 'name':'Wavelength'         , 'unit':'nm'},
                'ang': {'data':angles    , 'name':'Angle'              , 'unit':'degree'},
                'pha': {'data':pha       , 'name':'Phase function'     , 'unit':'N/A'},
                'ssa': {'data':ssa[:, index], 'name':'Single scattering albedo', 'unit':'N/A'},
                'asy': {'data':asy[:, index], 'name':'Asymmetry parameter'     , 'unit':'N/A'},
                'ref': {'data':ref          , 'name':'Effective radius'        , 'unit':'mm'}
                }



if __name__ == '__main__':

    pass
