import er3t
import h5py
import os
import numpy as np
from netCDF4 import Dataset


__alll__ = ['pha_mie']


def read_pmom():

    fname = '/Users/hoch4240/Chen/soft/er3t/tests/test-pha/wc.sol.mie.cdf'
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



class pha_mie:

    def __init__(self, wvl0=500.0):

        fdir_lrt = '/Users/hoch4240/Chen/soft/libradtran/v2.0.1'

        wvl, ref, ssa, asy, pmom = read_pmom()

        angles = np.concatenate((
            np.arange(  0.0,   2.0, 0.01),
            np.arange(  2.0,   5.0, 0.05),
            np.arange(  5.0,  10.0, 0.1),
            np.arange( 10.0,  15.0, 0.5),
            np.arange( 15.0, 176.0, 1.0),
            np.arange(176.0, 180.1, 0.25),
            ))

        Na = angles.size

        Np, Nr, Nl = pmom.shape

        index = np.argmin(np.abs(wvl-wvl0))

        print('Extracting wavelength=%.2fnm' % wvl[index])

        pha = np.zeros((Na, Nr), dtype=np.float64)

        for ir in range(Nr):

            pmom0 = pmom[:, ir, index]

            if pmom0[-1] > 0.1:
                print('Warning [pha_mie]: Ref=%.2f Legendre series did not converge.' % ref[ir])

            fname_tmp_inp = os.path.abspath('tmp_pmom_inp.txt')
            fname_tmp_out = os.path.abspath('tmp_pmom_out.txt')

            np.savetxt(fname_tmp_inp, pmom0/(2.0*np.arange(Np)+1.0))

            command = '%s/bin/phase -d -c -f %s > %s' % (fdir_lrt, fname_tmp_inp, fname_tmp_out)

            os.system(command)
            data0 = np.genfromtxt(fname_tmp_out)

            pha[:, ir] = np.interp(angles, data0[:, 0], data0[:, 1])

        self.data = {
                'ang': {'data':angles    , 'name':'Angle'              , 'unit':'degree'},
                'pha': {'data':pha       , 'name':'Phase function'     , 'unit':'N/A'}
                }



if __name__ == '__main__':

    pass
