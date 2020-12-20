import os
import sys
import copy
import pickle
import numpy as np
import h5py
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

    """
    Calculate Mie phase functions (water clouds) for a given wavelength and angles

    Input:
        wvl0: wavelength in nm, float value, default is 500.0
        angles: numpy array, angles to extract mie phase functions at
        *interpolate: boolen, whether to interpolate phase functions based on the wavelength, default is False
        reuse: boolen, whether to reuse the pre-existing phase functions stored at er3t/data/pha/mie, default is True
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

    fdir_pha_mie = '%s/mie/wc' % er3t.common.fdir_data_pha
    fname_coef = '%s/wc.sol.mie.cdf' % er3t.common.fdir_data_pha

    def __init__(self,
                 wvl0=500.0,
                 angles=np.concatenate((
                    np.arange(  0.0,   2.0, 0.01),
                    np.arange(  2.0,   5.0, 0.05),
                    np.arange(  5.0,  10.0, 0.1),
                    np.arange( 10.0,  15.0, 0.5),
                    np.arange( 15.0, 176.0, 1.0),
                    np.arange(176.0, 180.1, 0.25),
                 )),
                 interpolate=False,
                 reuse=True,
                 verbose=False):


        self.interpolate = interpolate
        self.reuse = reuse
        self.verbose = verbose

        print('+')
        self.get_data(wvl0, angles)
        print('-')


    def get_data(self,
            wvl0,
            angles,
            fdir='%s/pha/mie' % er3t.common.fdir_data,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = '%s/pha_mie_wc_%09.4fnm.pk' % (fdir, wvl0)

        if self.reuse:
            if os.path.exists(fname):
                with open(fname, 'rb') as f0:
                    data0 = pickle.load(f0)
                if np.abs(angles-data0['ang']['data']).sum() < 0.00000001:
                    print('Message [pha_mie_wc]: reuse phase function from "%s"' % fname)
                    self.data = copy.deepcopy(data0)
                else:
                    self.run(fname, wvl0, angles)
            else:
                self.run(fname, wvl0, angles)


    def run(self,
            fname,
            wvl0,
            angles,
            fdir_lrt = os.environ['LIBRADTRAN_PY']
            ):

        Na = angles.size
        wvl, ref, ssa, asy, pmom = read_pmom(self.fname_coef)
        Np, Nr, Nl = pmom.shape

        if not self.interpolate:
            index = np.argmin(np.abs(wvl-wvl0))
        else:
            sys.exit('Error   [pha_mie_wc]: has not implemented interpolation for phase function yet.')

        print('Message [pha_mie_wc]: calculating phase functions for %.2fnm ...' % wvl0)

        pha = np.zeros((Na, Nr), dtype=np.float64)

        for ir in range(Nr):

            pmom0 = pmom[:, ir, index]

            if pmom0[-1] > 0.1:
                if self.verbose:
                    print('Warning [pha_mie]: Ref=%.2f Legendre series did not converge.' % ref[ir])

            fname_tmp_inp = os.path.abspath('tmp_pmom_inp.txt')
            fname_tmp_out = os.path.abspath('tmp_pmom_out.txt')

            np.savetxt(fname_tmp_inp, pmom0/(2.0*np.arange(Np)+1.0))


            command = '%s/bin/phase -d -c -f %s > %s' % (fdir_lrt, fname_tmp_inp, fname_tmp_out)

            os.system(command)
            data0 = np.genfromtxt(fname_tmp_out)

            pha[:, ir] = np.interp(angles, data0[:, 0], data0[:, 1])

        os.system('rm -rf %s %s' % (fname_tmp_inp, fname_tmp_out))

        data = {
                'id'  : {'data':'Mie'     , 'name':'Mie'                , 'unit':'N/A'},
                'wvl0': {'data':wvl0      , 'name':'Given wavelength'   , 'unit':'nm'},
                'wvl' : {'data':wvl[index], 'name':'Actual wavelength'  , 'unit':'nm'},
                'ang' : {'data':angles    , 'name':'Angle'              , 'unit':'degree'},
                'pha' : {'data':pha       , 'name':'Phase function'     , 'unit':'N/A'},
                'ssa' : {'data':ssa[:, index], 'name':'Single scattering albedo', 'unit':'N/A'},
                'asy' : {'data':asy[:, index], 'name':'Asymmetry parameter'     , 'unit':'N/A'},
                'ref' : {'data':ref          , 'name':'Effective radius'        , 'unit':'mm'}
                }

        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        print('Message [pha_mie_wc]: phase function for %.2fnm has been store in %s.' % (wvl0, fname))

        self.data = data



if __name__ == '__main__':

    pass
