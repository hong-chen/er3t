import os
import sys
import copy
import struct
import warnings
import h5py
import numpy as np
from scipy import interpolate

import er3t

__all__ = ['shd_atm_1d', 'shd_atm_3d']



class shd_atm_1d:

    """
    Input:
        atm_obj: keyword argument, default=None, e.g. atm0 = atm_atmmod(levels=np.arange(21))
        abs_obj, keyword argument, default=None, e.g. abs0 = abs_16g(wavelength=600.0, atm_obj=atm0)

    Output:
        self.nml: Python dictionary, ig range from 0 to 15 (0, 1, ..., 15)
            ['NZ']
            ['WAVELEN']
            ['WAVENO']
            ['CKDFILE']
    """


    ID = 'SHDOM 1D Atmosphere'

    def __init__(self,
                 atm_obj = None, \
                 abs_obj = None, \
                 fname   = None, \
                 overwrite = True, \
                 force     = False,\
                 verbose   = False,\
                 quiet     = False \
            ):

        self.overwrite = overwrite
        self.verbose   = verbose
        self.quiet     = quiet

        if atm_obj is None:
            msg = 'Error [shd_atm_1d]: please provide an atm object for <atm_obj>.'
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if abs_obj is None:
            msg = 'Error [shd_atm_1d]: please provide an abs object for <abs_obj>.'
            raise OSError(msg)
        else:
            self.abs = abs_obj

        self.Ng = self.abs.Ng
        self.wvl_info = self.abs.wvl_info

        self.pre_shd_1d_atm()

        if fname is None:
            fname = 'shdom-ckd.txt'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_shd_ckd_file(fname, self.atm, self.abs)
            self.nml['CKDFILE'] = {'data':fname}
        else:
            self.gen_shd_ckd_file(fname, self.atm, self.abs)


    def pre_shd_1d_atm(self):

        self.nml= {}

        self.nml['NZ'] = {'data':self.atm.lay['altitude']['data'].size, 'name':'Nz', 'units':'N/A'}

        self.nml['WAVELEN'] = {'data':self.abs.wvl/1000.0, 'units':'micron', 'name':'Wavelength'}

        wvln_min = 1.0/self.abs.wvl_max_*1e7
        wvln_max = 1.0/self.abs.wvl_min_*1e7
        self.nml['WAVENO']  = {'data':'%.2f %.2f' % (wvln_min, wvln_max), 'units':'cm^-1', 'name':'Wave Number Range'}

        self.nml['GNDTEMP'] = {'data':self.atm.lay['temperature']['data'][0], 'units':'K', 'name':'Surface Temperature'}


    def gen_shd_ckd_file(
            self,
            fname,
            atm0,
            abs0,
            Nband=1
            ):

        if not self.quiet:
            print('Message [shd_atm_1d]: Creating 3D atm file <%s> for SHDOM...' % fname)

        with open(fname, 'w') as f:

            f.write('! correlated k-distribution file for SHDOM\n')
            f.write('%d ! number of bands\n' % 1)
            f.write('! Band# | Wave#1 | Wave#2 | SolFlx | Ng | g1 | g2 | ...\n')

            # wave number cm^-1
            wvln_min = 1.0/abs0.wvl_max_*1e7
            wvln_max = 1.0/abs0.wvl_min_*1e7

            sol = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()

            Ng = abs0.coef['weight']['data'].size

            g = ' '.join(['%.6f' % value for value in abs0.coef['weight']['data']])

            for iband in range(Nband):
                f.write('%d %.2f %.2f %.6f %d %s\n' % (iband+1, wvln_min, wvln_max, sol, Ng, g))

            f.write('%d\n' % atm0.lay['altitude']['data'].size)

            f.write('!\n')
            f.write('! Alt [km] | Pres [mb] | Temp [K]\n')

            alt = atm0.lay['altitude']['data'][::-1]
            # pres = atm0.lay['pressure']['data'][::-1]
            # temp = atm0.lay['temperature']['data'][::-1]

            indices_sort = np.argsort(abs0.coef['weight']['data'])
            kabs = abs0.coef['abso_coef']['data'][::-1, indices_sort]

            for j in range(alt.size):
                # f.write('%.6f %.2f %.2f\n' % (alt[j], pres[j], temp[j]))
                f.write('%.6f\n' % (alt[j]))

            f.write('! iBand | iLay | AbsCoef [km^-1]\n')

            for iband in range(Nband):
                for j in range(alt.size):
                    kabs_s = ' '.join(['%15.6e' % kabs0 for kabs0 in kabs[j, :]])
                    f.write('%d %d %s\n' % (iband+1, j+1, kabs_s))

        self.nml['CKDFILE'] = {'data':fname}

        if not self.quiet:
            print('Message [shd_atm_1d]: File <%s> is created.' % fname)



class shd_atm_3d:

    """
    Input:
        atm_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        cld_obj=: keyword argument, default=None, cloud object, for example, cld_obj = cld_les(fname='les.pk')
        abs_obj=: keyword argument, default=None, absorption object

        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['NX']
                ['NY']
                ['NZ']
                ['WAVELEN']
                ['WAVENO']
                ['GNDTEMP']
                ['PROPFILE']

        self.gen_shd_prp_file: method to create SHDOM property file of 3d atmosphere
    """


    ID = 'SHDOM 3D Atmosphere'

    def __init__(self,\
                 atm_obj   = None, \
                 abs_obj   = None, \
                 cld_obj   = None, \
                 fname     = None, \
                 overwrite = True, \
                 force     = False,\
                 verbose   = False,\
                 quiet     = False \
                 ):

        self.overwrite = overwrite
        self.verbose   = verbose
        self.quiet     = quiet

        if atm_obj is None:
            msg = 'Error [shd_atm_3d]: Please provide an atm object for <atm_obj>.'
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if abs_obj is None:
            msg = 'Error [shd_atm_3d]: Please provide an abs object for <abs_obj>.'
            raise OSError(msg)
        else:
            self.abs = abs_obj

        if cld_obj is None:
            msg = 'Error [shd_atm_3d]: Please provide an cld object for <cld_obj>.'
            raise OSError(msg)
        else:
            self.cld = cld_obj

        # Go through cloud layers and check whether atm is compatible
        # e.g., whether the sizes of the Altitude array (z) and Thickness array (dz) are the same
        if self.cld.lay['altitude']['data'].size != self.cld.lay['thickness']['data'].size: # layer number
            msg = 'Error [shd_atm_3d]: Incorrect number of cloud layers (%d) vs layer thicknesses (%d).' % (self.cld.lay['altitude']['data'].size, self.cld.lay['thickness']['data'].size)
            raise ValueError(msg)

        self.pre_shd_3d_atm()

        if fname is None:
            fname = 'shdom-prp.txt'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_shd_prp_file(fname, self.abs.wvl, self.atm, self.cld)
            self.nml['PROPFILE'] = {'data':fname}
        else:
            self.gen_shd_prp_file(fname, self.abs.wvl, self.atm, self.cld)


    def pre_shd_3d_atm(self):

        self.nml= {}

        self.nml['NX'] = copy.deepcopy(self.cld.lay['nx'])
        self.nml['NY'] = copy.deepcopy(self.cld.lay['ny'])

        self.nml['WAVELEN'] = {'data':self.abs.wvl/1000.0, 'units':'micron', 'name':'Wavelength'}

        wvln_min = 1.0/self.abs.wvl_max_*1e7
        wvln_max = 1.0/self.abs.wvl_min_*1e7
        self.nml['WAVENO']  = {'data':'%.2f %.2f' % (wvln_min, wvln_max), 'units':'cm^-1', 'name':'Wave Number Range'}

        self.nml['GNDTEMP'] = {'data':self.atm.lay['temperature']['data'][0], 'units':'K', 'name':'Surface Temperature'}

        logic_z_extra = np.logical_not(np.array([np.any(np.abs(self.atm.lay['altitude']['data'][i]-self.cld.lay['altitude']['data'])<1.0e-6) for i in range(self.atm.lay['altitude']['data'].size)]))
        self.Nz_extra = logic_z_extra.sum()
        self.z_extra = '%s' % '\n'.join(['%.4e %.4e' % tuple(item) for item in zip(self.atm.lay['altitude']['data'][logic_z_extra], self.atm.lay['temperature']['data'][logic_z_extra])])

        self.nml['NZ'] = {'data':self.Nz_extra+self.cld.lay['altitude']['data'].size, 'name':'Nz', 'units':'N/A'}

    def gen_shd_prp_file(
            self,
            fname,
            wavelength,
            atm0,
            cld0,
            Npha_max=1000,
            asy_tol=1.0e-2,
            pha_tol=1.0e-1,
            pol_tag='U',
            put_exe='put',
            prp_exe='propgen',
            ):

        fname_mie = er3t.rtm.shd.gen_mie_file(wavelength, wavelength)

        fname_ext = er3t.rtm.shd.gen_ext_file(fname.replace('prp', 'ext'), cld0)

        if len(self.z_extra) > 1000:
            msg = 'Error [shd_atm_3d]: <z_extra> is greater than 1000-character-limit.'
            raise OSError(msg)

        wavelength /= 1000.0

        command = '%s "1"\
 "%s" "1" "F" "%s"\
 "%d" "%.4e" "%.4e"\
 "%15.8e" "%.4f"\
 "%d" "%s"\
 "%s" "%s"\
 | %s' %\
            (put_exe,\
            fname_mie, fname_ext,\
            Npha_max, asy_tol, pha_tol,\
            wavelength, atm0.lev['pressure']['data'][0],\
            self.Nz_extra, self.z_extra,\
            pol_tag, fname,\
            prp_exe)

        if not self.quiet:
            print('Message [shd_atm_3d]: Creating 3D atm file <%s> for SHDOM...' % fname)

        os.system(command)

        if not self.quiet:
            print('Message [shd_atm_3d]: File <%s> is created.' % fname)

        self.nml['PROPFILE'] = {'data':fname}



if __name__ == '__main__':

    pass
