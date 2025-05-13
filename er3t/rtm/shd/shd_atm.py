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
                 alt_toa = 30.0, \
                 overwrite = True, \
                 force     = False,\
                 verbose   = False,\
                 quiet     = False \
            ):

        self.fname = fname
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
                self.gen_shd_ckd_file(fname, self.atm, self.abs, alt_toa=alt_toa)
            self.nml['CKDFILE'] = {'data':fname}
        else:
            self.gen_shd_ckd_file(fname, self.atm, self.abs, alt_toa=alt_toa)


    def pre_shd_1d_atm(self):

        self.nml= {}

        self.nml['NZ'] = {'data':self.atm.lay['altitude']['data'].size, 'name':'Nz', 'units':'N/A'}

        self.nml['WAVELEN'] = {'data':self.abs.wvl/1000.0, 'units':'micron', 'name':'Wavelength'}

        wvln_min = 1.0/self.abs.wvl_max_*1e7
        wvln_max = 1.0/self.abs.wvl_min_*1e7
        self.nml['WAVENO']  = {'data':'%.2f %.2f' % (wvln_min, wvln_max), 'units':'cm^-1', 'name':'Wave Number Range'}

        self.nml['GNDTEMP'] = {'data':self.atm.lay['temperature']['data'][0], 'units':'K', 'name':'Surface Temperature'}

        # calculate rayleight extinction
        self.atm_sca = er3t.util.cal_mol_ext_atm(self.abs.wvl/1000.0, self.atm) / (self.atm.lay['thickness']['data'])


    def gen_shd_ckd_file(
            self,
            fname,
            atm0,
            abs0,
            Nband=1,
            alt_toa=100.0,
            ):

        if not self.quiet:
            print('Message [shd_atm_1d]: Creating 1D CKDFile <%s> for SHDOM...' % fname)

        with open(fname, 'w') as f:

            f.write('! correlated k-distribution file for SHDOM\n')
            f.write('%d ! number of bands\n' % 1)
            f.write('! Band# | Wave#1 [%.2f nm] | Wave#2 [%.2f nm] | SolFlx | Ng | g1 | g2 | ...\n' % (abs0.wvl_max_, abs0.wvl_min_))

            # wave number cm^-1
            wvln_min = 1.0/abs0.wvl_max_*1e7
            wvln_max = 1.0/abs0.wvl_min_*1e7

            sol = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()

            Ng = abs0.coef['weight']['data'].size

            g = ' '.join(['%.12f' % value for value in abs0.coef['weight']['data']])

            for iband in range(Nband):
                f.write('%d %.2f %.2f %.12f %d %s\n' % (iband+1, wvln_min, wvln_max, sol, Ng, g))

            # calculating gas scatter (rayleigh) and gas absorption
            #╭────────────────────────────────────────────────────────────────────────────╮#
            # altitude
            #╭──────────────────────────────────────────────────────────────╮#
            # alt = atm0.lay['altitude']['data'][::-1]
            # thickness = atm0.lay['thickness']['data'][::-1]
            # zgrid = alt + thickness/2.0
            thickness = atm0.lay['thickness']['data'][::-1]
            zgrid = atm0.lev['altitude']['data'][1:][::-1]
            #╰──────────────────────────────────────────────────────────────╯#

            # gas scattering
            #╭──────────────────────────────────────────────────────────────╮#
            atm_sca = self.atm_sca[::-1]
            #╰──────────────────────────────────────────────────────────────╯#

            # gas absorption
            #╭──────────────────────────────────────────────────────────────╮#
            indices_sort = np.argsort(abs0.coef['weight']['data'])
            atm_abs = abs0.coef['abso_coef']['data'][::-1, indices_sort]
            for i in range(atm_abs.shape[0]):
                atm_abs[i, :] = atm_abs[i, :]/thickness[i]
            #╰──────────────────────────────────────────────────────────────╯#

            # add surface (z=0.0 km)
            #╭──────────────────────────────────────────────────────────────╮#
            if zgrid[-1] >= 1.0e-6:
                zgrid   = np.append(zgrid, 0.0)
                atm_sca = np.append(atm_sca, 0.0)
                atm_abs = np.concatenate((atm_abs, np.zeros((1, indices_sort.size), dtype=np.float32)))
                self.nml['NZ']['data'] += 1
            #╰──────────────────────────────────────────────────────────────╯#

            # add toa (z=alt_toa[100.0] km)
            #╭──────────────────────────────────────────────────────────────╮#
            if zgrid[0] <= (alt_toa-1.0e-6):
                zgrid   = np.append(alt_toa, zgrid)
                atm_sca = np.append(0.0, atm_sca)
                atm_abs = np.concatenate((np.zeros((1, indices_sort.size), dtype=np.float32), atm_abs))
                self.nml['NZ']['data'] += 1
            #╰──────────────────────────────────────────────────────────────╯#
            #╰────────────────────────────────────────────────────────────────────────────╯#

            f.write('%d\n' % zgrid.size)

            f.write('!\n')
            f.write('! Alt [km] | ScaCoef [km^-1]\n')

            for j in range(zgrid.size):
                f.write('%10.6f %15.6e\n' % (zgrid[j], atm_sca[j]))

            f.write('! iBand | iLay | AbsCoef [km^-1]\n')

            for iband in range(Nband):
                for j in range(zgrid.size):
                    atm_abs_s = ' '.join(['%15.6e' % atm_abs0 for atm_abs0 in atm_abs[j, :]])
                    f.write('%4d %4d %s\n' % (iband+1, j+1, atm_abs_s))

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
                 alt_toa   = 30.0, \
                 overwrite = True, \
                 force     = False,\
                 verbose   = False,\
                 quiet     = False,\
                 fname_atm_1d = None, \
                 ):

        self.fname = fname
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

        self.pre_shd_3d_atm(alt_toa=alt_toa)

        if fname is None:
            fname = 'shdom-prp.txt'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_shd_prp_file(fname, self.abs.wvl, self.atm, self.cld, fname_atm_1d=fname_atm_1d)
            self.nml['PROPFILE'] = {'data':fname}
        else:
            self.gen_shd_prp_file(fname, self.abs.wvl, self.atm, self.cld, fname_atm_1d=fname_atm_1d)


    def pre_shd_3d_atm(self, alt_toa=30.0):

        self.nml= {}

        self.nml['NX'] = copy.deepcopy(self.cld.lay['nx'])
        self.nml['NY'] = copy.deepcopy(self.cld.lay['ny'])

        self.nml['WAVELEN'] = {'data':self.abs.wvl/1000.0, 'units':'micron', 'name':'Wavelength'}

        wvln_min = 1.0/self.abs.wvl_max_*1e7
        wvln_max = 1.0/self.abs.wvl_min_*1e7
        self.nml['WAVENO']  = {'data':'%.2f %.2f' % (wvln_min, wvln_max), 'units':'cm^-1', 'name':'Wave Number Range'}

        self.nml['GNDTEMP'] = {'data':self.atm.lev['temperature']['data'][0], 'units':'K', 'name':'Surface Temperature'}

        # zgrid_atm = self.atm.lay['altitude']['data']+self.atm.lay['thickness']['data']/2.0
        zgrid_atm = self.atm.lev['altitude']['data'][1:]

        temp_atm  = self.atm.lay['temperature']['data']

        # zgrid_cld = self.cld.lay['altitude']['data']+self.cld.lay['thickness']['data']/2.0
        # zgrid_cld = self.cld.lev['altitude']['data'][1:]
        zgrid_cld = self.cld.lev['altitude']['data'][:-1]

        logic_z_extra = np.logical_not(np.array([np.any(np.abs(zgrid_atm[i]-zgrid_cld)<1.0e-6) for i in range(zgrid_atm.size)]))
        self.Nz_extra = logic_z_extra.sum()
        self.z_extra = '%s' % '\n'.join(['%.4e %.4e' % tuple(item) for item in zip(zgrid_atm[logic_z_extra], temp_atm[logic_z_extra])])
        if (zgrid_atm[0]>=1.0e-6) and (zgrid_cld[0]>=1.0e-6):
            self.z_extra = '%.4e %.4e\n%s' % (0.0, self.atm.lev['temperature']['data'][0], self.z_extra)
            self.Nz_extra += 1

        # thickness0 = self.atm.lay['thickness']['data'][-1]
        # if (zgrid_atm[-1]<=(alt_toa-1.0e-6-thickness0)) and (zgrid_cld[-1]<=(alt_toa-1.0e-6-thickness0)):
        #     self.z_extra = '%s\n%.4e %.4e' % (self.z_extra, zgrid_atm[-1]+thickness0, self.atm.lev['temperature']['data'][-1])
        #     self.Nz_extra += 1

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
            fname_atm_1d=None,
            ):

        fname_mie = er3t.rtm.shd.gen_mie_file(wavelength, wavelength)

        if fname_atm_1d is not None:
            fname_inp = er3t.rtm.shd.gen_ext_file(fname.replace('prp', 'ext'), cld0, fname_atm_1d=fname_atm_1d)
        else:
            fname_inp = er3t.rtm.shd.gen_lwc_file(fname.replace('prp', 'lwc'), cld0)

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
            fname_mie, fname_inp,\
            Npha_max, asy_tol, pha_tol,\
            wavelength, atm0.lev['pressure']['data'][0],\
            self.Nz_extra, self.z_extra,\
            pol_tag, fname,\
            prp_exe)

        if not self.quiet:
            print('Message [shd_atm_3d]: Creating 3D PROPFile <%s> for SHDOM...' % fname)

        os.system(command)

        if not self.quiet:
            print('Message [shd_atm_3d]: File <%s> is created.' % fname)

        self.nml['PROPFILE'] = {'data':fname}



if __name__ == '__main__':

    pass
