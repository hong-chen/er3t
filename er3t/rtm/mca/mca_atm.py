import os
import sys
import copy
import struct
import warnings
import h5py
import numpy as np
from scipy import interpolate

from er3t.util import cal_mol_ext, cal_sol_fac, get_lay_index, cal_mol_ext_atm



__all__ = ['mca_atm_1d', 'mca_atm_3d']



class mca_atm_1d:

    """
    Input:
        atm_obj: keyword argument, default=None, e.g. atm0 = atm_atmmod(levels=np.arange(21))
        abs_obj, keyword argument, default=None, e.g. abs0 = abs_16g(wavelength=600.0, atm_obj=atm0)

    Output:
        self.nml: Python dictionary, ig range from 0 to 15 (0, 1, ..., 15)
            [ig]['Atm_zgrd0']
            [ig]['Atm_wkd0']
            [ig]['Atm_mtprof']
            [ig]['Atm_tmp1d']
            [ig]['Atm_nkd']
            [ig]['Atm_nz']
            [ig]['Atm_ext1d']
            [ig]['Atm_omg1d']
            [ig]['Atm_apf1d']
            [ig]['Atm_abs1d']


    """


    ID = 'MCARaTS 1D Atmosphere'


    def __init__(self,
                 atm_obj = None, \
                 abs_obj = None
            ):

        if atm_obj is None:
            msg = 'Error [mca_atm_1d]: please provide an \'atm\' object for <atm_obj>.'
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if abs_obj is None:
            msg = 'Error [mca_atm_1d]: please provide an \'abs\' object for <abs_obj>.'
            raise OSError(msg)
        else:
            self.abs = abs_obj

        self.Ng = self.abs.Ng
        self.wvl_info = self.abs.wvl_info

        self.pre_mca_1d_atm()


    def pre_mca_1d_atm(self):

        self.nml = {i:{} for i in range(self.Ng)}

        for ig in range(self.Ng):

            self.nml[ig]['Atm_zgrd0'] = {'data':self.atm.lev['altitude']['data']*1000.0, 'units':'m'  , 'name':'Layer boundaries'}
            self.nml[ig]['Atm_wkd0']  = {'data':1.0                                    , 'units':'N/A', 'name':'Weight coefficients'}
            self.nml[ig]['Atm_mtprof']= {'data':0                                      , 'units':'N/A', 'name':'Temperature profile flag'}
            self.nml[ig]['Atm_tmp1d'] = {'data':self.atm.lay['temperature']['data']    , 'units':'K'  , 'name':'Temperature profile'}
            self.nml[ig]['Atm_nkd']   = {'data':1                                      , 'units':'N/A', 'name':'Number of K-distribution'}
            self.nml[ig]['Atm_np1d']  = {'data':1                                      , 'units':'N/A', 'name':'Number of 1D atmospheric constituents'}

            nz = self.atm.lay['altitude']['data'].size

            self.nml[ig]['Atm_nz']    = {'data':nz                                     , 'units':'N/A', 'name':'Number of z grid points'}

            # Use Bodhaine formula to calculate Rayleigh Optical Depth
            # atm_sca = cal_mol_ext(self.abs.wvl*0.001, self.atm.lev['pressure']['data'][:-1], self.atm.lev['pressure']['data'][1:]) \
            #          / (self.atm.lay['thickness']['data']*1000.0)
            atm_sca = cal_mol_ext_atm(self.abs.wvl*0.001, self.atm) / (self.atm.lay['thickness']['data']*1000.0)

            # Absorption coefficient
            atm_abs = self.abs.coef['abso_coef']['data'][:, ig] / (self.atm.lay['thickness']['data']*1000.0)
            self.nml[ig]['Atm_abs1d(1:, 1)'] = {'data':atm_abs, 'units':'/m' , 'name':'Absorption coefficients'}

            atm_ext = atm_sca
            self.nml[ig]['Atm_ext1d(1:, 1)'] = {'data':atm_ext, 'units':'/m' , 'name':'Extinction coefficients'}

            # Single Scattering Albedo
            atm_omg = np.repeat(1.0, nz)
            self.nml[ig]['Atm_omg1d(1:, 1)'] = {'data':atm_omg, 'units':'N/A', 'name':'Single scattering albedo'}

            # Asymmetry parameter
            atm_apf = np.repeat(-1 , nz)
            self.nml[ig]['Atm_apf1d(1:, 1)'] = {'data':atm_apf, 'units':'N/A', 'name':'Phase function'}


    def add_mca_1d_atm(self, ext1d=None, omg1d=None, apf1d=None, z_bottom=None, z_top=None):

        if (ext1d is None) or (omg1d is None) or (apf1d is None):
            msg = 'Error [mca_atm_1d]: Please provide values of <ext1d>, <omg1d>, and <apf1d>.'
            raise OSError(msg)

        for ig in range(self.Ng):

            nz = self.atm.lay['altitude']['data'].size

            atm_ext = np.zeros(nz)
            atm_ext[:] = ext1d

            atm_omg = np.zeros(nz)
            atm_omg[:] = omg1d

            atm_apf = np.zeros(nz)
            atm_apf[:] = apf1d

            if z_bottom is not None:
                atm_ext[self.atm.lay['altitude']['data']<z_bottom] = 0.0
                atm_omg[self.atm.lay['altitude']['data']<z_bottom] = 0.0
                atm_apf[self.atm.lay['altitude']['data']<z_bottom] = 0.0
            if z_top is not None:
                atm_ext[self.atm.lay['altitude']['data']>z_top] = 0.0
                atm_omg[self.atm.lay['altitude']['data']>z_top] = 0.0
                atm_apf[self.atm.lay['altitude']['data']>z_top] = 0.0

            N = self.nml[ig]['Atm_np1d']['data'] + 1

            self.nml[ig]['Atm_ext1d(1:, %d)' % N] = {'data':atm_ext, 'units':'/m' , 'name':'Extinction coefficients'}
            self.nml[ig]['Atm_omg1d(1:, %d)' % N] = {'data':atm_omg, 'units':'N/A', 'name':'Single scattering albedo'}
            self.nml[ig]['Atm_apf1d(1:, %d)' % N] = {'data':atm_apf, 'units':'N/A', 'name':'Phase function'}

            self.nml[ig]['Atm_np1d']['data'] += 1




class mca_atm_3d:

    """
    Input:
        atm_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        cld_obj=: keyword argument, default=None, cloud object, for example, cld_obj = cld_les(fname='les.pk')
        pha_obj=: keyword argument, default=None, phase function object (under development)

        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['Atm_nx']
                ['Atm_ny']
                ['Atm_dx']
                ['Atm_dy']
                ['Atm_nz3']
                ['Atm_iz3l']
                ['Atm_tmpa3d']
                ['Atm_abst3d']
                ['Atm_extp3d']
                ['Atm_omgp3d']
                ['Atm_apfp3d']

        self.gen_mca_3d_atm_file: method to create binary file of 3d atmosphere

        self.save_h5: method to save data into HDF5 file

    """


    ID = 'MCARaTS 3D Atmosphere'


    def __init__(self,\
                 atm_obj   = None, \
                 cld_obj   = None, \
                 pha_obj   = None, \
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
            msg = 'Error [mca_atm_3d]: Please provide an \'atm\' object for <atm_obj>.'
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if cld_obj is None:
            msg = 'Error [mca_atm_3d]: Please provide an \'cld\' object for <cld_obj>.'
            raise OSError(msg)
        else:
            self.cld = cld_obj

        if pha_obj is None:
            if self.verbose:
                msg = 'Warning [mca_atm_3d]: No phase function set specified - ignore thermodynamic phase/effective radius with g=0.85 (Henyey-Greenstein).'
                warnings.warn(msg)
        self.pha = pha_obj

        # Go through cloud layers and check whether atm is compatible
        # e.g., whether the sizes of the Altitude array (z) and Thickness array (dz) are the same
        if self.cld.lay['altitude']['data'].size != self.cld.lay['thickness']['data'].size: # layer number
            msg = 'Error [mca_atm_3d]: Incorrect number of cloud layers (%d) vs layer thicknesses (%d).' % (self.cld.lay['altitude']['data'].size, self.cld.lay['thickness']['data'].size)
            raise ValueError(msg)

        self.pre_mca_3d_atm()

        if fname is None:
            fname = 'mca_atm_3d.bin'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_mca_3d_atm_file(fname)
            self.nml['Atm_inpfile'] = {'data':fname}
        else:
            self.gen_mca_3d_atm_file(fname)


    def pre_mca_3d_atm(self):

        self.nml= {}

        lay_index = get_lay_index(self.cld.lay['altitude']['data'], self.atm.lay['altitude']['data'])


        nx   = self.cld.lay['nx']['data']
        ny   = self.cld.lay['ny']['data']
        nz3  = int(lay_index.size)
        iz3l = int(lay_index[0] + 1)

        if (iz3l+nz3) > self.atm.lay['altitude']['data'].size:
            msg = 'Error [mca_atm_3d]: Non-homogeneous layer top exceeds atmosphere top.'
            raise ValueError(msg)

        atm_tmp = np.zeros((nx, ny, nz3)   , dtype=np.float32) # deviation from layer temperature (-)
        atm_abs = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # deviation from atmospheric absorption (-)
        atm_ext = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # extinction
        atm_omg = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # single scattering albedo (set to 1.0)
        atm_apf = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # phase function

        for i in range(nz3):

            atm_tmp[:, :, i]    = self.cld.lay['temperature']['data'][:, :, i] - self.atm.lay['temperature']['data'][lay_index[i]]
            atm_ext[:, :, i, 0] = self.cld.lay['extinction']['data'][:, :, i]


        if self.pha is None:
            atm_omg[...] = 1.0
            atm_apf[...] = 0.85

        else:
            # Rayleigh scattering by default, will later assign different values for clouds
            atm_omg[...] = 1.0
            atm_apf[...] = -1.0

            if isinstance(self.cld.lay['extinction']['data'], np.ma.MaskedArray):
                logic_cld = (self.cld.lay['extinction']['data'].data > 0.0)
            elif isinstance(self.cld.lay['extinction']['data'], np.ndarray):
                logic_cld = (self.cld.lay['extinction']['data'] > 0.0)

            if self.pha.data['id']['data'].lower() == 'hg':

                index = np.argmin(np.abs(self.pha.data['asy']['data']-0.85)) + 1.0
                atm_apf[logic_cld, 0] = index

            elif self.pha.data['id']['data'].lower() == 'mie':

                if isinstance(self.cld.lay['cer']['data'], np.ma.MaskedArray):
                    cer = self.cld.lay['cer']['data'].data
                elif isinstance(self.cld.lay['cer']['data'], np.ndarray):
                    cer = self.cld.lay['cer']['data']

                ref = self.pha.data['ref']['data']
                ssa = self.pha.data['ssa']['data']
                asy = self.pha.data['asy']['data']
                ind = np.arange(float(ref.size)) + 1.0

                f_interp_ssa = interpolate.interp1d(ref, ssa, bounds_error=False, fill_value='extrapolate')
                f_interp_ind = interpolate.interp1d(ref, ind, bounds_error=False, fill_value='extrapolate')
                f_interp_asy = interpolate.interp1d(ref, asy, bounds_error=False, fill_value='extrapolate')

                atm_omg[logic_cld, 0] = f_interp_ssa(cer[logic_cld])
                atm_apf[logic_cld, 0] = f_interp_asy(cer[logic_cld])

                # if preffered to use the previous version's index interpolation setting
                # for the phase function, uncomment the following lines

                # atm_apf[logic_cld, 0] = f_interp_ind(cer[logic_cld])
                # set left-outbound to left-most value
                #╭────────────────────────────────────────────────────────────────────────────╮#
                # logic0 = (atm_apf>0.0) & (atm_apf<ind[0])
                # atm_omg[logic0] = ssa[0]
                # atm_apf[logic0] = ind[0]
                #╰────────────────────────────────────────────────────────────────────────────╯#

                # set right-outbound to right-most value
                #╭────────────────────────────────────────────────────────────────────────────╮#
                # logic1 = (atm_apf>ind[-1])
                # atm_omg[logic1] = ssa[-1]
                # atm_apf[logic1] = ind[-1]
                #╰────────────────────────────────────────────────────────────────────────────╯#

        self.nml['Atm_nx']     = copy.deepcopy(self.cld.lay['nx'])
        self.nml['Atm_ny']     = copy.deepcopy(self.cld.lay['ny'])

        self.nml['Atm_dx']     = copy.deepcopy(self.cld.lay['dx'])
        self.nml['Atm_dx']['data']  *= 1000.0
        self.nml['Atm_dx']['units']  = 'm'

        self.nml['Atm_dy']     = copy.deepcopy(self.cld.lay['dy'])
        self.nml['Atm_dy']['data']  *= 1000.0
        self.nml['Atm_dy']['units']  = 'm'

        self.nml['Atm_nz3']    = {'data':nz3   , 'unit':'N/A', 'name':'number of 3D layer'}
        self.nml['Atm_iz3l']   = {'data':iz3l+1, 'unit':'N/A', 'name':'layer index of first 3D layer'}

        self.nml['Atm_tmpa3d'] = {'data':atm_tmp, 'units':'K'  , 'name':'Temperature deviation'}
        self.nml['Atm_abst3d'] = {'data':atm_abs, 'units':'/m' , 'name':'Absorption coefficients deviation'}
        self.nml['Atm_extp3d'] = {'data':atm_ext, 'units':'/m' , 'name':'Extinction coefficients'}
        self.nml['Atm_omgp3d'] = {'data':atm_omg, 'units':'N/A', 'name':'Single scattering Albedo'}
        self.nml['Atm_apfp3d'] = {'data':atm_apf, 'units':'N/A', 'name':'Phase function'}
        self.nml['Atm_np3d']   = {'data':1      , 'units':'N/A', 'name': 'Number of 3D atmospheric constituents'}


    def add_mca_3d_atm(self, ext3d=None, omg3d=None, apf3d=None):

        if (ext3d is None) or (omg3d is None) or (apf3d is None):
            msg = 'Error [mca_atm_3d]: Please provide an <ext3d>, <omg3d>, and <apf3d>.'
            raise OSError(msg)
        else:

            if isinstance(ext3d, np.ndarray):
                if ext3d.ndim != 3:
                    msg = 'Error [mca_atm_3d]: <ext3d> should be in the dimension of (nx, ny, nz).'
                    raise ValueError(msg)

            if isinstance(omg3d, np.ndarray):
                if omg3d.ndim != 3:
                    msg = 'Error [mca_atm_3d]: <omg3d> should be in the dimension of (nx, ny, nz).'
                    raise ValueError(msg)

            if isinstance(apf3d, np.ndarray):
                if apf3d.ndim != 3:
                    msg = 'Error [mca_atm_3d]: <apf3d> should be in the dimension of (nx, ny, nz).'
                    raise ValueError(msg)


        atm_ext = np.concatenate((self.nml['Atm_extp3d']['data'], ext3d[..., np.newaxis]), axis=-1)
        atm_omg = np.concatenate((self.nml['Atm_omgp3d']['data'], omg3d[..., np.newaxis]), axis=-1)
        atm_apf = np.concatenate((self.nml['Atm_apfp3d']['data'], apf3d[..., np.newaxis]), axis=-1)

        self.nml['Atm_extp3d']['data'] = atm_ext
        self.nml['Atm_omgp3d']['data'] = atm_omg
        self.nml['Atm_apfp3d']['data'] = atm_apf
        self.nml['Atm_np3d']['data'] += 1


    def gen_mca_3d_atm_file(self, fname):

        if not self.quiet:
            print('Message [mca_atm_3d]: Creating 3D atm file <%s> for MCARaTS ...' % fname)

        fname = os.path.abspath(fname)

        self.nml['Atm_inpfile'] = {'data':fname}

        f = open(fname, 'wb')
        f.write(struct.pack('<%df' % self.nml['Atm_tmpa3d']['data'].size, *self.nml['Atm_tmpa3d']['data'].flatten(order='F')))
        f.write(struct.pack('<%df' % self.nml['Atm_abst3d']['data'].size, *self.nml['Atm_abst3d']['data'].flatten(order='F')))
        for i in range(self.nml['Atm_np3d']['data']):
            f.write(struct.pack('<%df' % self.nml['Atm_extp3d']['data'][..., i].size, *self.nml['Atm_extp3d']['data'][..., i][..., np.newaxis].flatten(order='F')))
            f.write(struct.pack('<%df' % self.nml['Atm_omgp3d']['data'][..., i].size, *self.nml['Atm_omgp3d']['data'][..., i][..., np.newaxis].flatten(order='F')))
            f.write(struct.pack('<%df' % self.nml['Atm_apfp3d']['data'][..., i].size, *self.nml['Atm_apfp3d']['data'][..., i][..., np.newaxis].flatten(order='F')))
        f.close()

        if not self.quiet:
            print('Message [mca_atm_3d]: File <%s> is created.' % fname)


    def save_h5(self, fname):

        fname = os.path.abspath(fname)

        self.nml['Atm_inpfile'] = {'data':fname}

        f = h5py.File(fname, 'w')
        for key in self.nml.keys():
            f[key] = self.nml[key]['data']
        f.close()

        if not self.quiet:
            print('Message [mca_atm_3d]: File <%s> is created.' % fname)




if __name__ == '__main__':

    pass
