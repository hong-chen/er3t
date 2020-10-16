import sys
import copy
import h5py
import struct
import numpy as np

from er3t.util import cal_mol_ext, cal_sol_fac, get_lay_index



__all__ = ['mca_atm_1d', 'mca_atm_3d']




class mca_atm_1d:

    """
    Input:
        atm_obj: keyword argument, default=None, e.g. atm0 = atm_atmmod(levels=np.arange(21))
        abs_obj, keyword argument, default=None, e.g. abs0 = abs_16g(wavelength=600.0, atm_obj=atm0)

    Output:
        self.nml: Python dictionary, ig range from 0 to 15 (0, 1, ..., 15)
            [ig]['Atm_zgrd0']
            [ig]['Atm_nz']
            [ig]['Atm_ext1d']
            [ig]['Atm_omg1d']
            [ig]['Atm_apf1d']
            [ig]['Atm_abs1d']


    """


    ID = 'MCARaTS 1D Atmosphere'


    def __init__(self,
                 atm_obj = None, \
                 abs_obj = None, \
            ):

        if atm_obj is None:
            sys.exit('Error   [cld_mca]: please provide an \'atm\' object for \'atm_obj\'.')
        else:
            self.atm = atm_obj

        if abs_obj is None:
            sys.exit('Error   [cld_mca]: please provide an \'abs\' object for \'abs_obj\'.')
        else:
            self.abs = abs_obj

        self.Ng = self.abs.Ng

        self.pre_mca_1d_atm()


    def pre_mca_1d_atm(self):

        self.nml = {i:{} for i in range(self.Ng)}

        for ig in range(self.Ng):


            self.nml[ig]['Atm_zgrd0'] = {'data':self.atm.lev['altitude']['data']*1000.0, 'units':'m'  , 'name':'Layer boundaries'}

            nz = self.atm.lay['altitude']['data'].size
            self.nml[ig]['Atm_nz']    = {'data':nz, 'units':'N/A', 'name':'Number of z grid points'}

            if 'wvl' in self.abs.coef.keys():
                self.nml[ig]['Wld_nwl'] = {'data': 1, 'units':'N/A', 'name': 'Number of wavelengths'}

                # Extinction coefficient
                # Use Bodhaine formula to calculate Rayleigh Optical Deption
                atm_ext = cal_mol_ext(self.abs.coef['wvl']['data']*0.001, self.atm.lev['pressure']['data'][:-1], self.atm.lev['pressure']['data'][1:]) \
                         / (self.atm.lay['thickness']['data']*1000.0)
                self.nml[ig]['Atm_ext1d'] = {'data':atm_ext, 'units':'/m' , 'name':'Extinction coefficients'}

                # Absorption coefficient
                atm_abs = self.abs.coef['abso_coef']['data'][:, ig] / (self.atm.lay['thickness']['data']*1000.0)
                self.nml[ig]['Atm_abs1d'] = {'data':atm_abs, 'units':'/m' , 'name':'Absorption coefficients'}

            elif 'wvls' in self.abs.coef.keys():
                self.nml[ig]['Wld_nwl'] = {'data': self.abs.coef['wvls']['data'].size, 'units':'N/A', 'name': 'Number of wavelengths'}

                # Extinction coefficient
                # Use Bodhaine formula to calculate Rayleigh Optical Deption
                atm_ext = cal_mol_ext(self.abs.coef['wvls']['data'].mean()*0.001, self.atm.lev['pressure']['data'][:-1], self.atm.lev['pressure']['data'][1:]) \
                         / (self.atm.lay['thickness']['data']*1000.0)
                self.nml[ig]['Atm_ext1d'] = {'data':atm_ext, 'units':'/m' , 'name':'Extinction coefficients'}

                # Absorption coefficient
                atm_abs = self.abs.coef['abso_coef']['data'][:, :, ig] / (self.atm.lay['thickness']['data']*1000.0)
                for iwl in range(self.abs.coef['wvls']['data'].size):
                    self.nml[ig]['Atm_abs1d(1:, %d)' % (iwl+1)] = {'data':atm_abs[iwl, :], 'units':'/m' , 'name':'Absorption coefficients'}

            # Assign single scattering albedo (omega), phase function (p(theta))
            self.nml[ig]['Atm_omg1d'] = {'data':np.repeat(1.0, nz), 'units':'N/A', 'name':'Single scattering albedo'}
            self.nml[ig]['Atm_apf1d'] = {'data':np.repeat(-1 , nz), 'units':'N/A', 'name':'Phase function'}




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
                ['Atm_dy']
                ['Atm_dy']
                ['Atm_nz3']
                ['Atm_iz3l']
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
                 fname     = None,
                 overwrite = True, \
                 verbose   = False,\
                 quiet     = False \
                 ):

        self.overwrite = overwrite
        self.verbose   = verbose
        self.quiet     = quiet

        if not self.quiet:
            print('+ <mca_atm_3d>')

        if atm_obj is None:
            sys.exit('Error   [mca_atm_3d]: Please provide an \'atm\' object for \'atm_obj\'.')
        else:
            self.atm = atm_obj

        if cld_obj is None:
            sys.exit('Error   [mca_atm_3d]: Please provide an \'cld\' object for \'cld_obj\'.')
        else:
            self.cld = cld_obj

        if pha_obj is None:
            if self.verbose:
                print("Warning [mca_atm_3d]: No phase function set specified - ignore thermodynamic phase/effective radius with g=0.85 (Henyey-Greenstein).")
        else:
            if self.verbose:
                print("Warning [mca_atm_3d]: Phase functions were specified, but are not yet implemented.")
            self.pha = pha_obj

        # Go through cloud layers and check whether atm is compatible
        # e.g., whether the sizes of the Altitude array (z) and Thickness array (dz) are the same
        if self.cld.lay['altitude']['data'].size != self.cld.lay['thickness']['data'].size: # layer number
            sys.exit("Error   [mca_atm_3d]: Incorrect number of cloud layers (%d) vs layer thicknesses (%d)." % (self.cld.lay['altitude']['data'].size, self.cld.lay['thickness']['data'].size))

        self.pre_mca_3d_atm()

        if self.overwrite:
            if fname is None:
                fname = 'mca_atm_3d.bin'
            self.gen_mca_3d_atm_file(fname)

        if not self.quiet:
            print('-')


    def pre_mca_3d_atm(self):

        self.nml= {}

        lay_index = get_lay_index(self.cld.lay['altitude']['data'], self.atm.lay['altitude']['data'])

        nx   = self.cld.lay['nx']['data']
        ny   = self.cld.lay['ny']['data']
        nz3  = int(lay_index.size)
        iz3l = int(lay_index[0] + 1)

        if (iz3l+nz3) > self.atm.lay['altitude']['data'].size:
            sys.exit('Error   [mca_atm_3d]: Non-homogeneous layer top exceeds atmosphere top.')

        atm_abs = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # deviation from atmospheric absorption (-)
        atm_ext = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # extinction
        atm_omg = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # single scattering albedo (set to 1.0)
        atm_apf = np.zeros((nx, ny, nz3, 1), dtype=np.float32) # phase function

        for i in range(nz3):

            atm_ext[:, :, i, 0] = self.cld.lay['extinction']['data'][:, :, i]

        atm_omg[...] = 1.0
        atm_apf[...] = 0.85

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

        self.nml['Atm_abst3d'] = {'data':atm_abs, 'units':'/m' , 'name':'Absorption coefficients deviation'}
        self.nml['Atm_extp3d'] = {'data':atm_ext, 'units':'/m' , 'name':'Extinction coefficients'}
        self.nml['Atm_omgp3d'] = {'data':atm_omg, 'units':'N/A', 'name':'Single scattering Albedo'}
        self.nml['Atm_apfp3d'] = {'data':atm_apf, 'units':'N/A', 'name':'Phase function'}


    def gen_mca_3d_atm_file(self, fname, abs3d=False):


        self.nml['Atm_atm3dfile'] = {'data':fname}
        if abs3d:
            self.nml['Atm_atm3dabs'] = {'data':1}
        else:
            self.nml['Atm_atm3dabs'] = {'data':0}

        f = open(fname, 'wb')
        f.write(struct.pack('<%df' % self.nml['Atm_extp3d']['data'].size, *self.nml['Atm_extp3d']['data'].flatten(order='F')))
        f.write(struct.pack('<%df' % self.nml['Atm_omgp3d']['data'].size, *self.nml['Atm_omgp3d']['data'].flatten(order='F')))
        f.write(struct.pack('<%df' % self.nml['Atm_apfp3d']['data'].size, *self.nml['Atm_apfp3d']['data'].flatten(order='F')))
        if abs3d:
            f.write(struct.pack('<%df' % self.nml['Atm_abst3d']['data'].size, *self.nml['Atm_abst3d']['data'].flatten(order='F')))
        f.close()

        if not self.quiet:
            print('Message [mca_atm_3d]: File \'%s\' is created.' % fname)


    def save_h5(self, fname):

        self.nml['Atm_atm3dfile'] = {'data':fname}

        f = h5py.File(fname, 'w')
        for key in self.nml.keys():
            f[key] = self.nml[key]['data']
        f.close()

        if not self.quiet:
            print('Message [mca_atm_3d]: File \'%s\' is created.' % fname)




if __name__ == '__main__':

    pass
