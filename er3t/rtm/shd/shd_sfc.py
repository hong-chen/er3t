import os
import sys
import copy
import struct
import warnings
import h5py
import numpy as np

import er3t.common


__all__ = ['shd_sfc_2d']



class shd_sfc_2d:

    """
    Input:
        atm_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        sfc_obj=: keyword argument, default=None, surface object, for example, sfc_obj = sfc_sat(fname='mod09.pk')
        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['Sfc_nxb']
                ['Sfc_nyb']
                ['Sfc_tmps2d']
                ['Sfc_jsfc2d']
                ['Sfc_psfc2d']

        self.gen_shd_2d_sfc_file: method to create binary file of 2d surface

        self.save_h5: method to save data into HDF5 file
    """


    ID = 'SHDOM 2D Surface'


    def __init__(self,\
                 atm_obj   = None, \
                 sfc_obj   = None, \
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
            msg = '\nError [shd_sfc_2d]: Please provide an <atm> object for <atm_obj>.'
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if sfc_obj is None:
            msg = '\nError [shd_sfc_2d]: Please provide an <sfc> object for <sfc_obj>.'
            raise OSError(msg)
        else:
            self.sfc = sfc_obj

        self.pre_shd_2d_sfc()

        if fname is None:
            fname = 'shdom-sfc_2d.txt'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_shd_2d_sfc_file(fname)
            self.nml['SFCFILE'] = {'data':fname}
        else:
            self.gen_shd_2d_sfc_file(fname)


    def pre_shd_2d_sfc(self):

        self.nml= {}

        self.nml['NX'] = copy.deepcopy(self.sfc.data['nx'])
        self.nml['NY'] = copy.deepcopy(self.sfc.data['ny'])
        self.nml['dx'] = copy.deepcopy(self.sfc.data['dx'])
        self.nml['dy'] = copy.deepcopy(self.sfc.data['dy'])

        self.Nx = self.nml['NX']['data']
        self.Ny = self.nml['NY']['data']
        self.dx = self.nml['dx']['data']
        self.dy = self.nml['dy']['data']

        if ('lambertian' in self.sfc.data['sfc']['name'].lower()):

            self.nml['header'] = dict(data='L', name='Header for SHDOM Surface File', units='N/A')
            self.sfc_data = self.sfc.data['sfc']['data']

        elif ('brdf-lsrt' in self.sfc.data['sfc']['name'].lower()):

            self.nml['header'] = dict(data='T', name='Header for SHDOM Surface File', units='N/A')
            self.sfc_data = self.sfc.data['sfc']['data']

        elif ('brdf-ocean' in self.sfc.data['sfc']['name'].lower()):

            self.nml['header'] = dict(data='O', name='Header for SHDOM Surface File', units='N/A')
            self.sfc_data = self.sfc.data['sfc']['data']

        else:

            msg = '\nError [shd_sfc_2d]: Cannot determine surface type - currently only supports Lambertian surface and LSRT BRDF surface (e.g., MCD43A1).'
            raise OSError(msg)


    def gen_shd_2d_sfc_file(
            self,
            fname,
            ):

        fname = os.path.abspath(fname)

        if not self.quiet:
            print('Message [shd_sfc_2d]: Creating 2D SFCFile <%s> for SHDOM...' % fname)

        with open(fname, 'w') as f:
            f.write('%s\n' % self.nml['header']['data'])
            f.write('%d %d %.4f %.4f\n' % (self.Nx, self.Ny, self.dx, self.dy))
            for ix in np.arange(self.Nx):
                for iy in np.arange(self.Ny):
                    string1 = '%d %d %.2f ' % ((ix+1), (iy+1), self.atm.lay['temperature']['data'][0])
                    string2 = ('%.6e ' * self.sfc_data[ix, iy, :].size) % tuple(self.sfc_data[ix, iy, :])
                    string3 = '\n'
                    f.write(string1+string2[:-1]+string3) # [:-1] is used to get rid of last empty space

        self.nml['SFCFILE'] = {'data':fname}

        # f = open(fname, 'wb')
        # f.write(struct.pack('<%df' % self.nml['Sfc_psfc2d']['data'].size, *self.nml['Sfc_psfc2d']['data'].flatten(order='F')))
        # f.close()

        if not self.quiet:
            print('Message [shd_sfc_2d]: File <%s> is created.' % fname)


if __name__ == '__main__':

    pass
