import os
import sys
import copy
import struct
import warnings
import h5py
import numpy as np



__all__ = ['mca_sfc_2d']



class mca_sfc_2d:

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

        self.gen_mca_2d_sfc_file: method to create binary file of 2d surface

        self.save_h5: method to save data into HDF5 file
    """


    ID = 'MCARaTS 2D Surface'


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
            msg = 'Error [mca_sfc_2d]: Please provide an \'atm\' object for <atm_obj>.'
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if sfc_obj is None:
            msg = 'Error [mca_sfc_2d]: Please provide an \'sfc\' object for <sfc_obj>.'
            raise OSError(msg)
        else:
            self.sfc = sfc_obj

        self.pre_mca_2d_sfc()

        if fname is None:
            fname = 'mca_sfc_2d.bin'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_mca_2d_sfc_file(fname)
            self.nml['Sfc_inpfile'] = {'data':fname}
        else:
            self.gen_mca_2d_sfc_file(fname)


    def pre_mca_2d_sfc(self):

        self.nml= {}

        self.nml['Sfc_nxb'] = copy.deepcopy(self.sfc.data['nx'])
        self.nml['Sfc_nyb'] = copy.deepcopy(self.sfc.data['ny'])

        self.nml['Sfc_tmps2d'] = dict(data=np.zeros_like(self.sfc.data['alb']['data']), name='Temperature anomalies'    , units='K')    # temperature anomaly
        self.nml['Sfc_jsfc2d'] = dict(data=np.ones_like(self.sfc.data['alb']['data']) , name='Surface distribution type', units='N/A')  # lambertian

        sfc_psfc         = np.zeros((self.sfc.Nx, self.sfc.Ny, 5), dtype=np.float64)
        sfc_psfc[..., 0] = self.sfc.data['alb']['data']
        self.nml['Sfc_psfc2d'] = dict(data=sfc_psfc, name='Surface distribution parameters', units='N/A')  # lambertian


    def gen_mca_2d_sfc_file(self, fname):

        fname = os.path.abspath(fname)

        self.nml['Sfc_inpfile'] = {'data':fname}

        f = open(fname, 'wb')
        f.write(struct.pack('<%df' % self.nml['Sfc_tmps2d']['data'].size, *self.nml['Sfc_tmps2d']['data'].flatten(order='F')))
        f.write(struct.pack('<%df' % self.nml['Sfc_jsfc2d']['data'].size, *self.nml['Sfc_jsfc2d']['data'].flatten(order='F')))
        f.write(struct.pack('<%df' % self.nml['Sfc_psfc2d']['data'].size, *self.nml['Sfc_psfc2d']['data'].flatten(order='F')))
        f.close()

        if not self.quiet:
            print('Message [mca_sfc_2d]: File <%s> is created.' % fname)


    def save_h5(self, fname):

        fname = os.path.abspath(fname)

        self.nml['Sfc_inpfile'] = {'data':fname}

        f = h5py.File(fname, 'w')
        for key in self.nml.keys():
            f[key] = self.nml[key]['data']
        f.close()

        if not self.quiet:
            print('Message [mca_sfc_2d]: File <%s> is created.' % fname)



if __name__ == '__main__':

    pass
