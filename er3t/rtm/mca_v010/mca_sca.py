import sys
import copy
import h5py
import struct
import numpy as np



__all__ = ['mca_sca']



class mca_sca:

    """
    Input:
        pha_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        abs_obj=: keyword argument, default=None, surface object, for example, sfc_obj = sfc_sat(fname='mod09.pk')
        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['Sca_nxb']
                ['Sca_nyb']
                ['Sca_tmps2d']
                ['Sca_jsfc2d']
                ['Sca_psfc2d']

        self.gen_mca_2d_sfc_file: method to create binary file of 2d surface

        self.save_h5: method to save data into HDF5 file
    """


    ID = 'MCARaTS Scattering'


    def __init__(self,\
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
            print('+ <mca_sca>')

        if pha_obj is None:
            sys.exit('Error   [mca_sca]: Please provide an \'pha\' object for \'pha_obj\'.')
        else:
            self.pha = pha_obj

        self.pre_mca_sca()

        if self.overwrite:
            if fname is None:
                fname = 'mca_sca.bin'
            self.gen_mca_sca_file(fname)

        if not self.quiet:
            print('-')


    def pre_mca_sca(self, nskip=0, nanci=0):

        self.nml= {}

        self.nml['Sca_npf']   = dict(data=self.pha.data['pha']['data'].shape[1], name='Number of tabulated phase functions', units='N/A')
        self.nml['Sca_nskip'] = dict(data=nskip, name='Number of phase functions to be skipped', units='N/A')
        self.nml['Sca_nanci'] = dict(data=nanci, name='Number of ancillary data', units='N/A')
        self.nml['Sca_nangi'] = dict(data=self.pha.data['angles']['data'].size, name='Number of angles', units='N/A')


    def gen_mca_sca_file(self, fname):

        self.nml['Sca_inpfile'] = {'data':fname}

        f = open(fname, 'wb')
        f.write(struct.pack('<%df' % self.pha.data['angles']['data'].size, *self.pha.data['angles']['data'].flatten(order='F')))
        for i in range(self.nml['Sca_npf']['data']):
            f.write(struct.pack('<%df' % self.pha.data['pha']['data'][:, i].size, *self.pha.data['pha']['data'][:, i].flatten(order='F')))
        f.close()

        if not self.quiet:
            print('Message [mca_sca]: File \'%s\' is created.' % fname)


    def save_h5(self, fname):

        self.nml['Sca_inpfile'] = {'data':fname}

        f = h5py.File(fname, 'w')
        for key in self.nml.keys():
            f[key] = self.nml[key]['data']
        f.close()

        if not self.quiet:
            print('Message [mca_sca]: File \'%s\' is created.' % fname)



if __name__ == '__main__':

    pass
