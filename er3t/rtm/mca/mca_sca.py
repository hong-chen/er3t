import os
import sys
import copy
import struct
import warnings
import h5py
import numpy as np



__all__ = ['mca_sca']



class mca_sca:

    """
    Input:
        pha_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['Sca_npf']
                ['Sca_nskip']
                ['Sca_nanci']
                ['Sca_nangi']
               *['Sca_inpfile']

        *self.gen_mca_sca_file: method to create binary file of tabulated phase functions

        self.save_h5: method to save data into HDF5 file
    """


    ID = 'MCARaTS Scattering'


    def __init__(self,\
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

        if pha_obj is None:
            msg = 'Error [mca_sca]: Please provide an \'pha\' object for <pha_obj>.'
            raise OSError(msg)
        else:
            self.pha = pha_obj

        self.pre_mca_sca()

        if fname is None:
            fname = 'mca_sca.bin'

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_mca_sca_file(fname)
            self.nml['Sca_inpfile'] = {'data':fname}
        else:
            self.gen_mca_sca_file(fname)


    def pre_mca_sca(self, nskip=0, nanci=0):

        self.nml= {}

        self.nml['Sca_npf']   = dict(data=self.pha.data['pha']['data'].shape[1], name='Number of tabulated phase functions', units='N/A')
        self.nml['Sca_nskip'] = dict(data=nskip, name='Number of phase functions to be skipped', units='N/A')
        self.nml['Sca_nanci'] = dict(data=nanci, name='Number of ancillary data', units='N/A')
        self.nml['Sca_nangi'] = dict(data=self.pha.data['ang']['data'].size, name='Number of angles', units='N/A')


    def gen_mca_sca_file(self, fname):

        fname = os.path.abspath(fname)

        self.nml['Sca_inpfile'] = {'data':fname}

        f = open(fname, 'wb')
        f.write(struct.pack('<%df' % self.pha.data['ang']['data'].size, *self.pha.data['ang']['data'].flatten(order='F')))
        for i in range(self.nml['Sca_npf']['data']):
            f.write(struct.pack('<%df' % self.pha.data['pha']['data'][:, i].size, *self.pha.data['pha']['data'][:, i].flatten(order='F')))
        f.close()

        if not self.quiet:
            print('Message [mca_sca]: File <%s> is created.' % fname)


    def save_h5(self, fname):

        fname = os.path.abspath(fname)

        self.nml['Sca_inpfile'] = {'data':fname}

        f = h5py.File(fname, 'w')
        for key in self.nml.keys():
            f[key] = self.nml[key]['data']

        g = f.create_group('pha')
        for key in self.pha.data.keys():
            g[key] = self.pha.data[key]['data']
        f.close()

        if not self.quiet:
            print('Message [mca_sca]: File <%s> is created.' % fname)



if __name__ == '__main__':

    pass
