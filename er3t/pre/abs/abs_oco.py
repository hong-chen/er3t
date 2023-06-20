import os
import sys
import pickle
import multiprocessing as mp
import h5py
from scipy.io import readsav
from scipy.interpolate import interp2d
import copy
import numpy as np

import er3t




__all__ = ['abs_oco', 'abs_oco_idl', 'abs_oco_h5']




class abs_oco:

    """
    1. Process the atmospheric gases profile (self.lay)
    2. Calculate the factors (self.fac)
    3. From the input wavelength, prepare variables for getting absorption coefficient (self.abso)
    4: Get absorption coefficient, SSFR slit function, and solar constant from Correlated-K database (self.coef)

    Input:
        wavelength: wavelength in nm
        fname     : file path for the correlated-k coefficients (in pickle format)
        atm_obj   : atmosphere object, e.g., atm_obj = atm_atmmod(levels=np.arange(21))

    Note:
        If wavelength is provided but fname does not exisit:
            calculate correlated-k coefficients and save data into fname

        If wavelength is provided but fname is None:
            calculate correlated-k coefficients without saving the data

        if wavelength is not provided but fname is provided (also existed):
            read out the data from fname

        if wavelength and fname are neither provided:
            exit with error message

    Output:
        self.coef['wavelength']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """


    def __init__(self, \
                 fname       = None,  \
                 fname_idl   = None,  \
                 atm_obj     = None,  \
                 overwrite   = False, \
                 verbose     = False):

        self.verbose   = verbose
        self.fname_idl = fname_idl

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((fname_idl is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((fname_idl is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(atm_obj)
            self.dump(fname)

        elif ((fname_idl is not None) and (fname is None)):

            self.run(atm_obj)

        else:

            sys.exit('Error   [abs_oco_idl]: Please provide \'fname_idl\' to proceed.')


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'coef'):
                if self.verbose:
                    print('Message [abs_oco_idl]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.wvl   = obj.wvl
                self.nwl   = obj.nwl
                self.coef  = obj.coef
                self.Ng    = obj.Ng
            else:
                sys.exit('Error   [abs_oco_idl]: \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, atm_obj):

        if self.fname_idl is None:
            sys.exit('Error   [abs_oco]: Please provide \'fname_idl\' (IDL gas absorption file) to proceed.')

        if not os.path.exists(self.fname_idl):
            sys.exit('Error   [abs_oco]: Failed to locate \'fname_idl=%s\'.' % self.fname_idl)

        # self.coef
        #     self.coef['wvl']
        #     self.coef['abso_coef']
        #     self.coef['slit_func']
        #     self.coef['solar']
        self.get_coefficient(atm_obj)


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [abs_oco_idl]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def get_coefficient(self, atm_obj, wvl_threshold=1.0):

        f = readsav(self.fname_idl)

        wvls      = f.lamx*1000.0
        abso_coef = f.absgl
        Ng        = abso_coef.shape[0]

        slit_func0     = f.absgy
        slit_func      = np.empty(abso_coef.shape, dtype=slit_func0.dtype)
        slit_func[...] = slit_func0[..., None]

        solar     = f.solx

        weight    = np.zeros_like(solar)
        for i in range(wvls.size):
            weight[:, i] = slit_func0[:, i] / (slit_func0[:, i].sum())

        self.Ng   = Ng
        self.nwl  = wvl.size
        self.coef = {
                'wvls'       : {'name':'Wavelengths (Nwl)'                    , 'data':wvls, 'units':'nm'},
                'abso_coef'  : {'name':'Absorption Coefficient (Nwl, Nz, Ng)' , 'data':np.swapaxes(np.transpose(abso_coef), 0, 1)},
                'slit_func'  : {'name':'Slit Function (Nwl, Nz, Ng)'          , 'data':np.swapaxes(np.transpose(slit_func), 0, 1)},
                'solar'      : {'name':'Solar Factor (Nwl, Ng)'               , 'data':np.transpose(solar)},
                'weight'     : {'name':'Weight (Nwl, Ng)'                     , 'data':np.transpose(weight)}
                     }




class abs_oco_idl:

    """
    This module is based on the database developed by Odele Coddington (Odele.Coddington@lasp.colorado.edu).

    1. Process the atmospheric gases profile (self.lay)
    2. Calculate the factors (self.fac)
    3. From the input wavelength, prepare variables for getting absorption coefficient (self.abso)
    4: Get absorption coefficient, SSFR slit function, and solar constant from Correlated-K database (self.coef)

    Input:
        wavelength: wavelength in nm
        fname     : file path for the correlated-k coefficients (in pickle format)
        atm_obj   : atmosphere object, e.g., atm_obj = atm_atmmod(levels=np.arange(21))

    Note:
        If wavelength is provided but fname does not exisit:
            calculate correlated-k coefficients and save data into fname

        If wavelength is provided but fname is None:
            calculate correlated-k coefficients without saving the data

        if wavelength is not provided but fname is provided (also existed):
            read out the data from fname

        if wavelength and fname are neither provided:
            exit with error message

    Output:
        self.coef['wvl']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """




    def __init__(self, \
                 wavelength  = None,  \
                 fname       = None,  \
                 fname_idl   = None,  \
                 atm_obj     = None,  \
                 overwrite   = False, \
                 verbose     = False):

        if fname_idl is None:
            self.fname_idl = '%s/abs/abs_oco_11.sav' % er3t.common.fdir_data
        else:
            self.fname_idl = fname_idl

        self.verbose   = verbose
        self.wvl       = wavelength
        self.wvl_info  = '%.4f nm (applied OCO-2 slit)' % wavelength

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((wavelength is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((wavelength is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(atm_obj)
            self.dump(fname)

        elif ((wavelength is not None) and (fname is None)):

            self.run(atm_obj)

        else:

            sys.exit('Error   [abs_oco_idl]: Please provide \'wavelength\' to proceed.')


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'coef'):
                if self.verbose:
                    print('Message [abs_oco_idl]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.wvl   = obj.wvl
                self.nwl   = obj.nwl
                self.coef  = obj.coef
                self.Ng    = obj.Ng
                self.wvl_info   = obj.wvl_info
            else:
                sys.exit('Error   [abs_oco_idl]: \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, atm_obj):

        if not os.path.exists(self.fname_idl):
            sys.exit('Error   [abs_oco_idl]: Missing IDL database.')

        # self.coef
        #     self.coef['wvl']
        #     self.coef['abso_coef']
        #     self.coef['slit_func']
        #     self.coef['solar']
        self.get_coefficient(atm_obj)


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [abs_oco_idl]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def get_coefficient(self, atm_obj, wvl_threshold=1.0):

        f = readsav(self.fname_idl)

        wvl_center_oco = f.lamx*1000.0

        index_wvl = np.argmin(np.abs(wvl_center_oco-self.wvl))
        if abs(wvl_center_oco[index_wvl]-self.wvl) >= wvl_threshold:
            sys.exit('Error [abs_oco_idl]: Cannot pick a close wavelength for %.2fnm from \'%s\'.' % (self.wvl, self.fname_idl))
        else:
            if self.verbose:
                print('Message [abs_oco_idl]: Picked wvl=%.2f from \'%s\' for input wavelength %.2fnm.' % (wvl_center_oco[index_wvl], self.fname_idl, self.wvl))
            self.wvl = wvl_center_oco[index_wvl]

        Ng        = f.absgn[index_wvl]
        wvls      = f.absgx[:Ng, index_wvl] * 1000.0

        abso_coef0 = f.absgl[:Ng, index_wvl, :]
        alt0       = (f.atm_zgrd[1:]+f.atm_zgrd[:-1])/2000.0
        alt        = atm_obj.lay['altitude']['data']
        x_         = np.arange(abso_coef0.shape[0])

        f_interp = interp2d(alt0, x_, abso_coef0)
        abso_coef = f_interp(alt, x_)

        slit_func0     = f.absgy[:Ng, index_wvl]
        slit_func      = np.empty(abso_coef.shape, dtype=slit_func0.dtype)
        slit_func[...] = slit_func0[:, None]

        solar     = f.solx[:Ng, index_wvl]
        weight    = slit_func0/slit_func0.sum()

        self.Ng   = Ng
        self.nwl  = wvls.size
        self.coef = {
                'wvl'       : {'name':'Wavelength'                     , 'data':self.wvl, 'units':'nm'},
                'abso_coef' : {'name':'Absorption Coefficient (Nz, Ng)', 'data':np.transpose(abso_coef)},
                'slit_func' : {'name':'Slit Function (Nz, Ng)'         , 'data':np.transpose(slit_func)},
                'solar'     : {'name':'Solar Factor (Ng)'              , 'data':solar},
                'weight'    : {'name':'Weight (Ng)'                    , 'data':weight}
                     }


class abs_oco_h5:

    """
    Added by Yu-Wen Chen (Yu-Wen.Chen@colorado.edu)
    Date: 2023.06.20

    This part is modified from abs_oco_idl but read a self-defined absorption coefficient output.

    Input:
        wavelength: wavelength in nm
        fname     : file path for the correlated-k coefficients (in pickle format)
        atm_obj   : atmosphere object, e.g., atm_obj = atm_atmmod(levels=np.arange(21))

    Note:
        If wavelength is provided but fname does not exisit:
            calculate correlated-k coefficients and save data into fname

        If wavelength is provided but fname is None:
            calculate correlated-k coefficients without saving the data

        if wavelength is not provided but fname is provided (also existed):
            read out the data from fname

        if wavelength and fname are neither provided:
            exit with error message

    Output:
        self.coef['wvl']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """

    def __init__(self, \
                 wavelength  = None,  \
                 fname       = None,  \
                 fname_h5   = None,  \
                 atm_obj     = None,  \
                 overwrite   = False, \
                 verbose     = False):

        if fname_h5 is None:
            self.fname_h5 = '%s/abs/abs_oco_11.h5' % er3t.common.fdir_data
        else:
            self.fname_h5 = fname_h5

        self.verbose   = verbose
        self.wvl       = wavelength

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((wavelength is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((wavelength is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(atm_obj)
            self.dump(fname)

        elif ((wavelength is not None) and (fname is None)):

            self.run(atm_obj)

        else:

            sys.exit('Error   [abs_oco_h5]: Please provide \'wavelength\' to proceed.')


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'coef'):
                if self.verbose:
                    print('Message [abs_oco_h5]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.wvl   = obj.wvl
                self.nwl   = obj.nwl
                self.coef  = obj.coef
                self.Ng    = obj.Ng
            else:
                sys.exit('Error   [abs_oco_h5]: \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, atm_obj):

        if not os.path.exists(self.fname_h5):
            sys.exit('Error   [abs_oco_h5]: Missing IDL database.')

        # self.coef
        #     self.coef['wvl']
        #     self.coef['abso_coef']
        #     self.coef['slit_func']
        #     self.coef['solar']
        self.get_coefficient()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [abs_oco_h5]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def get_coefficient(self, wvl_threshold=1.0):
        with h5py.File(self.fname_h5, 'r') as f:
            wvl_center_oco = f['lamx'][...]*1000.0

            index_wvl = np.argmin(np.abs(wvl_center_oco-self.wvl))
            if abs(wvl_center_oco[index_wvl]-self.wvl) >= wvl_threshold:
                sys.exit('Error   [abs_oco_h5]: Cannot pick a close wavelength for %.2fnm from \'%s\'.' % (self.wvl, self.fname_h5))
            else:
                if self.verbose:
                    print('Message [abs_oco_h5]: Picked wvl=%.2f from \'%s\' for input wavelength %.2fnm.' % (wvl_center_oco[index_wvl], self.fname_h5, self.wvl))
                self.wvl = wvl_center_oco[index_wvl]

            Ng              = f['absgn'][...][index_wvl]
            wvls            = f['absgx'][...][index_wvl, :Ng].T * 1000.0
            abso_coef       = f['absgl'][...][:, index_wvl, :Ng].T
            slit_func0      = f['absgy'][...][index_wvl, :Ng].T
            slit_func       = np.empty(abso_coef.shape, dtype=slit_func0.dtype)
            slit_func[...]  = slit_func0[:, None]
            solar           = f['solx'][index_wvl, :Ng].T
            weight          = slit_func0/slit_func0.sum()
            
            self.Ng         = Ng
            self.nwl        = wvls.size
            self.coef = {'wvl'       : {'name':'Wavelength'                     , 'data':self.wvl, 'units':'nm'},
                        'abso_coef' : {'name':'Absorption Coefficient (Nz, Ng)', 'data':np.transpose(abso_coef)},
                        'slit_func' : {'name':'Slit Function (Nz, Ng)'         , 'data':np.transpose(slit_func)},
                        'solar'     : {'name':'Solar Factor (Ng)'              , 'data':solar},
                        'weight'    : {'name':'Weight (Ng)'                    , 'data':weight}
                        }


if __name__ == '__main__':

    pass
