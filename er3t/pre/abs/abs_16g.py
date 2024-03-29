import os
import sys
import pickle
import multiprocessing as mp
import h5py
import copy
import numpy as np

import er3t



__all__ = ['abs_16g', 'abs_16g_txt', 'gen_h5_abs_16g']



class abs_16g:

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
        self.coef['wavelength']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """

    Ng       = 16
    group_s  = '/solar_v1.3'
    fname_h5 = '%s/abs_16g.h5' % er3t.common.fdir_data_abs
    reference = 'Coddington, O., Schmidt, K. S., Pilewskie, P., Gore, W. J., Bergstrom, R., Roman, M., Redemann, J., Russell, P. B., Liu, J., and Schaaf, C. C.: Aircraft measurements of spectral surface albedo and its consistency with ground based and space-borne observations, J. Geophys. Res., 113, D17209, doi:10.1029/2008JD010089, 2008.'

    def __init__(self, \
                 wavelength = er3t.common.params['wavelength'],  \
                 fname      = None,  \
                 atm_obj    = None,  \
                 overwrite  = False, \
                 verbose    = False):

        self.verbose   = verbose
        self.wvl       = wavelength
        self.nwl       = 1

        if self.reference not in er3t.common.references:
            er3t.common.references.append(self.reference)

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((wavelength is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((wavelength is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(atm_obj)
            self.dump(fname)

        elif ((wavelength is not None) and (fname is None)):

            self.run(atm_obj)

        else:

            sys.exit('Error   [abs_16g]: Please provide \'wavelength\' to proceed.')


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'coef'):
                if self.verbose:
                    print('Message [abs_16g]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.wvl   = obj.wvl
                self.nwl   = obj.nwl
                self.coef  = obj.coef
                self.wvl_info = obj.wvl_info
            else:
                sys.exit('Error   [abs_16g]: \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, atm_obj):

        self.wvl_info  = '%.2f nm (applied SSFR slit)' % self.wvl

        if not os.path.exists(self.fname_h5):
            sys.exit('Error   [abs_16g]: Missing HDF5 database.')

        f_h5 = h5py.File(self.fname_h5, 'r')

        # self.lay: Python dictionary
        #   self.lay['pressure']: Python dictionary, contain 'name', 'data' etc
        #   self.lay['altitude']
        #   self.lay['temperature']
        #   self.lay['thickness']
        #   self.lay['factor']
        #   self.lay['h2o']
        #   self.lay['co2']
        #   self.lay['ch4']
        #   self.lay['o3']
        #   self.lay['o2']
        #   self.lay['no2']
        if atm_obj is not None:
            self.prep_atmosphere(atm_obj)
        else:
            sys.exit('Error   [abs_16g]: \'atm_obj\' needs to be provided.')

        # self.fac
        #   self.fac['jpd']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['jpu']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['jtd']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['jtu']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['fac_Tp'] : Python dictionary, contain 'name', 'data' etc
        #   self.fac['fac_vTp']: Python dictionary, contain 'name', 'data' etc
        self.cal_factor()

        # self.abso
        #   self.abso[0]; self.abso[1]; self.abso[2] ...
        #     self.abso[0]['wvl']
        #     self.abso[0]['jpd']
        #     self.abso[0]['jpu']
        #     self.abso[0]['jtd']
        #     self.abso[0]['jtu']
        #     self.abso[0]['factor']
        #     self.abso[0]['absorber']
        self.prep_absorption()

        # self.coef
        #     self.coef['wvl']
        #     self.coef['abso_coef']
        #     self.coef['slit_func']
        #     self.coef['solar']
        self.get_coefficient(f_h5)

        f_h5.close()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [abs_16g]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def prep_atmosphere(self, atm_obj):

        self.lay = copy.deepcopy(atm_obj.lay)

        # rescale atmospheric profiles from [#/cm^3] to [#/cm^2]
        # A 1e-20 factor was applied when calculating the ks to prevent overflow problems
        # The 1e5 factor below is to convert the layer depth (in km) to cm
        factor = 1.0e-20 * 1.0e5 * self.lay['thickness']['data']
        for key in self.lay.keys():
            if key not in ['pressure', 'temperature', 'altitude', 'thickness']:
                self.lay[key]['data'] = self.lay[key]['data'] * factor

        self.Nz      = self.lay['altitude']['data'].size


    def cal_factor(self):

        # get index for atmosphere with respect to reference atmosphere
        pref, pref_log, tref, vref, vref_log = self.load_reference()

        p_log       = np.log(self.lay['pressure']['data'])
        jpd         = np.int_(35.0 - 5.0*(p_log+0.04))
        jpd[jpd<0]  = 0
        jpd[jpd>57] = 57
        jpu         = jpd + 1

        # calculate pressure interpolation factor
        fpu          = np.zeros_like(p_log, dtype=np.float64)
        div          = pref_log[jpd] - pref_log[jpu]
        indices      = np.where((div > 0.001))[0]
        fpu[indices] = (pref_log[jpd[indices]]-p_log[indices]) / div[indices]
        fpu[fpu>1.0] = 1.0
        fpu[fpu<0.0] = 0.0
        fpd          = 1 - fpu

        # calculate temperature interpolation factor
        atm_temp   = self.lay['temperature']['data']
        # delt       = np.array([-30, -15, 0, 15, 30])
        jtd        = np.int_(2.0 + (atm_temp-tref[jpd])/15.0)
        jtd[jtd<0] = 0
        jtd[jtd>3] = 3
        jtu        = np.int_(2.0 + (atm_temp-tref[jpu])/15.0)
        jtu[jtu<0] = 0
        jtu[jtu>3] = 3
        ftd        = (atm_temp-tref[jpd])/15.0 - np.float_(jtd-2.0)
        ftu        = (atm_temp-tref[jpu])/15.0 - np.float_(jtu-2.0)

        # calculate water vapor mixing ratio interpolation factor
        atm_h2o_mix = self.lay['h2o']['data'] / self.lay['factor']['data']
        atm_h2o_mix_log = np.log(atm_h2o_mix)
        atm_h2o_mix_log[atm_h2o_mix_log<-1.2206e+01] = -1.2206e+01
        atm_h2o_mix_log[atm_h2o_mix_log>-3.2061e+00] = -3.2061e+00
        jwd         = np.int_(12.2 + atm_h2o_mix_log)
        jwd[jwd<0]  = 0
        jwd[jwd>8]  = 8
        jwu         = jwd + 1
        fvd         = atm_h2o_mix_log - vref_log[jwd]

        # set coefficients for k interpolation (for each layer of input atmosphere)
        Nz      = p_log.size
        fac_Tp  = np.zeros((Nz, 2, 2)   , dtype=np.float64)
        fac_vTp = np.zeros((Nz, 2, 2, 2), dtype=np.float64)

        # pressure 0:
        fac_Tp[:, 0, 0] = (1.0-ftd) * fpd
        fac_Tp[:, 1, 0] = ftd       * fpd
        # pressure 1:
        fac_Tp[:, 0, 1] = (1.0-ftu) * fpu
        fac_Tp[:, 1, 1] = ftu       * fpu

        # pressure 0:
        fac_vTp[:,0,0,0] = fvd      * (1.-ftd) * fpd    # fac000
        fac_vTp[:,1,0,0] = (1.-fvd) * (1.-ftd) * fpd    # fac100
        fac_vTp[:,0,1,0] = fvd      * ftd      * fpd    # fac010
        fac_vTp[:,1,1,0] = (1.-fvd) * ftd      * fpd    # fac110
        # pressure 1:
        fac_vTp[:,0,0,1] = fvd      * (1.-ftu) * fpu    # fac001
        fac_vTp[:,1,0,1] = (1.-fvd) * (1.-ftu) * fpu    # fac101
        fac_vTp[:,0,1,1] = fvd      * ftu      * fpu    # fac011
        fac_vTp[:,1,1,1] = (1.-fvd) * ftu      * fpu    # fac111

        self.fac = {}
        self.fac['jpd']     = {'name':'jpd'    , 'data':jpd}
        self.fac['jpu']     = {'name':'jpu'    , 'data':jpu}
        self.fac['jtd']     = {'name':'jtd'    , 'data':jtd}
        self.fac['jtu']     = {'name':'jtu'    , 'data':jtu}
        self.fac['jwd']     = {'name':'jwd'    , 'data':jwd}
        self.fac['jwu']     = {'name':'jwu'    , 'data':jwu}
        self.fac['fac_Tp']  = {'name':'fac_Tp' , 'data':fac_Tp}
        self.fac['fac_vTp'] = {'name':'fac_vTp', 'data':fac_vTp}


    def prep_absorption(self, wvl_join=980):

        self.abso = {}

        abso_ref = {}
        abso_ref['wvl']      = {'name':'Wavelength', 'data':self.wvl}
        abso_ref['slit']     = {'name':'slit'      , 'data':True}
        abso_ref['solar']    = {'name':'solar'     , 'data':True}
        abso_ref['jpd']      = copy.deepcopy(self.fac['jpd'])
        abso_ref['jpu']      = copy.deepcopy(self.fac['jpu'])
        abso_ref['jtd']      = copy.deepcopy(self.fac['jtd'])
        abso_ref['jtu']      = copy.deepcopy(self.fac['jtu'])
        abso_ref['jwd']      = copy.deepcopy(self.fac['jwd'])
        abso_ref['jwu']      = copy.deepcopy(self.fac['jwu'])

        if self.wvl < 300.0:
            sys.exit('Error   [abs_16g]: Wavelength too short - no absorption data available.')

        elif (300.0 <= self.wvl < 448.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_uv' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['o3'])
            self.abso[0]      = abso0

        elif (448.0 <= self.wvl < 500.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_uv' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            abso0['slit']['data'] = False
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_uv' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (500.0 <= self.wvl < 620.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_k' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_k' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (620.0 <= self.wvl < 640.0) | (680.0 <= self.wvl < 700.0) | \
             (750.0 <= self.wvl < 760.0) | (770.0 <= self.wvl < 780.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['O2_cont5','kgo2']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_o2' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['o2'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_o2' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (640.0 <= self.wvl < 680.0) | (700.0 <= self.wvl < 750.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_k' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_k' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

            abso2             = copy.deepcopy(abso_ref)
            abso2['absorber'] = {'name':'absorber', 'data':['O2_cont5','kgo2']}
            abso2['group_s']  = {'name':'group_s' , 'data':'%s/solar_k' % self.group_s}
            abso2['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso2['gas_conc'] = copy.deepcopy(self.lay['o2'])
            abso2['slit']['data']  = False
            abso2['solar']['data'] = False
            self.abso[2]      = abso2

        elif (760.0 <= self.wvl < 770.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_o2' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            abso0['slit']['data']  = False
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_o2' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

            abso2             = copy.deepcopy(abso_ref)
            abso2['absorber'] = {'name':'absorber', 'data':['O2_cont5','kgo2']}
            abso2['group_s']  = {'name':'group_s' , 'data':'%s/solar_o2' % self.group_s}
            abso2['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso2['gas_conc'] = copy.deepcopy(self.lay['o2'])
            abso2['solar']['data'] = False
            self.abso[2]      = abso2

        elif (780.0 <= self.wvl < wvl_join):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_k' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

        elif (wvl_join <= self.wvl < 1240.0) | (1630.0 <= self.wvl < 1940.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O/k_arraynir', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

        elif (1240.0 <= self.wvl < 1300.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['O2_cont5/k_arraynir', 'kgo2']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['o2'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['CO2', 'kgco2']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['co2'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (1300.0 <= self.wvl < 1420.0) | (1450.0 <= self.wvl < 1560.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O/k_arraynir', 'kgh2o']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['CO2', 'kgco2']}
            abso1['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['co2'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (1420.0 <= self.wvl < 1450.0) | (1560.0 <= self.wvl < 1630.0) | \
             (1940.0 <= self.wvl < 2150.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['CO2', 'kgco2']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['co2'])
            self.abso[0]      = abso0

        elif (2150.0 <= self.wvl < 2500.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['CH4', 'kgch4']}
            abso0['group_s']  = {'name':'group_s' , 'data':'%s/solar_nir' % self.group_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['ch4'])
            self.abso[0]      = abso0

        elif self.wvl > 2500.0:
            sys.exit('Error   [abs_16g]: Wavelength too large - no absorption data available.')


    def get_coefficient(self, f_h5):

        abso_coef = np.zeros((self.Nz, self.Ng), dtype=np.float64)
        slit_func = np.zeros((self.Nz, self.Ng), dtype=np.float64)
        solar     = np.zeros(self.Ng, dtype=np.float64)

        for ia, key in enumerate(sorted(self.abso.keys())):

            abso_coef0 = np.zeros_like(abso_coef)

            abso_dict = copy.deepcopy(self.abso[key])

            group_s = '%s/solar_taug.%d' % (abso_dict['group_s']['data'], np.round(abso_dict['wvl']['data']))
            if group_s not in f_h5:
                sys.exit('Error   [abs_16g]: Cannot find \'%s\'.' % group_s)

            # read solar file
            # =============================================================================================
            v1, v2, dv, npts, sol_min, sol_max, sol_int = f_h5['%s/params' % group_s][...]
            s0                                          = f_h5['%s/data' % group_s][...][:, -1]

            l1 = 1.0e7 / v2
            l2 = 1.0e7 / v1
            cv = (v2-v1) / (l2-l1)

            if abso_dict['solar']['data']:
                solar = s0 * cv
            # =============================================================================================


            # read absorption coefficient and slit function
            # =============================================================================================
            if abso_dict['factor']['data'].ndim == 3:

                for iz in range(self.Nz):

                    vname_pdtd = '/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_pdtu = '/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_putd = '/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_putu = '/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    data_pdtd = f_h5[vname_pdtd][...]
                    data_pdtu = f_h5[vname_pdtu][...]
                    data_putd = f_h5[vname_putd][...]
                    data_putu = f_h5[vname_putu][...]

                    if abso_dict['slit']['data']:
                        slit_func[iz, :] = data_pdtd[:, -1]
                    abso_coef0[iz, :] = abso_dict['gas_conc']['data'][iz] * \
                            (abso_dict['factor']['data'][iz,0,0]*data_pdtd[:, 2] + \
                             abso_dict['factor']['data'][iz,1,0]*data_pdtu[:, 2] + \
                             abso_dict['factor']['data'][iz,0,1]*data_putd[:, 2] + \
                             abso_dict['factor']['data'][iz,1,1]*data_putu[:, 2])

            elif abso_dict['factor']['data'].ndim == 4:

                for iz in range(self.Nz):

                    vname_pdtdl = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_pdtul = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_putdl = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_putul = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_pdtdh = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_pdtuh = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_putdh = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    vname_putuh = '/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    data_pdtdl = f_h5[vname_pdtdl][...]
                    data_pdtul = f_h5[vname_pdtul][...]
                    data_putdl = f_h5[vname_putdl][...]
                    data_putul = f_h5[vname_putul][...]
                    data_pdtdh = f_h5[vname_pdtdh][...]
                    data_pdtuh = f_h5[vname_pdtuh][...]
                    data_putdh = f_h5[vname_putdh][...]
                    data_putuh = f_h5[vname_putuh][...]

                    if abso_dict['slit']['data']:
                        slit_func[iz, :] = data_pdtdl[:, -1]
                    abso_coef0[iz, :] = abso_dict['gas_conc']['data'][iz] * \
                            (abso_dict['factor']['data'][iz,0,0,0]*data_pdtdl[:, 2] + \
                             abso_dict['factor']['data'][iz,0,1,0]*data_pdtul[:, 2] + \
                             abso_dict['factor']['data'][iz,0,0,1]*data_putdl[:, 2] + \
                             abso_dict['factor']['data'][iz,0,1,1]*data_putul[:, 2] + \
                             abso_dict['factor']['data'][iz,1,0,0]*data_pdtdh[:, 2] + \
                             abso_dict['factor']['data'][iz,1,1,0]*data_pdtuh[:, 2] + \
                             abso_dict['factor']['data'][iz,1,0,1]*data_putdh[:, 2] + \
                             abso_dict['factor']['data'][iz,1,1,1]*data_putuh[:, 2])

            # =============================================================================================

            abso_coef += abso_coef0

        weight = self.load_weight()

        self.coef = {
                'wvl'       : {'name':'Wavelength'                     , 'data':self.wvl, 'units':'nm'},
                'abso_coef' : {'name':'Absorption Coefficient (Nz, Ng)', 'data':abso_coef},
                'slit_func' : {'name':'Slit Function (Nz, Ng)'         , 'data':slit_func},
                'solar'     : {'name':'Solar Factor (Ng)'              , 'data':solar},
                'weight'    : {'name':'Weight (Ng)'                    , 'data':weight}
                     }


    def load_reference(self):

        """
        returns data of reference atmosphere (MLS up to 100 km)
        """

        # pressure
        pref = np.array([
                1.05363E+03,8.62642E+02,7.06272E+02,5.78246E+02,4.73428E+02,\
                3.87610E+02,3.17348E+02,2.59823E+02,2.12725E+02,1.74164E+02,\
                1.42594E+02,1.16746E+02,9.55835E+01,7.82571E+01,6.40715E+01,\
                5.24573E+01,4.29484E+01,3.51632E+01,2.87892E+01,2.35706E+01,\
                1.92980E+01,1.57998E+01,1.29358E+01,1.05910E+01,8.67114E+00,\
                7.09933E+00,5.81244E+00,4.75882E+00,3.89619E+00,3.18993E+00,\
                2.61170E+00,2.13828E+00,1.75067E+00,1.43333E+00,1.17351E+00,\
                9.60789E-01,7.86628E-01,6.44036E-01,5.27292E-01,4.31710E-01,\
                3.53455E-01,2.89384E-01,2.36928E-01,1.93980E-01,1.58817E-01,\
                1.30029E-01,1.06458E-01,8.71608E-02,7.13612E-02,5.84256E-02,\
                4.78349E-02,3.91639E-02,3.20647E-02,2.62523E-02,2.14936E-02,\
                1.75975E-02,1.44076E-02,1.17959E-02,9.65769E-03], dtype=np.float64)

        pref_log = np.array([
                6.9600E+00, 6.7600E+00, 6.5600E+00, 6.3600E+00, 6.1600E+00,\
                5.9600E+00, 5.7600E+00, 5.5600E+00, 5.3600E+00, 5.1600E+00,\
                4.9600E+00, 4.7600E+00, 4.5600E+00, 4.3600E+00, 4.1600E+00,\
                3.9600E+00, 3.7600E+00, 3.5600E+00, 3.3600E+00, 3.1600E+00,\
                2.9600E+00, 2.7600E+00, 2.5600E+00, 2.3600E+00, 2.1600E+00,\
                1.9600E+00, 1.7600E+00, 1.5600E+00, 1.3600E+00, 1.1600E+00,\
                9.6000E-01, 7.6000E-01, 5.6000E-01, 3.6000E-01, 1.6000E-01,\
                -4.0000E-02,-2.4000E-01,-4.4000E-01,-6.4000E-01,-8.4000E-01,\
                -1.0400E+00,-1.2400E+00,-1.4400E+00,-1.6400E+00,-1.8400E+00,\
                -2.0400E+00,-2.2400E+00,-2.4400E+00,-2.6400E+00,-2.8400E+00,\
                -3.0400E+00,-3.2400E+00,-3.4400E+00,-3.6400E+00,-3.8400E+00,\
                -4.0400E+00,-4.2400E+00,-4.4400E+00,-4.6400E+00], dtype=np.float64)

        # temperature
        tref = np.array([
                2.9420E+02, 2.8799E+02, 2.7894E+02, 2.6925E+02, 2.5983E+02,\
                2.5017E+02, 2.4077E+02, 2.3179E+02, 2.2306E+02, 2.1578E+02,\
                2.1570E+02, 2.1570E+02, 2.1570E+02, 2.1706E+02, 2.1858E+02,\
                2.2018E+02, 2.2174E+02, 2.2328E+02, 2.2479E+02, 2.2655E+02,\
                2.2834E+02, 2.3113E+02, 2.3401E+02, 2.3703E+02, 2.4022E+02,\
                2.4371E+02, 2.4726E+02, 2.5085E+02, 2.5457E+02, 2.5832E+02,\
                2.6216E+02, 2.6606E+02, 2.6999E+02, 2.7340E+02, 2.7536E+02,\
                2.7568E+02, 2.7372E+02, 2.7163E+02, 2.6955E+02, 2.6593E+02,\
                2.6211E+02, 2.5828E+02, 2.5360E+02, 2.4854E+02, 2.4348E+02,\
                2.3809E+02, 2.3206E+02, 2.2603E+02, 2.2000E+02, 2.1435E+02,\
                2.0887E+02, 2.0340E+02, 1.9792E+02, 1.9290E+02, 1.8809E+02,\
                1.8329E+02, 1.7849E+02, 1.7394E+02, 1.7212E+02], dtype=np.float64)

        # water vapor mixing ratio
        vref = np.array([
                5.0000E-06, 1.3591E-05, 3.6945E-05, 1.0043E-04, 2.7299E-04,\
                7.4207E-04, 2.0171E-03, 5.4832E-03, 1.4905E-02, 4.0515E-02], dtype=np.float64)

        vref_log = np.array([
                -1.2206E+01,-1.1206E+01,-1.0206E+01,-9.2061E+00,-8.2061E+00,\
                -7.2061E+00,-6.2061E+00,-5.2061E+00,-4.2061E+00,-3.2061E+00], dtype=np.float64)

        return pref, pref_log, tref, vref, vref_log


    def load_weight(self):

        weight = np.array([\
                0.1527534276, 0.1491729617, 0.1420961469, 0.1316886544, \
                0.1181945205, 0.1019300893, 0.0832767040, 0.0626720116, \
                0.0424925000, 0.0046269894, 0.0038279891, 0.0030260086, \
                0.0022199750, 0.0014140010, 0.0005330000, 0.000075])

        return weight


    def download(self):

        pass



class abs_16g_txt:

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
        self.coef['wavelength']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """

    Ng     = 16
    fdir_k = '/argus/seven2/hofmann'
    fdir_s = '/argus/seven2/hofmann/solar_v1.3'


    def __init__(self, \
                 wavelength = None,  \
                 fname      = None,  \
                 atm_obj    = None,  \
                 overwrite  = False, \
                 verbose    = False):

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

            sys.exit('Error   [abs_16g_txt]: please provide \'wavelength\' to proceed.')


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'coef'):
                if self.verbose:
                    print('Message [abs_16g_txt]: loading %s ...' % fname)
                self.fname = obj.fname
                self.wvl   = obj.wvl
                self.coef  = obj.coef
            else:
                sys.exit('Error   [abs_16g_txt]: %s is not the correct \'pickle\' file to load.' % fname)


    def run(self, atm_obj):

        if not (os.path.exists(self.fdir_k) & os.path.exists(self.fdir_s)):
            sys.exit('Error   [abs_16g_txt]: missing database at %s.' % self.fdir_k)

        # self.lay: Python dictionary
        #   self.lay['pressure']: Python dictionary, contain 'name', 'data' etc
        #   self.lay['altitude']
        #   self.lay['temperature']
        #   self.lay['thickness']
        #   self.lay['factor']
        #   self.lay['h2o']
        #   self.lay['co2']
        #   self.lay['ch4']
        #   self.lay['o3']
        #   self.lay['o2']
        #   self.lay['no2']
        if atm_obj is not None:
            self.prep_atmosphere(atm_obj)
        else:
            sys.exit('Error   [abs_16g_txt]: \'atm_obj\' needs to be provided.')

        # self.fac
        #   self.fac['jpd']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['jpu']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['jtd']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['jtu']    : Python dictionary, contain 'name', 'data' etc
        #   self.fac['fac_Tp'] : Python dictionary, contain 'name', 'data' etc
        #   self.fac['fac_vTp']: Python dictionary, contain 'name', 'data' etc
        self.cal_factor()

        # self.abso
        #   self.abso[0]; self.abso[1]; self.abso[2] ...
        #     self.abso[0]['wvl']
        #     self.abso[0]['fdir_k']
        #     self.abso[0]['jpd']
        #     self.abso[0]['jpu']
        #     self.abso[0]['jtd']
        #     self.abso[0]['jtu']
        #     self.abso[0]['fdir_s']
        #     self.abso[0]['factor']
        #     self.abso[0]['absorber']
        self.prep_absorption()

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
                print('Message [abs_16g_txt]: saving object into %s ...' % fname)
            pickle.dump(self, f)


    def prep_atmosphere(self, atm_obj):

        self.lay = copy.deepcopy(atm_obj.lay)

        # rescale atmospheric profiles from [#/cm^3] to [#/cm^2]
        # A 1e-20 factor was applied when calculating the ks to prevent overflow problems
        # The 1e5 factor below is to convert the layer depth (in km) to cm
        factor = 1.0e-20 * 1.0e5 * self.lay['thickness']['data']
        for key in self.lay.keys():
            if key not in ['pressure', 'temperature', 'altitude', 'thickness']:
                self.lay[key]['data'] = self.lay[key]['data'] * factor

        self.Nz      = self.lay['altitude']['data'].size


    def cal_factor(self):

        # get index for atmosphere with respect to reference atmosphere
        pref, pref_log, tref, vref, vref_log = self.load_reference()

        p_log       = np.log(self.lay['pressure']['data'])
        jpd         = np.int_(35.0 - 5.0*(p_log+0.04))
        jpd[jpd<0]  = 0
        jpd[jpd>57] = 57
        jpu         = jpd + 1

        # calculate pressure interpolation factor
        fpu          = np.zeros_like(p_log, dtype=np.float64)
        div          = pref_log[jpd] - pref_log[jpu]
        indices      = np.where((div > 0.001))[0]
        fpu[indices] = (pref_log[jpd[indices]]-p_log[indices]) / div[indices]
        fpu[fpu>1.0] = 1.0
        fpu[fpu<0.0] = 0.0
        fpd          = 1 - fpu

        # calculate temperature interpolation factor
        atm_temp   = self.lay['temperature']['data']
        # delt       = np.array([-30, -15, 0, 15, 30])
        jtd        = np.int_(2.0 + (atm_temp-tref[jpd])/15.0)
        jtd[jtd<0] = 0
        jtd[jtd>3] = 3
        jtu        = np.int_(2.0 + (atm_temp-tref[jpu])/15.0)
        jtu[jtu<0] = 0
        jtu[jtu>3] = 3
        ftd        = (atm_temp-tref[jpd])/15.0 - np.float_(jtd-2.0)
        ftu        = (atm_temp-tref[jpu])/15.0 - np.float_(jtu-2.0)

        # calculate water vapor mixing ratio interpolation factor
        atm_h2o_mix = self.lay['h2o']['data'] / self.lay['factor']['data']
        atm_h2o_mix_log = np.log(atm_h2o_mix)
        atm_h2o_mix_log[atm_h2o_mix_log<-1.2206e+01] = -1.2206e+01
        atm_h2o_mix_log[atm_h2o_mix_log>-3.2061e+00] = -3.2061e+00
        jwd         = np.int_(12.2 + atm_h2o_mix_log)
        jwd[jwd<0]  = 0
        jwd[jwd>8]  = 8
        jwu         = jwd + 1
        fvd         = atm_h2o_mix_log - vref_log[jwd]

        # set coefficients for k interpolation (for each layer of input atmosphere)
        Nz      = p_log.size
        fac_Tp  = np.zeros((Nz, 2, 2)   , dtype=np.float64)
        fac_vTp = np.zeros((Nz, 2, 2, 2), dtype=np.float64)

        # pressure 0:
        fac_Tp[:, 0, 0] = (1.0-ftd) * fpd
        fac_Tp[:, 1, 0] = ftd       * fpd
        # pressure 1:
        fac_Tp[:, 0, 1] = (1.0-ftu) * fpu
        fac_Tp[:, 1, 1] = ftu       * fpu

        # pressure 0:
        fac_vTp[:,0,0,0] = fvd      * (1.-ftd) * fpd    # fac000
        fac_vTp[:,1,0,0] = (1.-fvd) * (1.-ftd) * fpd    # fac100
        fac_vTp[:,0,1,0] = fvd      * ftd      * fpd    # fac010
        fac_vTp[:,1,1,0] = (1.-fvd) * ftd      * fpd    # fac110
        # pressure 1:
        fac_vTp[:,0,0,1] = fvd      * (1.-ftu) * fpu    # fac001
        fac_vTp[:,1,0,1] = (1.-fvd) * (1.-ftu) * fpu    # fac101
        fac_vTp[:,0,1,1] = fvd      * ftu      * fpu    # fac011
        fac_vTp[:,1,1,1] = (1.-fvd) * ftu      * fpu    # fac111

        self.fac = {}
        self.fac['jpd']     = {'name':'jpd'    , 'data':jpd}
        self.fac['jpu']     = {'name':'jpu'    , 'data':jpu}
        self.fac['jtd']     = {'name':'jtd'    , 'data':jtd}
        self.fac['jtu']     = {'name':'jtu'    , 'data':jtu}
        self.fac['jwd']     = {'name':'jwd'    , 'data':jwd}
        self.fac['jwu']     = {'name':'jwu'    , 'data':jwu}
        self.fac['fac_Tp']  = {'name':'fac_Tp' , 'data':fac_Tp}
        self.fac['fac_vTp'] = {'name':'fac_vTp', 'data':fac_vTp}


    def prep_absorption(self, wvl_join=980):

        self.abso = {}

        abso_ref = {}
        abso_ref['wvl']      = {'name':'Wavelength', 'data':self.wvl}
        abso_ref['fdir_k']   = {'name':'fdir_k'    , 'data':self.fdir_k}
        abso_ref['slit']     = {'name':'slit'      , 'data':True}
        abso_ref['solar']    = {'name':'solar'     , 'data':True}
        abso_ref['jpd']      = copy.deepcopy(self.fac['jpd'])
        abso_ref['jpu']      = copy.deepcopy(self.fac['jpu'])
        abso_ref['jtd']      = copy.deepcopy(self.fac['jtd'])
        abso_ref['jtu']      = copy.deepcopy(self.fac['jtu'])
        abso_ref['jwd']      = copy.deepcopy(self.fac['jwd'])
        abso_ref['jwu']      = copy.deepcopy(self.fac['jwu'])

        if self.wvl < 300.0:
            sys.exit('Error   [abs_16g_txt]: Wavelength too short - no absorption data available.')

        elif (300.0 <= self.wvl < 448.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_uv' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['o3'])
            self.abso[0]      = abso0

        elif (448.0 <= self.wvl < 500.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_uv' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            abso0['slit']['data'] = False
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_uv' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (500.0 <= self.wvl < 620.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_k' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_k' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (620.0 <= self.wvl < 640.0) | (680.0 <= self.wvl < 700.0) | \
             (750.0 <= self.wvl < 760.0) | (770.0 <= self.wvl < 780.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['O2_cont5','kgo2']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_o2' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['o2'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_o2' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (640.0 <= self.wvl < 680.0) | (700.0 <= self.wvl < 750.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_k' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_k' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

            abso2             = copy.deepcopy(abso_ref)
            abso2['absorber'] = {'name':'absorber', 'data':['O2_cont5','kgo2']}
            abso2['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_k' % self.fdir_s}
            abso2['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso2['gas_conc'] = copy.deepcopy(self.lay['o2'])
            abso2['slit']['data']  = False
            abso2['solar']['data'] = False
            self.abso[2]      = abso2

        elif (760.0 <= self.wvl < 770.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_o2' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            abso0['slit']['data']  = False
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['O3', 'kgo3']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_o2' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['o3'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

            abso2             = copy.deepcopy(abso_ref)
            abso2['absorber'] = {'name':'absorber', 'data':['O2_cont5','kgo2']}
            abso2['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_o2' % self.fdir_s}
            abso2['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso2['gas_conc'] = copy.deepcopy(self.lay['o2'])
            abso2['solar']['data'] = False
            self.abso[2]      = abso2

        elif (780.0 <= self.wvl < wvl_join):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_k' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

        elif (wvl_join <= self.wvl < 1240.0) | (1630.0 <= self.wvl < 1940.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O/k_arraynir', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

        elif (1240.0 <= self.wvl < 1300.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['O2_cont5/k_arraynir', 'kgo2']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['o2'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['CO2', 'kgco2']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['co2'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (1300.0 <= self.wvl < 1420.0) | (1450.0 <= self.wvl < 1560.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['H2O/k_arraynir', 'kgh2o']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['h2o'])
            self.abso[0]      = abso0

            abso1             = copy.deepcopy(abso_ref)
            abso1['absorber'] = {'name':'absorber', 'data':['CO2', 'kgco2']}
            abso1['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso1['factor']   = copy.deepcopy(self.fac['fac_Tp'])
            abso1['gas_conc'] = copy.deepcopy(self.lay['co2'])
            abso1['slit']['data']  = False
            abso1['solar']['data'] = False
            self.abso[1]      = abso1

        elif (1420.0 <= self.wvl < 1450.0) | (1560.0 <= self.wvl < 1630.0) | \
             (1940.0 <= self.wvl < 2150.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['CO2', 'kgco2']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['co2'])
            self.abso[0]      = abso0

        elif (2150.0 <= self.wvl < 2500.0):
            abso0             = copy.deepcopy(abso_ref)
            abso0['absorber'] = {'name':'absorber', 'data':['CH4', 'kgch4']}
            abso0['fdir_s']   = {'name':'fdir_s'  , 'data':'%s/solar_nir' % self.fdir_s}
            abso0['factor']   = copy.deepcopy(self.fac['fac_vTp'])
            abso0['gas_conc'] = copy.deepcopy(self.lay['ch4'])
            self.abso[0]      = abso0

        elif self.wvl > 2500.0:
            sys.exit('Error   [abs_16g_txt]: Wavelength too large - no absorption data available.')


    def get_coefficient(self):

        abso_coef = np.zeros((self.Nz, self.Ng), dtype=np.float64)
        slit_func = np.zeros((self.Nz, self.Ng), dtype=np.float64)
        solar     = np.zeros(self.Ng, dtype=np.float64)

        for ia, key in enumerate(sorted(self.abso.keys())):

            abso_coef0 = np.zeros_like(abso_coef)

            abso_dict = copy.deepcopy(self.abso[key])

            fname_s = '%s/solar_taug.%d' % (abso_dict['fdir_s']['data'], np.round(abso_dict['wvl']['data']))
            if not os.path.isfile(fname_s):
                sys.exit('Error   [abs_16g_txt]: cannot find \'%s\'.' % fname_s)

            # read solar file
            # =============================================================================================
            try:
                with open(fname_s) as f:
                    v1, v2, dv, npts, sol_min, sol_max = np.fromstring(f.readline(), sep=' ', dtype=np.float64)
                    sol_int, = np.fromstring(f.readline(), sep=' ', dtype=np.float64)
                s0 = np.genfromtxt(fname_s, skip_header=2)[:, -1]
            except:
                with open(fname_s) as f:
                    v1, v2, dv, npts, sol_min, sol_max, sol_int = np.fromstring(f.readline(), sep=' ', dtype=np.float64)
                s0 = np.genfromtxt(fname_s, skip_header=1)[:, -1]

            l1 = 1.0e7 / v2
            l2 = 1.0e7 / v1
            cv = (v2-v1) / (l2-l1)

            if abso_dict['solar']['data']:
                solar = s0 * cv
            # =============================================================================================


            # read absorption coefficient and slit function
            # =============================================================================================
            if abso_dict['factor']['data'].ndim == 3:

                for iz in range(self.Nz):

                    fname_pdtd = '%s/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_pdtu = '%s/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_putd = '%s/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_putu = '%s/%s/pressure.%d/temperature.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1,    \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    data_pdtd = np.loadtxt(fname_pdtd)
                    data_pdtu = np.loadtxt(fname_pdtu)
                    data_putd = np.loadtxt(fname_putd)
                    data_putu = np.loadtxt(fname_putu)

                    if abso_dict['slit']['data']:
                        slit_func[iz, :] = data_pdtd[:, -1]
                    abso_coef0[iz, :] = abso_dict['gas_conc']['data'][iz] * \
                            (abso_dict['factor']['data'][iz,0,0]*data_pdtd[:, 2] + \
                             abso_dict['factor']['data'][iz,1,0]*data_pdtu[:, 2] + \
                             abso_dict['factor']['data'][iz,0,1]*data_putd[:, 2] + \
                             abso_dict['factor']['data'][iz,1,1]*data_putu[:, 2])

            elif abso_dict['factor']['data'].ndim == 4:

                for iz in range(self.Nz):

                    fname_pdtdl = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_pdtul = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_putdl = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_putul = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwd']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_pdtdh = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_pdtuh = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpd']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_putdh = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtd']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    fname_putuh = '%s/%s/pressure.%d/temperature.%d/wv.%d/%s.%d' % \
                            (abso_dict['fdir_k']['data'], abso_dict['absorber']['data'][0], \
                             abso_dict['jpu']['data'][iz]+1, abso_dict['jtu']['data'][iz]+1, abso_dict['jwu']['data'][iz]+1, \
                             abso_dict['absorber']['data'][1], abso_dict['wvl']['data'])

                    data_pdtdl = np.loadtxt(fname_pdtdl)
                    data_pdtul = np.loadtxt(fname_pdtul)
                    data_putdl = np.loadtxt(fname_putdl)
                    data_putul = np.loadtxt(fname_putul)
                    data_pdtdh = np.loadtxt(fname_pdtdh)
                    data_pdtuh = np.loadtxt(fname_pdtuh)
                    data_putdh = np.loadtxt(fname_putdh)
                    data_putuh = np.loadtxt(fname_putuh)

                    if abso_dict['slit']['data']:
                        slit_func[iz, :] = data_pdtdl[:, -1]
                    abso_coef0[iz, :] = abso_dict['gas_conc']['data'][iz] * \
                            (abso_dict['factor']['data'][iz,0,0,0]*data_pdtdl[:, 2] + \
                             abso_dict['factor']['data'][iz,0,1,0]*data_pdtul[:, 2] + \
                             abso_dict['factor']['data'][iz,0,0,1]*data_putdl[:, 2] + \
                             abso_dict['factor']['data'][iz,0,1,1]*data_putul[:, 2] + \
                             abso_dict['factor']['data'][iz,1,0,0]*data_pdtdh[:, 2] + \
                             abso_dict['factor']['data'][iz,1,1,0]*data_pdtuh[:, 2] + \
                             abso_dict['factor']['data'][iz,1,0,1]*data_putdh[:, 2] + \
                             abso_dict['factor']['data'][iz,1,1,1]*data_putuh[:, 2])

            # =============================================================================================

            abso_coef += abso_coef0

        weight = self.load_weight()

        self.coef = {
                'wvl'       : {'name':'Wavelength'                     , 'data':self.wvl, 'units':'nm'},
                'abso_coef' : {'name':'Absorption Coefficient (Nz, Ng)', 'data':abso_coef},
                'slit_func' : {'name':'Slit Function (Nz, Ng)'         , 'data':slit_func},
                'solar'     : {'name':'Solar Factor (Ng)'              , 'data':solar},
                'weight'    : {'name':'Weight (Ng)'                    , 'data':weight}
                     }


    def load_reference(self):

        """
        returns data of reference atmosphere (MLS up to 100 km)
        """

        # pressure
        pref = np.array([
                1.05363E+03,8.62642E+02,7.06272E+02,5.78246E+02,4.73428E+02,\
                3.87610E+02,3.17348E+02,2.59823E+02,2.12725E+02,1.74164E+02,\
                1.42594E+02,1.16746E+02,9.55835E+01,7.82571E+01,6.40715E+01,\
                5.24573E+01,4.29484E+01,3.51632E+01,2.87892E+01,2.35706E+01,\
                1.92980E+01,1.57998E+01,1.29358E+01,1.05910E+01,8.67114E+00,\
                7.09933E+00,5.81244E+00,4.75882E+00,3.89619E+00,3.18993E+00,\
                2.61170E+00,2.13828E+00,1.75067E+00,1.43333E+00,1.17351E+00,\
                9.60789E-01,7.86628E-01,6.44036E-01,5.27292E-01,4.31710E-01,\
                3.53455E-01,2.89384E-01,2.36928E-01,1.93980E-01,1.58817E-01,\
                1.30029E-01,1.06458E-01,8.71608E-02,7.13612E-02,5.84256E-02,\
                4.78349E-02,3.91639E-02,3.20647E-02,2.62523E-02,2.14936E-02,\
                1.75975E-02,1.44076E-02,1.17959E-02,9.65769E-03], dtype=np.float64)

        pref_log = np.array([
                6.9600E+00, 6.7600E+00, 6.5600E+00, 6.3600E+00, 6.1600E+00,\
                5.9600E+00, 5.7600E+00, 5.5600E+00, 5.3600E+00, 5.1600E+00,\
                4.9600E+00, 4.7600E+00, 4.5600E+00, 4.3600E+00, 4.1600E+00,\
                3.9600E+00, 3.7600E+00, 3.5600E+00, 3.3600E+00, 3.1600E+00,\
                2.9600E+00, 2.7600E+00, 2.5600E+00, 2.3600E+00, 2.1600E+00,\
                1.9600E+00, 1.7600E+00, 1.5600E+00, 1.3600E+00, 1.1600E+00,\
                9.6000E-01, 7.6000E-01, 5.6000E-01, 3.6000E-01, 1.6000E-01,\
                -4.0000E-02,-2.4000E-01,-4.4000E-01,-6.4000E-01,-8.4000E-01,\
                -1.0400E+00,-1.2400E+00,-1.4400E+00,-1.6400E+00,-1.8400E+00,\
                -2.0400E+00,-2.2400E+00,-2.4400E+00,-2.6400E+00,-2.8400E+00,\
                -3.0400E+00,-3.2400E+00,-3.4400E+00,-3.6400E+00,-3.8400E+00,\
                -4.0400E+00,-4.2400E+00,-4.4400E+00,-4.6400E+00], dtype=np.float64)

        # temperature
        tref = np.array([
                2.9420E+02, 2.8799E+02, 2.7894E+02, 2.6925E+02, 2.5983E+02,\
                2.5017E+02, 2.4077E+02, 2.3179E+02, 2.2306E+02, 2.1578E+02,\
                2.1570E+02, 2.1570E+02, 2.1570E+02, 2.1706E+02, 2.1858E+02,\
                2.2018E+02, 2.2174E+02, 2.2328E+02, 2.2479E+02, 2.2655E+02,\
                2.2834E+02, 2.3113E+02, 2.3401E+02, 2.3703E+02, 2.4022E+02,\
                2.4371E+02, 2.4726E+02, 2.5085E+02, 2.5457E+02, 2.5832E+02,\
                2.6216E+02, 2.6606E+02, 2.6999E+02, 2.7340E+02, 2.7536E+02,\
                2.7568E+02, 2.7372E+02, 2.7163E+02, 2.6955E+02, 2.6593E+02,\
                2.6211E+02, 2.5828E+02, 2.5360E+02, 2.4854E+02, 2.4348E+02,\
                2.3809E+02, 2.3206E+02, 2.2603E+02, 2.2000E+02, 2.1435E+02,\
                2.0887E+02, 2.0340E+02, 1.9792E+02, 1.9290E+02, 1.8809E+02,\
                1.8329E+02, 1.7849E+02, 1.7394E+02, 1.7212E+02], dtype=np.float64)

        # water vapor mixing ratio
        vref = np.array([
                5.0000E-06, 1.3591E-05, 3.6945E-05, 1.0043E-04, 2.7299E-04,\
                7.4207E-04, 2.0171E-03, 5.4832E-03, 1.4905E-02, 4.0515E-02], dtype=np.float64)

        vref_log = np.array([
                -1.2206E+01,-1.1206E+01,-1.0206E+01,-9.2061E+00,-8.2061E+00,\
                -7.2061E+00,-6.2061E+00,-5.2061E+00,-4.2061E+00,-3.2061E+00], dtype=np.float64)

        return pref, pref_log, tref, vref, vref_log


    def load_weight(self):

        weight = np.array([\
                0.1527534276, 0.1491729617, 0.1420961469, 0.1316886544, \
                0.1181945205, 0.1019300893, 0.0832767040, 0.0626720116, \
                0.0424925000, 0.0046269894, 0.0038279891, 0.0030260086, \
                0.0022199750, 0.0014140010, 0.0005330000, 0.000075])

        return weight



def gen_h5_abs_16g(fname_h5):

    fdir0 = '/argus/seven2/hofmann'

    if True:

        f = h5py.File(fname_h5, 'w')

        for sub in ['O3', 'CO2', 'CH4', 'H2O', 'O2_cont5']:

            fdir = '%s/%s' % (fdir0, sub)

            fnames = er3t.util.all_files(fdir)
            print(len(fnames))

            for fname in fnames:

                vname = fname.split('/')[-1]

                if ('temperature' in fname) and ('pressure' in fname) and (vname.split('.')[0] in ['kgo2', 'kgo3', 'kgh2o', 'kgco2', 'kgch4']) and ('old' not in fname) and ('test' not in fname):

                    group = fname.replace(fdir0, '').replace('/%s' % os.path.basename(fname), '')
                    if group not in f:
                        g = f.create_group(group)
                        print(group)
                    g[vname] = np.loadtxt(fname)

        f.close()

    if True:

        f = h5py.File(fname_h5, 'r+')

        fdir = '%s/solar_v1.3' % fdir0
        fnames = er3t.util.all_files(fdir)

        for fname in fnames:

            if ('solar_taug' in fname) and ('~' not in fname):

                group = fname.replace(fdir0, '')

                if group not in f:
                    g = f.create_group(group)
                    print(group)

                try:
                    with open(fname) as f0:
                        v1, v2, dv, npts, sol_min, sol_max = np.fromstring(f0.readline(), sep=' ', dtype=np.float64)
                        sol_int, = np.fromstring(f0.readline(), sep=' ', dtype=np.float64)

                    params = np.array([v1, v2, dv, npts, sol_min, sol_max, sol_int])
                    data   = np.genfromtxt(fname, skip_header=2)

                except:
                    with open(fname) as f0:
                        v1, v2, dv, npts, sol_min, sol_max, sol_int = np.fromstring(f0.readline(), sep=' ', dtype=np.float64)

                    params = np.array([v1, v2, dv, npts, sol_min, sol_max, sol_int])
                    data   = np.genfromtxt(fname, skip_header=1)

                if data.size > 0:
                    g['params'] = params
                    g['data']   = data

        f.close()



if __name__ == '__main__':

    pass
