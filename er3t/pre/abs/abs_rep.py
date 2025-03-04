import os
import sys
import glob
import pickle
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate

import er3t.common
import er3t.util
from .util import *



__all__ = ['abs_rep']



class abs_rep:

    """
    This module is based on the REPTRAN database developed by Gasteiger et al., 2014.
    Database can be downloaded at http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:reptran_2017_all.tar.gz

    Input:
        wavelength: wavelength in nm
        fname     : file path for the absorption coefficients (in pickle format)
        atm_obj   : atmosphere object, e.g., atm_obj = atm_atmmod(levels=np.arange(21))

    Output:
        self.wvl
        self.wvl_info
        self.coef['wavelength']
        self.coef['abso_coef']
        self.coef['slit_func']
        self.coef['solar']
        self.coef['weight']
    """

    fdir_data = '%s/reptran' % er3t.common.fdir_data_abs
    reference = '\nREPTRAN (Gasteiger et al., 2014):\n- Gasteiger, J., Emde, C., Mayer, B., Buras, R., Buehler, S. A., and Lemke, O.: Representative wavelengths absorption parameterization applied to satellite channels and spectral bands, J. Quant. Spectrosc. Radiat. Transf., 148, 99-115, https://doi.org/10.1016/j.jqsrt.2014.06.024, 2014.'

    def __init__(
            self,\
            wavelength=er3t.common.params['wavelength'],\
            target='fine',  \
            fname=None,     \
            atm_obj=None,   \
            band_name=None, \
            slit_func=None, \
            overwrite=False,\
            verbose=er3t.common.params['verbose'],\
            ):

        er3t.util.add_reference(self.reference)

        if wavelength < 5025.0:
            source = 'solar'
        else:
            source = 'thermal'

        self.wvl       = wavelength
        self.nwl       = 1
        self.target    = target.lower()
        self.source    = source.lower()
        self.atm_obj   = atm_obj
        self.slit_func = slit_func
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((wavelength is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((wavelength is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(wavelength, band_name=band_name)
            self.dump(fname)

        elif ((wavelength is not None) and (fname is None)):

            self.run(wavelength)

        else:

            msg = '\nError [abs_rep]: Please provide <wavelength> to proceed.'
            raise OSError(msg)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'coef'):
                if self.verbose:
                    msg = 'Message [abs_rep]: Loading <%s> ...' % fname
                    print(msg)
                self.fname = obj.fname
                self.wvl   = obj.wvl
                self.nwl   = obj.nwl
                self.Ng    = obj.Ng
                self.coef  = obj.coef
                self.wvl_info = obj.wvl_info
            else:
                msg = '\nError [abs_rep]: <%s> is not the correct pickle file to load.' % fname
                raise OSError(msg)


    def run(self, wavelength, band_name=None):

        if not os.path.exists(self.fdir_data):
            msg = '\nError [abs_rep]: Missing REPTRAN database.'
            raise OSError(msg)

        self.load_main(wavelength, band_name=band_name)
        self.cal_coef()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                msg = 'Message [abs_rep]: Saving object into <%s> ...' % fname
                print(msg)
            pickle.dump(self, f)


    def load_main(self, wavelength, band_name=None):

        f0 = Dataset('%s/reptran_%s_%s.cdf' % (self.fdir_data, self.source, self.target), 'r')

        # read out band names
        #╭────────────────────────────────────────────────────────────────────────────╮#
        band_bytes = f0.variables['band_name'][:]
        Nband, Nchar = band_bytes.shape
        bands = [band.decode('utf-8').replace(' ', '') for band in band_bytes.view('S%d' % Nchar).ravel()]
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # select band
        # if <band_name=> is specified, use <band_name=>
        # if <band_name=> is not specified, fall back to <wavelength=>
        #╭────────────────────────────────────────────────────────────────────────────╮#
        wvl_min = f0.variables['wvlmin'][:]
        wvl_max = f0.variables['wvlmax'][:]

        if band_name is not None:
            if band_name not in bands:
                bands_info = '\n'.join(bands)
                msg = '\nError [abs_rep]: <band_name=\'%s\'> is invalid, please specify one from the following \n%s' % (band_name, bands_info)
                raise OSError(msg)
            else:
                index_band = bands.index(band_name)
                self.band_name  = band_name
                self.run_reptran = True
        else:

            logic = (wavelength>=wvl_min) & (wavelength<wvl_max)
            indices = np.where(logic)[0]
            N_ = logic.sum()
            if N_ == 0:
                msg = '\nError [abs_rep]: %.4f nm is outside REPTRAN-%s-%s supported wavelength range.' % (wavelength, self.source, self.target)
                raise OSError(msg)
            elif N_ > 1:
                bands_ = [bands[i] for i in indices]
                bands_info_ = '\n'.join(bands_)
                msg = '\nError [abs_rep]: found more than one band matching the wavelength criteria, please specify one from the following at <band_name>\n%s\nfor example, <band_name=\'%s\'>' % (bands_info_, bands_[0])
                raise OSError(msg)
            elif N_ == 1:
                index_band = indices[0]
                self.band_name  = bands[index_band]
                self.run_reptran = True
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read out gases
        #╭────────────────────────────────────────────────────────────────────────────╮#
        gas_bytes = f0.variables['species_name'][:]
        Ngas, Nchar = gas_bytes.shape
        gases = [gas.decode('utf-8').replace(' ', '') for gas in gas_bytes.view('S%d' % Nchar).ravel()]
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # get band information
        #╭────────────────────────────────────────────────────────────────────────────╮#
        wvl_min0 = f0.variables['wvlmin'][:][index_band]
        wvl_max0 = f0.variables['wvlmax'][:][index_band]
        wvl_int0 = f0.variables['wvl_integral'][:][index_band]
        avg_err0 = f0.variables['avg_error'][:][index_band]
        nwvl0    = f0.variables['nwvl_in_band'][:][index_band]
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # get representative wavelength information
        #╭────────────────────────────────────────────────────────────────────────────╮#
        wvl_indices0 = f0.variables['iwvl'][:][:, index_band]
        wvl_indices = wvl_indices0[wvl_indices0>0] - 1
        wvl_weights0 = f0.variables['iwvl_weight'][:][:, index_band]
        wvl_weights = wvl_weights0[wvl_weights0>0]

        # this is actually number of wavelength, use Ng for consistency
        self.Ng = wvl_weights.size

        wvl = f0.variables['wvl'][:][wvl_indices]
        sol = f0.variables['extra'][:][wvl_indices] / 1000.0 # convert units to Wm^-2nm^-1

        gas_indices = np.where(np.sum(f0.variables['cross_section_source'][:][wvl_indices, :], axis=0)>0)[0]

        self.wvl_all = f0.variables['wvl'][:].data
        self.wvl_  = wvl
        self.sol   = sol
        self.wgt   = wvl_weights
        self.gases = [gases[index] for index in gas_indices]
        #╰────────────────────────────────────────────────────────────────────────────╯#

        f0.close()


    def cal_coef(self, logp=False):

        Nz = self.atm_obj.lay['altitude']['data'].size
        Ng = self.Ng

        if logp:
            p_ = np.log(self.atm_obj.lay['pressure']['data']*100.0) # from hPa to Pa (log-scale pressure)
        else:
            p_ = self.atm_obj.lay['pressure']['data']*100.0 # from hPa to Pa

        t_ = self.atm_obj.lay['temperature']['data']

        self.coef = {}

        self.coef['wvl'] = {
                'name': 'Wavelength',
                'units': 'nm',
                'data': self.wvl_.data
                }

        # this is actually number of wavelength, use Ng for consistency
        self.coef['weight'] = {
                'name': 'Weight (Ng)',
                'data': self.wgt.data
                }

        self.coef['solar'] = {
                'name': 'Solar Factor (Ng)',
                'data': self.sol.data
                }

        # sol0 = self.sol.data.copy()
        # print('+'*20)
        # print(self.wvl)
        # print(sol0)
        # if (self.source == 'solar') and (self.target in ['fine', 'medium', 'coarse']):
        #     self.coef['solar']['data'][...] = cal_solar_kurudz(self.wvl, slit_func=self.slit_func)
        # else:
        #     self.coef['solar']['data'][...] = np.sum(self.sol.data*self.wgt.data)
        # sol1 = self.coef['solar']['data']
        # print(sol1)
        # print()
        # print(self.wgt.data)
        # sol0_ = np.sum(sol0*self.wgt.data)
        # sol1_ = np.sum(sol1*self.wgt.data)
        # print(sol0_)
        # print(sol1_)
        # print((sol1_-sol0_)/sol0_ * 100.0)
        # print()
        # print('-'*20)
        # print()

        self.coef['slit_func'] = {
                'name': 'Slit Function (Nz, Ng)',
                'data': np.ones((Nz, Ng), dtype=np.float64)
                }

        self.coef['abso_coef'] = {
                'name':'Absorption Coefficient (Nz, Ng)',
                'data': np.zeros((Nz, Ng), dtype=np.float64)
                }

        gases = self.gases.copy()

        for i, wvl0 in enumerate(self.wvl_):

            if (wvl0 >= 116.0) & (wvl0 <= 850.0):
                xsec = cal_xsec_o3_molina(wvl0, self.atm_obj.lay['temperature']['data'])
                abso_coef0 = xsec * self.atm_obj.lay['o3']['data'] * 1e5 * self.atm_obj.lay['thickness']['data']
                abso_coef0[abso_coef0<0.0] = 0.0
                self.coef['abso_coef']['data'][:, i] += abso_coef0

                gases.append('O3')

            if (wvl0 >= 301.4) & (wvl0 <= 1338.2):
                xsec = cal_xsec_o4_greenblatt(wvl0)
                abso_coef0 = xsec * self.atm_obj.lay['o2']['data'] * 1e-41 * self.atm_obj.lay['thickness']['data']
                abso_coef0[abso_coef0<0.0] = 0.0
                self.coef['abso_coef']['data'][:, i] += abso_coef0

                gases.append('O4')

            if (wvl0 >= 230.91383) & (wvl0 <= 794.04565):
                xsec = cal_xsec_no2_burrows(wvl0)
                abso_coef0 = xsec * self.atm_obj.lay['no2']['data'] * 1e5 * self.atm_obj.lay['thickness']['data']
                abso_coef0[abso_coef0<0.0] = 0.0
                self.coef['abso_coef']['data'][:, i] += abso_coef0

                gases.append('NO2')

            if self.run_reptran:

                for gas_type in self.gases:

                    if gas_type.lower() in self.atm_obj.lay.keys():

                        f0 = Dataset('%s/reptran_%s_%s.lookup.%s.cdf' % (self.fdir_data, self.source, self.target, gas_type), 'r')
                        xsec     = np.squeeze(f0.variables['xsec'][:])
                        t_ref    = f0.variables['t_ref'][:]
                        dt_ref   = f0.variables['t_pert'][:]
                        vmr_ref  = f0.variables['vmrs'][:]
                        wvl_ref  = f0.variables['wvl'][:]
                        if logp:
                            p_ref = np.log(f0.variables['pressure'][:]) # log-scale pressure
                        else:
                            p_ref = f0.variables['pressure'][:]
                        f0.close()

                        i_sort_p = np.argsort(p_ref)
                        dt_ = t_ - np.interp(p_, p_ref[i_sort_p], t_ref[i_sort_p])

                        iwvl = np.argmin(np.abs(wvl_ref-wvl0))

                        # for water vapor (H2O)
                        #╭──────────────────────────────────────────────────────────────╮#
                        if xsec.ndim == 4:
                            points = (dt_ref, vmr_ref, p_ref[i_sort_p])
                            f_interp = interpolate.RegularGridInterpolator(points, xsec[:, :, iwvl, i_sort_p])

                            # vmr_ = np.log(self.atm_obj.lay['h2o']['data'] / self.atm_obj.lay['factor']['data'])
                            vmr_ = self.atm_obj.lay['h2o']['data'] / self.atm_obj.lay['factor']['data']
                            
                            if vmr_.max() > vmr_ref.max():
                                vmr_[vmr_>vmr_ref.max()] = vmr_ref.max()
                            
                            f_points = np.transpose(np.vstack((dt_, vmr_, p_)))
                        #╰──────────────────────────────────────────────────────────────╯#

                        # for other gases (CH4, CO2, CO, N2, N2O, O2, O3)
                        #╭──────────────────────────────────────────────────────────────╮#
                        else:
                            points = (dt_ref, p_ref[i_sort_p])
                            f_interp = interpolate.RegularGridInterpolator(points, xsec[:, iwvl, i_sort_p])

                            f_points = np.transpose(np.vstack((dt_, p_)))
                        #╰──────────────────────────────────────────────────────────────╯#

                        # from <libRadtran>/src/molecular.c: <The lookup table file contains the cross sections in units of 10^(-20)m^2; here we need cm^2, thus we multiply with 10^(-16)>
                        # first factor: 10^(-16) for converting units from m^-2 to cm^-2
                        # second factor: 10^(5) for converting km to cm by timing layer thickness <final output is absorption optical depth>
                        # thus 10^(-11) as scale factor
                        abso_coef0 = f_interp(f_points) * self.atm_obj.lay[gas_type.lower()]['data'] * 1e-11 * self.atm_obj.lay['thickness']['data']
                        abso_coef0[abso_coef0<0.0] = 0.0

                        self.coef['abso_coef']['data'][:, i] += abso_coef0

        self.gases = gases
        self.wvl_info = '%.2f nm (REPTRAN [Nwvl=%d|%s])' % (self.wvl, self.wvl_.size, ','.join(self.gases))



if __name__ == '__main__':

    pass
