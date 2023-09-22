import os
import sys
import glob
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate

import er3t.common


from .util import *



__all__ = ['abs_rep']



class abs_rep:

    fdir_data = '%s/reptran' % er3t.common.fdir_data_abs
    reference = 'Gasteiger, J., Emde, C., Mayer, B., Buras, R., Buehler,  S.A., and Lemke, O.: Representative wavelengths absorption parameterization applied to satellite channels and spectral bands, J. Quant. Spectrosc. Radiat. Transf., 148, 99-115, https://doi.org/10.1016/j.jqsrt.2014.06.024, 2014.'

    def __init__(
            self,
            wavelength=650.0,
            target='coarse',
            atm_obj=None,
            band_name=None,
            slit_func=None,
            ):

        if wavelength < 5025.0:
            source = 'solar'
        else:
            source = 'thermal'

        self.target     = target.lower()
        self.source     = source.lower()
        self.atm_obj    = atm_obj
        self.slit_func  = slit_func

        self.load_main(wavelength, band_name=band_name)

        self.cal_coef()

    def load_main(self, wavelength, band_name=None):

        f0 = Dataset('%s/reptran_%s_%s.cdf' % (self.fdir_data, self.source, self.target), 'r')

        # read out band names
        #/----------------------------------------------------------------------------\#
        band_bytes = f0.variables['band_name'][:]
        Nband, Nchar = band_bytes.shape
        bands = [band.decode('utf-8').replace(' ', '') for band in band_bytes.view('S%d' % Nchar).ravel()]
        #\----------------------------------------------------------------------------/#


        # select band
        # if <band_name=> is specified, use <band_name=>
        # if <band_name=> is not specified, fall back to <wavelength=>
        #/----------------------------------------------------------------------------\#
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
        else:

            logic = (wavelength>wvl_min) & (wavelength<wvl_max)
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
        #\----------------------------------------------------------------------------/#

        # read out gases
        #/----------------------------------------------------------------------------\#
        gas_bytes = f0.variables['species_name'][:]
        Ngas, Nchar = gas_bytes.shape
        gases = [gas.decode('utf-8').replace(' ', '') for gas in gas_bytes.view('S%d' % Nchar).ravel()]
        #\----------------------------------------------------------------------------/#


        # get band information
        #/----------------------------------------------------------------------------\#
        wvl_min0 = f0.variables['wvlmin'][:][index_band]
        wvl_max0 = f0.variables['wvlmax'][:][index_band]
        wvl_int0 = f0.variables['wvl_integral'][:][index_band]
        avg_err0 = f0.variables['avg_error'][:][index_band]
        nwvl0    = f0.variables['nwvl_in_band'][:][index_band]
        #\----------------------------------------------------------------------------/#

        # get representative wavelength information
        #/----------------------------------------------------------------------------\#
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
        self.wvl   = wavelength
        self.wvl_  = wvl
        self.sol   = sol
        self.wgt   = wvl_weights
        self.gases = [gases[index] for index in gas_indices]

        self.wvl_info  = '%.2f nm (REPTRAN [Nwvl=%d|%s])' % (wavelength, wvl.size, ','.join(self.gases))
        #\----------------------------------------------------------------------------/#

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

        self.coef['solar'] = {
                'name': 'Solar Factor (Ng)',
                'data': self.sol.data
                }

        # this is actually number of wavelength, use Ng for consistency
        self.coef['weight'] = {
                'name': 'Weight (Ng)',
                'data': self.wgt.data
                }

        self.coef['slit_func'] = {
                'name': 'Slit Function (Nz, Ng)',
                'data': np.ones((Nz, Ng), dtype=np.float64)
                }

        self.coef['abso_coef'] = {
                'name':'Absorption Coefficient (Nz, Ng)',
                'data': np.zeros((Nz, Ng), dtype=np.float64)
                }


        for i, wvl0 in enumerate(self.wvl_):

            print(self.coef['solar']['data'])
            if self.source == 'solar':
                self.coef['solar']['data'][i] = cal_solar_kurudz(wvl0, slit_func=self.slit_func)
            print(self.coef['solar']['data'])

            if (wvl0 >= 116.0) & (wvl0 <= 850.0):
                xsec = cal_xsec_o3_molina(wvl0, self.atm_obj.lay['temperature']['data'])
                abso_coef0 = xsec * self.atm_obj.lay['o3']['data'] * 1e5 * self.atm_obj.lay['thickness']['data']
                abso_coef0[abso_coef0<0.0] = 0.0
                self.coef['abso_coef']['data'][:, i] += abso_coef0

            if (wvl0 >= 301.4) & (wvl0 <= 1338.2):
                xsec = cal_xsec_o4_greenblatt(wvl0)
                abso_coef0 = xsec * self.atm_obj.lay['o2']['data'] * 1e-41 * self.atm_obj.lay['thickness']['data']
                abso_coef0[abso_coef0<0.0] = 0.0
                self.coef['abso_coef']['data'][:, i] += abso_coef0

            if (wvl0 >= 230.91383) & (wvl0 <= 794.04565):
                xsec = cal_xsec_no2_burrows(wvl0)
                abso_coef0 = xsec * self.atm_obj.lay['no2']['data'] * 1e5 * self.atm_obj.lay['thickness']['data']
                abso_coef0[abso_coef0<0.0] = 0.0
                self.coef['abso_coef']['data'][:, i] += abso_coef0

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
                        #/--------------------------------------------------------------\#
                        if xsec.ndim == 4:
                            points = (dt_ref, vmr_ref, p_ref[i_sort_p])
                            f_interp = interpolate.RegularGridInterpolator(points, xsec[:, :, iwvl, i_sort_p])

                            # vmr_ = np.log(self.atm_obj.lay['h2o']['data'] / self.atm_obj.lay['factor']['data'])
                            vmr_ = self.atm_obj.lay['h2o']['data'] / self.atm_obj.lay['factor']['data']
                            f_points = np.transpose(np.vstack((dt_, vmr_, p_)))
                        #\--------------------------------------------------------------/#

                        # for other gases (CH4, CO2, CO, N2, N2O, O2, O3)
                        #/--------------------------------------------------------------\#
                        else:
                            points = (dt_ref, p_ref[i_sort_p])
                            f_interp = interpolate.RegularGridInterpolator(points, xsec[:, iwvl, i_sort_p])

                            f_points = np.transpose(np.vstack((dt_, p_)))
                        #\--------------------------------------------------------------/#

                        # from <libRadtran>/src/molecular.c: <The lookup table file contains the cross sections in units of 10^(-20)m^2; here we need cm^2, thus we multiply with 10^(-16)>
                        # first factor: 10^(-16) for converting units from m^-2 to cm^-2
                        # second factor: 10^(5) for converting km to cm by timing layer thickness <final output is absorption optical depth>
                        # thus 10^(-11) as scale factor
                        abso_coef0 = f_interp(f_points) * self.atm_obj.lay[gas_type.lower()]['data'] * 1e-11 * self.atm_obj.lay['thickness']['data']
                        abso_coef0[abso_coef0<0.0] = 0.0

                        self.coef['abso_coef']['data'][:, i] += abso_coef0




if __name__ == '__main__':

    pass
