import os
import sys
import glob
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate

import er3t.common



__all__ = ['abs_rep']



class abs_rep:

    fdir_data = '%s/reptran' % er3t.common.fdir_data_abs
    reference = 'Gasteiger, J., Emde, C., Mayer, B., Buras, R., Buehler,  S.A., and Lemke, O.: Representative wavelengths absorption parameterization applied to satellite channels and spectral bands, J. Quant. Spectrosc. Radiat. Transf., 148, 99-115, https://doi.org/10.1016/j.jqsrt.2014.06.024, 2014.'

    def __init__(
            self,
            wavelength=650.0,
            target='MODIS',
            atm_obj=None,
            band_name=None,
            band='B01',
            ):

        if wavelength < 5025.0:
            source = 'solar'
        else:
            source = 'thermal'

        self.target     = target.lower()
        self.source     = source.lower()
        self.atm_obj    = atm_obj

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
        self.wvl   = wvl
        self.sol   = sol
        self.wgt   = wvl_weights
        self.gases = [gases[index] for index in gas_indices]
        #\----------------------------------------------------------------------------/#

        f0.close()

    def cal_coef(self):

        Nz = self.atm_obj.lay['altitude']['data'].size
        Ng = self.Ng

        self.coef = {}

        self.coef['wvl'] = {
                'name': 'Wavelength',
                'units': 'nm',
                'data': self.wvl.data
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


        abso_coef = np.zeros((Nz, Ng), dtype=np.float64)

        for gas_type in self.gases:

            if gas_type.lower() in self.atm_obj.lay.keys():

                f0 = Dataset('%s/reptran_%s_%s.lookup.%s.cdf' % (self.fdir_data, self.source, self.target, gas_type), 'r')
                xsec     = np.squeeze(f0.variables['xsec'][:])
                t_ref    = f0.variables['t_ref'][:]
                dt_ref   = f0.variables['t_pert'][:]
                vmr_ref  = f0.variables['vmrs'][:]
                wvl_ref  = f0.variables['wvl'][:]
                # wvl_ref = self.wvl_all[f0.variables['wvl_index'][:]-1]
                logp_ref = np.log(f0.variables['pressure'][:])
                f0.close()


                logp_ = np.log(self.atm_obj.lay['pressure']['data']*100.0)
                i_sort_logp = np.argsort(logp_ref)
                dt_ = self.atm_obj.lay['temperature']['data'] - np.interp(logp_, logp_ref[i_sort_logp], t_ref[i_sort_logp])

                for i, wvl0 in enumerate(self.wvl):

                    iwvl = np.argmin(np.abs(wvl_ref-wvl0))

                    # for water vapor (H2O)
                    #/--------------------------------------------------------------\#
                    if xsec.ndim == 4:
                        points = (dt_ref, vmr_ref, logp_ref[i_sort_logp])
                        f_interp = interpolate.RegularGridInterpolator(points, xsec[:, :, iwvl, i_sort_logp])

                        vmr_ = self.atm_obj.lay['h2o']['data'] / self.atm_obj.lay['factor']['data']
                        f_points = np.transpose(np.vstack((dt_, vmr_, logp_)))
                    #\--------------------------------------------------------------/#

                    # for other gases (CH4, CO2, CO, N2, N2O, O2, O3)
                    #/--------------------------------------------------------------\#
                    else:
                        points = (dt_ref, logp_ref[i_sort_logp])
                        f_interp = interpolate.RegularGridInterpolator(points, xsec[:, iwvl, i_sort_logp])

                        f_points = np.transpose(np.vstack((dt_, logp_)))
                    #\--------------------------------------------------------------/#

                    abso_coef0 = f_interp(f_points) * self.atm_obj.lay[gas_type.lower()]['data'] * 1e-12
                    abso_coef[:, i] += abso_coef0

        self.coef['abso_coef'] = {
                'name':'Absorption Coefficient (Nz, Ng)',
                'data':abso_coef
                }



if __name__ == '__main__':

    pass
