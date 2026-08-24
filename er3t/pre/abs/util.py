import numpy as np
from scipy import interpolate
import h5py
import os


import er3t.common
import er3t.util


__all__ = [
        'cal_xsec_o3_molina',
        'cal_xsec2_o4_greenblatt',
        'cal_xsec_no2_burrows',
        'cal_solar_kurudz',
        ]



def interp1d(x0, x, y, slit_func=None, method='auto', cubic_range=5):

    if slit_func is not None:

        x0_ = x0 + slit_func['wavelength']['data']

        if method == 'auto':
            if (x0_.max() <= x[cubic_range]) | (x0_.min() >= x[-cubic_range]):
                f0 = interpolate.interp1d(x, y, kind='linear', fill_value=0.0)
            else:
                f0 = interpolate.interp1d(x, y, kind='cubic', fill_value=0.0)
        else:
            f0 = interpolate.interp1d(x, y, kind=method, fill_value=0.0)

    else:

        if method == 'auto':
            if (x0 <= x[cubic_range]) | (x0 >= x[-cubic_range]):
                f0 = interpolate.interp1d(x, y, kind='linear')
            else:
                f0 = interpolate.interp1d(x, y, kind='cubic')
        else:
            f0 = interpolate.interp1d(x, y, kind=method)

    return f0



def cal_xsec_o3_molina(wvl0, t, t_ref=273.13, slit_func=None, method='auto', fname='%s/crs/crs_o3_mol_cf.dat' % er3t.common.fdir_data_abs):

    reference = '\nO₃ Absorption Cross Section (Molina and Molina, 1986):\n- Molina, L. T. and Molina, M. J.: Absolute Absorption Cross Sections of Ozone in the 185- to 350-nm Wavelength Range, J. Geophys. Res.-Atmos., 91, 4719, https://doi.org/10.1029/JD091iD13p14501, 1986.'
    er3t.util.add_reference(reference)

    data_ = np.loadtxt(fname)

    f0 = interp1d(wvl0, data_[:, 0], data_[:, 1], slit_func=slit_func, method=method)
    f1 = interp1d(wvl0, data_[:, 0], data_[:, 2], slit_func=slit_func, method=method)
    f2 = interp1d(wvl0, data_[:, 0], data_[:, 3], slit_func=slit_func, method=method)

    c0 = f0(wvl0)
    c1 = f1(wvl0)
    c2 = f2(wvl0)

    # units: cm^2 x molecule^-1 (1.0e-20 is a scale factor)
    sigma = 1.0e-20 * (c0 + c1*(t-t_ref) + c2*(t-t_ref)**2)

    return sigma



def cal_xsec2_o4_greenblatt(wvl0, slit_func=None, method='auto', fname='%s/crs/crs_o4_greenblatt.dat' % er3t.common.fdir_data_abs):

    reference = '\nO₂-O₂ Absorption Cross Section (Greenblatt et al., 1990):\n- Greenblatt, G. D., Orlando, J., Burkholder, J. B., and Ravishankara, A. R.: Absorption measurements of oxygen between 330 and 1140 nm, J. Geophys. Res., 95, 18577–18582, https://doi.org/10.1029/JD095iD11p18577, 1990.'
    er3t.util.add_reference(reference)

    data_ = np.loadtxt(fname)

    f0 = interp1d(wvl0, data_[:, 0], data_[:, 1], slit_func=slit_func, method=method)
    c0 = f0(wvl0)

    # units: cm^5 x molecule^-2 (1.0e-46 is a scale factor)
    sigma2 = 1.0e-46 * c0

    return sigma2



def cal_xsec_no2_burrows(wvl0, slit_func=None, method='auto', fname='%s/crs/crs_no2_gom.dat' % er3t.common.fdir_data_abs):

    reference = '\nNO₂ Absorption Cross Section (Burrows et al., 1998):\n- Burrows, J. P., Dehn, A., Deters, B., Himmelmann, S., Richter, A., Voigt, S., and Orphal, J.: Atmospheric remote-sensing reference data from GOME: 1. Temperature-dependent absorption cross-sections of NO2 in the 231–794 nm range, J. Quant. Spectrosc. Ra., 60, 1025–1031, https://doi.org/10.1016/S0022-4073(97)00197-0, 1998.'
    er3t.util.add_reference(reference)

    data_ = np.loadtxt(fname)

    f0 = interp1d(wvl0, data_[:, 0], data_[:, 1], slit_func=slit_func, method=method)
    c0 = f0(wvl0)

    # units: cm^2 x molecule^-1 (no scale factor needed)
    sigma = c0 * 1.0

    return sigma



def cal_solar_kurudz(wvl0, slit_func=None, method='auto', kurudz_file='%s/kurudz_0.1nm.dat' % er3t.common.fdir_data_solar):

    reference = '\nKurucz Solar Spectrum (Kurucz, 1992):\n- Kurucz, R. L.: Synthetic infrared spectra, in: Proceedings of the 154th Symposium of the International Astronomical Union (IAU), Tucson, Arizona, 2–6 March 1992, Kluwer, Acad., Norwell, MA, 154, 523–531, https://doi.org/10.1017/S0074180900124805, 1992.'
    er3t.util.add_reference(reference)

    data_= np.loadtxt(kurudz_file)
    data_[:, 1] /= 1000.0

    f0 = interp1d(wvl0, data_[:, 0], data_[:, 1], slit_func=slit_func, method=method)

    if slit_func is None:
        c0 = f0(wvl0)
    else:
        wvls = wvl0 + slit_func['wavelength']['data']
        c0 = np.average(f0(wvls), weights=slit_func['weight']['data'])

    return c0



def gen_h5_abs_ssfr(fname_h5):

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
