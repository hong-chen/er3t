import numpy as np
from scipy import interpolate


import er3t.common


__all__ = [
        'cal_xsec_o3_molina',
        'cal_xsec_o4_greenblatt',
        'cal_xsec_no2_burrows',
        'cal_solar_kurudz',
        ]



def cal_xsec_o3_molina(wvl0, t, t_ref=273.13, fname='%s/crs/crs_o3_mol_cf.dat' % er3t.common.fdir_data_abs):

    data_ = np.loadtxt(fname)

    if (wvl0 <= data_[5, 0]) | (wvl0 >= data_[-5, 0]):
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='linear')
        f1 = interpolate.interp1d(data_[:, 0], data_[:, 2], kind='linear')
        f2 = interpolate.interp1d(data_[:, 0], data_[:, 3], kind='linear')
    else:
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='cubic')
        f1 = interpolate.interp1d(data_[:, 0], data_[:, 2], kind='cubic')
        f2 = interpolate.interp1d(data_[:, 0], data_[:, 3], kind='cubic')

    c0 = f0(wvl0)
    c1 = f1(wvl0)
    c2 = f2(wvl0)

    sigma = 1e-20 * (c0 + c1*(t-t_ref) + c2*(t-t_ref)**2)

    return sigma



def cal_xsec_o4_greenblatt(wvl0, fname='%s/crs/crs_o4_greenblatt.dat' % er3t.common.fdir_data_abs):

    data_ = np.loadtxt(fname)

    if (wvl0 <= data_[5, 0]) | (wvl0 >= data_[-5, 0]):
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='linear')
    else:
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='cubic')
    c0 = f0(wvl0)

    sigma = 1e-20 * c0

    return sigma



def cal_xsec_no2_burrows(wvl0, fname='%s/crs/crs_no2_gom.dat' % er3t.common.fdir_data_abs):

    data_ = np.loadtxt(fname)

    if (wvl0 <= data_[5, 0]) | (wvl0 >= data_[-5, 0]):
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='linear')
    else:
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='cubic')
    c0 = f0(wvl0)

    sigma = c0 * 1.0

    return sigma



def cal_solar_kurudz(wvl0, slit_func=None, kurudz_file='%s/kurudz_0.1nm.dat' % er3t.common.fdir_data_solar):

    data_= np.loadtxt(kurudz_file)
    data_[:, 1] /= 1000.0

    if (wvl0 <= data_[5, 0]) | (wvl0 >= data_[-5, 0]):
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='linear')
    else:
        f0 = interpolate.interp1d(data_[:, 0], data_[:, 1], kind='cubic')

    if slit_func is None:
        c0 = f0(wvl0)
    else:
        wvls = wvl0 + slit_func['wavelength']['data']
        c0 = np.average(f0(wvls), weights=slit_func['weight']['data'])

    return c0



if __name__ == '__main__':

     pass
