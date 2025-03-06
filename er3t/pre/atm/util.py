import warnings
import numpy as np
from scipy import interpolate


import er3t.common
import er3t.util


__all__ = [
        'cal_num_den_from_rel_hum',\
        'cal_num_den_from_mix_rat',\
        'interp_pres_from_alt_temp',\
        'interp_ch4',\
        ]



def cal_num_den_from_rel_hum(rel_hum, temp):

    """
    Calculate water vapor number density from relative humidity
    Inputs:
        rel_hum: relative humidity, units %
        temp: temperature, units K

    Output:
        num_den: number density of water vapor
    """

    R            = 8.314                                    # J/mol/K
    Na           = 6.02297*1.0e23

    vap_pres_sat = 10.0**((7.5*temp)/(temp+237.3)+0.7858)   # mb
    vap_pres     = rel_hum/100.0 * vap_pres_sat             # mb
    num_den      = (vap_pres*100.0) / (R*(temp+273.15))     # mol/m^3
    num_den      = num_den*Na/1.0e6                         # #/cm^3

    return num_den



def cal_num_den_from_mix_rat(mix_rat, pres, temp):

    """
    Calculate water vapor number density from mixing ratio
    Inputs:
        mix_rat: mixing ratio, unitless
        pres: pressure, units hPa
        temp: temperature, units K

    Output:
        num_den: number density of water vapor
    """

    Na          = 6.02297*1.0e23
    water_vapor = mix_rat / 18.01528 * Na

    num_den = (pres*100.0)/(8.314*(temp+273.15))   # units: mol/m^3
    num_den = num_den*28.97/1000.0                 # units: kg/m^3
    num_den = num_den/1.0e6                        # /cm^3
    num_den = water_vapor*num_den                  # /cm3

    return num_den



def modify_h2o(date, tmhr_range, levels, fname_atmmod='/Users/hoch4240/Chen/soft/libradtran/v2.0.1/data/atmmod/afglss.dat'):

    """
    Old code from ARISE, incomplete
    """

    old_data = np.loadtxt(fname_atmmod)

    atm_atmmod0 = atm_atmmod(levels=levels, fname_atmmod=fname_atmmod)

    atm0    = atm_atmmod0.lev.copy()
    alt_atm = atm0['altitude']['data']

    hsk0 = read_ict_hsk(date)

    tmhr_hsk  = hsk0.data['Start_UTC']/3600.0
    logic_hsk = (tmhr_hsk>=tmhr_range[0])&(tmhr_hsk<=tmhr_range[1])

    alt_hsk   = hsk0.data['GPS_Altitude'][logic_hsk]/1000.0
    # temperature
    temp_hsk  = hsk0.data['Static_Air_Temp'][logic_hsk]+273.15
    # wvnd: water vapor number density
    wvnd_hsk  = cal_number_density_from_relative_humidity(hsk0.data['Relative_Humidity'][logic_hsk], hsk0.data['Static_Air_Temp'][logic_hsk])
    # wvnd_hsk  = cal_number_density_from_mixing_ratio(hsk0.data['Mixing_Ratio'][logic_hsk], hsk0.data['Static_Pressure'][logic_hsk], hsk0.data['Static_Air_Temp'][logic_hsk])

    logic_alt   = (alt_atm<=alt_hsk.max()) & (alt_atm>=alt_hsk.min())
    indices_alt = np.where(logic_alt)[0]

    f_interp     = interpolate.interp1d(alt_hsk, wvnd_hsk, bounds_error=False, fill_value='extrapolate', kind='linear')
    atm0['h2o']['data'][logic_alt] = f_interp(alt_atm[logic_alt])
    scale_factor = atm0['h2o']['data'][indices_alt[-1]] / atm0['h2o']['data'][indices_alt[-1]+1]
    atm0['h2o']['data'][indices_alt[-1]+1:] *= scale_factor

    f_interp  = interpolate.interp1d(alt_hsk, temp_hsk, bounds_error=False, fill_value='extrapolate', kind='linear')
    atm0['temperature']['data'][logic_alt] = f_interp(alt_atm[logic_alt])
    scale_factor = atm0['temperature']['data'][indices_alt[-1]] / atm0['temperature']['data'][indices_alt[-1]+1]
    atm0['temperature']['data'][indices_alt[-1]+1:] *= scale_factor

    atm0['h2o']['data'][0:indices_alt[0]] = wvnd_hsk[np.argmin(alt_hsk)]
    atm0['temperature']['data'][0:indices_alt[0]] = temp_hsk[np.argmin(alt_hsk)]

    air = 6.02214179e23/8.314472*atm0['pressure']['data']/atm0['temperature']['data']*1e-4
    for gas_tag in atm_atmmod0.gases:
        atm0[gas_tag]['data'] = atm0[gas_tag]['data']/atm0['factor']['data']*air
    atm0['air']['data'] = air

    new_vars = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
    new_data = np.zeros((atm0['altitude']['data'].size, 9))

    for i, vname in enumerate(new_vars):
        new_data[:, i] = atm0[vname]['data'][::-1]

    np.savetxt('ARISE_ATM_%s.txt' % date.strftime('%Y%m%d'), new_data)



def interp_pres_from_alt_temp(pres, alt, temp, alt_inp, temp_inp):

    """
    Use Barometric formula (https://en.wikipedia.org/wiki/Barometric_formula)
    to interpolate pressure from height and temperature

    Input:
        pressure: numpy array, original pressure in hPa
        altitude: numpy array, original altitude in km
        temperature: numpy array, original temperature in K
        altitude_to_interp: numpy array, altitude to be interpolate
        temperature_interp: numpy array, temperature to be interpolate

    Output:
        pn: interpolated pressure based on the input
    """

    indices = np.argsort(alt)
    h = np.float_(alt[indices])
    p = np.float_(pres[indices])
    t = np.float_(temp[indices])

    indices = np.argsort(alt_inp)
    hn = np.float_(alt_inp[indices])
    tn = np.float_(temp_inp[indices])

    n = p.size - 1
    a = 0.5*(t[1:]+t[:-1]) / (h[:-1]-h[1:]) * np.log(p[1:]/p[:-1])
    z = 0.5*(h[1:]+h[:-1])

    z0  = np.min(z) ; z1  = np.max(z)
    hn0 = np.min(hn); hn1 = np.max(hn)

    if hn0 < z0:
        a = np.hstack((a[0], a))
        z = np.hstack((hn0, z))
        if z0 - hn0 > 2.0:
            msg = '\nWarning [interp_pres_from_alt_temp]: Standard atmosphere not sufficient (lower boundary).'
            warnings.warn(msg)

    if hn1 > z1:
        a = np.hstack((a, z[n-1]))
        z = np.hstack((z, hn1))
        if hn1-z1 > 10.0:
            msg = '\nWarning [interp_pres_from_alt_temp]: Standard atmosphere not sufficient (upper boundary).'
            warnings.warn(msg)

    an = np.interp(hn, z, a)
    pn = np.zeros_like(hn)

    if hn.size == 1:
        hi = np.argmin(np.abs(hn-h))
        pn = p[hi]*np.exp(-an*(hn-h[hi])/tn)
        return pn

    for i in range(pn.size):
        hi = np.argmin(np.abs(hn[i]-h))
        pn[i] = p[hi]*np.exp(-an[i]*(hn[i]-h[hi])/tn[i])

    dp = pn[:-1] - pn[1:]
    pl = 0.5 * (pn[1:]+pn[:-1])
    zl = 0.5 * (hn[1:]+hn[:-1])

    for i in range(n-2):
        indices = (zl >= h[i]) & (zl < h[i+1])
        ind = np.where(indices==True)[0]
        ni  = indices.sum()
        if ni >= 2:
            dpm = dp[ind].sum()

            i0 = np.min(ind)
            i1 = np.max(ind)

            x1 = pl[i0]
            x2 = pl[i1]
            y1 = dp[i0]
            y2 = dp[i1]

            bb = (y2-y1) / (x2-x1)
            aa = y1 - bb*x1
            rescale = dpm / (aa+bb*pl[indices]).sum()

            if np.abs(rescale-1.0) > 0.1:
                msg = '\nWarning [interp_pres_from_alt_temp]: Pressure smoothing failed at %.1f to %.1f km, rescaled with %f ...' % (h[i], h[i+1], rescale)
                warnings.warn(msg)
            else:
                dp[indices] = rescale*(aa+bb*pl[indices])

    for i in range(dp.size):
        pn[i+1] = pn[i] - dp[i]

    return pn



def interp_ch4(alt_inp):

    """
    input:
        levels: numpy array, height in km
    output:
        ch4mix: mixing ratio of CH4
    """

    # height
    ch4h   = np.array([ 0.000000,      0.100000,      0.200000,      0.300000, \
                        0.400000,      0.500000,      1.000000,      2.000000, \
                        3.000000,      4.000000,      5.000000,      6.000000, \
                        7.000000,      8.000000,      9.000000,     10.000000, \
                       11.000000,     12.000000,     13.000000,     14.000000, \
                       15.000000,     16.000000,     17.000000,     18.000000, \
                       19.000000,     20.000000,     21.000000,     22.000000, \
                       23.000000,     24.000000,     25.000000,     27.000000, \
                       29.000000,     31.000000,     33.000000,     35.000000, \
                       37.000000,     40.000000])

    # CH4 number concentration
    ch4m   = np.array([  1.70000e-06,   1.70000e-06,   1.70000e-06,   1.70000e-06, \
                         1.70000e-06,   1.70000e-06,   1.70000e-06,   1.70000e-06, \
                         1.70000e-06,   1.70000e-06,   1.70000e-06,   1.70000e-06, \
                         1.69900e-06,   1.69700e-06,   1.69300e-06,   1.68500e-06, \
                         1.67485e-06,   1.66200e-06,   1.64753e-06,   1.62915e-06, \
                         1.60500e-06,   1.58531e-06,   1.55875e-06,   1.52100e-06, \
                         1.48145e-06,   1.42400e-06,   1.38858e-06,   1.34258e-06, \
                         1.28041e-06,   1.19173e-06,   1.05500e-06,   1.02223e-06, \
                         9.63919e-07,   9.04935e-07,   8.82387e-07,   8.48513e-07, \
                         7.91919e-07,   0.000000000])

    ch4mix = np.interp(alt_inp, ch4h, ch4m)

    return ch4mix



if __name__ == '__main__':

    pass
