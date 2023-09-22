import numpy as np
from scipy import interpolate


import er3t.common
import er3t.util


__all__ = ['cal_num_den_from_rel_hum', 'cal_num_den_from_mix_rat']



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



if __name__ == '__main__':

    pass
