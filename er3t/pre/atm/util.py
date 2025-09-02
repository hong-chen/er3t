import warnings
import numpy as np
from scipy import interpolate
import h5py
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import netCDF4 as nc
from er3t.util.modis import modis_07


import er3t.common
import er3t.util
import er3t.util.constants as constants


__all__ = [
        'cal_num_den_from_rel_hum',\
        'cal_num_den_from_mix_rat',\
        'interp_pres_from_alt_temp',\
        'interp_ch4',\
        'create_modis_dropsonde_atm',\
        'zpt_plot',\
        'zpt_plot_combine'
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


def mmr_to_vmr(mmr, gas='co2', units='ppmv'):
    """
    Convert mass mixing ratio (MMR) to volume mixing ratio (VMR).

    For O3, the formula is: VMR = 28.9644 / 47.9982 * 1e9 * MMR
    For CO the formula is: VMR = 28.9644 / 28.0101 * 1e9 * MMR
    For NO2 the formula is: VMR = 28.9644 / 46.0055 * 1e9 * MMR
    For SO2 the formula is: VMR = 28.9644 / 64.0638 * 1e9 * MMR
    For CO2 the formula is: VMR = 28.9644 / 44.0095 * 1e9 * MMR
    For CH4 the formula is: VMR = 28.9644 / 16.0425 * 1e9 * MMR
    Source: https://forum.ecmwf.int/t/convert-mass-mixing-ratio-mmr-to-mass-concentration-or-to-volume-mixing-ratio-vmr/1253
    """
    gas_species = gas.lower()
    if gas_species not in constants.molar_masses.keys():
        raise ValueError("Only {} are supported. Received unknown gas species: {}".format(constants.molar_masses.keys(), gas_species))
    if units == 'ppmv':
        return (28.9644 / constants.molar_masses[gas_species]) * 1e6 * mmr
    elif units == 'ppbv':
        return (28.9644 / constants.molar_masses[gas_species]) * 1e9 * mmr
    else:
        raise ValueError("Only `ppmv` and `ppbv` are supported as units, received: {}".format(units))


def number_density_to_volume_mixing_ratio(gas_number_density, pressure=None, temperature=None, air_number_density=None, units='ppmv'):
    """
    Convert number density to volume mixing ratio (VMR).
    Formula: VMR = ( R * T * n)  / (P * Na)
    VMR will be in units of ppmv or ppbv or dimensionless based on `units`.

    Parameters
    ----------
    gas_number_density : float
        Number density of the gas (molecules/cm^3).
    pressure : float, optional
        Total pressure (hPa).
    temperature : float, optional
        Temperature (K).
    air_number_density : float, optional
        Number density of air (molecules/cm^3). If not provided, it will be calculated using ideal gas law.
    units : str
        Desired output units ('ppmv' or 'ppbv'). Anything else will return dimensionless volume mixing ratio.

    Returns
    -------
    float
        Volume mixing ratio (VMR) in ppmv.
    """

    if air_number_density is None:
        if temperature is None or pressure is None:
            raise ValueError("Either air_number_density or both temperature and pressure must be provided.")
        # 1e4 is the unit conversion factor accounting for pressure being in hPa and number density in #/cm3
        # Calculate air number density using ideal gas law: n = P/(kB*T)
        # Convert pressure from hPa to Pa (×100) and get molecules/cm³ (÷1e6)
        air_number_density = (constants.R/constants.NA) * (temperature/pressure) * 1e4

    vmr = gas_number_density/air_number_density

    # update units if needed
    if units == 'ppbv':
        vmr = vmr * 1e9

    elif units == 'ppmv':
        vmr = vmr * 1e6

    return vmr


# def modify_h2o(date, tmhr_range, levels, fname_atmmod='/Users/hoch4240/Chen/soft/libradtran/v2.0.1/data/atmmod/afglss.dat'):

#     """
#     Old code from ARISE, incomplete
#     """

#     old_data = np.loadtxt(fname_atmmod)

#     atm_atmmod0 = atm_atmmod(levels=levels, fname_atmmod=fname_atmmod)

#     atm0    = atm_atmmod0.lev.copy()
#     alt_atm = atm0['altitude']['data']

#     hsk0 = read_ict_hsk(date)

#     tmhr_hsk  = hsk0.data['Start_UTC']/3600.0
#     logic_hsk = (tmhr_hsk>=tmhr_range[0])&(tmhr_hsk<=tmhr_range[1])

#     alt_hsk   = hsk0.data['GPS_Altitude'][logic_hsk]/1000.0
#     # temperature
#     temp_hsk  = hsk0.data['Static_Air_Temp'][logic_hsk]+273.15
#     # wvnd: water vapor number density
#     wvnd_hsk  = cal_number_density_from_relative_humidity(hsk0.data['Relative_Humidity'][logic_hsk], hsk0.data['Static_Air_Temp'][logic_hsk])
#     # wvnd_hsk  = cal_number_density_from_mixing_ratio(hsk0.data['Mixing_Ratio'][logic_hsk], hsk0.data['Static_Pressure'][logic_hsk], hsk0.data['Static_Air_Temp'][logic_hsk])

#     logic_alt   = (alt_atm<=alt_hsk.max()) & (alt_atm>=alt_hsk.min())
#     indices_alt = np.where(logic_alt)[0]

#     f_interp     = interpolate.interp1d(alt_hsk, wvnd_hsk, bounds_error=False, fill_value='extrapolate', kind='linear')
#     atm0['h2o']['data'][logic_alt] = f_interp(alt_atm[logic_alt])
#     scale_factor = atm0['h2o']['data'][indices_alt[-1]] / atm0['h2o']['data'][indices_alt[-1]+1]
#     atm0['h2o']['data'][indices_alt[-1]+1:] *= scale_factor

#     f_interp  = interpolate.interp1d(alt_hsk, temp_hsk, bounds_error=False, fill_value='extrapolate', kind='linear')
#     atm0['temperature']['data'][logic_alt] = f_interp(alt_atm[logic_alt])
#     scale_factor = atm0['temperature']['data'][indices_alt[-1]] / atm0['temperature']['data'][indices_alt[-1]+1]
#     atm0['temperature']['data'][indices_alt[-1]+1:] *= scale_factor

#     atm0['h2o']['data'][0:indices_alt[0]] = wvnd_hsk[np.argmin(alt_hsk)]
#     atm0['temperature']['data'][0:indices_alt[0]] = temp_hsk[np.argmin(alt_hsk)]

#     air = 6.02214179e23/8.314472*atm0['pressure']['data']/atm0['temperature']['data']*1e-4
#     for gas_tag in atm_atmmod0.gases:
#         atm0[gas_tag]['data'] = atm0[gas_tag]['data']/atm0['factor']['data']*air
#     atm0['air']['data'] = air

#     new_vars = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
#     new_data = np.zeros((atm0['altitude']['data'].size, 9))

#     for i, vname in enumerate(new_vars):
#         new_data[:, i] = atm0[vname]['data'][::-1]

#     np.savetxt('ARISE_ATM_%s.txt' % date.strftime('%Y%m%d'), new_data)


def parse_afgl_data_to_dataframe(afgl_fname):
    """
    Convert AFGL atmospheric data lines to a pandas DataFrame.

    Parameters
    ----------
    afgl_fname : str
        Path to the AFGL atmospheric data file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: altitude, pressure, temperature, air, o3, o2, h2o, co2, no2
        Units: km, mb, K, cm^-3, cm^-3, cm^-3, cm^-3, cm^-3, cm^-3
    """
    import pandas as pd

    with open(afgl_fname) as f0:
        afgl_data_lines = f0.readlines()

    # Extract header information and data lines
    header_lines = []
    data_lines = []

    for line in afgl_data_lines:
        line_stripped = line.strip()
        if line_stripped.startswith('#'):
            header_lines.append(line_stripped)
        elif line_stripped and not line_stripped.startswith('#'):
            # This is a data line
            data_lines.append(line_stripped)

    # Parse column names from header
    # Expected format: '#     z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)'
    column_names = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
    column_units = ['km', 'mb', 'K', 'cm^-3', 'cm^-3', 'cm^-3', 'cm^-3', 'cm^-3', 'cm^-3']

    # Parse data lines
    data_rows = []
    for line in data_lines:
        # Split by whitespace and convert to float
        values = line.split()
        if len(values) >= 9:  # Ensure we have all required columns
            try:
                # Convert scientific notation strings to float
                row_data = []
                for val in values[:9]:  # Take first 9 values
                    # Handle scientific notation like '4.307612E+11'
                    row_data.append(float(val))
                data_rows.append(row_data)
            except ValueError as e:
                print(f"Warning: Could not parse line '{line}': {e}")
                continue

    # Create DataFrame
    if data_rows:
        df = pd.DataFrame(data_rows, columns=column_names)

        # Add metadata as attributes
        df.attrs['units'] = dict(zip(column_names, column_units))
        df.attrs['description'] = 'AFGL Atmospheric Profile'
        df.attrs['header_info'] = header_lines

        # Sort by altitude (ascending)
        df = df.sort_values('altitude').reset_index(drop=True)

        return df
    else:
        raise ValueError("No valid data rows found in AFGL data")


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

    # Sort original data by altitude
    indices = np.argsort(alt)
    h = np.float64(alt[indices])
    p = np.float64(pres[indices])
    t = np.float64(temp[indices])

    # Sort interpolation data by altitude
    indices = np.argsort(alt_inp)
    hn = np.float64(alt_inp[indices])
    tn = np.float64(temp_inp[indices])

    # Calculate scale factor 'a' for barometric formula at layer midpoints
    n = p.size - 1
    a = 0.5*(t[1:]+t[:-1]) / (h[:-1]-h[1:]) * np.log(p[1:]/p[:-1])
    z = 0.5*(h[1:]+h[:-1])  # midpoint altitudes

    # Determine altitude bounds
    z0  = np.min(z) ; z1  = np.max(z)
    hn0 = np.min(hn); hn1 = np.max(hn)

    # Extend scale factor array if interpolation goes below original data
    if hn0 < z0:
        a = np.hstack((a[0], a))
        z = np.hstack((hn0, z))
        if z0 - hn0 > 2.0:
            msg = '\nWarning [interp_pres_from_alt_temp]: Standard atmosphere not sufficient (lower boundary).'
            warnings.warn(msg)

    # Extend scale factor array if interpolation goes above original data
    if hn1 > z1:
        a = np.hstack((a, z[n-1]))
        z = np.hstack((z, hn1))
        if hn1-z1 > 10.0:
            msg = '\nWarning [interp_pres_from_alt_temp]: Standard atmosphere not sufficient (upper boundary).'
            warnings.warn(msg)

    # Interpolate scale factors to target altitudes
    an = np.interp(hn, z, a)
    pn = np.zeros_like(hn)

    # Handle single point case
    if hn.size == 1:
        hi = np.argmin(np.abs(hn-h))
        pn = p[hi]*np.exp(-an*(hn-h[hi])/tn)
        return pn

    # Apply barometric formula to each target altitude
    for i in range(pn.size):
        hi = np.argmin(np.abs(hn[i]-h))  # find closest original altitude
        pn[i] = p[hi]*np.exp(-an[i]*(hn[i]-h[hi])/tn[i]) # barometric formula

    # Calculate pressure differences and layer midpoints for smoothing
    dp = pn[:-1] - pn[1:]  # pressure differences between adjacent levels
    pl = 0.5 * (pn[1:]+pn[:-1])  # pressure at layer midpoints
    zl = 0.5 * (hn[1:]+hn[:-1])  # altitude at layer midpoints

    # Smooth pressure profile by rescaling pressure differences within original layers
    for i in range(n-2):
        # Find which interpolated layers fall within this original layer
        indices = (zl >= h[i]) & (zl < h[i+1])
        ind = np.where(indices)[0]
        ni  = indices.sum()

        if ni >= 2:
            # Calculate total pressure difference in this layer
            dpm = dp[ind].sum()

            # Get boundary indices for linear interpolation
            i0 = np.min(ind)
            i1 = np.max(ind)

            # Set up linear interpolation for pressure differences
            x1 = pl[i0]
            x2 = pl[i1]
            y1 = dp[i0]
            y2 = dp[i1]

            # Calculate linear interpolation coefficients
            bb = (y2-y1) / (x2-x1)
            aa = y1 - bb*x1

            # Calculate rescaling factor to conserve total pressure difference
            rescale = dpm / (aa+bb*pl[indices]).sum()

            # Apply rescaling with warning if large correction needed
            if np.abs(rescale-1.0) > 0.1:
                msg = '\nWarning [interp_pres_from_alt_temp]: Pressure smoothing failed at %.1f to %.1f km, rescaled with %f ...' % (h[i], h[i+1], rescale)
                warnings.warn(msg)
            else:
                dp[indices] = rescale*(aa+bb*pl[indices])

    # Reconstruct final pressure profile from smoothed pressure differences
    for i in range(dp.size):
        pn[i+1] = pn[i] - dp[i]

    return pn



def interp_ch4(alt_inp):

    """
    Interpolate methane (CH4) mixing ratio as a function of altitude.

    This function provides CH4 volume mixing ratios based on a standard atmospheric
    profile that decreases with altitude from 1.7 ppmv at surface to near-zero
    at 40 km altitude.

    Parameters
    ----------
    alt_inp : array_like
        Altitude levels in kilometers above sea level where CH4 mixing ratios
        are desired. Can be a scalar or array.

    Returns
    -------
    ch4mix : ndarray
        CH4 volume mixing ratio (dimensionless) at the requested altitude levels.
        Values range from 1.7 ppmv at surface to 0 at 40+ km altitude.

    Notes
    -----
    The reference profile is based on standard atmospheric CH4 concentrations
    with linear interpolation between discrete altitude levels. The profile
    assumes constant mixing ratio in the troposphere (~1.7 ppmv) with gradual
    decrease in the stratosphere to zero at the stratopause.
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


def create_modis_dropsonde_atm(o2mix=0.20935,
                               output_dir='.',
                               output='zpt.h5',
                               fname_mod07=None,
                               dropsonde_df=None,
                               extent=None,
                               levels=None,
                               new_h_edge=None,
                               sfc_T_set=None, # in K
                               sfc_h_to_zero=True,
                               plot=True):
    """
    Use MODIS 07 product to create a vertical profile of temperature, dew temperature, pressure, O2 and H2O number density, and H2O volume mixing ratio.
    """

    if fname_mod07 is None or dropsonde_df is None:
        sys.exit("[Error] sat and dropsonde information must be provided!")
    else:
        # Get reanalysis from met and CO2 prior sounding data
        try_time = 0
        sfc_p = [np.nan]
        while try_time < 15 and np.isnan(np.nanmean(sfc_p)):
            boundary_width = 0.1
            extent = [extent[0]-boundary_width*try_time, extent[1]+boundary_width*try_time,
                      extent[2]-boundary_width*try_time, extent[3]+boundary_width*try_time]
            try_time += 1
            mod07 = modis_07(fnames=fname_mod07, extent=extent)
            cld_mask = mod07.data['cld_mask']['data']
            ### cloud mask: 0=cloudy, 1=uncetain, 2=probably clear, 3=confident clear
            sfc_p = mod07.data['p_sfc']['data']                             # surface pressure in hPa

        if np.isnan(np.nanmean(sfc_p)):
            print("cloud mask: ", cld_mask)
            return 'error', None

        # get profile data
        pprf_l_single = mod07.data['p_level']['data']            # pressure in hPa
        hprf_l = mod07.data['h_level_retrieved']['data']         # retrieved height profile in m
        tprf_l = mod07.data['T_level_retrieved']['data']         # retrieved temperature profile in K
        dewTprf_l = mod07.data['dewT_level_retrieved']['data']   # retrieved dew point temperature profile in K
        mwvmxprf_l = mod07.data['wvmx_level_retrieved']['data']  # retrieved water vapor mixing ratio profile in g/kg

        sfc_h = mod07.data['h_sfc']['data'] # surface elevation in m
        # assume surface geopotential height is the same as surface height (in m)
        skin_temp = mod07.data['t_skin']['data'] # skin temp in K
        sza = np.nanmean(mod07.data['sza']['data'])
        vza = np.nanmean(mod07.data['vza']['data'])

        # replicate pressure levels for all geographical locations
        pprf_l = np.repeat(pprf_l_single, mwvmxprf_l.shape[1]).reshape(mwvmxprf_l.shape)
        r = mwvmxprf_l/1000 # mass mixing ratio to kg/kg
        eprf_l = pprf_l*r/(constants.EPSILON+r) # vapor pressure in hPa

        # Compute virtual temperature
        Tv = tprf_l/(1 - (r/(r + constants.EPSILON)) * (1 - constants.EPSILON))

        air_layer = pprf_l*100/(constants.kb*tprf_l)/1e6  # air number density in molec/cm3
        dry_air_layer = (pprf_l-eprf_l)*100/(constants.kb*tprf_l)/1e6  # air number density in molec/cm3
        o2_layer = dry_air_layer*o2mix          # O2 number density in molec/cm3
        h2o_layer = eprf_l*100/(constants.kb*tprf_l)/1e6  # H2O number density in molec/cm3
        # h2o_vmr = h2o_layer/dry_air_layer       # H2O volume mixing ratio
        h2o_vmr = eprf_l/(pprf_l-eprf_l)       # H2O volume mixing ratio

        # Compute means in the area
        sfc_p_mean = np.nanmean(sfc_p)
        sfc_h_mean = np.nanmean(sfc_h)
        skin_temp_mean = np.nanmean(skin_temp)
        pprf_lev_mean   = np.nanmean(pprf_l, axis=1)    # pressure mid grid in hPa
        tprf_lev_mean   = np.nanmean(tprf_l, axis=1)         # temperature mid grid in K
        dewTprf_lev_mean= np.nanmean(dewTprf_l, axis=1)      # dew temperature mid grid in K
        d_o2_lev_mean   = np.nanmean(o2_layer, axis=1)
        d_h2o_lev_mean  = np.nanmean(h2o_layer, axis=1)
        hprf_lev_mean   = np.nanmean(hprf_l, axis=1)/1000     # height mid grid in km
        h2o_vmr_mean    = np.nanmean(h2o_vmr, axis=1)


        if sfc_h_to_zero:
            sfc_h_mean = 0

        # interpolate to surface
        # can change to a more physically reasonable way later
        f_temp = interp1d(pprf_lev_mean[:-1], tprf_lev_mean[:-1], fill_value='extrapolate')
        f_h2o_vmr = interp1d(pprf_lev_mean[:-1], h2o_vmr_mean[:-1], fill_value='extrapolate')
        pprf_lev_mean[-1] = sfc_p_mean
        tprf_lev_mean[-1] = f_temp(sfc_p_mean)
        hprf_lev_mean[-1] = sfc_h_mean
        h2o_vmr_mean[-1] = f_h2o_vmr(sfc_p_mean)

        if sfc_T_set is not None:
            tprf_lev_mean[-1] = sfc_T_set

        # Process dropsonde profile
        p_drop = np.array(dropsonde_df['p']) # in hPa
        alt_drop = np.array(dropsonde_df['alt']/1000) # in km
        t_dry_drop = np.array(dropsonde_df['t_dry'])
        t_dew_drop = np.array(dropsonde_df['t_dew'])
        mr_drop = np.array(dropsonde_df['h2o_mr'])
        r_drop = mr_drop/1000 # mass mixing ratio to kg/kg
        eprf_drop = p_drop*r_drop/(constants.EPSILON+r_drop)
        air_drop = p_drop*100/(constants.kb*t_dry_drop)/1e6  # air number density in molec/cm3
        o2_drop = air_drop*o2mix          # O2 number density in molec/cm3
        h2o_drop = eprf_drop*100/(constants.kb*t_dry_drop)/1e6  # H2O number density in molec/cm3
        h2o_vmr_drop = eprf_drop/(p_drop-eprf_drop)       # H2O volume mixing ratio

        # calculate 10m wind speed from dropsonde data
        ws_10m = np.array(dropsonde_df['ws'])
        ws_10m_nan_mask = np.isnan(ws_10m)
        ws10m = np.interp(0.01, alt_drop[~ws_10m_nan_mask], ws_10m[~ws_10m_nan_mask])   # calculate 10m wind speed


    if new_h_edge is not None:
        levels = new_h_edge
    else:
        if levels is not None:
            levels = np.array(levels)
        else:
            levels = np.concatenate((np.linspace(sfc_h_mean, 4.0, 11),
                                    np.arange(5.0, 10.1, 1.0),
                                    np.array([12.5, 15, 17.5, 20. , 25. , 30., 40.])))
    output_path = os.path.join(output_dir, output)
    if os.path.isfile(output_path):
        print(f'[Warning] Output file {output} exists - overwriting!')
    print('Saving to file '+output)
    with h5py.File(output_path, 'w') as h5_output:
        h5_output.create_dataset('sfc_p',       data=sfc_p_mean)
        h5_output['sfc_p'].attrs['units'] = 'hPa'
        h5_output['sfc_p'].attrs['description'] = 'Surface pressure'

        h5_output.create_dataset('sfc_h',       data=sfc_h_mean)
        h5_output['sfc_h'].attrs['units'] = 'km'
        h5_output['sfc_h'].attrs['description'] = 'Surface height above sea level'

        h5_output.create_dataset('skin_temp',   data=skin_temp_mean)
        h5_output['skin_temp'].attrs['units'] = 'K'
        h5_output['skin_temp'].attrs['description'] = 'Skin temperature from MODIS'

        h5_output.create_dataset('level_sim',      data=levels)
        h5_output['level_sim'].attrs['units'] = 'km'
        h5_output['level_sim'].attrs['description'] = 'Simulation levels height above sea level'

        h5_output.create_dataset('h_lev',       data=hprf_lev_mean)
        h5_output['h_lev'].attrs['units'] = 'km'
        h5_output['h_lev'].attrs['description'] = 'Height profile from MODIS retrievals'

        h5_output.create_dataset('p_lev',       data=pprf_lev_mean)
        h5_output['p_lev'].attrs['units'] = 'hPa'
        h5_output['p_lev'].attrs['description'] = 'Pressure profile from MODIS retrievals'

        h5_output.create_dataset('t_lev',       data=tprf_lev_mean)
        h5_output['t_lev'].attrs['units'] = 'K'
        h5_output['t_lev'].attrs['description'] = 'Temperature profile from MODIS retrievals'

        h5_output.create_dataset('dewT_lev',    data=dewTprf_lev_mean)
        h5_output['dewT_lev'].attrs['units'] = 'K'
        h5_output['dewT_lev'].attrs['description'] = 'Dew point temperature profile from MODIS retrievals'

        h5_output.create_dataset('d_o2_lev',    data=d_o2_lev_mean)
        h5_output['d_o2_lev'].attrs['units'] = 'molec/cm3'
        h5_output['d_o2_lev'].attrs['description'] = 'O2 number density profile from MODIS retrievals'

        h5_output.create_dataset('d_h2o_lev',   data=d_h2o_lev_mean)
        h5_output['d_h2o_lev'].attrs['units'] = 'molec/cm3'
        h5_output['d_h2o_lev'].attrs['description'] = 'H2O number density profile from MODIS retrievals'

        h5_output.create_dataset('h2o_vmr',     data=h2o_vmr_mean)
        h5_output['h2o_vmr'].attrs['units'] = 'dimensionless'
        h5_output['h2o_vmr'].attrs['description'] = 'H2O volume mixing ratio from MODIS retrievals'

        h5_output.create_dataset('p_drop',     data=p_drop)
        h5_output['p_drop'].attrs['units'] = 'hPa'
        h5_output['p_drop'].attrs['description'] = 'Pressure profile from dropsonde measurements'

        h5_output.create_dataset('t_dry_drop', data=t_dry_drop)
        h5_output['t_dry_drop'].attrs['units'] = 'K'
        h5_output['t_dry_drop'].attrs['description'] = 'Dry temperature profile from dropsonde measurements'

        h5_output.create_dataset('t_dew_drop', data=t_dew_drop)
        h5_output['t_dew_drop'].attrs['units'] = 'K'
        h5_output['t_dew_drop'].attrs['description'] = 'Dew point temperature profile from dropsonde measurements'

        h5_output.create_dataset('h2o_vmr_drop', data=h2o_vmr_drop)
        h5_output['h2o_vmr_drop'].attrs['units'] = 'dimensionless'
        h5_output['h2o_vmr_drop'].attrs['description'] = 'H2O volume mixing ratio from dropsonde measurements'

        h5_output.create_dataset('alt_drop', data=alt_drop)
        h5_output['alt_drop'].attrs['units'] = 'km'
        h5_output['alt_drop'].attrs['description'] = 'Altitude profile from dropsonde measurements'

        h5_output.create_dataset('air_drop', data=air_drop)
        h5_output['air_drop'].attrs['units'] = 'molec/cm3'
        h5_output['air_drop'].attrs['description'] = 'Air number density from dropsonde measurements'

        h5_output.create_dataset('o2_drop', data=o2_drop)
        h5_output['o2_drop'].attrs['units'] = 'molec/cm3'
        h5_output['o2_drop'].attrs['description'] = 'O2 number density from dropsonde measurements'

        h5_output.create_dataset('h2o_drop', data=h2o_drop)
        h5_output['h2o_drop'].attrs['units'] = 'molec/cm3'
        h5_output['h2o_drop'].attrs['description'] = 'H2O number density from dropsonde measurements'

        h5_output.create_dataset('ws10m',       data=ws10m)
        h5_output['ws10m'].attrs['units'] = 'm/s'
        h5_output['ws10m'].attrs['description'] = '10-meter wind speed interpolated from dropsonde data'

        h5_output.create_dataset('o2_mix',      data=o2mix)
        h5_output['o2_mix'].attrs['units'] = 'dimensionless'
        h5_output['o2_mix'].attrs['description'] = 'O2 mixing ratio used in calculations'

        h5_output.create_dataset('sza',         data=sza)
        h5_output['sza'].attrs['units'] = 'degrees'
        h5_output['sza'].attrs['description'] = 'Solar zenith angle from MODIS'

        h5_output.create_dataset('vza',         data=vza)
        h5_output['vza'].attrs['units'] = 'degrees'
        h5_output['vza'].attrs['description'] = 'Viewing zenith angle from MODIS'

        h5_output.create_dataset('lat',         data=np.mean(extent[2:]))
        h5_output['lat'].attrs['units'] = 'degrees_north'
        h5_output['lat'].attrs['description'] = 'Mean latitude of the study area'

    # zpt_plot(pprf_lev_mean, tprf_lev_mean, dewTprf_lev_mean, h2o_vmr_mean, output=f"{output_dir}/{output.replace('.h5', '.png')}")
    # zpt_plot(p_drop, t_dry_drop, t_dew_drop, h2o_vmr_drop, output=f"{output_dir}/{output.replace('.h5', '_dropsonde.png')}")
    if plot:
        zpt_plot_combine(pprf_lev_mean, tprf_lev_mean, dewTprf_lev_mean, h2o_vmr_mean,
                        p_drop, t_dry_drop, t_dew_drop, h2o_vmr_drop,
                        output=os.path.join(output_dir, output.replace('.h5', '_modis_dropsonde.png')))

    return 'success', ws10m

def h2o_vmr_axis_setting(ax, pmin=100, pmax=1000):

    ax.set_ylim(pmax, pmin)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.set_ylabel('Pressure (hPa)', fontsize=14)
    ax.set_xlabel('H$_2$O mixing ratio', fontsize=14, color='b')

def vmr_axis_setting(ax, xlabel, xlabel_color, pmin=100, pmax=1000):

    ax.set_ylim(pmax, pmin)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.set_ylabel('Pressure (hPa)', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14, color=xlabel_color)

def zpt_plot(p_lay, t_lay, dewT_lay, h2o_vmr, output, pmin=100, pmax=1000):
    from metpy.plots import SkewT
    from metpy.units import units

    p_prf = p_lay * units.hPa
    T_prf = (t_lay * units.kelvin).to(units.degC)
    Td_prf = (dewT_lay * units.kelvin).to(units.degC)

    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(12, 6.75))
    ax1.set_visible(False)
    skew = SkewT(fig=fig, subplot=(1, 2, 1), aspect=120.5)
    skew.plot(p_prf, T_prf, 'r', label='Temperature', linewidth=3)
    skew.plot(p_prf, Td_prf, 'g', label='Dew Point', linewidth=3)
    skew.ax.set_xlabel('Temperature ($\N{DEGREE CELSIUS}$)', fontsize=14)
    skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)
    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_ylim(pmax, pmin)
    skew.ax.legend()
    skew.ax.text(0.02, 1.07, '(a)', transform=skew.ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='left')


    ax2.plot(h2o_vmr, p_prf, 'b:', label='H$_2$O', linewidth=3)
    h2o_vmr_axis_setting(ax2, pmin=pmin, pmax=pmax)
    ax2.legend()
    ax2.text(1.28, 1.07, '(b)', transform=skew.ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='left')
    ax2.grid(which='minor', axis='y', linestyle='-', linewidth=1, color='lightgrey')
    fig.tight_layout()
    fig.savefig(output, dpi=300)

def zpt_plot_gases(p_lay, h2o_vmr, co2_vmr, o3_vmr,
                   output, pmin=100, pmax=1000):
    from metpy.plots import SkewT
    from metpy.units import units

    p_prf = p_lay * units.hPa

    fig, (ax2, ax3) =plt.subplots(1, 2, figsize=(12, 6.75))



    ax2.plot(h2o_vmr, p_prf, 'b:', label='H$_2$O', linewidth=3)
    h2o_vmr_axis_setting(ax2, pmin=pmin, pmax=pmax)
    ax2.legend()
    ax2.text(1.28, 1.07, '(a)', fontsize=16, fontweight='bold', va='center', ha='left')
    ax2.grid(which='minor', axis='y', linestyle='-', linewidth=1, color='lightgrey')

    ax4 = ax3.twiny()
    ax3_line = ax3.plot(co2_vmr*1e6, p_prf, 'orange', label='CO$_2$', linewidth=3)
    ax4_line = ax4.plot(o3_vmr*1e6, p_prf, 'purple', label='O$_3$', linewidth=3)
    ax3.set_xlabel('CO$_2$ mixing ratio', fontsize=14, color='orange')
    ax4.set_xlabel('O$_3$ mixing ratio', fontsize=14, color='purple')
    vmr_axis_setting(ax3, 'CO$_2$ mixing ratio (ppbv)', 'orange', pmin=pmin, pmax=pmax)
    vmr_axis_setting(ax4, 'O$_3$ mixing ratio (ppbv)', 'purple', pmin=pmin, pmax=pmax)
    legends = ax3_line + ax4_line
    labels = [i.get_label() for i in legends]
    ax3.legend(legends, labels)
    ax3.text(1.28, 1.07, '(b)', fontsize=16, fontweight='bold', va='center', ha='left')
    ax3.grid(which='minor', axis='y', linestyle='-', linewidth=1, color='lightgrey')

    fig.tight_layout()
    fig.savefig(output, dpi=300)


def zpt_plot_combine(p_lay, t_lay, dewT_lay, h2o_vmr,
                    p_drop, t_dry_drop, t_dew_drop, h2o_vmr_drop,
                    output, pmin=100, pmax=1000):
    from metpy.plots import SkewT
    from metpy.units import units

    p_prf = p_lay * units.hPa
    T_prf = (t_lay * units.kelvin).to(units.degC)
    Td_prf = (dewT_lay * units.kelvin).to(units.degC)

    p_drop = p_drop * units.hPa
    t_dry_drop = (t_dry_drop * units.kelvin).to(units.degC)
    t_dew_drop = (t_dew_drop * units.kelvin).to(units.degC)

    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(12, 6.75))
    ax1.set_visible(False)
    skew = SkewT(fig=fig, subplot=(1, 2, 1), aspect=120.5)
    skew.plot(p_prf, T_prf, 'coral', label='Temperature (modis07)', linewidth=4, alpha=0.85)
    skew.plot(p_prf, Td_prf, 'limegreen', label='Dew Point (modis07)', linewidth=4, alpha=0.85)
    skew.plot(p_drop, t_dry_drop, 'r', label='Temperature (dropsonde)', linewidth=2)
    skew.plot(p_drop, t_dew_drop, 'g', label='Dew Point (dropsonde)', linewidth=2)

    skew.ax.set_xlabel('Temperature ($\N{DEGREE CELSIUS}$)', fontsize=14)
    skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)
    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_ylim(pmax, pmin)
    skew.ax.legend()
    skew.ax.text(0.02, 1.07, '(a)', transform=skew.ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='left')

    ax2.plot(h2o_vmr, p_prf, 'deepskyblue', label='H$_2$O (modis07)', linewidth=3)
    ax2.plot(h2o_vmr_drop, p_drop, 'b', label='H$_2$O (dropsonde)', linewidth=3)
    h2o_vmr_axis_setting(ax2, pmin=pmin, pmax=pmax)
    ax2.legend()
    ax2.text(1.28, 1.07, '(b)', transform=skew.ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='left')
    ax2.grid(which='minor', axis='y', linestyle='-', linewidth=1, color='lightgrey')

    fig.tight_layout()
    fig.savefig(output, dpi=300)



if __name__ == '__main__':

    pass
