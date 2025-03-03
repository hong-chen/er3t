import numpy as np


import er3t.common
import er3t.util


__all__ = [
        'cal_ocean_brdf',\
        ]



def cal_ocean_brdf(
        wvl=er3t.common.params['wavelength'],\
        u10=1.0,\
        sal=34.3,\
        pcl=0.01,\
        whitecaps=True,\
        ):

    """
    This code is adapted from <libRadtran/libsrc_f/oceabrdf.f>

    Input parameters:
        u10: 10m wind speed, units: m/s, default=1.0, can be either value or 2D array for a domain
        sal: salinity, units: per mille [0.1% or ppt or psu], default=34.3
        pcl: pigment concentration, units: mg/m^3, default=0.01
    """

    # check data dimension
    #╭────────────────────────────────────────────────────────────────────────────╮#
    try:
        Nx, Ny = u10.shape
        u10 = np.float_(u10)
        ndim = u10.ndim
    except Exception as error:
        # print(error)
        u10 = float(u10)
        ndim = 0

    if ndim == 2:
        wvl_ = np.zeros_like(u10)
        wvl_[...] = wvl
        wvl = wvl_

        sal_ = np.zeros_like(u10)
        sal_[...] = sal
        sal = sal_
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # refractive index of water as a function of wavelength and salinity
    #╭────────────────────────────────────────────────────────────────────────────╮#
    reference = '\nRefractive Index of Water (Hale and Querry, 1973):\n- Hale, G. M., and Querry, M. R.: Optical Constants of Water in the 200-nm to 200-μm Wavelength Region, Appl. Opt. 12, 555-563, https://doi.org/10.1364/AO.12.000555, 1973.'
    er3t.util.add_reference(reference)

    refractive_index_water = {
            'wvl': np.array([ \
                   0.250,0.275,0.300,0.325,0.345,0.375,0.400,0.425,0.445,0.475,\
                   0.500,0.525,0.550,0.575,0.600,0.625,0.650,0.675,0.700,0.725,\
                   0.750,0.775,0.800,0.825,0.850,0.875,0.900,0.925,0.950,0.975,\
                   1.000,1.200,1.400,1.600,1.800,2.000,2.200,2.400,2.600,2.650,\
                   2.700,2.750,2.800,2.850,2.900,2.950,3.000,3.050,3.100,3.150,\
                   3.200,3.250,3.300,3.350,3.400,3.450,3.500,3.600,3.700,3.800,\
                   3.900,4.000], dtype=np.float64) * 1000.0,

           'real': np.array([ \
                   1.362,1.354,1.349,1.346,1.343,1.341,1.339,1.338,1.337,1.336,\
                   1.335,1.334,1.333,1.333,1.332,1.332,1.331,1.331,1.331,1.330,\
                   1.330,1.330,1.329,1.329,1.329,1.328,1.328,1.328,1.327,1.327,\
                   1.327,1.324,1.321,1.317,1.312,1.306,1.296,1.279,1.242,1.219,\
                   1.188,1.157,1.142,1.149,1.201,1.292,1.371,1.426,1.467,1.483,\
                   1.478,1.467,1.450,1.432,1.420,1.410,1.400,1.385,1.374,1.364,\
                   1.357,1.351], dtype=np.float64),

      'imaginary': np.array([ \
                   3.35E-08,2.35E-08,1.60E-08,1.08E-08,6.50E-09,\
                   3.50E-09,1.86E-09,1.30E-09,1.02E-09,9.35E-10,\
                   1.00E-09,1.32E-09,1.96E-09,3.60E-09,1.09E-08,\
                   1.39E-08,1.64E-08,2.23E-08,3.35E-08,9.15E-08,\
                   1.56E-07,1.48E-07,1.25E-07,1.82E-07,2.93E-07,\
                   3.91E-07,4.86E-07,1.06E-06,2.93E-06,3.48E-06,\
                   2.89E-06,9.89E-06,1.38E-04,8.55E-05,1.15E-04,\
                   1.10E-03,2.89E-04,9.56E-04,3.17E-03,6.70E-03,\
                   1.90E-02,5.90E-02,1.15E-01,1.85E-01,2.68E-01,\
                   2.98E-01,2.72E-01,2.40E-01,1.92E-01,1.35E-01,\
                   9.24E-02,6.10E-02,3.68E-02,2.61E-02,1.95E-02,\
                   1.32E-02,9.40E-03,5.15E-03,3.60E-03,3.40E-03,\
                   3.80E-03,4.60E-03], dtype=np.float64)
                 }

    refrac_r = np.interp(wvl, refractive_index_water['wvl'], refractive_index_water['real'])
    refrac_i = np.interp(wvl, refractive_index_water['wvl'], refractive_index_water['imaginary'])

    # salinity corrections
    #╭──────────────────────────────────────────────────────────────╮#
    reference = '\nSalinity Correction (Friedman, 1969):\n- Friedman, D.: Infrared Characteristics of Ocean Water (1.5 –15 μ), Appl. Opt. 8, 2073-2078, https://doi.org/10.1364/AO.8.002073, 1969.'
    er3t.util.add_reference(reference)

    refrac_r += 0.006*(sal/34.3)
    # refrac_i += 0.000*(sal/34.3)
    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # variance of micro-scopic surface slope
    #╭────────────────────────────────────────────────────────────────────────────╮#
    reference = '\nCox-Munk Parameterization (Cox and Munk, 1954):\n- Cox, C., and Munk, W.: Measurement of the Roughness of the Sea Surface from Photographs of the Sun’s Glitter, J. Opt. Soc. Am. 44, 838-850, https://doi.org/10.1364/JOSA.44.000838, 1954.'
    er3t.util.add_reference(reference)

    slope = 0.00512*u10 + 0.003
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # whitecaps treatment
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if whitecaps:

        reference = '\nWhitecaps (Koepke, 1984):\n- Koepke, P.: Effective reflectance of oceanic whitecaps, Appl. Opt. 23, 1816-1824, https://doi.org/10.1364/AO.23.001816, 1984.'
        er3t.util.add_reference(reference)

        reflectance_whitecaps = {
                'wvl': np.arange(200.0, 4001.0, 100.0),
                'ref': np.array([
                       0.220,0.220,0.220,0.220,0.220,0.220,0.215,0.210,0.200,0.190,
                       0.175,0.155,0.130,0.080,0.100,0.105,0.100,0.080,0.045,0.055,
                       0.065,0.060,0.055,0.040,0.000,0.000,0.000,0.000,0.000,0.000,
                       0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000
                       ]),
                }

        diffuse_frac = 2.95e-06 * (u10**3.52)
        diffuse_alb  = np.interp(wvl, reflectance_whitecaps['wvl'], reflectance_whitecaps['ref'])

    else:

        diffuse_frac = 0.0*u10
        diffuse_alb  = 0.0*u10
    #╰────────────────────────────────────────────────────────────────────────────╯#

    params = {
          'diffuse_alb': diffuse_alb,
         'diffuse_frac': diffuse_frac,
             'refrac_r': refrac_r,
             'refrac_i': refrac_i,
                'slope': slope,
            }

    return params




if __name__ == '__main__':

    pass
