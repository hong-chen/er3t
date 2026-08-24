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
        ref_whitecap = np.interp(wvl, reflectance_whitecaps['wvl'], reflectance_whitecaps['ref'])

    else:

        diffuse_frac = 0.0*u10
        ref_whitecap = 0.0*u10
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # underwater backscatter
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # subsurface reflectance from Case-I water (Morel et al., 1988)
    # This is the pigment-dependent term used by oceabrdf.f and
    # shd_sfc__brdfOcean.  It is Lambertian below the water surface.

    reference = '\nWhitecaps (Koepke, 1984):\n- Koepke, P.: Effective reflectance of oceanic whitecaps, Appl. Opt. 23, 1816-1824, https://doi.org/10.1364/AO.23.001816, 1984.'
    er3t.util.add_reference(reference)

    morel_wvl = np.arange(0.400, 0.701, 0.005)

    kw = np.array([
        0.0209,0.0200,0.0196,0.0189,0.0183,0.0182,0.0171,0.0170,0.0168,0.0166,
        0.0168,0.0170,0.0173,0.0174,0.0175,0.0184,0.0194,0.0203,0.0217,0.0240,
        0.0271,0.0320,0.0384,0.0445,0.0490,0.0505,0.0518,0.0543,0.0568,0.0615,
        0.0640,0.0640,0.0717,0.0762,0.0807,0.0940,0.1070,0.1280,0.1570,0.2000,
        0.2530,0.2790,0.2960,0.3030,0.3100,0.3150,0.3200,0.3250,0.3300,0.3400,
        0.3500,0.3700,0.4050,0.4180,0.4300,0.4400,0.4500,0.4700,0.5000,0.5500,
        0.6500])

    xc = np.array([
        0.1100,0.1110,0.1125,0.1135,0.1126,0.1104,0.1078,0.1065,0.1041,0.0996,
        0.0971,0.0939,0.0896,0.0859,0.0823,0.0788,0.0746,0.0726,0.0690,0.0660,
        0.0636,0.0600,0.0578,0.0540,0.0498,0.0475,0.0467,0.0450,0.0440,0.0426,
        0.0410,0.0400,0.0390,0.0375,0.0360,0.0340,0.0330,0.0328,0.0325,0.0330,
        0.0340,0.0350,0.0360,0.0375,0.0385,0.0400,0.0420,0.0430,0.0440,0.0445,
        0.0450,0.0460,0.0475,0.0490,0.0515,0.0520,0.0505,0.0440,0.0390,0.0340,
        0.0300])

    exponent = np.array([
        0.668,0.672,0.680,0.687,0.693,0.701,0.707,0.708,0.707,0.704,
        0.701,0.699,0.700,0.703,0.703,0.703,0.703,0.704,0.702,0.700,
        0.700,0.695,0.690,0.685,0.680,0.675,0.670,0.665,0.660,0.655,
        0.650,0.645,0.640,0.630,0.623,0.615,0.610,0.614,0.618,0.622,
        0.626,0.630,0.634,0.638,0.642,0.647,0.653,0.658,0.663,0.667,
        0.672,0.677,0.682,0.687,0.695,0.697,0.693,0.665,0.640,0.620,
        0.600])

    bw = np.array([
        0.0076,0.0072,0.0068,0.0064,0.0061,0.0058,0.0055,0.0052,0.0049,0.0047,
        0.0045,0.0043,0.0041,0.0039,0.0037,0.0036,0.0034,0.0033,0.0031,0.0030,
        0.0029,0.0027,0.0026,0.0025,0.0024,0.0023,0.0022,0.0022,0.0021,0.0020,
        0.0019,0.0018,0.0018,0.0017,0.0017,0.0016,0.0016,0.0015,0.0015,0.0014,
        0.0014,0.0013,0.0013,0.0012,0.0012,0.0011,0.0011,0.0010,0.0010,0.0010,
        0.0010,0.0009,0.0008,0.0008,0.0008,0.0007,0.0007,0.0007,0.0007,0.0007,
        0.0007])

    kw = np.interp(wvl, morel_wvl, kw)
    xc = np.interp(wvl, morel_wvl, xc)
    exponent = np.interp(wvl, morel_wvl, exponent)

    bw = np.interp(wvl, morel_wvl, bw)
    valid = (wvl >= 0.4) & (wvl <= 0.7)

    pigment = np.maximum(np.asarray(pcl), 0.0)
    bb = 0.5*bw
    kd = kw
    with_pigment = pigment >= 1.0e-4
    b = 0.30*pigment**0.62
    bbt = 0.002 + 0.02*(0.5 - 0.25*np.log10(np.maximum(pigment, 1.0e-4)))*0.550/np.maximum(wvl, 1.0e-12)
    bb = np.where(with_pigment, 0.5*bw + bbt*b, bb)
    kd = np.where(with_pigment, kw + xc*pigment**exponent, kd)

    ref0 = 0.33*bb/(0.75*kd)

    for _ in range(100):
        u = 0.90*(1.0-ref0)/(1.0+2.25*ref0)
        ref_underwater = 0.33*bb/(u*kd)
        if np.all(np.abs((ref_underwater-ref0)/np.maximum(ref_underwater, 1.0e-30)) < 1.0e-4):
            break
        ref0 = ref_underwater

    ref_underwater = np.where(valid, ref_underwater, 0.0)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    diffuse_alb = (1.0 - ref_whitecap)*ref_underwater + ref_whitecap
    # diffuse_alb = ref_whitecap

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
