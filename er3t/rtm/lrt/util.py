import os
import numpy as np



__all__ = ['gen_cloud_1d', 'gen_aerosol_1d', 'gen_wavelength_file', 'gen_surface_albedo_file']



def gen_cloud_1d(cld_cfg):

    cld_cfg['cloud_file'] = os.path.abspath(cld_cfg['cloud_file'])

    altitude = cld_cfg['cloud_altitude']

    alt = np.sort(altitude)[::-1]
    lwc = np.zeros_like(alt)
    cer = np.zeros_like(alt)

    lwc[1:] = cld_cfg['liquid_water_content']
    cer[1:] = cld_cfg['cloud_effective_radius']

    with open(cld_cfg['cloud_file'], 'w') as f:
        f.write('# Altitude[km]    Liquid Water Content [g/m3]    Cloud Effective Radius [um]\n')
        for i, alt0 in enumerate(altitude):
            f.write('     %.4f                   %.4f                          %.4f\n' % (alt[i], lwc[i], cer[i]))
        if abs(alt[-1]-0.0)>0.001:
            f.write('     %.4f                   %.4f                          %.4f\n' % (0.0, 0.0, 0.0))

    return cld_cfg



def gen_aerosol_1d(aer_cfg):

    filename_aer = os.path.basename(aer_cfg['aerosol_file'])

    altitude = aer_cfg['aerosol_altitude']
    alt  = np.sort(altitude)[::-1]


    # aod
    # =====================================================================================
    data0 = aer_cfg['aerosol_optical_depth']
    if np.isscalar(data0):
        data = np.zeros_like(alt)
        data[1:] = data0
    elif isinstance(data0, np.ndarray):
        data = data0
    else:
        msg = 'Error [gen_aerosol_1d]: Only support scalar or array for <aerosol_optical_depth>.'
        raise OSError(msg)

    filename_aod = 'aer-aod_%s' % filename_aer
    fname_aod    = aer_cfg['aerosol_file'].replace(filename_aer, filename_aod)

    with open(fname_aod, 'w') as f:
        f.write('# Altitude[km]   AOD\n')
        for i, alt0 in enumerate(altitude):
            f.write('     %.4f                %.4f\n' % (alt[i], data[i]))
        if abs(alt[-1]-0.0)>0.001:
            f.write('     %.4f                %.4f\n' % (0.0, 0.0))

    aer_cfg['aerosol_file_aod'] = os.path.abspath(fname_aod)
    # =====================================================================================


    # ssa
    # =====================================================================================
    data0 = aer_cfg['single_scattering_albedo']
    if np.isscalar(data0):
        data = np.zeros_like(alt)
        data[1:] = data0
    elif isinstance(data0, np.ndarray):
        data = data0
    else:
        msg = 'Error [gen_aerosol_1d]: Only support scalar or array for <single_scattering_albedo>.'
        raise OSError(msg)

    filename_ssa = 'aer-ssa_%s' % filename_aer
    fname_ssa    = aer_cfg['aerosol_file'].replace(filename_aer, filename_ssa)

    with open(fname_ssa, 'w') as f:
        f.write('# Altitude[km]   SSA\n')
        for i, alt0 in enumerate(altitude):
            f.write('     %.4f                %.4f\n' % (alt[i], data[i]))
        if abs(alt[-1]-0.0)>0.001:
            f.write('     %.4f                %.4f\n' % (0.0, 0.0))

    aer_cfg['aerosol_file_ssa'] = os.path.abspath(fname_ssa)
    # =====================================================================================


    # asy
    # =====================================================================================
    data0 = aer_cfg['asymmetry_parameter']
    if np.isscalar(data0):
        data = np.zeros_like(alt)
        data[1:] = data0
    elif isinstance(data0, np.ndarray):
        data = data0
    else:
        msg = 'Error [gen_aerosol_1d]: only support scalar or array for <asymmetry_parameter>.'
        raise OSError(msg)

    filename_asy = 'aer-asy_%s' % filename_aer
    fname_asy    = aer_cfg['aerosol_file'].replace(filename_aer, filename_asy)

    with open(fname_asy, 'w') as f:
        f.write('# Altitude[km]   ASY\n')
        for i, alt0 in enumerate(altitude):
            f.write('     %.4f                %.4f\n' % (alt[i], data[i]))
        if abs(alt[-1]-0.0)>0.001:
            f.write('     %.4f                %.4f\n' % (0.0, 0.0))

    aer_cfg['aerosol_file_asy'] = os.path.abspath(fname_asy)
    # =====================================================================================

    return aer_cfg



def gen_wavelength_file(fname_wvl, wvls):

    np.savetxt(fname_wvl, wvls, fmt='%.3f')



def gen_surface_albedo_file(fname_alb, wvls, albs):

    np.savetxt(fname_alb, np.column_stack((wvls, albs)), fmt='%.3f')



if __name__ == '__main__':

    pass
