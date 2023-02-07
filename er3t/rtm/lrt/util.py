import os
import sys
import glob
import datetime
import numpy as np
import h5py

import er3t.common
import er3t.rtm.lrt as lrt


__all__ = [
        'gen_cloud_1d', 'gen_aerosol_1d', 'gen_wavelength_file', 'gen_surface_albedo_file', \
        'gen_bispectral_lookup_table',
        ]



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



def cal_radiative_property(f_up, f_down, topN=-1, bottomN=0, scaleN=1.0, tag='all'):

    """
    Calculate radiative properties such as tranmisttance, reflectance etc. based on
    given upwelling and downwelling irradiances

    Inputs:
        f_up: upwelling irradiance
        f_down: downwelling irradiance

    Outputs:
        specified radiative property (by `tag`, by default is `tag='all'`)
    """

    transmittance  = f_down[bottomN, ...]/f_down[topN, ...] * scaleN
    albedo_bottom  = f_up[bottomN, ...]/f_down[bottomN, ...] * scaleN
    albedo_top     = f_up[topN, ...]/f_down[topN, ...] * scaleN

    f_net_top      = f_down[topN, ...] - f_up[topN, ...]
    f_net_bottom   = f_down[bottomN, ...] - f_up[bottomN, ...]

    absorptance    = (f_net_top-f_net_bottom)/f_down[topN, ...] * scaleN
    reflectance    = (f_up[topN, ...] - f_up[bottomN, ...]) / f_down[topN, ...] * scaleN

    if tag.lower() == 'transmittance':
        return transmittance
    elif tag.lower() == 'reflectance-top':
        return reflectance
    elif tag.lower() == 'absorptance':
        return absorptance
    elif tag.lower() == 'albedo-top':
        return albedo_top
    elif tag.lower() == 'albedo-bottom':
        return albedo_bottom
    elif tag.lower() == 'all':
        return [transmittance, reflectance, absorptance, albedo_top, albedo_bottom]



def gen_bispectral_lookup_table(
        date=datetime.date.today(),                            # date (datetime.date)
        wavelength_pair=(860.0, 2130.0),                       # two wavelengths (list or tuple)
        surface_albedo_pair=(0.03, 0.03),                      # two surface albedos (list or tuple)
        solar_zenith_angle=0.0,                                # solar zenith angle (float)
        solar_azimuth_angle=0.0,                               # solar azimuth angle (float)
        sensor_zenith_angle=0.0,                               # sensor zenith angle (float)
        sensor_azimuth_angle=0.0,                              # sensor azimuth angle (float)
        cloud_type='water',                                    # water cloud or ice cloud
        cloud_altitude=np.arange(0.4, 1.51, 0.1),              # vertical location of the clouds
        cloud_optical_thickness_all=np.arange(0.0, 50.1, 2.0), # cloud optical thickness array (numpy.ndarray)
        cloud_effective_radius_all=np.arange(4.0, 25.1, 1.0),  # cloud effective radius (numpy.ndarray)
        aerosol_optical_depth=0.0,                             # aerosol optical depth
        aerosol_single_scattering_albedo=0.8,                  # aerosol single scattering albedo
        aerosol_asymmetry_parameter=0.7,                       # aerosol asymmetry parameter
        aerosol_altitude=np.arange(2.9, 6.01, 0.1),            # vertical location of the aerosols
        output_altitude=np.array([0.8, 2.0]),                  # output altitude for libRadtran calculations
        fname=None,                                            # output file
        fdir_tmp='tmp-data',                                   # directory to store temporary data (string)
        fdir_lut='data/lut',                                   # directory to store lookup table data
        prop_tag='radiance',                                   # property tag, can be "radiance", "albedo-top", "albedo-bottom", "reflectance", "transmittance", "absorptance" (string)
        atmosphere_file='%s/afglus.dat' % er3t.common.fdir_data_atmmod, # atmosphere profile
        overwrite=True,
        ):

    # create temporary data directory
    #/----------------------------------------------------------------------------\#
    fdir_tmp = os.path.abspath(fdir_tmp)
    if not os.path.exists(fdir_tmp):
        os.system('mkdir -p %s' % fdir_tmp)
    #\----------------------------------------------------------------------------/#


    # delete old files if overwrite is specified
    #/----------------------------------------------------------------------------\#
    if overwrite:
        if len(glob.glob('%s/*.txt' % fdir_tmp)) > 0:
            os.system('find %s -name "*.txt" | xargs rm -f' % fdir_tmp)
    #\----------------------------------------------------------------------------/#


    # assign parameters
    #/----------------------------------------------------------------------------\#
    wvl_x, wvl_y = wavelength_pair
    alb_x, alb_y = surface_albedo_pair
    sza = solar_zenith_angle
    saa = solar_azimuth_angle
    vza = sensor_zenith_angle
    vaa = sensor_azimuth_angle
    #\----------------------------------------------------------------------------/#


    # RT setup
    #/----------------------------------------------------------------------------\#
    # initialization
    #/--------------------------------------------------------------\#

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = atmosphere_file

    # cloud setup
    #/--------------------------------------------------------------\#
    cld_cfg = lrt.get_cld_cfg()
    if cloud_type.lower() == 'water':
        cld_cfg['wc_properties'] = 'mie'
    else:
        msg = '\nError [gen_bispectral_lookup_table]: currently we do not support <cloud_type="%s">' % (cloud_type)
        sys.exit(msg)
    #\--------------------------------------------------------------/#

    # aerosol setup
    #/--------------------------------------------------------------\#
    if aerosol_optical_depth > 0.0:
        aer_cfg = lrt.get_aer_cfg()
        aer_cfg['aerosol']
        aer_cfg['aerosol_file'] = '%s/lrt_aerosol.txt' % (fdir_tmp)
        aer_cfg['aerosol_optical_depth']    = aerosol_optical_depth
        aer_cfg['single_scattering_albedo'] = aerosol_single_scattering_albedo
        aer_cfg['asymmetry_parameter']      = aerosol_asymmetry_parameter
        aer_cfg['aerosol_altitude']         = aerosol_altitude
    else:
        aer_cfg = None
    #\--------------------------------------------------------------/#


    # for toa downwelling
    #/--------------------------------------------------------------\#
    init_x0 = lrt.lrt_init_mono(
            output_altitude='toa',
            input_file='%s/lrt_inpfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_x),
            output_file='%s/lrt_outfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_x),
            date=date,
            surface_albedo=alb_x,
            wavelength=wvl_x,
            solar_zenith_angle=sza,
            lrt_cfg=lrt_cfg,
            )

    init_y0 = lrt.lrt_init_mono(
            output_altitude='toa',
            input_file='%s/lrt_inpfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_y),
            output_file='%s/lrt_outfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_y),
            date=date,
            surface_albedo=alb_y,
            wavelength=wvl_y,
            solar_zenith_angle=sza,
            lrt_cfg=lrt_cfg,
            )
    #\--------------------------------------------------------------/#

    inits_x = []
    inits_y = []

    for cot in cloud_optical_thickness_all:
        for cer in cloud_effective_radius_all:

            cld_cfg['cloud_file']              = '%s/lrt_cloud_%06.2f_%06.2f.txt' % (fdir_tmp, cot, cer)
            cld_cfg['cloud_altitude']          = cloud_altitude
            cld_cfg['cloud_optical_thickness'] = cot
            cld_cfg['cloud_effective_radius']  = cer

            input_file_x  = '%s/lrt_inpfile_wvl-%4.4dnm_cot-%04.1f_cer-%04.1f_sza-%04.1f_alb-%04.2f.txt' % (fdir_tmp, wvl_x, cot, cer, sza, alb_x)
            output_file_x = '%s/lrt_outfile_wvl-%4.4dnm_cot-%04.1f_cer-%04.1f_sza-%04.1f_alb-%04.2f.txt' % (fdir_tmp, wvl_x, cot, cer, sza, alb_x)

            input_file_y  = '%s/lrt_inpfile_wvl-%4.4dnm_cot-%04.1f_cer-%04.1f_sza-%04.1f_alb-%04.1f.txt' % (fdir_tmp, wvl_y, cot, cer, sza, alb_y)
            output_file_y = '%s/lrt_outfile_wvl-%4.4dnm_cot-%04.1f_cer-%04.1f_sza-%04.1f_alb-%04.1f.txt' % (fdir_tmp, wvl_y, cot, cer, sza, alb_y)

            if prop_tag.lower() in ['radiance', 'rad', 'reflectance', 'ref']:

                init_x = lrt.lrt_init_mono_rad(
                        output_altitude=output_altitude,
                        input_file=input_file_x,
                        output_file=output_file_x,
                        date=date,
                        surface_albedo=alb_x,
                        wavelength=wvl_x,
                        solar_zenith_angle=sza,
                        solar_azimuth_angle=saa,
                        sensor_zenith_angle=vza,
                        sensor_azimuth_angle=vaa,
                        lrt_cfg=lrt_cfg,
                        cld_cfg=cld_cfg,
                        aer_cfg=aer_cfg,
                        )

                init_y = lrt.lrt_init_mono_rad(
                        output_altitude=output_altitude,
                        input_file=input_file_y,
                        output_file=output_file_y,
                        date=date,
                        surface_albedo=alb_y,
                        wavelength=wvl_y,
                        solar_zenith_angle=sza,
                        solar_azimuth_angle=saa,
                        sensor_zenith_angle=vza,
                        sensor_azimuth_angle=vaa,
                        lrt_cfg=lrt_cfg,
                        cld_cfg=cld_cfg,
                        aer_cfg=aer_cfg,
                        )

            elif prop_tag.lower() in ['transmittance', 'reflectance', 'absorptance', 'albedo-top', 'albedo-bottom', 'all']:

                init_x = lrt.lrt_init_mono(
                        output_altitude=output_altitude,
                        input_file=input_file_x,
                        output_file=output_file_x,
                        date=date,
                        surface_albedo=alb_x,
                        wavelength=wvl_x,
                        solar_zenith_angle=sza,
                        lrt_cfg=lrt_cfg,
                        cld_cfg=cld_cfg,
                        aer_cfg=aer_cfg,
                        )


                init_y = lrt.lrt_init_mono(
                        output_altitude=output_altitude,
                        input_file=input_file_y,
                        output_file=output_file_y,
                        date=date,
                        surface_albedo=alb_y,
                        wavelength=wvl_y,
                        solar_zenith_angle=sza,
                        lrt_cfg=lrt_cfg,
                        cld_cfg=cld_cfg,
                        aer_cfg=aer_cfg,
                        )

            else:
                msg = '\nError [gen_bispectral_lookup_table]: currently we do not support <prop_tag="%s">' % (prop_tag)
                sys.exit(msg)

            inits_x.append(init_x)
            inits_y.append(init_y)
    #\--------------------------------------------------------------/#

    # run
    #/--------------------------------------------------------------\#
    lrt.lrt_run_mp(inits_x+inits_y+[init_x0, init_y0])
    #\--------------------------------------------------------------/#

    # read output
    #/--------------------------------------------------------------\#
    if prop_tag.lower() in ['radiance', 'rad', 'reflectance', 'ref']:

        data_x = lrt.lrt_read_uvspec_rad(inits_x)
        data_y = lrt.lrt_read_uvspec_rad(inits_y)

        prop_x = np.squeeze(data_x.rad).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        prop_y = np.squeeze(data_y.rad).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))

        if prop_tag.lower() in ['reflectance', 'ref']:

            data_x0 = lrt.lrt_read_uvspec([init_x0])
            data_y0 = lrt.lrt_read_uvspec([init_y0])
            prop_x = np.pi*prop_x/(np.squeeze(data_x0.f_down))
            prop_y = np.pi*prop_y/(np.squeeze(data_y0.f_down))

    elif prop_tag.lower() in ['transmittance', 'reflectance-top', 'absorptance', 'albedo-top', 'albedo-bottom']:

        data_x = lrt.lrt_read_uvspec(inits_x)
        data_y = lrt.lrt_read_uvspec(inits_y)

        # process calculations
        #/--------------------------------------------------------------\#
        prop_x = cal_radiative_property(data_x.f_up, data_x.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        prop_y = cal_radiative_property(data_y.f_up, data_y.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        #\--------------------------------------------------------------/#

    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#



    # save RT calculations
    #/----------------------------------------------------------------------------\#
    # create data directory
    #/--------------------------------------------------------------\#
    fdir_lut = os.path.abspath(fdir_lut)

    if not os.path.exists(fdir_lut):
        os.system('mkdir -p %s' % fdir_lut)
    #\--------------------------------------------------------------/#

    if fname is None:
        fname = 'lut_%4.4dnm-%4.4dnm.h5' % wavelength_pair

    f = h5py.File(fname, 'w')

    g = f.create_group('params')
    g['wvl_x'] = wvl_x
    g['wvl_x'].attrs['description'] = 'Wavelength x [nm]'

    g['wvl_y'] = wvl_y
    g['wvl_y'].attrs['description'] = 'Wavelength y [nm]'

    g['alb_x'] = alb_x
    g['alb_x'].attrs['description'] = 'Surface albedo x'

    g['alb_y'] = alb_y
    g['alb_y'].attrs['description'] = 'Surface albedo y'

    g['sza'] = sza
    g['sza'].attrs['description'] = 'Solar zenith angle'

    g['saa'] = saa
    g['saa'].attrs['description'] = 'Solar azimuth angle'

    g['vza'] = vza
    g['vza'].attrs['description'] = 'Viewing zenith angle'

    g['vaa'] = vaa
    g['vaa'].attrs['description'] = 'Viewing azimuth angle'

    g['atm_file'] = atmosphere_file
    g['atm_file'].attrs['description'] = 'Atmospheric profile'

    f['prop_x'] = prop_x
    f['prop_x'].dims[0].label = 'Cloud Optical Thickness'
    f['prop_x'].dims[1].label = 'Cloud Effective Radius [micron]'
    f['prop_x'].attrs['description'] = '%s at %.2f nm' % (prop_tag.title(), wvl_x)

    f['prop_y'] = prop_y
    f['prop_y'].dims[0].label = 'Cloud Optical Thickness'
    f['prop_y'].dims[1].label = 'Cloud Effective Radius [micron]'
    f['prop_y'].attrs['description'] = '%s at %.2f nm' % (prop_tag.title(), wvl_y)

    f['cot']    = cloud_optical_thickness_all
    f['cot'].attrs['description'] = 'Cloud Optical Thickness'

    f['cer']    = cloud_effective_radius_all
    f['cer'].attrs['description'] = 'Cloud Effective Radius [micron]'
    f.close()
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    pass
