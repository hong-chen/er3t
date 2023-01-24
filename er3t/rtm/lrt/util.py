import os
import sys
import datetime
import numpy as np

import er3t.common
import er3t.rtm.lrt as lrt


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
    elif tag.lower() == 'reflectance':
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
        cloud_altitude=np.arange(0.9, 1.51, 0.1),              # vertical location of the clouds
        cloud_optical_thickness_all=np.arange(0.0, 50.1, 2.0), # cloud optical thickness array (numpy.ndarray)
        cloud_effective_radius_all=np.arange(4.0, 25.1, 1.0),  # cloud effective radius (numpy.ndarray)
        aerosol_optical_depth=0.0,                             # aerosol optical depth
        aerosol_single_scattering_albedo=0.8,                  # aerosol single scattering albedo
        aerosol_asymmetry_parameter=0.7,                       # aerosol asymmetry parameter
        aerosol_altitude=np.arange(2.9, 6.01, 0.1),            # vertical location of the aerosols
        output_altitude=np.array([0.8, 2.0]),                  # output altitude for libRadtran calculations
        fdir_tmp='tmp-data',                                   # directory to store temporary data (string)
        fdir_lut='data/lut',                                   # directory to store lookup table data
        prop_tag='radiance',                                   # property tag, can be "radiance", "albedo-top", "albedo-bottom", "reflectance", "transmittance", "absorptance" (string)
        atmosphere_file='%s/afglus.dat' % er3t.common.fdir_data_atmmod, # atmosphere profile
        overwrite=True,
        plot=True
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
    inits_x = []
    inits_y = []

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = atmosphere_file

    cld_cfg = lrt.get_cld_cfg()
    if cloud_type.lower() == 'water':
        cld_cfg['wc_properties'] = 'mie'
    else:
        msg = '\nError [gen_bispectral_lookup_table]: currently we do not support <cloud_type="%s">' % (cloud_type)
        sys.exit(msg)

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

    for cot in cloud_optical_thickness_all:
        for cer in cloud_effective_radius_all:

            cld_cfg['cloud_file']              = '%s/lrt_cloud_%06.2f_%06.2f.txt' % (fdir_tmp, cot, cer)
            cld_cfg['cloud_altitude']          = cloud_altitude
            cld_cfg['cloud_optical_thickness'] = cot
            cld_cfg['cloud_effective_radius']  = cer

            input_file_x  = '%s/lrt_inpfile_%4.4dnm_%04.1f_%04.1f_%04.1f_%04.2f.txt' % (fdir_tmp, wvl_x, cot, cer, sza, alb_x)
            output_file_x = '%s/lrt_outfile_%4.4dnm_%04.1f_%04.1f_%04.1f_%04.2f.txt' % (fdir_tmp, wvl_x, cot, cer, sza, alb_x)

            input_file_y  = '%s/lrt_inpfile_%4.4dnm_%04.1f_%04.1f_%04.1f_%04.1f.txt' % (fdir_tmp, wvl_y, cot, cer, sza, alb_y)
            output_file_y = '%s/lrt_outfile_%4.4dnm_%04.1f_%04.1f_%04.1f_%04.1f.txt' % (fdir_tmp, wvl_y, cot, cer, sza, alb_y)

            if prop_tag.lower() in ['radiance', 'rad']:

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

            elif prop_tag.lower() in ['flux', 'iiradiance']:

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
    lrt.lrt_run_mp(inits_x+inits_y)
    #\--------------------------------------------------------------/#

    # read output
    #/--------------------------------------------------------------\#
    data_x = lrt.lrt_read_uvspec(inits_x)
    data_y = lrt.lrt_read_uvspec(inits_y)
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # process calculations
    #/----------------------------------------------------------------------------\#
    prop_x = cal_radiative_property(data_x.f_up, data_x.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
    prop_y = cal_radiative_property(data_y.f_up, data_y.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
    #\----------------------------------------------------------------------------/#


    # save RT calculations
    #/----------------------------------------------------------------------------\#
    # create data directory
    #/--------------------------------------------------------------\#
    fdir_lut = os.path.abspath(fdir_lut)

    if not os.path.exists(fdir_lut):
        os.system('mkdir -p %s' % fdir_lut)
    #\--------------------------------------------------------------/#

    f = h5py.File(fname, 'w')

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
    f['cer'].attrs['description'] = 'Cloud Effective Radius'
    f.close()
    #\----------------------------------------------------------------------------/#


    # plot lookup table
    #/----------------------------------------------------------------------------\#
    if plot:
        plot_bispectral_lookup_table(fname)
    #\----------------------------------------------------------------------------/#



def plot_bispectral_lookup_table(fname, prop_x0=-1.0, prop_y0=-1.0, fdir_out='data'):

    filename = fname.split('/')[-1][:-3]
    words    = filename.split('_')
    prop     = words[2].replace('-', ' ').title()
    wvl_pair = []
    for word in words:
        if 'nm' in word:
            wavelengths = word.split('-')
            for wavelength in wavelengths:
                if wavelength[0] == '0':
                    wavelength = wavelength[1:]
                wvl_pair.append(wavelength)

    f = h5py.File(fname, 'r')
    cer = f['cer'][...]
    cot = f['cot'][...]
    prop_x = f['prop_x'][...]
    prop_y = f['prop_y'][...]
    f.close()

    rcParams['font.size'] = 15
    fig = plt.figure(figsize=(6.5, 6.0))
    ax1 = fig.add_subplot(111)

    for i in range(cer.size):
        ax1.plot(prop_x[:, i], prop_y[:, i], color='r', zorder=0, lw=0.5, alpha=0.2)
        ax1.text(prop_x[-1, i]+0.008, prop_y[-1, i], '%.1f' % cer[i], fontsize=6, color='r', va='center', zorder=1, weight=0)

    for i in range(cot.size):
        ax1.plot(prop_x[i, :], prop_y[i, :], color='b', lw=0.5, zorder=1, alpha=0.2)
        ax1.text(prop_x[i, -1], prop_y[i, -1]-0.02, '%.1f' % cot[i], fontsize=6, color='b', ha='center', zorder=1, weight=0)

    ax1.scatter(prop_x0, prop_y0, c='k', s=1)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('%s [%s]' % (prop, wvl_pair[0]))
    ax1.set_ylabel('%s [%s]' % (prop, wvl_pair[1]))
    plt.savefig('%s/%s.svg' % (fdir_out, filename))
    plt.close(fig)



if __name__ == '__main__':

    pass
