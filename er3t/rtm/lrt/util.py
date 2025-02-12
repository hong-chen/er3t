import os
import sys
import glob
import datetime
import numpy as np
import h5py
from scipy.interpolate import interp1d

import er3t.common
import er3t.rtm.lrt as lrt


__all__ = [
        'gen_cloud_1d', 'gen_aerosol_1d', 'gen_wavelength_file', 'gen_surface_albedo_file', \
        'gen_bispectral_lookup_table', \
        'func_ref_vs_cot', \
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
    data0 = aer_cfg['aerosol_optical_depth']/(alt.size-1)
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



def cal_radiative_property(f_up, f_down, topN=-1, bottomN=0, scaleN=1.0, tag='albedo-top'):

    """
    Calculate radiative properties such as tranmisttance, reflectance etc. based on
    given upwelling and downwelling irradiances

    Inputs:
        f_up: upwelling irradiance
        f_down: downwelling irradiance

    Outputs:
        specified radiative property (by `tag`)
    """

    # number of layer check
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if (f_up.shape[1] < 2) and (tag.lower() not in ['albedo-top', 'albedo-bottom']):
        msg = '\nError [cal_radiative_property]: Insufficient number of layers for calculating radiative property.'
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if tag.lower() == 'transmittance':

        transmittance = f_down[0, bottomN, :]/f_down[0, topN, :] * scaleN
        return transmittance

    elif tag.lower() == '_reflectance':

        reflectance = (f_up[0, topN, :] - f_up[0, bottomN, :]) / f_down[0, topN, :] * scaleN
        return reflectance

    elif tag.lower() == 'absorptance':

        f_net_top    = f_down[0, topN, :] - f_up[0, topN, :]
        f_net_bottom = f_down[0, bottomN, :] - f_up[0, bottomN, :]
        absorptance  = (f_net_top-f_net_bottom)/f_down[0, topN, :] * scaleN
        return absorptance

    elif tag.lower() == 'albedo-top':

        albedo_top = f_up[0, topN, :]/f_down[0, topN, :] * scaleN
        return albedo_top

    elif tag.lower() == 'albedo-bottom':

        albedo_bottom = f_up[0, bottomN, :]/f_down[0, bottomN, :] * scaleN
        return albedo_bottom



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
        prop_tag='reflectance',                                # property tag, can be "radiance", "reflectance", "albedo-top", "albedo-bottom", "_reflectance", "transmittance", "absorptance" (string)
        atmosphere_file='%s/afglus.dat' % er3t.common.fdir_data_atmmod, # atmosphere profile
        overwrite=True,
        ):

    # create temporary data directory
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir_tmp = os.path.abspath(fdir_tmp)
    if not os.path.exists(fdir_tmp):
        os.system('mkdir -p %s' % fdir_tmp)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # delete old files if overwrite is specified
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if overwrite:
        if len(glob.glob('%s/*.txt' % fdir_tmp)) > 0:
            os.system('find %s -name "*.txt" | xargs rm -f' % fdir_tmp)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # assign parameters
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvl_x, wvl_y = wavelength_pair
    alb_x, alb_y = surface_albedo_pair
    sza = solar_zenith_angle
    saa = solar_azimuth_angle
    vza = sensor_zenith_angle
    vaa = sensor_azimuth_angle
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # RT setup
    #╭────────────────────────────────────────────────────────────────────────────╮#

    # initialization
    #╭──────────────────────────────────────────────────────────────╮#
    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = atmosphere_file

    # cloud setup
    #╭────────────────────────────────────────────────╮#
    cld_cfg = lrt.get_cld_cfg()
    if cloud_type.lower() == 'water':
        cld_cfg['wc_properties'] = 'mie'
    else:
        msg = '\nError [gen_bispectral_lookup_table]: <cloud_type="%s"> is NOT supported.' % (cloud_type)
        raise OSError(msg)
    #╰────────────────────────────────────────────────╯#

    # aerosol setup
    #╭────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────╯#


    # for toa downwelling
    #╭────────────────────────────────────────────────╮#
    init_x0 = lrt.lrt_init_mono_flx(
            output_altitude='toa',
            input_file='%s/lrt_inpfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_x),
            output_file='%s/lrt_outfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_x),
            date=date,
            surface_albedo=alb_x,
            wavelength=wvl_x,
            solar_zenith_angle=sza,
            lrt_cfg=lrt_cfg,
            )

    init_y0 = lrt.lrt_init_mono_flx(
            output_altitude='toa',
            input_file='%s/lrt_inpfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_y),
            output_file='%s/lrt_outfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_y),
            date=date,
            surface_albedo=alb_y,
            wavelength=wvl_y,
            solar_zenith_angle=sza,
            lrt_cfg=lrt_cfg,
            )
    #╰────────────────────────────────────────────────╯#

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

            elif prop_tag.lower() in ['transmittance', '_reflectance', 'absorptance', 'albedo-top', 'albedo-bottom', 'all']:

                init_x = lrt.lrt_init_mono_flx(
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


                init_y = lrt.lrt_init_mono_flx(
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
    #╰──────────────────────────────────────────────────────────────╯#

    # run
    #╭──────────────────────────────────────────────────────────────╮#
    lrt.lrt_run_mp([init_x0, init_y0]+inits_x+inits_y)
    #╰──────────────────────────────────────────────────────────────╯#

    # read output
    #╭──────────────────────────────────────────────────────────────╮#
    if prop_tag.lower() in ['radiance', 'rad', 'reflectance', 'ref']:

        data_x = lrt.lrt_read_uvspec_rad(inits_x)
        data_y = lrt.lrt_read_uvspec_rad(inits_y)

        prop_x = data_x.rad[0, -1, :].reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        prop_y = data_y.rad[0, -1, :].reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))

        if prop_tag.lower() in ['reflectance', 'ref']:

            data_x0 = lrt.lrt_read_uvspec_flx([init_x0])
            data_y0 = lrt.lrt_read_uvspec_flx([init_y0])
            prop_x = np.pi*prop_x/(np.squeeze(data_x0.f_down))
            prop_y = np.pi*prop_y/(np.squeeze(data_y0.f_down))

    elif prop_tag.lower() in ['transmittance', '_reflectance', 'absorptance', 'albedo-top', 'albedo-bottom']:

        data_x = lrt.lrt_read_uvspec_flx(inits_x)
        data_y = lrt.lrt_read_uvspec_flx(inits_y)

        # process calculations
        #╭────────────────────────────────────────────────╮#
        prop_x = cal_radiative_property(data_x.f_up, data_x.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        prop_y = cal_radiative_property(data_y.f_up, data_y.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        #╰────────────────────────────────────────────────╯#

    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # save RT calculations
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # create data directory
    #╭──────────────────────────────────────────────────────────────╮#
    fdir_lut = os.path.abspath(fdir_lut)

    if not os.path.exists(fdir_lut):
        os.system('mkdir -p %s' % fdir_lut)
    #╰──────────────────────────────────────────────────────────────╯#

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
    #╰────────────────────────────────────────────────────────────────────────────╯#



def retrieve(prop1, prop2, cld_tau, cld_ref, prop1_data, prop2_data):

    """
    under development

    inputs:
        prop1: observed property (e.g., reflectance) of x;
        prop2: observed property (e.g., reflectance) of y;
        cld_tau(N_tau): array of cloud optical thickness;
        cld_ref(N_ref): array of cloud effective radius;
        prop1_data(N_tau, N_ref): modeled property (e.g., reflectance) of x;
        prop2_data(N_tau, N_ref): modeled property (e.g., reflectance) of y;

    Method:
        1. Use matplotlib.path.contain_points to identify the LUT grid that the observational point falls into.
        2. Apply bilinear interpolation to obtain the cloud optical thickness and effective radius for the
        observed point from four adjacent modeled points.

    Notes:
        The cloud optical thickness and cloud effective radius will NOT be retrieved if the observed point is
        outside the LUT.

    Example:

    import h5py
    fname = 'data/0860nm_1620nm_00.h5'
    f = h5py.File(fname, 'r')
    prop1_data = f['wvl_0860'][...]
    prop2_data = f['wvl_1620'][...]
    cld_tau    = f['cld_tau'][...]
    cld_ref    = f['cld_ref'][...]
    f.close()

    a1, a2 = retrieve(0.5, 0.4, cld_tau, cld_ref, prop1_data, prop2_data)
    print(a1, a2)
    a1, a2 = retrieve(0.0, 0.0, cld_tau, cld_ref, prop1_data, prop2_data)
    print(a1, a2)
    """

    import numpy as np
    import matplotlib.path as mpl_path
    import multiprocessing as mp
    from scipy import interpolate

    # find the peripheral points of the whole LUT
    points_x1 = prop1_data[0, :]        ; points_y1 = prop2_data[0, :]
    points_x2 = prop1_data[:, -1]       ; points_y2 = prop2_data[:, -1]
    points_x3 = prop1_data[-1, :][::-1] ; points_y3 = prop2_data[-1, :][::-1]
    points_x4 = prop1_data[:, 0][::-1]  ; points_y4 = prop2_data[:, 0][::-1]

    points_x  = np.hstack((points_x1, points_x2, points_x3, points_x4))
    points_y  = np.hstack((points_y1, points_y2, points_y3, points_y4))
    points_xy = np.transpose(np.vstack((points_x, points_y)))
    grid_path_full = mpl_path.Path(points_xy, closed=True)

    # do cloud retrievals if observed point is inside the LUT
    if grid_path_full.contains_point((prop1, prop2)):

        Nx = cld_tau.size
        Ny = cld_ref.size

        # loop to find index_x for cloud optical thickness
        index_x   = 0
        points_x  = np.append(prop1_data[index_x, :], prop1_data[index_x+1, :][::-1])
        points_y  = np.append(prop2_data[index_x, :], prop2_data[index_x+1, :][::-1])
        points_xy = np.transpose(np.vstack((points_x, points_y)))
        grid_path = mpl_path.Path(points_xy, closed=True)

        while (not grid_path.contains_point((prop1, prop2))) and (index_x<=Nx-3):

            index_x  += 1
            points_x  = np.append(prop1_data[index_x, :], prop1_data[index_x+1, :][::-1])
            points_y  = np.append(prop2_data[index_x, :], prop2_data[index_x+1, :][::-1])
            points_xy = np.transpose(np.vstack((points_x, points_y)))
            grid_path = mpl_path.Path(points_xy, closed=True)

        # loop to find index_y for cloud effective radius
        index_y   = 0
        points_x  = np.append(prop1_data[:, index_y], prop1_data[:, index_y+1][::-1])
        points_y  = np.append(prop2_data[:, index_y], prop2_data[:, index_y+1][::-1])
        points_xy = np.transpose(np.vstack((points_x, points_y)))
        grid_path = mpl_path.Path(points_xy, closed=True)

        while (not grid_path.contains_point((prop1, prop2))) and (index_y<=Ny-3):

            index_y  += 1
            points_x  = np.append(prop1_data[:, index_y], prop1_data[:, index_y+1][::-1])
            points_y  = np.append(prop2_data[:, index_y], prop2_data[:, index_y+1][::-1])
            points_xy = np.transpose(np.vstack((points_x, points_y)))
            grid_path = mpl_path.Path(points_xy, closed=True)

        points = np.zeros((4, 2))
        points_tau = np.zeros(4)
        points_ref = np.zeros(4)

        points[0, :]  = np.array([prop1_data[index_x, index_y], prop2_data[index_x, index_y]])
        points_tau[0] = cld_tau[index_x]
        points_ref[0] = cld_ref[index_y]
        points[1, :]  = np.array([prop1_data[index_x, index_y+1], prop2_data[index_x, index_y+1]])
        points_tau[1] = cld_tau[index_x]
        points_ref[1] = cld_ref[index_y+1]
        points[2, :]  = np.array([prop1_data[index_x+1, index_y], prop2_data[index_x+1, index_y]])
        points_tau[2] = cld_tau[index_x+1]
        points_ref[2] = cld_ref[index_y]
        points[3, :]  = np.array([prop1_data[index_x+1, index_y+1], prop2_data[index_x+1, index_y+1]])
        points_tau[3] = cld_tau[index_x+1]
        points_ref[3] = cld_ref[index_y+1]

        tau = interpolate.griddata(points, points_tau, (prop1, prop2), method='linear')
        ref = interpolate.griddata(points, points_ref, (prop1, prop2), method='linear')

    else:

        tau = -1.0
        ref = -1.0

    return tau, ref



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
        prop_tag='reflectance',                                # property tag, can be "radiance", "reflectance", "albedo-top", "albedo-bottom", "_reflectance", "transmittance", "absorptance" (string)
        atmosphere_file='%s/afglus.dat' % er3t.common.fdir_data_atmmod, # atmosphere profile
        overwrite=True,
        ):

    # create temporary data directory
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir_tmp = os.path.abspath(fdir_tmp)
    if not os.path.exists(fdir_tmp):
        os.system('mkdir -p %s' % fdir_tmp)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # delete old files if overwrite is specified
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if overwrite:
        if len(glob.glob('%s/*.txt' % fdir_tmp)) > 0:
            os.system('find %s -name "*.txt" | xargs rm -f' % fdir_tmp)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # assign parameters
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvl_x, wvl_y = wavelength_pair
    alb_x, alb_y = surface_albedo_pair
    sza = solar_zenith_angle
    saa = solar_azimuth_angle
    vza = sensor_zenith_angle
    vaa = sensor_azimuth_angle
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # RT setup
    #╭────────────────────────────────────────────────────────────────────────────╮#

    # initialization
    #╭──────────────────────────────────────────────────────────────╮#
    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = atmosphere_file

    # cloud setup
    #╭────────────────────────────────────────────────╮#
    cld_cfg = lrt.get_cld_cfg()
    if cloud_type.lower() == 'water':
        cld_cfg['wc_properties'] = 'mie'
    else:
        msg = '\nError [gen_bispectral_lookup_table]: <cloud_type="%s"> is NOT supported.' % (cloud_type)
        raise OSError(msg)
    #╰────────────────────────────────────────────────╯#

    # aerosol setup
    #╭────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────╯#


    # for toa downwelling
    #╭────────────────────────────────────────────────╮#
    init_x0 = lrt.lrt_init_mono_flx(
            output_altitude='toa',
            input_file='%s/lrt_inpfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_x),
            output_file='%s/lrt_outfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_x),
            date=date,
            surface_albedo=alb_x,
            wavelength=wvl_x,
            solar_zenith_angle=sza,
            lrt_cfg=lrt_cfg,
            )

    init_y0 = lrt.lrt_init_mono_flx(
            output_altitude='toa',
            input_file='%s/lrt_inpfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_y),
            output_file='%s/lrt_outfile_%4.4dnm_toa.txt' % (fdir_tmp, wvl_y),
            date=date,
            surface_albedo=alb_y,
            wavelength=wvl_y,
            solar_zenith_angle=sza,
            lrt_cfg=lrt_cfg,
            )
    #╰────────────────────────────────────────────────╯#

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

            elif prop_tag.lower() in ['transmittance', '_reflectance', 'absorptance', 'albedo-top', 'albedo-bottom', 'all']:

                init_x = lrt.lrt_init_mono_flx(
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


                init_y = lrt.lrt_init_mono_flx(
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
    #╰──────────────────────────────────────────────────────────────╯#

    # run
    #╭──────────────────────────────────────────────────────────────╮#
    lrt.lrt_run_mp([init_x0, init_y0]+inits_x+inits_y)
    #╰──────────────────────────────────────────────────────────────╯#

    # read output
    #╭──────────────────────────────────────────────────────────────╮#
    if prop_tag.lower() in ['radiance', 'rad', 'reflectance', 'ref']:

        data_x = lrt.lrt_read_uvspec_rad(inits_x)
        data_y = lrt.lrt_read_uvspec_rad(inits_y)

        prop_x = data_x.rad[0, -1, :].reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        prop_y = data_y.rad[0, -1, :].reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))

        if prop_tag.lower() in ['reflectance', 'ref']:

            data_x0 = lrt.lrt_read_uvspec_flx([init_x0])
            data_y0 = lrt.lrt_read_uvspec_flx([init_y0])
            prop_x = np.pi*prop_x/(np.squeeze(data_x0.f_down))
            prop_y = np.pi*prop_y/(np.squeeze(data_y0.f_down))

    elif prop_tag.lower() in ['transmittance', '_reflectance', 'absorptance', 'albedo-top', 'albedo-bottom']:

        data_x = lrt.lrt_read_uvspec_flx(inits_x)
        data_y = lrt.lrt_read_uvspec_flx(inits_y)

        # process calculations
        #╭────────────────────────────────────────────────╮#
        prop_x = cal_radiative_property(data_x.f_up, data_x.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        prop_y = cal_radiative_property(data_y.f_up, data_y.f_down, tag=prop_tag).reshape((cloud_optical_thickness_all.size, cloud_effective_radius_all.size))
        #╰────────────────────────────────────────────────╯#

    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # save RT calculations
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # create data directory
    #╭──────────────────────────────────────────────────────────────╮#
    fdir_lut = os.path.abspath(fdir_lut)

    if not os.path.exists(fdir_lut):
        os.system('mkdir -p %s' % fdir_lut)
    #╰──────────────────────────────────────────────────────────────╯#

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
    #╰────────────────────────────────────────────────────────────────────────────╯#



class func_ref_vs_cot:

    def __init__(self,
            cot,
            cer0=10.0,
            fdir=er3t.common.params['fdir_tmp'],
            date=er3t.common.params['date'],
            wavelength=er3t.common.params['wavelength'],
            surface_albedo=er3t.common.params['surface_albedo'],
            atmospheric_profile=er3t.common.params['atmospheric_profile'],
            solar_file='%s/solar_16g_1.0nm.dat' % (er3t.common.fdir_data_solar),
            solar_zenith_angle=er3t.common.params['solar_zenith_angle'],
            solar_azimuth_angle=er3t.common.params['solar_azimuth_angle'],
            sensor_zenith_angle=er3t.common.params['sensor_zenith_angle'],
            sensor_azimuth_angle=er3t.common.params['sensor_azimuth_angle'],
            sensor_altitude=er3t.common.params['sensor_altitude'],
            cloud_top_height=2.0,
            cloud_geometrical_thickness=1.0,
            Ncpu=er3t.common.params['Ncpu'],
            output_tag=er3t.common.params['output_tag'],
            overwrite=er3t.common.params['overwrite'],
            ):


        self.cot  = cot
        self.cer0 = cer0
        self.wvl0 = wavelength
        self.sza0 = solar_zenith_angle
        self.saa0 = solar_azimuth_angle
        self.vza0 = sensor_zenith_angle
        self.vaa0 = sensor_azimuth_angle
        self.alt0 = sensor_altitude
        self.cth0 = cloud_top_height
        self.cbh0 = cloud_top_height-cloud_geometrical_thickness
        self.alb0 = surface_albedo
        self.fdir = fdir
        self.output_tag = output_tag
        self.cpu0 = Ncpu
        self.date0 = date
        self.fname_atm = atmospheric_profile
        self.fname_sol = solar_file

        self.mu0  = np.cos(np.deg2rad(self.sza0))
        self.ref_2s = er3t.util.cal_r_twostream(cot, a=self.alb0, mu=self.mu0)

        if not overwrite:
            try:
                self.load_all()
            except:
                self.run_all()
                self.load_all()
        else:
            self.run_all()
            self.load_all()

    def get_inits(self):


        # initialization
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # lrt setup
        #╭──────────────────────────────────────────────────────────────╮#
        lrt_cfg = lrt.get_lrt_cfg(spectral_resolution=1.0)
        lrt_cfg['atmosphere_file'] = self.fname_atm
        lrt_cfg['solar_file'] = self.fname_sol
        lrt_cfg['mol_abs_param'] = 'reptran medium'
        #╰──────────────────────────────────────────────────────────────╯#

        # cloud setup
        #╭──────────────────────────────────────────────────────────────╮#
        cld_cfg = lrt.get_cld_cfg()
        cld_cfg['wc_properties'] = 'mie'
        #╰──────────────────────────────────────────────────────────────╯#

        # aerosol setup
        #╭──────────────────────────────────────────────────────────────╮#
        aer_cfg = None
        #╰──────────────────────────────────────────────────────────────╯#
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # for toa downwelling
        #╭────────────────────────────────────────────────────────────────────────────╮#
        init_toa = lrt.lrt_init_mono_flx(
                output_altitude='toa',
                input_file='%s/lrt-toa-inp_%4.4dnm.txt' % (self.fdir, self.wvl0),
                output_file='%s/lrt-toa-out_%4.4dnm.txt' % (self.fdir, self.wvl0),
                date=self.date0,
                surface_albedo=self.alb0,
                wavelength=self.wvl0,
                solar_zenith_angle=self.sza0,
                spectral_resolution=1.0,
                lrt_cfg=lrt_cfg,
                )
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # for radiance
        #╭────────────────────────────────────────────────────────────────────────────╮#
        inits_rad = []

        for cot0 in self.cot:

            cld_cfg['cloud_file']              = '%s/lrt-cld_cot-%06.2f_cer-%06.2f.txt' % (self.fdir, cot0, self.cer0)
            cld_cfg['cloud_altitude']          = np.arange(self.cbh0, self.cth0+0.1, 0.1)
            cld_cfg['cloud_optical_thickness'] = cot0
            cld_cfg['cloud_effective_radius']  = self.cer0

            inp_file_rad = '%s/lrt-rad-inp_wvl-%4.4dnm_cot-%04.1f_cer-%04.1f_sza-%04.1f_alb-%04.2f.txt' % (self.fdir, self.wvl0, cot0, self.cer0, self.sza0, self.alb0)
            out_file_rad = '%s/lrt-rad-out_wvl-%4.4dnm_cot-%04.1f_cer-%04.1f_sza-%04.1f_alb-%04.2f.txt' % (self.fdir, self.wvl0, cot0, self.cer0, self.sza0, self.alb0)

            init_rad = lrt.lrt_init_mono_rad(
                    input_file=inp_file_rad,
                    output_file=out_file_rad,
                    date=self.date0,
                    surface_albedo=self.alb0,
                    wavelength=self.wvl0,
                    solar_zenith_angle=self.sza0,
                    solar_azimuth_angle=self.saa0,
                    sensor_zenith_angle=self.vza0,
                    sensor_azimuth_angle=self.vaa0,
                    spectral_resolution=1.0,
                    lrt_cfg=lrt_cfg,
                    cld_cfg=cld_cfg,
                    aer_cfg=aer_cfg,
                    output_altitude='toa',
                    )

            inits_rad.append(init_rad)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # inits
        #╭────────────────────────────────────────────────────────────────────────────╮#
        self.inits_toa = [init_toa]
        self.inits_rad = inits_rad
        #╰────────────────────────────────────────────────────────────────────────────╯#

    def run_all(self):

        os.system('rm -rf %s' % self.fdir)
        os.makedirs(self.fdir)

        self.get_inits()

        # run
        #╭────────────────────────────────────────────────────────────────────────────╮#
        lrt.lrt_run_mp(self.inits_toa+self.inits_rad)
        #╰────────────────────────────────────────────────────────────────────────────╯#

    def load_all(self):

        self.get_inits()

        # read output
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # radiance
        #╭──────────────────────────────────────────────────────────────╮#
        data_rad = lrt.lrt_read_uvspec_rad(self.inits_rad)
        self.rad = np.squeeze(data_rad.rad)
        #╰──────────────────────────────────────────────────────────────╯#

        # toa
        #╭──────────────────────────────────────────────────────────────╮#
        data_toa0 = lrt.lrt_read_uvspec_flx(self.inits_toa)
        self.toa0 = np.squeeze(data_toa0.f_down)
        #╰──────────────────────────────────────────────────────────────╯#

        # reflectance
        #╭──────────────────────────────────────────────────────────────╮#
        self.ref = np.pi*self.rad/self.toa0
        #╰──────────────────────────────────────────────────────────────╯#
        #╰────────────────────────────────────────────────────────────────────────────╯#

    def get_cot_from_ref(self, ref, method='cubic', mode='rt'):

        if mode == '2s':
            f = interp1d(self.ref_2s, self.cot, kind=method, bounds_error=False, fill_value='extrapolate')
        elif mode == 'rt':
            f = interp1d(self.ref, self.cot, kind=method, bounds_error=False, fill_value='extrapolate')

        return f(ref)

    def get_ref_from_cot(self, cot, method='cubic', mode='rt'):

        if mode == '2s':
            f = interp1d(self.cot, self.ref_2s, kind=method, bounds_error=False)
        elif mode == 'rt':
            f = interp1d(self.cot, self.ref, kind=method, bounds_error=False)

        return f(cot)



if __name__ == '__main__':

    pass
