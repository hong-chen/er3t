import os
import sys
import numpy as np
import datetime
import er3t.rtm.lrt as lrt

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches



def test_flux_01_clear_sky():

    """
    This following is an example for clear-sky calculation with default parameterization.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_flux_01_clear_sky'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)


    init = lrt.lrt_init_mono(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp
            )

    lrt.lrt_run(init)

    data = lrt.lrt_read_uvspec([init])

    # the flux calculated can be accessed through
    print(data.f_up)
    print(data.f_down)
    print(data.f_down_diffuse)
    print(data.f_down_direct)

def test_flux_02_clear_sky():

    """
    The following example is similar to Example 1 but with user's input of surface albedo, solar zenith angle,
    wavelength etc.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_flux_02_clear_sky'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    init = lrt.lrt_init_mono(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp,
            date        = datetime.datetime(2014, 9, 11),
            surface_albedo     = 0.8,
            solar_zenith_angle = 70.0,
            wavelength         = 532.31281,
            output_altitude    = 5.0,
            lrt_cfg            = lrt_cfg,
            )
    lrt.lrt_run(init)

    data = lrt.lrt_read_uvspec([init])

    # the flux calculated can be accessed through
    print(data.f_up)
    print(data.f_down)
    print(data.f_down_diffuse)
    print(data.f_down_direct)

def test_flux_03_clear_sky():

    """
    The following example is similar to Example 2 but for multiple calculations at different solar zenith angles.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_flux_03_clear_sky'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    inits = []
    for i, sza0 in enumerate(sza):
        init = lrt.lrt_init_mono(
                input_file  = '%s/input%2.2d.txt' % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.8,
                solar_zenith_angle = sza0,
                wavelength         = 532.31281,
                output_altitude    = 5.0,
                lrt_cfg            = lrt_cfg
                )
        inits.append(init)

    # run with multi cores
    lrt.lrt_run_mp(inits, ncpu=6)

    data = lrt.lrt_read_uvspec(inits)

    # the flux calculated can be accessed through
    print(data.f_up)
    print(data.f_down)
    print(data.f_down_diffuse)
    print(data.f_down_direct)

def test_flux_04_cloud():

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_flux_04_cloud'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = lrt.get_cld_cfg()
    cld_cfg['cloud_file'] = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    inits = []
    for i, sza0 in enumerate(sza):

        init = lrt.lrt_init_mono(
                input_file  = '%s/input%2.2d.txt'  % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.8,
                solar_zenith_angle = sza0,
                wavelength         = 532.31281,
                output_altitude    = 5.0,
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg
                )
        inits.append(init)

    # run with multi cores
    lrt.lrt_run_mp(inits, ncpu=6)

    data = lrt.lrt_read_uvspec(inits)

    # the flux calculated can be accessed through
    print(data.f_up)
    print(data.f_down)
    print(data.f_down_diffuse)
    print(data.f_down_direct)

def test_flux_05_cloud_and_aerosol():

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_flux_05_cloud_and_aerosol'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    aer_cfg = lrt.get_aer_cfg()
    aer_cfg['aerosol_file'] = '%s/aerosol.txt' % fdir_tmp
    aer_cfg['aerosol_optical_depth']    = 0.4
    aer_cfg['asymmetry_parameter']      = 0.6
    aer_cfg['single_scattering_albedo'] = 0.85
    aer_cfg['aerosol_altitude'] = np.arange(3.0, 6.1, 0.2)

    inits = []
    for i, sza0 in enumerate(sza):

        init = lrt.lrt_init_mono(
                input_file  = '%s/input%2.2d.txt'  % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.8,
                solar_zenith_angle = sza0,
                wavelength         = 532.31281,
                output_altitude    = 5.0,
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg,
                aer_cfg            = aer_cfg
                )
        inits.append(init)

    # run with multi cores
    lrt.lrt_run_mp(inits, ncpu=6)

    data = lrt.lrt_read_uvspec(inits)

    # the flux calculated can be accessed through
    print(data.f_up)
    print(data.f_down)
    print(data.f_down_diffuse)
    print(data.f_down_direct)




def test_rad_01_clear_sky():

    """
    This following is an example for clear-sky calculation with default parameterization.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_rad_01_clear_sky'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    init = lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp
            )
    lrt.lrt_run(init)

    data = lrt.lrt_read_uvspec_rad([init])

    # the radiance calculated can be accessed through
    print(data.rad)

def test_rad_02_clear_sky():

    """
    The following example is similar to Example 1 but with user's input of surface albedo,
    solar zenith angle, solar azimuth angle, sensor zenith angle, sensor azimuth angle
    wavelength etc.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_rad_02_clear_sky'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    init = lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp,
            date        = datetime.datetime(2014, 9, 11),
            surface_albedo     = 0.03,
            solar_zenith_angle = 45.0,
            solar_azimuth_angle = 0.0,
            sensor_zenith_angle = 45.0,
            sensor_azimuth_angle = 45.0,
            wavelength         = 532.31281,
            output_altitude    = 'toa',
            lrt_cfg            = lrt_cfg,
            )
    lrt.lrt_run(init)

    data = lrt.lrt_read_uvspec_rad([init])

    # the radiance calculated can be accessed through
    print(data.rad)

def test_rad_03_clear_sky():

    """
    The following example is similar to Example 2 but for multiple calculations at different solar zenith angles.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_rad_03_clear_sky'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    inits = []
    for i, sza0 in enumerate(sza):
        init = lrt.lrt_init_mono_rad(
                input_file  = '%s/input%2.2d.txt' % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.03,
                solar_zenith_angle = sza0,
                solar_azimuth_angle = 30.0,
                sensor_zenith_angle = 30.0,
                sensor_azimuth_angle = 0.0,
                wavelength         = 532.31281,
                output_altitude    = 'toa',
                lrt_cfg            = lrt_cfg
                )
        inits.append(init)

    # run with multi cores
    lrt.lrt_run_mp(inits, ncpu=6)

    data = lrt.lrt_read_uvspec_rad(inits)

    # the radiance calculated can be accessed through
    print(data.rad)

def test_rad_04_cloud():

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_rad_04_cloud'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = lrt.get_cld_cfg()
    cld_cfg['cloud_file'] = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    inits = []
    for i, sza0 in enumerate(sza):

        init = lrt.lrt_init_mono_rad(
                input_file  = '%s/input%2.2d.txt'  % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.03,
                solar_zenith_angle = sza0,
                solar_azimuth_angle = 30.0,
                sensor_zenith_angle = 30.0,
                sensor_azimuth_angle = 0.0,
                wavelength         = 532.31281,
                output_altitude    = 'toa',
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg
                )
        inits.append(init)

    # run with multi cores
    lrt.lrt_run_mp(inits, ncpu=6)

    data = lrt.lrt_read_uvspec_rad(inits)

    # the radiance calculated can be accessed through
    print(data.rad)

def test_rad_05_cloud_and_aerosol():

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/test_rad_05_cloud_and_aerosol'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    aer_cfg = lrt.get_aer_cfg()
    aer_cfg['aerosol_file'] = '%s/aerosol.txt' % fdir_tmp
    aer_cfg['aerosol_optical_depth']    = 0.4
    aer_cfg['asymmetry_parameter']      = 0.6
    aer_cfg['single_scattering_albedo'] = 0.85
    aer_cfg['aerosol_altitude'] = np.arange(3.0, 6.1, 0.2)

    inits = []
    for i, sza0 in enumerate(sza):

        init = lrt.lrt_init_mono_rad(
                input_file  = '%s/input%2.2d.txt'  % (fdir_tmp, i),
                output_file = '%s/output%2.2d.txt' % (fdir_tmp, i),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.8,
                solar_zenith_angle = sza0,
                solar_azimuth_angle = 30.0,
                sensor_zenith_angle = 30.0,
                sensor_azimuth_angle = 0.0,
                wavelength         = 532.31281,
                output_altitude    = 5.0,
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg,
                aer_cfg            = aer_cfg
                )
        inits.append(init)

    # run with multi cores
    lrt.lrt_run_mp(inits, ncpu=6)

    data = lrt.lrt_read_uvspec_rad(inits)

    # the radiance calculated can be accessed through
    print(data.rad)





def example_rad_01_sun_glint(wvl0=532.0, sza0=60.0, saa0=0.0, vza0=60.0):

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/example_rad_01_sun_glint'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    vaa = np.arange(0.0, 361.0, 5.0)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    input_dict_extra = {'brdf_cam': 'u10 1'}

    # radiance calculations without aerosol
    # ===========================================================================================
    init = lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt'  % (fdir_tmp),
            output_file = '%s/output.txt' % (fdir_tmp),
            date        = datetime.datetime(2014, 9, 11),
            surface_albedo     = 0.03,
            solar_zenith_angle = sza0,
            solar_azimuth_angle = saa0,
            sensor_zenith_angle = vza0,
            sensor_azimuth_angle = vaa,
            wavelength         = wvl0,
            output_altitude    = 'toa',
            input_dict_extra   = input_dict_extra,
            mute_list          = ['albedo'],
            lrt_cfg            = lrt_cfg,
            cld_cfg            = None,
            aer_cfg            = None,
            )

    # run with multi cores
    lrt.lrt_run(init)

    data = lrt.lrt_read_uvspec_rad([init])
    rad  = np.squeeze(data.rad)
    # ===========================================================================================


    # =============================================================================
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='polar')
    ax1.plot(np.deg2rad(vaa), rad, color='r')

    ax1.scatter(np.deg2rad(saa0), rad.max()*1.1, s=400, c='orange', lw=0.0, alpha=0.8)

    ax1.set_title('Radiance at %d nm (SZA=%d$^\circ$, SAA=%d$^\circ$, VZA=%d$^\circ$)' % (wvl0, sza0, saa0, vza0), y=1.08)

    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)

    plt.savefig('00_er3t_lrt-example_rad_01.png', bbox_inches='tight')

    plt.show()
    plt.close(fig)
    # =============================================================================

def example_rad_02_anisotropy(wvl0=532.0, sza0=60.0, saa0=0.0, vza0=60.0):

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    fdir_tmp = 'tmp-data/00_er3t_lrt/example_rad_02_anisotropy'
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    vaa = np.arange(0.0, 361.0, 5.0)

    lrt_cfg = lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    aer_cfg = lrt.get_aer_cfg()
    aer_cfg['aerosol_file'] = '%s/aerosol.txt' % fdir_tmp
    aer_cfg['aerosol_optical_depth']    = 0.2
    aer_cfg['asymmetry_parameter']      = 0.6
    aer_cfg['single_scattering_albedo'] = 0.85
    aer_cfg['aerosol_altitude'] = np.arange(3.0, 6.1, 0.2)

    input_dict_extra = {'brdf_cam': 'u10 1'}

    # radiance calculations without aerosol
    # ===========================================================================================
    init = lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt'  % (fdir_tmp),
            output_file = '%s/output.txt' % (fdir_tmp),
            date        = datetime.datetime(2014, 9, 11),
            surface_albedo     = 0.03,
            solar_zenith_angle = sza0,
            solar_azimuth_angle = saa0,
            sensor_zenith_angle = vza0,
            sensor_azimuth_angle = vaa,
            wavelength         = wvl0,
            output_altitude    = 'toa',
            mute_list          = ['albedo'],
            input_dict_extra   = input_dict_extra,
            lrt_cfg            = lrt_cfg,
            cld_cfg            = cld_cfg,
            aer_cfg            = None,
            )

    # run with multi cores
    lrt.lrt_run(init)

    data1 = lrt.lrt_read_uvspec_rad([init])
    rad1  = np.squeeze(data1.rad)
    # ===========================================================================================

    # radiance calculations with aerosol
    # ===========================================================================================
    init = lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt'  % (fdir_tmp),
            output_file = '%s/output.txt' % (fdir_tmp),
            date        = datetime.datetime(2014, 9, 11),
            surface_albedo     = 0.03,
            solar_zenith_angle = sza0,
            solar_azimuth_angle = saa0,
            sensor_zenith_angle = vza0,
            sensor_azimuth_angle = vaa,
            wavelength         = wvl0,
            output_altitude    = 'toa',
            lrt_cfg            = lrt_cfg,
            cld_cfg            = cld_cfg,
            aer_cfg            = aer_cfg,
            )

    # run with multi cores
    lrt.lrt_run(init)

    data2 = lrt.lrt_read_uvspec_rad([init])
    rad2  = np.squeeze(data2.rad)
    # ===========================================================================================


    # =============================================================================
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='polar')
    ax1.plot(np.deg2rad(vaa), rad1, color='r')
    ax1.plot(np.deg2rad(vaa), rad2, color='b')

    ax1.scatter(np.deg2rad(saa0), max([rad1.max()*1.1, rad2.max()*1.1]), s=400, c='orange', lw=0.0, alpha=0.8)

    patches_legend = [
                mpatches.Patch(color='red'   , label='Cloud'),
                mpatches.Patch(color='blue'  , label='Cloud+Aerosol')
                ]
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

    ax1.set_title('Radiance at %d nm (SZA=%d$^\circ$, SAA=%d$^\circ$, VZA=%d$^\circ$)' % (wvl0, sza0, saa0, vza0), y=1.08)

    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)

    plt.savefig('00_er3t_lrt-example_rad_02.png', bbox_inches='tight')

    plt.show()
    plt.close(fig)
    # =============================================================================





if __name__ == '__main__':


    test_flux_01_clear_sky()
    test_flux_02_clear_sky()
    test_flux_03_clear_sky()
    test_flux_04_cloud()
    test_flux_05_cloud_and_aerosol()


    test_rad_01_clear_sky()
    test_rad_02_clear_sky()
    test_rad_03_clear_sky()
    test_rad_04_cloud()
    test_rad_05_cloud_and_aerosol()


    example_rad_01_sun_glint(wvl0=532.0, sza0=60.0, saa0=0.0, vza0=60.0)
    example_rad_02_anisotropy(wvl0=532.0, sza0=60.0, saa0=0.0, vza0=60.0)

    pass
