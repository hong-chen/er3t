"""
by Hong Chen (hong.chen.cu@gmail.com)

This code has been tested under:
    1) Linux on 2023-06-27 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64
"""

import os
import sys
import h5py
import numpy as np
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
mpl.use('Agg')


import er3t




# global variables
#/--------------------------------------------------------------\#
name_tag = '00_er3t_lrt'
fdir0    = er3t.common.fdir_examples
#\--------------------------------------------------------------/#




def test_flux_01_clear_sky():

    """
    This following is an example for clear-sky calculation with default parameterization.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)


    init = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp
            )

    er3t.rtm.lrt.lrt_run(init)

    data = er3t.rtm.lrt.lrt_read_uvspec_flx([init])

    # the flux calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Upwelling flux: ', np.squeeze(data.f_up))
    print('  Downwelling flux: ', np.squeeze(data.f_down))
    print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    print()

def test_flux_02_clear_sky():

    """
    The following example is similar to Example 1 but with user's input of surface albedo, solar zenith angle,
    wavelength etc.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    init = er3t.rtm.lrt.lrt_init_mono_flx(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp,
            date        = datetime.datetime(2014, 9, 11),
            surface_albedo     = 0.8,
            solar_zenith_angle = 70.0,
            wavelength         = 532.31281,
            output_altitude    = 5.0,
            lrt_cfg            = lrt_cfg,
            )
    er3t.rtm.lrt.lrt_run(init)

    data = er3t.rtm.lrt.lrt_read_uvspec_flx([init])

    # the flux calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Upwelling flux: ', np.squeeze(data.f_up))
    print('  Downwelling flux: ', np.squeeze(data.f_down))
    print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    print()

def test_flux_03_clear_sky():

    """
    The following example is similar to Example 2 but for multiple calculations at different solar zenith angles.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    inits = []
    for i, sza0 in enumerate(sza):
        init = er3t.rtm.lrt.lrt_init_mono_flx(
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
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_flx(inits)

    # the flux calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Upwelling flux: ', np.squeeze(data.f_up))
    print('  Downwelling flux: ', np.squeeze(data.f_down))
    print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    print()

def test_flux_04_cloud():

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file'] = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    inits = []
    for i, sza0 in enumerate(sza):

        init = er3t.rtm.lrt.lrt_init_mono_flx(
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
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_flx(inits)

    # the flux calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Upwelling flux: ', np.squeeze(data.f_up))
    print('  Downwelling flux: ', np.squeeze(data.f_down))
    print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    print()

def test_flux_05_cloud_and_aerosol():

    """
    The following example is similar to Example 4 but for cloud and aerosol calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km and
                   a homogeneous aerosol layer (AOD=0.4, ASY=0.6, SSA=0.85) located at 3.0 to 6.0 km
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    aer_cfg = er3t.rtm.lrt.get_aer_cfg()
    aer_cfg['aerosol_file'] = '%s/aerosol.txt' % fdir_tmp
    aer_cfg['aerosol_optical_depth']    = 0.4
    aer_cfg['asymmetry_parameter']      = 0.6
    aer_cfg['single_scattering_albedo'] = 0.85
    aer_cfg['aerosol_altitude'] = np.arange(3.0, 6.1, 0.2)

    inits = []
    for i, sza0 in enumerate(sza):

        init = er3t.rtm.lrt.lrt_init_mono_flx(
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
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_flx(inits)

    # the flux calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Upwelling flux: ', np.squeeze(data.f_up))
    print('  Downwelling flux: ', np.squeeze(data.f_down))
    print('  Down-diffuse flux: ', np.squeeze(data.f_down_diffuse))
    print('  Down-direct flux: ', np.squeeze(data.f_down_direct))
    print()




def test_rad_01_clear_sky():

    """
    This following is an example for clear-sky calculation with default parameterization.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    init = er3t.rtm.lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt' % fdir_tmp,
            output_file = '%s/output.txt' % fdir_tmp
            )
    er3t.rtm.lrt.lrt_run(init)

    data = er3t.rtm.lrt.lrt_read_uvspec_rad([init])

    # the radiance calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Radiance: ', np.squeeze(data.rad))
    print()

def test_rad_02_clear_sky():

    """
    The following example is similar to Example 1 but with user's input of surface albedo,
    solar zenith angle, solar azimuth angle, sensor zenith angle, sensor azimuth angle
    wavelength etc.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    init = er3t.rtm.lrt.lrt_init_mono_rad(
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
    er3t.rtm.lrt.lrt_run(init)

    data = er3t.rtm.lrt.lrt_read_uvspec_rad([init])

    # the radiance calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Radiance: ', np.squeeze(data.rad))
    print()

def test_rad_03_clear_sky():

    """
    The following example is similar to Example 2 but for multiple calculations at different solar zenith angles.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    inits = []
    for i, sza0 in enumerate(sza):
        init = er3t.rtm.lrt.lrt_init_mono_rad(
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
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_rad(inits)

    # the radiance calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Radiance: ', np.squeeze(data.rad))
    print()

def test_rad_04_cloud():

    """
    The following example is similar to Example 3 but for cloud calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file'] = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    inits = []
    for i, sza0 in enumerate(sza):

        init = er3t.rtm.lrt.lrt_init_mono_rad(
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
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_rad(inits)

    # the radiance calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Radiance: ', np.squeeze(data.rad))
    print()

def test_rad_05_cloud_and_aerosol():

    """
    The following example is similar to Example 4 but for cloud and aerosol calculations.
    Assume we have a homogeneous cloud layer (COT=10.0, CER=12.0) located at 0.5 to 1.0 km and
                   a homogeneous aerosol layer (AOD=0.4, ASY=0.6, SSA=0.85) located at 3.0 to 6.0 km
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir_tmp = '%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    sza = np.arange(60.0, 65.1, 0.5)

    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = 10.0
    cld_cfg['cloud_effective_radius']  = 12.0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)

    aer_cfg = er3t.rtm.lrt.get_aer_cfg()
    aer_cfg['aerosol_file'] = '%s/aerosol.txt' % fdir_tmp
    aer_cfg['aerosol_optical_depth']    = 0.4
    aer_cfg['asymmetry_parameter']      = 0.6
    aer_cfg['single_scattering_albedo'] = 0.85
    aer_cfg['aerosol_altitude'] = np.arange(3.0, 6.1, 0.2)

    inits = []
    for i, sza0 in enumerate(sza):

        init = er3t.rtm.lrt.lrt_init_mono_rad(
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
    er3t.rtm.lrt.lrt_run_mp(inits, Ncpu=6)

    data = er3t.rtm.lrt.lrt_read_uvspec_rad(inits)

    # the radiance calculated can be accessed through
    print('Results for <%s>:' % _metadata['Function'])
    print('  Radiance: ', np.squeeze(data.rad))
    print()





def example_rad_01_sun_glint(
        wvl0=532.0,
        sza0=60.0,
        saa0=0.0,
        vza0=60.0,
        plot=True
        ):

    """
    This example code is used to provide simulated radiance for ocean BRDF surface.
    """

    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    job_tag  = 'sza%3.3d_saa%3.3d_vza%3.3d' % (sza0, saa0, vza0)
    fdir_tmp = '%s/tmp-data/%s/%s/%s' % (fdir0, name_tag, _metadata['Function'], job_tag)
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)

    vaa = np.arange(0.0, 361.0, 5.0)

    # rt initialization
    #/----------------------------------------------------------------------------\#
    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    input_dict_extra = {'brdf_cam': 'u10 1'}
    #\----------------------------------------------------------------------------/#

    # rt setup
    #/----------------------------------------------------------------------------\#
    init = er3t.rtm.lrt.lrt_init_mono_rad(
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
    #\----------------------------------------------------------------------------/#

    # run rt
    #/----------------------------------------------------------------------------\#
    print('Running calculations for <%s> ...' % (_metadata['Function']))
    er3t.rtm.lrt.lrt_run(init)
    #\----------------------------------------------------------------------------/#

    # read output
    #/--------------------------------------------------------------\#
    data = er3t.rtm.lrt.lrt_read_uvspec_rad([init])
    rad  = np.squeeze(data.rad)
    #\--------------------------------------------------------------/#


    if plot:
        #/----------------------------------------------------------------------------\#
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.plot(np.deg2rad(vaa), rad, color='red', lw=6.0, zorder=1)

        ax1.scatter(np.deg2rad(saa0), rad.max()*1.1, s=400, c='orange', lw=0.0, alpha=0.8)

        ax1.set_title('Radiance at %d nm (SZA=%d$^\circ$, SAA=%d$^\circ$, VZA=%d$^\circ$)' % (wvl0, sza0, saa0, vza0), y=1.08)

        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)

        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], job_tag)
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
        #\----------------------------------------------------------------------------/#

        print('Results for <%s> is saved at <%s>.' % (_metadata['Function'], fname_png))
        print()

def example_rad_02_libera_adm(
        sza0=60.0,
        saa0=0.0,
        vza0=45.0,
        cot0=10.0,
        cer0=12.0,
        vaa=np.arange(0.0, 361.0, 5.0),
        wvl=np.arange(350.0, 701.0, 5.0),
        plot=True,
        ):

    """
    This example code is used to provide simulated ADM at 555 nm and VIS band (350 - 700 nm).

    Default parameter settings can be used to produce Figure 8d in

    Gristey, J. J., Schmidt, K. S., Chen, H., Feldman, D. R., Kindel, B. C., Mauss, J., van den Heever,
    M., Hakuba, M. Z., and Pilewskie, P.: Angular Sampling of a Monochromatic, Wide-Field-of-View Camera
    to Augment Next-Generation Earth Radiation Budget Satellite Observations, Atmos. Meas. Tech. Discuss.
    [preprint], https://doi.org/10.5194/amt-2023-7, in review, 2023.
    """

    # create tmp directory
    #/----------------------------------------------------------------------------\#
    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    job_tag  = 'sza%3.3d_saa%3.3d_vza%3.3d_cot%3.3d_cer%3.3d' % (sza0, saa0, vza0, cot0, cer0)
    fdir_tmp = '%s/tmp-data/%s/%s/%s' % (fdir0, name_tag, _metadata['Function'], job_tag)
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    #\----------------------------------------------------------------------------/#


    # rt setup
    #/----------------------------------------------------------------------------\#
    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')

    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
    cld_cfg['cloud_file']  = '%s/cloud.txt' % fdir_tmp
    cld_cfg['cloud_optical_thickness'] = cot0
    cld_cfg['cloud_effective_radius']  = cer0
    cld_cfg['cloud_altitude'] = np.arange(0.5, 1.1, 0.1)
    #\----------------------------------------------------------------------------/#


    # get initializations
    #/----------------------------------------------------------------------------\#
    inits_rad = []
    inits_flx = []
    for i, wvl0 in enumerate(wvl):

        # lrt initialization (radiance)
        #/----------------------------------------------------------------------------\#
        init_rad = er3t.rtm.lrt.lrt_init_mono_rad(
                input_file  = '%s/input_%4.4dnm_rad.txt'  % (fdir_tmp, wvl0),
                output_file = '%s/output_%4.4dnm_rad.txt' % (fdir_tmp, wvl0),
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
                )
        inits_rad.append(init_rad)
        #\----------------------------------------------------------------------------/#


        # lrt initialization (flux)
        #/----------------------------------------------------------------------------\#
        init_flx = er3t.rtm.lrt.lrt_init_mono_flx(
                input_file  = '%s/input_%4.4dnm_flx.txt'  % (fdir_tmp, wvl0),
                output_file = '%s/output_%4.4dnm_flx.txt' % (fdir_tmp, wvl0),
                date        = datetime.datetime(2014, 9, 11),
                surface_albedo     = 0.03,
                solar_zenith_angle = sza0,
                wavelength         = wvl0,
                output_altitude    = 'toa',
                lrt_cfg            = lrt_cfg,
                cld_cfg            = cld_cfg,
                )
        inits_flx.append(init_flx)
        #\----------------------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # run rt
    #/----------------------------------------------------------------------------\#
    print('Running calculations for <%s> ...' % (_metadata['Function']))
    er3t.rtm.lrt.lrt_run_mp(inits_rad, Ncpu=12)
    er3t.rtm.lrt.lrt_run_mp(inits_flx, Ncpu=12)
    #\----------------------------------------------------------------------------/#


    # get output
    #/----------------------------------------------------------------------------\#
    data_ = er3t.rtm.lrt.lrt_read_uvspec_flx(inits_flx)
    toa   = np.squeeze(data_.f_down)

    rad = np.zeros((wvl.size, vaa.size), dtype=np.float64)
    ref  = np.zeros_like(rad)
    for i, wvl0 in enumerate(wvl):
        try:
            data = er3t.rtm.lrt.lrt_read_uvspec_rad([inits_rad[i]])
            rad[i, :]  = np.squeeze(data.rad)
            ref[i, :]  = np.pi*rad[i, :]/toa[i]
        except:
            rad[i, :]  = np.nan
            ref[i, :]  = np.nan
    #\----------------------------------------------------------------------------/#


    # rad_555/ref_555
    #/----------------------------------------------------------------------------\#
    wvl0 = 555.0
    iwvl = np.argmin(np.abs(wvl-wvl0))
    rad_555 = rad[iwvl, :]
    ref_555 = ref[iwvl, :]
    #\----------------------------------------------------------------------------/#


    # rad_vis/ref_vis
    #/----------------------------------------------------------------------------\#
    ref_vis = np.zeros(vaa.size, dtype=np.float64)
    rad_vis = np.zeros(vaa.size, dtype=np.float64)
    for i in range(vaa.size):
        rad_vis[i] = np.trapz(rad[:, i], x=wvl)
        ref_vis[i] = np.pi*np.trapz(rad[:, i], x=wvl) / np.trapz(toa, x=wvl)
    #\----------------------------------------------------------------------------/#


    # write output file
    #/----------------------------------------------------------------------------\#
    fname = '%s/libera_adm_%s.h5' % (fdir_tmp, job_tag)
    f = h5py.File(fname, 'w')
    f['wvl'] = wvl
    f['vaa'] = vaa
    f['vza0'] = vza0
    f['sza0'] = sza0
    f['saa0'] = saa0
    f['cot0'] = cot0
    f['cer0'] = cer0
    f['rad'] = rad
    f['ref'] = ref
    f['toa'] = toa
    f['rad_555'] = rad_555
    f['ref_555'] = ref_555
    f['rad_vis'] = rad_vis
    f['ref_vis'] = ref_vis
    f.close()
    #\----------------------------------------------------------------------------/#

    if plot:

        #/----------------------------------------------------------------------------\#
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection='polar', aspect='equal')

        # 555 nm
        #/----------------------------------------------------------------------------\#
        ax1.plot(np.deg2rad(vaa), ref_555, color='darkblue', lw=12.0, alpha=1.0, zorder=0)
        #\----------------------------------------------------------------------------/#

        # vis band
        #/----------------------------------------------------------------------------\#
        scale_factor = ref_555[0]/ref_vis[0]
        ax1.plot(np.deg2rad(vaa), ref_vis*scale_factor, color='orange', ls='--', lw=6.0, zorder=1)
        #\----------------------------------------------------------------------------/#

        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)

        patches_legend = [
                          mpatches.Patch(color='darkblue' , label='555 nm'), \
                          mpatches.Patch(color='orange'   , label='Rescaled VIS'), \
                         ]
        ax1.legend(handles=patches_legend, loc='center', fontsize=16)

        ax1.set_rmax(1.0)
        ax1.set_rlabel_position(180.0)  # Move radial labels away from plotted line

        ax1.set_title('Anisotropic Reflectance (SZA=%d$^\circ$, VZA=%d$^\circ$, COT=%d, SF=%.2f)' % (sza0, vza0, cot0, scale_factor), y=1.08)

        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], job_tag)
        fig.savefig(fname_png, bbox_inches='tight', metadata=_metadata)
        #\----------------------------------------------------------------------------/#

        print('Results for <%s> is saved at <%s>.' % (_metadata['Function'], fname_png))
        print()



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


    example_rad_01_sun_glint(wvl0=532.0, sza0=60.0, saa0=0.0, vza0=60.0, plot=True)
    example_rad_02_libera_adm(sza0=60.0, saa0=0.0, vza0=45.0, cot0=10.0, cer0=12.0, plot=True)

    pass
