import os
import sys
import datetime
import importlib.util

has_mcarats     = ('MCARATS_V010_EXE' in dict(os.environ))
has_libradtran  = ('LIBRADTRAN_V2_DIR' in dict(os.environ))
has_token       = ('EARTHDATA_TOKEN' in dict(os.environ))
has_netcdf4     = (importlib.util.find_spec('netCDF4') is not None)
has_hdf4        = (importlib.util.find_spec('pyhdf') is not None)
has_hdf5        = (importlib.util.find_spec('h5py') is not None)
has_xarray      = (importlib.util.find_spec('xarray') is not None)
has_mpi         = False

fdir_er3t        = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

fdir_data        = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
fdir_data_solar  = os.path.join(fdir_data, 'solar')
fdir_data_abs    = os.path.join(fdir_data, 'abs')
fdir_data_pha    = os.path.join(fdir_data, 'pha')
fdir_data_atmmod = os.path.join(fdir_data, 'atmmod')
fdir_data_slit   = os.path.join(fdir_data, 'slit')
fdir_data_ssfr   = os.path.join(fdir_data_slit, 'ssfr')

fdir_data_tmp    = os.path.join(fdir_er3t, 'tmp-data')
fdir_examples    = os.path.join(fdir_er3t, 'examples')
fdir_tests       = os.path.join(fdir_er3t, 'tests')
fdir_logs        = os.path.join(fdir_er3t, 'logs')

params = {
                 'wavelength': 650.0,
                       'date': datetime.datetime.now(),
         'solar_zenith_angle': 0.0,
        'solar_azimuth_angle': 0.0,
        'sensor_zenith_angle': 0.0,
       'sensor_azimuth_angle': 0.0,
            'sensor_altitude': 705000.0,
                     'target': '3d radiance',
                     'solver': 'mcarats',
        'atmospheric_profile': '%s/afglus.dat' % fdir_data_atmmod,
                 'absorption': 'abs_16g',
             'surface_albedo': 0.03,
                'phase_cloud': 'mie',
                    'Nphoton': 1e8,
                       'Ncpu': 12,
                   'fdir_tmp': 'tmp-data/%s' % datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                 'output_tag': 'rtm-out_rad-3d',
                  'overwrite': True,
                    'verbose': True,
               'earth_radius': 6371.009,
        }

references = [
'EaR³T (Chen et al., 2023):\n- Chen, H., Schmidt, K. S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold, G., Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) - Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals, Atmos. Meas. Tech., 16, 1971–2000, https://doi.org/10.5194/amt-16-1971-2023, 2023.'
        ]

_today_dt    = datetime.datetime.now(datetime.timezone.utc)
_today_dt    = _today_dt.replace(tzinfo=None) # so that timedelta does not raise an error
_date_today_ = _today_dt.strftime('%d %B, %Y')

# sdown references
_sat_tags_support_ = {

        'MODRGB': {
                'dataset_tag': 'MODRGB',
                   'dict_key': 'mod_rgb',
                'description': 'Terra MODIS True Color (RGB) Imagery',
                    'website': 'https://worldview.earthdata.nasa.gov',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'We acknowledge the use of imagery from the NASA Worldview application (https://worldview.earthdata.nasa.gov/), part of the NASA Earth Observing System Data and Information System (EOSDIS).',
                },

        'MYDRGB': {
                'dataset_tag': 'MYDRGB',
                   'dict_key': 'myd_rgb',
                'description': 'Aqua MODIS True Color (RGB) Imagery',
                    'website': 'https://worldview.earthdata.nasa.gov',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'We acknowledge the use of imagery from the NASA Worldview application (https://worldview.earthdata.nasa.gov/), part of the NASA Earth Observing System Data and Information System (EOSDIS).',
                },

        'MOD03': {
                'dataset_tag': '61/MOD03',
                   'dict_key': 'mod_03',
                'description': 'Terra MODIS Geolocation Fields Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD03.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS Geolocation Fields Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MOD03.061 (last access: %s), 2017.' % _date_today_,
                },

        'MYD03': {
                'dataset_tag': '61/MYD03',
                   'dict_key': 'myd_03',
                'description': 'Aqua MODIS Geolocation Fields Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD03.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS Geolocation Fields Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MYD03.061 (last access: %s), 2017.' % _date_today_,
                },

        'MOD02QKM': {
                'dataset_tag': '61/MOD02QKM',
                   'dict_key': 'mod_02',
                'description': 'Terra MODIS Level 1b (250m) Calibrated Radiances Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD02QKM.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS 250m Calibrated Radiances Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MOD02QKM.061 (last access: %s), 2017.' % _date_today_,
                },

        'MYD02QKM': {
                'dataset_tag': '61/MYD02QKM',
                   'dict_key': 'myd_02',
                'description': 'Aqua MODIS Level 1b (250m) Calibrated Radiances Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD02QKM.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS 250m Calibrated Radiances Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MYD02QKM.061 (last access: %s), 2017.' % _date_today_,
                },

        'MOD02HKM': {
                'dataset_tag': '61/MOD02HKM',
                   'dict_key': 'mod_02',
                'description': 'Terra MODIS Level 1b (500m) Calibrated Radiances Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD02HKM.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS 500m Calibrated Radiances Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MOD02HKM.061 (last access: %s), 2017.' % _date_today_,
                },

        'MYD02HKM': {
                'dataset_tag': '61/MYD02HKM',
                   'dict_key': 'myd_02',
                'description': 'Aqua MODIS Level 1b (250m) Calibrated Radiances Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD02HKM.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS 500m Calibrated Radiances Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MYD02HKM.061 (last access: %s), 2017.' % _date_today_,
                },

        'MOD021KM': {
                'dataset_tag': '61/MOD021KM',
                   'dict_key': 'mod_02',
                'description': 'Terra MODIS Level 1b (1km) Calibrated Radiances Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD021KM.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS 1km Calibrated Radiances Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MOD021KM.061 (last access: %s), 2017.' % _date_today_,
                },

        'MYD021KM': {
                'dataset_tag': '61/MYD021KM',
                   'dict_key': 'myd_02',
                'description': 'Aqua MODIS Level 1b (1km) Calibrated Radiances Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD021KM.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'MODIS Characterization Support Team: MODIS 1km Calibrated Radiances Product, NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MYD021KM.061 (last access: %s), 2017.' % _date_today_,
                },

        'MOD06_L2': {
                'dataset_tag': '61/MOD06_L2',
                   'dict_key': 'mod_l2',
                'description': 'Terra MODIS Atmosphere Level 2 Cloud Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD06_L2.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'Platnick, S., Ackerman, S. A., King, M. D. , Meyer, K., Menzel, W. P. , Holz, R. E., Baum, B. A., and Yang, P., 2015: MODIS atmosphere L2 cloud product (06_L2), NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MOD06_L2.061 (last access: %s), 2015.' % _date_today_,
                },

        'MYD06_L2': {
                'dataset_tag': '61/MYD06_L2',
                   'dict_key': 'myd_l2',
                'description': 'Aqua MODIS Atmosphere Level 2 Cloud Product',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD06_L2.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Platnick, S., Ackerman, S. A., King, M. D. , Meyer, K., Menzel, W. P. , Holz, R. E., Baum, B. A., and Yang, P., 2015: MODIS atmosphere L2 cloud product (06_L2), NASA MODIS Adaptive Processing System [data set], Goddard Space Flight Center, USA, http://dx.doi.org/10.5067/MODIS/MOD06_L2.061 (last access: %s), 2015.' % _date_today_,
                },

        'MOD35_L2': {
                'dataset_tag': '61/MOD35_L2',
                   'dict_key': 'mod_l2',
                'description': 'Terra MODIS Atmosphere Level 2 Cloud Mask',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD35_L2.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'Ackerman, S., P. Menzel, R. Frey, B.Baum, 2017. MODIS Atmosphere L2 Cloud Mask Product. NASA MODIS Adaptive Processing System, Goddard Space Flight Center, [doi:10.5067/MODIS/MOD35_L2.061'
                },

        'MYD35_L2': {
                'dataset_tag': '61/MYD35_L2',
                   'dict_key': 'myd_l2',
                'description': 'Aqua MODIS Atmosphere Level 2 Cloud Mask',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD35_L2.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Ackerman, S., P. Menzel, R. Frey, B.Baum, 2017. MODIS Atmosphere L2 Cloud Mask Product. NASA MODIS Adaptive Processing System, Goddard Space Flight Center, [doi:10.5067/MODIS/MYD35_L2.061]'
                },

        # uncomment when product becomes available
        # 'MOD_CLDMSK_L2': {
        #         'dataset_tag': '5110/CLDMSK_L2_MODIS_Terra',
        #            'dict_key': 'mod_cldmsk_l2',
        #         'description': 'MODIS/Terra Cloud Mask 5-Min Swath 1 km',
        #                 'doi': '10.5067/MODIS/CLDMSK_L2_MODIS_Terra.001',
        #             'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_MODIS_Terra',
        #           'satellite': 'Terra',
        #          'instrument': 'MODIS',
        #           'reference': 'Ackerman, S., et al., 2019. MODIS/Terra Cloud Mask and Spectral Test Results 5-Min L2 Swath 1km, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/ CLDMSK_L2_MODIS_Terra.001',
        #         },

        # 'TERRA_CLDMSK_L2': {
        #         'dataset_tag': '5110/CLDMSK_L2_MODIS_Terra',
        #            'dict_key': 'mod_cldmsk_l2',
        #         'description': 'MODIS/Terra Cloud Mask 5-Min Swath 1 km',
        #                 'doi': '10.5067/MODIS/CLDMSK_L2_MODIS_Terra.001',
        #             'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_MODIS_Terra',
        #           'satellite': 'Terra',
        #          'instrument': 'MODIS',
        #           'reference': 'Ackerman, S., et al., 2019. MODIS/Terra Cloud Mask and Spectral Test Results 5-Min L2 Swath 1km, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/ CLDMSK_L2_MODIS_Terra.001',
        #         },

        'MYD_CLDMSK_L2': {
                'dataset_tag': '5110/CLDMSK_L2_MODIS_Aqua',
                   'dict_key': 'myd_cldmsk_l2',
                'description': 'Aqua MODIS Continuity Cloud Mask (MVCM) 5-Min Swath 1 km',
                        'doi': '10.5067/MODIS/CLDMSK_L2_MODIS_Aqua.001',
                    'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_MODIS_Aqua',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Ackerman, S., et al., 2019. MODIS/Aqua Cloud Mask and Spectral Test Results 5-Min L2 Swath 1km, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/CLDMSK_L2_MODIS_Aqua.001',
                },

        'AQUA_CLDMSK_L2': {
                'dataset_tag': '5110/CLDMSK_L2_MODIS_Aqua',
                   'dict_key': 'myd_cldmsk_l2',
                'description': 'Aqua MODIS Continuity Cloud Mask (MVCM) 5-Min Swath 1 km',
                        'doi': '10.5067/MODIS/CLDMSK_L2_MODIS_Aqua.001',
                    'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_MODIS_Aqua',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Ackerman, S., et al., 2019. MODIS/Aqua Cloud Mask and Spectral Test Results 5-Min L2 Swath 1km, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/CLDMSK_L2_MODIS_Aqua.001',
                },

        'MOD09': {
                'dataset_tag': '61/MOD09',
                   'dict_key': 'mod_09',
                'description': 'Terra MODIS Atmosphere Level 2 Atmospherically Corrected Surface Reflectance',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MOD09.061',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'Eric Vermote - NASA GSFC and MODAPS SIPS - NASA. (2015). MOD09 MODIS/Terra L2 Surface Reflectance, 5-Min Swath 250m, 500m, and 1km. NASA LP DAAC. http://doi.org/10.5067/MODIS/MOD09.061'
                },

        'MYD09': {
                'dataset_tag': '61/MYD09',
                   'dict_key': 'myd_09',
                'description': 'Aqua MODIS Atmosphere Level 2 Atmospherically Corrected Surface Reflectance',
                    'website': 'http://dx.doi.org/10.5067/MODIS/MYD09.061',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Eric Vermote - NASA GSFC and MODAPS SIPS - NASA. (2015). MYD09 MODIS/Aqua L2 Surface Reflectance, 5-Min Swath 250m, 500m, and 1km. NASA LP DAAC. http://doi.org/10.5067/MODIS/MYD09.061'
                },

        'MCD43A3': {
                'dataset_tag': '61/MCD43A3',
                   'dict_key': 'mod_43',
                'description': 'MODIS BRDF/Albedo Level 3 Surface Product',
                    'website': 'https://doi.org/10.5067/MODIS/MCD43A3.061',
                  'satellite': 'Terra & Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Schaaf, C., and Wang, Z.: MODIS/Terra+Aqua BRDF/Albedo Daily L3 Global - 500m V061, NASA EOSDIS Land Processes DAAC [data set], https://doi.org/10.5067/MODIS/MCD43A3.061 (last access: %s), 2021.' % _date_today_,
                },

        'VNP02IMG': {
                'dataset_tag': '5200/VNP02IMG',
                   'dict_key': 'vnp_02',
                'description': 'Suomi-NPP VIIRS Level 1b (375m) Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VNP02IMG.002',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2016-09-01. VIIRS/NPP Imagery Resolution 6-Min L1B Swath 375m. Version 1. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VNP02IMG.001',
                },

        'VJ102IMG': {
                'dataset_tag': '5201/VJ102IMG',
                   'dict_key': 'vj1_02',
                'description': 'JPSS1 (NOAA-20) VIIRS Level 1b (375m) Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ102IMG.021',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2019-08-01. VIIRS/JPSS1 Imagery Resolution 6-Min L1B Swath 375m. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ102IMG.002',
                },

         'VJ202IMG': {
                'dataset_tag': '5200/VJ202IMG',
                   'dict_key': 'vj2_02',
                'description': 'JPSS2 (NOAA-21) VIIRS Level 1b (375m) Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ202IMG.021',
                  'satellite': 'NOAA21',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2019-08-01. VIIRS/JPSS2 Imagery Resolution 6-Min L1B Swath 375m. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ202IMG.002',
                },

        'VNP02MOD': {
                'dataset_tag': '5200/VNP02MOD',
                   'dict_key': 'vnp_02',
                'description': 'Suomi-NPP VIIRS Level 1b (750m) Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VNP02MOD.002',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2021-07-12. VIIRS/NPP Moderate Resolution Terrain Corrected Geolocation 6-Min L1 Swath 750 m . Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VNP02MOD.002.',
                },

        'VJ102MOD': {
                'dataset_tag': '5201/VJ102MOD',
                   'dict_key': 'vj1_02',
                'description': 'JPSS1 (NOAA-20) VIIRS Level 1b (750m) Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ102MOD.021',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2019-08-01. VIIRS/JPSS1 Moderate Resolution 6-Min L1B Swath 750m. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ102MOD.002',
                },

        'VJ202MOD': {
                'dataset_tag': '5200/VJ202MOD',
                   'dict_key': 'vj2_02',
                'description': 'JPSS2 (NOAA-21) VIIRS Level 1b (750m) Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ202MOD.021',
                  'satellite': 'NOAA21',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2019-08-01. VIIRS/JPSS2 Moderate Resolution 6-Min L1B Swath 750m. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ202MOD.002',
                },

        'VNP03IMG': {
                'dataset_tag': '5200/VNP03IMG',
                   'dict_key': 'vnp_03',
                'description': 'Suomi-NPP VIIRS (375m) Geolocation Fields Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VNP03IMG.002',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. VIIRS/NPP Imagery Resolution Terrain-Corrected Geolocation 6-Min L1 Swath 375m Light. Version 1. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VNP03IMGLL.001',
                },

        'VJ103IMG': {
                'dataset_tag': '5201/VJ103IMG',
                   'dict_key': 'vj1_03',
                'description': 'JPSS1 (NOAA-20) VIIRS (375m) Geolocation Fields Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ103IMG.021',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. VIIRS/JPSS1 Imagery Resolution Terrain-Corrected Geolocation 6-Min L1 Swath 375m Light. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ103IMG.021',
                },

        'VJ203IMG': {
                'dataset_tag': '5200/VJ203IMG',
                   'dict_key': 'vj2_03',
                'description': 'JPSS2 (NOAA-21) VIIRS (375m) Geolocation Fields Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ203IMG.021',
                  'satellite': 'NOAA21',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. VIIRS/JPSS2 Imagery Resolution Terrain-Corrected Geolocation 6-Min L1 Swath 375m Light. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ203IMG.021',
                },

        'VNP03MOD': {
                'dataset_tag': '5200/VNP03MOD',
                   'dict_key': 'vnp_03',
                'description': 'Suomi-NPP VIIRS (750m) Geolocation Fields Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VNP03MOD.002',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2021-07-12. VIIRS/NPP Moderate Resolution Terrain-Corrected Geolocation L1 6-Min Swath 750 m. Version 2. LAADS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VNP03MOD.002',
                },

        'VJ103MOD': {
                'dataset_tag': '5201/VJ103MOD',
                   'dict_key': 'vj1_03',
                'description': 'JPSS1 (NOAA-20) VIIRS (750m) Geolocation Fields Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ103MOD.021',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2019-08-01. VIIRS/JPSS1 Moderate Resolution Terrain Corrected Geolocation 6-Min L1 Swath 750m. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ103MOD.002',
                },

         'VJ203MOD': {
                'dataset_tag': '5200/VJ203MOD',
                   'dict_key': 'vj2_03',
                'description': 'JPSS2 (NOAA-21) VIIRS (750m) Geolocation Fields Product',
                    'website': 'https://doi.org/10.5067/VIIRS/VJ203MOD.021',
                  'satellite': 'NOAA21',
                 'instrument': 'VIIRS',
                  'reference': 'VCST Team. 2019-08-01. VIIRS/JPSS1 Moderate Resolution Terrain Corrected Geolocation 6-Min L1 Swath 750m. Version 2. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS). https://doi.org/10.5067/VIIRS/VJ203MOD.002',
                },

        'VNPRGB': {
                'dataset_tag': '5200/VNPRGB',
                   'dict_key': 'vnp_rgb',
                'description': 'Suomi-NPP VIIRS True Color (RGB) Imagery',
                    'website': 'https://worldview.earthdata.nasa.gov',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'We acknowledge the use of imagery from the NASA Worldview application (https://worldview.earthdata.nasa.gov/), part of the NASA Earth Observing System Data and Information System (EOSDIS).',
                },

        'VJ1RGB': {
                'dataset_tag': '5201/VJ1RGB',
                   'dict_key': 'vj1_rgb',
                'description': 'JPSS1 (NOAA-20) VIIRS True Color (RGB) Imagery',
                    'website': 'https://worldview.earthdata.nasa.gov',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'We acknowledge the use of imagery from the NASA Worldview application (https://worldview.earthdata.nasa.gov/), part of the NASA Earth Observing System Data and Information System (EOSDIS).',
                },

        'VJ2RGB': {
                'dataset_tag': '5200/VJ2RGB',
                   'dict_key': 'vj2_rgb',
                'description': 'JPSS2 (NOAA-21) VIIRS True Color (RGB) Imagery',
                    'website': 'https://worldview.earthdata.nasa.gov',
                  'satellite': 'NOAA21',
                 'instrument': 'VIIRS',
                  'reference': 'We acknowledge the use of imagery from the NASA Worldview application (https://worldview.earthdata.nasa.gov/), part of the NASA Earth Observing System Data and Information System (EOSDIS).',
                },

        'VNP_CLDPROP_L2': {
                'dataset_tag': '5111/CLDPROP_L2_VIIRS_SNPP',
                   'dict_key': 'vnp_l2',
                'description': 'Suomi-NPP VIIRS Cloud Properties Product',
                    'website': 'https://doi.org/10.5067/VIIRS/CLDPROP_L2_VIIRS_SNPP.011',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'NASA VIIRS Atmosphere SIPS. 2019-11-15. VIIRS/SNPP Cloud Properties 6-min L2 Swath 750m. Version 1.1. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS).',
                },

        'VJ1_CLDPROP_L2': {
                'dataset_tag': '5111/CLDPROP_L2_VIIRS_NOAA20',
                   'dict_key': 'vj1_l2',
                'description': 'JPSS1 (NOAA-20) VIIRS Cloud Properties Product',
                    'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDPROP_L2_VIIRS_NOAA20',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'NASA VIIRS Atmosphere SIPS. 2019-11-15. VIIRS/JPSS1 Cloud Properties 6-min L2 Swath 750m. Version 1.1. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS).',
                },

        ## commenting this out as cloud products for NOAA-21 are not available yet
        # 'VJ2_CLDPROP_L2': {
        #         'dataset_tag': '5111/CLDPROP_L2_VIIRS_NOAA21',
        #            'dict_key': 'vj2_l2',
        #         'description': 'JPSS2 (NOAA-21) VIIRS Cloud Properties Product',
        #             'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDPROP_L2_VIIRS_NOAA21',
        #           'satellite': 'NOAA21',
        #          'instrument': 'VIIRS',
        #           'reference': 'NASA VIIRS Atmosphere SIPS. 2019-11-15. VIIRS/JPSS2 Cloud Properties 6-min L2 Swath 750m. Version 1.1. MODAPS at NASA/GSFC. Archived by National Aeronautics and Space Administration, U.S. Government, L1 and Atmosphere Archive and Distribution System (LAADS).',
        #         },

        'VNP_CLDMSK_L2': {
                'dataset_tag': '5110/CLDMSK_L2_VIIRS_SNPP',
                   'dict_key': 'vnp_cldmsk_l2',
                'description': 'SNPP VIIRS Continuity Cloud Mask (MVCM) 6-Min Swath 750 m',
                        'doi': '10.5067/VIIRS/CLDMSK_L2_VIIRS_SNPP.001',
                    'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_VIIRS_SNPP',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'Ackerman, S., et al., 2019. VIIRS/SNPP Cloud Mask and Spectral Test Results 6-Min L2 Swath 750m, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/CLDMSK_L2_VIIRS_SNPP.001',
                },

         'VJ1_CLDMSK_L2': {
                'dataset_tag': '5110/CLDMSK_L2_VIIRS_NOAA20',
                   'dict_key': 'vj1_cldmsk_l2',
                'description': 'NOAA20 (JPSS1) VIIRS Continuity Cloud Mask (MVCM) 6-Min Swath 750 m',
                        'doi': '10.5067/VIIRS/CLDMSK_L2_VIIRS_NOAA20.001',
                    'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_VIIRS_NOAA20',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'Ackerman, S., et al., 2019. VIIRS/NOAA20 Cloud Mask and Spectral Test Results 6-Min L2 Swath 750m, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/CLDMSK_L2_VIIRS_NOAA20.001',
                },

          'VJ2_CLDMSK_L2': {
                'dataset_tag': '5110/CLDMSK_L2_VIIRS_NOAA21',
                   'dict_key': 'vj2_cldmsk_l2',
                'description': 'NOAA21 (JPSS2) VIIRS Continuity Cloud Mask (MVCM) 6-Min Swath 750 m',
                        'doi': '10.5067/VIIRS/CLDMSK_L2_VIIRS_NOAA21.001',
                    'website': 'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/CLDMSK_L2_VIIRS_NOAA21',
                  'satellite': 'NOAA21',
                 'instrument': 'VIIRS',
                  'reference': 'Ackerman, S., et al., 2019. VIIRS/NOAA21 Cloud Mask and Spectral Test Results 6-Min L2 Swath 750m, Version-1. NASA Level-1 and Atmosphere Archive & Distribution System (LAADS) Distributed Active Archive Center (DAAC), Goddard Space Flight Center, USA: https://dx.doi.org/10.5067/VIIRS/CLDMSK_L2_VIIRS_NOAA21.001',
                },

        'VNP09': {
                'dataset_tag': '5200/VNP09',
                   'dict_key': 'vnp_09',
                'description': 'Suomi-NPP VIIRS Atmospherically Corrected Surface Reflectance Product',
                    'website': 'n/a',
                  'satellite': 'SNPP',
                 'instrument': 'VIIRS',
                  'reference': 'Roger, J. C., Vermote, E. F., Devadiga, S., & Ray, J. P. (2016). Suomi-NPP VIIRS Surface Reflectance User’s Guide. V1 Re-processing (NASA Land SIPS).'
                },

        'VJ109': {
                'dataset_tag': '5200/VJ109',
                   'dict_key': 'vj1_09',
                'description': 'JPSS1 (NOAA-20) Atmospherically Corrected Surface Reflectance Product',
                    'website': 'n/a',
                  'satellite': 'NOAA20',
                 'instrument': 'VIIRS',
                  'reference': 'Roger, J. C., Vermote, E. F., Devadiga, S., & Ray, J. P. (2016). Suomi-NPP VIIRS Surface Reflectance User’s Guide. V1 Re-processing (NASA Land SIPS).'
                },

        'MOD29': {
                  'dataset_tag': 'MOST/MOD29.061',
                   'dict_key': 'mod_29',
                'description': 'MODIS/Terra Sea Ice Extent 5-Min L2 Swath 1km',
                    'website': 'https://nsidc.org/data/mod29/versions/61',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'Hall, D. K. & Riggs, G. (2021). MODIS/Terra Sea Ice Extent 5-Min L2 Swath 1km. (MOD29, Version 61). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD29.061. [describe subset used if applicable]. Date Accessed 03-01-2025.'
                },

        'MYD29': {
                  'dataset_tag': 'MOST/MYD29.061',
                   'dict_key': 'myd_29',
                'description': 'MODIS/Aqua Sea Ice Extent 5-Min L2 Swath 1km',
                    'website': 'https://nsidc.org/data/myd29/versions/61',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Hall, D. K. & Riggs, G. A. (2021). MODIS/Aqua Sea Ice Extent 5-Min L2 Swath 1km. (MYD29, Version 61). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MYD29.061. [describe subset used if applicable]. Date Accessed 03-01-2025.'
                },

        'MOD29_NRT': {
                  'dataset_tag': '61/MOD29',
                   'dict_key': 'mod_29',
                'description': 'MODIS/Terra Sea Ice Extent 5-Min L2 Swath 1km',
                    'website': 'https://nsidc.org/data/mod29/versions/61',
                  'satellite': 'Terra',
                 'instrument': 'MODIS',
                  'reference': 'Hall, D. K. & Riggs, G. (2021). MODIS/Terra Sea Ice Extent 5-Min L2 Swath 1km. (MOD29, Version 61). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD29.061. [describe subset used if applicable]. Date Accessed 03-01-2025.'
                },

        'MYD29_NRT': {
                  'dataset_tag': '61/MYD29',
                   'dict_key': 'myd_29',
                'description': 'MODIS/Aqua Sea Ice Extent 5-Min L2 Swath 1km',
                    'website': 'https://nsidc.org/data/myd29/versions/61',
                  'satellite': 'Aqua',
                 'instrument': 'MODIS',
                  'reference': 'Hall, D. K. & Riggs, G. A. (2021). MODIS/Aqua Sea Ice Extent 5-Min L2 Swath 1km. (MYD29, Version 61). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MYD29.061. [describe subset used if applicable]. Date Accessed 03-01-2025.'
                },

        'oco2_L1bScND': {
                'dataset_tag': 'oco2_L1bScND',
                   'dict_key': 'oco_l1b',
                'description': 'OCO-2 L1B Calibrated Radiances Product',
                    'website': 'https://doi.org/10.5067/6O3GEUK7U2JG',
                  'satellite': 'OCO-2',
                 'instrument': 'OCO-2',
                  'reference': 'OCO-2 Science Team/Gunson, M., and Eldering, A.: OCO-2 Level 1B calibrated, geolocated science spectra, Retrospective Processing V10r, Goddard Earth Sciences Data and Information Services Center (GES DISC) [data set], Greenbelt, MD, USA, https://doi.org/10.5067/6O3GEUK7U2JG (last access: %s), 2019.' % _date_today_,
                },

        'oco2_L2MetND': {
                'dataset_tag': 'oco2_L2MetND',
                   'dict_key': 'oco_met_l2',
                'description': 'OCO-2 L2 Meteorological Parameters Product',
                    'website': 'https://doi.org/10.5067/OJZZW0LIGSDH',
                  'satellite': 'OCO-2',
                 'instrument': 'OCO-2',
                  'reference': 'OCO-2 Science Team/Gunson, M., and Eldering, A.: OCO-2 Level 2 meteorological parameters interpolated from global assimilation model for each sounding, Retrospective Processing V10r, Goddard Earth Sciences Data and Information Services Center (GES DISC) [data set], Greenbelt, MD, USA, https://doi.org/10.5067/OJZZW0LIGSDH (last access: %s), 2019.' % _date_today_,
                },

        'oco2_L2StdND': {
                'dataset_tag': 'oco2_L2StdND',
                   'dict_key': 'oco_ret_l2',
                'description': 'OCO-2 L2 XCO2 Retrieval Product',
                    'website': 'https://doi.org/10.5067/6SBROTA57TFH',
                  'satellite': 'OCO-2',
                 'instrument': 'OCO-2',
                  'reference': 'OCO-2 Science Team/Gunson, M., and Eldering, A.: OCO-2 Level 2 geolocated XCO2 retrievals results, physical model, Retrospective Processing V10r, Goddard Earth Sciences Data and Information Services Center (GES DISC) [data set], Greenbelt, MD, USA, https://doi.org/10.5067/6SBROTA57TFH (last access: %s), 2020.' % _date_today_,
                },

        }
#\----------------------------------------------------------------------------/#
