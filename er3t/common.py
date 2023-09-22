import os
import sys
import logging
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
'EaR³T (Chen et al., 2023):\n\
- Chen, H., Schmidt, K. S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold, G., Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) - Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals, Atmos. Meas. Tech., 16, 1971–2000, https://doi.org/10.5194/amt-16-1971-2023, 2023.\n'
        ]
