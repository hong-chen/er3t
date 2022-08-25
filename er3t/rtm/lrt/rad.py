import os
import sys
import glob
import datetime
import warnings
import math
import numpy as np
import multiprocessing as mp
from collections import OrderedDict as OD

from .util import *
from .cfg import *



__all__ = ['lrt_init_mono_rad', 'lrt_init_spec_rad', 'lrt_read_uvspec_rad']


def convert_azimuth_angle(azimuth_angle):

    return (azimuth_angle+180.0) % 360.0



class lrt_init_mono_rad:

    """
    Purpose: use libRadtran to calculation monochromatic irradiance for SSFR

    Features: 'lrt_cfg' for general libRadtran parameter specification
              'cld_cfg' for 1D cloud layer specification
              'aer_cfg' for 1D aerosol layer specification

    See <examples/00_er3t_lrt.py> for detailed usage.
    """

    def __init__(self, \
            input_file          = None,
            output_file         = None,
            date                = None, # datetime.date object
            surface_albedo      = None, # unitless: 0 ~ 1
            solar_zenith_angle  = None, # units: degree
            solar_azimuth_angle = None, # units: degree
            sensor_zenith_angle = None, # units: degree
            sensor_azimuth_angle= None, # units: degree
            wavelength          = None, # units: nm
            output_altitude     = None, # units: km
            output_format       = 'lambda uu',
            spectral_resolution = 0.1,
            mute_list           = [],
            input_dict_extra    = None,
            lrt_cfg = None,
            cld_cfg = None,
            aer_cfg = None,
            verbose = False
            ):

        if lrt_cfg is None:
            self.lrt_cfg         = get_lrt_cfg()
        else:
            self.lrt_cfg         = lrt_cfg

        self.mute_list       = mute_list

        # executable file
        self.executable_file = lrt_cfg['executable_file']

        # input file
        if input_file is None:
            dtime_tmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            input_file = 'lrt_input_%s.txt' % dtime_tmp
            if verbose:
                print('Message [lrt_init_mono]: <input_file> is missing, assigning input_file = %s.' % input_file)
        self.input_file = input_file

        # output file
        if output_file is None:
            output_file = 'lrt_output_%s.txt' % dtime_tmp
            if verbose:
                print('Message [lrt_init_mono]: <output_file> is missing, assigning output_file = %s.' % output_file)
        self.output_file = output_file

        # date
        if date is None:
            date = datetime.date.today()
            if verbose:
                print('Message [lrt_init_mono]: <date> is missing, assigning date = datetime.date.today().')

        # surface albedo
        if surface_albedo is None:
            surface_albedo = 0.03
            if verbose:
                print('Message [lrt_init_mono]: <surface_albedo> is missing, assigning surface_albedo = 0.03.')

        # solar zenith angle
        if solar_zenith_angle is None:
            solar_zenith_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <solar_zenith_angle> is missing, assigning solar_zenith_angle = 0.0.')

        # solar azimuth angle
        if solar_azimuth_angle is None:
            solar_azimuth_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <solar_azimuth_angle> is missing, assigning solar_azimuth_angle = 0.0.')
        solar_azimuth_angle = convert_azimuth_angle(solar_azimuth_angle)

        # sensor zenith angle
        if sensor_zenith_angle is None:
            sensor_zenith_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <sensor_zenith_angle> is missing, assigning sensor_zenith_angle = 0.0.')

        # sensor azimuth angle
        if sensor_azimuth_angle is None:
            sensor_azimuth_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <sensor_azimuth_angle> is missing, assigning sensor_azimuth_angle = 0.0.')

        if not isinstance(sensor_azimuth_angle, str):
            if isinstance(sensor_azimuth_angle, (list, np.ndarray)):
                self.Nvar = len(sensor_azimuth_angle) + 1
                sensor_azimuth_angle = ' '.join(str(vaa0) for vaa0 in sensor_azimuth_angle)
            else:
                sensor_azimuth_angle = str(sensor_azimuth_angle)
                self.Nvar = 2
        else:
            self.Nvar = 2

        # wavelength
        if wavelength is None:
            wavelength = 500.0
            if verbose:
                print('Message [lrt_init_mono]: <wavelength> is missing, assigning wavelength = 500.0.')
        self.Nx = 1

        # slit function
        if wavelength < 950.0:
            slit_function_file    = lrt_cfg['slit_function_file_vis']
            wavelength_half_width = 8.0
        else:
            slit_function_file    = lrt_cfg['slit_function_file_nir']
            wavelength_half_width = 16.0
        if verbose:
            print('Message [lrt_init_mono]: slit_function_file = \'%s\'.' % slit_function_file)

        # output altitude
        if output_altitude is None:
            output_altitude = 'TOA'
            if verbose:
                print('Message [lrt_init_mono]: <output_altitude> is missing, assigning output_altitude = \'TOA\'.')


        if not isinstance(output_altitude, str):
            if isinstance(output_altitude, (list, np.ndarray)):
                output_altitude = ' '.join(str(zout) for zout in output_altitude)
            else:
                output_altitude = str(output_altitude)

        self.Ny = len(output_altitude.split())

        day_of_year = date.timetuple().tm_yday
        wavelength_s  = np.round(wavelength-wavelength_half_width-spectral_resolution, decimals=int(-math.log10(spectral_resolution)))
        wavelength_e  = np.round(wavelength+wavelength_half_width+spectral_resolution, decimals=int(-math.log10(spectral_resolution)))

        self.input_dict = OD([
                            ('atmosphere_file'   , lrt_cfg['atmosphere_file']),
                            ('source solar'      , lrt_cfg['solar_file']),
                            ('day_of_year'       , str(day_of_year)),
                            ('albedo'            , '%.6f' % surface_albedo),
                            ('sza'               , '%.4f' % solar_zenith_angle),
                            ('phi0'              , '%.4f' % solar_azimuth_angle),
                            ('umu'               , '%.8f' % np.cos(np.deg2rad(sensor_zenith_angle))),
                            ('phi'               , sensor_azimuth_angle),
                            ('rte_solver'        , lrt_cfg['rte_solver']),
                            ('number_of_streams' , str(lrt_cfg['number_of_streams'])),
                            ('wavelength'        , '%.1f %.1f' % (wavelength_s, wavelength_e)),
                            ('spline'            , '%.3f %.3f %.3f' % (wavelength, wavelength, spectral_resolution)),
                            ('slit_function_file', slit_function_file),
                            ('mol_abs_param'     , lrt_cfg['mol_abs_param']),
                            ('output_user'       , output_format),
                            ('zout'              , output_altitude)
                            ])


        self.input_dict_extra = input_dict_extra

        if cld_cfg is not None:

            if all(cld_cfg[key] is not None for key in cld_cfg.keys()):
                self.cld_cfg = cld_cfg

                if not os.path.exists(cld_cfg['cloud_file']):
                    gen_cloud_1d(cld_cfg)

                if cld_cfg['cloud_type'] == 'water':
                    prefix = 'wc'
                elif cld_cfg['cloud_type'] == 'ice':
                    prefix = 'ic'

                if self.input_dict_extra is not None:

                    self.input_dict_extra['%s_file 1D '       % prefix] = cld_cfg['cloud_file']
                    self.input_dict_extra['%s_properties %s'  % (prefix, cld_cfg['%s_properties' % prefix])] = 'interpolate'
                    self.input_dict_extra['%s_modify tau set' % prefix] = str(cld_cfg['cloud_optical_thickness'])

                else:

                    self.input_dict_extra = OD([
                        ('%s_file 1D '       % prefix                                     , cld_cfg['cloud_file']),
                        ('%s_properties %s'  % (prefix, cld_cfg['%s_properties' % prefix]), 'interpolate'),
                        ('%s_modify tau set' % prefix                                     , str(cld_cfg['cloud_optical_thickness']))
                        ])

            else:
                msg = 'Error [lrt_init_mono]: <cld_cfg> is not complete.'
                raise OSError(msg)



        if aer_cfg is not None:

            if all(aer_cfg[key] is not None for key in aer_cfg.keys()):
                self.aer_cfg = aer_cfg

                if not os.path.exists(aer_cfg['aerosol_file']):
                    aer_cfg = gen_aerosol_1d(aer_cfg)

                if self.input_dict_extra is not None:

                    self.input_dict_extra['aerosol_default']  = ''
                    self.input_dict_extra['aerosol_file tau'] = aer_cfg['aerosol_file_aod']
                    self.input_dict_extra['aerosol_file ssa'] = aer_cfg['aerosol_file_ssa']
                    self.input_dict_extra['aerosol_file gg']  = aer_cfg['aerosol_file_asy']

                else:
                    self.input_dict_extra = OD([
                        ('aerosol_default' , ''),
                        ('aerosol_file tau', aer_cfg['aerosol_file_aod']),
                        ('aerosol_file ssa', aer_cfg['aerosol_file_ssa']),
                        ('aerosol_file gg' , aer_cfg['aerosol_file_asy'])
                        ])

            else:
                msg = 'Error [lrt_init_mono]: <aer_cfg> is not complete.'
                raise OSError(msg)



class lrt_init_spec_rad:

    """
    Purpose: use libRadtran to calculation multichromatic irradiance for SSFR

    Features: can take in multiple wavelengths, e.g., [350.0, 355.0, ..., 750.0]
              cld_cfg for 1D cloud layer specification
              aer_cfg for 1D aerosol layer specification

    Limitations: so far, can only deal with wavelengths all smaller than 950nm or wavelengths all greater than 950nm
                 since different wavelength ranges use different slit functions

    See <examples/00_er3t_lrt.py> for detailed usage.
    """

    def __init__(self, \
            input_file          = None, # string; ascii file that contains input parameters for libRadtran
            output_file         = None, # string; ascii file created by libRadtran that contains results
            date                = None, # datetime.date object; e.g., datetime.datetime(2014, 9, 13)
            surface_albedo      = None, # float number; unitless: 0 ~ 1
            surface_albedo_file = None, # string; ascii file that contains two columns - 1: wavelength 2: albedo
            solar_zenith_angle  = None, # float number; units: degree
            solar_azimuth_angle = None, # units: degree
            sensor_zenith_angle = None, # units: degree
            sensor_azimuth_angle= None, # units: degree
            wavelength_file     = None, # string; ascii file that contains one column - 1: wavelength
            output_altitude     = None, # string, list, or numpy.array; units: km
            output_format       = 'lambda uu',
            spectral_resolution = 0.1,
            lrt_cfg = None,
            cld_cfg = None,
            aer_cfg = None,
            verbose = False
            ):

        if lrt_cfg is None:
            self.lrt_cfg         = get_lrt_cfg()
        else:
            self.lrt_cfg         = lrt_cfg

        # executable file
        self.executable_file = lrt_cfg['executable_file']

        # input file
        if input_file is None:
            dtime_tmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            input_file = 'lrt_input_%s.txt' % dtime_tmp
            if verbose:
                print('Message [lrt_init_spec]: <input_file> is missing, assigning input_file = %s.' % input_file)
        self.input_file = input_file

        # output file
        if output_file is None:
            output_file = 'lrt_output_%s.txt' % dtime_tmp
            if verbose:
                print('Message [lrt_init_spec]: <output_file> is missing, assigning output_file = %s.' % output_file)
        self.output_file = output_file

        # date
        if date is None:
            date = datetime.date.today()
            if verbose:
                print('Message [lrt_init_spec]: <date> is missing, assigning date = datetime.date.today().')
        day_of_year = date.timetuple().tm_yday

        # wavelength
        if wavelength_file is None:
            wavelength = np.arange(350.0, 951.0, 5.0)
            if verbose:
                print('Message [lrt_init_spec]: <wavelength_file> is missing, assigning wavelength = [350.0, 355.0, ..., 950.0].')
            wavelength_file = 'lrt_wvl_%s.txt' % dtime_tmp
            gen_wavelength_file(wavelength_file, wavelength)
        else:
            wavelength = np.loadtxt(wavelength_file)
        self.Nx = wavelength.size

        # surface albedo
        if surface_albedo_file is None and surface_albedo is None:
            surface_albedo = 0.03
            if verbose:
                print('Message [lrt_init_spec]: <surface_albedo_file> is missing, assigning surface_albedo = [0.03, 0.03, ..., 0.03].')

        # solar zenith angle
        if solar_zenith_angle is None:
            solar_zenith_angle = 0.0
            if verbose:
                print('Message [lrt_init_spec]: <solar_zenith_angle> is missing, assigning solar_zenith_angle = 0.0.')

        # solar azimuth angle
        if solar_azimuth_angle is None:
            solar_azimuth_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <solar_azimuth_angle> is missing, assigning solar_azimuth_angle = 0.0.')
        solar_azimuth_angle = convert_azimuth_angle(solar_azimuth_angle)

        # sensor zenith angle
        if sensor_zenith_angle is None:
            sensor_zenith_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <sensor_zenith_angle> is missing, assigning sensor_zenith_angle = 0.0.')

        # sensor azimuth angle
        if sensor_azimuth_angle is None:
            sensor_azimuth_angle = 0.0
            if verbose:
                print('Message [lrt_init_mono]: <sensor_azimuth_angle> is missing, assigning sensor_azimuth_angle = 0.0.')
        sensor_azimuth_angle = convert_azimuth_angle(sensor_azimuth_angle)

        # slit function
        wvl_min = wavelength.min()
        wvl_max = wavelength.max()

        if wvl_max <= 950.0:
            slit_function_file    = lrt_cfg['slit_function_file_vis']
            wavelength_half_width = 8.0
        elif wvl_min >= 950.0:
            slit_function_file    = lrt_cfg['slit_function_file_nir']
            wavelength_half_width = 16.0
        else:
            msg = 'Error [lrt_init_spec]: Invalid wavelength detected, can only deal with wavelengths within the range of <=950nm or >=950nm.'
            raise ValueError(msg)

        if verbose:
            print('Message [lrt_init_spec]: slit_function_file = \'%s\'.' % slit_function_file)


        # output altitude
        if output_altitude is None:
            output_altitude = 'TOA'
            if verbose:
                print('Message [lrt_init_spec]: <output_altitude> is missing, assigning output_altitude = \'TOA\'.')

        if not isinstance(output_altitude, str):
            if isinstance(output_altitude, (list, np.ndarray)):
                output_altitude = ' '.join(str(zout) for zout in output_altitude)
            else:
                output_altitude = str(output_altitude)
        self.Ny = len(output_altitude.split())

        if surface_albedo_file is None:
            self.input_dict = OD([
                                ('atmosphere_file'   , lrt_cfg['atmosphere_file']),
                                ('source solar'      , lrt_cfg['solar_file']),
                                ('day_of_year'       , str(day_of_year)),
                                ('albedo'            , str(np.round(surface_albedo, decimals=10))),
                                ('sza'               , str(np.round(solar_zenith_angle, decimals=10))),
                                ('phi0'              , '%.4f' % solar_azimuth_angle),
                                ('umu'               , '%.8f' % np.cos(np.deg2rad(sensor_zenith_angle))),
                                ('phi'               , '%.4f' % sensor_azimuth_angle),
                                ('rte_solver'        , lrt_cfg['rte_solver']),
                                ('number_of_streams' , str(lrt_cfg['number_of_streams'])),
                                ('wavelength'        , '%.1f %.1f' % (wavelength.min()-wavelength_half_width, wavelength.max()+wavelength_half_width)),
                                ('spline_file'       , wavelength_file),
                                ('slit_function_file', slit_function_file),
                                ('mol_abs_param'     , lrt_cfg['mol_abs_param']),
                                ('output_user'       , output_format),
                                ('zout'              , output_altitude)
                                ])
        else:
            self.input_dict = OD([
                                ('atmosphere_file'   , lrt_cfg['atmosphere_file']),
                                ('source solar'      , lrt_cfg['solar_file']),
                                ('day_of_year'       , str(day_of_year)),
                                ('albedo_file'       , surface_albedo_file),
                                ('sza'               , str(solar_zenith_angle)),
                                ('phi0'              , '%.4f' % solar_azimuth_angle),
                                ('umu'               , '%.4f' % sensor_zenith_angle),
                                ('phi'               , '%.4f' % sensor_azimuth_angle),
                                ('rte_solver'        , lrt_cfg['rte_solver']),
                                ('number_of_streams' , str(lrt_cfg['number_of_streams'])),
                                ('wavelength'        , '%.1f %.1f' % (wavelength.min()-wavelength_half_width, wavelength.max()+wavelength_half_width)),
                                ('spline_file'       , wavelength_file),
                                ('slit_function_file', slit_function_file),
                                ('mol_abs_param'     , lrt_cfg['mol_abs_param']),
                                ('output_user'       , output_format),
                                ('zout'              , output_altitude)
                                ])



        self.input_dict_extra = None

        if cld_cfg is not None:

            if all(cld_cfg[key] is not None for key in cld_cfg.keys()):
                self.cld_cfg = cld_cfg

                if not os.path.exists(cld_cfg['cloud_file']):
                    gen_cloud_1d(cld_cfg)

                if cld_cfg['cloud_type'] == 'water':
                    prefix = 'wc'
                elif cld_cfg['cloud_type'] == 'ice':
                    prefix = 'ic'

                self.input_dict_extra = OD([
                    ('%s_file 1D '       % prefix                                     , cld_cfg['cloud_file']),
                    ('%s_properties %s'  % (prefix, cld_cfg['%s_properties' % prefix]), 'interpolate'),
                    ('%s_modify tau set' % prefix                                     , str(cld_cfg['cloud_optical_thickness']))
                    ])

            else:
                msg = 'Error [lrt_init_spec]: <cld_cfg> is not complete.'
                raise OSError(msg)


        if aer_cfg is not None:

            if all(aer_cfg[key] is not None for key in aer_cfg.keys()):
                self.aer_cfg = aer_cfg

                if not os.path.exists(aer_cfg['aerosol_file']):
                    gen_aerosol_1d(aer_cfg)

                if self.input_dict_extra:

                    self.input_dict_extra['aerosol_default']  = ''
                    self.input_dict_extra['aerosol_file tau'] = aer_cfg['aerosol_file']
                    self.input_dict_extra['aerosol_modify tau set'] = str(aer_cfg['aerosol_optical_depth'])
                    self.input_dict_extra['aerosol_modify gg set']  = str(aer_cfg['asymmetry_parameter'])
                    self.input_dict_extra['aerosol_modify ssa set'] = str(aer_cfg['single_scattering_albedo'])

                else:
                    self.input_dict_extra = OD([
                        ('aerosol_default' , ''),
                        ('aerosol_file tau', aer_cfg['aerosol_file']),
                        ('aerosol_modify tau set', str(aer_cfg['aerosol_optical_depth'])),
                        ('aerosol_modify gg set' , str(aer_cfg['asymmetry_parameter'])),
                        ('aerosol_modify ssa set', str(aer_cfg['single_scattering_albedo']))
                        ])

            else:
                msg = 'Error [lrt_init_spec]: <aer_cfg> is not complete.'
                raise OSError(msg)



class lrt_read_uvspec_rad:

    """
    Input:
        list of lrt_init objects

    return:
        self.rad      : radiance

        f_down, f_down_direct, f_down_diffuse, f_up have the dimension of
        (Nx, Ny, Nz), where
        Nx: number of wavelength, for monochromatic, Nx=1
        Ny: number of output altitude
        Nz: number of lrt_init objects

        one can use numpy.squeeze to remove the axis where the Nx/Ny/Nz = 1.
    """


    def __init__(self, inits):

        Nx = inits[0].Nx
        Ny = inits[0].Ny
        Nz = len(inits)
        Nvar = inits[0].Nvar

        self.dims  = {'X':'Wavelength', 'Y':'Altitude', 'Z':'Files'}
        rad  = np.zeros((Nx, Ny, Nz, Nvar-1), dtype=np.float64)

        for i, init in enumerate(inits):

            data = np.loadtxt(init.output_file).reshape((Nx, Ny, Nvar))
            rad[:, :, i, :]  = data[:, :, 1:]/1000.0

        self.rad = rad


    def __add__(self, data):

        self.rad = np.vstack((self.rad, data.rad))

        return self



if __name__ == '__main__':

    pass
