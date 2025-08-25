import os
import sys
import copy
import pickle
import pandas as pd
import yaml
import numpy as np


import er3t.common
import er3t.util
from er3t.pre.atm.util import interp_pres_from_alt_temp, interp_ch4
from er3t.util import constants


__all__ = ['atm_atmmod', 'ARCSIXAtmModel']



class atm_atmmod:
    """
    Atmospheric model class for creating 1D atmospheric profiles with gas concentrations.
    This class interpolates atmospheric data from a base atmosphere file (AFGL atmospheric profile)
    to user-specified altitude levels and calculates atmospheric properties including pressure,
    temperature, and gas concentrations for various atmospheric constituents.
    The class can operate in three modes:
    1. Create new atmospheric profile from levels and save to file
    2. Load existing atmospheric profile from file
    3. Create atmospheric profile without saving

    Input:

        levels=      : keyword argument, numpy array, height in km
        fname=       : keyword argument, string, default=None, the atmoshpere file user wants to name
        fname_atmmod=: keyword argument, string, defult='mca-data/atmmod/afglus.dat', the base atmosphere file to interpolate to levels and layers
        overwrite=   : keyword argument, boolen, default=False, whether or not user wants to overwrite the atmosphere file
        verbose=     : keyword argument, boolen, default=False, whether or not print detailed messages

    Note:
    If levels is provided but fname does not exisit:
        calculate atmospheric gases profile and save data into fname

    if levels is not provided but fname is provided (also exists):
        read out the data from fname

    if levels and fname are neither provided:
        exit with error message

    Output:
        self.lev['pressure']
        self.lev['temperature']
        self.lev['altitude']
        self.lev['h2o']
        self.lev['o2']
        self.lev['o3']
        self.lev['co2']
        self.lev['no2']
        self.lev['ch4']
        self.lev['factor']

        self.lay['pressure']
        self.lay['temperature']
        self.lay['altitude']
        self.lay['thickness']
        self.lay['h2o']
        self.lay['o2']
        self.lay['o3']
        self.lay['co2']
        self.lay['no2']
        self.lay['ch4']
        self.lay['factor']
    """


    ID     = 'Atmosphere 1D'

    gases  = ['o3', 'o2', 'h2o', 'co2', 'no2', 'ch4']

    reference = '\nAFGL Atmospheric Profile (Anderson et al., 1986):\n- Anderson, G. P., Clough, S. A., Kneizys, F. X., Chetwynd, J. H., and Shettle, E. P.: AFGL atmospheric constituent profiles (0-120 km), Tech. Rep. AFGL-TR-86-0110, Air Force Geophys. Lab., Hanscom Air Force Base, Bedford, Massachusetts, USA, 1986.'

    def __init__(self,                \
                 levels       = None, \
                 fname        = None, \
                 fname_atmmod = '%s/afglus.dat' % er3t.common.fdir_data_atmmod, \
                 overwrite    = False, \
                 verbose      = False):

        er3t.util.add_reference(self.reference)

        self.verbose      = verbose
        self.fname_atmmod = fname_atmmod

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((levels is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((levels is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(levels)
            self.dump(fname)

        elif ((levels is not None) and (fname is None)):

            self.run(levels)

        else:

            sys.exit('Error   [atm_atmmod]: Please check if \'%s\' exists or provide \'levels\' to proceed.' % fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [atm_atmmod]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.lev   = obj.lev
                self.lay   = obj.lay
            else:
                sys.exit('Error   [atm_atmmod]: File \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, levels):

        if levels.size > 1:
            self.levels = levels
            self.layers = 0.5 * (levels[1:]+levels[:-1])
        else:
            msg = '\nError [atm_atmmod]: Size of <levels> must be greater than 1.'
            raise ValueError(msg)

        # self.atm0: Python dictionary
        #   self.atm0['altitude']
        #   self.atm0['pressure']
        #   self.atm0['temperature']
        #   self.atm0['co2']
        #   self.atm0['no2']
        #   self.atm0['h2o']
        #   self.atm0['o3']
        #   self.atm0['o2']
        self.atmmod()

        # self.lev, self.lay: Python dictionary
        #   self.lev['altitude']    | self.lay['altitude']
        #   self.lev['pressure']    | self.lay['pressure']
        #   self.lev['temperature'] | self.lay['temperature']
        #   self.lev['co2']         | self.lay['co2']
        #   self.lev['no2']         | self.lay['no2']
        #   self.lev['h2o']         | self.lay['h2o']
        #   self.lev['o3']          | self.lay['o3']
        #   self.lev['o2']          | self.lay['o2']
        self.interp()

        # add self.lev['ch4'] and self.lay['ch4']
        self.add_ch4()

        # covert mixing ratio [unitless] to number density [cm-3]
        self.cal_num_den()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [atm_atmmod]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def atmmod(self):

        vnames = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
        units  = ['km', 'mb', 'K', 'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3']
        data   = np.genfromtxt(self.fname_atmmod)

        # read original data from *.dat file into Python dictionary that contains 'data', 'name', and 'units'
        self.atm0 = {}
        for i, vname in enumerate(vnames):
            self.atm0[vname] = {'data':data[:, i], 'name':vname, 'units':units[i]}

        # 1. change the values in array from descending order to ascending order
        indices = np.argsort(self.atm0['altitude']['data'])
        for key in self.atm0.keys():
            self.atm0[key]['data'] = self.atm0[key]['data'][indices]

        # 2. calculate the mixing ratio from volume number density for each gas
        for key in self.atm0.keys():
            if key in self.gases:
                self.atm0[key]['data']  = self.atm0[key]['data']/self.atm0['air']['data']
                self.atm0[key]['units'] = 'N/A'


    def interp(self):
        """
        Interpolate atmospheric data to specified levels and layers.
        This method interpolates atmospheric properties from the original atmosphere
        model (self.atm0) to user-defined vertical levels and layers. It performs
        linear interpolation for most atmospheric variables and uses the barometric
        formula for pressure interpolation to maintain physical consistency.
        The method creates two new atmospheric profiles:
        - self.lev: Interpolated data at specified levels
        - self.lay: Interpolated data at layer midpoints

        Notes:
            - All atmospheric variables except 'altitude' and 'pressure' are
            interpolated linearly
            - Pressure is interpolated using the barometric formula via
            interp_pres_from_alt_temp() function
            - Layer thickness is calculated as the difference between consecutive levels
            - The method assumes self.levels and self.layers are already defined
        """

        # check whether the input height is within the atmosphere height range
        if self.levels.min() < self.atm0['altitude']['data'].min():
            sys.exit('Error   [atm_atmmod]: Input levels too low.')
        if self.levels.max() > self.atm0['altitude']['data'].max():
            sys.exit('Error   [atm_atmmod]: Input levels too high.')

        self.lev = {}
        self.lev = copy.deepcopy(self.atm0)
        self.lev['altitude']['data']  = self.levels

        self.lay = {}
        self.lay = copy.deepcopy(self.atm0)
        self.lay['altitude']['data']  = self.layers
        self.lay['thickness'] = { \
                 'name' : 'Thickness', \
                 'units':'km', \
                 'data':self.levels[1:]-self.levels[:-1]}

        # Linear interpolate to input levels and layers
        for key in self.atm0.keys():
            if key not in ['altitude', 'pressure']:
                self.lev[key]['data'] = np.interp(self.lev['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])
                self.lay[key]['data'] = np.interp(self.lay['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])

        # Use Barometric formula to interpolate pressure
        self.lev['pressure']['data'] = interp_pres_from_alt_temp(self.atm0['pressure']['data'], self.atm0['altitude']['data'], self.atm0['temperature']['data'], \
                self.lev['altitude']['data'], self.lev['temperature']['data'])
        self.lay['pressure']['data'] = interp_pres_from_alt_temp(self.atm0['pressure']['data'], self.atm0['altitude']['data'], self.atm0['temperature']['data'], \
                self.lay['altitude']['data'], self.lay['temperature']['data'])


    def add_ch4(self):

        ch4 = {'name':'ch4', 'units':'cm-3', 'data':interp_ch4(self.levels)}
        self.lev['ch4'] = ch4

        ch4 = {'name':'ch4', 'units':'cm-3', 'data':interp_ch4(self.layers)}
        self.lay['ch4'] = ch4


    def cal_num_den(self):

        self.lev['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':constants.Na/constants.R*self.lev['pressure']['data']/self.lev['temperature']['data']*1.0e-4}

        self.lay['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':constants.Na/constants.R*self.lay['pressure']['data']/self.lay['temperature']['data']*1.0e-4}

        for key in self.lev.keys():
            if key in self.gases:
                self.lev[key]['data']  = self.lev[key]['data'] * self.lev['factor']['data']
                self.lev[key]['units'] = 'cm-3'
                self.lay[key]['data']  = self.lay[key]['data'] * self.lay['factor']['data']
                self.lay[key]['units'] = 'cm-3'


class ARCSIXAtmModel:
    """
    ARCSIXAtmModel is similar to atm_atmmod but specifically designed for the ARCSIX project.
    It can be used to create atmospheric models based on the ARCSIX requirements and AFGL profiles.

    It is designed to be modular to allow for specific atmospheric modeling needs using a combination of reanalyses,
    satellite data, AFGL profiles, and ARCSIX measurements.
    """

    ID = 'ARCSIX Atmosphere Model'

    def __init__(self,
                 levels       = None,
                 fname_out    = None,
                 config_file  = None,
                 verbose      = False):

        self.levels       = levels
        self.fname_out    = fname_out
        self.verbose      = verbose
        self.config_file  = config_file

        # Load configuration if provided
        if config_file is not None:
            config = self.load_initial_config(config_file)

            if 'fname_base_atmmod' in config:
                self.fname_base_atmmod = config['fname_base_atmmod']
            else:
                sys.exit('Error [ARCSIXAtmModel]: Configuration file must specify "fname_base_atmmod".')

            if 'gases' in config:
                self.gases = config['gases']
            else:
                self.gases = None

            # Check for external data sources and load their variables
            if 'external_data_sources' in config:
                self.external_data_sources = config['external_data_sources']
                external_data_vars = self.external_data_sources.keys()
                if ('altitude' not in external_data_vars) or ('pressure' not in external_data_vars):
                    raise ValueError('Error [ARCSIXAtmModel]: Configuration file must specify "altitude" and "pressure" in "external_data_sources".')

            else:
                self.external_data_sources = None

        self.run()


    def create_base_atmosphere(self):
        """Create base atmospheric model from AFGL data"""

        # Standard AFGL variables and units
        vnames = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
        units = ['km', 'mb', 'K', 'cm^-3', 'cm^-3', 'cm^-3', 'cm^-3', 'cm^-3', 'cm^-3']
        self.afgl_vnames = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']

        # Load AFGL data
        data = np.genfromtxt(self.fname_base_atmmod)

        # Create base atmospheric dictionary
        self.atm0 = {}
        for i, vname in enumerate(vnames):
            self.atm0[vname] = {
                'data': data[:, i],
                'name': vname,
                'units': units[i],
                'source': os.path.basename(self.fname_base_atmmod)
            }

        # Sort by ascending altitude
        indices = np.argsort(self.atm0['altitude']['data'])
        for key in self.atm0.keys():
            self.atm0[key]['data'] = self.atm0[key]['data'][indices]

        self.afgl_levels = np.array(self.atm0['altitude']['data']) # create a copy for later use if needed

        # Convert gas densities to mixing ratios for gases
        for key in self.atm0.keys():
            if key in self.gases:
                self.atm0[key]['data'] = self.atm0[key]['data'] / self.atm0['air']['data']
                self.atm0[key]['units'] = 'kg kg**-1'

        if self.verbose:
            print('Message [ARCSIXAtmModel]: Created base atmosphere from AFGL data')


    def setup_vertical_levels_layers(self):
        """Determine vertical levels and layers from external data, user input, or AFGL default"""

        if self.levels is not None:
            # Use user-provided levels
            if self.levels.size <= 1:
                raise ValueError('Error [ARCSIXAtmModel]: Size of levels must be greater than 1.')
            if self.verbose:
                print('Message [ARCSIXAtmModel]: Using user-provided altitude levels')

        elif 'altitude' in self.external_data_sources:
            # Use external altitude data
            alt_data = self.load_external_data('altitude')
            self.levels = alt_data['data']

            # we want altitude in km
            if alt_data['units'].lower() == 'm':
                self.levels /= 1000.  # Convert meters to kilometers

            self.altitude_units = 'km'

            if self.verbose:
                print(f'Message [ARCSIXAtmModel]: Using altitude levels from {self.external_data_sources["altitude"]["file"]}')

        else:
            # Use AFGL altitude levels as default
            self.levels = self.afgl_levels
            self.altitude_units = 'km'

            if self.verbose:
                print('Message [ARCSIXAtmModel]: Using AFGL altitude levels as default')

        # Sort by ascending altitude
        # will need to re-sort the other variables later too
        self.levels = np.sort(self.levels)

        # Calculate layer midpoints
        self.layers = 0.5 * (self.levels[1:] + self.levels[:-1])


    def load_initial_config(self, config_file):
        """ load configuration from a YAML file """
        config = {}
        with open(config_file, 'r') as f:
            fconfig = yaml.safe_load(f)
            config['fname_base_atmmod'] = fconfig.get("fname_base_atmmod")
            config['gases'] = fconfig.get("gases")
            config['external_data_sources'] = fconfig.get("external_data_sources")

        return config


    def read_dat_file(self, fname):
        """
        Read .dat file of the atmospheric profile and return a df.
        Expects column names to be variables, row 0 to be units.
        """
        df = pd.read_table(fname, sep=',')
        print(dict(zip(list(df.columns), list(df.iloc[0].values)))) # expecting first row to be units
        df = df[1:].reset_index(drop=True) # drop the units row
        df = df.astype('float64') # convert all columns to float

        return df


    def run(self):

        # step 1: create base atmospheric model from AFGL data/base_fname_atmmod
        # creates self.atm0: Python dictionary
        self.create_base_atmosphere()

        # step 2: set up vertical levels and layers
        # creates self.levels and self.layers
        self.setup_vertical_levels_layers()

        # step 3: now bring in other data sources
        self.add_external_data_sources()

        # sort variables by ascending altitude before interpolating
        indices = np.argsort(self.atm0['altitude']['data'])
        for key in self.atm0.keys():
            if key not in self.afgl_vars_persist: # otherwise unchanged afgl vars will get sorted wrongly
                self.atm0[key]['data'] = self.atm0[key]['data'][indices]

        # step 4: interpolate to desired levels
        self.interpolate_to_levels()

        # step 5: convert mixing ratio [unitless or kg/kg] to number density [cm^-3]
        self.calculate_number_density_from_mass_mixing_ratio()

        # step 6: (optional) save to file
        if self.fname_out is not None:
            self.save_to_file()


    def add_external_data_sources(self):
        """Add external data sources to the atmospheric model."""

        if self.external_data_sources is not None:
            external_vars = self.external_data_sources.keys()

            # check which variables in afgl will be preserved
            # as these will need additional interpolation step to the new altitudes
            self.afgl_vars_persist = list(set(self.afgl_vnames) - set(external_vars))
            self.afgl_vars_overriden = list(set(self.afgl_vnames) & set(external_vars))
            self.new_vars_added = list(set(external_vars) - set(self.afgl_vnames))

            if len(self.afgl_vars_persist) > 0:
                print(f'Message [ARCSIXAtmModel]: Preserving the following AFGL variables, will need additional interpolation: {self.afgl_vars_persist}')

            if len(self.afgl_vars_overriden) > 0:
                print(f'Message [ARCSIXAtmModel]: Overriding the following AFGL variables with external data: {self.afgl_vars_overriden}')

            if len(self.new_vars_added) > 0:
                print(f'Message [ARCSIXAtmModel]: Adding the following new variables from external data: {self.new_vars_added}')

            for var in external_vars:

                if self.verbose:
                    print(f'Message [ARCSIXAtmModel]: Adding external data for {var} from {self.external_data_sources[var]["file"]}')


                # external_data_var is a dict containing the keys 'name', 'units', 'source', and 'data'
                external_data_var = self.load_external_data(var=var)

                # Override AFGL data with external data
                if var in self.atm0:
                    self.atm0[var] = external_data_var

                    # Update units for altitude separately
                    if var == 'altitude':
                        self.atm0[var]['name'] = 'altitude'
                        self.atm0[var]['data'] = self.levels
                        self.atm0[var]['units'] = self.altitude_units
                        self.atm0[var]['source'] = os.path.basename(self.external_data_sources[var]['file'])

                        print(self.altitude_units)

                    else:
                        self.atm0[var] = external_data_var

                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Replaced {var} with external data from {self.external_data_sources[var]["file"]}')

                # Add new variable not in AFGL
                else:
                    self.atm0[var] = external_data_var
                    self.atm0[var]['source'] = os.path.basename(self.external_data_sources[var]['file'])
                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Added new variable {var} from external data')

        # proceed with just AFGL that was already read in earlier
        else:
            print('Message [ARCSIXAtmModel]: No external data sources found. Proceeding with AFGL data only.')


    def load_external_data(self, var):
        """Load external data as a dictionary for a given variable from specified file"""
        df = self.read_dat_file(fname=self.external_data_sources[var]['file'])
        data_column_name = self.external_data_sources[var]['data_column']
        data = df[data_column_name].values

        external_data = {'name': var, 'units': self.external_data_sources[var]['units'], 'source':os.path.basename(self.external_data_sources[var]['file']), 'data': data}
        return external_data


    def interpolate_to_levels(self):
        """Interpolate atmospheric data to specified levels and layers"""

        # Check altitude range
        if self.levels.min() < self.atm0['altitude']['data'].min():
            print('Warning [ARCSIXAtmModel]: Input levels extend below AFGL range')
        if self.levels.max() > self.atm0['altitude']['data'].max():
            print('Warning [ARCSIXAtmModel]: Input levels extend above AFGL range')

        # Initialize level and layer dictionaries
        self.lev = copy.deepcopy(self.atm0)
        self.lev['altitude']['data'] = self.levels

        # initialize layer from atm0
        self.lay = copy.deepcopy(self.atm0)
        self.lay['altitude']['data'] = self.layers
        self.lay['thickness'] = {
            'name': 'Thickness',
            'units': 'km',
            'data': self.levels[1:] - self.levels[:-1],
            'source': 'calculated from levels'
        }

        # Interpolate all variables except altitude and pressure to the user input/final altitudes in self.lev and self.lay
        for key in self.atm0.keys():
            if key not in ['altitude', 'pressure']:

                # if afgl variables was preserved, it will need to be updated
                # to the new altitudes first before interpolation
                if key in self.afgl_vars_persist:
                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Interpolating AFGL variable {key}')

                    # Store original AFGL data for interpolation
                    original_afgl_data = self.atm0[key]['data'].copy()
                    original_afgl_altitudes = self.afgl_levels.copy()

                    self.lev[key]['data'] = np.interp(self.lev['altitude']['data'], original_afgl_altitudes, original_afgl_data)
                    self.lay[key]['data'] = np.interp(self.lay['altitude']['data'], original_afgl_altitudes, original_afgl_data)

                else:
                    self.lev[key]['data'] = np.interp(self.lev['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])
                    self.lay[key]['data'] = np.interp(self.lay['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])

                    # print(key, '\n non afgl', self.lev[key]['data'])

        # else: # Use barometric formula for pressure interpolation
        self.lev['pressure']['data'] = interp_pres_from_alt_temp(self.atm0['pressure']['data'], self.atm0['altitude']['data'], self.atm0['temperature']['data'], self.lev['altitude']['data'], self.lev['temperature']['data'])

        self.lay['pressure']['data'] = interp_pres_from_alt_temp(self.atm0['pressure']['data'], self.atm0['altitude']['data'], self.atm0['temperature']['data'], self.lay['altitude']['data'], self.lay['temperature']['data'])


    def calculate_air_number_density(self):
        """
        Calculate number density of air and update lev and lay in-place

        Step 1: calculate air density in kg/m3 using ideal gas law rho = P / (R * T)
        Step 2: convert air density from kg/m3 to number density in cm^-3 using: n = rho * Na / M_dry
        where M_dry is molar mass in kg/mol, Na is Avogadro's number

        Full equation n = (P * Na)/(M_dry * R * T)
        """

        # 100 is for hPa to Pa
        self.lev['air']['data'] = {
            'units': 'cm^-3',
            'data': (self.lev['pressure']['data'] * 100 * constants.Na) / (constants.M_dry * constants.R * self.lev['temperature']['data'])
        }
        self.lay['air']['data'] = {
            'units': 'cm^-3',
            'data': (self.lay['pressure']['data'] * 100 * constants.Na) / (constants.M_dry * constants.R * self.lay['temperature']['data'])
        }


    def calculate_number_density_from_mass_mixing_ratio(self):
        """
        Calculate number density from mass mixing ratio for all gases and update lev and lay in-place

        Units of mass mixing ratio must be kg/kg, output number density will be in #/cm^3 (or simply, cm^-3)
        Calculation is done via ideal gas law:
        number density of gas x:
        n_x = (Na * P * C_x) / (R * T)

        where
        Na: Avogadro's number (/mol)
        R: Gas constant (J/(mol*K))
        P: Pressure (mb)
        T: Temperature (K)

        Reference: Eq. (7): https://projects.iq.harvard.edu/files/acmg/files/intro_atmo_chem_bookchap1.pdf
        """

        # 100 is the conversion from hPa to Pa; all others being SI results in /m^3 so an additional factor of 1e-6 is needed to convert to /cm^3
        for key in self.gases:

            if (key == 'air') and ('air' in self.afgl_vars_persist): # needs to be dealt with separately
                self.calculate_air_number_density() # updates lev and key in place
                continue

            if key in self.lev:
                self.lev[key]['data'] = (constants.Na * self.lev['pressure']['data'] * 100 * self.lev[key]['data']) * 10**-6 / (constants.R * self.lev['temperature']['data'])
                self.lev[key]['units'] = 'cm^-3'

            if key in self.lay:
                self.lay[key]['data'] = (constants.Na * self.lay['pressure']['data'] * 100 * self.lay[key]['data']) * 10**-6 / (constants.R * self.lay['temperature']['data'])
                self.lay[key]['units'] = 'cm^-3'


    def save_to_file(self, plot=True):
        """Save atmospheric model to hdf5"""
        outdir = os.path.dirname(self.fname_out)
        if len(outdir) > 0 and (not os.path.exists(outdir)):
            os.makedirs(outdir)
        elif len(outdir) == 0:
            outdir = './'
            self.fname_out = os.path.join(outdir, self.fname_out)

        if not self.fname_out.endswith('.h5') and os.path.splitext(self.fname_out)[1] == '':
            self.fname_out += '.h5'

        elif not self.fname_out.endswith('.h5') and os.path.splitext(self.fname_out)[1] != '':
            self.fname_out = self.fname_out.replace(os.path.splitext(self.fname_out)[1], '.h5')

        #TODO: Improve attribute naming and descriptions in the file

        # save to hdf5
        import h5py

        with h5py.File(self.fname_out, 'w') as f:
            # Create groups for levels and layers
            lev_group = f.create_group('levels')
            lay_group = f.create_group('layers')

            # Save level data
            for var in self.lev.keys():
                lev_group.create_dataset(var, data=self.lev[var]['data'])
                lev_group[var].attrs['units'] = self.lev[var]['units']
                lev_group[var].attrs['source'] = self.lev[var]['source']

            # Save layer data
            for var in self.lay.keys():
                lay_group.create_dataset(var, data=self.lay[var]['data'])
                lay_group[var].attrs['units'] = self.lay[var]['units']
                lay_group[var].attrs['source'] = self.lay[var]['source']

        if self.verbose:
            print(f'Message [ARCSIXAtmModel]: Saved to {self.fname_out}')


        if plot:
            from er3t.util.plot_util import set_plot_fonts, MPL_STYLE_PATH
            import matplotlib.pyplot as plt

            # Plot level data
            n_panels = len(self.lev.keys()) - 1
            var_names = list(self.lev.keys())
            var_names.remove('pressure')  # Remove pressure for plotting
            fig, axes = plt.subplots(1, n_panels, figsize=(30, 10))
            plt.style.use(MPL_STYLE_PATH)

            for i, var_name in enumerate(var_names):
                axes[i].plot(self.lev[var_name]['data'], self.lev['pressure']['data'], label=var_name)
                axes[i].set_xlabel(f'{var_name} {self.lev[var_name]["units"]}')
                axes[i].set_title(f'{var_name} Profile')
                axes[i].invert_yaxis()
                axes[i].grid(True, alpha=0.3)

            axes[0].set_ylabel('Pressure (hPa)')

            fig.savefig(self.fname_out.replace('.h5', '_levels.png'), bbox_inches='tight')
            plt.close('all')

            # Plot layer data
            n_panels = len(self.lev.keys()) - 1
            var_names = list(self.lev.keys())
            var_names.remove('pressure')  # Remove pressure for plotting
            fig, axes = plt.subplots(1, n_panels, figsize=(30, 10))
            plt.style.use(MPL_STYLE_PATH)

            for i, var_name in enumerate(var_names):
                axes[i].plot(self.lay[var_name]['data'], self.lay['pressure']['data'], label=var_name)
                axes[i].set_xlabel(f'{var_name} {self.lay[var_name]["units"]}')
                axes[i].set_title(f'{var_name} Profile')
                axes[i].invert_yaxis()
                axes[i].grid(True, alpha=0.3)

            axes[0].set_ylabel('Pressure (hPa)')

            fig.savefig(self.fname_out.replace('.h5', '_layers.png'), bbox_inches='tight')
            plt.close('all')
            print(f'Saved figure to {os.path.dirname(self.fname_out)}')

if __name__ == '__main__':

    # define levels in km
    # levels = np.concatenate((np.arange(0, 2.1, 0.2),
    #                         np.arange(2.5, 4.1, 0.5),
    #                         np.arange(5.0, 10.1, 2.5),
    #                         np.array([15, 20, 30., 40., 50.])))
    arcsix_atm_mod = ARCSIXAtmModel(levels=None, config_file='/Users/vikas/workspace/arctic/er3t-git/arcsix_atm_profile_config.yaml', verbose=1, fname_out='/Users/vikas/workspace/arctic/er3t-git/data/test_data/arcsix_atm_profile_output.h5')
