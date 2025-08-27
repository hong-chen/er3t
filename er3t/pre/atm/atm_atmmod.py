import os
import sys
import copy
import pickle
import datetime
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

    def __init__(self,
                 levels       = None,
                 fname        = None,
                 fname_atmmod = '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
                 overwrite    = False,
                 plot         = False,
                 verbose      = False):

        er3t.util.add_reference(self.reference)

        self.verbose      = verbose
        self.fname_atmmod = fname_atmmod
        self.plot         = plot

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):
            # Decide how to load based on extension
            ext = os.path.splitext(fname)[1].lower()
            if ext in ['.pkl', '.pickle', '.pk']:
                self.load_pickle(fname)
            elif ext in ['.h5', '.hdf5']:
                raise NotImplementedError("HDF5 load not yet implemented for atm_atmmod. Provide 'levels' to regenerate or use a pickle file.")
            else:
                self.load_pickle(fname)  # fallback

        elif ((levels is not None) and (fname is not None)):
            # Create and auto-save (supports .h5 or pickle based on extension)
            self.run(levels)
            self._auto_save(fname)

        elif ((levels is not None) and (fname is None)):
            # Just create in memory
            self.run(levels)

        else:
            sys.exit("Error   [atm_atmmod]: Please provide 'levels' or an existing file to load.")



    def load(self, fname):
        return self.load_pickle(fname)

    def load_pickle(self, fname):

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

        # self.save_to_hdf5(fname='afgl_atm_atmmod_output.h5')


    def dump(self, fname):
        """Backward-compatible pickle dump (deprecated if extension is .h5)."""
        ext = os.path.splitext(fname)[1].lower()
        if ext in ['.h5', '.hdf5']:
            if self.verbose:
                print(f"Warning [atm_atmmod]: Attempted to pickle with HDF5 extension '{ext}'. Saving as true HDF5 instead.")
            self.save_to_hdf5(fname)
            return
        self.dump_pickle(fname)


    def dump_pickle(self, fname):
        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [atm_atmmod]: Saving pickle object into %s ...' % fname)
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    def save_to_hdf5(self, fname):
        """Save atmospheric profile (levels & layers) to an HDF5 file.

        Fixes prior issue where users passed a .h5 filename to __init__, which
        resulted in a pickle file with an HDF5 extension (causing 'file signature not found').

        Parameters
        ----------
        fname : str
            Target filename. Will enforce .h5 extension.
        """
        if self.lev is None or self.lay is None:
            raise RuntimeError("save_to_hdf5 called before atmosphere was generated. Run run(levels) first.")

        # Directory handling
        outdir = os.path.dirname(fname)
        if outdir and (not os.path.exists(outdir)):
            os.makedirs(outdir, exist_ok=True)

        # Normalize extension
        root, ext = os.path.splitext(fname)
        if ext.lower() not in ['.h5', '.hdf5']:
            fname = root + '.h5'

        import h5py
        with h5py.File(fname, 'w') as f:
            lev_group = f.create_group('levels')
            lay_group = f.create_group('layers')

            # Helper to write dict entries
            def _write_group(src_dict, grp):
                for var, dd in src_dict.items():
                    if not isinstance(dd, dict) or 'data' not in dd:
                        continue  # skip malformed entries
                    data = np.asarray(dd['data'])
                    dset = grp.create_dataset(var, data=data)
                    # Attributes (only add if present)
                    if 'units' in dd:
                        dset.attrs['units'] = dd['units']
                    if 'name' in dd:
                        dset.attrs['name'] = dd['name']

            _write_group(self.lev, lev_group)
            _write_group(self.lay, lay_group)

            # Metadata
            f.attrs['title'] = 'Atmospheric Model Profiles from AFGL'
            f.attrs['description'] = 'AFGL Atmospheric Model Profiles'
            f.attrs['computer'] = os.uname()[1]
            f.attrs['software'] = f'EaR3T file {os.path.abspath(__file__)}, class {self.__class__.__name__}'
            f.attrs['created_on'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')

        if self.verbose:
            print(f"Message [atm_atmmod]: Saved HDF5 to {fname}")

        if self.plot:
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

            fig.savefig(fname.replace('.h5', '_levels.png'), bbox_inches='tight')
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

            fig.savefig(fname.replace('.h5', '_layers.png'), bbox_inches='tight')
            plt.close('all')
            print(f'Saved figure to {os.path.dirname(fname)}')



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

        # 2. calculate the volume mixing ratio from volume number density for each gas
        for key in self.atm0.keys():
            if key in self.gases:
                self.atm0[key]['data']  = self.atm0[key]['data']/self.atm0['air']['data']
                self.atm0[key]['units'] = 'dimensionless' # kg/kg


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
          'data':constants.NA/constants.R*self.lev['pressure']['data']/self.lev['temperature']['data']*1.0e-4}

        self.lay['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':constants.NA/constants.R*self.lay['pressure']['data']/self.lay['temperature']['data']*1.0e-4}

        for key in self.lev.keys():
            if key in self.gases:
                self.lev[key]['data']  = self.lev[key]['data'] * self.lev['factor']['data']
                self.lev[key]['units'] = 'cm-3'
                self.lay[key]['data']  = self.lay[key]['data'] * self.lay['factor']['data']
                self.lay[key]['units'] = 'cm-3'


class ARCSIXAtmModel:
    """
    ARCSIXAtmModel is similar to atm_atmmod but specifically designed for the ARCSIX project.
    It can be used to create atmospheric models using ARCSIX data, reanalyses, and AFGL profiles.

    It is designed to be modular to allow for specific atmospheric modeling needs using a combination of reanalyses,
    satellite data, AFGL profiles, and ARCSIX measurements. This is achieved via a flexible data handling system
    that can incorporate various data sources and seamlessly integrate different altitude grids to the desired one.
    """

    ID = 'ARCSIX Atmosphere Model'

    def __init__(self,
                 levels        = None,
                 levels_source = 'external',
                 fname_out     = None,
                 config_file   = None,
                 plot          = False,
                 verbose       = False):

        self.levels       = levels
        self.fname_out    = fname_out
        self.verbose      = verbose
        self.config_file  = config_file
        self.plot         = plot

        # set levels_source based on input
        if (levels is None) and (levels_source == 'afgl'):
            self.levels_source = 'afgl'
            print('Message [ARCSIXAtmModel]: Using AFGL default altitude levels')

        elif (levels is None) and (levels_source == 'external'):
            self.levels_source = 'external'
            print('Message [ARCSIXAtmModel]: Using external altitude levels')

        elif (levels is not None):
            self.levels_source = 'user'
            print('Message [ARCSIXAtmModel]: Using user-defined altitude levels.\nPlease ensure that that they are in units of km')

        else:
            raise ValueError('Error [ARCSIXAtmModel]: Please provide valid levels or levels_source.')

        # Initialize grid tracking for modular data source handling
        self.variable_grid_mapping = {}  # Track which altitude grid each variable uses

        # Load configuration if provided
        if config_file is not None:
            config = self.load_initial_config(config_file)

            if 'fname_base_atmmod' in config:
                self.fname_base_atmmod = config['fname_base_atmmod']
                # add legacy variable
                self.fname_atmmod = config['fname_base_atmmod']
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

                # Only require altitude and pressure for external levels mode
                if self.levels_source == 'external':
                    if ('altitude' not in external_data_vars) or ('pressure' not in external_data_vars):
                        raise ValueError('Error [ARCSIXAtmModel]: Configuration file must specify "altitude" and "pressure" in "external_data_sources" when using external levels.')

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

        self.afgl_levels = np.sort(np.array(self.atm0['altitude']['data'])) # create a copy for later use if needed

        # Track that all AFGL variables start on the AFGL grid
        for vname in vnames:
            self.variable_grid_mapping[vname] = 'afgl'

        # Convert gas densities to mixing ratios for gases
        for key in self.atm0.keys():
            if key in self.gases:
                self.atm0[key]['data'] = self.atm0[key]['data'] / self.atm0['air']['data']
                self.atm0[key]['units'] = 'kg kg**-1'

        if self.verbose:
            print('Message [ARCSIXAtmModel]: Created base atmosphere from AFGL data')


    def setup_vertical_levels_layers(self):
        """Determine vertical levels and layers from external data, user input, or AFGL default"""

        if self.levels_source == 'user':
            # Use user-provided levels
            if self.levels.size <= 1:
                raise ValueError('Error [ARCSIXAtmModel]: Size of levels must be greater than 1.')
            if self.verbose:
                print('Message [ARCSIXAtmModel]: Using user-provided altitude levels')

            # and also copy to another variable for sorting and interpolation later and set others to None
            self.user_levels = np.sort(self.levels.copy())
            self.atm0['altitude']['data'] = self.user_levels
            self.external_levels = None

            # Update grid mapping for altitude since it's now on the target grid
            self.variable_grid_mapping['altitude'] = 'target'

        elif self.levels_source == 'external':
            # Use external altitude data
            alt_data_dict = self.load_external_data('altitude')

            # we want altitude in km
            if alt_data_dict['units'].lower() == 'm':
                alt_data_dict['data'] /= 1000.  # Convert meters to kilometers

            self.levels = alt_data_dict['data']

            if self.verbose:
                print(f'Message [ARCSIXAtmModel]: Using altitude levels from {self.external_data_sources["altitude"]["file"]}')

            # but also copy to another variable for sorting and interpolation, set others to None
            self.external_levels = alt_data_dict['data'].copy()
            self.user_levels = None
            self.atm0['altitude']['data'] = self.external_levels

            # Update grid mapping for altitude since it's now on the external grid
            self.variable_grid_mapping['altitude'] = 'external'

        else:
            # Use AFGL altitude levels as default, set others to None
            self.levels = self.afgl_levels.copy()
            self.external_levels = None
            self.user_levels = None

            if self.verbose:
                print('Message [ARCSIXAtmModel]: Using AFGL altitude levels as default')

        # Sort by ascending altitude
        # will need to re-sort the other variables later too
        self.levels = np.sort(self.levels)

        # Calculate layer midpoints
        self.layers = 0.5 * (self.levels[1:] + self.levels[:-1])

        # altitude must be in km
        self.altitude_units = 'km'


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
        # print(dict(zip(list(df.columns), list(df.iloc[0].values)))) # expecting first row to be units
        df = df[1:].reset_index(drop=True) # drop the units row
        df = df.astype('float64') # convert all columns to float

        return df


    def run(self):

        # step 1: create base atmospheric model from AFGL data/base_fname_atmmod
        # creates self.atm0: Python dictionary
        self.create_base_atmosphere()

        # step 2: set up vertical levels and layers
        # creates self.levels and self.layers, might update self.atm0 altitude data
        self.setup_vertical_levels_layers()

        # step 3: now bring in other data sources
        self.add_external_data_sources()

        # step 4: validate grid consistency before processing
        # this step is important to ensure that sorting and interpolation are done correctly
        self.validate_grid_consistency()

        # step 5: sort variables by their respective altitude grids before interpolating
        self.sort_atmospheric_data()

        # step 6: interpolate to desired levels
        self.interpolate_to_levels()

        # step 7: convert mixing ratio [unitless or kg/kg] to number density [cm^-3]
        self.calculate_number_density_from_mixing_ratio()

        # step 8: (optional) save to file
        if self.fname_out is not None:
            self.save_to_hdf5()


    def sort_atmospheric_data(self):
        """
        Sort atmospheric data by ascending altitude, handling mixed data sources correctly.

        This method handles the complex case where some variables come from external sources
        (with potentially different altitude grids) and others remain from AFGL data.
        """

        afgl_sort_indices = np.argsort(self.afgl_levels)
        self.afgl_levels = self.afgl_levels[afgl_sort_indices] # sort AFGL levels
        if self.external_levels is not None:
            external_sort_indices = np.argsort(self.external_levels)
            self.external_levels = self.external_levels[external_sort_indices] # sort external levels

        # Apply appropriate sorting to each variable based on its grid assignment
        for key in self.atm0.keys():
            if key != 'altitude': # only sort non-altitude vars
                grid_type = self.variable_grid_mapping.get(key, 'afgl')

                if grid_type == 'afgl':
                    self.atm0[key]['data'] = self.atm0[key]['data'][afgl_sort_indices]
                elif grid_type == 'external' and self.external_levels is not None:
                    self.atm0[key]['data'] = self.atm0[key]['data'][external_sort_indices]
                # Don't sort target grid variables - they're already on final grid

        if self.verbose:
            print('Message [ARCSIXAtmModel]: Completed atmospheric data sorting')


    def add_external_data_sources(self):
        """Add external data sources to the atmospheric model."""

        if self.external_data_sources is not None:
            external_vars = self.external_data_sources.keys()

            # check which variables in afgl will be preserved
            # as these will need additional interpolation step to the new altitudes
            self.afgl_vars_persist = list(set(self.afgl_vnames) - set(external_vars))
            self.afgl_vars_overriden = list(set(self.afgl_vnames) & set(external_vars))
            self.new_vars_added = list(set(external_vars) - set(self.afgl_vnames))

            if (len(self.afgl_vars_persist) > 0) and self.verbose:
                print(f'Message [ARCSIXAtmModel]: Preserving the following AFGL variables, will need additional interpolation: {self.afgl_vars_persist}')

            if (len(self.afgl_vars_overriden) > 0) and self.verbose:
                print(f'Message [ARCSIXAtmModel]: Overriding the following AFGL variables with external data: {self.afgl_vars_overriden}')

            if (len(self.new_vars_added) > 0) and self.verbose:
                print(f'Message [ARCSIXAtmModel]: Adding the following new variables from external data: {self.new_vars_added}')

            # Initialize external levels tracking
        self.external_data_grids = {}  # Track altitude grids for each external variable

        if self.external_data_sources is not None:
            external_vars = list(self.external_data_sources.keys())

            # identify which variables are overridden vs preserved vs new
            # as these will need additional interpolation step to the new altitudes
            self.afgl_vars_persist = list(set(self.afgl_vnames) - set(external_vars))
            self.afgl_vars_overriden = list(set(self.afgl_vnames) & set(external_vars))
            self.new_vars_added = list(set(external_vars) - set(self.afgl_vnames))

            if (len(self.afgl_vars_persist) > 0) and self.verbose:
                print(f'Message [ARCSIXAtmModel]: Preserving the following AFGL variables, will need additional interpolation: {self.afgl_vars_persist}')

            if (len(self.afgl_vars_overriden) > 0) and self.verbose:
                print(f'Message [ARCSIXAtmModel]: Overriding the following AFGL variables with external data: {self.afgl_vars_overriden}')

            if (len(self.new_vars_added) > 0) and self.verbose:
                print(f'Message [ARCSIXAtmModel]: Adding the following new variables from external data: {self.new_vars_added}')

            for var in external_vars:

                if self.verbose:
                    print(f'Message [ARCSIXAtmModel]: Adding external data for {var} from {self.external_data_sources[var]["file"]}')

                # external_data_var is a dict containing the keys 'name', 'units', 'source', and 'data'
                external_data_var = self.load_external_data(var=var)

                # Load and store the corresponding altitude grid for this variable
                if var != 'altitude':
                    try:
                        # Try to get altitude data from the same file
                        var_df = self.read_dat_file(fname=self.external_data_sources[var]['file'])
                        if 'altitude_km' in var_df.columns:
                            altitude_data = var_df['altitude_km'].values
                            if self.external_data_sources[var]['units'].lower() == 'm':
                                altitude_data /= 1000.
                            self.external_data_grids[var] = altitude_data
                        elif 'altitude' in var_df.columns:
                            altitude_data = var_df['altitude'].values
                            self.external_data_grids[var] = altitude_data
                    except Exception:
                        # If can't get altitude, will use AFGL grid
                        pass

                # Override AFGL data with external data
                if var in self.atm0:

                    # for altitude, we don't want to replace anything already done unless external mode was selected
                    if (var == 'altitude'):
                        external_levels = external_data_var['data'].copy() # make copy for sorting
                        if external_data_var['units'].lower() == 'm': # ensure units are consistent (in km)
                            external_levels /= 1000.

                        self.external_levels = np.sort(external_levels) # also save as separate variable for sorting

                        if self.levels_source == 'external': # only update the altitude dictionary if levels source was external
                            self.atm0[var]['name'] = 'altitude'
                            self.atm0[var]['data'] = self.levels
                            self.atm0[var]['units'] = self.altitude_units
                            self.atm0[var]['source'] = os.path.basename(self.external_data_sources[var]['file'])
                            self.variable_grid_mapping[var] = 'target'  # Final target grid

                        elif self.levels_source == 'user': # if user provided levels, do not change anything except source
                            self.atm0[var]['source'] = 'user-defined'
                            self.variable_grid_mapping[var] = 'target'  # Final target grid

                    elif (var != 'altitude'):
                        # update or add the variable as is
                        self.atm0[var] = external_data_var
                        self.variable_grid_mapping[var] = 'external'  # External source grid

                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Replaced {var} with external data from {self.external_data_sources[var]["file"]}')

                # if new variable not in AFGL, update source and print
                else:
                    # update or add the variable as is
                    self.atm0[var] = external_data_var
                    self.atm0[var]['source'] = os.path.basename(self.external_data_sources[var]['file'])
                    self.variable_grid_mapping[var] = 'external'  # External source grid
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


    def get_source_grid_for_variable(self, var):
        """Get the source grid type for a given variable"""
        return self.variable_grid_mapping.get(var, 'afgl')


    def get_altitudes_for_grid(self, grid_type):
        """Get the altitude array for a specific grid type"""
        if grid_type == 'afgl':
            return self.afgl_levels
        elif grid_type == 'external':
            return self.external_levels  # May be None
        elif grid_type == 'target':
            # Return appropriate target grid based on levels source
            if self.levels_source == 'user':
                return self.user_levels
            elif self.levels_source == 'external':
                return self.external_levels
            else:
                return self.levels
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")


    def get_altitudes_for_variable(self, var):
        """Get the altitude array for a specific variable, handling variable-specific grids"""
        grid_type = self.get_source_grid_for_variable(var)

        if grid_type == 'external' and var in self.external_data_grids:
            # Use variable-specific grid if available
            return self.external_data_grids[var]
        else:
            # Use standard grid type
            return self.get_altitudes_for_grid(grid_type)


    def validate_grid_consistency(self):
        """Validate that all variables have consistent grid assignments"""
        for var, grid_type in self.variable_grid_mapping.items():
            if var in self.atm0:
                grid_altitudes = self.get_altitudes_for_grid(grid_type)
                if grid_altitudes is None:
                    # Skip validation for variables on grids that don't exist
                    continue

                expected_size = grid_altitudes.shape[0]
                actual_size = self.atm0[var]['data'].shape[0]

                if actual_size != expected_size:
                    raise ValueError(f"Variable '{var}' has {actual_size} points but "
                                   f"its assigned grid '{grid_type}' has {expected_size} points")

        if self.verbose:
            print('Message [ARCSIXAtmModel]: Grid consistency validation passed')


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

        # Interpolate all variables except altitude and pressure using grid-aware approach
        for key in self.atm0.keys():
            if key not in ['altitude', 'pressure']:
                source_altitudes = self.get_altitudes_for_variable(key)

                # Skip interpolation if source grid doesn't exist
                if source_altitudes is None:
                    if self.verbose:
                        print(f'Warning [ARCSIXAtmModel]: Skipping {key} - source grid not available')
                    continue

                if self.verbose:
                    grid_type = self.get_source_grid_for_variable(key)
                    print(f'Message [ARCSIXAtmModel]: Interpolating {key} from {grid_type} grid')

                # Ensure source data and altitudes have compatible shapes
                if self.atm0[key]['data'].shape[0] != source_altitudes.shape[0]:
                    raise ValueError(f"Shape mismatch for {key}: data shape {self.atm0[key]['data'].shape} "
                                   f"doesn't match altitude grid shape {source_altitudes.shape}")

                self.lev[key]['data'] = np.interp(
                    self.lev['altitude']['data'],
                    source_altitudes,
                    self.atm0[key]['data']
                )


                self.lay[key]['data'] = np.interp(
                    self.lay['altitude']['data'],
                    source_altitudes,
                    self.atm0[key]['data']
                )        # Handle pressure interpolation with consistent reference grid
        # Find the grid that has both pressure and temperature data
        pressure_grid = self.get_source_grid_for_variable('pressure')
        temperature_grid = self.get_source_grid_for_variable('temperature')

        if pressure_grid == temperature_grid:
            # Same grid - use directly
            ref_pressure_grid = pressure_grid
        else:
            # Different grids - use AFGL as reference and interpolate if needed
            ref_pressure_grid = 'afgl'
            if pressure_grid != 'afgl':
                # Interpolate pressure to AFGL grid
                pressure_source_altitudes = self.get_altitudes_for_grid(pressure_grid)
                self.atm0['pressure']['data'] = np.interp(
                    self.afgl_levels,
                    pressure_source_altitudes,
                    self.atm0['pressure']['data']
                )
            if temperature_grid != 'afgl':
                # Interpolate temperature to AFGL grid
                temp_source_altitudes = self.get_altitudes_for_grid(temperature_grid)
                self.atm0['temperature']['data'] = np.interp(
                    self.afgl_levels,
                    temp_source_altitudes,
                    self.atm0['temperature']['data']
                )

        ref_altitudes = self.get_altitudes_for_grid(ref_pressure_grid)

        if self.verbose:
            print(f'Message [ARCSIXAtmModel]: Using {ref_pressure_grid} grid for pressure interpolation')

        # Use barometric formula for pressure interpolation
        self.lev['pressure']['data'] = interp_pres_from_alt_temp(
            self.atm0['pressure']['data'], ref_altitudes, self.atm0['temperature']['data'],
            self.lev['altitude']['data'], self.lev['temperature']['data']
        )

        self.lay['pressure']['data'] = interp_pres_from_alt_temp(
            self.atm0['pressure']['data'], ref_altitudes, self.atm0['temperature']['data'],
            self.lay['altitude']['data'], self.lay['temperature']['data']
        )


    def calculate_air_number_density(self):
        """
        Calculate number density of air and update lev and lay in-place

        Step 1: calculate air density in kg/m3 using ideal gas law rho = P / (R * T)
        Step 2: convert air density from kg/m3 to number density in cm^-3 using: n = rho * Na / M_dry
        where M_dry is molar mass in kg/mol, Na is Avogadro's number

        Full equation n = (P * Na)/(M_dry * R * T)
        """

        # 100 is for hPa to Pa, 1e-6 is for m3 to cm3, so final factor is 1e-4
        self.lev['air']['data'] = {
            'units': 'cm^-3',
            'data': (self.lev['pressure']['data'] * constants.NA) * 1e-4 / (constants.M_dry * constants.R * self.lev['temperature']['data'])
        }
        self.lay['air']['data'] = {
            'units': 'cm^-3',
            'data': (self.lay['pressure']['data'] * constants.NA) * 1e-4 / (constants.M_dry * constants.R * self.lay['temperature']['data'])
        }


    def calculate_number_density_from_mixing_ratio(self):
        """
        Calculate number density from mass or volume mixing ratios for all gases and update lev and lay in-place

        For mass mixing ratios (kg/kg):
        number density of gas x: n_x = (Na * P * C_x) / (R * T)

        For volume mixing ratios (#/cm3):
        First convert to dimensionless volume mixing ratio, then apply factor

        where
        Na: Avogadro's number (/mol)
        R: Gas constant (J/(mol*K))
        P: Pressure (mb)
        T: Temperature (K)

        Reference: Eq. (7): https://projects.iq.harvard.edu/files/acmg/files/intro_atmo_chem_bookchap1.pdf
        """

        # 100 is the conversion from hPa to Pa; all others being SI results in /m^3 so an additional factor of 1e-6 is needed to convert to /cm^3 and therefore the 1e-4 is used as the final unit conversion factor
        self.lev['factor'] = {
            'name': 'number density factor',
            'units': 'cm^-3',
            'data': (constants.NA / constants.R) * (self.lev['pressure']['data'] / self.lev['temperature']['data']) * 1.0e-4,
            'source': 'calculated'
        }

        self.lay['factor']  = {
          'name': 'number density factor',
          'units': 'cm^-3',
          'data': (constants.NA / constants.R) * (self.lay['pressure']['data'] / self.lay['temperature']['data']) * 1.0e-4,
          'source': 'calculated'
        }

        mmr_units = ['kg/kg', 'n/a', 'unitless', 'dimensionless', 'g/g']
        vmr_units = ['#/cm3', 'cm^-3', '/cm3']

        for gas in self.gases:

            if (gas == 'air') and ('air' in self.afgl_vars_persist): # needs to be dealt with separately
                self.calculate_air_number_density() # updates lev and key in place
                continue

            # Handle levels
            if gas in self.lev:
                if self.lev[gas]['units'] in mmr_units:
                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Converting {gas} from mass mixing ratio to number density')

                    self.lev[gas]['data'] = self.lev[gas]['data'] * self.lev['factor']['data'] # apply factor
                    self.lev[gas]['units'] = 'cm^-3'

                elif self.lev[gas]['units'] in vmr_units:
                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Converting {gas} from volume mixing ratio to number density')

                    vmr_lev_data = self.lev[gas]['data'] / self.lev['air']['data'] # dimensionless vol. mixing ratio
                    self.lev[gas]['data'] = vmr_lev_data * self.lev['factor']['data'] # apply factor to go back to volume units
                    self.lev[gas]['units'] = 'cm^-3'

                else:
                    raise ValueError(f'Error [ARCSIXAtmModel]: Unrecognized units for {gas} in levels: {self.lev[gas]["units"]}.\nIf the gas is in mass mixing ratio units, ensure it is in {mmr_units}, or if it is in volume mixing ratio units, ensure it is in {vmr_units}.')

            # Handle layers
            if gas in self.lay:
                if self.lay[gas]['units'] in mmr_units:
                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Converting {gas} from mass mixing ratio to number density')

                    self.lay[gas]['data'] = self.lay[gas]['data'] * self.lay['factor']['data'] # apply factor
                    self.lay[gas]['units'] = 'cm^-3'

                elif self.lay[gas]['units'] in vmr_units:
                    if self.verbose:
                        print(f'Message [ARCSIXAtmModel]: Converting {gas} from volume mixing ratio to number density')

                    vmr_lay_data = self.lay[gas]['data'] / self.lay['air']['data'] # dimensionless vol. mixing ratio
                    self.lay[gas]['data'] = vmr_lay_data * self.lay['factor']['data'] # apply factor to go back to volume units
                    self.lay[gas]['units'] = 'cm^-3'

                else:
                    raise ValueError(f'Error [ARCSIXAtmModel]: Unrecognized units for {gas} in levels: {self.lay[gas]["units"]}.\nIf the gas is in mass mixing ratio units, ensure it is in {mmr_units}, or if it is in volume mixing ratio units, ensure it is in {vmr_units}.')


    def save_to_hdf5(self):
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

            # add metadata
            f.attrs['title'] = 'ARCSIX Atmospheric Model Profiles'
            f.attrs['description'] = 'ARCSIX Atmospheric Model Profiles'
            f.attrs['computer'] = os.uname()[1]
            f.attrs['software'] = f'EaR3T file {os.path.abspath(__file__)}, class {self.__class__.__name__}'
            f.attrs['created_on'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')# utc time

        if self.verbose:
            print(f'Message [ARCSIXAtmModel]: Saved to {self.fname_out}')


        if self.plot:
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
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))
    # arcsix_atm_mod = ARCSIXAtmModel(levels=levels, levels_source='user', config_file='er3t/pre/atm/arcsix_atm_profile_config.yaml', verbose=1, fname_out='data/test_data/arcsix_atm_profile_output.h5', plot=False)
    afgl_atm_mod = atm_atmmod(levels=levels, fname_atmmod="er3t/data/atmmod/afglss.dat", verbose=1, plot=True)
