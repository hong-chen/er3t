import os
import sys
import numpy as np
from collections import OrderedDict as OD
import er3t.common



__all__ = ['get_lrt_cfg', 'get_cld_cfg', 'get_aer_cfg']



def get_lrt_cfg(
        lrt_fdir   = None,
        ssfr_fdir  = er3t.common.fdir_data_ssfr,
        spectral_resolution=0.1
        ):

    if lrt_fdir is None:
        if er3t.common.has_libradtran:
            lrt_fdir = os.environ['LIBRADTRAN_V2_DIR']
        else:
            sys.exit('Error   [er3t.rtm.lrt]: Cannot locate libRadtran. Please make sure libRadtran is installed and specified at enviroment variable <LIBRADTRAN_V2_DIR>.')

    lrt_cfg = {
            'executable_file'    : '%s/bin/uvspec' % lrt_fdir,
            'atmosphere_file'    : '%s/data/atmmod/afglus.dat' % lrt_fdir,
            'solar_file'         : '%s/data/solar_flux/kurudz_%.1fnm.dat' % (lrt_fdir, spectral_resolution),
            'data_files_path'    : '%s/data' % lrt_fdir,
            'rte_solver'         : 'disort',
            'number_of_streams'  : 32,
            'mol_abs_param'      : 'LOWTRAN', # use 'reptran fine' for higher resolution
            'slit_function_file_vis' : '%s/vis_%.1fnm_s.dat' % (ssfr_fdir, spectral_resolution),
            'slit_function_file_nir' : '%s/nir_%.1fnm_s.dat' % (ssfr_fdir, spectral_resolution)
            }

    return lrt_cfg



def get_cld_cfg():

    cld_cfg = {
            'cloud_file'                : 'LRT_cloud_profile.txt',
            'cloud_optical_thickness'   : 20.0,
            'cloud_effective_radius'    : 10.0,
            'liquid_water_content'      : 0.02,
            'cloud_type'                : 'water',  # or ice
            'wc_properties'             : 'mie',
            'cloud_altitude'            : np.arange(0.9, 1.31, 0.1)
            }

    return cld_cfg



def get_aer_cfg():

    aer_cfg = {
            'aerosol_file': 'LRT_aerosol_profile.txt',
            'aerosol_optical_depth'    : 0.2,
            'asymmetry_parameter'      : 0.0,
            'single_scattering_albedo' : 0.8,
            'aerosol_altitude'         : np.arange(3.0, 6.01, 0.2)
            }

    return aer_cfg



if __name__ == '__main__':

    pass
