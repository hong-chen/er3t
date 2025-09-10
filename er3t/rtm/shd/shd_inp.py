import os
import sys
import psutil
import warnings
import datetime
from collections import OrderedDict
import numpy as np

from er3t.util import nice_array_str



__all__ = ['shd_inp_file']



def load_shd_inp_nml():

    # SHDOM initialization namelist
    #╭────────────────────────────────────────────────────────────────────────────╮#
    '''
    RUNNAME    : label for the run (also for multiple processor log file names)

    PROPFILE   : name of the input medium property file

    SFCFILE    : name of the input surface property file (or NONE)

    CKDFILE    : name of the input correlated k-distribution file (or NONE)

    INSAVEFILE : name of the input binary save file (or NONE)

    OUTSAVEFILE: name of the output binary save file (or NONE)

    NSTOKES    : number of Stokes parameters
                   1 for I, (i.e. unpolarized);
                   3 for I, Q, U;
                   4 for I, Q, U, V

    NX, NY, NZ : base grid size in X, Y and Z
                   NX and NY are the number of grid points horizontally;
                     (for periodic boundaries there is actually an extra plane
                      of grid points on the horizontal boundaries)
                   NZ is the number grid cells vertically (>1);

    NMU, NPHI  : number of discrete ordinates covering -1<mu<1 and 0<phi<2π

    BCFLAG     : Bit flags to specify the horizontal boundary conditions:
                   0 for periodic in X & Y,
                   1 for open in X,
                   2 for open in Y,
                   3 for open in X & Y.

    IPFLAG     : Bit flags for independent pixel mode: 0 for 3D,
                   1 for independent (2D) scans in X,
                   2 for 2D scans in Y (X-Z planes),
                   3 for indepedent pixels (i.e. bit 0 for X and bit 1 for Y).
                 Bit 2 of IPFLAG means do the direct beam in 3D, e.g. IPFLAG=7
                   means 3D direct beam but IP diffuse radiative transfer.

    DELTAM     : T (true) for delta-M scaling of medium and Nakajima and Tanaka
                   method of computing radiances

    GRIDTYPE   : E for even Z base grid between bottom and top,
                 P for Z base grid levels from property file,
                 F for Z base grid levels from file: <zgrid.inp>.
    '''

    shdom_nml_init = OrderedDict([
             ('_header', '$SHDOMINPUT'),

             ('RUNNAME', 'shdom-run_%s' % datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
             ('PROPFILE','shdom-prp.txt'),
             ('SFCFILE', 'NONE'),
             ('CKDFILE', 'NONE'),
             ('INSAVEFILE', 'NONE'),
             ('OUTSAVEFILE', 'NONE'),

             ('NSTOKES', 1),
             ('NX', 100),
             ('NY', 100),
             ('NZ', 20),
             ('NMU', 9),
             ('NPHI', 18),

             ('BCFLAG', 0),
             ('IPFLAG', 0),
             ('DELTAM', '.TRUE.'),
             ('GRIDTYPE', 'P'),
        ])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # SHDOM radiation namelist
    #╭────────────────────────────────────────────────────────────────────────────╮#
    """
    UNITS      : 'R' for radiance units (W/m^2 ster),
                 'T' for brightness temperature (Rayleigh-Jeans assumed)

    SRCTYPE    : 'S' for solar source, 'T' for thermal source, 'B' for both

    SOLARFLUX  : top of medium solar flux on a horizontal surface (any units)
                   For k-distribution this is a multiplier for the solar flux
                   in the CKD file (i.e. normally should be 1).

    SOLARMU    : cosine of the solar zenith angle (this represents the
                   direction of travel of the solar beam, so is forced
                   to be negative although it can be specified positive).

    SOLARAZ    : solar beam azimuthal angle; specified in degrees but
                   immediately converted to radians for use in code.
                 0 is beam going in positive X direction, 90 is positive Y.

    SKYRAD     : isotropic diffuse radiance incident from above

    GNDALBEDO  : bottom surface Lambertian albedo

    GNDTEMP    : ground temperature in Kelvin

    WAVELEN    : wavelength in microns for 'R' units;
                   WAVELEN not needed for solar sources (except for Ocean surface)
                   (GNDTEMP and WAVELEN used for Both source)

    WAVENO(2)  : wavenumber range (cm^-1) for correlated k-distribution.
                   This particular range must be in the CKD file.
                   If KDIST then WAVELEN set to 10000/(average wavenumber),
                   and UNITS='B' for band.
    """

    shdom_nml_rad = OrderedDict([
             ('UNITS', 'R'),
             ('SRCTYPE', 'S'),
             ('SOLARFLUX', 1.0),
             ('SOLARMU', 1.0),
             ('SOLARAZ', 0.0),
             ('SKYRAD', 0.0),
             ('GNDALBEDO', 0.0),
             ('GNDTEMP', 288),
             ('WAVELEN', 0.67),
             ('WAVENO', '500.0, 800.0'),
        ])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # shdom output namelist
    #╭────────────────────────────────────────────────────────────────────────────╮#
    shdom_nml_out = OrderedDict([
             ('NUMOUT', 1),
             ('OUTTYPES(1)', 'R'),
             ('OUTPARMS(1,1)', '0.9, 0.055, 0.055, 0.0, 0.0, 1, 1.0, 0.00'),
             ('OUTFILES(1)', 'shdom-out.txt'),
             ('SENFILE', 'NONE'),
             ('OutFileNC', 'NONE'),
        ])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # shdom parameter namelist
    #╭────────────────────────────────────────────────────────────────────────────╮#
    """
    ACCELFLAG : T (true) to do the sequence acceleration. An acceleration
                  extrapolation of the source function is done every other
                  iteration.

    SOLACC    : solution accuracy - tolerance for solution criterion

    MAXITER   : maximum number of iterations allowed

    SPLITACC  : cell splitting accuracy; grid cells that have the adaptive
                  splitting criterion above this value are split.
                This is an absolute measure, but cannot be easily associated
                  with the resulting radiometric accuracy.
                Set to zero or negative for no adaptive cell splitting.

    SHACC     : adaptive spherical harmonic truncation accuracy; the
                  spherical harmonic source function series is truncated
                  after the terms are below this level. Truncation can still
                  happens if SHACC=0 (for 0 source terms).  This is also
                  an absolute measure, and is approximately the level of accuracy.

    MAX_TOTAL_MB        : approximate maximum memory to use (MB for 4 byte reals)

    ADAPT_GRID_FACTOR   : ratio of total grid points to base grid points

    NUM_SH_TERM_FACTOR  : ratio of average number of spherical harmonic terms
                           to total possible (NLM)

    CELL_TO_POINT_RATIO : ratio of number of grid cells to grid points
    """
    shdom_nml_param = OrderedDict([
             ('ACCELFLAG', '.TRUE.'),
             ('SOLACC', 1.0E-5),
             ('MAXITER', 100),
             ('SPLITACC', 0.1),
             ('SHACC', 0.003),
             ('MAX_TOTAL_MB', psutil.virtual_memory().total / 1024.0**2.0 / 2.0),
             ('ADAPT_GRID_FACTOR', 2.2),
             ('NUM_SH_TERM_FACTOR', 0.6),
             ('CELL_TO_POINT_RATIO', 1.5),

             ('_footer', '$END'),
            ])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    shdom_nml_all = OrderedDict([
                   ('shdom_nml_init' , shdom_nml_init),
                   ('shdom_nml_rad'  , shdom_nml_rad),
                   ('shdom_nml_out'  , shdom_nml_out),
                   ('shdom_nml_param', shdom_nml_param),
        ])

    return shdom_nml_all


def load_shd_inp_nml_info():

    # SHDOM input parameter descriptions
    #╭────────────────────────────────────────────────────────────────────────────╮#
    shdom_nml_all_info = OrderedDict([
            ('RUNNAME', 'label for the run (also for multiple processor log file names)'),
            ('PROPFILE', 'name of the input medium property file'),
            ('SFCFILE', 'name of the input surface property file (or NONE)'),
            ('CKDFILE', 'name of the input correlated k-distribution file (or NONE)'),
            ('INSAVEFILE' , 'name of the input binary save file (or NONE)'),
            ('OUTSAVEFILE', 'name of the output binary save file (or NONE)'),

            ('NSTOKES', 'number of Stokes parameters (1 for I, i.e. unpolarized;\n\
                                                      3 for I, Q, U;\n\
                                                   or 4 for I, Q, U, V)'),
            ('NX', 'base grid size in X; NX is the number of grid points horizontally;\n\
                    for periodic boundaries there is actually an extra plane'),
            ('NY', 'base grid size in Y; NY is the number of grid points horizontally;\n\
                    for periodic boundaries there is actually an extra plane'),
            ('NZ', 'base grid size in Z; NZ is one more than the number grid cells vertically'),
            ('NMU', 'number of discrete ordinates covering -1<mu<1'),
            ('NPHI','number of discrete ordinates covering 0<phi<2pi'),
            ('BCFLAG', 'Bit flags to specify the horizontal boundary conditions:\n\
                          0 for periodic in X & Y,\n\
                          1 for open in X,\n\
                          2 for open in Y,\n\
                          3 for open in X & Y.'),
            ('IPFLAG',''),
            ('DELTAM',''),
            ('GRIDTYPE',''),

            ('UNITS',''),
            ('SRCTYPE',''),
            ('SOLARFLUX',''),
            ('SOLARMU',''),
            ('SOLARAZ',''),
            ('SKYRAD',''),
            ('GNDALBEDO',''),
            ('GNDTEMP',''),
            ('WAVELEN',''),
            ('WAVENO',''),

            ('ACCELFLAG',''),
            ('SOLACC',''),
            ('MAXITER',''),
            ('SPLITACC',''),
            ('SHACC',''),
            ('MAX_TOTAL_MB',''),
            ('ADAPT_GRID_FACTOR',''),
            ('NUM_SH_TERM_FACTOR',''),
            ('CELL_TO_POINT_RATIO',''),

            ('NUMOUT',''),
            ('OutFileNC',''),
            ])
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return shdom_nml_all_info


def shd_inp_nml(input_dict, verbose=True, comment=False):

    shdom_nml_all      = load_shd_inp_nml()
    shdom_nml_all_info = load_shd_inp_nml_info()

    nml_ordered_keys_full = []
    nml_ordered_item_full = []
    for nml_key in shdom_nml_all.keys():
        for var_key in shdom_nml_all[nml_key].keys():
            nml_ordered_keys_full.append(var_key)
            nml_ordered_item_full.append(nml_key)

    # Check input variables
    #
    # If input variable in the full dictionary keys, assign data to
    # the input variable
    #
    # If not
    #   1. If is a typo in input variables, exit with error message
    #   2. If in array-like way, e.g., 'OUTPARMS(1,1)' is similar to
    #      'OUTPARMS', update dictionary with new variable and assign data
    #      to the updated variable.
    #
    # `shdom_nml_all` and `shdom_nml_all_info` will be updated in the loop
    for key in input_dict.keys():
        if key not in nml_ordered_keys_full:
            if '(' in key and ')' in key:
                ee          = key.index('(')
                key_ori     = key[:ee]
                index_ori   = nml_ordered_keys_full.index(key_ori)

                key_more = [xx for xx in nml_ordered_keys_full if key_ori in xx and '(' in xx]

                if len(key_more) >= 1:
                    key0     = key_more[-1]
                    index    = nml_ordered_keys_full.index(key0)
                else:
                    index = index_ori

                nml_ordered_keys_full.insert(index+1, key)
                nml_ordered_item_full.insert(index+1, nml_ordered_item_full[index])
                shdom_nml_all[nml_ordered_item_full[index]][key] = input_dict[key]
                if comment:
                    shdom_nml_all_info[key] = shdom_nml_all_info[key_ori]
            else:
                msg = 'Error [shd_inp_nml]: please check input variable <%s>.' % key
                raise OSError(msg)
        else:
            index   = nml_ordered_keys_full.index(key)
            shdom_nml_all[nml_ordered_item_full[index]][key] = input_dict[key]

    # create a full dictionary to link variable back to namelist section
    # examples:
    #   'RUNNAME': 'shdom_nml_init'
    shdom_nml_input = OrderedDict(zip(nml_ordered_keys_full, nml_ordered_item_full))

    return shdom_nml_all, shdom_nml_all_info, shdom_nml_input


def shd_inp_file(input_fname, input_dict, verbose=True, comment=False):

    shdom_nml_all, shdom_nml_all_info, shdom_nml_input \
        = shd_inp_nml(input_dict, verbose=verbose, comment=comment)

    input_fname = os.path.abspath(input_fname)
    fdir_inp    = os.path.dirname(input_fname)
    if not os.path.exists(fdir_inp):
        os.system('mkdir -p %s' % fdir_inp)

    # creating input file for SHDOM
    f = open(input_fname, 'w')

    for nml_key in shdom_nml_all.keys():

        vars_key = [xx for xx in shdom_nml_input.keys() if shdom_nml_input[xx]==nml_key]
        for var_key in vars_key:

            var = shdom_nml_all[nml_key][var_key]

            if (var_key[0] != '_'):

                if isinstance(var, str):

                    if '*' in var or (var_key in ['DELTAM', 'WAVENO', 'OUTPARMS(1,1)', 'ACCELFLAG']):
                        f.write(' %-15s = %s\n' % (var_key, var))
                    else:
                        f.write(' %-15s = \'%s\'\n' % (var_key, var))

                elif isinstance(var, (int, float, np.int32, np.int64, np.float32, np.float64)):

                    f.write(' %-15s = %-.16g\n' % (var_key, var))


                elif isinstance(var, np.ndarray):

                    if var.size > 1:

                        var_str = nice_array_str(var)

                        if len(var_str) <= 80:
                            f.write(' %-15s = %s\n' % (var_key, var_str))
                        else:
                            f.write(' %-15s =\n' % var_key)
                            f.write('%s\n' % var_str)

                    elif var.size == 1:
                        f.write(' %-15s = %-g\n' % (var_key, var))

                else:
                    msg = 'Error [shd_inp_file]: only types of int, float, str, ndarray are supported (do not support <%s> as %s).' % (var_key, type(var))
                    raise ValueError(msg)

                if comment:
                    var_detail = shdom_nml_all_info[var_key]
                    f.write(' !------------------------------- This is a comment for above parameter -------------------------------------\n')
                    if '\n' in var_detail:
                        lines = var_detail.split('\n')
                        for line in lines:
                            f.write(' !----> %s\n' % line)
                    else:
                        f.write(' !----> %s\n' % shdom_nml_all_info[var_key])
                    f.write(' !-----------------------------------------------------------------------------------------------------------\n')
                    f.write('\n')

            else:
                f.write(' %s\n' % (var))

    f.close()


if __name__ == '__main__':

    pass
