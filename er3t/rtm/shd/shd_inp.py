import os
import sys
import warnings
from collections import OrderedDict
import numpy as np

from er3t.util import nice_array_str



__all__ = ['shd_inp_file']



def load_shd_inp_nml():

    # SHDOM initialization namelist
    #╭────────────────────────────────────────────────────────────────────────────╮#
    '''
    RUNNAME     label for the run (also for multiple processor log file names)
    PROPFILE    name of the input medium property file
    SFCFILE     name of the input surface property file (or NONE)
    CKDFILE     name of the input correlated k-distribution file (or NONE)
    INSAVEFILE  name of the input binary save file (or NONE)
    OUTSAVEFILE name of the output binary save file (or NONE)
    '''
    shdom_nml_init = OrderedDict([
             ('RUNNAME', 'SHDOM-run'),
             ('PROPFILE','shdom_prp_file.txt'),
             ('SFCFILE', 'NONE'),
             ('CKDFILE', 'NONE'),
             ('INSAVEFILE', 'stcu067a.bin'),
             ('OUTSAVEFILE', 'NONE'),
        ])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    shdom_nml_all = OrderedDict([
             ('_header', '$SHDOMINPUT'),
             ('RUNNAME', 'SHDOM-run'),
             ('PROPFILE','shdom_prp_file.txt'),
             ('SFCFILE', 'NONE'),
             ('CKDFILE', 'NONE'),
             ('INSAVEFILE', 'stcu067a.bin'),
             ('OUTSAVEFILE', 'NONE'),
             ('NSTOKES', 1),
             ('NX', 64),
             ('NY', 64),
             ('NZ', 22),
             ('NMU', 18),
             ('NPHI', 9),
             ('BCFLAG', 0),
             ('IPFLAG', 0),
             ('DELTAM', '.TRUE.'),
             ('GRIDTYPE', 'P'),
             ('SRCTYPE', 'S'),
             ('SOLARFLUX', 1.0),
             ('SOLARMU', 0.8),
             ('SOLARAZ', 0.0),
             ('SKYRAD', 0.0),
             ('GNDALBEDO', 0.6),
             ('UNITS', 'R'),
             ('GNDTEMP', 290),
             ('WAVENO', '500.0, 800.0'),
             ('WAVELEN', 0.67),
             ('ACCELFLAG', '.FALSE.'),
             ('SOLACC', 1.0E-5),
             ('MAXITER', 100),
             ('SPLITACC', 0.1),
             ('SHACC', 0.003),
             ('NUMOUT', 1),
             ('OUTTYPES(1)', 'R'),
             ('OUTPARMS(1,1)', '0.9, 0.055, 0.055, 0.0, 0.0, 1, 1.0, 0.00'),
             ('OUTFILES(1)', 'shdom-out.txt'),
             ('OutFileNC', 'NONE'),
             ('MAX_TOTAL_MB', 120.0),
             ('ADAPT_GRID_FACTOR', 2.2),
             ('NUM_SH_TERM_FACTOR', 0.6),
             ('CELL_TO_POINT_RATIO', 1.5),
             ('_footer', '$END'),
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

            ])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    return shdom_nml_all_info


def shd_inp_nml(input_dict, verbose=True, comment=True):

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
    #   2. If in array-like way, e.g., 'Atm_ext1d(1:, 1)' is similar to
    #      'Atm_ext1d', update dictionary with new variable and assign data
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
    #   'Wld_mverb': 'mcarWld_nml_init'
    #   'Atm_zgrd0': 'mcarAtm_nml_job'
    shdom_nml_input = OrderedDict(zip(nml_ordered_keys_full, nml_ordered_item_full))

    return shdom_nml_all, shdom_nml_all_info, shdom_nml_input


def shd_inp_file(input_fname, input_dict, verbose=True, comment=True):

    shdom_nml_all, shdom_nml_all_info, shdom_nml_input \
        = shd_inp_nml(input_dict, verbose=verbose, comment=comment)

    input_fname = os.path.abspath(input_fname)
    fdir_inp    = os.path.dirname(input_fname)
    if not os.path.exists(fdir_inp):
        os.system('mkdir -p %s' % fdir_inp)

    # creating input file for SHDOM
    f = open(input_fname, 'w')

    for nml_key in shdom_nml_all.keys():
        f.write('&%s\n' % nml_key)

        vars_key = [xx for xx in shdom_nml_input.keys() if shdom_nml_input[xx]==nml_key]
        for var_key in vars_key:
            var = shdom_nml_all[nml_key][var_key]
            if var is not None:

                if isinstance(var, (int, float, np.int32, np.int64, np.float32, np.float64)):
                    f.write(' %-15s = %-.16g\n' % (var_key, var))
                elif isinstance(var, str):
                    if '*' in var:
                        f.write(' %-15s = %s\n' % (var_key, var))
                    else:
                        f.write(' %-15s = \'%s\'\n' % (var_key, var))
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

        f.write('/\n')

    f.close()



if __name__ == '__main__':


    nml = load_shd_inp_nml()
    for key in nml.keys():
        print(key, nml[key])
    pass
