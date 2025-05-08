import os
import sys
import glob
import datetime
import warnings
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import h5py
import numpy as np
from scipy import interpolate

import er3t.common

from er3t.util import cal_sol_fac, h5dset_to_pydict


__all__ = ['get_shd_data_out', 'get_shd_data_out_ori', 'shd_out_ng', 'shd_out_raw']


def get_shd_data_out_ori(
        fname,
        verbose=False,
        ):

    if verbose:
        print('Message [get_shd_data_out]: Reading SHDOM output ...')
        print('╭────────────────────────────────────────────────────────────────────────────╮')

    headers = []
    with open(fname, 'r') as f:
        line = f.readline().strip()
        while line[0] == '!':
            headers.append(line)
            line = f.readline().strip()

    Nline_output_type = [i for i in range(len(headers)) if ('OUTPUT_TYPE=' in headers[i])][0]
    output_type = headers[Nline_output_type].split('=')[-1].lower().strip()

    # Nvar
    #╭────────────────────────────────────────────────────────────────────────────╮#
    line_xy = [line for line in headers[Nline_output_type:] if ('  X  ' in line) or ('  Y  ' in line) or (' Z ' in line)][0]
    Nvar_s = 0
    if '  X  ' in line_xy:
        Nvar_s += 1
    if '  Y  ' in line_xy:
        Nvar_s += 1
    if ' Z ' in line_xy:
        Nvar_s += 1

    data_ = np.loadtxt(fname, comments='!')[:, Nvar_s:]
    Nvar = data_.shape[-1]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # Nx, Ny, Nz
    #╭────────────────────────────────────────────────────────────────────────────╮#
    line_xy = [line for line in headers[Nline_output_type:] if ('NXO=' in line) or ('NYO=' in line)]
    if len(line_xy) == 1:
        Nz = 1
        Nx = int(line_xy[0][1:].split('NXO=')[1].split()[0])
        Ny = int(line_xy[0][1:].split('NYO=')[1].split()[0])
        if 'NDIR' in line_xy[0]:
            Nset = int(line_xy[0][1:].split('NDIR=')[1].split()[0])
        else:
            Nset = 1
    elif len(line_xy) == 0:
        Nx = int(headers[2][1:].split('NX=')[1].split()[0])
        Ny = int(headers[2][1:].split('NY=')[1].split()[0])
        Nz = data_.shape[0]//Nx//Ny
        Nset = 1
    #╰────────────────────────────────────────────────────────────────────────────╯#

    data = np.zeros((Nset, Ny, Nx, Nz, Nvar), dtype=np.float32)
    for i in range(Nvar):
        data[..., i] = data_[:, i].reshape((Nset, Ny, Nx, Nz))

    # from (Nset, Ny, Nx, Nz, Nvar) to (Ny, Nx, Nz, Nset, Nvar)
    data = np.moveaxis(data, 0, -2)
    # from (Ny, Nx, Nz, Nset, Nvar) to (Nx, Ny, Nz, Nset, Nvar)
    data = np.moveaxis(data, 0, 1)

    if verbose:
        print('target file: <%s>' % os.path.abspath(fname))
        print('%s (Nx, Ny, Nz, Nset, Nvar): %s' % (output_type.title(), data.shape))
        print('╰────────────────────────────────────────────────────────────────────────────╯')

    return data


def get_shd_data_out(
        fname,
        verbose=False,
        ):

    if verbose:
        print('Message [get_shd_data_out]: Reading SHDOM output ...')
        print('╭────────────────────────────────────────────────────────────────────────────╮')


    # read headers
    #╭────────────────────────────────────────────────────────────────────────────╮#
    headers = []
    with open(fname, 'r') as f:
        line = f.readline().strip()
        while (line) and (line[0]=='!'):
            headers.append(line)
            line = f.readline().strip()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # extract information
    #╭────────────────────────────────────────────────────────────────────────────╮#
    Nline_output_type = [i for i in range(len(headers)) if ('OUTPUT_TYPE=' in headers[i])][0]
    output_type = headers[Nline_output_type].split('=')[-1].lower().strip()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # Nx, Ny
    #╭────────────────────────────────────────────────────────────────────────────╮#
    line_xy = [line for line in headers[Nline_output_type:] if ('NXO=' in line) or ('NYO=' in line)]
    if len(line_xy) == 1:
        Nx = int(line_xy[0][1:].split('NXO=')[1].split()[0])
        Ny = int(line_xy[0][1:].split('NYO=')[1].split()[0])
    elif len(line_xy) == 0:
        Nx = int(headers[2][1:].split('NX=')[1].split()[0])
        Ny = int(headers[2][1:].split('NY=')[1].split()[0])
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # output data variable
    #╭────────────────────────────────────────────────────────────────────────────╮#
    mode = headers[-3].replace('!', '').strip().upper()
    if mode in ['R', 'F1', 'F2', 'F3', 'F4', 'F5', 'H1', 'H2', 'H3', 'S1', 'S2', 'J1', 'J2', 'M1', 'M2']:
        binary = True
    else:
        binary = False
    #╰────────────────────────────────────────────────────────────────────────────╯#


    if binary:

        fname_data = fname + headers[-2].replace('!', '').strip()

        if mode in ['F5', 'H3', 'S2', 'J2', 'M2']:

            Ndata, Nvar = [int(num_s.strip()) for num_s in headers[-1].replace('!', '').split(',')]
            shape = (Ndata, Nvar)

            data = np.fromfile(fname_data, dtype='<f4').reshape(shape, order='F')

        else:

            Nx_, Ny_, Nz, Nset, Nvar = [int(num_s.strip()) for num_s in headers[-1].replace('!', '').split(',')]

            if mode in ['R', 'F2']:

                shape = (Nz, Nx_, Ny_, Nset, Nvar)
                axes_swap = [2, 0, 1, 3, 4]

            else:

                shape = (Nz, Ny_, Nx_, Nset, Nvar)
                axes_swap = [2, 1, 0, 3, 4]

            data_ = np.fromfile(fname_data, dtype='<f4').reshape(shape, order='F')

            data = np.moveaxis(data_, [0, 1, 2, 3, 4], axes_swap)
            data = data[:Nx, :Ny, :, :, :]

    else:

        # Nvar
        #╭────────────────────────────────────────────────────────────────────────────╮#
        line_xy = [line for line in headers[Nline_output_type:] if ('  X  ' in line) or ('  Y  ' in line) or (' Z ' in line)][0]
        Nvar_s = 0
        if '  X  ' in line_xy:
            Nvar_s += 1
        if '  Y  ' in line_xy:
            Nvar_s += 1
        if ' Z ' in line_xy:
            Nvar_s += 1

        data_ = np.loadtxt(fname, comments='!')[:, Nvar_s:]
        Nvar = data_.shape[-1]
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Nx, Ny, Nz
        #╭────────────────────────────────────────────────────────────────────────────╮#
        line_xy = [line for line in headers[Nline_output_type:] if ('NXO=' in line) or ('NYO=' in line)]
        if len(line_xy) == 1:
            Nz = 1
            Nx = int(line_xy[0][1:].split('NXO=')[1].split()[0])
            Ny = int(line_xy[0][1:].split('NYO=')[1].split()[0])
            if 'NDIR' in line_xy[0]:
                Nset = int(line_xy[0][1:].split('NDIR=')[1].split()[0])
            else:
                Nset = 1
        elif len(line_xy) == 0:
            Nx = int(headers[2][1:].split('NX=')[1].split()[0])
            Ny = int(headers[2][1:].split('NY=')[1].split()[0])
            Nz = data_.shape[0]//Nx//Ny
            Nset = 1
        #╰────────────────────────────────────────────────────────────────────────────╯#

        fname_data = fname

        data = np.zeros((Nset, Ny, Nx, Nz, Nvar), dtype=np.float32)
        for i in range(Nvar):
            data[..., i] = data_[:, i].reshape((Nset, Ny, Nx, Nz))

        # from (Nset, Ny, Nx, Nz, Nvar) to (Ny, Nx, Nz, Nset, Nvar)
        data = np.moveaxis(data, 0, -2)

        # from (Ny, Nx, Nz, Nset, Nvar) to (Nx, Ny, Nz, Nset, Nvar)
        data = np.moveaxis(data, 0, 1)

    if verbose:
        print('target file: <%s>' % os.path.abspath(fname))
        print('  data file: <%s>' % os.path.abspath(fname_data))
        if data.ndim > 2:
            print('%s (Nx, Ny, Nz, Nset, Nvar): %s' % (output_type.title(), data.shape))
        else:
            print('%s (Ndata, Nvar): %s' % (output_type.title(), data.shape))
        print('╰────────────────────────────────────────────────────────────────────────────╯')

    return data


class shd_out_ng:

    """
    Read SHDOM (Ng) output files (flux)

    Input:
        fname=    : HDF5 file path of the many SHDOM binary files to be written/read, string, default=None
        shd_obj=  : shdom object, e.g., shd_obj = shdom(...), default=None
        abs_obj=  : abs object, e.g., abs_obj = abs_16g(...), default=None
        mode=     : mode, string, can be 'mean', 'std', 'all', default='mean'
        overwrite=: whether to overwrite the HDF5 file specifed by fname, default=False
        squeeze=  : whether to remove axis where dimension=1, default=True
        quiet=    : quiet flag, default=False
        verbose=  : verbose flag, default=False

    Output:
        self.data
            ['f_up']
            ['f_down']
            ['f_down_direct']
            ['f_down_diffuse']

    Note:
        If fname is specified and exists, read the data from fname;
        If fname is specified but does not exist while shd_obj and abs_obj are specified, create file first and then read the data from the file;
        If fname is not specified, shd_obj and abs_obj are specified, read data directly from SHDOM binary files
    """


    def __init__(self, \
                 fname     = None,  \
                 shd_obj   = None,  \
                 abs_obj   = None,  \
                 mode      = 'mean',\
                 overwrite = False, \
                 squeeze   = True,  \
                 quiet     = False, \
                 verbose   = False
                 ):


        self.mode      = mode
        self.quiet     = quiet
        self.verbose   = verbose
        self.overwrite = overwrite
        self.squeeze   = squeeze


        self.fname     = fname
        self.shd       = shd_obj
        self.abs       = abs_obj


        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load()

        elif (((shd_obj is not None) and (abs_obj is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((shd_obj is not None) and (abs_obj is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run()
            self.dump()

        elif (((shd_obj is not None) and (abs_obj is not None)) and (fname is None)):

            self.run()

        else:

            msg = 'Error [shd_out_ng]: Please provide both <shd_obj> and <abs_obj> to proceed.'
            raise OSError(msg)


    def load(self):

        if self.verbose:
            print('Message [shd_out_ng]: Reading <%s> from <%s> ...' % (self.shd.target.lower(), self.fname))

        self.data = {}

        f = h5py.File(self.fname, 'r')
        g = f[self.mode]

        for key in g.keys():

            self.data[key] = h5dset_to_pydict(g[key])

        f.close()


    def run(self):

        if self.verbose:
            print('Message [shd_out_ng]: Reading <%s> ...' % self.shd.target.lower())

        if self.shd.target in ['flux', 'flux0']: # ['f', 'flux', 'irradiance', 'heating rate', 'hr']:
            self.data = read_flux_shd_out(self.shd, self.abs, squeeze=self.squeeze)

        elif self.shd.target == 'radiance':
            self.data = read_radiance_shd_out(self.shd, self.abs, squeeze=self.squeeze)


    def dump(self):

        if not self.quiet:
            print('Message [shd_out_ng]: Saving <%s> into <%s> ...' % (self.shd.target.lower(), self.fname))

        mode = self.mode.lower()

        f = h5py.File(self.fname, 'w')

        g = f.create_group(mode)
        for key in self.data.keys():

            if isinstance(self.data[key]['data'], np.ndarray):
                g.create_dataset(key, data=self.data[key]['data'], compression='gzip', compression_opts=9, chunks=True)
            else:
                g[key] = self.data[key]['data']

            for key0 in self.data[key].keys():
                if key0 != 'data':
                    if key0 == 'dims_info':
                        g[key].attrs[key0]  = np.string_(self.data[key][key0])
                    else:
                        g[key].attrs[key0]  = self.data[key][key0]

        f.close()




class shd_out_raw:

    """
    Read a single SHDOM output binary file based on the information provided in the .ctl file

    Input:
        fname_txt: positional argument, string type, file path of the shdom binary file

    Output:
        self.data: Python list
    """


    def __init__(self, fname_txt, verbose=False):
        self.verbose = verbose

        if not os.path.isfile(fname_txt):
            msg = 'Error [shd_out_raw]: Cannot find <%s>.' % fname_txt
            raise OSError(msg)

        self.fname_txt = fname_txt

        self.data = []
        self.read_txt()


    def read_txt(self):

        if self.verbose:
            print('Message [get_shd_data_out]: Reading SHDOM output ...')
            print('╭────────────────────────────────────────────────────────────────────────────╮')


        # read headers
        #╭────────────────────────────────────────────────────────────────────────────╮#
        headers = []
        with open(self.fname_txt, 'r') as f:
            line = f.readline().strip()
            while (line) and (line[0]=='!'):
                headers.append(line)
                line = f.readline().strip()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # extract information
        #╭────────────────────────────────────────────────────────────────────────────╮#
        Nline_output_type = [i for i in range(len(headers)) if ('OUTPUT_TYPE=' in headers[i])][0]
        output_type = headers[Nline_output_type].split('=')[-1].lower().strip()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Nx, Ny
        #╭────────────────────────────────────────────────────────────────────────────╮#
        line_xy = [line for line in headers[Nline_output_type:] if ('NXO=' in line) or ('NYO=' in line)]
        if len(line_xy) == 1:
            Nx = int(line_xy[0][1:].split('NXO=')[1].split()[0])
            Ny = int(line_xy[0][1:].split('NYO=')[1].split()[0])
        elif len(line_xy) == 0:
            Nx = int(headers[2][1:].split('NX=')[1].split()[0])
            Ny = int(headers[2][1:].split('NY=')[1].split()[0])
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # output data variable
        #╭────────────────────────────────────────────────────────────────────────────╮#
        mode = headers[-3].replace('!', '').strip().upper()
        if mode in ['R', 'F1', 'F2', 'F3', 'F4', 'F5', 'H1', 'H2', 'H3', 'S1', 'S2', 'J1', 'J2', 'M1', 'M2']:
            binary = True
        else:
            binary = False
        #╰────────────────────────────────────────────────────────────────────────────╯#


        if binary:

            fname_data = self.fname_txt + headers[-2].replace('!', '').strip()

            if mode in ['F5', 'H3', 'S2', 'J2', 'M2']:

                Ndata, Nvar = [int(num_s.strip()) for num_s in headers[-1].replace('!', '').split(',')]
                shape = (Ndata, Nvar)

                data = np.fromfile(fname_data, dtype='<f4').reshape(shape, order='F')

            else:

                Nx_, Ny_, Nz, Nset, Nvar = [int(num_s.strip()) for num_s in headers[-1].replace('!', '').split(',')]

                if mode in ['R', 'F2']:

                    shape = (Nz, Nx_, Ny_, Nset, Nvar)
                    axes_swap = [2, 0, 1, 3, 4]

                else:

                    shape = (Nz, Ny_, Nx_, Nset, Nvar)
                    axes_swap = [2, 1, 0, 3, 4]

                data_ = np.fromfile(fname_data, dtype='<f4').reshape(shape, order='F')

                data = np.moveaxis(data_, [0, 1, 2, 3, 4], axes_swap)
                data = data[:Nx, :Ny, :, :, :]

        else:

            # Nvar
            #╭────────────────────────────────────────────────────────────────────────────╮#
            line_xy = [line for line in headers[Nline_output_type:] if ('  X  ' in line) or ('  Y  ' in line) or (' Z ' in line)][0]
            Nvar_s = 0
            if '  X  ' in line_xy:
                Nvar_s += 1
            if '  Y  ' in line_xy:
                Nvar_s += 1
            if ' Z ' in line_xy:
                Nvar_s += 1

            data_ = np.loadtxt(self.fname_txt, comments='!')[:, Nvar_s:]
            Nvar = data_.shape[-1]
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # Nx, Ny, Nz
            #╭────────────────────────────────────────────────────────────────────────────╮#
            line_xy = [line for line in headers[Nline_output_type:] if ('NXO=' in line) or ('NYO=' in line)]
            if len(line_xy) == 1:
                Nz = 1
                Nx = int(line_xy[0][1:].split('NXO=')[1].split()[0])
                Ny = int(line_xy[0][1:].split('NYO=')[1].split()[0])
                if 'NDIR' in line_xy[0]:
                    Nset = int(line_xy[0][1:].split('NDIR=')[1].split()[0])
                else:
                    Nset = 1
            elif len(line_xy) == 0:
                Nx = int(headers[2][1:].split('NX=')[1].split()[0])
                Ny = int(headers[2][1:].split('NY=')[1].split()[0])
                Nz = data_.shape[0]//Nx//Ny
                Nset = 1
            #╰────────────────────────────────────────────────────────────────────────────╯#

            fname_data = self.fname_txt

            data = np.zeros((Nset, Ny, Nx, Nz, Nvar), dtype=np.float32)
            for i in range(Nvar):
                data[..., i] = data_[:, i].reshape((Nset, Ny, Nx, Nz))

            # from (Nset, Ny, Nx, Nz, Nvar) to (Ny, Nx, Nz, Nset, Nvar)
            data = np.moveaxis(data, 0, -2)

            # from (Ny, Nx, Nz, Nset, Nvar) to (Nx, Ny, Nz, Nset, Nvar)
            data = np.moveaxis(data, 0, 1)

        if self.verbose:
            print('target file: <%s>' % os.path.abspath(self.fname_txt))
            print('  data file: <%s>' % os.path.abspath(fname_data))
            if data.ndim > 2:
                print('%s (Nx, Ny, Nz, Nset, Nvar): %s' % (output_type.title(), data.shape))
            else:
                print('%s (Ndata, Nvar): %s' % (output_type.title(), data.shape))
            print('╰────────────────────────────────────────────────────────────────────────────╯')

        self.data = [{} for i in range(Nvar)]
        for i in range(Nvar):
            self.data[i]['name']      = 'var_%02d' % (i)
            self.data[i]['dims']      = [Ny, Nx, Nz, Nset]
            self.data[i]['dims_info'] = ['Nx', 'Ny', 'Nz', 'Nset']
            self.data[i]['data']      = data[..., i]

def save_h5_shd_out(fname, shd_obj, abs_obj, mode='mean', squeeze=True):

    """
    Save fluxes from SHDOM output files into a HDF5 file

    Input:
        fname   : positional argument, string type, file path of HDF5 file
        shd_obj : positional argument, shd object, e.g., shd_obj = shdom(...)
        abs_obj : positional argument, abs object, e.g., abs_obj = abs_16g(...)
        mode=   : keyword argument, string, default='mean', can be 'mean', 'all'
        squeeze=: keyword argument, boolen, default=True, whether to keep axis that has dimension of 1 or not, True:remove, False:keep

    Output:
        HDF5 file named fname: HDF5 file
        f[/group/dataset]
            group can be 'mean', 'std', 'raw', while dataset contains:
                'f_up'            : Global upwelling irradiance
                'f_down'          : Global downwelling irradiance
                'f_down_direct'   : Direct downwelling irradiance
                'f_down_diffuse'  : Diffuse downwelling irradiance
    """

    mode = mode.lower()

    f = h5py.File(fname, 'w')

    if shd_obj.target == 'flux':
        data = read_flux_shd_out(shd_obj, abs_obj, mode=mode, squeeze=squeeze)
    elif shd_obj.target == 'radiance':
        data = read_radiance_shd_out(shd_obj, abs_obj, mode=mode, squeeze=squeeze)

    g = f.create_group(mode)
    for key in data.keys():
        g[key] = data[key]['data']
        for key0 in data[key].keys():
            if key0 != 'data':
                if key0 == 'dims_info':
                    g[key].attrs[key0]  = np.string_(data[key][key0])
                else:
                    g[key].attrs[key0]  = data[key][key0]


    f.close()

def read_flux_shd_out(shd_obj, abs_obj, squeeze=True):

    """
    Read fluxes from SHDOM output files
    Input:
        shd_obj : positional argument, shd object, e.g., shd_obj = shdom(...)
        abs_obj : positional argument, abs object, e.g., abs_obj = abs_16g(...)
        squeeze=: keyword argument, boolen, default=True, whether to keep axis that has dimension of 1 or not, True:remove, False:keep

    Output:
        data: Python dictionary
            data['f_up']            : Global upwelling irradiance (mean/std/raw)
            data['f_down']          : Global downwelling irradiance (mean/std/raw)
            data['f_down_direct']   : Direct downwelling irradiance (mean/std/raw)
            data['f_down_diffuse']  : Diffuse downwelling irradiance (mean/std/raw)
    """

    # +
    # read one file to get the information of dataset dimension
    out0      = shd_out_raw(shd_obj.fnames_out[0], verbose=False)
    dims_info = out0.data[0]['dims_info']
    dims      = out0.data[0]['dims']
    # -


    # +
    # calculate factors using sol_fac (sun-earth distance, weight, slit functions etc.)
    Nz = dims[dims_info.index('Nz')]

    zz     = np.arange(Nz)
    if Nz > 1:
        zz[-1] = zz[-2]

    sol_fac = cal_sol_fac(shd_obj.date)

    norm    = np.zeros(Nz, dtype=np.float32)
    factors = np.zeros((Nz, shd_obj.Ng), dtype=np.float32)

    factors[...] = 1.0
    # for iz in range(Nz):
    #     norm[iz] = sol_fac/(abs_obj.coef['weight']['data'] * abs_obj.coef['slit_func']['data'][zz[iz], :]).sum()
    #     for ig in range(shd_obj.Ng):
    #         factors[iz, ig] = norm[iz]*abs_obj.coef['solar']['data'][ig]*abs_obj.coef['weight']['data'][ig]*abs_obj.coef['slit_func']['data'][zz[iz], ig]
    # -


    # +
    # calculate fluxes
    if squeeze:
        dims_info  = [dims_info[i] for i in range(len(dims)) if dims[i] > 1]
        dims       = [i for i in dims if i>1]

    dims_info += ['Nr']

    f_down_direct = np.zeros(dims, dtype=np.float32)
    f_down        = np.zeros(dims, dtype=np.float32)
    f_up          = np.zeros(dims, dtype=np.float32)

    for ig in range(shd_obj.Ng):

        fname0         = shd_obj.fnames_out[ig]

        out0           = shd_out_raw(fname0, verbose=False)
        f_down_direct0 = out0.data[0]['data']
        f_down0        = out0.data[1]['data']
        f_up0          = out0.data[2]['data']

        for iz in range(Nz):
            f_down_direct0[:, :, iz, :] *= factors[iz, ig]
            f_down0[:, :, iz, :]        *= factors[iz, ig]
            f_up0[:, :, iz, :]          *= factors[iz, ig]

        if squeeze:
            f_down_direct[...] += np.squeeze(f_down_direct0)
            f_down[...]        += np.squeeze(f_down0)
            f_up[...]          += np.squeeze(f_up0)
        else:
            f_down_direct[...] += f_down_direct0
            f_down[...]        += f_down0
            f_up[...]          += f_up0
    # -


    # +
    # store data into Python dictionary
    data_dict = {}

    toa = np.sum(sol_fac * abs_obj.coef['solar']['data'] * abs_obj.coef['weight']['data'])
    data_dict['toa'] = {'data':toa, 'name':'TOA without SZA' , 'units':'W/m^2/nm'}

    data_dict['f_down']          = {'data':f_down              , 'name':'Global downwelling flux' , 'units':'W/m^2/nm', 'dims_info':dims_info}
    data_dict['f_up']            = {'data':f_up                , 'name':'Global upwelling flux'   , 'units':'W/m^2/nm', 'dims_info':dims_info}
    data_dict['f_down_direct']   = {'data':f_down_direct       , 'name':'Direct downwelling flux' , 'units':'W/m^2/nm', 'dims_info':dims_info}
    data_dict['f_down_diffuse']  = {'data':f_down-f_down_direct, 'name':'Diffuse downwelling flux', 'units':'W/m^2/nm', 'dims_info':dims_info}

    return data_dict
    # -

def read_radiance_shd_out(shd_obj, abs_obj, squeeze=True):

    """
    Read fluxes from SHDOM output files
    Input:
        shd_obj : positional argument, shd object, e.g., shd_obj = shdom(...)
        abs_obj : positional argument, abs object, e.g., abs_obj = abs_16g(...)
        squeeze=: keyword argument, boolen, default=True, whether to keep axis that has dimension of 1 or not, True:remove, False:keep

    Output:
        data: Python dictionary
            data['rad']            : Radiance (mean/std/raw)
    """

    # +
    # read one file to get the information of dataset dimension
    out0      = shd_out_raw(shd_obj.fnames_out[0], verbose=False)
    dims_info = out0.data[0]['dims_info']
    dims      = out0.data[0]['dims']
    # -

    # +
    # calculate factors using sol_fac (sun-earth distance, weight, slit functions etc.)
    Nz = dims[dims_info.index('Nz')]

    zz     = np.arange(Nz)
    if Nz > 1:
        zz[-1] = zz[-2]

    sol_fac = cal_sol_fac(shd_obj.date)

    norm    = np.zeros(Nz, dtype=np.float32)
    factors = np.zeros((Nz, shd_obj.Ng), dtype=np.float32)

    factors[...] = 1.0
    # for iz in range(Nz):
    #     norm[iz] = sol_fac/(abs_obj.coef['weight']['data'] * abs_obj.coef['slit_func']['data'][zz[iz], :]).sum()
    #     for ig in range(shd_obj.Ng):
    #         factors[iz, ig] = norm[iz]*abs_obj.coef['solar']['data'][ig]*abs_obj.coef['weight']['data'][ig]*abs_obj.coef['slit_func']['data'][zz[iz], ig]
    # -


    # +
    # calculate radiances
    if squeeze:
        dims_info  = [dims_info[i] for i in range(len(dims)) if dims[i] > 1]
        dims       = [i for i in dims if i>1]

    dims_info += ['Nr']

    rad = np.zeros(dims, dtype=np.float32)

    for ig in range(shd_obj.Ng):

        fname0 = shd_obj.fnames_out[ig]

        out0   = shd_out_raw(fname0, verbose=False)
        rad0   = out0.data[0]['data']

        for iz in range(Nz):
            rad0[:, :, iz, :] *= factors[iz, ig]

        if squeeze:
            rad[...] += np.squeeze(rad0)
        else:
            rad[...] += rad0
    # -


    # +
    # store data into Python dictionary
    data_dict = {}

    toa = np.sum(sol_fac * abs_obj.coef['solar']['data'] * abs_obj.coef['weight']['data'])
    data_dict['toa'] = {'data':toa, 'name':'TOA without SZA' , 'units':'W/m^2/nm'}

    data_dict['rad']      = {'data':rad, 'name':'Radiance' , 'units':'W/m^2/nm/sr', 'dims_info':dims_info}

    return data_dict
    # -

if __name__ == '__main__':

    pass
