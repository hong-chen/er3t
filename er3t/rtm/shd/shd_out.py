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


__all__ = ['get_shd_data_out', 'get_shd_data_out_ori']


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


if __name__ == '__main__':

    pass
