import os
import sys
import h5py
import warnings
from collections import OrderedDict
import numpy as np

from er3t.util import cal_sol_fac, h5dset_to_pydict



__all__ = ['mca_out_raw', 'mca_out_ng']



class mca_out_raw:

    """
    Read a single MCARaTS output binary file based on the information provided in the .ctl file

    Input:
        fname_bin: positional argument, string type, file path of the mcarats binary file

    Output:
        self.data: Python list
    """


    def __init__(self, fname_bin):

        if not os.path.isfile(fname_bin):
            msg = 'Error [mca_out_raw]: Cannot find <%s>.' % fname_bin
            raise OSError(msg)

        fname_ctl = fname_bin + '.ctl'
        if not os.path.isfile(fname_ctl):
            msg = 'Error [mca_out_raw]: Cannot find <%s>.' % fname_ctl
            raise OSError(msg)

        self.fname_bin = fname_bin
        self.fname_ctl = fname_ctl

        self.data = []
        self.read_ctl()
        self.read_bin()


    def read_ctl(self):

        f     = open(self.fname_ctl, 'r')
        lines = f.readlines()

        Ns = 0
        for i, line in enumerate(lines):
            line = line.strip()

            if 'XDEF' in line:
                line = line.replace('XDEF', '').replace('LINEAR', '').strip()
                Nx, S, I = np.int_(line.split())

            elif 'YDEF' in line:
                line = line.replace('YDEF', '').replace('LINEAR', '').strip()
                Ny, S, I = np.int_(line.split())

            elif 'TDEF' in line:
                line = line.replace('TDEF', '').strip()
                Nt = int(line.split()[0])

            elif 'VARS' in line and 'ENDVARS' not in line:
                line      = line.replace('VARS', '').strip()
                Nvar      = int(line)
                self.Nvar = Nvar

                for j in range(i+1, i+Nvar+1):
                    line       = lines[j].strip()
                    words      = line.split()
                    vname      = ' '.join(([words[0]]+['(%s)' % ' '.join(words[3:])]))

                    Nz         = int(words[1])
                    dims       = [Nx, Ny, Nz, Nt]

                    Ne         = Ns + Nx*Ny*Nz*Nt

                    self.data.append({'name':vname, 'dims':dims, 'dims_info': ['Nx', 'Ny', 'Nz', 'Nt'], \
                            'Index_Start':Ns, 'Index_End':Ne})

                    Ns         = Ne

                i = j

        f.close()


    def read_bin(self, dtype='<f4'):

        data_bin  = np.fromfile(self.fname_bin, dtype=dtype)

        for i, info in enumerate(self.data):

            self.data[i]['name']      = info['name']
            self.data[i]['dims']      = info['dims']
            self.data[i]['dims_info'] = info['dims_info']
            self.data[i]['data']      = data_bin[info['Index_Start']:info['Index_End']].reshape(info['dims'], order='F')



class mca_out_ng:

    """
    Read MCARaTS (16g) output files (flux)

    Input:
        fname=    : HDF5 file path of the many MCARaTS binary files to be written/read, string, default=None
        mca_obj=  : mcarats object, e.g., mca_obj = mcarats(...), default=None
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
        If fname is specified but does not exist while mca_obj and abs_obj are specified, create file first and then read the data from the file;
        If fname is not specified, mca_obj and abs_obj are specified, read data directly from MCARaTS binary files
    """


    def __init__(self, \
                 fname     = None,  \
                 mca_obj   = None,  \
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
        self.mca       = mca_obj
        self.abs       = abs_obj


        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load()

        elif (((mca_obj is not None) and (abs_obj is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((mca_obj is not None) and (abs_obj is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run()
            self.dump()

        elif (((mca_obj is not None) and (abs_obj is not None)) and (fname is None)):

            self.run()

        else:

            msg = 'Error [mca_out_ng]: Please provide both <mca_obj> and <abs_obj> to proceed.'
            raise OSError(msg)


    def load(self):

        if self.verbose:
            print('Message [mca_out_ng]: Reading \'%s\' from \'%s\' ...' % (self.mca.target.lower(), self.fname))

        self.data = {}

        f = h5py.File(self.fname, 'r')
        g = f[self.mode]

        for key in g.keys():

            self.data[key] = h5dset_to_pydict(g[key])

        f.close()


    def run(self):

        if self.verbose:
            print('Message [mca_out_ng]: Reading \'%s\' ...' % self.mca.target.lower())

        if self.mca.target == 'flux': # ['f', 'flux', 'irradiance', 'heating rate', 'hr']:
            self.data = read_flux_mca_out(self.mca, self.abs, mode=self.mode, squeeze=self.squeeze)

        elif self.mca.target == 'radiance':
            self.data = read_radiance_mca_out(self.mca, self.abs, mode=self.mode, squeeze=self.squeeze)


    def dump(self):

        if not self.quiet:
            print('Message [mca_out_ng]: Saving \'%s\' into \'%s\' ...' % (self.mca.target.lower(), self.fname))

        mode = self.mode.lower()

        f = h5py.File(self.fname, 'w')

        g = f.create_group(mode)
        for key in self.data.keys():
            g[key] = self.data[key]['data']
            for key0 in self.data[key].keys():
                if key0 != 'data':
                    if key0 == 'dims_info':
                        g[key].attrs[key0]  = np.string_(self.data[key][key0])
                    else:
                        g[key].attrs[key0]  = self.data[key][key0]

        f.close()



def save_h5_mca_out(fname, mca_obj, abs_obj, mode='mean', squeeze=True):

    """
    Save fluxes from MCARaTS output files into a HDF5 file

    Input:
        fname   : positional argument, string type, file path of HDF5 file
        mca_obj : positional argument, mca object, e.g., mca_obj = mcarats(...)
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

    if mca_obj.target == 'flux':
        data = read_flux_mca_out(mca_obj, abs_obj, mode=mode, squeeze=squeeze)
    elif mca_obj.target == 'radiance':
        data = read_radiance_mca_out(mca_obj, abs_obj, mode=mode, squeeze=squeeze)

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



def read_flux_mca_out(mca_obj, abs_obj, mode='mean', squeeze=True):

    """
    Read fluxes from MCARaTS output files
    Input:
        mca_obj : positional argument, mca object, e.g., mca_obj = mcarats(...)
        abs_obj : positional argument, abs object, e.g., abs_obj = abs_16g(...)
        mode=   : keyword argument, string, default='mean', can be 'mean', 'all'
        squeeze=: keyword argument, boolen, default=True, whether to keep axis that has dimension of 1 or not, True:remove, False:keep

    Output:
        data: Python dictionary
            data['f_up']            : Global upwelling irradiance (mean/std/raw)
            data['f_down']          : Global downwelling irradiance (mean/std/raw)
            data['f_down_direct']   : Direct downwelling irradiance (mean/std/raw)
            data['f_down_diffuse']  : Diffuse downwelling irradiance (mean/std/raw)
    """

    mode = mode.lower()

    # +
    # read one file to get the information of dataset dimension
    out0      = mca_out_raw(mca_obj.fnames_out[0][0])
    dims_info = out0.data[0]['dims_info']
    dims      = out0.data[0]['dims']
    # -


    # +
    # calculate factors using sol_fac (sun-earth distance, weight, slit functions etc.)
    Nz = dims[dims_info.index('Nz')]

    zz     = np.arange(Nz)
    if Nz > 1:
        zz[-1] = zz[-2]

    sol_fac = cal_sol_fac(mca_obj.date)

    norm    = np.zeros(Nz, dtype=np.float64)
    factors = np.zeros((Nz, mca_obj.Ng), dtype=np.float64)

    for iz in range(Nz):
        norm[iz] = sol_fac/(abs_obj.coef['weight']['data'] * abs_obj.coef['slit_func']['data'][zz[iz], :]).sum()
        for ig in range(mca_obj.Ng):
            factors[iz, ig] = norm[iz]*abs_obj.coef['solar']['data'][ig]*abs_obj.coef['weight']['data'][ig]*abs_obj.coef['slit_func']['data'][zz[iz], ig]
    # -


    # +
    # calculate fluxes
    if squeeze:
        dims_info  = [dims_info[i] for i in range(len(dims)) if dims[i] > 1]
        dims       = [i for i in dims if i>1]

    dims_info += ['Nr']
    dims      += [mca_obj.Nrun]

    f_down_direct = np.zeros(dims, dtype=np.float64)
    f_down        = np.zeros(dims, dtype=np.float64)
    f_up          = np.zeros(dims, dtype=np.float64)

    for ir in range(mca_obj.Nrun):
        for ig in range(mca_obj.Ng):

            fname0         = mca_obj.fnames_out[ir][ig]

            out0           = mca_out_raw(fname0)
            f_down_direct0 = out0.data[0]['data']
            f_down0        = out0.data[1]['data']
            f_up0          = out0.data[2]['data']

            for iz in range(Nz):
                f_down_direct0[:, :, iz, :] *= factors[iz, ig]
                f_down0[:, :, iz, :]        *= factors[iz, ig]
                f_up0[:, :, iz, :]          *= factors[iz, ig]

            if squeeze:
                f_down_direct[..., ir] += np.squeeze(f_down_direct0)
                f_down[..., ir]        += np.squeeze(f_down0)
                f_up[..., ir]          += np.squeeze(f_up0)
            else:
                f_down_direct[..., ir] += f_down_direct0
                f_down[..., ir]        += f_down0
                f_up[..., ir]          += f_up0
    # -


    # +
    # store data into Python dictionary
    data_dict = {}

    toa = np.sum(sol_fac * abs_obj.coef['solar']['data'] * abs_obj.coef['weight']['data'])
    data_dict['toa'] = {'data':toa, 'name':'TOA without SZA' , 'units':'W/m^2/nm'}

    if mode == 'all':

        data_dict['f_down']          = {'data':f_down              , 'name':'Global downwelling flux' , 'units':'W/m^2/nm', 'dims_info':dims_info}
        data_dict['f_up']            = {'data':f_up                , 'name':'Global upwelling flux'   , 'units':'W/m^2/nm', 'dims_info':dims_info}
        data_dict['f_down_direct']   = {'data':f_down_direct       , 'name':'Direct downwelling flux' , 'units':'W/m^2/nm', 'dims_info':dims_info}
        data_dict['f_down_diffuse']  = {'data':f_down-f_down_direct, 'name':'Diffuse downwelling flux', 'units':'W/m^2/nm', 'dims_info':dims_info}

        data_dict['N_photon']        = {'data':mca_obj.photons     , 'name': 'Number of photons'      , 'units': 'N/A'}
        data_dict['N_run']           = {'data':mca_obj.Nrun        , 'name': 'Number of runs'         , 'units': 'N/A'}

    elif mode == 'mean':

        data_dict['f_down']          = {'data':np.mean(f_down              , axis=-1), 'name':'Global downwelling flux (mean)' , 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}
        data_dict['f_up']            = {'data':np.mean(f_up                , axis=-1), 'name':'Global upwelling flux (mean)'   , 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}
        data_dict['f_down_direct']   = {'data':np.mean(f_down_direct       , axis=-1), 'name':'Direct downwelling flux (mean)' , 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}
        data_dict['f_down_diffuse']  = {'data':np.mean(f_down-f_down_direct, axis=-1), 'name':'Diffuse downwelling flux (mean)', 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}

        data_dict['f_down_std']          = {'data':np.std(f_down              , axis=-1), 'name':'Global downwelling flux (standard deviation)' , 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}
        data_dict['f_up_std']            = {'data':np.std(f_up                , axis=-1), 'name':'Global upwelling flux (standard deviation)'   , 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}
        data_dict['f_down_direct_std']   = {'data':np.std(f_down_direct       , axis=-1), 'name':'Direct downwelling flux (standard deviation)' , 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}
        data_dict['f_down_diffuse_std']  = {'data':np.std(f_down-f_down_direct, axis=-1), 'name':'Diffuse downwelling flux (standard deviation)', 'units':'W/m^2/nm', 'dims_info':dims_info[:-1]}

        data_dict['N_photon']        = {'data':mca_obj.photons     , 'name': 'Number of photons'      , 'units': 'N/A'}
        data_dict['N_run']           = {'data':mca_obj.Nrun        , 'name': 'Number of runs'         , 'units': 'N/A'}

    else:

        msg = 'Error [read_flux_mca_out]: Do not support <mode=%s>.' % mode
        raise OSError(msg)

    return data_dict
    # -



def read_radiance_mca_out(mca_obj, abs_obj, mode='mean', squeeze=True):

    """
    Read fluxes from MCARaTS output files
    Input:
        mca_obj : positional argument, mca object, e.g., mca_obj = mcarats(...)
        abs_obj : positional argument, abs object, e.g., abs_obj = abs_16g(...)
        mode=   : keyword argument, string, default='mean', can be 'mean', 'all'
        squeeze=: keyword argument, boolen, default=True, whether to keep axis that has dimension of 1 or not, True:remove, False:keep

    Output:
        data: Python dictionary
            data['rad']            : Radiance (mean/std/raw)
    """

    mode = mode.lower()

    # +
    # read one file to get the information of dataset dimension
    out0      = mca_out_raw(mca_obj.fnames_out[0][0])
    dims_info = out0.data[0]['dims_info']
    dims      = out0.data[0]['dims']
    # -

    # +
    # calculate factors using sol_fac (sun-earth distance, weight, slit functions etc.)
    Nz = dims[dims_info.index('Nz')]

    zz     = np.arange(Nz)
    if Nz > 1:
        zz[-1] = zz[-2]

    sol_fac = cal_sol_fac(mca_obj.date)

    norm    = np.zeros(Nz, dtype=np.float64)
    factors = np.zeros((Nz, mca_obj.Ng), dtype=np.float64)

    for iz in range(Nz):
        norm[iz] = sol_fac/(abs_obj.coef['weight']['data'] * abs_obj.coef['slit_func']['data'][zz[iz], :]).sum()
        for ig in range(mca_obj.Ng):
            factors[iz, ig] = norm[iz]*abs_obj.coef['solar']['data'][ig]*abs_obj.coef['weight']['data'][ig]*abs_obj.coef['slit_func']['data'][zz[iz], ig]
    # -


    # +
    # calculate radiances
    if squeeze:
        dims_info  = [dims_info[i] for i in range(len(dims)) if dims[i] > 1]
        dims       = [i for i in dims if i>1]

    dims_info += ['Nr']
    dims      += [mca_obj.Nrun]

    rad = np.zeros(dims, dtype=np.float64)

    for ir in range(mca_obj.Nrun):
        for ig in range(mca_obj.Ng):

            fname0 = mca_obj.fnames_out[ir][ig]

            out0   = mca_out_raw(fname0)
            rad0   = out0.data[0]['data']

            for iz in range(Nz):
                rad0[:, :, iz, :] *= factors[iz, ig]

            if squeeze:
                rad[..., ir] += np.squeeze(rad0)
            else:
                rad[..., ir] += rad0
    # -


    # +
    # store data into Python dictionary
    data_dict = {}

    toa = np.sum(sol_fac * abs_obj.coef['solar']['data'] * abs_obj.coef['weight']['data'])
    data_dict['toa'] = {'data':toa, 'name':'TOA without SZA' , 'units':'W/m^2/nm'}

    if mode == 'all':
        data_dict['rad']      = {'data':rad, 'name':'Radiance' , 'units':'W/m^2/nm/sr', 'dims_info':dims_info}
        data_dict['N_photon'] = {'data':mca_obj.photons     , 'name': 'Number of photons'      , 'units': 'N/A'}
        data_dict['N_run']    = {'data':mca_obj.Nrun        , 'name': 'Number of runs'         , 'units': 'N/A'}
    elif mode == 'mean':
        data_dict['rad']      = {'data':np.mean(rad, axis=-1), 'name':'Radiance (mean)' , 'units':'W/m^2/nm/sr', 'dims_info':dims_info[:-1]}
        data_dict['rad_std']  = {'data':np.std(rad, axis=-1), 'name':'Radiance (standard deviation)' , 'units':'W/m^2/nm/sr', 'dims_info':dims_info[:-1]}
        data_dict['N_photon'] = {'data':mca_obj.photons     , 'name': 'Number of photons'      , 'units': 'N/A'}
        data_dict['N_run']    = {'data':mca_obj.Nrun        , 'name': 'Number of runs'         , 'units': 'N/A'}
    else:
        msg = 'Error [read_radiance_mca_out]: Do not support <mode=%s>.' % mode
        raise OSError(msg)

    return data_dict
    # -



if __name__ == "__main__":

    pass
