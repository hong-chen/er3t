"""
by Hong Chen (hong.chen@lasp.colorado.edu)

This code serves as an example code to produce training data for CNN described in Nataraja et al. (2022).

The processes include:
    1) `func_cot_vs_rad`: run simulations to establish IPA relationship between COT and radiance for a given wavelength

    2) `main_les`: coarsen a given LES cloud scene by factors of 2 and 4 and run radiance simulations for both the
       original LES scene and coarsened scenes.

    3) `split_data_native_resolution`: upscale the coarsened LES scenes to original spatial resolution of 100m and split
       them into data with size of 480x480

    4) `crop_select_cloud_scene`: crop the 480x480 data into 64x64 mini tiles that each contains ground truth of cloud
       optical thickness from LES and realistic radiance simulated by EaR3T. A random selection is applied to evenly select
       data at different cloud fractions and cloud inhomogeneities to avoid biasing CNN model.

This code has been tested under:
    1) Linux on 2023-03-14 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64
"""

import os
import h5py
import glob
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
import datetime
import time
from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams, ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches



import er3t



# global variables
#/--------------------------------------------------------------\#
params = {
                    'name_tag' : os.path.relpath(__file__).replace('.py', ''),
                  'wavelength' : 600.0,
                        'date' : datetime.datetime(2019, 10, 5),
               'surface_albedo': 0.03,
          'solar_zenith_angle' : 28.9,
         'solar_azimuth_angle' : 296.83,
         'sensor_zenith_angle' : 0.0,
        'sensor_azimuth_angle' : 0.0,
             'sensor_altitude' : 705000.0,
                      'photon' : 1e8,
            'cloud_top_height' : 2.0,
 'cloud_geometrical_thickness' : 1.0,
         'atmospheric_profile' : '%s/afglus.dat' % er3t.common.fdir_data_atmmod,
                    'fname_les': '%s/data/00_er3t_mca/aux/les.nc' % er3t.common.fdir_examples,
                  'photon_ipa' : 1e7,
                     'cer_ipa' : 10.0,
                     'cot_ipa' : np.concatenate((       \
               np.arange(0.0, 2.0, 0.5),     \
               np.arange(2.0, 30.0, 2.0),    \
               np.arange(30.0, 60.0, 5.0),   \
               np.arange(60.0, 100.0, 10.0), \
               np.arange(100.0, 201.0, 50.0) \
               )),
        }
#\--------------------------------------------------------------/#




def run_mca_coarse_case(f_mca, wavelength, fname_les, fdir0, fdir_out='tmp-data/%s/03_sim-ori' % params['name_tag'], coarsen_factor=2, solver='3D', overwrite=True):

    fdir = '%s/%dnm' % (fdir0, wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    levels = np.arange(0.0, 20.1, 0.4)
    fname_atm = '%s/atm.pk' % fdir
    atm0       = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod=params['atmospheric_profile'], overwrite=overwrite)

    fname_abs = '%s/abs.pk' % fdir
    abs0      = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)


    # read in LES cloud
    #/----------------------------------------------------------------------------\#
    fname_les_pk = '%s/les.pk' % fdir
    cld0 = er3t.pre.cld.cld_les(fname_nc=fname_les, fname=fname_les_pk, coarsen=[1, 1, 10], overwrite=overwrite)
    cld0.lay['dx']['data'] *= coarsen_factor
    cld0.lay['dy']['data'] *= coarsen_factor
    #\----------------------------------------------------------------------------/#


    # mca_sca object (enable/disable mie scattering)
    #/----------------------------------------------------------------------------\#
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    #\----------------------------------------------------------------------------/#


    # define atmosphere (clouds, aerosols, trace gases)
    # coarsen if needed
    #/----------------------------------------------------------------------------\#
    atm1d0    = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm3d0    = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)

    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]
    #\----------------------------------------------------------------------------/#

    mca0 = er3t.rtm.mca.mcarats_ng(
            date=params['date'],
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sca=sca,
            Ng=abs0.Ng,
            target='radiance',
            surface_albedo=params['surface_albedo'],
            solar_zenith_angle=params['solar_zenith_angle'],
            solar_azimuth_angle=params['solar_azimuth_angle'],
            sensor_zenith_angle=params['sensor_zenith_angle'],
            sensor_azimuth_angle=params['sensor_azimuth_angle'],
            sensor_altitude=params['sensor_altitude'],
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            photons=params['photon'],
            weights=abs0.coef['weight']['data'],
            solver='3D',
            Ncpu=24,
            mp_mode='py',
            overwrite=overwrite)

    out0 = er3t.rtm.mca.mca_out_ng(fname='%s/mca-out-rad-%s_%.2fnm.h5' % (fdir0, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    rad_3d = out0.data['rad']['data']
    rad_3d[np.isnan(rad_3d)] = 0.0
    rad_3d[rad_3d<0.0] = 0.0

    ref_3d = np.pi*rad_3d/(out0.data['toa']['data']*np.cos(np.deg2rad(params['solar_zenith_angle'])))

    cot_true      = np.sum(cld0.lay['cot']['data'], axis=-1)
    cot_true[np.isnan(cot_true)] = 0.0
    cot_true[cot_true<0.0] = 0.0

    cot_1d      = f_mca.get_cot_from_ref(ref_3d)
    logic_out = (cot_1d<f_mca.cot[0]) | (cot_1d>f_mca.cot[-1])
    logic_low = (logic_out) & (ref_3d<np.median(ref_3d[cot_true>0.0]))
    logic_high = logic_out & np.logical_not(logic_low)
    cot_1d[logic_low]  = f_mca.cot[0]
    cot_1d[logic_high] = f_mca.cot[-1]
    #\----------------------------------------------------------------------------/#

    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fname_new = '%s/data_%s_%dnm.h5' % (fdir_out, os.path.basename(fdir0), wavelength)
    f = h5py.File(fname_new, 'w')

    f['cot_true'] = cot_true
    f['rad_3d']   = rad_3d
    f['ref_3d']   = ref_3d
    f['cot_1d']   = cot_1d

    f.close()




def split_data_native_resolution(fname, coarsen_factor=2, fdir_out='tmp-data/%s/04_sim-native' % params['name_tag']):

    with h5py.File(fname, 'r+') as f:

        for key in f.keys():

            if 'expand' not in key:

                data0 = f[key][...]
                Nx = data0.shape[0]
                Ny = data0.shape[1]

                data = np.zeros((Nx*coarsen_factor, Ny*coarsen_factor), dtype=data0.dtype)
                for i in range(coarsen_factor):
                    for j in range(coarsen_factor):
                        data[i::coarsen_factor , j::coarsen_factor]  = data0

                key_new = '%s_expand' % key
                if key_new in f.keys():
                    del(f[key_new])
                f[key_new] = data

    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    with h5py.File(fname, 'r') as f0:
        for i in range(coarsen_factor):
            for j in range(coarsen_factor):
                N = i + j*coarsen_factor
                fname_new = '%s/%s_%4.4d.h5' % (fdir_out, os.path.basename(fname).replace('.h5', ''), N)
                with h5py.File(fname_new, 'w') as f:
                    for key in f0.keys():
                        if 'expand' in key:
                            x_start = i*Nx
                            x_end   = (i+1) * Nx
                            y_start = j*Ny
                            y_end   = (j+1) * Ny
                            f[key.replace('_expand', '')] = f0[key][...][x_start:x_end, y_start:y_end]




def crop_select_cloud_scene():

    def coarsen(fname, vname, coarsen_factor):

        with h5py.File(fname, 'r') as f:

            data0 = f[vname][...]
            Nx = data0.shape[0]
            Ny = data0.shape[1]

            data = np.zeros((Nx*coarsen_factor, Ny*coarsen_factor), dtype=data0.dtype)
            for i in range(coarsen_factor):
                for j in range(coarsen_factor):
                    data[i::coarsen_factor , j::coarsen_factor]  = data0

        data_split = {}

        for i in range(coarsen_factor):
            for j in range(coarsen_factor):
                vname = i + j*coarsen_factor
                x_start = i*Nx
                x_end   = (i+1) * Nx
                y_start = j*Ny
                y_end   = (j+1) * Ny
                data_split[vname] = data[x_start:x_end, y_start:y_end]

        return data_split

    def cal_std_mean(ref, Np=64, Dp=32):

        Nx, Ny = ref.shape
        x_s = np.arange(Np, Nx-Np-Dp, Dp)
        y_s = np.arange(Np, Ny-Np-Dp, Dp)

        ref_std = np.zeros((x_s.size, y_s.size), dtype=np.float64)
        ref_mean= np.zeros((x_s.size, y_s.size), dtype=np.float64)
        x       = np.zeros((x_s.size, y_s.size), dtype=np.int32)
        y       = np.zeros((x_s.size, y_s.size), dtype=np.int32)

        for i in range(x_s.size):
            for j in range(y_s.size):
                ref_std[i, j]  = np.nanstd(ref[x_s[i]:x_s[i]+Np, y_s[j]:y_s[j]+Np])
                ref_mean[i, j] = np.nanmean(ref[x_s[i]:x_s[i]+Np, y_s[j]:y_s[j]+Np])
                x[i, j]  = x_s[i]
                y[i, j]  = y_s[j]

        return ref_std.ravel(), ref_mean.ravel(), x.ravel(), y.ravel()

    def get_ref_std_ref_mean(coarsen_factor=2, Np=64, Dp=32, sza=29.162360459281544, fdir='tmp-data/%s/04_sim-native' % params['name_tag']):

        fnames = sorted(glob.glob('%s/*coa-fac-%d*600nm*.h5' % (fdir, coarsen_factor)))

        cot = {}
        for fname in fnames:
            cot[fname] = coarsen(fname, 'cot_true', coarsen_factor)

        ref = {}
        for fname in fnames:
            ref[fname] = coarsen(fname, 'ref_3d', coarsen_factor)

        fnames_all = []
        ref_std  = np.array([], dtype=np.float64)
        ref_mean = np.array([], dtype=np.float64)
        index_var= np.array([], dtype=np.int32)
        index_x  = np.array([], dtype=np.int32)
        index_y  = np.array([], dtype=np.int32)
        for fname in fnames:

            for key in cot[fname]:

                ref_std0, ref_mean0, x0, y0 = cal_std_mean(ref[fname][key], Np=Np, Dp=Dp)

                ref_std  = np.append(ref_std, ref_std0)
                ref_mean = np.append(ref_mean, ref_mean0)
                index_x  = np.append(index_x, x0)
                index_y  = np.append(index_y, y0)
                index_var = np.append(index_var, np.repeat(key, ref_std0.size))
                fnames_all += [fname]*ref_std0.size

        data = {}
        data['fnames']    = fnames_all
        data['factor']    = np.repeat(coarsen_factor, ref_std.size)
        data['ref_std']   = ref_std
        data['ref_mean']  = ref_mean
        data['index_var'] = index_var
        data['index_x']   = index_x
        data['index_y']   = index_y

        return data


    data1  = get_ref_std_ref_mean(coarsen_factor=1)
    data2  = get_ref_std_ref_mean(coarsen_factor=2)
    data4  = get_ref_std_ref_mean(coarsen_factor=4)


    data = {}
    for key in data1.keys():
        if key == 'fnames':
            # 'fnames' is Python list
            data[key] = data1[key] + data2[key] + data4[key]
        else:
            # other variables are numpy array
            data[key] = np.concatenate((data1[key], data2[key], data4[key]))


    ref_std  = data['ref_std']
    ref_mean = data['ref_mean']
    indices  = np.arange(ref_std.size)

    # even out the selection probability so coarsened data won't have higher
    # possibility to be selected simply because it has larger data volum
    #/----------------------------------------------------------------------------\#
    prob1 = np.zeros_like(data1['ref_mean'])
    prob1[...] = 0.33333/ prob1.size

    prob2 = np.zeros_like(data2['ref_mean'])
    prob2[...] = 0.33333 / prob2.size

    prob4 = np.zeros_like(data4['ref_mean'])
    prob4[...] = 0.333333 / prob4.size

    prob = np.concatenate((prob1, prob2, prob4))
    data['prob'] = prob
    #\----------------------------------------------------------------------------/#


    # random selection
    # maximum of 100 tiles in one ref-mean vs ref-std grid set by variable <Nselect>
    #/----------------------------------------------------------------------------\#
    xedges = np.arange(0.0, 1.01, 0.02)
    yedges = np.arange(0.0, 0.45, 0.02)
    heatmap, xedges, yedges = np.histogram2d(ref_mean.ravel(), ref_std.ravel(), bins=(xedges, yedges))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)

    Nselect = 100
    indices_select = np.array([], dtype=np.int32)
    indices_plot = []
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            logic = (ref_mean>=xedges[i]) & (ref_mean<xedges[i+1]) & (ref_std>=yedges[j]) & (ref_std<yedges[j+1])
            if logic.sum() > 0:
                prob00 = data['prob'][logic]
                prob0 = prob00/prob00.sum()
                prob0[-1] = 1.0-prob0[:-1].sum()
                indices_select0 = np.random.choice(indices[logic], min([logic.sum(), Nselect]), replace=False, p=prob0)
                indices_select = np.append(indices_select, indices_select0)
                indices_plot.append(indices_select0[0])

    fnames = []
    for index_select in indices_select:
        fnames.append(data['fnames'][index_select])
    #\----------------------------------------------------------------------------/#

    # sort data by LES file name
    # why: this way the LES file will only be opened once to select mini tiles that come from the
    #      same LES file to save time
    #/----------------------------------------------------------------------------\#
    indices_sorted = np.array(sorted(range(len(fnames)), key=fnames.__getitem__))

    fnames_sorted = []
    for index in indices_sorted:
        fnames_sorted.append(fnames[index])

    data_sorted = {}
    data_sorted['fnames'] = fnames_sorted
    for key in data.keys():
        if key != 'fnames':
            data_sorted[key] = data[key][indices_select][indices_sorted]
    #\----------------------------------------------------------------------------/#


    # create mini tiles
    #/----------------------------------------------------------------------------\#
    fdir_out = 'tmp-data/%s/05_sim-select' % params['name_tag']
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fname0 = ''
    for i, index_sorted in enumerate(indices_sorted):

        while data_sorted['fnames'][i] != fname0:
            try:
                f0.close()
            except:
                pass
            cot_true_split = coarsen(data_sorted['fnames'][i], 'cot_true', data_sorted['factor'][i])
            cot_1d_split   = coarsen(data_sorted['fnames'][i], 'cot_1d', data_sorted['factor'][i])
            rad_3d_split   = coarsen(data_sorted['fnames'][i], 'rad_3d', data_sorted['factor'][i])
            ref_3d_split   = coarsen(data_sorted['fnames'][i], 'ref_3d', data_sorted['factor'][i])
            fname0 = data_sorted['fnames'][i]
            f0 = h5py.File(fname0, 'r')

        index_x_s = data_sorted['index_x'][i]
        index_y_s = data_sorted['index_y'][i]
        index_x_e = data_sorted['index_x'][i] + 64
        index_y_e = data_sorted['index_y'][i] + 64

        cot0 = cot_true_split[data_sorted['index_var'][i]][index_x_s:index_x_e, index_y_s:index_y_e]
        ref0 = ref_3d_split[data_sorted['index_var'][i]][index_x_s:index_x_e, index_y_s:index_y_e]

        index_str = '[%8.8d](%3.3d-%3.3d_%3.3d-%3.3d)' % (indices_select[index_sorted], index_x_s, index_x_e, index_y_s, index_y_e)

        fname = '%s/%8.8d_%.4f_%.4f_%s_%s' % (fdir_out, i, np.nanmean(ref0), np.nanstd(ref0), index_str, os.path.basename(fname0))
        f = h5py.File(fname, 'w')
        f['cot_true'] = cot_true_split[data_sorted['index_var'][i]][index_x_s:index_x_e, index_y_s:index_y_e]
        f['cot_1d']   = cot_1d_split[data_sorted['index_var'][i]][index_x_s:index_x_e, index_y_s:index_y_e]
        f['rad_3d']   = rad_3d_split[data_sorted['index_var'][i]][index_x_s:index_x_e, index_y_s:index_y_e]
        f['ref_3d']   = ref_3d_split[data_sorted['index_var'][i]][index_x_s:index_x_e, index_y_s:index_y_e]
        f.close()
    #\----------------------------------------------------------------------------/#




if __name__ == '__main__':


    # step 1
    # derive relationship of COT vs Radiance at a given wavelength
    # data stored under <tmp-data/ipa-0600.0nm_alb-0.03>
    #/----------------------------------------------------------------------------\#
    # fdir1 = 'tmp-data/ipa-%06.1fnm_alb-%04.2f' % (params['wavelength'], params['surface_albedo'])
    # f_mca = er3t.rtm.mca.func_ref_vs_cot(
    #         params['cot_ipa'],
    #         cer0=params['cer_ipa'],
    #         fdir=fdir1,
    #         date=params['date'],
    #         wavelength=params['wavelength'],
    #         surface_albedo=params['surface_albedo'],
    #         solar_zenith_angle=params['solar_zenith_angle'],
    #         solar_azimuth_angle=params['solar_azimuth_angle'],
    #         sensor_altitude=params['sensor_altitude'],
    #         sensor_zenith_angle=params['sensor_zenith_angle'],
    #         sensor_azimuth_angle=params['sensor_azimuth_angle'],
    #         cloud_top_height=params['cloud_top_height'],
    #         cloud_geometrical_thickness=params['cloud_geometrical_thickness'],
    #         Nx=2,
    #         Ny=2,
    #         dx=0.1,
    #         dy=0.1,
    #         photon_number=params['photon_ipa'],
    #         overwrite=False
    #         )
    #\----------------------------------------------------------------------------/#


    # step 2
    # run EaR3T for LES scenes at specified coarsening factor
    # (spatial resolution depends on coarsening factor)
    # raw processing data is stored under <tmp-data/05_cnn-les_rad-sim/02_sim-raw>
    # simulation output data is stored under <tmp-data/05_cnn-les_rad-sim/03_sim-ori>
    #/----------------------------------------------------------------------------\#
    # fdir1 = 'tmp-data/ipa-%06.1fnm_alb-%04.2f' % (params['wavelength'], params['surface_albedo'])
    # f_mca =  er3t.rtm.mca.func_ref_vs_cot(params['cot_ipa'], cer0=params['cer_ipa'], fdir=fdir1, wavelength=params['wavelength'], overwrite=False)
    # for coarsen_factor in [1, 2, 4]:
    #     fdir2 = 'tmp-data/%s/02_sim-raw/les_coa-fac-%d' % (params['name_tag'], coarsen_factor)
    #     run_mca_coarse_case(f_mca, params['wavelength'], params['fname_les'], fdir2, coarsen_factor=coarsen_factor, overwrite=True)
    #\----------------------------------------------------------------------------/#


    # step 3
    # split/upsample the calculation so the spatial resolution is 100 m
    # data stored under <tmp-data/05_cnn-les_rad-sim/04_sim-native>
    #/----------------------------------------------------------------------------\#
    # for fname in sorted(glob.glob('tmp-data/%s/03_sim-ori/*coa-fac-1_600nm.h5' % params['name_tag'])):
    #     split_data_native_resolution(fname, coarsen_factor=1)
    # for fname in sorted(glob.glob('tmp-data/%s/03_sim-ori/*coa-fac-2_600nm.h5' % params['name_tag'])):
    #     split_data_native_resolution(fname, coarsen_factor=2)
    # for fname in sorted(glob.glob('tmp-data/%s/03_sim-ori/*coa-fac-4_600nm.h5' % params['name_tag'])):
    #     split_data_native_resolution(fname, coarsen_factor=4)
    #\----------------------------------------------------------------------------/#


    # step 4
    # split data into 64x64 mini tiles
    # perform random selection based on Mean vs STD grids
    # data stored under <tmp-data/05_cnn-les_rad-sim/05_sim-select>
    #/----------------------------------------------------------------------------\#
    # crop_select_cloud_scene()
    #\----------------------------------------------------------------------------/#

    pass
