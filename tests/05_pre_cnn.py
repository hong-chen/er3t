#!/bin/env python

#SBATCH --partition=shas
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hong.chen-1@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=les_04

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

from er3t.util.cloud import clouds

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g, abs_oco_idl
from er3t.pre.cld import cld_sat, cld_les
from er3t.pre.sfc import sfc_sat
from er3t.util.modis import modis_l1b, modis_l2, modis_09a1, grid_modis_by_extent, grid_modis_by_lonlat, download_modis_https, get_sinusoidal_grid_tag
from er3t.util import cal_r_twostream, cal_ext

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng



class func_cot_vs_rad:

    def __init__(self,
            fdir,
            wavelength,
            cot=np.concatenate((np.arange(0.0, 1.0, 0.1),
                                np.arange(1.0, 10.0, 1.0),
                                np.arange(10.0, 20.0, 2.0),
                                np.arange(20.0, 50.0, 5.0),
                                np.arange(50.0, 100.0, 10.0),
                                np.arange(100.0, 200.0, 20.0),
                                np.arange(200.0, 401.0, 50.0))),
            run=False,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        self.fdir       = fdir
        self.wavelength = wavelength
        self.cot        = cot
        self.rad        = np.array([])

        if run:
            self.run_all()

        for i in range(self.cot.size):
            cot0 = self.cot[i]
            fname = '%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot0)
            out0  = mca_out_ng(fname=fname, mode='mean', squeeze=True)
            self.rad = np.append(self.rad, out0.data['rad']['data'].mean())

    def run_all(self):

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(cot0)

    def run_mca_one(self, cot):

        levels    = np.linspace(0.0, 20.0, 21)

        fname_atm = '%s/atm.pk' % self.fdir
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)

        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_16g(wavelength=self.wavelength, fname=fname_abs, atm_obj=atm0, overwrite=True)

        cot_2d    = np.zeros((2, 2), dtype=np.float64); cot_2d[...] = cot
        cer_2d    = np.zeros((2, 2), dtype=np.float64); cer_2d[...] = 12.0
        ext_3d    = np.zeros((2, 2, 2), dtype=np.float64)

        fname_nc  = '/data/hong/mygit/er3t/tests/data/les.nc'
        fname_les = '%s/les.pk' % self.fdir
        cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=True)

        cld0.lev['altitude']['data']    = cld0.lay['altitude']['data'][2:5]

        cld0.lay['x']['data']           = cld0.lay['x']['data'][:2]
        cld0.lay['y']['data']           = cld0.lay['y']['data'][:2]
        cld0.lay['nx']['data']          = 2
        cld0.lay['ny']['data']          = 2
        cld0.lay['altitude']['data']    = cld0.lay['altitude']['data'][2:4]
        cld0.lay['pressure']['data']    = cld0.lay['pressure']['data'][2:4]
        cld0.lay['temperature']['data'] = cld0.lay['temperature']['data'][:2, :2, 2:4]
        cld0.lay['cot']['data']         = cot_2d
        cld0.lay['thickness']['data']   = cld0.lay['thickness']['data'][2:4]

        ext_3d[:, :, 0]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 1]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        cld0.lay['extinction']['data']  = ext_3d

        atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % self.fdir)
        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]

        mca0 = mcarats_ng(
                date=datetime.datetime(2016, 8, 29),
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                Ng=abs0.Ng,
                target='radiance',
                surface_albedo=0.0,
                solar_zenith_angle=29.162360459281544,
                solar_azimuth_angle=-63.16777636586792,
                sensor_zenith_angle=0.0,
                sensor_azimuth_angle=0.0,
                fdir='%s/%.2f/les_rad_3d' % (self.fdir, cot),
                Nrun=1,
                photons=1e7,
                solver='3D',
                Ncpu=24,
                mp_mode='py',
                overwrite=True)

        out0 = mca_out_ng(fname='%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot), mca_obj=mca0, abs_obj=abs0, mode='all', squeeze=True, verbose=True, overwrite=True)

    def interp_from_rad(self, rad, method='cubic'):

        f = interp1d(self.rad, self.cot, kind=method, bounds_error=False)

        return f(rad)

    def interp_from_cot(self, cot, method='cubic'):

        f = interp1d(self.cot, self.rad, kind=method, bounds_error=False)

        return f(cot)





def run_mca_coarse_case(f_mca, wavelength, fname_nc, fdir0, coarsen_factor=2, overwrite=True):

    fdir = '%s/%dnm' % (fdir0, wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    levels = np.arange(0.0, 20.1, 1.0)

    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)

    fname_les = '%s/les.pk' % fdir
    cld0      = cld_les(fname_nc=fname_nc, fname=fname_les, altitude=atm0.lay['altitude']['data'], coarsing=[1, 1, 1, 1], overwrite=overwrite)

    # radiance 3d
    # =======================================================================================================
    target    = 'radiance'
    solver    = '3D'
    atm1d0    = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm3d0    = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)

    # coarsen
    atm3d0.nml['Atm_dx']['data'] *= coarsen_factor
    atm3d0.nml['Atm_dy']['data'] *= coarsen_factor

    atm_1ds   = [atm1d0]
    atm_3ds   = [atm3d0]

    mca0 = mcarats_ng(
            date=datetime.datetime(2016, 8, 29),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            Ng=abs0.Ng,
            target=target,
            surface_albedo=0.0,
            solar_zenith_angle=29.162360459281544,
            solar_azimuth_angle=-63.16777636586792,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            # photons=1e8*coarsen_factor,
            photons=2e9,
            weights=abs0.coef['weight']['data'],
            solver=solver,
            Ncpu=24,
            mp_mode='py',
            overwrite=overwrite)

    out0 = mca_out_ng(fname='%s/mca-out-rad-%s_%.2fnm.h5' % (fdir0, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    rad_3d      = out0.data['rad']['data']
    rad_3d[np.isnan(rad_3d)] = 0.0
    rad_3d[rad_3d<0.0] = 0.0

    cot_true      = np.sum(cld0.lay['cot']['data'], axis=-1)
    cot_true[np.isnan(cot_true)] = 0.0
    cot_true[cot_true<0.0] = 0.0

    cot_1d      = f_mca.interp_from_rad(rad_3d)
    cot_1d[np.isnan(cot_1d)] = 0.0
    cot_1d[cot_1d<0.0] = 0.0
    # =======================================================================================================

    fname_new = 'data/data_%s_coa-fac-%d_%dnm.h5' % (os.path.basename(fdir0), coarsen_factor, wavelength)
    f = h5py.File(fname_new, 'w')

    f['cot_true'] = cot_true
    f['rad_3d']   = rad_3d
    f['cot_1d']   = cot_1d

    f.close()

def run(fdir, fname_nc, coarsen_factor=2):

    for wvl in [600.0]:
        f_mca =  func_cot_vs_rad('data/ret/%3.3d' % wvl, wvl, run=False)
        run_mca_coarse_case(f_mca, wvl, fname_nc, fdir, coarsen_factor=coarsen_factor, overwrite=True)

def main_les_tak(coarsen_factor=2):

    fnames_nc = [
            'data/7seas/x48km_TB_nt035_undg_tau1h_nndg_tau1h_v03_control/7SEAS_480x480x150_dx100m_dz40m_dt2sec_480_0000081000_mod.nc',
            'data/7seas/x48km_TB_nt035_undg_tau1h_nndg_tau1h_v03_shear/7SEAS_480x480x150_dx100m_dz40m_dt2sec_480_0000081000_mod.nc',
            'data/7seas/x48km_TB_nt150_undg_tau1h_nndg_tau1h_v03_control/7SEAS_480x480x150_dx100m_dz40m_dt2sec_480_0000081000_mod.nc',
            'data/7seas/x48km_TB_nt150_undg_tau1h_nndg_tau1h_v03_shear/7SEAS_480x480x150_dx100m_dz40m_dt2sec_480_0000081000_mod.nc',
            'data/7seas/x48km_TB_nt230_undg_tau1h_nndg_tau1h_v03_control/7SEAS_480x480x150_dx100m_dz40m_dt2sec_480_0000081000_mod.nc',
            'data/7seas/x48km_TB_nt230_undg_tau1h_nndg_tau1h_v03_shear/7SEAS_480x480x150_dx100m_dz40m_dt2sec_480_0000081000_mod.nc'
            ]

    for fname_nc in fnames_nc:
        run('tmp-data/%s_coa-fac-%d' % (fname_nc.split('/')[-2], coarsen_factor), fname_nc, coarsen_factor=coarsen_factor)





def split_data(fname, coarsen_factor=2):

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

    with h5py.File(fname, 'r') as f0:
        for i in range(coarsen_factor):
            for j in range(coarsen_factor):
                N = i + j*coarsen_factor
                fname_new = 'data/%s_%4.4d.h5' % (os.path.basename(fname).replace('.h5', ''), N)
                with h5py.File(fname_new, 'w') as f:
                    for key in f0.keys():
                        if 'expand' in key:
                            x_start = i*Nx
                            x_end   = (i+1) * Nx
                            y_start = j*Ny
                            y_end   = (j+1) * Ny
                            f[key.replace('_expand', '')] = f0[key][...][x_start:x_end, y_start:y_end]





def select_cloud_scene():

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

    def get_ref_std_ref_mean(coarsen_factor=2, Np=64, Dp=32, sza=29.162360459281544):

        fnames = sorted(glob.glob('data/*coa-fac-%d_coa-fac-%d*600nm*.h5' % (coarsen_factor, coarsen_factor)))

        cot = {}
        for fname in fnames:
            cot[fname] = coarsen(fname, 'cot_true', coarsen_factor)


        fnames_all = []
        ref_std  = np.array([], dtype=np.float64)
        ref_mean = np.array([], dtype=np.float64)
        index_var= np.array([], dtype=np.int32)
        index_x  = np.array([], dtype=np.int32)
        index_y  = np.array([], dtype=np.int32)
        for fname in fnames:

            for key in cot[fname]:

                ref = cal_r_twostream(cot[fname][key], a=0.0, g=0.85, mu=np.cos(np.deg2rad(sza)))
                ref_std0, ref_mean0, x0, y0 = cal_std_mean(ref, Np=Np, Dp=Dp)

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
            data[key] = data1[key] + data2[key] + data4[key]
        else:
            data[key] = np.concatenate((data1[key], data2[key], data4[key]))


    ref_std  = data['ref_std']
    ref_mean = data['ref_mean']
    indices  = np.arange(ref_std.size)

    prob1 = np.zeros_like(data1['ref_mean'])
    prob1[...] = 0.33333/ prob1.size

    prob2 = np.zeros_like(data2['ref_mean'])
    prob2[...] = 0.33333 / prob2.size

    prob4 = np.zeros_like(data4['ref_mean'])
    prob4[...] = 0.333333 / prob4.size

    prob = np.concatenate((prob1, prob2, prob4))
    data['prob'] = prob


    # xedges = np.linspace(0.0, 0.3, 50)
    # yedges = np.linspace(0.0, 0.3, 50)
    xedges = np.arange(0.0, 1.01, 0.02)
    yedges = np.arange(0.0, 0.45, 0.02)
    heatmap, xedges, yedges = np.histogram2d(ref_mean.ravel(), ref_std.ravel(), bins=(xedges, yedges))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)

    Nselect = 500
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

    # print(indices_select.size)
    # print(np.unique(indices_select).size)
    # exit()


    # print(ref_mean.size)
    # print(XX.size, (heatmap>0).sum(), heatmap[heatmap>0].min())
    # values, counts = np.unique(heatmap[heatmap>0], return_counts=True)
    # for value, count in zip(values, counts):
    #     print(int(value), count)
    # print()
    # exit()


    fnames = []
    for index_select in indices_select:
        fnames.append(data['fnames'][index_select])

    # for all the data
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # fnames = data['fnames'].copy()
    # indices_select = np.arange(len(fnames))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    indices_sorted = np.array(sorted(range(len(fnames)), key=fnames.__getitem__))

    fnames_sorted = []
    for index in indices_sorted:
        fnames_sorted.append(fnames[index])

    data_sorted = {}
    data_sorted['fnames'] = fnames_sorted
    for key in data.keys():
        if key != 'fnames':
            print(key)
            data_sorted[key] = data[key][indices_select][indices_sorted]




if __name__ == '__main__':

    # step 1
    # derive relationship of COT vs Radiance at a given wavelength
    # =============================================================================
    # wvl = 600.0
    # f_mca =  func_cot_vs_rad('data/ret/%3.3d' % wvl, wvl, run=True)
    # =============================================================================


    # step 2
    # run ERT for LES scenes at specified coarsening factor
    # (spatial resolution depends on coarsening factor)
    # =============================================================================
    # for coarsen_factor in [1, 2, 4, 8]:
        # main_les_tak(coarsen_factor=coarsen_factor)
    # =============================================================================


    # step 3
    # split/upsample the calculation so the spatial resolution is 100 m
    # =============================================================================
    # for fname in sorted(glob.glob('data/*coa-fac-2_coa-fac-2_600nm.h5')):
    #     split_data(fname, coarsen_factor=2)
    # for fname in sorted(glob.glob('data/*coa-fac-4_coa-fac-4_600nm.h5')):
    #     split_data(fname, coarsen_factor=4)
    # for fname in sorted(glob.glob('data/*coa-fac-8_coa-fac-8_600nm.h5')):
    #     split_data(fname, coarsen_factor=8)
    # =============================================================================


    # step 4
    # split data into 64x64 mini tiles
    # perform random selection based on Mean vs STD grids
    # =============================================================================
    # select_cloud_scene_new()
    # =============================================================================

    pass
