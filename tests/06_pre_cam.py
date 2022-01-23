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





class aircraft:

    """
    create an aircraft object
    """

    def __init__(
            self,
            date        = datetime.datetime(2019, 10, 5),
            speed       = 250.0,
            pitch_angle = 0.0,
            roll_angle  = 0.0,
            heading_angle = 0.0,
            altitude    = 6000.0
            ):

        """
        aircraft navigational info
        """

        self.date  = date
        self.speed = speed
        self.pitch_angle = pitch_angle
        self.roll_angle = roll_angle
        self.heading_angle = heading_angle

    def camera(
            self,
            camera_type = 'all-sky',
            field_of_view = 180.0,
            sensor_zenith_angle = 0.0,
            sensor_azimuth_angle = 0.0,
            pitch_angle_offset = 0.0,
            roll_angle_offset = 0.0
            ):

        """
        camera info
        """

        self.camera = {
            'camera_type'         : camera_type,
            'field_of_view'       : field_of_view,
            'sensor_zenith_angle' : sensor_zenith_angle,
            'sensor_azimuth_angle': sensor_azimuth_angle,
            'pitch_angle_offset'  : pitch_angle_offset,
            'roll_angle_offset'   : roll_angle_offset
            }

    def geoinfo(
            self,
            Nx = 480,
            Ny = 480,
            dx = 100.0,
            dy = 100.0,
            xpos = 0.5,
            ypos = 0.0,
            ):

        """
        geoinfo
        """

        self.geoinfo = {
                'Nx': Nx,
                'Ny': Ny,
                'xpos': xpos,
                'ypos': ypos,
                'solar_zenith_angle': ,
                'solar_azimuth_angle': ,
                'Nfly': 0
                }

    def fly(
            delta_seconds=1.0
            ):

        travel_dist_x = self.speed * delta_seconds * np.sin(np.deg2rad(self.heading_angle))
        travel_dist_y = self.speed * delta_seconds * np.cos(np.deg2rad(self.heading_angle))

        self.geoinfo['xpos'] += (travel_dist_x // self.geoinfo['dx']) / self.geoinfo['Nx']
        self.geoinfo['ypos'] += (travel_dist_y // self.geoinfo['dy']) / self.geoinfo['Ny']
        self.geoinfo['Nfly'] += 1


    def flyover_view(
            self,
            fdir0='tmp-data/06',
            date = datetime.datetime(2019, 10, 5),
            photons = 1e8,
            solver = '3D',
            wavelength = 600.0,
            surface_albedo = 0.03,
            ):

        # def run_rad_sim(aircraft, fname_nc, fdir0, wavelength=600, overwrite=True):

        """
        core function to run radiance simulation
        """

        fdir = '%s/scene_%3.3d/%4.4dnm' % (fdir0, self.geoinfo['Nfly'], wavelength)

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # setup atmosphere (1D) and clouds (3D)
        # =======================================================================================================
        levels = np.arange(0.0, 20.1, 1.0)

        fname_atm = '%s/atm.pk' % fdir
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=False)
        fname_abs = '%s/abs.pk' % fdir
        abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=False)

        fname_les = '%s/les.pk' % fdir
        cld0      = cld_les(fname_nc='data/les.nc', fname=fname_les, altitude=atm0.lay['altitude']['data'], coarsing=[1, 1, 25, 1], overwrite=False)

        atm1d0    = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
        atm3d0    = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]
        # =======================================================================================================

        mca0 = mcarats_ng(
                date=date,
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                Ng=abs0.Ng,
                target='radiance',
                surface_albedo=surface_albedo,
                solar_zenith_angle=self.geoinfo['solar_zenith_angle'],
                solar_azimuth_angle=self.geoinfo['solar_azimuth_angle'],
                sensor_zenith_angle=self.camera['sensor_zenith_angle'],
                sensor_azimuth_angle=self.camera['sensor_azimuth_angle'],
                sensor_type = self.camera['sensor_type'],
                sensor_xpos = self.geoinfo['xpos'],
                sensor_ypos = self.geoinfo['xpos'],
                fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
                Nrun=3,
                photons=photons,
                weights=abs0.coef['weight']['data'],
                solver=solver,
                Ncpu=12,
                mp_mode='py',
                overwrite=overwrite)

        out0 = mca_out_ng(fname='%s/mca-out-rad-%s_%.2fnm.h5' % (fdir0, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)






if __name__ == '__main__':

    # step 1
    # create an aircraft object
    # =============================================================================
    aircraft0 = aircraft()
    # =============================================================================
