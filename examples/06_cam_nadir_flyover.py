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
import numpy as np
import datetime
import time
from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams, ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_sat, cld_les
from er3t.util import cal_r_twostream

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng



def cal_solar_angles(dtime, longitude, latitude, altitude):

    dtime = dtime.replace(tzinfo=datetime.timezone.utc)

    sza_i = 90.0 - pysolar.solar.get_altitude(latitude, longitude, dtime, elevation=altitude)
    if sza_i < 0.0 or sza_i > 90.0:
        sza_i = np.nan

    saa_i = pysolar.solar.get_azimuth(latitude, longitude, dtime, elevation=altitude)
    if saa_i >= 0.0:
        if 0.0<=saa_i<=180.0:
            saa_i = 180.0 - saa_i
        elif 180.0<saa_i<=360.0:
            saa_i = 540.0 - saa_i
        else:
            saa_i = np.nan
    elif saa_i < 0.0:
        if -180.0<=saa_i<0.0:
            saa_i = -saa_i + 180.0
        elif -360.0<=saa_i<-180.0:
            saa_i = -saa_i - 180.0
        else:
            saa_i = np.nan

    return sza_i, saa_i

class aircraft:

    """
    create an aircraft object
    """

    def __init__(
            self,
            date        = datetime.datetime(2019, 10, 5, 2),
            speed       = 200.0,
            pitch_angle = 0.0,
            roll_angle  = 0.0,
            heading_angle = 0.0,
            altitude    = 6000.0
            ):

        """
        aircraft navigational info
        """

        self.datetime = date
        self.speed = speed
        self.pitch_angle = pitch_angle
        self.roll_angle = roll_angle
        self.heading_angle = heading_angle
        self.altitude = altitude

    def install_camera(
            self,
            sensor_type = 'all-sky camera',
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
            'sensor_type'         : sensor_type,
            'field_of_view'       : field_of_view,
            'sensor_zenith_angle' : sensor_zenith_angle,
            'sensor_azimuth_angle': sensor_azimuth_angle,
            'pitch_angle_offset'  : pitch_angle_offset,
            'roll_angle_offset'   : roll_angle_offset
            }

    def initialize_geoinfo(
            self,
            extent = [122.55, 123.0, 15.55, 16.0],
            Nx = 480,
            Ny = 480,
            dx = 100.0,
            dy = 100.0,
            xpos = 0.2,
            ypos = 0.1,
            ):

        """
        geoinfo
        """

        lon = extent[0]+xpos*(extent[1]-extent[0])
        lat = extent[2]+ypos*(extent[3]-extent[2])
        sza, saa = cal_solar_angles(self.datetime, lon, lat, self.altitude)

        self.geoinfo = {
                'extent': extent,
                'Nx': Nx,
                'Ny': Ny,
                'dx': dx,
                'dy': dy,
                'xpos': xpos,
                'ypos': ypos,
                'lon' : lon,
                'lat' : lat,
                'solar_zenith_angle': sza,
                'solar_azimuth_angle': saa,
                'Nfly': 0
                }

    def fly_to_next(
            self,
            delta_seconds=1.0
            ):

        travel_dist_x = self.speed * delta_seconds * np.sin(np.deg2rad(self.heading_angle))
        travel_dist_y = self.speed * delta_seconds * np.cos(np.deg2rad(self.heading_angle))

        self.geoinfo['xpos'] += (travel_dist_x // self.geoinfo['dx']) / self.geoinfo['Nx']
        self.geoinfo['ypos'] += (travel_dist_y // self.geoinfo['dy']) / self.geoinfo['Ny']

        lon   = self.geoinfo['extent'][0]+self.geoinfo['xpos']*(self.geoinfo['extent'][1]-self.geoinfo['extent'][0])
        lat   = self.geoinfo['extent'][2]+self.geoinfo['ypos']*(self.geoinfo['extent'][3]-self.geoinfo['extent'][2])
        self.geoinfo['lon']   = lon
        self.geoinfo['lat']   = lat

        self.datetime += datetime.timedelta(seconds=delta_seconds)

        sza, saa = cal_solar_angles(self.datetime, lon, lat, self.altitude)
        self.geoinfo['solar_zenith_angle'] = sza
        self.geoinfo['solar_azimuth_angle'] = saa

        self.geoinfo['Nfly'] += 1

    def flyover_view(
            self,
            fdir0='tmp-data/06',
            date = datetime.datetime(2019, 10, 5),
            photons = 1e8,
            solver = '3D',
            wavelength = 600.0,
            surface_albedo = 0.03,
            plot=True,
            overwrite=True
            ):

        # def run_rad_sim(aircraft, fname_nc, fdir0, wavelength=600, overwrite=True):

        """
        core function to run radiance simulation
        """

        fdir = '%s/%4.4dnm' % (fdir0, wavelength)
        fdir_scene = '%s/scene_%3.3d' % (fdir, self.geoinfo['Nfly'])

        if not os.path.exists(fdir_scene):
            os.makedirs(fdir_scene)

        # setup atmosphere (1D) and clouds (3D)
        # =======================================================================================================
        levels = np.arange(0.0, 20.1, 1.0)

        fname_atm = '%s/atm.pk' % fdir0
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=False)
        fname_abs = '%s/abs.pk' % fdir0
        abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=False)

        fname_les = '%s/les.pk' % fdir0
        # cld0      = cld_les(fname_nc='data/les.nc', fname=fname_les, coarsing=[1, 1, 25, 1], overwrite=False)
        cld0      = cld_les(fname_nc='data/les.nc', altitude=levels, fname=fname_les, overwrite=False)

        atm1d0    = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
        atm3d0    = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir0)

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
                sensor_altitude = self.altitude,
                sensor_type = self.camera['sensor_type'],
                sensor_xpos = self.geoinfo['xpos'],
                sensor_ypos = self.geoinfo['ypos'],
                fdir='%s/rad_%s' % (fdir_scene, solver.lower()),
                Nrun=3,
                photons=photons,
                weights=abs0.coef['weight']['data'],
                solver=solver,
                Ncpu=12,
                mp_mode='py',
                overwrite=overwrite)

        fname_out = '%s/mca-out-rad-%s_%.2fnm_%3.3d.h5' % (fdir0, solver.lower(), wavelength, self.geoinfo['Nfly'])
        out0 = mca_out_ng(fname=fname_out, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

        # plot
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if plot:
            fname_png = os.path.basename(fname_out).replace('.h5', '.png')

            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            cs = ax1.imshow(np.transpose(out0.data['rad']['data']), cmap='Greys_r', vmin=0.0, vmax=0.3, origin='lower')
            plt.colorbar(cs)
            ax1.set_xlabel('X Index')
            ax1.set_ylabel('Y Index')
            ax1.set_title('All-Sky Camera Radiance Simulation (%s Mode, %d nm, Scene %2.2d)' % (solver, wavelength, self.geoinfo['Nfly']))
            plt.savefig(fname_png, bbox_inches='tight')
            plt.close(fig)
        # ------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    # forward simulation
    # =============================================================================
    # aircraft0 = aircraft()
    # aircraft0.install_camera()
    # aircraft0.initialize_geoinfo()
    # aircraft0.flyover_view()

    # for i in range(20):
    #     aircraft0.fly_to_next()
    #     aircraft0.flyover_view()
    # =============================================================================
