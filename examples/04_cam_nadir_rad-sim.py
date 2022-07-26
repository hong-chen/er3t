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




# passed test

def get_cld_rtv_ipa(ref_2d, lon_2d, lat_2d, sza0):

    a0         = 0.03
    mu0        = np.cos(np.deg2rad(sza0))

    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)

    threshold  = a0 * 1.0
    indices    = np.where(ref_2d>threshold)
    indices_x  = indices[0]
    indices_y  = indices[1]
    lon        = lon_2d[indices_x, indices_y]
    lat        = lat_2d[indices_x, indices_y]

    Nx, Ny = ref_2d.shape
    cot_2d_2s = np.zeros_like(ref_2d)
    cer_2d_2s = np.zeros_like(ref_2d); cer_2d_2s[...] = 1.0
    for i in range(indices_x.size):
        cot_2d_2s[indices_x[i], indices_y[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
        cer_2d_2s[indices_x[i], indices_y[i]] = 12.0

    return cot_2d_2s, cer_2d_2s




def predict_64_by_64(rad_3d, model):

    cot_class_l = np.concatenate((np.arange(0.0, 1.0, 0.1),
                               np.arange(1.0, 10.0, 1.0),
                               np.arange(10.0, 20.0, 2.0),
                               np.arange(20.0, 50.0, 5.0),
                               np.arange(50.0, 101.0, 10.0)))

    cot_class_r = np.append(cot_class_l[1:], 200.0)

    cot_class_m = (cot_class_l+cot_class_r)/2.0

    cot_class = cot_class_l.copy()

    Ncot = cot_class.size
    Nx, Ny = rad_3d.shape

    cot_max = np.zeros((Nx, Ny), dtype=np.float64); cot_max[...] = np.nan
    cot_wei = np.zeros((Nx, Ny), dtype=np.float64); cot_wei[...] = np.nan
    prob = np.zeros((Nx, Ny, Ncot), dtype=np.float64); prob[...] = np.nan

    for i in range(0, Nx, 64):
        for j in range(0, Ny, 64):

            index_x_s = i
            index_x_e = i+64
            index_y_s = j
            index_y_e = j+64

            data0 = np.zeros((64, 64), dtype=np.float64)
            if (index_x_e > Nx) or (index_y_e > Ny):
                if index_x_e > Nx:
                    index_x_e = Nx
                if index_y_e > Ny:
                    index_y_e = Ny

            data0[:index_x_e-index_x_s, :index_y_e-index_y_s] = rad_3d[index_x_s:index_x_e, index_y_s:index_y_e]

            prob0 = np.squeeze(model.predict(preprocess(data0)))

            cot_max0 = cot_class[np.argmax(prob0, axis=-1)]
            cot_max[index_x_s:index_x_e, index_y_s:index_y_e] = cot_max0[:index_x_e-index_x_s, :index_y_e-index_y_s]

            cot_wei0 = np.dot(prob0, cot_class)
            cot_wei[index_x_s:index_x_e, index_y_s:index_y_e] = cot_wei0[:index_x_e-index_x_s, :index_y_e-index_y_s]

            prob[index_x_s:index_x_e, index_y_s:index_y_e, :] = prob0[:index_x_e-index_x_s, :index_y_e-index_y_s, :]

    return cot_max, cot_wei, prob, cot_class

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):

    """
    by Vikas Nataraja (vikas.hanasogenataraja@lasp.colorado.edu)
    """

    import keras.backend as K

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    loss = -alpha * K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred)

    return K.mean(loss, axis=-1)

def preprocess(img, resize_dims=None, normalize=False):

    """
    by Vikas Nataraja (vikas.hanasogenataraja@lasp.colorado.edu)
    Pre-processing steps for an image
    """

    if normalize:
        img = standard_normalize(img)
    if resize_dims is not None:
        img = resize_img(img, resize_dims)

    if len(img.shape) > 2:
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    else:
        img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    return img




class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad(date, geometry, cloud, wavelength=600.0, cth=2.0, photons=1e7, fdir='tmp-data/04_cam_nadir_rad-sim', solver='3D', overwrite=True):

    """
    Simulate radiance for camera using IPA/CNN based cloud optical thickness
    """

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # atm object
    # =================================================================================
    levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sca object (enable/disable mie scattering)
    # =================================================================================
    #pha0 = pha_mie(wvl0=wavelength)
    #sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    pha0 = None
    # =================================================================================


    # cld object
    # =================================================================================
    modl1b    =  sat_tmp(cloud)
    fname_cld = '%s/cld.pk' % fdir
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=2.0, cgt=1.0, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================

    # run mcarats
    mca0 = mcarats_ng(
            date=date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface_albedo=0.03,
            #sca=sca, # disable/enable mie scattering
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = geometry['sza']['data'],
            solar_azimuth_angle  = geometry['saa']['data'],
            sensor_altitude      = geometry['alt']['data'],
            fdir='%s/%.4fnm/cam/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=photons,
            solver=solver,
            Ncpu=12,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-cam-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)




def main_pre_ipa():

    # read in raw all-sky camera imagery data
    f = h5py.File('data/04_cam_nadir_rad-sim/aux/cam_nadir_imagery.h5', "r")
    extent = f['gridded/ext'][...]
    red    = f['gridded/red'][...]
    lon    = f['gridded/lon'][...]
    lat    = f['gridded/lat'][...]
    sza0   = f['geometry/sza'][...]
    f.close()

    # estimate reflectance based on red-channel radiance (isotropic assumption)
    f_toa = 1.782035
    ref = np.pi*red/(f_toa*np.cos(np.deg2rad(sza0)))

    # use two-stream approximation to estimate cloud optical thickness from reflectance
    # special note: cloud effective radius is constantly set to 12 micron
    cot_ipa, cer_ipa = get_cld_rtv_ipa(ref, lon, lat, sza0)

    # save pre-processed data
    f = h5py.File('data/04_cam_nadir_rad-sim/pre-data_ipa.h5', 'w')
    f['rad'] = red
    f['ref'] = ref
    f['lon'] = lon
    f['lat'] = lat
    f['extent'] = extent
    g = f.create_group('ipa')
    g['cot'] = cot_ipa
    g['cer'] = cer_ipa
    f.close()

def main_pre_cnn(
        fname='data/04_cam_nadir_rad-sim/pre-data_ipa.h5',
        model_path='/data/vikas/weights/chosen_uniform_fine_4x_16.h5'
        ):

    import tensorflow as tf
    from keras.models import load_model

    f = h5py.File('data/04_cam_nadir_rad-sim/pre-data_cnn.h5', 'w')

    f0 = h5py.File(fname, 'r')
    rad = f0['rad'][...]

    # load cnn model
    model = load_model('{}'.format(model_path), custom_objects={"tf":tf, "focal_loss":focal_loss})

    # predict cloud optical thickness based on radiance
    cot_max, cot_wei, prob, cot_class = predict_64_by_64(rad, model)

    f['lon'] = f0['lon'][...]
    f['lat'] = f0['lat'][...]
    f['rad'] = f0['rad'][...]
    f['extent'] = f0['extent'][...]

    g = f.create_group('cnn')
    g['cot_wei'] = cot_wei
    g['cot_max'] = cot_max
    g['cot_class'] = cot_class
    g['prob'] = prob

    f0.close()

    f.close()

def main_sim():

    # read in solar geometries
    geometry = {}
    f = h5py.File('data/04_cam_nadir_rad-sim/aux/cam_nadir_imagery.h5', 'r')
    geometry['sza'] = dict(name='Solar Zenith Angle'              , units='degree'    , data=f['geometry/sza'][...])
    geometry['saa'] = dict(name='Solar Azimuth Angle'             , units='degree'    , data=f['geometry/saa'][...])
    geometry['alt'] = dict(name='Altitude'                        , units='meter'     , data=f['geometry/altitude'][...])
    f.close()

    # read in IPA based cloud data
    cloud_ipa = {}
    f = h5py.File('data/04_cam_nadir_rad-sim/pre-data_ipa.h5', 'r')
    cot_2d = f['ipa/cot'][...]
    cer_2d = f['ipa/cer'][...]
    cloud_ipa['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['lon'][...])
    cloud_ipa['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['lat'][...])
    cloud_ipa['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d)
    cloud_ipa['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d)
    f.close()

    # read in CNN based cloud data
    cloud_cnn = {}
    if os.path.exists('data/04_cam_nadir_rad-sim/pre-data_cnn.h5'):
        f = h5py.File('data/04_cam_nadir_rad-sim/pre-data_cnn.h5', 'r')
    else:
        f = h5py.File('data/04_cam_nadir_rad-sim/aux/pre-data_cnn.h5', 'r')
    cot_2d = f['cnn/cot_wei'][...]
    cer_2d = np.zeros_like(cot_2d); cer_2d[...] = 12.0
    cloud_cnn['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['lon'][...])
    cloud_cnn['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['lat'][...])
    cloud_cnn['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d)
    cloud_cnn['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d)
    f.close()

    # run simulations using EaR3T
    date = datetime.datetime(2019, 10, 5)
    cal_mca_rad(date, geometry, cloud_ipa, wavelength=600.0, cth=2.0, photons=1e7, fdir='tmp-data/04_cam_nadir_rad-sim/ipa', solver='3D', overwrite=True)
    cal_mca_rad(date, geometry, cloud_cnn, wavelength=600.0, cth=2.0, photons=1e7, fdir='tmp-data/04_cam_nadir_rad-sim/cnn', solver='3D', overwrite=True)

def main_post():
    pass




if __name__ == '__main__':

    # Step 1. Pre-process all-sky camera data
    #    a. convert red channel radiance into reflectance
    #    b. estimate cloud optical thickness (cot) based on reflectance through two-stream approximation
    #    c. store data in <pre-data.h5> under data/04_cam_nadir_rad-sim
    # main_pre_ipa()

    # Step 2. Use CNN to predict cloud optical thickness from camera red channel radiance
    # Special note: to run the following function, tensorflow needs to be installed.
    #               If you have problems in setting up tensorflow enviroment, you can skip this step and
    #               use <pre-data_cnn.h5> under data/04_cam_nadir_rad-sim/aux
    # main_pre_cnn()

    # Step 3. Use EaR3T to run radiance simulations for both cot_ipa and cot_cnn
    #    a. 3D radiance simulation using cot_ipa
    #    b. 3D radiance simulation using cot_cnn
    # main_sim()

    # Step 3. Post-process and plot
    #    a. save data in <post-data.h5> under data/04_cam_nadir_rad-sim
    #        1) raw radiance measurements
    #        2) radiance simulation based on cot_ipa
    #        3) radiance simulation based on cot_cnn
    #    b. plot
    # main_post()




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
