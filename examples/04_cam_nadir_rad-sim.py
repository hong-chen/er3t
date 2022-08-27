"""
by Hong Chen (hong.chen.cu@gmail.com)

This code serves as an example code to reproduce 3D radiance simulations for App. 4 in Chen et al. (2022).

The processes include:
    1) `main_pre_ipa()`: pre-process all-sky camera data
        a) read in camera radiance observations at red channel
        b) convert radiance into reflectance and perform IPA method (two-stream approximation) to retrieve
           cloud optical thickness (cot_ipa)

    2) `main_pre_cnn()`: use CNN model to predict cloud optical thickness (cot_cnn) based on camera measured radiance
        a) load CNN model
        b) predict cot_cnn

    3) `main_sim()`: use EaR3T to run 3D radiance simulations based on
        a) cot_ipa
        b) cot_cnn

    4) `main_post()`: post-process data and plot
        a) read in
            i) camera observed radiance
            j) 3D radiance simulation based on cot_ipa
            k) 3D radiance simulation based on cot_cnn
        b) plot

This code has been tested under:
    1) Linux on 2022-07-26 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64

"""

import os
import h5py
import numpy as np
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_sat, cld_les
from er3t.util import cal_r_twostream

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng




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
        fname = 'data/04_cam_nadir_rad-sim/pre-data_cnn.h5'
    else:
        fname = 'data/04_cam_nadir_rad-sim/aux/pre-data_cnn.h5'
    f = h5py.File(fname, 'r')
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

def main_post(plot=True):

    # read in camera measured red channel radiance
    f = h5py.File('data/04_cam_nadir_rad-sim/pre-data_ipa.h5', 'r')
    extent = f['extent'][...]
    rad_cam = f['rad'][...]
    f.close()

    # read in simulated 3D radiance based on cot_ipa
    f = h5py.File('tmp-data/04_cam_nadir_rad-sim/ipa/mca-out-rad-cam-3d_600.0000nm.h5', 'r')
    rad_sim_ipa = f['mean/rad'][...]
    f.close()

    # read in simulated 3D radiance based on cot_cnn
    f = h5py.File('tmp-data/04_cam_nadir_rad-sim/cnn/mca-out-rad-cam-3d_600.0000nm.h5', 'r')
    rad_sim_cnn = f['mean/rad'][...]
    f.close()

    # save data into <post-data.h5> under data/04_cam_nadir_rad-sim
    f = h5py.File('data/04_cam_nadir_rad-sim/post-data.h5', 'w')
    f['extent'] = extent
    f['rad_cam'] = rad_cam
    f['rad_sim-3d_cot-ipa'] = rad_sim_ipa
    f['rad_sim-3d_cot-cnn'] = rad_sim_cnn
    f.close()

    if plot:

        # =============================================================================
        fig = plt.figure(figsize=(13, 8))
        ax1 = fig.add_subplot(231)
        ax1.imshow(rad_cam.T, extent=extent, origin='lower', cmap='Greys_r', vmin=0.0, vmax=0.6, alpha=0.3)
        rad_cam0 = rad_cam.copy()
        rad_cam0[-7:, :] = np.nan
        rad_cam0[:, -7:] = np.nan
        rad_cam0[:7, :] = np.nan
        rad_cam0[:, :7] = np.nan
        ax1.imshow(rad_cam0.T, extent=extent, origin='lower', cmap='jet', vmin=0.0, vmax=0.6, alpha=1.0)
        ax1.set_xlabel('X [km]')
        ax1.set_ylabel('Y [km]')
        ax1.set_title('Radiance Observation (Red)')

        ax2 = fig.add_subplot(232)
        ax2.imshow(rad_sim_ipa.T, extent=extent, origin='lower', cmap='Greys_r', vmin=0.0, vmax=0.6, alpha=0.3)
        rad_sim_ipa0 = rad_sim_ipa.copy()
        rad_sim_ipa0[-7:, :] = np.nan
        rad_sim_ipa0[:, -7:] = np.nan
        rad_sim_ipa0[:7, :] = np.nan
        rad_sim_ipa0[:, :7] = np.nan
        ax2.imshow(rad_sim_ipa0.T, extent=extent, origin='lower', cmap='jet', vmin=0.0, vmax=0.6, alpha=1.0)
        ax2.set_xlabel('X [km]')
        ax2.set_ylabel('Y [km]')
        ax2.set_title('3D Radiance Simulation (COT$\mathrm{_{IPA}}$)')

        ax3 = fig.add_subplot(233)
        xedges = np.arange(0.0, 0.61, 0.03)
        yedges = np.arange(0.0, 0.61, 0.03)
        heatmap, xedges, yedges = np.histogram2d(rad_cam0.ravel(), rad_sim_ipa0.ravel(), bins=(xedges, yedges))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)
        levels = np.power(10, np.linspace(0.0, 1.5, 151))

        cs = ax3.contourf(XX, YY, heatmap, levels, extend='both')
        ax3.plot([0.0, 1.0], [0.0, 1.0], lw=1.5, color='gray', ls='--', zorder=2)
        ax3.set_xlim([0.0, 0.6])
        ax3.set_ylim([0.0, 0.6])
        ax3.set_xlabel('Radiance Observation')
        ax3.set_ylabel('Radiance Simulation (COT$\mathrm{_{IPA}}$)')

        ax4 = fig.add_subplot(235)
        ax4.imshow(rad_sim_cnn.T, extent=extent, origin='lower', cmap='Greys_r', vmin=0.0, vmax=0.6, alpha=0.3)
        rad_sim_cnn0 = rad_sim_cnn.copy()
        rad_sim_cnn0[-7:, :] = np.nan
        rad_sim_cnn0[:, -7:] = np.nan
        rad_sim_cnn0[:7, :] = np.nan
        rad_sim_cnn0[:, :7] = np.nan
        ax4.imshow(rad_sim_cnn0.T, extent=extent, origin='lower', cmap='jet', vmin=0.0, vmax=0.6, alpha=1.0)
        ax4.set_xlabel('X [km]')
        ax4.set_ylabel('Y [km]')
        ax4.set_title('3D Radiance Simulation (COT$\mathrm{_{CNN}}$)')

        ax5 = fig.add_subplot(236)
        xedges = np.arange(0.0, 0.61, 0.03)
        yedges = np.arange(0.0, 0.61, 0.03)
        heatmap, xedges, yedges = np.histogram2d(rad_cam0.ravel(), rad_sim_cnn0.ravel(), bins=(xedges, yedges))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)
        levels = np.power(10, np.linspace(0.0, 1.5, 151))

        cs = ax5.contourf(XX, YY, heatmap, levels, extend='both')
        ax5.plot([0.0, 1.0], [0.0, 1.0], lw=1.5, color='gray', ls='--', zorder=2)
        ax5.set_xlim([0.0, 0.6])
        ax5.set_ylim([0.0, 0.6])
        ax5.set_xlabel('Radiance Observation')
        ax5.set_ylabel('Radiance Simulation (COT$\mathrm{_{CNN}}$)')

        plt.subplots_adjust(hspace=0.45, wspace=0.45)

        plt.savefig('04_cam_nadir_rad-sim.png', bbox_inches='tight')
        plt.close(fig)
        # =============================================================================




if __name__ == '__main__':

    # Step 1. Pre-process all-sky camera data
    #    a. convert red channel radiance into reflectance
    #    b. estimate cloud optical thickness (cot) based on reflectance through two-stream approximation
    #    c. store data in <pre-data.h5> under data/04_cam_nadir_rad-sim
    # =============================================================================
    # main_pre_ipa()
    # =============================================================================

    # Step 2. Use CNN to predict cloud optical thickness from camera red channel radiance
    # Special note: to run the following function, tensorflow needs to be installed.
    #               If you have problems in setting up tensorflow enviroment, you can skip this step and
    #               use <pre-data_cnn.h5> provided under data/04_cam_nadir_rad-sim/aux instead.
    #               CNN model credit: Nataraja et al. 2022 (https://doi.org/10.5194/amt-2022-45)
    # =============================================================================
    # main_pre_cnn()
    # =============================================================================

    # Step 3. Use EaR3T to run radiance simulations for both cot_ipa and cot_cnn
    #    a. 3D radiance simulation using cot_ipa
    #    b. 3D radiance simulation using cot_cnn
    # =============================================================================
    # main_sim()
    # =============================================================================

    # Step 4. Post-process and plot
    #    a. save data in <post-data.h5> under data/04_cam_nadir_rad-sim
    #        1) raw camera radiance measurements
    #        2) radiance simulation based on cot_ipa
    #        3) radiance simulation based on cot_cnn
    #    b. plot
    # =============================================================================
    # main_post()
    # =============================================================================

    pass
