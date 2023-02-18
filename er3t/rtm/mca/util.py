import os
import sys
import glob
import datetime
import numpy as np
from scipy.interpolate import interp1d
import h5py

import er3t.common as common_
import er3t.pre.atm as atm_
import er3t.pre.abs as abs_
import er3t.pre.cld as cld_
import er3t.pre.pha as pha_
import er3t.rtm.mca as rtm_
import er3t.util as util_


__all__ = [
        'func_ref_vs_cot',
        ]


class func_ref_vs_cot:

    def __init__(self,
            fdir='tmp-data',
            wavelength=650.0,
            surface_albedo=0.03,
            solar_zenith_angle=30.0,
            solar_azimuth_angle=0.0,
            sensor_zenith_angle=0.0,
            sensor_azimuth_angle=0.0,
            cot=np.concatenate((
                np.arange(0.0, 1.0, 0.2),
                np.arange(1.0, 10.0, 1.0),
                np.arange(10.0, 30.0, 2.0),
                np.arange(30.0, 51.0, 5.0),
                )),
            cer=20.0,
            run=False,
            ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        self.fdir = fdir
        self.wvl = wavelength
        self.sza = solar_zenith_angle
        self.saa = solar_azimuth_angle
        self.vza = sensor_zenith_angle
        self.vaa = sensor_azimuth_angle
        self.alb = surface_albedo
        self.cot = cot
        self.cer = cer

        self.mu = np.cos(np.deg2rad(self.sza))
        self.ref_2s = util_.cal_r_twostream(cot, a=self.alb, mu=self.mu)

        if run:
            self.run_all()

        self.ref = np.array([])
        self.ref_std = np.array([])
        for i in range(self.cot.size):
            cot0 = self.cot[i]

            name_tag = 'cot-%04.1f_cer-%04.1f' % (cot0, self.cer)

            # read data
            #/----------------------------------------------------------------------------\#
            fname = '%s/mca-out-rad-3d_%s.h5' % (self.fdir, name_tag)
            f0 = h5py.File(fname, 'r')
            rad0     = f0['mean/rad'][...].mean()
            rad_std0 = f0['mean/rad_std'][...].mean()
            toa0     = f0['mean/toa'][...]
            f0.close()
            #\----------------------------------------------------------------------------/#

            # convert from rad to ref
            #/----------------------------------------------------------------------------\#
            ref0 = np.pi*rad0 / (toa0*self.mu)
            ref_std0 = np.pi*rad_std0 / (toa0*self.mu)
            #\----------------------------------------------------------------------------/#

            self.ref     = np.append(self.ref, ref0)
            self.ref_std = np.append(self.ref_std, ref_std0)

    def run_all(self):

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(cot0, self.cer)

    def run_mca_one(self, cot0, cer0):

        name_tag = 'cot-%04.1f_cer-%04.1f' % (cot0, cer0)

        # atm object
        #/----------------------------------------------------------------------------\#
        levels = np.arange(0.0, 20.1, 1.0)
        fname_atm = '%s/atm.pk' % self.fdir
        atm0   = atm_.atm_atmmod(levels=levels, fname=fname_atm, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # abs object
        #/----------------------------------------------------------------------------\#
        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_.abs_16g(wavelength=self.wvl, fname=fname_abs, atm_obj=atm0, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # define cloud
        #/----------------------------------------------------------------------------\#
        fname_cld = '%s/cld_%s.pk' % (self.fdir, name_tag)
        cld0 = cld_.cld_gen_hom(fname=fname_cld, altitude=atm0.lay['altitude']['data'][1:3], atm_obj=atm0, Nx=4, Ny=4, cot0=cot0, cer0=cer0, overwrite=True)
        #\----------------------------------------------------------------------------/#


        # mca_sca object
        #/----------------------------------------------------------------------------\#
        pha0 = pha_.pha_mie_wc(wavelength=self.wvl)

        fname_sca = '%s/mca_sca_%s.bin' % (self.fdir, name_tag)
        sca  = rtm_.mca_sca(pha_obj=pha0, fname=fname_sca, overwrite=True)
        #\----------------------------------------------------------------------------/#


        # mca_cld object
        #/----------------------------------------------------------------------------\#
        atm1d0  = rtm_.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

        fname_atm_3d = '%s/mca_atm_3d_%s.bin' % (self.fdir, name_tag)
        atm3d0  = rtm_.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname=fname_atm_3d, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]
        #\----------------------------------------------------------------------------/#


        # run mcarats
        #/----------------------------------------------------------------------------\#
        mca0 = rtm_.mcarats_ng(
                date=datetime.datetime.now(),
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                surface_albedo=self.alb,
                sca=sca,
                Ng=abs0.Ng,
                target='radiance',
                solar_zenith_angle   = self.sza,
                solar_azimuth_angle  = self.saa,
                sensor_zenith_angle  = self.vza,
                sensor_azimuth_angle = self.vaa,
                fdir='%s/%05.1f/rad' % (self.fdir, cot0),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=1e6,
                solver='3D',
                Ncpu=12,
                mp_mode='py',
                overwrite=True
                )
        #\----------------------------------------------------------------------------/#


        # mcarats output
        #/----------------------------------------------------------------------------\#
        out0 = rtm_.mca_out_ng(fname='%s/mca-out-rad-3d_%s.h5' % (self.fdir, name_tag), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
        #\----------------------------------------------------------------------------/#

    def cot_from_ref(self, ref, method='cubic', mode='rt'):

        if mode == '2s':
            f = interp1d(self.ref_2s, self.cot, kind=method, bounds_error=False)
        elif mode == 'rt':
            f = interp1d(self.ref, self.cot, kind=method, bounds_error=False)

        return f(ref)

    def ref_from_cot(self, cot, method='cubic', mode='rt'):

        if mode == '2s':
            f = interp1d(self.cot, self.ref_2s, kind=method, bounds_error=False)
        elif mode == 'rt':
            f = interp1d(self.cot, self.ref, kind=method, bounds_error=False)

        return f(cot)



if __name__ == '__main__':

    cot = np.array([0.0, 1.0, 5.0, 10.0, 30.0, 50.0, 100.0])

    f0 = func_ref_vs_cot(cot=cot, run=True)
    # sys.exit()

    # f0 = func_ref_vs_cot(cot=cot, run=False)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.path as mpl_path
    import matplotlib.image as mpl_img
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams, ticker
    from matplotlib.ticker import FixedLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # import cartopy.crs as ccrs
    # mpl.use('Agg')

    # figure
    #/----------------------------------------------------------------------------\#
    plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    # fig.suptitle('Figure')
    # plot
    #/--------------------------------------------------------------\#
    ax1 = fig.add_subplot(111)
    # ax1.errorbar(f0.cot, f0.ref, yerr=f0.ref_std, s=6, c='k', lw=0.0)
    ax1.errorbar(f0.cot, f0.ref, yerr=f0.ref_std, c='k', lw=1.0)
    ax1.plot(f0.cot, f0.ref_2s, c='r', lw=1.0)
    # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
    ax1.set_xlim((0.0, 100.0))
    ax1.set_ylim((0.0, 1.0))
    # ax1.set_xlabel('')
    # ax1.set_ylabel('')
    # ax1.set_title('')
    # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
    # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
    #\--------------------------------------------------------------/#
    # add colorbar
    #/--------------------------------------------------------------\#
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', '5%', pad='3%')
    # cbar = fig.colorbar(cs, cax=cax)
    # cbar.set_label('', rotation=270, labelpad=4.0)
    # cbar.set_ticks([])
    # cax.axis('off')
    #\--------------------------------------------------------------/#
    # save figure
    #/--------------------------------------------------------------\#
    # plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    # plt.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
    #\--------------------------------------------------------------/#
    plt.show()
    sys.exit()
    #\----------------------------------------------------------------------------/#


    pass
