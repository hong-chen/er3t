import os
import sys
import glob
import datetime
import numpy as np
from scipy.interpolate import interp1d
import h5py

import er3t


__all__ = [
        'func_ref_vs_cot',
        ]



class func_ref_vs_cot:

    def __init__(self,
            cot,
            cer0=20.0,
            fdir=er3t.common.params['fdir_tmp'],
            wavelength=er3t.common.params['wavelength'],
            surface_albedo=er3t.common.params['surface_albedo'],
            solar_zenith_angle=er3t.common.params['solar_zenith_angle'],
            solar_azimuth_angle=er3t.common.params['solar_azimuth_angle'],
            sensor_zenith_angle=er3t.common.params['sensor_zenith_angle'],
            sensor_azimuth_angle=er3t.common.params['sensor_azimuth_angle'],
            output_tag=er3t.common.params['output_tag'],
            overwrite=er3t.common.params['overwrite'],
            ):


        self.cot  = cot
        self.cer0 = cer0
        self.wvl0 = wavelength
        self.sza0 = solar_zenith_angle
        self.saa0 = solar_azimuth_angle
        self.vza0 = sensor_zenith_angle
        self.vaa0 = sensor_azimuth_angle
        self.alb0 = surface_albedo
        self.fdir = fdir
        self.output_tag = output_tag

        self.mu0  = np.cos(np.deg2rad(self.sza0))
        self.ref_2s = er3t.util.cal_r_twostream(cot, a=self.alb0, mu=self.mu0)

        if not overwrite:
            try:
                self.load_all()
            except:
                self.run_all()
                self.load_all()
        else:
            self.run_all()
            self.load_all()

    def load_all(self):

        self.rad     = np.array([])
        self.rad_std = np.array([])
        for i in range(self.cot.size):
            name_tag = 'cot-%05.1f_cer-%04.1f' % (self.cot[i], self.cer0)

            # read data
            #/----------------------------------------------------------------------------\#
            fname = '%s/%s_%s.h5' % (self.fdir, self.output_tag, name_tag)
            f0 = h5py.File(fname, 'r')
            rad0     = f0['mean/rad'][...].mean()
            rad_std0 = f0['mean/rad_std'][...].mean()
            toa0     = f0['mean/toa'][...]
            f0.close()
            #\----------------------------------------------------------------------------/#

            self.rad     = np.append(self.rad, rad0)
            self.rad_std = np.append(self.rad_std, rad_std0)

        # convert from rad to ref
        #/----------------------------------------------------------------------------\#
        self.ref     = np.pi*self.rad      / (toa0*self.mu0)
        self.ref_std = np.pi*self.rad_std  / (toa0*self.mu0)
        #\----------------------------------------------------------------------------/#

    def run_all(self):

        os.system('rm -rf %s' % self.fdir)
        os.makedirs(self.fdir)

        for cot0 in self.cot:
            self.run_one(cot0, self.cer0)

    def run_one(self, cot0, cer0, Nx=10, Ny=10):

        name_tag = 'cot-%05.1f_cer-%04.1f' % (cot0, cer0)

        # atm object
        #/----------------------------------------------------------------------------\#
        levels = np.arange(0.0, 20.1, 1.0)
        fname_atm = '%s/atm.pk' % self.fdir
        atm0   = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # abs object
        #/----------------------------------------------------------------------------\#
        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = er3t.pre.abs.abs_16g(wavelength=self.wvl0, fname=fname_abs, atm_obj=atm0, overwrite=False)
        #\----------------------------------------------------------------------------/#

        # cloud object
        #/----------------------------------------------------------------------------\#
        fname_cld = '%s/cld_%s.pk' % (self.fdir, name_tag)
        cld0 = er3t.pre.cld.cld_gen_hom(fname=fname_cld, altitude=atm0.lay['altitude']['data'][1:3], atm_obj=atm0, Nx=Nx, Ny=Ny, cot0=cot0, cer0=cer0, overwrite=True)
        #\----------------------------------------------------------------------------/#

        # phase function
        #/----------------------------------------------------------------------------\#
        pha0 = er3t.pre.pha.pha_mie_wc(wavelength=self.wvl0)
        #\----------------------------------------------------------------------------/#

        # mca_sca object
        #/----------------------------------------------------------------------------\#
        fname_sca = '%s/mca_sca_%s.bin' % (self.fdir, name_tag)
        sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname=fname_sca, overwrite=True)
        #\----------------------------------------------------------------------------/#

        # mca_cld object
        #/----------------------------------------------------------------------------\#
        atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

        fname_atm_3d = '%s/mca_atm_3d_%s.bin' % (self.fdir, name_tag)
        atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname=fname_atm_3d, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]
        #\----------------------------------------------------------------------------/#

        # run mcarats
        #/----------------------------------------------------------------------------\#
        mca0 = er3t.rtm.mca.mcarats_ng(
                date=datetime.datetime.now(),
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                sca=sca,
                target='radiance',
                surface_albedo       = self.alb0,
                solar_zenith_angle   = self.sza0,
                solar_azimuth_angle  = self.saa0,
                sensor_zenith_angle  = self.vza0,
                sensor_azimuth_angle = self.vaa0,
                fdir='%s/%s_%s/rad' % (self.fdir, self.output_tag, name_tag),
                Nrun=3,
                Ng=abs0.Ng,
                weights=abs0.coef['weight']['data'],
                photons=1e7,
                solver='3D',
                Ncpu=12,
                mp_mode='py',
                overwrite=True
                )
        #\----------------------------------------------------------------------------/#


        # mcarats output
        #/----------------------------------------------------------------------------\#
        out0 = er3t.rtm.mca.mca_out_ng(fname='%s/%s_%s.h5' % (self.fdir, self.output_tag, name_tag), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
        #\----------------------------------------------------------------------------/#

    def get_cot_from_ref(self, ref, method='cubic', mode='rt'):

        if mode == '2s':
            f = interp1d(self.ref_2s, self.cot, kind=method, bounds_error=False)
        elif mode == 'rt':
            f = interp1d(self.ref, self.cot, kind=method, bounds_error=False)

        return f(ref)

    def get_ref_from_cot(self, cot, method='cubic', mode='rt'):

        if mode == '2s':
            f = interp1d(self.cot, self.ref_2s, kind=method, bounds_error=False)
        elif mode == 'rt':
            f = interp1d(self.cot, self.ref, kind=method, bounds_error=False)

        return f(cot)



if __name__ == '__main__':

    pass
