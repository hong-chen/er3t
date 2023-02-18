import os
import sys
import glob
import datetime
import numpy as np
import h5py

import er3t.common
import er3t.rtm.mca as mca


__all__ = [
        'func_ref_vs_cot',
        ]


class func_ref_vs_cot:

    def __init__(self,
            fdir='tmp-data',
            wavelength=650.0,
            surface_albedo=0.03,
            solar_zenith_angle=0.0,
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
        self.ref = np.array([])
        self.ref_std = np.array([])

        self.ref_2s = cal_r_twostream(cot, a=self.alb, mu=np.cos(np.deg2rad(self.sza)))

        if run:
            self.run_all()

        for i in range(self.cot.size):
            cot0 = self.cot[i]

            # read data
            #/----------------------------------------------------------------------------\#
            fname = '%s/mca-out-rad-3d_cot-%05.1f.h5' % (self.fdir, cot0)
            f0 = h5py.File(fname, 'r')
            rad0     = f0['mean/rad'][...].mean()
            rad_std0 = f0['mean/rad_std'][...].mean()
            toa0     = f0['mean/toa'][...]
            # toa00 = (1.383910e+03+7.580696e-13)/1000.0
            # print(toa0)
            # print(toa0*(np.cos(np.deg2rad(_solar_zenith_angle))))
            # print(toa00)

            f0.close()
            #\----------------------------------------------------------------------------/#

            # convert from rad to ref
            #/----------------------------------------------------------------------------\#
            ref0 = np.pi*rad0 / (toa0*np.cos(np.deg2rad(_solar_zenith_angle)))
            ref_std0 = np.pi*rad_std0 / (toa0*np.cos(np.deg2rad(_solar_zenith_angle)))
            # ref0 = np.pi*rad0 / (toa0)*(np.cos(np.deg2rad(_solar_zenith_angle)))
            # ref_std0 = np.pi*rad_std0 / (toa0)*(np.cos(np.deg2rad(_solar_zenith_angle)))
            # ref0 = np.pi*rad0 / (toa00)
            # ref_std0 = np.pi*rad_std0 / (toa00)
            # ref0 = rad0
            # ref_std0 = rad_std0
            #\----------------------------------------------------------------------------/#

            self.ref     = np.append(self.ref, ref0)
            self.ref_std = np.append(self.ref_std, ref_std0)

    def run_all(self):

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(cot0)

    def run_mca_one(self, cot):

        # atm object
        #/----------------------------------------------------------------------------\#
        levels = np.arange(0.0, 20.1, 1.0)
        fname_atm = '%s/atm.pk' % self.fdir
        atm0   = atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod=_fname_atmmod, overwrite=False)
        #\----------------------------------------------------------------------------/#


        # abs object
        #/----------------------------------------------------------------------------\#
        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_16g(wavelength=self.wvl, fname=fname_abs, atm_obj=atm0, overwrite=False)
        #\----------------------------------------------------------------------------/#


        # define cloud
        #/----------------------------------------------------------------------------\#
        fname_les_pk  = '%s/les.pk' % self.fdir
        cld0          = cld_les(fname_nc=_fnames_les[0], fname=fname_les_pk, coarsen=[1, 1, 25], overwrite=False)

        iz = 1
        Nx = 3
        Ny = 3
        Nz = 2
        cot_2d    = np.zeros((Nx, Ny), dtype=np.float64); cot_2d[...] = cot
        # cer_2d    = np.zeros((Nx, Ny), dtype=np.float64); cer_2d[...] = self.cer
        cer_3d    = np.zeros((Nx, Ny, Nz), dtype=np.float64); cer_3d[...] = self.cer
        ext_3d    = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        cld0.lev['altitude']['data']    = cld0.lay['altitude']['data'][iz:iz+Nz+1]

        cld0.lay['x']['data']           = cld0.lay['x']['data'][:Nx]
        cld0.lay['y']['data']           = cld0.lay['y']['data'][:Ny]
        cld0.lay['nx']['data']          = Nx
        cld0.lay['ny']['data']          = Ny
        cld0.lay['altitude']['data']    = cld0.lay['altitude']['data'][iz:iz+Nz]
        cld0.lay['pressure']['data']    = cld0.lay['pressure']['data'][iz:iz+Nz]
        cld0.lay['temperature']['data'] = cld0.lay['temperature']['data'][:Nx, :Ny, iz:iz+Nz]
        cld0.lay['cot']['data']         = cot_2d
        cld0.lay['cer']['data']         = cer_3d
        cld0.lay['thickness']['data']   = cld0.lay['thickness']['data'][iz:iz+Nz]

        # ext_3d[:, :, 0]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        # ext_3d[:, :, 1]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 0]  = cot_2d/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 1]  = cot_2d/(cld0.lay['thickness']['data'].sum()*1000.0)
        cld0.lay['extinction']['data']  = ext_3d
        #\----------------------------------------------------------------------------/#


        # mca_sca object
        #/----------------------------------------------------------------------------\#
        pha0 = pha_mie(wavelength=self.wvl)
        sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % self.fdir, overwrite=True)
        #\----------------------------------------------------------------------------/#


        # mca_cld object
        #/----------------------------------------------------------------------------\#
        atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % self.fdir, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]
        #\----------------------------------------------------------------------------/#


        # run mcarats
        #/----------------------------------------------------------------------------\#
        mca0 = mcarats_ng(
                date=_date,
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
                fdir='%s/%05.1f/rad' % (self.fdir, cot),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=_photon_ipa,
                solver='3D',
                Ncpu=12,
                mp_mode='py',
                overwrite=True
                )
        #\----------------------------------------------------------------------------/#


        # mcarats output
        #/----------------------------------------------------------------------------\#
        out0 = mca_out_ng(fname='%s/mca-out-rad-3d_cot-%05.1f.h5' % (self.fdir, cot), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
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

    pass
