import os
import sys
import copy
import time
import datetime
import warnings
import multiprocessing as mp
import numpy as np

import er3t.common
import er3t.util
from er3t.rtm.mca import mca_inp_file
from er3t.rtm.mca import mca_run



__all__ = ['mcarats_ng', 'cal_mca_azimuth']



class mcarats_ng:

    """
    Input:

        atm_1ds=: Python list, contains object of atm_1d
        atm_3ds=: Python list, contains object of atm_3d

        Ng=    : integer, number of gs, e.g., for abs_16g, Ng=16
        target=: string type, can be one of 'flux', 'radiance', and 'heating rate'
        fdir=  : string type, this will be a directory to store input/output files

        date=  : keyword argument, datetime.datetime object, the date to calculate sun-earth distance
        target=: keyword argument, string, can be 'flux', 'radiance', default='flux'
        surface_albedo=     : keyword argument, float, surface albedo, default=0.03
        solar_zenith_angle= : keyword argument, float, solar zenith angle, default=30.0
        solar_azimuth_angle=: keyword argument, float, solar azimuth angle, default=0.0

        Nrun=     : keyword argument, integer, number of runs to calculate mean/std statistics, default=1

        solver=   : keyword argument, integer, 0:3d mode, 1:partial-3d mode, 2:ipa mode, default=0
        photons=  : keyword argument, integer, number of photons, default=1e6
        Ncpu=     : keyword argument, integer, number of CPUs to use, default=1
        tune=     : keyword argument, boolen, whether to tune the MCARaTS calculation, default=False
        overwrite=: keyword argument, boolen, whether to overwrite existing MCARaTS output files (rerun MCARaTS), default=True
        np_mode=: keyword argument, string, photon distribution, can be 'even', 'weight', default='even'
        mp_mode=: keyword argument, string, multiprocessing mode, can be 'py', 'sh', 'mpi', default='py'

        fdir=   : keyword argument, string, path to store MCARaTS input and output files
        comment=: keyword argument, boolen, whether to add comment in MCARaTS input file, default=False
        verbose=: keyword argument, boolen, verbose tag, default=False
        quiet=  : keyword argument, boolen, verbose tag, default=False

    Output:

        MCARaTS output files created under path specified by 'fdir'
    """

    reference = '\nMCARaTS (Iwabuchi, 2006; Iwabuchi and Okamura, 2017):\n- Iwabuchi, H.: Efficient Monte Carlo methods for radiative transfer modeling, J. Atmos. Sci., 63, 2324-2339, https://doi.org/10.1175/JAS3755.1, 2006.\n- Iwabuchi, H., and Okamura, R.: Multispectral Monte Carlo radiative transfer simulation by using the maximum cross-section method, Journal of Quantitative Spectroscopy and Radiative Transfer, 193, 40-46, https://doi.org/10.1016/j.jqsrt.2017.01.025, 2017.'


    def __init__(self,                                          \

                 atm_1ds             = [],                      \
                 atm_3ds             = [],                      \

                 sca                 = None,                    \

                 Ng                  = 16,                      \
                 weights             = None,                    \

                 fdir                = 'tmp-data/sim',          \
                 Nrun                = 3,                       \
                 Ncpu                = 'auto',                  \
                 mp_mode             = 'py',                    \
                 overwrite           = True,                    \

                 date                = datetime.datetime.now(), \
                 comment             = False,                   \
                 tune                = False,                   \
                 target              = 'flux',                  \
                 surface_albedo      = 0.03,                    \
                 solar_zenith_angle  = 30.0,                    \
                 solar_azimuth_angle = 0.0,                     \

                 sensor_zenith_angle = 0.0,                     \
                 sensor_azimuth_angle= 0.0,                     \
                 sensor_altitude     = 705000.0,                \
                 sensor_type         = 'satellite',             \
                 sensor_xpos         = 0.5,                     \
                 sensor_ypos         = 0.5,                     \

                 solver              = '3d',                    \
                 photons             = 1e7,                     \
                 base_ratio          = 0.05,                    \

                 verbose             = False,                   \
                 quiet               = False                    \
                 ):

        er3t.util.add_reference(self.reference)

        fdir = os.path.abspath(fdir)

        if not os.path.exists(fdir):
            os.makedirs(fdir)
            if not quiet:
                print('Message [mcarats_ng]: Directory <%s> is created.' % fdir)
        else:
            if verbose:
                print('Message [mcarats_ng]: Directory <%s> already exists.' % fdir)

        self.Ng      = Ng
        self.date    = date
        self.fdir    = fdir
        self.verbose = verbose
        self.quiet   = quiet
        self.overwrite = overwrite
        self.mp_mode   = mp_mode.lower()

        self.sca                 = sca

        self.surface_albedo      = surface_albedo
        self.solar_zenith_angle  = solar_zenith_angle
        self.solar_azimuth_angle = solar_azimuth_angle

        self.sensor_zenith_angle = sensor_zenith_angle
        self.sensor_azimuth_angle= sensor_azimuth_angle
        self.sensor_altitude     = sensor_altitude
        self.sensor_type         = sensor_type
        self.sensor_xpos         = sensor_xpos
        self.sensor_ypos         = sensor_ypos

        self.Nrun    = Nrun

        solver = solver.lower()
        if solver in ['3d', '3 d', 'three d']:
            self.solver = '3D'
        elif solver in ['p3d', 'p-3d', 'partial 3d', 'partial-3d']:
            self.solver = 'Partial 3D'
        elif solver in ['ipa', 'independent pixel approximation']:
            self.solver = 'IPA'
        else:
            msg = 'Error [mcarats_ng]: Cannot understand <solver=%s>.' % self.solver
            raise OSError(msg)

        self.target  = target

        # Nx, Ny
        #/----------------------------------------------------------------------------\#
        if len(atm_3ds) > 0:
            self.Nx = atm_3ds[0].nml['Atm_nx']['data']
            self.Ny = atm_3ds[0].nml['Atm_ny']['data']
        else:
            self.Nx = 1
            self.Ny = 1
        #\----------------------------------------------------------------------------/#

        # photon distribution over gs of correlated-k
        #/----------------------------------------------------------------------------\#
        if weights is None:
            self.np_mode = 'evenly'
            weights = np.repeat(1.0/self.Ng, Ng)
        else:
            self.np_mode = 'weighted'

        photons_dist = distribute_photon(photons, weights, base_ratio=base_ratio)

        self.photons = np.tile(photons_dist, Nrun)
        self.photons_per_set = photons_dist.sum()
        #\----------------------------------------------------------------------------/#

        # Determine how many CPUs to utilize
        #/----------------------------------------------------------------------------\#
        Ncpu_total = mp.cpu_count()
        self.Ncpu_total = Ncpu_total
        if Ncpu == 'auto':
            self.Ncpu = Ncpu_total - 1
        elif Ncpu > 1:
            if Ncpu > Ncpu_total:
                self.Ncpu = Ncpu_total
            else:
                self.Ncpu = Ncpu
        else:
            msg = 'Error [mcarats_ng]: Cannot understand <Ncpu=%s>.' % Ncpu
            raise OSError(msg)
        #\----------------------------------------------------------------------------/#

        # in file names, 'r' indicates #run, 'g' indicates #g, index of 'r' and 'g' both start from 0
        # self.fnames_inp/self.fnames_out is a list embeded with lists
        # E.g, in order to get the input file name of the 1st g in 1st run, self.fnames_inp[0][0]
        self.fnames_inp = []
        self.fnames_out = []
        for ir in range(self.Nrun):
            self.fnames_inp.append(['%s/r%2.2d.g%3.3d.inp.txt' % (self.fdir, ir, ig) for ig in range(self.Ng)])
            self.fnames_out.append(['%s/r%2.2d.g%3.3d.out.bin' % (self.fdir, ir, ig) for ig in range(self.Ng)])

        if not self.quiet and not self.overwrite:
            print('Message [mcarats_ng]: Reading mode ...')

        if overwrite:

            # initialize namelist (list contains 16 Python dictionaries)
            self.nml = [{} for ig in range(self.Ng)]

            # MCARaTS wld initialization
            self.init_wld(verbose=verbose, tune=tune,
                sensor_zenith_angle=sensor_zenith_angle, sensor_azimuth_angle=sensor_azimuth_angle, \
                sensor_type=sensor_type, sensor_altitude=sensor_altitude, sensor_xpos=sensor_xpos, sensor_ypos=sensor_ypos)

            # MCARaTS scattering initialization
            self.init_sca(sca=sca)

            # MCARaTS atm initialization
            self.init_atm(atm_1ds=atm_1ds, atm_3ds=atm_3ds)

            # MCARaTS surface initialization
            self.init_sfc(surface_albedo=surface_albedo)

            # MCARaTS source (e.g., solar) initialization
            self.init_src(solar_zenith_angle=solar_zenith_angle, solar_azimuth_angle=solar_azimuth_angle)

            # Create MCARaTS input files (ASCII)
            self.gen_mca_inp(comment=comment)

            # Run MCARaTS to get output files (Binary)
            self.gen_mca_out()


        if self.mp_mode not in ['batch', 'shell', 'bash', 'hpc', 'sh']:
            self.run_check()


    def init_wld(self, tune=False, verbose=False, \
        sensor_zenith_angle=0.0, sensor_azimuth_angle=0.0, \
        sensor_type='satellite', sensor_altitude=705000.0, sensor_xpos=0.5, sensor_ypos=0.5):

        if self.target.lower() in ['f', 'flux', 'irradiance']:
            self.target = 'flux'
        elif self.target.lower() in ['f0', 'flux0', 'irradiance0']:
            self.target = 'flux0'
        elif self.target.lower() in ['heating rate', 'hr']:
            self.target = 'heating rate'
        elif self.target.lower() in ['radiance', 'rad']:
            self.target = 'radiance'
        else:
            msg = 'Error [mcarats_ng]: Cannot understand <target=%s>.' % self.target
            raise OSError(msg)

        for ig in range(self.Ng):

            if verbose:
                self.nml[ig]['Wld_mverb'] = 3
            else:
                self.nml[ig]['Wld_mverb'] = 0

            if tune:
                self.nml[ig]['Wld_moptim'] = 2
            else:
                self.nml[ig]['Wld_moptim'] = 0


            self.nml[ig]['Wld_mbswap'] = 0
            self.nml[ig]['Wld_njob']   = 1


            if self.target == 'flux' :

                self.nml[ig]['Wld_mtarget'] = 1
                self.nml[ig]['Flx_mflx']    = 3
                self.nml[ig]['Flx_mhrt']    = 0

            elif self.target == 'flux0' :

                self.nml[ig]['Wld_mtarget'] = 1
                self.nml[ig]['Flx_mflx']    = 1
                self.nml[ig]['Flx_mhrt']    = 0

            elif self.target == 'heating rate':

                self.nml[ig]['Wld_mtarget'] = 1
                self.nml[ig]['Flx_mflx']    = 3
                self.nml[ig]['Flx_mhrt']    = 1

            elif self.target == 'radiance':

                self.nml[ig]['Wld_mtarget'] = 2

                if 'satellite' in sensor_type.lower():
                    self.nml[ig]['Rad_mrkind']  = 2
                elif 'all-sky' in sensor_type.lower():
                    self.nml[ig]['Rad_mrkind'] = 1
                    self.nml[ig]['Rad_qmax']   = 178.0
                    self.nml[ig]['Rad_apsize'] = 0.05
                    self.nml[ig]['Rad_xpos'] = sensor_xpos
                    self.nml[ig]['Rad_ypos'] = sensor_ypos


                self.nml[ig]['Rad_mplen']   = 0
                self.nml[ig]['Rad_mpmap']   = 1
                self.nml[ig]['Rad_nrad']    = 1

                self.nml[ig]['Rad_difr0']   = 7.5
                self.nml[ig]['Rad_difr1']   = 0.0025
                self.nml[ig]['Rad_the']     = 180.0 - sensor_zenith_angle
                self.nml[ig]['Rad_phi']     = cal_mca_azimuth(sensor_azimuth_angle)
                self.nml[ig]['Rad_zloc']    = sensor_altitude

            else:
                msg = 'Error [mcarats_ng]: Cannot understand <target=%s>.' % self.target
                raise OSError(msg)


    def init_sca(self, sca=None):

        for ig in range(self.Ng):
            # this parameter is important, otherwise random memory segmentation error will occur
            if sca is None:
                self.nml[ig]['Sca_npf'] = 0
            else:
                for key in sca.nml.keys():
                    if os.path.exists(sca.nml['Sca_inpfile']['data']):
                        sca.nml['Sca_inpfile']['data'] = os.path.relpath(sca.nml['Sca_inpfile']['data'], start=self.fdir)
                    self.nml[ig][key] = sca.nml[key]['data']


    def init_atm(self, atm_1ds=[], atm_3ds=[]):

        for ig in range(self.Ng):

            if len(atm_1ds) == 0:
                msg = 'Error [mcarats_ng]: need <atm_1ds> to proceed.'
                raise OSError(msg)
            else:

                for i, atm_1d in enumerate(atm_1ds):

                    for key in atm_1d.nml[ig].keys():

                        self.nml[ig][key] = atm_1d.nml[ig][key]['data']

                self.wvl_info = atm_1d.wvl_info

            if len(atm_3ds) > 0:

                for atm_3d in atm_3ds:

                    for key in atm_3d.nml.keys():

                        if key not in ['Atm_tmpa3d', 'Atm_abst3d', 'Atm_extp3d', 'Atm_omgp3d', 'Atm_apfp3d']:
                            if os.path.exists(atm_3d.nml['Atm_inpfile']['data']):
                                atm_3d.nml['Atm_inpfile']['data'] = os.path.relpath(atm_3d.nml['Atm_inpfile']['data'], start=self.fdir)
                            self.nml[ig][key] = atm_3d.nml[key]['data']

                self.Nx = atm_3d.nml['Atm_nx']['data']
                self.Ny = atm_3d.nml['Atm_ny']['data']
                self.dx = atm_3d.nml['Atm_dx']['data']
                self.dy = atm_3d.nml['Atm_dy']['data']

                if self.target == 'radiance':
                    if 'satellite' in self.sensor_type.lower():
                        if 'Atm_nx' in atm_3d.nml.keys() and 'Atm_ny' in atm_3d.nml.keys():
                            self.nml[ig]['Rad_nxr'] = atm_3d.nml['Atm_nx']['data']
                            self.nml[ig]['Rad_nyr'] = atm_3d.nml['Atm_ny']['data']
                        else:
                            self.nml[ig]['Rad_nxr'] = 1
                            self.nml[ig]['Rad_nyr'] = 1

                    elif 'all-sky' in self.sensor_type.lower():
                        self.nml[ig]['Rad_nxr'] = 500
                        self.nml[ig]['Rad_nyr'] = 500


    def init_src(self, solar_zenith_angle=0.0, solar_azimuth_angle=0.0):

        for ig in range(self.Ng):
            self.nml[ig]['Src_flx']   = 1.0
            self.nml[ig]['Src_qmax']  = 0.533133
            self.nml[ig]['Src_dwlen'] = 0.0
            self.nml[ig]['Src_mtype'] = 1
            self.nml[ig]['Src_mphi']  = 0
            self.nml[ig]['Src_the']   = 180.0 - solar_zenith_angle
            self.nml[ig]['Src_phi']   = cal_mca_azimuth(solar_azimuth_angle)


    def init_sfc(self, surface_albedo=0.03):

        for ig in range(self.Ng):

            if self.verbose:
                print('Message [mcarats_ng]: Assume Lambertian surface ...')

            if isinstance(surface_albedo, float):

                self.nml[ig]['Sfc_mbrdf'] = np.array([1, 0, 0, 0])
                self.nml[ig]['Sfc_mtype'] = 1
                self.nml[ig]['Sfc_param(1)'] = surface_albedo

                self.sfc_2d = False

            elif isinstance(surface_albedo, er3t.rtm.mca.mca_sfc_2d):

                for key in surface_albedo.nml.keys():
                    if '2d' not in key:
                        if os.path.exists(surface_albedo.nml['Sfc_inpfile']['data']):
                            surface_albedo.nml['Sfc_inpfile']['data'] = os.path.relpath(surface_albedo.nml['Sfc_inpfile']['data'], start=self.fdir)
                        self.nml[ig][key] = surface_albedo.nml[key]['data']

                self.sfc_2d = True

            else:

                msg = '\nError [mcarats_ng]: Cannot ingest <surface_albedo>.'
                raise ValueError(msg)


    def gen_mca_inp(self, comment=False):

        """
        Generate MCARaTS input files from namelists

        Input:
            fnames : positional argument, file name of the input files to be created
            nmls   : positional argument, Python dictionaries that contains MCARaTS namelist parameters
            comment: keyword argument, default=False, whether to add comments for each MCARaTS parameter in the input file

        Output:
            text files specified by fnames
        """

        # create input files for MCARaTS
        Nseed = int(time.time())
        rands = np.arange(self.Nrun*self.Ng).reshape((self.Nrun, self.Ng))
        np.random.shuffle(rands)
        for ir in range(self.Nrun):
            for ig in range(self.Ng):
                self.nml[ig]['Wld_jseed'] = Nseed + rands[ir, ig]
                mca_inp_file(self.fnames_inp[ir][ig], self.nml[ig], comment=comment)

        if not self.quiet:
            print('Message [mcarats_ng]: Created MCARaTS input files under <%s>.' % self.fdir)


    def gen_mca_out(self):

        """
        Run MCARaTS to get MCARaTS output files
        """

        # solver:
        #   0: Full 3D radiative transfer
        #   1: Partial 3D radiative transfer
        #   2: Independent Column Approximation
        solvers = {'3D':0, 'Partial 3D':1, 'IPA':2}

        fnames_inp = []
        fnames_out = []
        for ir in range(self.Nrun):
            fnames_inp += self.fnames_inp[ir]
            fnames_out += self.fnames_out[ir]

        if not self.quiet:
            print('Message [mcarats_ng]: Running MCARaTS to get output files under <%s> ...' % self.fdir)

        if not self.quiet:
            self.print_info()

        run0 = mca_run(fnames_inp, fnames_out, photons=self.photons, solver=solvers[self.solver], Ncpu=self.Ncpu, verbose=self.verbose, quiet=self.quiet, mp_mode=self.mp_mode)


    def run_check(self):

        check = []
        for ir in range(self.Nrun):
            for ig in range(self.Ng):
                fname = self.fnames_out[ir][ig]
                if not os.path.exists(fname):
                    check.append(False)
                else:
                    check.append(True)
        if not all(check):
            msg = 'Error [mcarats_ng]: Missing some output files.'
            raise OSError(msg)


    def print_info(self):

        print('----------------------------------------------------------')
        print('                 General Information                      ')
        print('               Simulation : %s %s' % (self.solver, self.target.title()))
        print('               Wavelength : %s' % (self.wvl_info))

        print('               Date (DOY) : %s (%d)' % (self.date.strftime('%Y-%m-%d'), self.date.timetuple().tm_yday))

        print('       Solar Zenith Angle : %.4f° (0 at local zenith)' % self.solar_zenith_angle)
        print('      Solar Azimuth Angle : %.4f° (0 at north; 90° at east)' % self.solar_azimuth_angle)

        if self.target == 'radiance':
            if self.sensor_zenith_angle < 90.0:
                print('      Sensor Zenith Angle : %.4f° (looking down, 0 straight down)' % self.sensor_zenith_angle)
            else:
                print('      Sensor Zenith Angle : %.4f° (looking up, 180° straight up)' % self.sensor_zenith_angle)
            print('     Sensor Azimuth Angle : %.4f° (0 at north; 90° at east)' % self.sensor_azimuth_angle)
            print('          Sensor Altitude : %.1f km' % (self.sensor_altitude/1000.0))

        if not self.sfc_2d:
            print('           Surface Albedo : %.2f' % self.surface_albedo)
        else:
            print('           Surface Albedo : 2D domain')

        if self.sca is None:
            print('           Phase Function : Henyey-Greenstein')
        else:
            print('           Phase Function : %s' % self.sca.pha.ID)

        if (self.Nx > 1) | (self.Ny > 1):
            print('     Domain Size (Nx, Ny) : (%d, %d)' % (self.Nx, self.Ny))
            print('      Pixel Res. (dx, dy) : (%.2f km, %.2f km)' % (self.dx/1000.0, self.dy/1000.0))

        print('  Number of Photons / Set : %.1e (%s over %d g)' % (self.photons_per_set, self.np_mode, self.Ng))
        print('           Number of Runs : %s (g) * %d (set)' % (self.Ng, self.Nrun))
        print('           Number of CPUs : %d (used) of %d (total)' % (self.Ncpu, self.Ncpu_total))
        print('----------------------------------------------------------')



def cal_mca_azimuth(normal_azimuth_angle):

    """
    Convert normal azimuth angle (0 pointing north, positive when clockwise) to photon azimuth in MCARaTS

    Input:
        normal_azimuth_angle: float/integer, normal azimuth angle (0 pointing north, positive when clockwise)

    Output:
        MCARaTS azimuth angle (0 sun shining from west, positive when counterclockwise)
    """

    while normal_azimuth_angle < 0.0:
        normal_azimuth_angle += 360.0

    while normal_azimuth_angle > 360.0:
        normal_azimuth_angle -= 360.0

    mca_azimuth = 270.0 - normal_azimuth_angle
    if mca_azimuth < 0.0:
        mca_azimuth += 360.0

    return mca_azimuth



def distribute_photon(Nphoton, weights, base_ratio=0.05):

    Ndist = weights.size
    photons_dist = np.int_(Nphoton*(1.0-base_ratio)*weights) + np.int_(Nphoton*base_ratio/Ndist)

    Ndiff = Nphoton - photons_dist.sum()

    if Ndiff >= 0:
        photons_dist[np.argmin(weights)] += Ndiff
    else:
        photons_dist[np.argmax(weights)] += Ndiff

    return photons_dist



if __name__ == '__main__':

    pass
