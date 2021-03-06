import os
import sys
import copy
import datetime
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from er3t.rtm.mca_v011 import mca_inp_file
from er3t.rtm.mca_v011 import mca_run



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
        mp_mode=: keyword argument, string, multiprocessing mode, can be 'py', 'sh', 'mpi', default='py'

        fdir=   : keyword argument, string, path to store MCARaTS input and output files
        verbose=: keyword argument, boolen, verbose tag, default=False
        quiet=  : keyword argument, boolen, verbose tag, default=False

    Output:

        MCARaTS output files created under path specified by 'fdir'
    """


    def __init__(self,                                          \

                 atm_1ds             = [],                      \
                 atm_3ds             = [],                      \

                 sfc_2d              = None,

                 Ng                  = 16,                      \
                 fdir                = 'data/tmp/mca',          \
                 Nrun                = 1,                       \
                 Ncpu                = 'auto',                  \
                 mp_mode             = 'py',                    \
                 overwrite           = True,                    \

                 date                = datetime.datetime.now(), \
                 tune                = False,                   \
                 target              = 'flux',                  \
                 surface_albedo      = 0.03,                    \
                 solar_zenith_angle  = 30.0,                    \
                 solar_azimuth_angle = 0.0,                     \

                 sensor_zenith_angle = 0.0,                     \
                 sensor_azimuth_angle= 0.0,                     \
                 sensor_altitude     = 705000.0,                \

                 solver              = 0,                       \
                 photons             = 1e6,                     \

                 verbose             = False,                   \
                 quiet               = False                    \
                 ):

        if not quiet:
            print('+ <mcarats_ng>')

        if not os.path.exists(fdir):
            os.makedirs(fdir)
            if not quiet:
                print('Message [mcarats_ng]: Directory \'%s\' is created.' % fdir)
        else:
            if verbose:
                print('Message [mcarats_ng]: Directory \'%s\' already exists.' % fdir)


        self.Ng      = Ng
        self.date    = date
        self.fdir    = fdir
        self.verbose = verbose
        self.quiet   = quiet
        self.overwrite = overwrite
        self.mp_mode   = mp_mode


        self.Nrun    = Nrun


        self.solver  = solver
        self.photons = photons
        self.target  = target

        # Determine how many CPUs to utilize
        Ncpu_total = mp.cpu_count()
        if Ncpu == 'auto':
            self.Ncpu = Ncpu_total - 1
        elif Ncpu > 1:
            if Ncpu > Ncpu_total:
                self.Ncpu = Ncpu_total - 1
            else:
                self.Ncpu = Ncpu
        else:
            exit('Error   [mcarats_ng]: Cannot understand \'Ncpu=%s\'.' % Ncpu)

        # in file names, 'r' indicates #run, 'g' indicates #g, index of 'r' and 'g' both start from 0
        # self.fnames_inp/self.fnames_out is a list embeded with lists
        # E.g, in order to get the input file name of the 1st g in 1st run, self.fnames_inp[0][0]
        self.fnames_inp = []
        self.fnames_out = []
        for ir in range(self.Nrun):
            self.fnames_inp.append(['%s/r%2.2d.g%2.2d.inp.txt' % (self.fdir, ir, ig) for ig in range(self.Ng)])
            self.fnames_out.append(['%s/r%2.2d.g%2.2d.out.bin' % (self.fdir, ir, ig) for ig in range(self.Ng)])

        if not self.quiet and not self.overwrite:
            print('Message [mcarats_ng]: Reading mode ...')

        if overwrite:

            # initialize namelist (list contains 16 Python dictionaries)
            self.nml = [{} for ig in range(self.Ng)]

            # MCARaTS wld initialization
            self.init_wld(verbose=verbose, tune=tune,
                sensor_zenith_angle=sensor_zenith_angle, sensor_azimuth_angle=sensor_azimuth_angle, sensor_altitude=sensor_altitude)

            # MCARaTS scattering initialization
            self.init_sca()

            # MCARaTS atm initialization
            self.init_atm(atm_1ds=atm_1ds, atm_3ds=atm_3ds)

            # MCARaTS surface initialization
            self.init_sfc(sfc_2d=sfc_2d, surface_albedo=surface_albedo)

            # MCARaTS source (e.g., solar) initialization
            self.init_src(solar_zenith_angle=solar_zenith_angle, solar_azimuth_angle=solar_azimuth_angle)

            # Create MCARaTS input files (ASCII)
            self.gen_mca_inp()

            if not self.quiet:
                print('-------------------------------------------')
                print('            General Information      ')
                print('               Date : %s' % self.date.strftime('%Y-%m-%d'))
                print(' Solar Zenith Angle : %.2f' % solar_zenith_angle)
                print('Solar Azimuth Angle : %.2f' % solar_azimuth_angle)
                print('     Surface Albedo : %.2f' % surface_albedo)
                print('     Number of Runs : %s(g) * %d(run)' % (self.Ng, self.Nrun))
                print('     Number of CPUs : %d(used) of %d(total)' % (self.Ncpu, Ncpu_total))
                print('    Photons per Run : %.1e' % self.photons)
                print('             Target : %s' % self.target.title())
                print('             Solver : %s' % self.solver)
                print('-------------------------------------------')

            # Run MCARaTS to get output files (Binary)
            self.gen_mca_out()


        self.run_check()

        if not quiet:
            print('-')


    def init_wld(self, tune=False, verbose=False, \
        sensor_zenith_angle=0.0, sensor_azimuth_angle=0.0, sensor_altitude=705000.0):

        if self.target.lower() in ['f', 'flux', 'irradiance']:
            self.target = 'flux'
        elif self.target.lower() in ['heating rate', 'hr']:
            self.target = 'heating rate'
        elif self.target.lower() in ['radiance', 'rad']:
            self.target = 'radiance'
        else:
            sys.exit('Error   [mcarats_ng]: Cannot understand \'target=%s\'.' % self.target)

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
            # self.nml[ig]['Wld_nwl']    = 1


            if self.target == 'flux' :

                self.nml[ig]['Wld_mtarget'] = 1
                self.nml[ig]['Flx_mflx']    = 3
                self.nml[ig]['Flx_mhrt']    = 0

            elif self.target == 'heating rate':

                self.nml[ig]['Wld_mtarget'] = 1
                self.nml[ig]['Flx_mflx']    = 3
                self.nml[ig]['Flx_mhrt']    = 1

            elif self.target == 'radiance':

                self.nml[ig]['Wld_mtarget'] = 2

                self.nml[ig]['Rad_mrkind']  = 2
                self.nml[ig]['Rad_mplen']   = 0
                self.nml[ig]['Rad_mpmap']   = 1
                self.nml[ig]['Rad_nrad']    = 1

                self.nml[ig]['Rad_nrad']    = 1
                self.nml[ig]['Rad_difr0']   = 7.5
                self.nml[ig]['Rad_difr1']   = 0.0025
                self.nml[ig]['Rad_the']     = 180.0 - sensor_zenith_angle
                self.nml[ig]['Rad_phi']     = cal_mca_azimuth(sensor_azimuth_angle)
                self.nml[ig]['Rad_zloc']    = sensor_altitude
                # self.nml[ig]['Rad_apsize']  = 0.05

            else:
                sys.exit('Error   [mcarats_ng]: Cannot understand \'target=%s\'.' % self.target)


    def init_sca(self):

        for ig in range(self.Ng):
            # this parameter is important, otherwise random memory segmentation error will occur
            self.nml[ig]['Sca_npf'] = 1
            self.nml[ig]['Sca_inpfile(1)'] = 'cld_0067_sca.dat'
            self.nml[ig]['Sca_npf(1)'] = 60
            self.nml[ig]['Sca_nangi(1)'] = 203


    def init_atm(self, atm_1ds=[], atm_3ds=[]):

        for ig in range(self.Ng):

            if len(atm_1ds) == 0:
                sys.exit('Error   [mcarats_ng]: need \'atm_1ds\' to proceed.')
            else:

                self.nml[ig]['Atm_np1d'] = len(atm_1ds)

                for i, atm_1d in enumerate(atm_1ds):

                    self.nml[ig]['Atm_iipfd1d(%d)' % (i+1)] = 1
                    for key in atm_1d.nml[ig].keys():

                        if key not in ['Atm_ext1d', 'Atm_omg1d', 'Atm_apf1d', 'Atm_abs1d']:
                            self.nml[ig][key] = atm_1d.nml[ig][key]['data']
                        else:
                            key_new = '%s(1:, %d)' % (key, i+1)
                            self.nml[ig][key_new] = atm_1d.nml[ig][key]['data']

            if len(atm_3ds) > 0:

                self.nml[ig]['Atm_np3d'] = len(atm_3ds)

                for i, atm_3d in enumerate(atm_3ds):

                    self.nml[ig]['Atm_iipfd3d(%d)' % (i+1)] = 1

                    for key in atm_3d.nml.keys():

                        if key[-2:] != '3d':
                            if os.path.exists(atm_3d.nml['Atm_atm3dfile']['data']):
                                os.system('mv %s %s' % (atm_3d.nml['Atm_atm3dfile']['data'], self.fdir))
                                atm_3d.nml['Atm_atm3dfile']['data'] = os.path.basename(atm_3d.nml['Atm_atm3dfile']['data'])
                            self.nml[ig][key] = atm_3d.nml[key]['data']

                if self.target == 'radiance':
                    if 'Atm_nx' in atm_3d.nml.keys() and 'Atm_ny' in atm_3d.nml.keys():
                        self.nml[ig]['Rad_nxr'] = atm_3d.nml['Atm_nx']['data']
                        self.nml[ig]['Rad_nyr'] = atm_3d.nml['Atm_ny']['data']
                    else:
                        self.nml[ig]['Rad_nxr'] = 1
                        self.nml[ig]['Rad_nyr'] = 1


    def init_src(self, solar_zenith_angle=0.0, solar_azimuth_angle=0.0):

        for ig in range(self.Ng):
            self.nml[ig]['Src_flx']   = 1.0
            self.nml[ig]['Src_qmax']  = 0.533133
            self.nml[ig]['Src_mphi']  = 0
            self.nml[ig]['Src_the']   = 180.0 - solar_zenith_angle
            self.nml[ig]['Src_phi']   = cal_mca_azimuth(solar_azimuth_angle)
            self.nml[ig]['Src_spcflx']= 1.0


    def init_sfc(self, sfc_2d=None, surface_albedo=0.03):

        for ig in range(self.Ng):

            if self.verbose:
                print('Message [mcarats_ng]: Assume Lambertian surface ...')

            if sfc_2d is not None:

                for key in sfc_2d.nml.keys():
                    if '2d' not in key:
                        if os.path.exists(sfc_2d.nml['Sfc_inpfile']['data']):
                            os.system('mv %s %s' % (sfc_2d.nml['Sfc_inpfile']['data'], self.fdir))
                            sfc_2d.nml['Sfc_inpfile']['data'] = os.path.basename(sfc_2d.nml['Sfc_inpfile']['data'])
                        self.nml[ig][key] = sfc_2d.nml[key]['data']

            else:
                # self.nml[ig]['Sfc_mbrdf'] = np.array([1, 0, 0, 0])
                self.nml[ig]['Sfc_mtype'] = 1
                self.nml[ig]['Sfc_param(1)'] = surface_albedo


    def gen_mca_inp(self):

        """
        Generate MCARaTS input files from namelists

        Input:
            fnames : positional argument, file name of the input files to be created
            nmls   : positional argument, Python dictionaries that contains MCARaTS namelist parameters

        Output:
            text files specified by fnames
        """

        # create input files for MCARaTS
        for ir in range(self.Nrun):
            for ig in range(self.Ng):
                mca_inp_file(self.fnames_inp[ir][ig], self.nml[ig])

        if not self.quiet:
            print('Message [mcarats_ng]: Created MCARaTS input files under \'%s\'.' % self.fdir)


    def gen_mca_out(self):

        """
        Run MCARaTS to get MCARaTS output files
        """


        solver = self.solver.lower()
        if solver in ['3d', '3 d', 'three d']:
            self.solver = '3D'
        elif solver in ['p3d', 'p-3d', 'partial 3d', 'partial-3d']:
            self.solver = 'Partial 3D'
        elif solver in ['ipa', 'independent pixel approximation']:
            self.solver = 'IPA'
        else:
            sys.exit('Error   [mcarats_ng]: Cannot understand \'solver=%s\'.' % self.solver)

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
            print('Message [mcarats_ng]: Running MCARaTS to get output files under \'%s\' ...' % self.fdir)

        run0 = mca_run(fnames_inp, fnames_out, data_path=self.fdir, photons=self.photons, solver=solvers[self.solver], Ncpu=self.Ncpu, verbose=self.verbose, quiet=self.quiet, mode=self.mp_mode)


    def run_check(self):

        check = []
        for ir in range(self.Nrun):
            for ig in range(self.Ng):
                fname = self.fnames_out[ir][ig]
                if not os.path.exists(fname):
                    print('Error   [mcarats_ng]: Cannot find file \'%s\'.' % fname)
                    check.append(False)
                else:
                    check.append(True)
        if not all(check):
            sys.exit()

        pass



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



if __name__ == '__main__':

    pass
