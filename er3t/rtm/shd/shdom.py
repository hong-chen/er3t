import os
import sys
import copy
import time
import psutil
import datetime
import warnings
import multiprocessing as mp
import numpy as np

import er3t.common
import er3t.util
from er3t.rtm.shd import shd_inp_file
from er3t.rtm.shd import shd_run



__all__ = ['shdom_ng']



class shdom_ng:

    """
    Input:

        atm_1ds=: Python list, contains object of atm_1d
        atm_3ds=: Python list, contains object of atm_3d

        Ng=    : integer, number of gs, e.g., for abs_16g, Ng=16
        target=: string type, can be one of 'flux', 'radiance', and 'heating rate'

        date=  : keyword argument, datetime.datetime object, the date to calculate sun-earth distance
        target=: keyword argument, string, can be 'flux', 'radiance', default='flux'
        surface_albedo=     : keyword argument, float, surface albedo, default=0.03
        solar_zenith_angle= : keyword argument, float, solar zenith angle, default=30.0
        solar_azimuth_angle=: keyword argument, float, solar azimuth angle, default=0.0

        Nrun=     : keyword argument, integer, number of runs to calculate mean/std statistics, default=1

        solver=   : keyword argument, integer, 0:3d mode, 1:partial-3d mode, 2:ipa mode, default=0
        Ncpu=     : keyword argument, integer, number of CPUs to use, default=1
        overwrite=: keyword argument, boolen, whether to overwrite existing SHDOM output files (rerun SHDOM), default=True
        mp_mode=: keyword argument, string, multiprocessing mode, can be 'py', 'sh', 'mpi', default='py'

        fdir=   : keyword argument, string, path to store SHDOM input and output files
        comment=: keyword argument, boolen, whether to add comment in SHDOM input file, default=False
        verbose=: keyword argument, boolen, verbose tag, default=False
        quiet=  : keyword argument, boolen, verbose tag, default=False

    Output:

        SHDOM output files created under path specified by 'fdir'
    """

    reference = '\nSHDOM (Evans, 1998):\n- Evans, K. F.: The Spherical Harmonics Discrete Ordinate Method for Three-Dimensional Atmospheric Radiative Transfer, J. Atmos. Sci., 55, 429–446, https://doi.org/10.1175/1520-0469(1998)055<0429:TSHDOM>2.0.CO;2, 1998.'


    def __init__(self,                                          \

                 date                = datetime.datetime.now(), \

                 atm_1ds             = [],                      \
                 atm_3ds             = [],                      \

                 Ng                  = 1,                       \
                 Niter               = 100,                     \

                 fdir                = 'tmp-data/sim-shd',      \
                 Ncpu                = 'auto',                  \
                 mp_mode             = 'py',                    \

                 surface_albedo      = 0.03,                    \

                 solar_zenith_angle  = 30.0,                    \
                 solar_azimuth_angle = 0.0,                     \

                 sensor_zenith_angles  = np.array([0.0]),       \
                 sensor_azimuth_angles = np.array([0.0]),       \
                 sensor_altitude       = 705000.0,              \
                 sensor_res_dx         = 100.0,                 \
                 sensor_res_dy         = 100.0,                 \

                 target              = 'flux',                  \
                 solver              = '3d',                    \

                 comment             = False,                   \
                 overwrite           = True,                    \
                 force               = False,                   \
                 verbose             = False,                   \
                 quiet               = False,                   \
                 ):

        er3t.util.add_reference(self.reference)

        fdir = os.path.abspath(fdir)

        if not os.path.exists(fdir):
            os.makedirs(fdir)
            if not quiet:
                print('Message [shdom_ng]: Directory <%s> is created.' % fdir)
        else:
            if verbose:
                print('Message [shdom_ng]: Directory <%s> already exists.' % fdir)

        self.Ng      = Ng
        self.date    = date
        self.fdir    = fdir
        self.verbose = verbose
        self.quiet   = quiet
        self.overwrite = overwrite
        self.force     = force
        self.mp_mode   = mp_mode.lower()

        self.surface_albedo      = surface_albedo
        self.solar_zenith_angle  = solar_zenith_angle
        self.solar_azimuth_angle = solar_azimuth_angle

        self.sensor_zenith_angle = sensor_zenith_angles
        self.sensor_azimuth_angle= sensor_azimuth_angles
        self.sensor_altitude     = sensor_altitude

        solver = solver.lower()
        if solver in ['3d', '3 d', 'three d']:
            self.solver = '3D'
        elif solver in ['ipa', 'independent pixel approximation']:
            self.solver = 'IPA'
        else:
            msg = 'Error [shdom_ng]: Cannot understand <solver=%s>.' % self.solver
            raise OSError(msg)

        self.target  = target

        # params
        #╭────────────────────────────────────────────────────────────────────────────╮#
        self.wvl_info = atm_1ds[0].wvl_info
        self.wvl  = atm_1ds[0].nml['WAVELEN']['data']*1000.0
        self.wvln = atm_1ds[0].nml['WAVENO']['data']

        self.fname_ckd = atm_1ds[0].nml['CKDFILE']['data']
        self.fname_prp = atm_3ds[0].nml['PROPFILE']['data']

        self.sfc_alb  = surface_albedo
        self.sfc_temp = atm_1ds[0].nml['GNDTEMP']['data']
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Nx, Ny, Nz
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if len(atm_3ds) > 0:
            self.Nx = atm_3ds[0].nml['NX']['data']
            self.Ny = atm_3ds[0].nml['NY']['data']
            self.Nz = atm_3ds[0].nml['NZ']['data']
        else:
            self.Nx = 1
            self.Ny = 1
            self.Nz = atm_1ds[0].nml['NZ']['data']

        self.dx = sensor_res_dx
        self.dy = sensor_res_dy
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Determine how many CPUs to utilize
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
            msg = 'Error [shdom_ng]: Cannot understand <Ncpu=%s>.' % Ncpu
            raise OSError(msg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # in file names, 'r' indicates #run, 'g' indicates #g, index of 'r' and 'g' both start from 0
        # self.fnames_inp/self.fnames_out is a list embeded with lists
        # E.g, in order to get the input file name of the 1st g in 1st run, self.fnames_inp[0][0]
        self.fnames_inp = []
        self.fnames_out = []
        self.fnames_sav = []
        for ig in range(self.Ng):
            self.fnames_inp.append('%s/shdom-inp_g-%3.3d.txt' % (self.fdir, ig))
            self.fnames_out.append('%s/shdom-out_g-%3.3d.txt' % (self.fdir, ig))
            self.fnames_sav.append('%s/shdom-sav_g-%3.3d.bin' % (self.fdir, ig))

        if not self.quiet and not self.overwrite:
            print('Message [shdom_ng]: Reading mode ...')

        if overwrite:

            # initialize namelist (list contains Ng Python dictionaries)
            self.nml = [{} for ig in range(self.Ng)]

            # SHDOM namelist init
            self.nml_init()

            # SHDOM namelist rad
            self.nml_rad(
                    solar_zenith_angle,
                    solar_azimuth_angle,
                    )

            # SHDOM namelist param
            self.nml_param(Niter)


            # SHDOM namelist out
            self.nml_out(
                    sensor_zenith_angles,
                    sensor_azimuth_angles,
                    sensor_altitude,
                    sensor_res_dx,
                    sensor_res_dy
                    )

            # Create SHDOM input files (ASCII)
            self.gen_shd_inp(comment=comment)

            # Run SHDOM to get output files (Binary)
            self.gen_shd_out()


    def nml_init(
            self, \
            Nmu=8,
            Nphi=16,
        ):

        for ig in range(self.Ng):

            self.nml[ig]['_header'] = '$SHDOMINPUT'
            self.nml[ig]['RUNNAME'] = 'shdom-run_g-%3.3d' % ig
            self.nml[ig]['PROPFILE'] = self.fname_prp
            self.nml[ig]['SFCFILE']  = '%s/shdom/data/shdom-sfc_land-lsrt.txt' % (os.environ['MYGIT'])
            self.nml[ig]['CKDFILE']  = self.fname_ckd
            if self.overwrite and self.force:
                self.nml[ig]['INSAVEFILE']  = 'NONE'
                self.nml[ig]['OUTSAVEFILE'] = self.fnames_sav[ig]
            else:
                self.nml[ig]['INSAVEFILE']  = self.fnames_sav[ig]
                self.nml[ig]['OUTSAVEFILE'] = self.fnames_sav[ig]

            self.nml[ig]['NSTOKES'] = 1
            self.nml[ig]['NX'] = self.Nx
            self.nml[ig]['NY'] = self.Ny
            self.nml[ig]['NZ'] = self.Nz
            self.nml[ig]['NMU']  = Nmu
            self.nml[ig]['NPHI'] = Nphi

            self.nml[ig]['BCFLAG'] = 0

            if (self.solver == '3D'):
                self.nml[ig]['IPFLAG'] = 0
            elif (self.solver == 'IPA'):
                self.nml[ig]['IPFLAG'] = 3

            self.nml[ig]['DELTAM'] = '.TRUE.'
            self.nml[ig]['GRIDTYPE'] = 'P'


    def nml_rad(
            self, \
            sza0, \
            saa0, \
        ):

        for ig in range(self.Ng):

            if self.wvl < 5025.0:
                self.nml[ig]['UNITS'] = 'R'
                self.nml[ig]['SRCTYPE'] = 'S'
            else:
                self.nml[ig]['UNITS'] = 'T'
                self.nml[ig]['SRCTYPE'] = 'T'

            self.nml[ig]['SOLARFLUX'] = er3t.util.cal_sol_fac(self.date)
            self.nml[ig]['SOLARMU'] = np.cos(np.deg2rad(sza0))
            self.nml[ig]['SOLARAZ'] = er3t.rtm.shd.cal_shd_saa(saa0)
            self.nml[ig]['SKYRAD'] = 0.0
            self.nml[ig]['GNDALBEDO'] = self.sfc_alb
            self.nml[ig]['GNDTEMP'] = self.sfc_temp
            self.nml[ig]['WAVELEN'] = self.wvl/1000.0
            self.nml[ig]['WAVENO'] = self.wvln


    def nml_out(
            self, \
            vza, \
            vaa, \
            alt0, \
            dx,
            dy,
        ):

        if self.target.lower() in ['f', 'flux', 'irradiance']:
            self.target = 'flux'
        elif self.target.lower() in ['f0', 'flux0', 'irradiance0']:
            self.target = 'flux0'
        elif self.target.lower() in ['heating rate', 'hr']:
            self.target = 'heating rate'
        elif self.target.lower() in ['radiance', 'rad']:
            self.target = 'radiance'
        else:
            msg = 'Error [shdom_ng]: Cannot understand <target=%s>.' % self.target
            raise OSError(msg)

        for ig in range(self.Ng):

            self.nml[ig]['NUMOUT'] = 1

            if self.target == 'radiance':

                vza_new = np.cos(np.deg2rad(np.array(vza)))
                vaa_new = np.zeros_like(vza)

                for i, vaa0 in enumerate(vaa):
                    vaa_new[i] = er3t.rtm.shd.cal_shd_vaa(vaa0)

                self.nml[ig]['OUTTYPES(1)'] = 'R'
                self.nml[ig]['OUTPARMS(1,1)'] = '%.4f, %.4f, %.4f, 0.0, 0.0, %d,\n%s'\
                        % (alt0/1000.0, dx/1000.0, dy/1000.0, vza_new.size, '\n'.join([' %.8f, %.4f,' % tuple(item) for item in zip(vza_new, vaa_new)]))

                self.nml[ig]['OUTPARMS(1,1)'] = self.nml[ig]['OUTPARMS(1,1)'][:-1] # get rid of comma (,) at the end

            elif self.target == 'flux':

                self.nml[ig]['OUTTYPES(1)'] = 'F'
                self.nml[ig]['OUTPARMS(1,1)'] = 4

            elif self.target == 'flux0':

                self.nml[ig]['OUTTYPES(1)'] = 'F'
                self.nml[ig]['OUTPARMS(1,1)'] = 1

            elif self.target == 'heating rate':

                self.nml[ig]['OUTTYPES(1)'] = 'H'
                self.nml[ig]['OUTPARMS(1,1)'] = 2

            self.nml[ig]['OUTFILES(1)'] = self.fnames_out[ig]
            self.nml[ig]['OutFileNC'] = 'NONE'


    def nml_param(
            self, \
            Niter,
        ):

        for ig in range(self.Ng):

            self.nml[ig]['ACCELFLAG'] = '.TRUE.'
            self.nml[ig]['SOLACC'] = 1.0e-5
            self.nml[ig]['MAXITER'] = Niter
            self.nml[ig]['SPLITACC'] = 0.01
            self.nml[ig]['SHACC'] = 0.003
            self.nml[ig]['MAX_TOTAL_MB'] = psutil.virtual_memory().total / 1024.0**2.0 / 2.0
            self.nml[ig]['ADAPT_GRID_FACTOR'] = 2.2
            self.nml[ig]['NUM_SH_TERM_FACTOR'] = 0.6
            self.nml[ig]['CELL_TO_POINT_RATIO'] = 1.5
            self.nml[ig]['_footer'] = '$END'


    def gen_shd_inp(self, comment=False):

        """
        Generate SHDOM input files from namelists

        Input:
            fnames : positional argument, file name of the input files to be created
            nmls   : positional argument, Python dictionaries that contains SHDOM namelist parameters
            comment: keyword argument, default=False, whether to add comments for each SHDOM parameter in the input file

        Output:
            text files specified by fnames
        """

        # create input files for SHDOM
        for ig in range(self.Ng):
            shd_inp_file(self.fnames_inp[ig], self.nml[ig], comment=comment)

        if not self.quiet:
            print('Message [shdom_ng]: Created SHDOM input files under <%s>.' % self.fdir)


    def gen_shd_out(self):

        """
        Run SHDOM to get SHDOM output files
        """

        fnames_inp = self.fnames_inp
        fnames_out = self.fnames_out

        if not self.quiet:
            print('Message [shdom_ng]: Running SHDOM to get output files under <%s> ...' % self.fdir)

        if not self.quiet:
            self.print_info()

        run0 = shd_run(fnames_inp, Ncpu=self.Ncpu, verbose=self.verbose, quiet=self.quiet, mp_mode=self.mp_mode)


    def print_info(self):

        print('╭────────────────────────────────────────────────────────╮')
        print('                 General Information                      ')
        print('               Simulation : %s %s' % (self.solver, self.target.title()))
        print('               Wavelength : %s' % (self.wvl_info))

        print('               Date (DOY) : %s (%d)' % (self.date.strftime('%Y-%m-%d'), self.date.timetuple().tm_yday))

        print('       Solar Zenith Angle : %.4f° (0 at local zenith)' % self.solar_zenith_angle)
        print('      Solar Azimuth Angle : %.4f° (0 at north; 90° at east)' % self.solar_azimuth_angle)

        if self.target == 'radiance':
            for i, vza0 in enumerate(self.sensor_zenith_angle):
                vaa0 = self.sensor_azimuth_angle[i]

                if vza0 < 90.0:
                    print('[%2.2d]  Sensor Zenith Angle : %.4f° (looking down, 0 straight down)' % (i, vza0))
                else:
                    print('[%2.2d]  Sensor Zenith Angle : %.4f° (looking up, 180° straight up)' % (i, vza0))

                print('[%2.2d] Sensor Azimuth Angle : %.4f° (0 at north; 90° at east)' % (i, vaa0))
            print('          Sensor Altitude : %.1f km' % (self.sensor_altitude/1000.0))

        print('           Surface Albedo : 2D domain')

        print('           Phase Function : %s' % 'Mie (Water Clouds)')

        if (self.Nx > 1) | (self.Ny > 1):
            print('     Domain Size (Nx, Ny) : (%d, %d)' % (self.Nx, self.Ny))
            print('      Pixel Res. (dx, dy) : (%.2f km, %.2f km)' % (self.dx/1000.0, self.dy/1000.0))

        print('           Number of CPUs : %d (used) of %d (total)' % (self.Ncpu, self.Ncpu_total))
        print('╰────────────────────────────────────────────────────────╯')



if __name__ == '__main__':

    pass
