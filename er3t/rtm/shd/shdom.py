import os
import sys
import copy
import time
import psutil
import datetime
import warnings
from collections import OrderedDict
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

        date=   : keyword argument, datetime.datetime object, the date to calculate sun-earth distance
        target= : keyword argument, string, can be 'flux', 'radiance', default='flux'
        surface=: keyword argument, float, surface albedo, default=0.03
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

    reference = f"\nSHDOM (Evans, 1998):\n- Evans, K. F.: The Spherical Harmonics Discrete Ordinate Method for Three-Dimensional Atmospheric Radiative Transfer, J. Atmos. Sci., 55, 429–446, https://doi.org/10.1175/1520-0469(1998)055<0429:TSHDOM>2.0.CO;2, 1998."


    def __init__(self,                                          \

                 date                = datetime.datetime.now(), \

                 atm_1ds             = [],                      \
                 atm_3ds             = [],                      \

                 surface             = None,                    \

                 Ng                  = 1,                       \
                 Niter               = 100,                     \
                 sol_acc             = 1.0e-5,                  \
                 Nmu                 = None,
                 Nphi                = None,
                 split_acc           = None,                    \
                 sh_acc              = None,                    \

                 fdir                = "tmp-data/sim-shd",      \
                 Ncpu                = "auto",                  \
                 mp_mode             = "py",                    \

                 solar_zenith_angle  = 30.0,                    \
                 solar_azimuth_angle = 0.0,                     \

                 sensor_type           = "radiometer",          \
                 sensor_zenith_angles  = np.array([0.0]),       \
                 sensor_azimuth_angles = np.array([0.0]),       \
                 sensor_altitude       = 705.0,                 \
                 sensor_dx             = 0.1,                   \
                 sensor_dy             = 0.1,                   \
                 sensor_xpos           = 0.5,                   \
                 sensor_ypos           = 0.5,                   \

                 target              = "flux",                  \
                 solver              = "3d",                    \

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
                print(f"Message [shdom_ng]: Directory <{fdir}> is created.")
        else:
            if verbose:
                print(f"Message [shdom_ng]: Directory <{fdir}> already exists.")

        self.Ng      = Ng
        self.Ng_     = 1 # currently SHDOM integrates gs within its calculation
        self.date    = date
        self.fdir    = fdir
        self.verbose = verbose
        self.quiet   = quiet
        self.overwrite = overwrite
        self.force     = force
        self.mp_mode   = mp_mode.lower()

        self.surface = surface
        if isinstance(self.surface, float) or isinstance(self.surface, np.float32) or isinstance(self.surface, np.float64):
            self.sfc_2d = False
        elif isinstance(surface, er3t.rtm.shd.shd_sfc_2d):
            self.sfc_2d = True

        self.solar_zenith_angle  = solar_zenith_angle
        self.solar_azimuth_angle = solar_azimuth_angle

        self.sensor_type          = sensor_type.lower()
        self.sensor_zenith_angle  = np.array(sensor_zenith_angles).ravel()
        self.sensor_azimuth_angle = np.array(sensor_azimuth_angles).ravel()
        self.sensor_altitude      = sensor_altitude
        self.sensor_xpos          = sensor_xpos
        self.sensor_ypos          = sensor_ypos

        solver = solver.lower()
        if solver in ['3d', '3 d', 'three d']:
            self.solver = '3D'
        elif solver in ['ipa', 'independent pixel approximation']:
            self.solver = 'IPA'
        else:
            msg = f"Error [shdom_ng]: Cannot understand <solver={self.solver}>."
            raise OSError(msg)

        target  = target.lower()
        if target in ['f', 'flux', 'irradiance']:
            self.target = 'flux'
        elif target in ['f0', 'flux0', 'irradiance0']:
            self.target = 'flux0'
        elif target in ['heating rate', 'hr']:
            self.target = 'heating rate'
        elif target in ['radiance', 'rad']:
            self.target = 'radiance'
        else:
            msg = f"Error [shdom_ng]: Cannot understand <target={self.target}>."
            raise OSError(msg)

        # params
        #╭────────────────────────────────────────────────────────────────────────────╮#
        self.wvl_info = atm_1ds[0].wvl_info
        self.wvl  = atm_1ds[0].nml['WAVELEN']['data']*1000.0
        self.wvln = atm_1ds[0].nml['WAVENO']['data']

        if not self.sfc_2d:
            self.fname_sfc = 'NONE'
        else:
            self.fname_sfc = self.surface.nml['SFCFILE']['data']

        self.fname_ckd = atm_1ds[0].nml['CKDFILE']['data']
        self.fname_prp = atm_3ds[0].nml['PROPFILE']['data']

        self.sfc_temp = atm_1ds[0].nml['GNDTEMP']['data']

        self.Nmu = Nmu
        self.Nphi= Nphi
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

        self.dx = sensor_dx
        self.dy = sensor_dy

        # overwrite Nx, Ny, dx, dy if 1D atm but 2D surface
        if self.sfc_2d and (self.Nx == 1) and (self.Ny == 1):
            self.Nx = self.surface.nml['NX']['data']
            self.Ny = self.surface.nml['NY']['data']
            self.dx = self.surface.nml['dx']['data']
            self.dy = self.surface.nml['dy']['data']
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Determine how many CPUs to utilize
        #╭────────────────────────────────────────────────────────────────────────────╮#
        Ncpu_total = mp.cpu_count()
        self.Ncpu_total = Ncpu_total
        if Ncpu == 'auto':
            self.Ncpu = Ncpu_total - 1
        elif Ncpu >= 1:
            if Ncpu > Ncpu_total:
                self.Ncpu = Ncpu_total
            else:
                self.Ncpu = Ncpu
        else:
            msg = f"Error [shdom_ng]: Cannot understand <Ncpu={Ncpu}>."
            raise OSError(msg)

        if (self.Nx == 1) and (self.Ny == 1):
            self.Ncpu = 1
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # in file names, 'r' indicates #run, 'g' indicates #g, index of 'r' and 'g' both start from 0
        # self.fnames_inp/self.fnames_out is a list embeded with lists
        # E.g, in order to get the input file name of the 1st g in 1st run, self.fnames_inp[0][0]
        self.fnames_inp = []
        self.fnames_out = []
        self.fnames_sav = []
        for ig in range(self.Ng_):
            self.fnames_inp.append(f"{self.fdir}/shdom-inp_g-{ig:03d}.txt")
            # if 'camera' in self.sensor_type:
            #     self.fnames_out.append(f"{self.fdir}/shdom-out_g-{ig:03d}.pgm")
            # else:
            self.fnames_out.append(f"{self.fdir}/shdom-out_g-{ig:03d}.txt")
            self.fnames_sav.append(f"{self.fdir}/shdom-sav_g-{ig:03d}.sHdOm-sav")

        if not self.quiet and not self.overwrite:
            print("Message [shdom_ng]: Reading mode ...")

        if overwrite:

            # initialize namelist (list contains Ng Python dictionaries)
            self.nml = [{} for ig in range(self.Ng_)]

            # SHDOM namelist init
            self.nml_init()

            # SHDOM namelist rad
            self.nml_rad(
                    solar_zenith_angle,
                    solar_azimuth_angle,
                    )

            # SHDOM namelist param
            self.nml_param(Niter, sol_acc=sol_acc, split_acc=split_acc, sh_acc=sh_acc)

            # SHDOM namelist out
            self.nml_out(
                    sensor_zenith_angles,
                    sensor_azimuth_angles,
                    sensor_altitude,
                    sensor_dx,
                    sensor_dy
                    )

            # Create SHDOM input files (ASCII)
            self.gen_shd_inp(comment=comment)

            # Run SHDOM to get output files (ASCII[info] + Binary[data])
            self.gen_shd_out()


    def nml_init(
            self, \
        ):

        for ig in range(self.Ng_):

            self.nml[ig]['_header'] = "$SHDOMINPUT"
            self.nml[ig]['RUNNAME'] = f"shdom-run_g-{ig:03d}"
            self.nml[ig]['PROPFILE'] = self.fname_prp
            self.nml[ig]['SFCFILE']  = self.fname_sfc
            self.nml[ig]['CKDFILE']  = self.fname_ckd

            if self.overwrite and self.force:
                self.nml[ig]['INSAVEFILE']  = "NONE"
                self.nml[ig]['OUTSAVEFILE'] = self.fnames_sav[ig]
            else:
                self.nml[ig]['INSAVEFILE']  = self.fnames_sav[ig]
                self.nml[ig]['OUTSAVEFILE'] = self.fnames_sav[ig]

            self.nml[ig]['NSTOKES'] = 1
            self.nml[ig]['NX'] = self.Nx
            self.nml[ig]['NY'] = self.Ny
            self.nml[ig]['NZ'] = self.Nz

            if self.Nmu is None:
                if self.target == "radiance":
                    if (self.Nx == 1) and (self.Ny == 1):
                        # follows Emde et al., 2019 (IPRT polarized radiative transfer model intercomparison project – phase A)
                        self.nml[ig]['NMU']  = 128
                    elif (self.Nx > 64) and (self.Ny > 64):
                        self.nml[ig]['NMU']  = 12
                    else:
                        self.nml[ig]['NMU']  = 18
                else:
                    if (self.Nx == 1) and (self.Ny == 1):
                        self.nml[ig]['NMU']  = 64
                    elif (self.Nx > 64) and (self.Ny > 64):
                        self.nml[ig]['NMU']  = 8
                    else:
                        self.nml[ig]['NMU']  = 12
            else:
                self.nml[ig]['NMU']  = self.Nmu

            if self.Nphi is None:
                if self.target == "radiance":
                    if (self.Nx == 1) and (self.Ny == 1):
                        # follows Emde et al., 2019 (IPRT polarized radiative transfer model intercomparison project – phase A)
                        self.nml[ig]['NPHI'] = 256
                    elif (self.Nx > 64) and (self.Ny > 64):
                        self.nml[ig]['NPHI'] = 24
                    else:
                        self.nml[ig]['NPHI'] = 36
                else:
                    if (self.Nx == 1) and (self.Ny == 1):
                        self.nml[ig]['NPHI'] = 1
                    elif (self.Nx > 64) and (self.Ny > 64):
                        self.nml[ig]['NPHI'] = 16
                    else:
                        self.nml[ig]['NPHI'] = 24
            else:
                self.nml[ig]['NPHI']  = self.Nphi

            self.nml[ig]['BCFLAG'] = 0

            if (self.solver == "3D"):
                self.nml[ig]['IPFLAG'] = 0
            elif (self.solver == "IPA"):
                self.nml[ig]['IPFLAG'] = 3

            if self.wvl < 5025.0:
                self.nml[ig]['DELTAM'] = ".TRUE."
            else:
                self.nml[ig]['DELTAM'] = ".FALSE."

            self.nml[ig]['GRIDTYPE'] = "P"


    def nml_rad(
            self, \
            sza0, \
            saa0, \
        ):

        for ig in range(self.Ng_):

            if self.wvl < 5025.0:
                self.nml[ig]['UNITS'] = "R"
                self.nml[ig]['SRCTYPE'] = "S"
            else:
                self.nml[ig]['UNITS'] = "T"
                self.nml[ig]['SRCTYPE'] = "T"

            self.nml[ig]['SOLARFLUX'] = er3t.util.cal_sol_fac(self.date)
            self.nml[ig]['SOLARMU'] = np.cos(np.deg2rad(sza0))
            self.nml[ig]['SOLARAZ'] = er3t.rtm.shd.cal_shd_saa(saa0)
            self.nml[ig]['SKYRAD']  = 0.0

            if not self.sfc_2d:
                self.nml[ig]['GNDALBEDO'] = self.surface
            else:
                self.nml[ig]['GNDALBEDO'] = 0.0
            self.nml[ig]['GNDTEMP'] = self.sfc_temp   # K
            self.nml[ig]['WAVELEN'] = self.wvl/1000.0 # micron
            self.nml[ig]['WAVENO']  = self.wvln       # cm^-1


    def nml_out(
            self, \
            vza, \
            vaa, \
            alt0, \
            dx,
            dy,
        ):

        vza = np.array(vza).ravel()
        vaa = np.array(vaa).ravel()

        for ig in range(self.Ng_):

            self.nml[ig]['NUMOUT'] = 1

            if self.target == "radiance":

                vza_new = vza
                mu_new  = np.cos(np.deg2rad(vza_new))
                vaa_new = np.array([er3t.rtm.shd.cal_shd_vaa(vaa0) for vaa0 in vaa])

                if self.sensor_type == "radiometer":

                    self.nml[ig]['OUTTYPES(1)'] = "R"

                    if mu_new.size <= 36:

                        view_str = "\n".join([f" {item[0]:.16f}, {item[1]:.4f}," for item in zip(mu_new, vaa_new)])
                        self.nml[ig]['OUTPARMS(1,1)'] = f"{alt0:.4f}, {dx:.8f}, {dy:.8f}, 0.0, 0.0, {mu_new.size},\n{view_str}"
                        self.nml[ig]['OUTPARMS(1,1)'] = self.nml[ig]['OUTPARMS(1,1)'][:-1] # get rid of comma (,) at the end

                    else:

                        data_sensor = OrderedDict()
                        data_sensor['vza'] = mu_new
                        data_sensor['vaa'] = vaa_new
                        fname_sensor = self.fnames_inp[ig].replace('shdom-inp', 'shdom-sen')
                        self.fname_sensor = er3t.rtm.shd.gen_sen_file(fname_sensor, data_sensor)

                        self.nml[ig]['SENFILE'] = f"{self.fname_sensor}"
                        self.nml[ig]['OUTPARMS(1,1)'] = f"{alt0:.4f}, {dx:.8f}, {dy:.8f}, 0.0, 0.0, {mu_new.size}"

                elif self.sensor_type == "camera1":

                    self.nml[ig]['OUTTYPES(1)'] = "V"
                    nbyte = 1
                    downscale = 1000
                    theta = 180.0
                    phi = 0.0
                    rotang = 0.0
                    nlines = 500
                    nsamps = 500
                    delline = 179.0/nlines
                    delsamp = 179.0/nsamps
                    self.nml[ig]['OUTPARMS(1,1)'] = f"1 {nbyte} {downscale} {self.sensor_xpos:.4f} {self.sensor_ypos:.4f} {self.sensor_altitude:.4f} {theta:.1f} {phi:.1f} {rotang:.1f} {nlines} {nsamps} {delline:.4f} {delsamp:.4f}"

                elif self.sensor_type == "camera2":

                    self.nml[ig]['OUTTYPES(1)'] = "V"
                    nbyte = 1
                    downscale = 1000
                    spacing = 50.0
                    scan1 = -80.0
                    scan2 = 80.0
                    delscan = 0.5

                    self.nml[ig]['OUTPARMS(1,1)'] = f"2 {nbyte} {downscale} {self.sensor_xpos:.4f} {self.sensor_ypos:.4f} {self.sensor_altitude:.4f} {self.sensor_xpos:.4f} {self.sensor_ypos+25000.0:.4f} {self.sensor_altitude:.4f} {spacing:.1f} {scan1:.1f} {scan2:.1f} {delscan:.4f}"

                elif self.sensor_type == "sensor":

                    self.nml[ig]['OUTTYPES(1)'] = "V"

                    data_sensor = OrderedDict()
                    data_sensor['x'] = self.sensor_xpos
                    data_sensor['y'] = self.sensor_ypos
                    data_sensor['z'] = self.sensor_altitude
                    data_sensor['vza'] = np.cos(np.deg2rad(180.0-vza_new))
                    data_sensor['vaa'] = np.pi - np.deg2rad(vaa_new)

                    if ((isinstance(data_sensor['x'], float)) or (isinstance(data_sensor['x'], int))):
                        data_sensor['x'] = np.repeat(data_sensor['x'], data_sensor['vza'].size)
                    if ((isinstance(data_sensor['y'], float)) or (isinstance(data_sensor['y'], int))):
                        data_sensor['y'] = np.repeat(data_sensor['y'], data_sensor['vza'].size)
                    if ((isinstance(data_sensor['z'], float)) or (isinstance(data_sensor['z'], int))):
                        data_sensor['z'] = np.repeat(data_sensor['z'], data_sensor['vza'].size)

                    fname_sensor = self.fnames_inp[ig].replace('shdom-inp', 'shdom-sen')
                    self.fname_sensor = er3t.rtm.shd.gen_sen_file(fname_sensor, data_sensor)

                    self.nml[ig]['SENFILE'] = f"{self.fname_sensor}"
                    self.nml[ig]['OUTPARMS(1,1)'] = f"3 {data_sensor['vza'].size} {len(data_sensor.keys())}"

            elif self.target == 'flux':

                self.nml[ig]['OUTTYPES(1)'] = "F"
                self.nml[ig]['OUTPARMS(1,1)'] = 4

            elif self.target == 'flux0':

                self.nml[ig]['OUTTYPES(1)'] = "F"
                self.nml[ig]['OUTPARMS(1,1)'] = 1

            elif self.target == 'heating rate':

                self.nml[ig]['OUTTYPES(1)'] = "H"
                self.nml[ig]['OUTPARMS(1,1)'] = 2

            else:

                msg = f"Error [shdom_ng]: Does NOT support <target={self.target}>."
                raise OSError(msg)

            self.nml[ig]['OUTFILES(1)'] = self.fnames_out[ig]
            self.nml[ig]['OutFileNC'] = "NONE"


    def nml_param(
            self,  \
            Niter, \
            sol_acc=1e-5,   \
            split_acc=None, \
            sh_acc=None,    \
        ):

        for ig in range(self.Ng_):

            self.nml[ig]['ACCELFLAG'] = ".TRUE."
            self.nml[ig]['SOLACC'] = sol_acc
            self.nml[ig]['MAXITER'] = Niter

            if split_acc is None:
                if (self.dx <= 0.100001) or (self.dy <= 0.100001):
                    # encounter error when grid resolution is fine, this is a temporary solution
                    # bug details [from shdom]:
                    # Note: The following floating-point exceptions are signalling: IEEE_DIVIDE_BY_ZERO IEEE_UNDERFLOW_FLAG
                    # STOP Error [_matchGridPnt]: No split direction.
                    self.nml[ig]['SPLITACC'] = 0.0
                else:
                    if (self.Nx == 1) and (self.Ny == 1):
                        self.nml[ig]['SPLITACC'] = 0.00003 # follows Emde et al., 2019 (IPRT polarized radiative transfer model intercomparison project – phase A)
                    elif (self.Nx > 64) and (self.Ny > 64):
                        self.nml[ig]['SPLITACC'] = 0.01
                    else:
                        self.nml[ig]['SPLITACC'] = 0.001
            else:
                self.nml[ig]['SPLITACC'] = split_acc

            if sh_acc is None:
                self.nml[ig]['SHACC'] = 0.003
            else:
                self.nml[ig]['SHACC'] = sh_acc

            self.nml[ig]['MAX_TOTAL_MB'] = psutil.virtual_memory().total / 1024.0**2.0 / 2.0
            if (self.Nx == 1) and (self.Ny == 1):
                self.nml[ig]['ADAPT_GRID_FACTOR'] = 50.0
                self.nml[ig]['CELL_TO_POINT_RATIO'] = 2.0
            else:
                self.nml[ig]['ADAPT_GRID_FACTOR'] = 3.2
                self.nml[ig]['CELL_TO_POINT_RATIO'] = 1.6
            self.nml[ig]['NUM_SH_TERM_FACTOR'] = 0.6
            self.nml[ig]['_footer'] = "$END"


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
        for ig in range(self.Ng_):
            shd_inp_file(self.fnames_inp[ig], self.nml[ig], comment=comment)

        if not self.quiet:
            print(f"Message [shdom_ng]: Created SHDOM input files under <{self.fdir}>.")


    def gen_shd_out(self):

        """
        Run SHDOM to get SHDOM output files
        """

        fnames_inp = self.fnames_inp
        fnames_out = self.fnames_out

        if not self.quiet:
            print(f"Message [shdom_ng]: Running SHDOM to get output files under <{self.fdir}> ...")

        if not self.quiet:
            self.print_info()

        run0 = shd_run(fnames_inp, Ncpu=self.Ncpu, verbose=self.verbose, quiet=self.quiet, mp_mode=self.mp_mode)


    def print_info(self):

        print( "╭──────────────────────────────────────────────────────────────╮")
        print( "                     General Information")
        print(f"                   Simulation : {self.solver} {self.target.title()}")
        print(f"                   Wavelength : {self.wvl_info}")

        print(f"                   Date (DOY) : {self.date.strftime('%Y-%m-%d')} ({self.date.timetuple().tm_yday})")

        print(f"           Solar Zenith Angle : {self.solar_zenith_angle:.4f}° (0 at local zenith)")
        print(f"          Solar Azimuth Angle : {self.solar_azimuth_angle:.4f}° (0 at north; 90° at east)")

        if (self.target == "radiance") and (self.sensor_type == "radiometer"):
            for i, vza0 in enumerate(self.sensor_zenith_angle[:min([2, self.sensor_zenith_angle.size])]):
                vaa0 = self.sensor_azimuth_angle[i]

                if vza0 < 90.0:
                    print(f"[{i:02d}]      Sensor Zenith Angle : {vza0:.4f}° (looking down, 0 straight down)")
                else:
                    print(f"[{i:02d}]      Sensor Zenith Angle : {vza0:.4f}° (looking up, 180° straight up)")
                print(f"[{i:02d}]     Sensor Azimuth Angle : {vaa0:.4f}° (0 at north; 90° at east)")

            if self.sensor_zenith_angle.size >= 3:
                if self.sensor_zenith_angle.size > 3:
                    print( "                         ...")
                i = self.sensor_zenith_angle.size
                vza0 = self.sensor_zenith_angle[-1]
                vaa0 = self.sensor_azimuth_angle[-1]
                if vza0 < 90.0:
                    print(f"[{i:02d}]      Sensor Zenith Angle : {vza0:.4f}° (looking down, 0 straight down)")
                else:
                    print(f"[{i:02d}]      Sensor Zenith Angle : {vza0:.4f}° (looking up, 180° straight up)")
                print(f"[{i:02d}]     Sensor Azimuth Angle : {vaa0:.4f}° (0 at north; 90° at east)")

            print(f"              Sensor Altitude : {self.sensor_altitude:.1f} km")

        else:
            if (self.sensor_type == "radiometer"):
                print(f"        User-Specified Sensor : {os.path.basename(self.fname_sensor)}")
            else:
                print(f"        User-Specified Sensor : Camera")


        if self.sfc_2d:
            print( "                 Surface BRDF : 2D domain")
        else:
            print(f"               Surface Albedo : {self.surface:.4f}")

        print( "               Phase Function : Mie (Water Clouds, from SHDOM)")

        if (self.Nx > 1) | (self.Ny > 1):
            print(f"         Domain Size (Nx, Ny) : ({self.Nx}, {self.Ny})")
            print(f"          Pixel Res. (dx, dy) : ({self.dx:.2f} km, {self.dy:.2f} km)")

        print(f"               Number of CPUs : {self.Ncpu} (used) of {self.Ncpu_total} (total)")
        print( "╰──────────────────────────────────────────────────────────────╯")



if __name__ == '__main__':

    pass
