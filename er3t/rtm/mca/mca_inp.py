import os
import sys
import warnings
from collections import OrderedDict
import numpy as np

from er3t.util import nice_array_str



__all__ = ['mca_inp_file']



def load_mca_inp_nml():

    # +
    # MCARaTS initialization
    """
    Wld_mverb=0  : verbose mode (0=quiet, 1=yes, 2=more, 3=most)
    Wld_jseed=0  : seed for random number generator (0 for automatic)
    Wld_mbswap=0 : flag for byte swapping for binary input files (0=no, 1=yes)
    Wld_mtarget=1: flag for target quantities
                   = 1 : fluxes and heating rates, calculated by MC
                   = 2 : radiances, calculated by MC
                   = 3 : quasi-radiances (or some signals), calculated by volume rendering
    Wld_moptim=2 : flag for optimization of calculation techniques
                   = -2 : no tuning (use default optimization)
                   = -1 : no optimization (deactivate all optimization)
                   =  0 : unbiased optimization (deactivate all biasing optimizations)
                   =  1 : conservative optimizations (possibly for smaller biases)
                   =  2 : standard optimizations (recommended)
                   =  3 : quick-and-dirty optimizations (for speed when small biases are acceptable)
    Wld_njob=1   : # of jobs in a single experiment
    Wld_nwl=1    : # of wavelengths or CKD terms in a single experiment
    """
    mcarats_env = OrderedDict([
                         ('Wld_mverb'  , None),
                         ('Wld_jseed'  , None),
                         ('Wld_mbswap' , None),
                         ('Wld_mtarget', None),
                         ('Wld_moptim' , None),
                         ('Wld_njob'   , None),
                         ('Wld_nwl'    , None)])

    """
    KNDFL=100
    Sca_npfd=0            : # of scattering phase function data files
    Sca_inpfile(KNDFL)=' ': file names for scattering phase functions
    Sca_npf(KNDFL)=0      : # of tabulated phase functions in each file
    Sca_nangi(KNDFL)=100  : # of angles in each file
    Sca_nskip(KNDFL)=0    : # of data record lines to be skipped
    Sca_nchi=4            : # of orders for truncation approximation (>= 2)
    Sca_ntg=20000         : # of table grids for angles & probabilities
    Sca_qtfmax=20.0       : geometrical truncation angle (deg.)

    KNP1D=KNP3D=100
    Atm_atm3dfile=' '     : file name for input of 3D otpical properties
    Atm_atm3dabs=1      : 0 (off) or 1 (on) to give the 3D gas absorption coefficinents
    Atm_np1d=1          : # of scattering components in 1D medium
    Atm_np3d=1          : # of scattering components in 3D medium
    Atm_nx=1            : # of X grid points
    Atm_ny=1            : # of Y grid points
    Atm_nz=1            : # of Z grid points
    Atm_iz3l=1          : layer index for the lowest 3-D layer
    Atm_nz3=0           : # of 3-D inhomogeneous layers
    Atm_nwl=1           : # of wavelengths
    Atm_nqlay=10        : # of Gaussian quadrature points per layer
    Atm_iipfd1d(KNP1D)=1: indices for phase function data files for 1D medium
    Atm_iipfd3d(KNP3D)=1: indices for phase function data files for 3D medium

    Sfc_inpfile=' '          : file name for input of surface properties
    Sfc_mbrdf(4)=(1, 1, 1, 1): flags of on/off status for BRDF models
    Sfc_nxb=1                : # of X grid points
    Sfc_nyb=1                : # of Y grid points
    Sfc_nsco=60              : # of uz0 for coefficient table
    Sfc_nsuz=200             : # of uz0 grids for albedo LUTs

    Src_nsrc=1: # of sources

    Flx_mflx=1      : flag for flux density calculations (0=off, 1=on)
    Flx_mhrt=1      : flag for heating rate calculations (0=off, 1=on)
    Flx_nxf=1       : # of cells along X
    Flx_nyf=1       : # of cells along Y
    Flx_diff0=1.0   : numerical diffusion parameter
    Flx_diff1=0.01  : numerical diffusion parameter
    Flx_cf_dtau=0.2 : Delta_tau, layer optical thickness for collision forcing

    Rad_mrkind=2   : a kind of radiance
                   = 0 : no radiance calculation
                   = 1 : 1st kind, local radiance averaged over solid angle
                   = 2 : 2nd kind, averaged over pixel (horizontal cross section of atmospheric column)
    Rad_mpmap=1    : method for pixel mapping
                   = 1 : polar       (U = theta * cos(phi), V = theta * sin(phi))
                   = 2 : rectangular (U = theta, V = phi)
    Rad_mplen=0    : method of calculation of pathlength statistics
                   = 0 : (npr = 0)  no calculation of pathlength statistics
                   = 1 : (npr = nz) layer-by-layer pathlength distribution
                   = 2 : (npr = nwf) average pathlengths with weighting functions
                   = 3 : (npr = ntp) histogram of total, integrated pathelength
    Rad_nrad=0     : # of radiances
    Rad_nxr=1      : # of X grid points
    Rad_nyr=1      : # of Y grid points
    Rad_nwf=1      : # of weighting functions
    Rad_ntp=100    : # of total pathlength bins
    Rad_tpmin=0.0  : min of total pathlength
    Rad_tpmax=1.0e5: max of total pathlength

    Vis_mrend=2      : method for rendering (0/1/2/3/4)
                     = 0 : integration of density along the ray (e.g. for calculating optical thickness)
                     = 1 : as 1, but with attenuation (assuming that the source function is uniformly 1)
                     = 2 : RTE-based solver, assuming horizontally-uniform source functions
                     = 3 : as 2, but taking into accout 3-D distribution of 1st-order source functions (J1)
                     = 4 : as 3, but with TMS correction for anisotropic scattering
    Vis_epserr=1.0e-4: convergence criterion for multiple scattering components
    Vis_fpsmth=0.5   : phase function smoothing fraction for relaxed TMS correction
    Vis_fatten=1.0   : attenuation factor (1 for physics-based rendering)
    Vis_nqhem=1      : # of Gaussian quadrature points in a hemisphere

    Pho_iso_SS=1       : scattering order at which 1-D transfer begins
    Pho_iso_tru=0      : truncation approximations are used after this scattering order
    Pho_iso_max=1000000: max scattering order for which radiation is sampled
    Pho_wmin=0.2       : min photon weight
    Pho_wmax=3.0       : max photon weight
    Pho_wfac=1.0       : factor for ideal photon weight
    Pho_pfpeak=30.0    : phase function peak threshold
    """
    mcarats_init = OrderedDict([
                       ('Sca_npfd'   , None),
                       ('Sca_inpfile', None),
                       ('Sca_npf'    , None),
                       ('Sca_nangi'  , None),
                       ('Sca_nskip'  , None),
                       ('Sca_nchi'   , None),
                       ('Sca_ntg'    , None),
                       ('Sca_qtfmax' , None),

                       ('Atm_atm3dfile', None),
                       ('Atm_atm3dabs' , None),
                       ('Atm_np1d'   , None),
                       ('Atm_np3d'   , None),
                       ('Atm_nx'     , None),
                       ('Atm_ny'     , None),
                       ('Atm_nz'     , None),
                       ('Atm_iz3l'   , None),
                       ('Atm_nz3'    , None),
                       ('Atm_nwl'    , None),
                       ('Atm_nqlay'  , None),
                       ('Atm_iipfd1d', None),
                       ('Atm_iipfd3d', None),

                       ('Sfc_inpfile', None),
                       ('Sfc_mbrdf'  , None),
                       ('Sfc_nxb'    , None),
                       ('Sfc_nyb'    , None),
                       ('Sfc_nsco'   , None),
                       ('Sfc_nsuz'   , None),

                       ('Src_nsrc', None),

                       ('Flx_mflx'   , None),
                       ('Flx_mhrt'   , None),
                       ('Flx_nxf'    , None),
                       ('Flx_nyf'    , None),
                       ('Flx_diff0'  , None),
                       ('Flx_diff1'  , None),
                       ('Flx_cf_dtau', None),

                       ('Rad_mrkind', None),
                       ('Rad_mpmap' , None),
                       ('Rad_mplen' , None),
                       ('Rad_nrad'  , None),
                       ('Rad_nxr'   , None),
                       ('Rad_nyr'   , None),
                       ('Rad_nwf'   , None),
                       ('Rad_ntp'   , None),
                       ('Rad_tpmin' , None),
                       ('Rad_tpmax' , None),

                       ('Vis_mrend'  , None),
                       ('Vis_epserr' , None),
                       ('Vis_fpsmth' , None),
                       ('Vis_fatten' , None),
                       ('Vis_nqhem'  , None),

                       ('Pho_iso_SS' , None),
                       ('Pho_iso_tru', None),
                       ('Pho_iso_max', None),
                       ('Pho_wmin'   , None),
                       ('Pho_wmax'   , None),
                       ('Pho_wfac'   , None),
                       ('Pho_pfpeak' , None)])
    # -

    # +
    # MCARaTS job
    """
    Wld_nplcf=0  : dummy variable (will be removed in the future)

    KNP1D=KNP3D=100
    KNWL=KNZ=3000
    Atm_idread=1           : location of data to be read
    Atm_dx=1.0e4           : X size of cell
    Atm_dy=1.0e4           : Y size of cell
    Atm_zgrd0(KNZ+1)=10.0  : Z (height) at layer interfaces
    Atm_ext1d(KNZ,KNP1D)   : extinction coefficients
    Atm_omg1d(KNZ,KNP1D) : single scattering albedos
    Atm_apf1d(KNZ,KNP1D) : phase function specification parameters
    Atm_abs1d(KNZ,KNWL)  : absorption coefficients
    Atm_fext1d(KNP1D)=1.0: scaling factor for Atm_ext1d
    Atm_fext3d(KNP3D)=1.0: scaling factor for Atm_ext3d
    Atm_fabs1d=1.0       : scaling factor for Atm_abs1d
    Atm_fabs3d=1.0       : scaling factor for Atm_abs3d
    Atm_mcs_rat=1.5      : threshold for max/mean extinction coefficient ratio
    Atm_mcs_frc=0.8      : threshold for fraction of super-voxels good for MCS
    Atm_mcs_dtauz=2.0    : Delta_tau_z,  threshold for super-voxel optical thickness
    Atm_mcs_dtauxy=4.0   : Delta_tau_xy, threshold for super-voxel optical thickness

    Sfc_NPAR=5
    Sfc_idread=1       : index of data to be read
    Sfc_mtype=1        : surface BRDF type
    Sfc_param(Sfc_NPAR): BRDF parameters
                       = (1.0, 0.0, 0.0, 0.0, 0.0)
    Sfc_nudsm=14       : # of table grid points for DSM model
    Sfc_nurpv=8        : # of table grid points for RPV model
    Sfc_nulsrt=14      : # of table grid points for LSRT model
    Sfc_nqpot=24       : # of quadrature points for preprocess
    Sfc_rrmax=5.0      : max factor for relative BRDF used for random directions
    Sfc_rrexp=0.5      : scaling exponent for relative BRDF used for random directions

    KNSRC=3000
    Src_mphi(KNSRC)=0   : flag for random azimuth
    Src_flx(KNSRC)=1.0  : source flux density
    Src_qmax(KNSRC)=0.0 : full cone angle
    Src_the(KNSRC)=120.0: zenith angle
    Src_phi(KNSRC)=0.0  : azimuth angle
    
    KNZ=KNRAD=3000; KNWF=30
    Rad_mrproj=0           : flag for angular weighting (for mrkind = 1 or 3)
                           = 0 : w = 1, results are radiances simply averaged over solid angle
                           = 1 : w = cosQ for Q = angle from the camera center direction, results are weighted average
                             Eamples: When FOV = hemisphere (nxr = 1, nyr = 1, mpmap = 2, umax = 90 deg.),
                                      mrproj = 0 for hemispherical-mean radiance (actinic flux density)
                                      mrproj = 1 for irradiance (flux density)
    Rad_difr0=10.0          : numerical diffusion parameter
    Rad_difr1=0.01          : numerical diffusion parameter
    Rad_zetamin=0.01        : threshold for radiance contribution function
    Rad_npwrn=1             : power exponent for scaling of near-field radiance contribution
    Rad_npwrf=1             : power exponent for scaling of  far-field radiance contribution
    Rad_cf_dmax=10.0        : Delta_Tau_s,max, max layer optical thickness for CF scattering
    Rad_cf_taus=5.0         : Tau_s,cf, scattering optical thickness for CF
    Rad_wfunc0(KNZ,KNWF)    : weighting functions used when Rad_mplen=2
    Rad_rmin0(KNRAD)=1.0e-17: min distance (from camera)
    Rad_rmid0(KNRAD)=1.0e3  : moderate distance (from camera)
    Rad_rmax0(KNRAD)=1.0e17 : max distance (from camera)

    Rad_phi(KNRAD)=0.0      : phi,   angle around Z0
    Rad_the(KNRAD)=0.0      : theta, angle around Y1
    Rad_psi(KNRAD)=0.0      : psi,   angle around Z2
                            Camera coordinates: Z-Y-Z, three rotations
                              1: rotation about Z0 (original Z-axis in world coordinates) by phi
                              2: rotation about Y1 (Y-axis in (X1,Y1,Z1) coordinates) by theta
                              3: rotation about Z2 (Z-axis in (X2,Y2,Z2) coordinates) by psi

    Rad_umax(KNRAD)=180.0   : max angle along U-direction
    Rad_vmax(KNRAD)=180.0   : max angle along V-direction
    Rad_qmax(KNRAD)=180.0   : max angle of FOV cone
    Rad_xpos(KNRAD)=0.5     : X relative position
    Rad_ypos(KNRAD)=0.5     : Y relative position
    Rad_zloc(KNRAD)=0.0     : Z location
    Rad_apsize(KNRAD)=0.0   : aperture size
    Rad_zref(KNRAD)=0.0     : Z location of the reference level height
    """
    mcarats_job  = OrderedDict([
                       ('Wld_nplcf', None),

                       ('Atm_idread'    , None),
                       ('Atm_dx'        , None),
                       ('Atm_dy'        , None),
                       ('Atm_zgrd0'     , None),
                       ('Atm_ext1d'     , None),
                       ('Atm_omg1d'     , None),
                       ('Atm_apf1d'     , None),
                       ('Atm_abs1d'     , None),
                       ('Atm_fext1d'    , None),
                       ('Atm_fext3d'    , None),
                       ('Atm_fabs1d'    , None),
                       ('Atm_fabs3d'    , None),
                       ('Atm_mcs_rat'   , None),
                       ('Atm_mcs_frc'   , None),
                       ('Atm_mcs_dtauz' , None),
                       ('Atm_mcs_dtauxy', None),

                       ('Sfc_idread', None),
                       ('Sfc_mtype' , None),
                       ('Sfc_param' , None),
                       ('Sfc_nudsm' , None),
                       ('Sfc_nurpv' , None),
                       ('Sfc_nulsrt', None),
                       ('Sfc_nqpot' , None),
                       ('Sfc_rrmax' , None),
                       ('Sfc_rrexp' , None),

                       ('Src_mphi' , None),
                       ('Src_flx'  , None),
                       ('Src_qmax' , None),
                       ('Src_the'  , None),
                       ('Src_phi'  , None),

                       ('Rad_mrproj' , None),
                       ('Rad_difr0'  , None),
                       ('Rad_difr1'  , None),
                       ('Rad_zetamin', None),
                       ('Rad_npwrn'  , None),
                       ('Rad_npwrf'  , None),
                       ('Rad_cf_dmax', None),
                       ('Rad_cf_taus', None),
                       ('Rad_wfunc0' , None),
                       ('Rad_rmin0'  , None),
                       ('Rad_rmid0'  , None),
                       ('Rad_rmax0'  , None),
                       ('Rad_phi'    , None),
                       ('Rad_the'    , None),
                       ('Rad_psi'    , None),
                       ('Rad_umax'   , None),
                       ('Rad_vmax'   , None),
                       ('Rad_qmax'   , None),
                       ('Rad_xpos'   , None),
                       ('Rad_ypos'   , None),
                       ('Rad_zloc'   , None),
                       ('Rad_apsize' , None),
                       ('Rad_zref'   , None)])

    # -

    mcarats_nml_all = OrderedDict([
                   ('mcarats_env', mcarats_env),
                   ('mcarats_init', mcarats_init),
                   ('mcarats_job', mcarats_job)])

    return mcarats_nml_all



def load_mca_inp_nml_info():

    # +
    # MCARaTS input parameter descriptions
    mcarats_nml_all_info = OrderedDict([
    ('KNDFL' , 'default=100'),
    ('KNP1D' , 'default=100'),
    ('KNP3D' , 'default=100'),
    ('KNWL'  , 'default=3000'),
    ('KNZ'   , 'default=3000'),
    ('KNSRC' , 'default=3000'),
    ('KNWF'  , 'default=30'),

    ('Wld_mverb'  , 'default=0 : verbose mode (0=quiet, 1=yes, 2=more, 3=most)'),
    ('Wld_jseed'  , 'default=0 : seed for random number generator (0 for automatic)'),
    ('Wld_mbswap' , 'default=0 : flag for byte swapping for binary input files (0=no, 1=yes)'),
    ('Wld_mtarget', 'default=1 : flag for target quantities\n\
      = 1 : fluxes and heating rates, calculated by MC\n\
      = 2 : radiances, calculated by MC\n\
      = 3 : quasi-radiances (or some signals), calculated by volume rendering'),
    ('Wld_moptim' , 'default=2 : flag for optimization of calculation techniques\n\
      = -2 : no tuning (use default optimization)\n\
      = -1 : no optimization (deactivate all optimization)\n\
      =  0 : unbiased optimization (deactivate all biasing optimizations)\n\
      =  1 : conservative optimizations (possibly for smaller biases)\n\
      =  2 : standard optimizations (recommended)\n\
      =  3 : quick-and-dirty optimizations (for speed when small biases are acceptable)'),
    ('Wld_njob'   , 'default=1 : # of jobs in a single experiment'),
    ('Wld_nwl'    , 'default=1 : # of wavelengths or CKD terms in a single experiment'),

    ('Sca_npfd'    , 'default=0 : # of scattering phase function data files'),
    ('Sca_inpfile' , 'default(KNDFL)=\' \' : file names for scattering phase functions'),
    ('Sca_npf'     , 'default(KNDFL)=0 : # of tabulated phase functions in each file'),
    ('Sca_nangi'   , 'default(KNDFL)=100 : # of angles in each file'),
    ('Sca_nskip'   , 'default(KNDFL)=0 : # of data record lines to be skipped'),
    ('Sca_nchi'    , 'default=4 : # of orders for truncation approximation (>= 2)'),
    ('Sca_ntg'     , 'default=20000 : # of table grids for angles & probabilities'),
    ('Sca_qtfmax'  , 'default=20.0  : geometrical truncation angle (deg.)'),

    ('Atm_atm3dfile' , 'default=\' \' : file name for input of 3D otpical properties'),
    ('Atm_atm3dabs'  , 'default=1 :  0 (off) or 1 (on) to give the 3D (perturbations of) \n\
        gaseous absorption coefficinents in the data file Atm_atm3dfile'),
    ('Atm_np1d'    , 'default=1 : # of scattering components in 1D medium'),
    ('Atm_np3d'    , 'default=1 : # of scattering components in 3D medium'),
    ('Atm_nx'      , 'default=1 : # of X grid points'),
    ('Atm_ny'      , 'default=1 : # of Y grid points'),
    ('Atm_nz'      , 'default=1 : # of Z grid points'),
    ('Atm_iz3l'    , 'default=1 : layer index for the lowest 3-D layer'),
    ('Atm_nz3'     , 'default=0 : # of 3-D inhomogeneous layers'),
    ('Atm_nwl'     , 'default=1 : # of wavelengths'),
    ('Atm_nqlay'   , 'default=10 : # of Gaussian quadrature points per layer'),
    ('Atm_iipfd1d' , 'default(KNP1D)=1 : indices for phase function data files for 1D medium'),
    ('Atm_iipfd3d' , 'default(KNP3D)=1 : indices for phase function data files for 3D medium'),

    ('Sfc_inpfile' , 'default=\' \' : file name for input of surface properties'),
    ('Sfc_mbrdf'   , 'default(4)=(1, 1, 1, 1) : flags of on/off status for BRDF models'),
    ('Sfc_nxb'     , 'default=1 : # of X grid points'),
    ('Sfc_nyb'     , 'default=1 : # of Y grid points'),
    ('Sfc_nsco'    , 'default=60 : # of uz0 for coefficient table'),
    ('Sfc_nsuz'    , 'default=200 : # of uz0 grids for albedo LUTs'),
    ('Src_nsrc'    , 'default=1 : # of sources'),

    ('Flx_mflx'    , 'default=1 : flag for flux density calculations (0=off, 1=on)'),
    ('Flx_mhrt'    , 'default=1 : flag for heating rate calculations (0=off, 1=on)'),
    ('Flx_nxf'     , 'default=1 : # of cells along X'),
    ('Flx_nyf'     , 'default=1 : # of cells along Y'),
    ('Flx_diff0'   , 'default=1.0 : numerical diffusion parameter'),
    ('Flx_diff1'   , 'default=0.01 : numerical diffusion parameter'),
    ('Flx_cf_dtau' , 'default=0.2 : Delta_tau, layer optical thickness for collision forcing'),

    ('Rad_mrkind'  , 'default=2 : a kind of radiance\n\
      = 0 : no radiance calculation\n\
      = 1 : 1st kind, local radiance averaged over solid angle\n\
      = 2 : 2nd kind, averaged over pixel (horizontal cross section of atmospheric column)'),
    ('Rad_mpmap'   , 'default=1 : method for pixel mapping\n\
      = 1 : polar       (U = theta * cos(phi), V = theta * sin(phi))\n\
      = 2 : rectangular (U = theta, V = phi)'),
    ('Rad_mplen'   , 'default=0 : method of calculation of pathlength statistics\n\
      = 0 : (npr = 0)  no calculation of pathlength statistics\n\
      = 1 : (npr = nz) layer-by-layer pathlength distribution\n\
      = 2 : (npr = nwf) average pathlengths with weighting functions\n\
      = 3 : (npr = ntp) histogram of total, integrated pathelength'),
    ('Rad_nrad'    , 'default=0 : # of radiances'),
    ('Rad_nxr'     , 'default=1 : # of X grid points'),
    ('Rad_nyr'     , 'default=1 : # of Y grid points'),
    ('Rad_nwf'     , 'default=1 : # of weighting functions'),
    ('Rad_ntp'     , 'default=100 : # of total pathlength bins'),
    ('Rad_tpmin'   , 'default=0.0 : min of total pathlength'),
    ('Rad_tpmax'   , 'default=1.0e5 : max of total pathlength'),

    ('Vis_mrend'   , 'default=2 : method for rendering (0/1/2/3/4)\n\
      = 0 : integration of density along the ray (e.g. for calculating optical thickness)\n\
      = 1 : as 1, but with attenuation (assuming that the source function is uniformly 1)\n\
      = 2 : RTE-based solver, assuming horizontally-uniform source functions\n\
      = 3 : as 2, but taking into accout 3-D distribution of 1st-order source functions (J1)\n\
      = 4 : as 3, but with TMS correction for anisotropic scattering'),
    ('Vis_epserr'  , 'default=1.0e-4 : convergence criterion for multiple scattering components'),
    ('Vis_fpsmth'  , 'default=0.5 : phase function smoothing fraction for relaxed TMS correction'),
    ('Vis_fatten'  , 'default=1.0 : attenuation factor (1 for physics-based rendering)'),
    ('Vis_nqhem'   , 'default=1 : # of Gaussian quadrature points in a hemisphere'),

    ('Pho_iso_SS'  , 'default=1 : scattering order at which 1-D transfer begins'),
    ('Pho_iso_tru' , 'default=0 : truncation approximations are used after this scattering order'),
    ('Pho_iso_max' , 'default=1000000 : max scattering order for which radiation is sampled'),
    ('Pho_wmin'    , 'default=0.2 : min photon weight'),
    ('Pho_wmax'    , 'default=3.0 : max photon weight'),
    ('Pho_wfac'    , 'default=1.0 : factor for ideal photon weight'),
    ('Pho_pfpeak'  , 'default=30.0 : phase function peak threshold'),

    ('Wld_nplcf'   , 'default=0 : dummy variable (will be removed in the future)'),

    ('Atm_idread'    , 'default=1 : location of data to be read'),
    ('Atm_dx'        , 'default=1.0e4 : X size of cell'),
    ('Atm_dy'        , 'default=1.0e4 : Y size of cell'),
    ('Atm_zgrd0'     , 'default(KNZ+1)=10.0 : Z (height) at layer interfaces'),
    ('Atm_ext1d'     , 'default(KNZ,KNP1D) : extinction coefficients'),
    ('Atm_omg1d'     , 'default(KNZ,KNP1D) : single scattering albedos'),
    ('Atm_apf1d'     , 'default(KNZ,KNP1D) : phase function specification parameters'),
    ('Atm_abs1d'     , 'default(KNZ,KNWL) : absorption coefficients'),
    ('Atm_fext1d'    , 'default(KNP1D)=1.0 : scaling factor for Atm_ext1d'),
    ('Atm_fext3d'    , 'default(KNP3D)=1.0 : scaling factor for Atm_ext3d'),
    ('Atm_fabs1d'    , 'default=1.0 : scaling factor for Atm_abs1d'),
    ('Atm_fabs3d'    , 'default=1.0 : scaling factor for Atm_abs3d'),
    ('Atm_mcs_rat'   , 'default=1.5 : threshold for max/mean extinction coefficient ratio'),
    ('Atm_mcs_frc'   , 'default=0.8 : threshold for fraction of super-voxels good for MCS'),
    ('Atm_mcs_dtauz' , 'default=2.0 : Delta_tau_z,  threshold for super-voxel optical thickness'),
    ('Atm_mcs_dtauxy', 'default=4.0 : Delta_tau_xy, threshold for super-voxel optical thickness'),

    ('Sfc_NPAR'  , 'default=5'),
    ('Sfc_idread', 'default=1 : index of data to be read'),
    ('Sfc_mtype' , 'default=1 : surface BRDF type'),
    ('Sfc_param' , 'default(Sfc_NPAR) : BRDF parameters\n\
      = (1.0, 0.0, 0.0, 0.0, 0.0)'),
    ('Sfc_nudsm' , 'default=14 : # of table grid points for DSM model'),
    ('Sfc_nurpv' , 'default=8 : # of table grid points for RPV model'),
    ('Sfc_nulsrt', 'default=14 : # of table grid points for LSRT model'),
    ('Sfc_nqpot' , 'default=24 : # of quadrature points for preprocess'),
    ('Sfc_rrmax' , 'default=5.0 : max factor for relative BRDF used for random directions'),
    ('Sfc_rrexp' , 'default=0.5 : scaling exponent for relative BRDF used for random directions'),

    ('Src_mphi'  , 'default(KNSRC)=0 : flag for random azimuth'),
    ('Src_flx'   , 'default(KNSRC)=1.0 : source flux density'),
    ('Src_qmax'  , 'default(KNSRC)=0.0 : full cone angle'),
    ('Src_the'   , 'default(KNSRC)=120.0 : zenith angle'),
    ('Src_phi'   , 'default(KNSRC)=0.0 : azimuth angle'),

    ('Rad_mrproj'    , 'default=0 : flag for angular weighting (for mrkind = 1 or 3)\n\
      = 0 : w = 1, results are radiances simply averaged over solid angle\n\
      = 1 : w = cosQ for Q = angle from the camera center direction, results are weighted average\n\
      Eamples: When FOV = hemisphere (nxr = 1, nyr = 1, mpmap = 2, umax = 90 deg.),\n\
      mrproj = 0 for hemispherical-mean radiance (actinic flux density)\n\
      mrproj = 1 for irradiance (flux density)'),
    ('Rad_difr0'     , 'default=10.0 : numerical diffusion parameter'),
    ('Rad_difr1'     , 'default=0.01 : numerical diffusion parameter'),
    ('Rad_zetamin'   , 'default=0.01 : threshold for radiance contribution function'),
    ('Rad_npwrn'     , 'default=1 : power exponent for scaling of near-field radiance contribution'),
    ('Rad_npwrf'     , 'default=1 : power exponent for scaling of  far-field radiance contribution'),
    ('Rad_cf_dmax'   , 'default=10.0 : Delta_Tau_s,max, max layer optical thickness for CF scattering'),
    ('Rad_cf_taus'   , 'default=5.0 : Tau_s,cf, scattering optical thickness for CF'),
    ('Rad_wfunc0'    , 'default(KNZ,KNWF) : weighting functions used when Rad_mplen=2'),
    ('Rad_rmin0'     , 'default(KNRAD)=1.0e-17 : min distance (from camera)'),
    ('Rad_rmid0'     , 'default(KNRAD)=1.0e3 : moderate distance (from camera)'),
    ('Rad_rmax0'     , 'default(KNRAD)=1.0e17 : max distance (from camera)'),
    ('Rad_phi'       , 'default(KNRAD)=0.0 : phi,   angle around Z0'),
    ('Rad_the'       , 'default(KNRAD)=0.0 : theta, angle around Y1'),
    ('Rad_psi'       , 'default(KNRAD)=0.0 : psi,   angle around Z2\n\
      Camera coordinates : Z-Y-Z, three rotations\n\
      1: rotation about Z0 (original Z-axis in world coordinates) by phi\n\
      2: rotation about Y1 (Y-axis in (X1,Y1,Z1) coordinates) by theta\n\
      3: rotation about Z2 (Z-axis in (X2,Y2,Z2) coordinates) by psi'),
    ('Rad_umax'      , 'default(KNRAD)=180.0 : max angle along U-direction'),
    ('Rad_vmax'      , 'default(KNRAD)=180.0 : max angle along V-direction'),
    ('Rad_qmax'      , 'default(KNRAD)=180.0 : max angle of FOV cone'),
    ('Rad_xpos'      , 'default(KNRAD)=0.5 : X relative position'),
    ('Rad_ypos'      , 'default(KNRAD)=0.5 : Y relative position'),
    ('Rad_zloc'      , 'default(KNRAD)=0.0 : Z location'),
    ('Rad_apsize'    , 'default(KNRAD)=0.0 : aperture size'),
    ('Rad_zref'      , 'default(KNRAD)=0.0 : Z location of the reference level height')])
    # -

    return mcarats_nml_all_info



def mca_inp_nml(input_dict, verbose=True, comment=True):

    mcarats_nml_all      = load_mca_inp_nml()
    mcarats_nml_all_info = load_mca_inp_nml_info()

    nml_ordered_keys_full = []
    nml_ordered_item_full = []
    for nml_key in mcarats_nml_all.keys():
        for var_key in mcarats_nml_all[nml_key].keys():
            nml_ordered_keys_full.append(var_key)
            nml_ordered_item_full.append(nml_key)

    # Check input variables
    #
    # If input variable in the full dictionary keys, assign data to
    # the input variable
    #
    # If not
    #   1. If is a typo in input variables, exit with error message
    #   2. If in array-like way, e.g., 'Atm_ext1d(1:, 1)' is similar to
    #      'Atm_ext1d', update dictionary with new variable and assign data
    #      to the updated variable.
    #
    # `mcarats_nml_all` and `mcarats_nml_all_info` will be updated in the loop
    for key in input_dict.keys():
        if key not in nml_ordered_keys_full:
            if '(' in key and ')' in key:
                ee          = key.index('(')
                key_ori     = key[:ee]
                index_ori   = nml_ordered_keys_full.index(key_ori)

                key_more = [xx for xx in nml_ordered_keys_full if key_ori in xx and '(' in xx]

                if len(key_more) >= 1:
                    key0     = key_more[-1]
                    index    = nml_ordered_keys_full.index(key0)
                else:
                    index = index_ori

                nml_ordered_keys_full.insert(index+1, key)
                nml_ordered_item_full.insert(index+1, nml_ordered_item_full[index])
                mcarats_nml_all[nml_ordered_item_full[index]][key] = input_dict[key]
                if comment:
                    mcarats_nml_all_info[key] = mcarats_nml_all_info[key_ori]
            else:
                msg = 'Error [mca_inp_nml]: please check input variable <%s>.' % key
                raise OSError(msg)
        else:
            index   = nml_ordered_keys_full.index(key)
            mcarats_nml_all[nml_ordered_item_full[index]][key] = input_dict[key]

    # create a full dictionary to link variable back to namelist section
    # examples:
    #   'Wld_mverb': 'mcarWld_nml_init'
    #   'Atm_zgrd0': 'mcarAtm_nml_job'
    mcarats_nml_input = OrderedDict(zip(nml_ordered_keys_full, nml_ordered_item_full))

    return mcarats_nml_all, mcarats_nml_all_info, mcarats_nml_input



def mca_inp_file(input_fname, input_dict, verbose=True, comment=True):

    mcarats_nml_all, mcarats_nml_all_info, mcarats_nml_input \
        = mca_inp_nml(input_dict, verbose=verbose, comment=comment)

    input_fname = os.path.abspath(input_fname)
    fdir_inp    = os.path.dirname(input_fname)
    if not os.path.exists(fdir_inp):
        os.system('mkdir -p %s' % fdir_inp)

    # creating input file for MCARaTS
    f = open(input_fname, 'w')

    for nml_key in mcarats_nml_all.keys():
        f.write('&%s\n' % nml_key)

        vars_key = [xx for xx in mcarats_nml_input.keys() if mcarats_nml_input[xx]==nml_key]
        for var_key in vars_key:
            var = mcarats_nml_all[nml_key][var_key]
            if var is not None:

                if isinstance(var, (int, float, np.int32, np.int64, np.float32, np.float64)):
                    f.write(' %-15s = %-.16g\n' % (var_key, var))
                elif isinstance(var, str):
                    if '*' in var:
                        f.write(' %-15s = %s\n' % (var_key, var))
                    else:
                        f.write(' %-15s = \'%s\'\n' % (var_key, var))
                elif isinstance(var, np.ndarray):

                    if var.size > 1:

                        var_str = nice_array_str(var)

                        if len(var_str) <= 80:
                            f.write(' %-15s = %s\n' % (var_key, var_str))
                        else:
                            f.write(' %-15s =\n' % var_key)
                            f.write('%s\n' % var_str)

                    elif var.size == 1:
                        f.write(' %-15s = %-g\n' % (var_key, var))

                else:
                    msg = 'Error [mca_inp_file]: only types of int, float, str, ndarray are supported (do not support <%s> as %s).' % (var_key, type(var))
                    raise ValueError(msg)

                if comment:
                    var_detail = mcarats_nml_all_info[var_key]
                    f.write(' !------------------------------- This is a comment for above parameter -------------------------------------\n')
                    if '\n' in var_detail:
                        lines = var_detail.split('\n')
                        for line in lines:
                            f.write(' !----> %s\n' % line)
                    else:
                        f.write(' !----> %s\n' % mcarats_nml_all_info[var_key])
                    f.write(' !-----------------------------------------------------------------------------------------------------------\n')
                    f.write('\n')

        f.write('/\n')

    f.close()



if __name__ == '__main__':

    pass
