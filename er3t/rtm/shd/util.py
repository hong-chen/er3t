import os
import sys
import glob
import struct
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import numpy as np
from scipy import interpolate
import er3t.common

__all__ = [
        'cal_shd_saa',
        'cal_shd_vaa',
        'gen_mie_file_wc',
        'gen_mie_file_ic',
        'gen_ext_file',
        'gen_lwc_file',
        'gen_mie_file_from_nc',
        'gen_ice_file_from_nc',
        'gen_sen_file',
        ]


def cal_shd_saa(normal_azimuth_angle):

    """
    Convert normal azimuth angle (0 pointing north, positive when clockwise) to viewing azimuth in SHDOM

    Input:
        normal_azimuth_angle: float/integer, normal azimuth angle (0 pointing north, positive when clockwise)

    Output:
        SHDOM solar azimuth angle (0 sun shining from west, positive when counterclockwise)
    """

    while normal_azimuth_angle < 0.0:
        normal_azimuth_angle += 360.0

    while normal_azimuth_angle > 360.0:
        normal_azimuth_angle -= 360.0

    shd_saa = 270.0 - normal_azimuth_angle
    if shd_saa < 0.0:
        shd_saa += 360.0

    return shd_saa


def cal_shd_vaa(normal_azimuth_angle):

    """
    Convert normal azimuth angle (0 pointing north, positive when clockwise) to viewing azimuth in SHDOM

    Input:
        normal_azimuth_angle: float/integer, normal azimuth angle (0 pointing north, positive when clockwise)

    Output:
        SHDOM sensor azimuth angle (0 sensor looking from east, positive when counterclockwise)
    """

    while normal_azimuth_angle < 0.0:
        normal_azimuth_angle += 360.0

    while normal_azimuth_angle > 360.0:
        normal_azimuth_angle -= 360.0

    shd_vaa = 90.0 - normal_azimuth_angle
    if shd_vaa < 0.0:
        shd_vaa += 360.0

    return shd_vaa


def gen_mie_file_wc(
        wavelength_s,
        wavelength_e,
        fname=None,
        pol_tag='F', # unpolarized
        par_tag='W', # water
        avg_tag='C', # central wavelength
        dist_tag='G', # gamma distribution
        alpha_tag='7 i',
        Nref=25,
        ref_s=1.0,
        ref_e=25.0,
        ref_tag='F', # even-spaced r_e
        ref_max=50.0,
        put_exe='put',
        mie_exe='make_mie_table',
        overwrite=False,
        ):

    if fname is None:

        fdir = f"{er3t.common.fdir_data_tmp}/shdom"
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = f"{fdir}/shdom-mie_{par_tag}_{pol_tag}_{wavelength_s:.4f}-{wavelength_e:.4f}.txt"

    if (not os.path.exists(fname)) or overwrite:

        wavelength_s /= 1000.0 #convert to micron
        wavelength_e /= 1000.0 #convert to micron

        command = f'{put_exe}\
 "{pol_tag}" "{wavelength_s:15.8e} {wavelength_e:15.8e}" "{par_tag}" "{avg_tag}"\
 "{dist_tag}" "{alpha_tag}"\
 "{Nref} {ref_s:.2f} {ref_e:.2f}"\
 "{ref_tag}" "{ref_max:.2f}"\
 "{fname}"\
 | {mie_exe}'

        os.system(command)

    return fname

def gen_mie_file_ic(
        wavelength_s,
        wavelength_e,
        fname=None,
        pol_tag='F', # unpolarized
        par_tag='I', # water
        avg_tag='C', # central wavelength
        dist_tag='G', # gamma distribution
        alpha_tag='7 i',
        Nref=2,
        ref_s=100.0,
        ref_e=150.0,
        ref_tag='F', # even-spaced r_e
        ref_max=200.0,
        put_exe='put',
        mie_exe='make_mie_table',
        overwrite=False,
        ):

    if fname is None:

        fdir = f"{er3t.common.fdir_data_tmp}/shdom"
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = f"{fdir}/shdom-mie_{par_tag}_{pol_tag}_{wavelength_s:.4f}-{wavelength_e:.4f}.txt"

    if (not os.path.exists(fname)) or overwrite:

        wavelength_s /= 1000.0 #convert to micron
        wavelength_e /= 1000.0 #convert to micron

        command = f'{put_exe}\
 "{pol_tag}" "{wavelength_s:15.8e} {wavelength_e:15.8e}" "{par_tag}" "{avg_tag}"\
 "{dist_tag}" "{alpha_tag}"\
 "{Nref} {ref_s:.2f} {ref_e:.2f}"\
 "{ref_tag}" "{ref_max:.2f}"\
 "{fname}"\
 | {mie_exe}'

        os.system(command)

    return fname


def gen_ext_file(
        fname,
        cld0,
        postfix='.sHdOmNG-ext',
        fname_atm_1d=None,
        ):

    # retrieve optical properties
    #╭────────────────────────────────────────────────────────────────────────────╮#
    cer = cld0.lay['cer']['data']
    ext = cld0.lay['extinction']['data'] * 1000.0

    # zgrid = cld0.lay['altitude']['data'] + cld0.lay['thickness']['data']/2.0
    zgrid = cld0.lev['altitude']['data'][1:]
    # zgrid = cld0.lev['altitude']['data'][:-1]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate extinction file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    temp = cld0.lay['temperature']['data']

    Nx, Ny, Nz = ext.shape

    with open(fname, 'w') as f:
        f.write('2 parameter extinction file for SHDOM\n')
        f.write('%d %d %d\n' % ext.shape)
        f.write('%.8e %.8e\n' % (cld0.lay['dx']['data'], cld0.lay['dy']['data']))
        f.write('%s\n' % ' '.join([str('%.6f' % alt0) for alt0 in zgrid]))
        f.write('%s\n' % ' '.join([str('%.4f' % np.nanmean(temp[:, :, iz])) for iz in range(Nz)]))

        f.write('! The following provides information for interpreting binary data:\n')
        f.write('! %s\n' % postfix)
        f.write('! %10d,%10d,%10d,%10d\n' % (Nx, Ny, Nz, 2))
        if fname_atm_1d is not None:
            f.write('! %s\n' % fname_atm_1d)

        # save gridded data into binary file
        #╭──────────────────────────────────────────────────────────────╮#
        with open('%s%s' % (fname, postfix), 'wb') as fb:
            # ext.T/cer.T converts the dimention from (Nx, Ny, Nz) to (Nz, Ny, Nx)
            fb.write(struct.pack('<%df' % ext.size, *ext.T.flatten(order='F')))
            fb.write(struct.pack('<%df' % cer.size, *cer.T.flatten(order='F')))
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname


def gen_lwc_file(
        fname,
        cld0,
        q_factor=2.0,
        ):

    # retrieve optical properties
    #╭────────────────────────────────────────────────────────────────────────────╮#
    cer = cld0.lay['cer']['data']

    const0 = 0.75*q_factor/(1000.0*1.0e-6)
    lwc = cld0.lay['extinction']['data']/(const0/cer) * 1000.0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate LWC file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    temp = cld0.lay['temperature']['data']

    Nx, Ny, Nz = lwc.shape

    with open(fname, 'w') as f:
        f.write('2 parameter LWC file for SHDOM\n')
        f.write('%d %d %d\n' % lwc.shape)
        f.write('%.8e %.8e\n' % (cld0.lay['dx']['data'], cld0.lay['dy']['data']))
        f.write('%s\n' % ' '.join([str('%.6f' % alt0) for alt0 in cld0.lay['altitude']['data']]))
        f.write('%s\n' % ' '.join([str('%.4f' % np.mean(temp[:, :, iz])) for iz in range(Nz)]))

        # save gridded data into ascii file
        #╭──────────────────────────────────────────────────────────────╮#
        for ix in np.arange(Nx):
            for iy in np.arange(Ny):
                for iz in np.arange(Nz):
                    f.write('%d %d %d %.6e %.6e\n' % ((ix+1), (iy+1), (iz+1), lwc[ix, iy, iz], cer[ix, iy, iz]))
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname


def gen_mie_file_from_nc(
        wavelength_s,
        wavelength_e,
        fname=None,
        fname_nc='%s/wc.sol.mie.cdf' % er3t.common.fdir_data_pha,
        pol_tag='F', # unpolarized
        par_tag='W', # water
        overwrite=True,
        ):

    if er3t.common.has_netcdf4:
        from netCDF4 import Dataset
    else:
        er3t.common.logger.error('Need netCDF4 to proceed.')

    if fname is None:

        fdir = '%s/shdom' % er3t.common.fdir_data_tmp
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = '%s/shdom-mie-nc_%s_%s_%.4f-%.4f.txt' % (fdir, par_tag, pol_tag, wavelength_s, wavelength_e)

    if (not os.path.exists(fname)) or overwrite:

        wavelength_s /= 1000.0 #convert to micron
        wavelength_e /= 1000.0 #convert to micron
        wvl = (wavelength_s+wavelength_e)/2.0

        # read data from nc file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f0 = Dataset(fname_nc, 'r')
        param_alpha = f0.getncattr('param_alpha')
        size_distr = f0.getncattr('size_distr')
        parameterization = f0.getncattr('parameterization')

        wavelen = f0.variables['wavelen'][:]
        index_wvl = np.argmin(np.abs(wavelen-wvl))

        reff = f0.variables['reff'][:]
        refre = f0.variables['refre'][:][index_wvl]
        refim = f0.variables['refim'][:][index_wvl]
        rho = f0.variables['rho'][:]

        ext = f0.variables['ext'][:][index_wvl, :]
        ssa = f0.variables['ssa'][:][index_wvl, :]

        pmom = f0.variables['pmom'][:][index_wvl, :, 0, :]

        f0.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        with open(fname, 'w') as f:
            f.write('! %s scattering table vs. effective radius (LWC=1 g/m^3)\n' % parameterization.title())
            f.write('    %.3f    %.3f  wavelength range (micron)\n' % (wavelength_s, wavelength_e))
            f.write(' %.3f  W   particle density (g/cm^3) and type (Water, Ice, Aerosol)\n' % rho.mean())
            f.write('  %.6e %.6e  particle index of refraction\n' % (refre, refim))
            f.write('%.6f %s shape parameter\n' % (param_alpha, size_distr.replace('.', '')))
            f.write('  %d    %.3f   %.3f  number, starting, ending effective radius\n' % (reff.size, reff[0], reff[-1]))

            for i, reff0 in enumerate(reff):
                pmom0 = pmom[i, :]
                logic = np.logical_not(np.isnan(pmom0)) & np.logical_not(np.isinf(pmom0))
                Nmom0 = logic.sum()

                pmom0_str = er3t.util.nice_array_str(pmom0[:Nmom0], numPerLine=200, useSci=True)

                f.write('  %7.4f    %.6e  %.12f   %4d  Reff  Ext  Alb  Nrank\n' % (reff0, ext[i], ssa[i], Nmom0-1))
                f.write('%s' % pmom0_str)

    return fname


def gen_ice_file_from_nc(
        wavelength_s,
        wavelength_e,
        fname=None,
        fname_nc=f"/Users/hchen/Work/soft/libradtran/v2.0.5/data/ic/baum/ic.sol.baum.cdf",
        pol_tag='F', # unpolarized
        par_tag='I', # water
        overwrite=True,
        ):

    if er3t.common.has_netcdf4:
        from netCDF4 import Dataset
    else:
        er3t.common.logger.error('Need netCDF4 to proceed.')

    if fname is None:

        fdir = f"{er3t.common.fdir_data_tmp}/shdom"
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = f"{fdir}/shdom-ice-nc_{par_tag}_{pol_tag}_{wavelength_s:.4f}-{wavelength_e:.4f}.txt"

    # if (not os.path.exists(fname)) or overwrite:
    if True:

        wavelength_s /= 1000.0 #convert to micron
        wavelength_e /= 1000.0 #convert to micron
        wvl = (wavelength_s+wavelength_e)/2.0

        # read data from nc file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f0 = Dataset(fname_nc, 'r')
        param_alpha = 0.0
        size_distr = 'L'
        parameterization = f0.getncattr('parameterization')

        wavelen = f0.variables['wavelen'][:]
        index_wvl = np.argmin(np.abs(wavelen-wvl))

        reff = f0.variables['reff'][:]
        refre = f0.variables['refre'][:][index_wvl]
        refim = f0.variables['refim'][:][index_wvl]
        rho = f0.variables['rho'][:]

        ext = f0.variables['ext'][:][index_wvl, :]
        ssa = f0.variables['ssa'][:][index_wvl, :]

        pmom = f0.variables['pmom'][:][index_wvl, :, 0, :]
        ang = f0.variables['theta'][:][index_wvl, :, 0, :]
        Nang = f0.variables['ntheta'][:][index_wvl, :, 0]
        pha = f0.variables['phase'][:][index_wvl, :, 0, :]

        f0.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = False
        if plot:
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
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            # fig.suptitle('Figure')
            # plot1
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(111)
            for i, reff0 in enumerate(reff[:1]):
                Nang0 = Nang[i]
                ang0 = ang[i, :Nang0]
                pha0 = pha[i, :Nang0]

                Nmom0 = 2001
                pmom0 = er3t.pre.pha.phase2pmom(ang0, pha0, Nleg=Nmom0, Ngauss=200)
                pha1 = er3t.pre.pha.pmom2phase(pmom0, np.cos(np.deg2rad(ang0)))

                ax1.plot(ang0, pha0, lw=2.0, color='k')
                ax1.plot(ang0, pha1, lw=1.0, color='r')
            # ax1.set_xlim((0, 1))
            # ax1.set_ylim((0, 1))
            ax1.set_yscale('log')
            # ax1.set_xlabel('X')
            # ax1.set_ylabel('Y')
            # ax1.set_title('Plot1')
            # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
            # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
            #╰──────────────────────────────────────────────────────────────╯#
            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.35, wspace=0.35)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function': sys._getframe().f_code.co_name, 'Date': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
            fname_fig = f"{_metadata_['Function']}.png"
            plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
            #╰──────────────────────────────────────────────────────────────╯#
            plt.show()
            sys.exit()
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        with open(fname, 'w') as f:
            f.write(f"! {parameterization.title()} scattering table vs. effective radius (LWC=1 g/m^3)\n")
            f.write(f"    {wavelength_s:.3f}    {wavelength_e:.3f}  wavelength range (micron)\n")
            f.write(f" {rho.mean():.3f}  I   particle density (g/cm^3) and type (Water, Ice, Aerosol)\n")
            f.write(f"  {refre:.6e} {refim:.6e}  particle index of refraction\n")
            # f.write(f"{param_alpha:.6f} {size_distr.replace('.', '')} shape parameter\n")
            f.write(f"7.00000 gamma size distribution shape parameter\n")
            f.write(f"  {reff.size:d}    {reff[0]:.3f}   {reff[-1]:.3f}  number, starting, ending effective radius\n")

            for i, reff0 in enumerate(reff):

                pmom0 = pmom[i, :]
                logic = np.logical_not(np.isnan(pmom0)) & np.logical_not(np.isinf(pmom0))
                Nmom0 = logic.sum()

                # Nang0 = Nang[i]
                # ang0 = ang[i, :Nang0]
                # pha0 = pha[i, :Nang0]

                # Nmom0 = 2001
                # pmom0 = er3t.pre.pha.phase2pmom(ang0, pha0, Nleg=Nmom0, Ngauss=200)

                pmom0_str = er3t.util.nice_array_str(pmom0[:Nmom0], numPerLine=200, useSci=True)

                f.write('  %7.4f    %.6e  %.12f   %4d  Reff  Ext  Alb  Nrank\n' % (reff0, ext[i], ssa[i], Nmom0-1))
                f.write('%s' % pmom0_str)

    return fname


def gen_sen_file(
        fname,
        data,
        postfix='.sHdOmNG-sen',
        ):

    params = data.keys()
    Nparam = len(params)

    N = 0
    for param in params:
        N += data[param].size

    if (N%Nparam != 0):
        msg = f"Error [gen_sen_file]: the size of sensor parameters does NOT match."
        raise OSError(msg)
    Ndata = data[param].size

    data_new = np.zeros((Nparam, Ndata), dtype=np.float32)
    for i, param in enumerate(params):
        data_new[i, :] = data[param].ravel()

    header = f"{Nparam}-parameter ({'|'.join(params)}) sensor file for SHDOM"

    # generate extinction file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    with open(fname, 'w') as f:
        f.write(f"{header}\n")

        f.write( "! The following provides information for interpreting binary data:\n")
        f.write(f"! {postfix}\n")
        f.write(f"! {Nparam:10d},{Ndata:10d}\n")

        # save gridded data into binary file
        #╭──────────────────────────────────────────────────────────────╮#
        with open('%s%s' % (fname, postfix), 'wb') as fb:
            fb.write(struct.pack(f"<{Nparam*Ndata}f", *data_new.flatten(order='F')))
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname


if __name__ == '__main__':

    fname = gen_ice_file_from_nc(550.0, 550.0, par_tag='I', fname_nc='/Users/hchen/Work/soft/libradtran/v2.0.5/data/ic/baum/ic.sol.baum.cdf')
    print(fname)

    pass
