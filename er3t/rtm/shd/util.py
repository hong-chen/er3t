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
        'gen_mie_file',
        'gen_ext_file',
        'gen_lwc_file',
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


def gen_mie_file(
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

        fdir = '%s/shdom' % er3t.common.fdir_data_tmp
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        fname = '%s/shdom-mie_%s_%s_%.4f-%.4f.txt' % (fdir, par_tag, pol_tag, wavelength_s, wavelength_e)

    if (not os.path.exists(fname)) or overwrite:

        wavelength_s /= 1000.0 #convert to micron
        wavelength_e /= 1000.0 #convert to micron

        command = '%s\
 "%s" "%15.8e %15.8e" "%s" "%s"\
 "%s" "%s"\
 "%d %.2f %.2f"\
 "%s" "%.2f"\
 "%s"\
 | %s' %\
            (put_exe,\
            pol_tag,  wavelength_s, wavelength_e, par_tag, avg_tag,\
            dist_tag, alpha_tag,\
            Nref, ref_s, ref_e,\
            ref_tag, ref_max,\
            fname,\
            mie_exe)

        os.system(command)

    return fname


def gen_ext_file(
        fname,
        cld0,
        postfix='.sHdOm-ext',
        fname_atm_1d=None,
        ):

    # retrieve optical properties
    #╭────────────────────────────────────────────────────────────────────────────╮#
    cer = cld0.lay['cer']['data']
    ext = cld0.lay['extinction']['data'] * 1000.0

    # zgrid = cld0.lay['altitude']['data'] + cld0.lay['thickness']['data']/2.0
    # zgrid = cld0.lev['altitude']['data'][1:]
    zgrid = cld0.lev['altitude']['data'][:-1]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate extinction file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    temp = cld0.lay['temperature']['data']

    Nx, Ny, Nz = ext.shape

    with open(fname, 'w') as f:
        f.write('2 parameter extinction file for SHDOM\n')
        f.write('%d %d %d\n' % ext.shape)
        f.write('%.4f %.4f\n' % (cld0.lay['dx']['data'], cld0.lay['dy']['data']))
        f.write('%s\n' % ' '.join([str('%.6f' % alt0) for alt0 in zgrid]))
        f.write('%s\n' % ' '.join([str('%.4f' % np.nanmean(temp[:, :, iz])) for iz in range(Nz)]))

        f.write('! The following provides information for interpreting binary data:\n')
        f.write('! %s\n' % postfix)
        f.write('! %6d,%6d,%6d,%6d\n' % (Nx, Ny, Nz, 2))
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
        f.write('%.4f %.4f\n' % (cld0.lay['dx']['data'], cld0.lay['dy']['data']))
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


if __name__ == '__main__':

    fname = gen_prp_file()
    print(fname)

    # date = datetime.datetime(2019, 9, 2)
    # sza = 34.93346840064262
    # saa = -144.90807471473198
    # vza = 14.44008093613803
    # vaa = -99.99851773953723


    # print('SOLARFLUX: %.6f' % er3t.util.cal_sol_fac(date))
    # print('SOLARMU: %.6f' % np.cos(np.deg2rad(sza)))
    # print('SOLARAZ: %.6f' % cal_shd_saa(saa))

    # print('SENSORMU: %.6f' % np.cos(np.deg2rad(vza)))
    # print('SENSORAZ: %.6f' % cal_shd_vaa(vaa))
    pass
