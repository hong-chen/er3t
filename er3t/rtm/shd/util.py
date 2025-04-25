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
        'gen_ckd_file',
        'gen_mie_file',
        'gen_ext_file',
        'gen_prp_file',
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


def gen_ckd_file(fname, atm0, abs0, Nband=1):

    with open(fname, 'w') as f:

        f.write('! correlated k-distribution file for SHDOM\n')
        f.write('%d ! number of bands\n' % 1)
        f.write('! Band# | Wave#1 | Wave#2 | SolFlx | Ng | g1 | g2 | ...\n')

        # wave number cm^-1
        wvln_min = 1.0/abs0.wvl_max_*1e7
        wvln_max = 1.0/abs0.wvl_min_*1e7

        sol = (abs0.coef['solar']['data']*abs0.coef['weight']['data']).sum()

        Ng = abs0.coef['weight']['data'].size

        g = ' '.join(['%.6f' % value for value in abs0.coef['weight']['data']])

        for iband in range(Nband):
            f.write('%d %.2f %.2f %.6f %d %s\n' % (iband+1, wvln_min, wvln_max, sol, Ng, g))

        f.write('%d\n' % atm0.lay['altitude']['data'].size)

        f.write('!\n')
        f.write('! Alt [km] | Pres [mb] | Temp [K]\n')

        alt = atm0.lay['altitude']['data'][::-1]
        # pres = atm0.lay['pressure']['data'][::-1]
        # temp = atm0.lay['temperature']['data'][::-1]

        indices_sort = np.argsort(abs0.coef['weight']['data'])
        kabs = abs0.coef['abso_coef']['data'][::-1, indices_sort]

        for j in range(alt.size):
            # f.write('%.6f %.2f %.2f\n' % (alt[j], pres[j], temp[j]))
            f.write('%.6f\n' % (alt[j]))

        f.write('! iBand | iLay | AbsCoef [km^-1]\n')

        for iband in range(Nband):
            for j in range(alt.size):
                kabs_s = ' '.join(['%15.6e' % kabs0 for kabs0 in kabs[j, :]])
                f.write('%d %d %s\n' % (iband+1, j+1, kabs_s))


def gen_ext_file(
        fname,
        cld0,
        postfix='.sHdOm-ext',
        ):

    # retrieve optical properties
    #╭────────────────────────────────────────────────────────────────────────────╮#
    cer = cld0.lay['cer']['data']
    ext = cld0.lay['extinction']['data'] * 1000.0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # generate extinction file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    temp = cld0.lay['temperature']['data']

    Nx, Ny, Nz = ext.shape

    with open(fname, 'w') as f:
        f.write('2 parameter extinction file for SHDOM\n')
        f.write('%d %d %d\n' % ext.shape)
        f.write('%.4f %.4f\n' % (cld0.lay['dx']['data'], cld0.lay['dy']['data']))
        f.write('%s\n' % ' '.join([str('%.4f' % alt0) for alt0 in cld0.lay['altitude']['data']]))
        f.write('%s\n' % ' '.join([str('%.4f' % np.mean(temp[:, :, iz])) for iz in range(Nz)]))

        f.write('! The following provides information for interpreting binary data:\n')
        f.write('! %s\n' % postfix)
        f.write('! %6d,%6d,%6d,%6d\n' % (Nx, Ny, Nz, 2))

        # save gridded data into binary file
        #╭──────────────────────────────────────────────────────────────╮#
        with open('%s%s' % (fname, postfix), 'wb') as fb:
            # ext.T/cer.T converts the dimention from (Nx, Ny, Nz) to (Nz, Ny, Nx)
            fb.write(struct.pack('<%df' % ext.size, *ext.T.flatten(order='F')))
            fb.write(struct.pack('<%df' % cer.size, *cer.T.flatten(order='F')))
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname


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


def gen_prp_file(
        fname,
        wavelength,
        atm0,
        cld0,
        Npha_max=1000,
        asy_tol=1.0e-2,
        pha_tol=1.0e-1,
        pol_tag='U',
        put_exe='put',
        prp_exe='propgen',
        ):

    fname_mie = er3t.rtm.shd.gen_mie_file(wavelength, wavelength)

    fname_ext = er3t.rtm.shd.gen_ext_file(fname.replace('prp', 'ext'), cld0)

    logic_z_extra = np.logical_not(np.array([np.any((atm0.lay['altitude']['data'][i]-cld0.lay['altitude']['data'])<1.0e-6) for i in range(atm0.lay['altitude']['data'].size)]))
    Nz_extra = logic_z_extra.sum()
    z_extra = '%s' % '\n'.join(['%.4e %.4e' % tuple(item) for item in zip(atm0.lay['altitude']['data'][logic_z_extra], atm0.lay['temperature']['data'][logic_z_extra])])

    if len(z_extra) > 1000:
        msg = 'Error [gen_prp_file]: <z_extra> is greater than 1000-character-limit.'
        raise OSError(msg)

    wavelength /= 1000.0

    command = '%s "1"\
 "%s" "1" "F" "%s"\
 "%d" "%.4e" "%.4e"\
 "%15.8e" "%.4f"\
 "%d" "%s"\
 "%s" "%s"\
 | %s' %\
        (put_exe,\
        fname_mie, fname_ext,\
        Npha_max, asy_tol, pha_tol,\
        wavelength, atm0.lev['pressure']['data'][0],\
        Nz_extra, z_extra,\
        pol_tag, fname,\
        prp_exe)

    os.system(command)

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
