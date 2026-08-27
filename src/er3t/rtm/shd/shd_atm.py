import os
import sys
import copy
import struct
import subprocess
import h5py
import numpy as np
from scipy import interpolate

import er3t
from er3t.core.numerics import cal_mol_ext_atm
from er3t.rtm.shd._vertical import (
    convert_propgen_to_layer_level,
    merged_level_grid,
    temperatures_on_grid,
)

__all__ = ["shd_atm_1d", "shd_atm_3d"]


class shd_atm_1d:
    """
    Input:
        atm_obj: keyword argument, default=None, e.g. atm0 = atm_atmmod(levels=np.arange(21))
        abs_obj, keyword argument, default=None, e.g. abs0 = abs_16g(wavelength=600.0, atm_obj=atm0)

    Output:
        self.nml: Python dictionary, ig range from 0 to 15 (0, 1, ..., 15)
            ['NZ']
            ['WAVELEN']
            ['WAVENO']
            ['CKDFILE']
    """

    ID = "SHDOM 1D Atmosphere"

    def __init__(
        self,
        atm_obj=None,
        abs_obj=None,
        fname=None,
        alt_toa=30.0,
        overwrite=True,
        force=False,
        verbose=False,
        quiet=False,
    ):
        self.overwrite = overwrite
        self.verbose = verbose
        self.quiet = quiet

        if atm_obj is None:
            msg = f"Please provide an atm object for <atm_obj>."
            er3t.common.logger.error(msg)
            raise OSError
        else:
            self.atm = atm_obj

        if abs_obj is None:
            msg = f"Please provide an abs object for <abs_obj>."
            er3t.common.logger.error(msg)
            raise OSError
        else:
            self.abs = abs_obj

        self.Ng = self.abs.Ng
        self.wvl_info = self.abs.wvl_info

        self.pre_shd_1d_atm(alt_toa=alt_toa)

        if fname is None:
            fname = "shdom-ckd.txt"
        fname = os.fspath(fname)
        self.fname = fname

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_shd_ckd_file(fname, self.atm, self.abs, alt_toa=alt_toa)
            self.nml["CKDFILE"] = {"data": fname}
        else:
            self.gen_shd_ckd_file(fname, self.atm, self.abs, alt_toa=alt_toa)

    def pre_shd_1d_atm(self, alt_toa=30.0):
        self.nml = {}

        self.z_levels = merged_level_grid(self.atm, alt_toa=alt_toa)

        self.nml["NZ"] = {
            "data": self.z_levels.size,
            "name": "Nz",
            "units": "N/A",
        }

        self.nml["WAVELEN"] = {
            "data": self.abs.wvl / 1000.0,
            "units": "micron",
            "name": "Wavelength",
        }

        wvln_min = 1.0 / self.abs.wvl_max_ * 1e7
        wvln_max = 1.0 / self.abs.wvl_min_ * 1e7
        self.nml["WAVENO"] = {
            "data": f"{wvln_min:.2f} {wvln_max:.2f}",
            "units": "cm^-1",
            "name": "Wave Number Range",
        }

        self.nml["GNDTEMP"] = {
            "data": self.atm.lev["temperature"]["data"][0],
            "units": "K",
            "name": "Surface Temperature",
        }

        # calculate rayleight extinction
        # self.atm_sca = er3t.core.cal_mol_ext(self.abs.wvl/1000.0, self.atm.lev['pressure']['data'][:-1], self.atm.lev['pressure']['data'][1:]) / (self.atm.lay['thickness']['data'])
        self.atm_sca = (
            cal_mol_ext_atm(self.abs.wvl / 1000.0, self.atm)
            / (self.atm.lay["thickness"]["data"])
        )

    def gen_shd_ckd_file(
        self,
        fname,
        atm0,
        abs0,
        Nband=1,
        alt_toa=100.0,
    ):
        if not self.quiet:
            msg = f"Creating 1D correlated-k file <{fname}> for SHDOM ..."
            er3t.common.logger.info(msg)

        with open(fname, "w") as f:
            f.write("! correlated k-distribution file for SHDOM\n")
            f.write(f"{1} ! number of bands\n")
            f.write(
                f"! Band# | Wave#1 [{abs0.wvl_max_:.2f} nm] | Wave#2 [{abs0.wvl_min_:.2f} nm] | Ng | SolFlx1| SolFlx2 | ... | g1 | g2 | ...\n"
            )

            # wave number cm^-1
            wvln_min = 1.0 / abs0.wvl_max_ * 1e7
            wvln_max = 1.0 / abs0.wvl_min_ * 1e7

            Ng = abs0.coef["weight"]["data"].size
            indices_sort = np.argsort(abs0.coef["weight"]["data"])

            sol = " ".join(
                [f"{value:.12f}" for value in abs0.coef["solar"]["data"][indices_sort]]
            )
            wgt = " ".join(
                [f"{value:.12f}" for value in abs0.coef["weight"]["data"][indices_sort]]
            )

            for iband in range(Nband):
                f.write(
                    "%d %.2f %.2f %d %s %s\n"
                    % (iband + 1, wvln_min, wvln_max, Ng, sol, wgt)
                )

            # calculating gas scatter (rayleigh) and gas absorption
            # ╭────────────────────────────────────────────────────────────────────────────╮#
            # altitude
            # ╭──────────────────────────────────────────────────────────────╮#
            thickness = atm0.lay["thickness"]["data"][::-1]
            # CKD profiles remain level-interpolated in aeria3d.  Place the
            # layer-mean coefficients at their physical midpoints so sampling
            # at a model-cell midpoint returns the original layer value.
            zgrid = atm0.lay["altitude"]["data"][::-1]
            # ╰──────────────────────────────────────────────────────────────╯#

            # gas scattering
            # ╭──────────────────────────────────────────────────────────────╮#
            atm_sca = self.atm_sca[::-1]
            # ╰──────────────────────────────────────────────────────────────╯#

            # gas absorption
            # ╭──────────────────────────────────────────────────────────────╮#
            atm_abs = abs0.coef["abso_coef"]["data"][::-1, indices_sort]
            for i in range(atm_abs.shape[0]):
                atm_abs[i, :] = atm_abs[i, :] / thickness[i]
            # ╰──────────────────────────────────────────────────────────────╯#

            # add surface (z=0.0 km)
            # ╭──────────────────────────────────────────────────────────────╮#
            if zgrid[-1] >= 1.0e-6:
                zgrid = np.append(zgrid, 0.0)
                atm_sca = np.append(atm_sca, 0.0)
                atm_abs = np.concatenate(
                    (atm_abs, np.zeros((1, indices_sort.size), dtype=np.float32))
                )
            # ╰──────────────────────────────────────────────────────────────╯#

            # add toa (z=alt_toa[100.0] km)
            # ╭──────────────────────────────────────────────────────────────╮#
            if zgrid[0] <= (alt_toa - 1.0e-6):
                zgrid = np.append(alt_toa, zgrid)
                atm_sca = np.append(0.0, atm_sca)
                atm_abs = np.concatenate(
                    (np.zeros((1, indices_sort.size), dtype=np.float32), atm_abs)
                )
            # ╰──────────────────────────────────────────────────────────────╯#
            # ╰────────────────────────────────────────────────────────────────────────────╯#

            # remove atmosphere by turning off gas absorption and scattering
            # ╭────────────────────────────────────────────────────────────────────────────╮#
            # atm_abs[...] = 0.0
            # atm_sca[...] = 0.0
            # ╰────────────────────────────────────────────────────────────────────────────╯#

            f.write(f"{int(zgrid.size)}\n")

            f.write("!\n")
            f.write("! Alt [km] | ScaCoef [km^-1]\n")

            for j in range(zgrid.size):
                f.write(f"{zgrid[j]:10.6f} {atm_sca[j]:15.6e}\n")

            f.write("! iBand | iLay | AbsCoef [km^-1]\n")

            for iband in range(Nband):
                for j in range(zgrid.size):
                    atm_abs_s = " ".join(
                        [f"{atm_abs0:15.6e}" for atm_abs0 in atm_abs[j, :]]
                    )
                    f.write(f"{int(iband + 1):4} {int(j + 1):4} {atm_abs_s}\n")

        self.nml["CKDFILE"] = {"data": fname}

        if not self.quiet:
            msg = f"File <{fname}> is created."
            er3t.common.logger.info(msg)


class shd_atm_3d:
    """
    Input:
        atm_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        cld_obj=: keyword argument, default=None, cloud object, for example, cld_obj = cld_les(fname='les.pk')
        abs_obj=: keyword argument, default=None, absorption object

        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['NX']
                ['NY']
                ['NZ']
                ['WAVELEN']
                ['WAVENO']
                ['GNDTEMP']
                ['PROPFILE']

        self.gen_shd_prp_file: method to create SHDOM property file of 3d atmosphere
    """

    ID = "SHDOM 3D Atmosphere"

    def __init__(
        self,
        atm_obj=None,
        abs_obj=None,
        cld_obj=None,
        fname=None,
        alt_toa=30.0,
        overwrite=True,
        force=False,
        verbose=False,
        quiet=False,
        fname_atm_1d=None,
    ):
        self.overwrite = overwrite
        self.verbose = verbose
        self.quiet = quiet

        if atm_obj is None:
            msg = f"Please provide an atm object for <atm_obj>."
            er3t.common.logger.error(msg)
            raise OSError
        else:
            self.atm = atm_obj

        if abs_obj is None:
            msg = f"Please provide an abs object for <abs_obj>."
            er3t.common.logger.error(msg)
            raise OSError
        else:
            self.abs = abs_obj

        if cld_obj is None:
            msg = f"Please provide an cld object for <cld_obj>."
            er3t.common.logger.error(msg)
            raise OSError
        else:
            self.cld = cld_obj

        # Go through cloud layers and check whether atm is compatible
        # e.g., whether the sizes of the Altitude array (z) and Thickness array (dz) are the same
        if (
            self.cld.lay["altitude"]["data"].size
            != self.cld.lay["thickness"]["data"].size
        ):  # layer number
            msg = f"Incorrect number of cloud layers ({self.cld.lay['altitude']['data'].size:d}) vs layer thicknesses ({self.cld.lay['thickness']['data'].size:d})."
            er3t.common.logger.error(msg)
            raise ValueError

        self.pre_shd_3d_atm(alt_toa=alt_toa)

        if fname is None:
            fname = "shdom-prp.txt"
        fname = os.fspath(fname)
        self.fname = fname

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_shd_prp_file(
                    fname, self.abs.wvl, self.atm, self.cld, fname_atm_1d=fname_atm_1d
                )
            self.nml["PROPFILE"] = {"data": fname}
        else:
            self.gen_shd_prp_file(
                fname, self.abs.wvl, self.atm, self.cld, fname_atm_1d=fname_atm_1d
            )

    def pre_shd_3d_atm(self, alt_toa=30.0):
        self.nml = {}

        self.nml["NX"] = copy.deepcopy(self.cld.lay["nx"])
        self.nml["NY"] = copy.deepcopy(self.cld.lay["ny"])

        self.nml["WAVELEN"] = {
            "data": self.abs.wvl / 1000.0,
            "units": "micron",
            "name": "Wavelength",
        }

        wvln_min = 1.0 / self.abs.wvl_max_ * 1e7
        wvln_max = 1.0 / self.abs.wvl_min_ * 1e7
        self.nml["WAVENO"] = {
            "data": f"{wvln_min:.2f} {wvln_max:.2f}",
            "units": "cm^-1",
            "name": "Wave Number Range",
        }

        self.nml["GNDTEMP"] = {
            "data": self.atm.lev["temperature"]["data"][0],
            "units": "K",
            "name": "Surface Temperature",
        }

        self.z_levels = merged_level_grid(self.atm, self.cld, alt_toa=alt_toa)
        self.z_layers = 0.5 * (self.z_levels[:-1] + self.z_levels[1:])
        self.temperature_levels = temperatures_on_grid(self.atm, self.z_levels)
        self.temperature_layers = temperatures_on_grid(self.atm, self.z_layers)

        # propgen receives one optical sample at each physical layer midpoint.
        # The generated file is converted to LAYER optics / LEVEL temperature
        # ownership after phase-function mixing, so no shifted extra levels are
        # needed.
        self.Nz_extra = 0
        self.z_extra = ""

        self.nml["NZ"] = {
            "data": self.z_levels.size,
            "name": "Nz",
            "units": "N/A",
        }

    def gen_shd_prp_file(
        self,
        fname,
        wavelength,
        atm0,
        cld0,
        Npha_max=1000,
        asy_tol=1.0e-2,
        pha_tol=1.0e-1,
        pol_tag="U",
        prp_exe="propgen",
        fname_atm_1d=None,
    ):
        # fname_mie = er3t.rtm.shd.gen_mie_file_from_nc(wavelength, wavelength)
        if "ice" in cld0.ID.lower():
            # fname_mie = er3t.rtm.shd.gen_mie_file_ic(wavelength, wavelength)
            fname_mie = er3t.rtm.shd.gen_ice_file_from_nc(wavelength, wavelength)
            # fname_mie = er3t.rtm.shd.gen_mie_file_from_nc(wavelength, wavelength)
        else:
            fname_mie = er3t.rtm.shd.gen_mie_file_wc(wavelength, wavelength)

        property_path = os.path.abspath(fname)
        if fname_atm_1d is not None:
            property_name = os.path.basename(property_path)
            ext_name = property_name.replace("prp", "ext")
            if ext_name == property_name:
                stem, suffix = os.path.splitext(property_name)
                ext_name = f"{stem}-ext{suffix}"
            fname_inp = er3t.rtm.shd.gen_ext_file(
                os.path.join(os.path.dirname(property_path), ext_name),
                cld0,
                fname_atm_1d=fname_atm_1d,
                zgrid=self.z_layers,
                temperature=self.temperature_layers,
            )
        else:
            property_name = os.path.basename(property_path)
            lwc_name = property_name.replace("prp", "lwc")
            if lwc_name == property_name:
                stem, suffix = os.path.splitext(property_name)
                lwc_name = f"{stem}-lwc{suffix}"
            fname_inp = er3t.rtm.shd.gen_lwc_file(
                os.path.join(os.path.dirname(property_path), lwc_name),
                cld0,
                zgrid=self.z_layers,
                temperature=self.temperature_layers,
            )

        if len(self.z_extra) > 5000:
            msg = f"<z_extra> [length={len(self.z_extra)}] is greater than 5000-character-limit."
            er3t.common.logger.error(msg)
            raise OSError

        wavelength /= 1000.0

        propgen_input = [
            "1",
            os.fspath(fname_mie),
            "1",
            "F",
            os.fspath(fname_inp),
            str(Npha_max),
            f"{asy_tol:.4e}",
            f"{pha_tol:.4e}",
            f"{wavelength:.8e}",
            f"{atm0.lev['pressure']['data'][0]:.4f}",
            str(self.Nz_extra),
        ]
        if self.z_extra:
            propgen_input.extend(self.z_extra.splitlines())
        propgen_input.extend([pol_tag, fname])

        if not self.quiet:
            msg = f"Creating 3D property file <{fname}> for SHDOM ..."
            er3t.common.logger.info(msg)

        try:
            subprocess.run(
                [prp_exe],
                input="\n".join(propgen_input) + "\n",
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"propgen failed while creating <{fname}>.") from error
        if not os.path.exists(fname):
            raise RuntimeError(f"propgen did not create <{fname}>.")

        convert_propgen_to_layer_level(
            fname,
            self.z_levels,
            self.temperature_levels,
        )

        if not self.quiet:
            msg = f"File <{fname}> is created."
            er3t.common.logger.info(msg)

        self.nml["PROPFILE"] = {"data": fname}


if __name__ == "__main__":
    pass
