import os
import sys
import copy
import struct
import warnings
import h5py
import numpy as np

import er3t.common


__all__ = ["drt_sfc_2d"]


class drt_sfc_2d:
    """
    Input:
        atm_obj=: keyword argument, default=None, atmosphere object, for example, atm_obj = atm_atmmod(fname='atm.pk')
        sfc_obj=: keyword argument, default=None, surface object, for example, sfc_obj = sfc_sat(fname='mod09.pk')
        verbose=: keyword argument, default=False, verbose tag
        quiet=  : keyword argument, default=False, quiet tag

    Output:
        self.nml: Python dictionary
                ['Sfc_nxb']
                ['Sfc_nyb']
                ['Sfc_tmps2d']
                ['Sfc_jsfc2d']
                ['Sfc_psfc2d']

        self.gen_drt_2d_sfc_file: method to create binary file of 2d surface

        self.save_h5: method to save data into HDF5 file
    """

    def __init__(
        self,
        atm_obj=None,
        sfc_obj=None,
        fname=None,
        overwrite=True,
        force=False,
        verbose=False,
        quiet=False,
    ):
        self.overwrite = overwrite
        self.verbose = verbose
        self.quiet = quiet

        if atm_obj is None:
            msg = "\nError [drt_sfc_2d]: Please provide an <atm> object for <atm_obj>."
            raise OSError(msg)
        else:
            self.atm = atm_obj

        if sfc_obj is None:
            msg = "\nError [drt_sfc_2d]: Please provide an <sfc> object for <sfc_obj>."
            raise OSError(msg)
        else:
            self.sfc = sfc_obj

        self.pre_drt_2d_sfc()

        if fname is None:
            fname = "drtom-sfc_2d.txt"

        if not self.overwrite:
            if (not os.path.exists(fname)) and (not force):
                self.gen_drt_2d_sfc_file(fname)
            self.nml["SFCFILE"] = {"data": fname}
        else:
            self.gen_drt_2d_sfc_file(fname)

    def pre_drt_2d_sfc(self):
        self.nml = {}

        self.nml["NX"] = copy.deepcopy(self.sfc.data["nx"])
        self.nml["NY"] = copy.deepcopy(self.sfc.data["ny"])
        self.nml["dx"] = copy.deepcopy(self.sfc.data["dx"])
        self.nml["dy"] = copy.deepcopy(self.sfc.data["dy"])

        self.Nx = self.nml["NX"]["data"]
        self.Ny = self.nml["NY"]["data"]
        self.dx = self.nml["dx"]["data"]
        self.dy = self.nml["dy"]["data"]

        if (self.Nx == 1) and (self.Ny == 1):
            self.ID = "Homogeneous Surface"
        else:
            self.ID = f"2D [{self.Nx}x{self.Ny}] Domain"

        if "lambertian" in self.sfc.data["sfc"]["name"].lower():
            self.nml["header"] = dict(
                data="L", name="Header for DISORT Surface File", units="N/A"
            )
            self.sfc_data = self.sfc.data["sfc"]["data"]

            self.ID = f"{self.ID} (Lambertian, for DISORT)"

        elif "brdf-lsrt-jiao" in self.sfc.data["sfc"]["name"].lower():
            self.nml["header"] = dict(
                data="J", name="Header for DISORT Surface File", units="N/A"
            )
            self.sfc_data = self.sfc.data["sfc"]["data"]

            self.ID = f"{self.ID} (LSRT-Jiao Snow, for DISORT)"

        elif "brdf-lsrt" in self.sfc.data["sfc"]["name"].lower():
            self.nml["header"] = dict(
                data="T", name="Header for DISORT Surface File", units="N/A"
            )
            self.sfc_data = self.sfc.data["sfc"]["data"]

            self.ID = f"{self.ID} (LSRT Land, for DISORT)"

        elif "brdf-ocean" in self.sfc.data["sfc"]["name"].lower():
            self.nml["header"] = dict(
                data="O", name="Header for DISORT Surface File", units="N/A"
            )
            self.sfc_data = self.sfc.data["sfc"]["data"]

            self.ID = f"{self.ID} (Cox-Munk Ocean, for DISORT)"

        elif "brdf-mixed" in self.sfc.data["sfc"]["name"].lower():
            self.nml["header"] = dict(
                data="X", name="Header for DISORT Surface File", units="N/A"
            )
            self.sfc_data = self.sfc.data["sfc"]["data"]
            self.ID = f"{self.ID} (Mixed BRDF Surface, for DISORT)"

        else:
            msg = "\nError [drt_sfc_2d]: Cannot determine surface type - currently only supports Lambertian surface and LSRT BRDF surface (e.g., MCD43A1)."
            raise OSError(msg)

    def gen_drt_2d_sfc_file(
        self,
        fname,
        postfix=".disort-sfc",
    ):
        fname = os.path.abspath(fname)
        Nparam = self.sfc_data.shape[-1]
        temp_sfc = self.atm.lay["temperature"]["data"][0]

        if not self.quiet:
            er3t.common.logger.info(
                f"Message [drt_sfc_2d]: Creating 2D SFCFile <{fname}> for DISORT..."
            )

        with open(fname, "w") as f:
            f.write(f"{self.nml['header']['data']}\n")
            f.write(f"{self.Nx} {self.Ny} {self.dx:.8e} {self.dy:.8e}\n")

            if self.Nx * self.Ny <= 36:
                for iy in np.arange(self.Ny):
                    for ix in np.arange(self.Nx):
                        string1 = f"{ix + 1} {iy + 1} {temp_sfc:.2f} "
                        string2 = ("%.8e " * self.sfc_data[ix, iy, :].size) % tuple(
                            self.sfc_data[ix, iy, :]
                        )
                        string3 = "\n"
                        f.write(
                            string1 + string2[:-1] + string3
                        )  # [:-1] is used to get rid of last empty space

            else:
                # add in edge pixels for DISORT
                data = np.zeros(
                    (self.Nx + 1, self.Ny + 1, Nparam + 1), dtype=np.float32
                )
                Ndata_t = self.Nx * self.Ny
                data[:-1, :-1, 0] = np.repeat(temp_sfc, Ndata_t).reshape(
                    self.Nx, self.Ny
                )
                data[:-1, :-1, 1:] = self.sfc_data

                f.write(
                    "! The following provides information for interpreting binary data:\n"
                )
                f.write(f"! {postfix}\n")
                f.write(f"! {self.Nx + 1:10d},{self.Ny + 1:10d},{Nparam + 1:10d}\n")

                with open("%s%s" % (fname, postfix), "wb") as fb:
                    Ndata = data.size
                    # data.T reshapes data from [Nx, Ny, Nparam], to [Nparam, Ny, Nx]
                    fb.write(struct.pack(f"<{Ndata}f", *data.T.flatten(order="F")))

        self.nml["SFCFILE"] = {"data": fname}

        if not self.quiet:
            er3t.common.logger.info(
                "Message [drt_sfc_2d]: File <%s> is created." % fname
            )


if __name__ == "__main__":
    pass
