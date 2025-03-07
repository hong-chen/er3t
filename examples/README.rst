Examples
~~~~~~~~

**The code in this directory is under active development. Please check for the latest code status before use.**

Additionally, these example codes require auxiliary data to run.
To download the required data, under current directory (``examples``) where it contains ``install-examples.sh``,
type in the following

::

    bash install-examples.sh

|

    If ``install-examples.sh`` fails to download the data from Google Drive for any reason, you can download the required data manually
    from `here <https://drive.google.com/file/d/1Oov75VffmuQSljxjoOS6q6egmfT6CmkI/view?usp=share_link>`_.

    After you download the file (``er3t-data-examples.tar.gz``), put it under ``er3t/examples`` directory where
    it contains ``install-examples.sh``, then run the command ``bash install-examples.sh`` through a terminal again.

|

Codes ``00`` can be used for quickly adapting the usage of ``er3t.rtm.mca`` (MCARaTS) and ``er3t.rtm.lrt`` (libRadtran).

The figure results for each example are provided in ``check`` directory (available after installing auxiliary data)
for validation.


|
|

=====================
00_er3t_mca.py
=====================

This program contains various examples using LES data.

#. ``example_01_flux_clear_sky``

   An example that calculates flux profile (Nz) under clear-sky condition.


#. ``example_02_flux_les_cloud_3d``

   An example that calculates flux fields(Nx, Ny, Nz) using 3D LES cloud field.


#. ``example_03_flux_les_cloud_3d_aerosol_1d``

   An example case that calculates flux fields (Nx, Ny, Nz) using 3D LES cloud field and a user-defined 1D aerosol layer above clouds.


#. ``example_04_flux_les_cloud_3d_aerosol_3d``

   An example case that calculates flux fields (Nx, Ny, Nz) using 3D LES cloud field and a user-defined 3D aerosol layer near surface.


#. ``example_05_rad_les_cloud_3d``

   An example case that calculates radiance field (Nx, Ny) using 3D LES cloud field.

#. ``example_06_rad_cld_gen_hem``

   An example case that calculates radiance field (Nx, Ny) for an artifical 3D cloud field generated by built-in hemispherical cloud generator (`er3t.pre.cld.cld_gen_hem`).


The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each example.
After saving the changes to the file, type in ``python 00_er3t_mca.py`` in a terminal under ``er3t/examples``.

|
|

=====================
00_er3t_lrt.py
=====================

This program contains various test and example cases of calculating radiance and flux using libRadtran.

The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each test case.
After saving the changes to the file, type in ``python 00_er3t_lrt.py`` in a terminal under ``er3t/examples``.
