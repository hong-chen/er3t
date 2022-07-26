Examples
~~~~~~~~

The example codes under this directory require auxiliary data to run.

To download the required data, under current directory (``examples``) where it contains ``install-examples.sh``,
type in the following

::

    bash install-examples.sh

|



=====================
01_oco2_rad-sim.py
=====================




=====================
02_modis_rad-sim.py
=====================




=====================
03_spns_flux-sim.py
=====================




=====================
04_cam_nadir_rad-sim.py
=====================



=====================
05_cnn-les_rad-sim.py
=====================




=====================
06_cam_nadir_flyover.py
=====================





=====================
test_mca.py
=====================

This program contains various test cases.

1. ``test_flux_clear_sky``

   A test case that calculates flux profile (Nz) under clear-sky condition.


2. ``test_flux_with_les_cloud3d``

   A test case that calculates flux fields(Nx, Ny, Nz) using LES input.


3. ``test_radiance_with_les_cloud3d``

   A test case that calculates radiance field (Nx, Ny) using LES input.


4. ``test_flux_with_les_cloud3d_aerosol1d``

   A test case that calculates flux fields (Nx, Ny, Nz) using LES input and a user-defined 1D aerosol layer.


5. ``test_flux_with_les_cloud3d_aerosol3d``

   A test case that calculates flux fields (Nx, Ny, Nz) using LES input and a user-defined 3D aerosol layer.


6. ``test_radiance_with_les_cloud3d_aerosol3d``

   A test case that calculates radiance field (Nx, Ny) using LES input and a user-defined 3D aerosol layer.


To run a specific test case, please comment/uncomment corresponding lines in the ``main`` function.
After saving the changes to the file, type in ``python test_mca.py`` in a terminal under ``er3t/examples``.
