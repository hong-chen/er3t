Tests
~~~~~


=============
01_test_atm.py
=============

This file include tests for ``atm`` modules under ``../er3t/pre/atm``.


=============
02_test_abs.py
=============

This file include tests for ``abs`` modules under ``../er3t/pre/abs``.


=============
03_test_cld.py
=============

This file include tests for ``cld`` modules under ``../er3t/pre/cld``.




=============
04_pre_mca.py
=============

This program contains various test cases.

1. ``test_flux_clear_sky``

   A test case that calculates flux profile (Nz) under clear-sky condition.


2. ``test_flux_with_les_cloud3d``

   A test case that calculates flux fields(Nx, Ny, Nz) using LES input.


3. ``test_radiance_with_les_cloud3d``

   A test case that calculates radiance field (Nx, Ny) using LES input.


4. ``test_radiance_with_modis_cloud_and_surface``

   A test case that calculates radiance field (Nx, Ny) using MODIS cloud and surface reflectance products.


5. ``test_flux_with_les_cloud3d_aerosol1d``

   A test case that calculates flux fields (Nx, Ny, Nz) using LES input and a user-defined 1D aerosol layer.


6. ``test_flux_with_les_cloud3d_aerosol3d``

   A test case that calculates flux fields (Nx, Ny, Nz) using LES input and a user-defined 3D aerosol layer.


7. ``test_radiance_with_les_cloud3d_aerosol3d``

   A test case that calculates radiance field (Nx, Ny) using LES input and a user-defined 3D aerosol layer.


To run a specific test case, please comment/uncomment corresponding lines in the ``main`` function.
After saving the changes to the file, type in ``python 04_pre_mca.py`` in a terminal under ``er3t/tests``.
