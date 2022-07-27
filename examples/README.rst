Examples
~~~~~~~~

The example codes under this directory require auxiliary data to run.

To download the required data, under current directory (``examples``) where it contains ``install-examples.sh``,
type in the following

::

    bash install-examples.sh

|

Besides showcasing EaR³T, codes ``01`` to ``04`` can be used to reproduce scientific results
discussed in `Chen et al. (2022) <https://doi.org/10.5194/amt-2022-143>`_.

=====================
01_oco2_rad-sim.py
=====================

This code provides realistic radiance simulations for OCO-2 (770 nm) based on publicly available MODIS surface and
cloud products. The procecsses involve
#. download satellite data;
#. process radiative properties for surface and clouds;
#. set up and run 3D radiative transfer model;
#. compare simulations with radiance observations from OCO-2.
Afore-mentioned steps are completely automated with minimum input of date and region of interest specified
by user.

The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each step.
After saving the changes to the file, type in ``python 01_oco2_rad-sim.py`` in a terminal under ``er3t/examples``.

=====================
02_modis_rad-sim.py
=====================

This code provides realistic radiance simulations for MODIS (650 nm) based on publicly available MODIS surface and
cloud products. The procecsses involve
#. download satellite data;
#. process radiative properties for surface and clouds;
#. set up and run 3D radiative transfer model;
#. compare simulations with radiance observations from MODIS.
Afore-mentioned steps are completely automated with minimum input of date and region of interest specified
by user.

The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each step.
After saving the changes to the file, type in ``python 02_modis_rad-sim.py`` in a terminal under ``er3t/examples``.

=====================
03_spns_flux-sim.py
=====================

This code provides realistic irradiance (flux) simulations (745 nm) for one CAMP²Ex flight track based on AHI
cloud products. The procecsses involve
#. partition flight track into mini flight track segments;
#. set up and run 3D radiative transfer model based on AHI cloud product for each mini flight track segment;
#. compare simulations with irradiance (flux) observations from SPN-S.

The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each step.
After saving the changes to the file, type in ``python 03_spns_flux-sim.py`` in a terminal under ``er3t/examples``.

=====================
04_cam_nadir_rad-sim.py
=====================

This code provides realistic radiance simulations (600 nm) for two cloud optical thickness fields derived from
one airborne camera imagery during CAMP²Ex - 1) tradiational IPA retrieved and 2) context-aware CNN retrieved.
#. apply IPA method (Two-Stream Approximation) to retrieve cloud optical thickness from camera imagery;
#. apply CNN method to retrieve cloud optical thickness from camera imagery;
#. set up and run 3D radiative transfer model for the two cloud optical thickness fields;
#. compare simulations with radiance observations from camera.

The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each step.
After saving the changes to the file, type in ``python 04_cam_nadir_rad-sim.py`` in a terminal under ``er3t/examples``.

=====================
05_cnn-les_rad-sim.py
=====================

This code provides realistic radiance simulations based on LES data. It produces extensive training dataset (ground
truth of cloud optical thickness, realistic radiance simulation) for training CNN.
#. artificially create more LES cloud fields through coarsening by factor of 2 and 4;
#. run radiance simulations for all the LES cloud fields (480x480);
#. crop radiance simulations and cloud optical thickness fields into mini tiles (64x64);
#. evenly select mini tiles based on the 1) cloud fraction (average radiance), and 2) cloud
   inhomogeneity (standard deviation of radiance) for training.

The executable lines are located after the line ``if __name__ == __main__:``.

To run the code, please comment/uncomment the line associated with each step.
After saving the changes to the file, type in ``python 05_cnn-les_rad-sim.py`` in a terminal under ``er3t/examples``.

=====================
test_mca.py
=====================

This program contains various test cases using LES data.

#. ``test_flux_clear_sky``

   A test case that calculates flux profile (Nz) under clear-sky condition.


#. ``test_flux_with_les_cloud3d``

   A test case that calculates flux fields(Nx, Ny, Nz) using LES input.


#. ``test_radiance_with_les_cloud3d``

   A test case that calculates radiance field (Nx, Ny) using LES input.


#. ``test_flux_with_les_cloud3d_aerosol1d``

   A test case that calculates flux fields (Nx, Ny, Nz) using LES input and a user-defined 1D aerosol layer.


#. ``test_flux_with_les_cloud3d_aerosol3d``

   A test case that calculates flux fields (Nx, Ny, Nz) using LES input and a user-defined 3D aerosol layer.


#. ``test_radiance_with_les_cloud3d_aerosol3d``

   A test case that calculates radiance field (Nx, Ny) using LES input and a user-defined 3D aerosol layer.


To run a specific test case, please comment/uncomment corresponding lines in the ``main`` function.
After saving the changes to the file, type in ``python test_mca.py`` in a terminal under ``er3t/examples``.
