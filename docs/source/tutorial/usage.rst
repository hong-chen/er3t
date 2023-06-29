=====
Usage
=====

Due to the lack of funding support, only limited documentation has been developed.
Despite the funding difficulties, we provide various examples extend from simple demo to complex research
project under `<root>/examples <https://github.com/hong-chen/er3t/tree/dev/examples>`_, will be referred to
as ``examples``)and hope users can learn the usage of EaR³T from the provided examples.


Examples
~~~~~~~~

.. warning::

    The example code is under active development. Please check for the `latest code status <https://discord.com/channels/681619528945500252/1004090233412923544/1017575066139103293>`_ before use.

To run the example codes, auxiliary data is required.
To download the data, under the example directory (``examples``) where it contains ``install-examples.sh``,
type in the following

    .. code-block:: bash

       bash install-examples.sh

    * If ``install-examples.sh`` fails to download the data from Google Drive for any reason, you can download the required data manually
      from `here <https://drive.google.com/file/d/1Oov75VffmuQSljxjoOS6q6egmfT6CmkI/view?usp=share_link>`_.

      After you download the file (``er3t-data-examples.tar.gz``), put it under ``examples`` directory where
      it contains ``install-examples.sh``, then run the command ``bash install-examples.sh`` through a terminal again.

Codes ``00`` (e.g., ``00_er3t_mca.py``) can be used to perform test runs.

Codes ``01`` to ``04`` can be used to reproduce scientific results discussed in
`Chen et al. (2023) <https://doi.org/10.5194/amt-16-1971-2023>`_.

The figure results for each example are provided in ``examples/check`` directory (available after installing auxiliary data)
for validation.

|

Quick Start
~~~~~~~~~~~

At current stage, we use ``MCARaTS`` as our default radiative transfer solver. To check whether EaR³T has been
successfully installed, one can use ``00_er3t_mca.py`` under ``examples``.

There are total of 6 examples provided in ``00_er3t_mca.py`` (see the following from the code).

.. code-block:: Python

   if __name__ == '__main__':

       # irradiance simulation
       #/-----------------------------------------------------------------------------\
       example_01_flux_clear_sky()
       example_02_flux_les_cloud_3d()
       example_03_flux_les_cloud_3d_aerosol_1d()
       example_04_flux_les_cloud_3d_aerosol_3d()
       #\-----------------------------------------------------------------------------/

       # radiance simulation
       #/-----------------------------------------------------------------------------\
       example_05_rad_les_cloud_3d()
       example_06_rad_cld_gen_hem()
       #\-----------------------------------------------------------------------------/

       pass

``example_01`` to ``example_04`` are irradiance (or flux density) simulations and ``example_05``
and ``example_06`` are radiance simulations. By default, all the simulation runs are enabled.
If you would like to only run selected simulations, simply commented out the unwanted simulations
in the ``00_er3t_mca.py`` code. For example, the following will only run ``example_05``

.. code-block:: Python

   if __name__ == '__main__':

       # irradiance simulation
       #/-----------------------------------------------------------------------------\
       # example_01_flux_clear_sky()
       # example_02_flux_les_cloud_3d()
       # example_03_flux_les_cloud_3d_aerosol_1d()
       # example_04_flux_les_cloud_3d_aerosol_3d()
       #\-----------------------------------------------------------------------------/

       # radiance simulation
       #/-----------------------------------------------------------------------------\
       example_05_rad_les_cloud_3d()
       # example_06_rad_cld_gen_hem()
       #\-----------------------------------------------------------------------------/

       pass

To run the code, type in the following command in a terminal under ``examples``

.. code-block:: bash

   python 00_er3t_mca.py

You would expect something similar to the following appear on your terminal screen as indication
for a successful installation

.. code-block:: text

   Message [cld_les]: Processing </data/hong/mygit/er3t/examples/data/00_er3t_mca/aux/les.nc> ...
   Message [cld_les]: Downscaling data from dimension (480, 480, 100) to (480, 480, 4) ...
   Message [cld_les]: Saving object into </data/hong/mygit/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/les.pk> ...
   Message [pha_mie_wc]: Phase function for 650.00nm has been stored at </data/hong/mygit/er3t/tmp-data/pha/mie/pha_mie_wc_0650.0000nm.pk>.
   Message [mca_sca]: File </data/hong/mygit/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/mca_sca.bin> is created.
   Message [mca_atm_3d]: Creating 3D atm file </data/hong/mygit/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/mca_atm_3d.bin> for MCARaTS ...
   Message [mca_atm_3d]: File </data/hong/mygit/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/mca_atm_3d.bin> is created.
   Message [mcarats_ng]: Created MCARaTS input files under </data/hong/mygit/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/0650/rad_3d>.
   Message [mcarats_ng]: Running MCARaTS to get output files under </data/hong/mygit/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/0650/rad_3d> ...
   ----------------------------------------------------------
                    General Information
                  Simulation : 3D Radiance
                  Wavelength : 650.00 nm (applied SSFR slit)
                  Date (DOY) : 2017-08-13 (225)
          Solar Zenith Angle : 30.0000° (0 at local zenith)
         Solar Azimuth Angle : 45.0000° (0 at north; 90° at east)
         Sensor Zenith Angle : 0.0000° (looking down, 0 straight down)
        Sensor Azimuth Angle : 0.0000° (0 at north; 90° at east)
             Sensor Altitude : 705.0 km
              Surface Albedo : 0.03
              Phase Function : Mie (Water Clouds)
        Domain Size (Nx, Ny) : (480, 480)
         Pixel Res. (dx, dy) : (0.10 km, 0.10 km)
     Number of Photons / Set : 1.0e+08 (weighted over 16 g)
              Number of Runs : 16 (g) * 3 (set)
              Number of CPUs : 12 (used) of 16 (total)
   ----------------------------------------------------------
     0%|                                                     | 0/48 [00:00<?, ?it/s]

After the run is completed, you will have a figure (e.g., ``00_er3t_mca-example_05_rad_les_cloud_3d_3d.png``) created under ``examples``, which you can
use to compare with the same figure under ``examples/check``.

At this point, congratulations! Your EaR³T is ready to go and you have done a successful 3D radiative transfer simulation!

.. note::

    If you encountered any error, please feel free to reach out at `Discord SUPPORT/examples <https://discord.com/channels/681619528945500252/1123343152477110453>`__
    for community support.
