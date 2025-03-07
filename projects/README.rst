Examples (`code status <https://discord.com/channels/681619528945500252/1004090233412923544/1017575066139103293>`_)
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

Codes ``01`` to ``04`` can be used to reproduce scientific results discussed in
`Chen et al. (2023) <https://doi.org/10.5194/amt-16-1971-2023>`_.

The figure results for each example are provided in ``check`` directory (available after installing auxiliary data)
for validation.


|
|

============================
Get Access to NASA EARTHDATA
============================

Code ``01`` and ``02`` contain some programs that can automatically download satellite data from NASA EARTHDATA.
To get the data access, you will need to register an account with NASA EARTHDATA.

#. Register an account at `NASA EARTHDATA <https://urs.earthdata.nasa.gov>`_ ;

#. Create a ``.netrc`` file under your home directory, e.g., type in ``touch ~/.netrc``;

#. Assume you registered with an username ``abc`` and a password ``123``, in the ``~/.netrc`` file, put in

   ::

     machine urs.earthdata.nasa.gov
     login abc
     password 123

#. Request a token (instructions `here <https://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens/>`_)
   for your EARTHDATA account and store the token under environment variable ``EARTHDATA_TOKEN``, e.g., ``export EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"``.

|
|

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

|
|

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

|
|

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

|
|

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

|
|

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
