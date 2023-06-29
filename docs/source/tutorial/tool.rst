Command Line Tools
==================

EaR³T offers command line tools under ``<root>/bin``.
Current available tools are:

Data Tool - ``lss``
-------------------

  can effortlessly display **brief** dataset information contained within a hierarchical
  data file (currently supports ``netcdf``, ``hdf5``, ``hdf4``, and ``IDL sav``) on a terminal screen.

  **An Example:**

  In a terminal, type in:

  .. code-block:: bash

     lss MYD09A1.A2019241.h09v05.006.2019250044127.hdf

  One will get:

  .. code-block:: bash

     + HDF4
     sur_refl_b01 ---------- : Dataset  (2400, 2400)
     sur_refl_b02 ---------- : Dataset  (2400, 2400)
     sur_refl_b03 ---------- : Dataset  (2400, 2400)
     sur_refl_b04 ---------- : Dataset  (2400, 2400)
     sur_refl_b05 ---------- : Dataset  (2400, 2400)
     sur_refl_b06 ---------- : Dataset  (2400, 2400)
     sur_refl_b07 ---------- : Dataset  (2400, 2400)
     sur_refl_day_of_year -- : Dataset  (2400, 2400)
     sur_refl_qc_500m ------ : Dataset  (2400, 2400)
     sur_refl_raz ---------- : Dataset  (2400, 2400)
     sur_refl_state_500m --- : Dataset  (2400, 2400)
     sur_refl_szen --------- : Dataset  (2400, 2400)
     sur_refl_vzen --------- : Dataset  (2400, 2400)
     -

|

Data Tool - ``lsa``
-------------------

  can effortlessly display **detailed** dataset information contained within a hierarchical
  data file (currently supports ``netcdf``, ``hdf5``, ``hdf4``, and ``IDL sav``) on a terminal screen.

  **An Example:**

  In a terminal, type in:

  .. code-block:: bash

     lsa MYD09A1.A2019241.h09v05.006.2019250044127.hdf

  One will get:

  .. code-block:: bash

     + HDF4
     1. sur_refl_b01 ------------ : Dataset  (2400, 2400)
       1.01|'_FillValue' -------- : -28672
       1.02|'add_offset' -------- : 0.0
       1.03|'add_offset_err' ---- : 0.0
       1.04|'calibrated_nt' ----- : 5
       1.05|'long_name' --------- : Surface_reflectance_for_band_1
       1.06|'scale_factor' ------ : 0.0001
       1.07|'scale_factor_err' -- : 0.0
       1.08|'units' ------------- : reflectance
       1.09|'valid_range' ------- : [-100, 16000]


     2. sur_refl_b02 ------------ : Dataset  (2400, 2400)
       2.01|'_FillValue' -------- : -28672
       2.02|'add_offset' -------- : 0.0
       2.03|'add_offset_err' ---- : 0.0
       2.04|'calibrated_nt' ----- : 5
       2.05|'long_name' --------- : Surface_reflectance_for_band_2
       2.06|'scale_factor' ------ : 0.0001
       2.07|'scale_factor_err' -- : 0.0
       2.08|'units' ------------- : reflectance
       2.09|'valid_range' ------- : [-100, 16000]
     ...
     -

|

Satellite Tool - ``sdown``
--------------------------

  can automatically download satellite data (currently supports MODIS and VIIRS data archived on
  LAADS DAAC and LANCE, and satellite RGB imageries from NASA WorldView) for a user specified
  date and region. The development of ``sdown`` is led by Vikas Nataraja.
