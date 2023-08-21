==================
Command Line Tools
==================

EaR³T offers command line tools under ``<root>/bin``.
Current available tools are:

Data Tool - ``lss``
-------------------

  This tool can be used to effortlessly display **brief** dataset information contained within a hierarchical
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

  This tool can be used to effortlessly display **detailed** dataset information contained within a hierarchical
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

Satellite Products Download Tool - ``sdown``
--------------------------------------------

  This tool can be used to effortlessly download satellite data (currently supports MODIS and VIIRS data archived on
  LAADS DAAC and LANCE, and satellite RGB imageries from NASA WorldView) for a user specified
  date and region. The development of ``sdown`` is led by Vikas Nataraja.

  Setup may be required to start using ``sdown``: 

  .. code-block:: bash

    python setup.py develop

  Once setup, sdown should be available via Terminal. To test if setup is correct, use:
  
  .. code-block:: bash

    sdown --run_demo

  The command above should download sample product files from the LAADS DAAC servers and store them in a directory called ``sat-data/``.

  Example Usage:
  To download MODIS-Terra and VIIRS-SNPP L1b products and a Worldview RGB image near Papua New Guinea/north Western Australia for March 26, 2023:

  .. code-block:: bash

    sdown --date 20230326 --lons 127 132 --lats -14 -10 --products mod021km vnp02mod modrgb vnprgb --verbose

  The following command line arguments are available to customize (use ``sdown --help`` to view all options in detail):
  
  .. code-block:: bash
    
    -f , --fdir           Directory where the files will be downloaded
                          By default, files will be downloaded to 'sat-data/'
                          
    -v, --verbose         Pass --verbose if status of your request should be reported frequently.

    -d , --date           Date for which you would like to download data. Use yyyymmdd format.
                          Example: --date 20210404
                          
    -t , --start_date     The start date of the range of dates for which you would like to download data. Use yyyymmdd format.
                          Example: --start_date 20210404
                          
    -u , --end_date       The end date of the range of dates for which you would like to download data. Use yyyymmdd format.
                          Example: --end_date 20210414
                          
    -e  [ ...], --extent  [ ...]
                          Extent of region of interest 
                          lon1 lon2 lat1 lat2 in West East South North format.
                          Example:  --extent -10 -5 25 30
                          
    -x  [ ...], --lons  [ ...]
                          The west-most and east-most longitudes of the region.
                          Alternative to providing the first two terms in `extent`.
                          Example:  --lons -10 -5
                          
    -y  [ ...], --lats  [ ...]
                          The south-most and north-most latitudes of the region.
                          Alternative to providing the last two terms in `extent`.
                          Example:  --lats 25 30
                          
    -n, --nrt             Pass --nrt if Near Real Time products should be downloaded.
                          This is disabled by default to automatically download standard products.
                          Currently, only MODIS NRT products can be downloaded.
    -p  [ ...], --products  [ ...]
                          Short prefix (case insensitive) for the product name.
                          Example:  --products MOD02QKM

  
  We currently support the download of the following products:

  .. code-block:: bash

    MOD02QKM:  Level 1b 250m (Terra) radiance product
    MYD02QKM:  Level 1b 250m (Aqua)  radiance product
    MOD02HKM:  Level 1b 500m (Terra) radiance product
    MYD02HKM:  Level 1b 500m (Aqua)  radiance product
    MOD021KM:  Level 1b 1km (Terra)  radiance product
    MYD021KM:  Level 1b 1km (Aqua)   radiance product
    MOD06_L2:  Level 2 (Terra) cloud product
    MYD06_L2:  Level 2 (Aqua)  cloud product
    MOD35_L2:  Level 2 (Terra) cloud mask product
    MYD35_L2:  Level 2 (Aqua)  cloud mask product
    MOD03   :  Solar/viewing geometry (Terra) product
    MYD03   :  Solar/viewing geometry (Aqua)  product
    MODRGB  :  True-Color RGB (Terra) imagery, useful for visuzliation
    MYDRGB  :  True-Color RGB (Aqua)  imagery, useful for visuzliation
    MCD43   :  Level 3 surface product MCD43A3
    VNPRGB  :  True-Color RGB (S-NPP) imagery, useful for visuzliation
    VJ1RGB  :  True-Color RGB (NOAA-20) imagery, useful for visuzliation
    VNP02IMG:  Level 1b 375m (S-NPP) radiance product
    VJ102IMG:  Level 1b 375m (NOAA-20) radiance product
    VNP03IMG:  Solar/viewing geometry 375m (S-NPP) product
    VJ103IMG:  Solar/viewing geometry 375m (NOAA-20) product
    VNP02MOD:  Level 1b 750m (S-NPP) radiance product
    VJ102MOD:  Level 1b 750m (NOAA-20) radiance product
    VNP03MOD:  Solar/viewing geometry 750m (S-NPP) product
    VJ103MOD:  Solar/viewing geometry 750m (NOAA-20) product
    VNP_CLDPROP_L2: Level 2 (S-NPP) cloud properties product
    VJ1_CLDPROP_L2: Level 2 (NOAA-20) cloud properties product

  We are working on supporting more products via this tool.

  
  




