Version 0.2.0
-------------
Important updates from previous version (v0.1.1):

* Initiates documentation at https://er3t.readthedocs.io/;

* Adds command line tool ``sdown`` for effortlessly downloading satellite data;

* Adds satellite data download support for NASA LANCE;

* Adds time estimation and stamping (accuracy within seconds) for NASA Worldview imagery;

* Implements BRDF surface support for Cox-Munk (ocean) and LSRT (land, e.g., MCD43A1 product);

* Implements gas absorption support for REPTRAN.


Version 0.1.1
-------------
Important updates from previous version (v0.1.0):

* Fixes Mie phase function for water clouds (details see Discord);

* Updates example code under ``examples``;

* Improves satellite data download process.


Version 0.1.0
-------------
First official public release.

Important updates from previous version (v0.0.1):

* implements Mie scattering phase function;

* contains ``examples`` directory that provides various mini examples;

* adds command line tools of ``lss`` and ``lsa`` for effortlessly displaying dataset information of
  meteorological data file through a terminal - currently supports ``HDF4``, ``HDF5``, ``netCDF``,
  and ``IDL sav``;

* includes support for libRadtran (``er3t.rtm.lrt``) in addition to MCARaTS (``er3t.rtm.mca``).


Version 0.0.1
-------------
First public pre-release (DOI:10.5281/zenodo.4110565).
