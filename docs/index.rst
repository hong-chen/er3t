EaR³T
=====

.. note::

    This documentation is under active development.

.. figure:: /_assets/er3t-logo.png
   :align: center
   :width: 400px

The Education and Research 3D Radiative Transfer Toolbox (EaR³T, /ɜːt/) is a Python
software designed for radiative transfer applications. It provides high-level interfaces
to automate the process of performing IPA/3D radiative transfer calculations for measured
or modeled cloud/aerosol fields using publicly available IPA/3D radiative transfer models
including MCARaTS, libRadtran (IPA only), and SHDOM (under development).

The code is publicly available at https://github.com/hong-chen/er3t.


**Applicable area:**

* Spaceborne and airborne remote sensing;

* 3D radiative effects (of clouds, aerosols, and trace gases etc.);

* Synthetic data generation (for CNN training);

* Novel retrieval algorithm development (e.g., CNN-based).


**Contributors:**

EaR³T was originally designed and developed by `Hong Chen <hong.chen@lasp.colorado.edu>`_ and
`Sebastian Schmidt <sebastian.schmidt@lasp.colorado.edu>`_ at the University of Colorado Boulder
and `Hong Chen <hong.chen@lasp.colorado.edu>`_ is the current maintainer.

Additionally, EaR³T has been actively contributed by the following developers:

* `Vikas Nataraja <Vikas.HanasogeNataraja@lasp.colorado.edu>`_ (Dec., 2022 - current)

   - improved the automated process of satellite data download (functions in ``er3t/util/util.py``)

   - added support for MODIS 35 product (functions in ``er3t/util/modis.py``)

   - implemented command line tool for satellite data download (``bin/sdown``)


* `Ken Hirata <Ken.Hirata@colorado.edu>`_ (Jan., 2023 - current)

   - contributed to the theoretical development of CPU multithreading optimization (functions in ``er3t/rtm/mca/mca_run.py``)

   - implementing the Mie scattering phase function support for aerosols (work in progress)

* `Yu-Wen Chen <Yu-Wen.Chen@colorado.edu>`_ (Apr., 2023 - current)

   - added support for MODIS 04 product (functions in ``er3t/util/modis.py``)

   - implementing spectroscopy support for OCO-2 (work in progress, functions in ``er3t/pre/abs/abs_oco.py``)


**How to cite:**

* `Chen et al., 2023 <https://doi.org/10.5194/amt-16-1971-2023>`_

   Chen, H., Schmidt, K. S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold, G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech., 16, 1971–2000, doi:10.5194/amt-16-1971-2023, 2023.


**Community Support:**

Please `join us on Discord <https://discord.gg/ntqsguwaWv>`_ for the latest information and community support.


|
|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   usage
   gallery
   publication
