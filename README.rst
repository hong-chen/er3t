EaR³T (Education and Research 3D Radiative Transfer Toolbox)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://dcbadge.vercel.app/api/server/ntqsguwaWv?style=flat&theme=discord-inverted
    :target: https://discord.gg/ntqsguwaWv
    :alt: Discord Server

.. image:: https://readthedocs.org/projects/er3t/badge/?version=latest
    :target: https://er3t.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/dynamic/json?color=blue&label=unique%20clone&query=uniques&url=https://gist.githubusercontent.com/hong-chen/54187b01bd3c5eac3b7645ad332f9ad3/raw/clone.json&logo=github
    :target: https://github.com/hong-chen/er3t
    :alt: Git Clone Counts

.. image:: https://img.shields.io/github/stars/hong-chen/er3t?color=blue&label=star&logo=github
    :target: https://github.com/hong-chen/er3t/stargazers
    :alt: GitHub Repo Stars

.. image:: https://img.shields.io/badge/cited_by-8-blue
    :target: https://er3t.readthedocs.io/en/latest/source/other/highlight.html#publications
    :alt: Citation Counts

.. image:: https://img.shields.io/badge/doi-10.5194%2Famt--16--1971--2023-blue
    :target: https://doi.org/10.5194/amt-16-1971-2023
    :alt: Publication DOI


.. image:: https://github.com/hong-chen/er3t/blob/gh-pages/docs/assets/er3t-logo.png
    :target: https://github.com/hong-chen/er3t
    :width: 400
    :align: center

|

We are preparing for an upcoming release, so please anticipate frequent changes to the main branch.
------------

|

`EaR³T <https://er3t.readthedocs.io/en/latest/>`_ (pronounced /ɜːt/) is a Python software package
developed for cutting-edge radiative transfer and remote sensing applications. It provides high-level
interfaces to automate the process of performing IPA/3D radiative transfer calculations for measured
or modeled cloud/aerosol fields using publicly available radiative transfer solvers. The automation
capability covers not only the simulation of radiometric quantities such as radiance and irradiance,
but also the downloading/processing of satellite data and the setup of atmospheric radiative properties,
which are essential for an end-to-end simulation pipeline.


Applications
------------

* Spaceborne and airborne remote sensing;

* 3D radiative effects (of clouds, aerosols, and trace gases etc.);

* Synthetic data generation (for CNN training);

* Novel retrieval algorithm development (e.g., CNN-based).


Features
--------
:Radiative Transfer:

  * **solver**: `MCARaTS <https://sites.google.com/site/mcarats/>`_ | `libRadtran <http://www.libradtran.org/>`_

  * **absorption**: `Correlated-k <https://doi.org/10.1029/90JD01945>`_ | `REPTRAN <https://doi.org/10.1016/j.jqsrt.2014.06.024>`_

  * **clouds**: `Mie (water) <https://doi.org/10.1364/AO.19.001505>`_

  * **surface**: `Cox-Munk (ocean) <https://doi.org/10.1364/JOSA.44.000838>`_ | `LSRT Model (land) <https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD43A1>`_

:Remote Sensing:

  * **downloader**: `Worldview <https://worldview.earthdata.nasa.gov>`_ | `LAADS DAAC <https://ladsweb.modaps.eosdis.nasa.gov/archive/>`_ | `GES DISC <https://oco2.gesdisc.eosdis.nasa.gov/data/>`_ | `LANCE <https://nrt3.modaps.eosdis.nasa.gov/archive>`_

  * **reader**: `MODIS <https://modis.gsfc.nasa.gov>`_ | `VIIRS <https://ncc.nesdis.noaa.gov/VIIRS/>`_ | `OCO-2 <https://ocov2.jpl.nasa.gov>`_ | `AHI <https://www.data.jma.go.jp/mscweb/en/index.html>`_

  * **simulator**: `ARISE <https://zenodo.org/record/4029241>`_ | `CAMP²Ex <https://zenodo.org/record/7358509>`_


Resources
---------

:Source Code: https://github.com/hong-chen/er3t/

  * `er3t/master (most stable) <https://github.com/hong-chen/er3t/tree/master>`_

  * `er3t/dev (most up-to-date) <https://github.com/hong-chen/er3t/tree/dev>`_

  * `er3t/gh-pages (latest docs) <https://github.com/hong-chen/er3t/tree/gh-pages>`_

  * `Releases <https://github.com/hong-chen/er3t/releases>`_


:Documentation: https://er3t.readthedocs.io/

  * `How to install <https://er3t.readthedocs.io/en/latest/source/tutorial/install.html>`_

  * `How to use <https://er3t.readthedocs.io/en/latest/source/tutorial/usage.html>`_

  * `How to cite <https://er3t.readthedocs.io/en/latest/#how-to-cite>`_

  * `How to contribute <https://er3t.readthedocs.io/en/latest/source/tutorial/contribute.html>`_

  * `FAQ <https://er3t.readthedocs.io/en/latest/source/other/faq.html>`_

:Community Support: `Discord <https://discord.gg/ntqsguwaWv>`_

  * `Installation-related  <https://discord.com/channels/681619528945500252/1123343093417119754>`_

  * `Examples-related <https://discord.com/channels/681619528945500252/1123343152477110453>`_

  * `Satellite-related <https://discord.com/channels/681619528945500252/1123343438121799690>`_

  * `MCARaTS-related <https://discord.com/channels/681619528945500252/1123343304126365837>`_

  * `libRadtran-related <https://discord.com/channels/681619528945500252/1123343342730760222>`_

  * `Discussion <https://discord.com/channels/681619528945500252/1001181810782388414>`_


Gallery
-------

.. list-table::

    * - Multi-Angle (space view [land BRDF])

      - Sunrise to Sunset (space view [ocean BRDF])

    * - .. image:: https://github.com/hong-chen/er3t/blob/gh-pages/docs/assets/multi-angle_space.gif
            :width: 400

      - .. image:: https://github.com/hong-chen/er3t/blob/gh-pages/docs/assets/sunrise-sunset_space.gif
            :width: 400

    * - Multi-Angle (ground view)

      - Sunrise to Sunset (ground view)

    * - .. image:: https://github.com/hong-chen/er3t/blob/gh-pages/docs/assets/multi-angle_ground.gif
            :width: 400

      - .. image:: https://github.com/hong-chen/er3t/blob/gh-pages/docs/assets/sunrise-sunset_ground.gif
            :width: 400
