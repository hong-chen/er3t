EaR³T (Education and Research 3D Radiative Transfer Toolbox)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://er3t.readthedocs.io/en/latest/_images/er3t-logo.png
    :target: https://github.com/hong-chen/er3t
    :wdith: 200
    :align: left

.. image:: https://readthedocs.org/projects/er3t/badge/?version=latest
    :target: https://er3t.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://discordapp.com/api/guilds/681619528945500252/widget.png?style=shield
   :target: https://discord.gg/ntqsguwaWv
   :alt: Discord Server


EaR³T (pronounced [ɜːt]) provides high-level interfaces that can automate the process of performing IPA/3D
radiative transfer calculations for measured or modeled cloud/aerosol fields using
publicly available IPA/3D radiative transfer models including MCARaTS (**implemented**),
libRadtran (**implemented**, IPA only), and SHDOM (under development).

Applicable area:

* Spaceborne and airborne remote sensing;

* 3D radiative effects (of clouds, aerosols, and trace gases etc.);

* Synthetic data generation (for CNN training);

* Novel retrieval algorithm development (e.g., CNN-based).

|

.. list-table:: **Demo**

    * - Multi-Angle (space view)

      - Sunrise to Sunset (space view)

    * - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/multi-angle_space.gif

      - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/sunrise-sunset_space.gif

    * - Multi-Angle (ground view)

      - Sunrise to Sunset (ground view)

    * - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/multi-angle_ground.gif

      - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/sunrise-sunset_ground.gif

|

**How to cite:**

* `Chen et al., 2023 <https://doi.org/10.5194/amt-16-1971-2023>`_

   Chen, H., Schmidt, K. S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold, G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech., 16, 1971–2000, doi:10.5194/amt-16-1971-2023, 2023.

|

Please `join us on Discord <https://discord.gg/ntqsguwaWv>`_ for the latest information and community support.

|
|

.. include:: docs/source/tutorial/install.rst

|
|

.. include:: docs/source/tutorial/usage.rst


|
|

.. include:: docs/source/other/acknowledge.rst

|
|

.. include:: docs/source/other/highlight.rst

|
|

============
Contributors
============

Current and past contributors are:

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

|

If you are interested in making contributions to the package,
please refer to `CONTRIBUTING <https://github.com/hong-chen/er3t/blob/dev/CONTRIBUTING.rst>`_
doc for further information.

|
|

=====
F.A.Q
=====

1. How to update the local ``er3t`` repository?

::

    git checkout master
    git pull origin master

    python setup.py develop


2. What to do if encounter conflicts in file change when ``git pull``?

::

    git checkout master
    git fetch --all
    git reset --hard origin/master
    git pull origin master

    python setup.py develop


3. How to clean up local branches?

::

    git branch -a
    git remote prune origin --dry-run

    git remote prune origin
    git branch -a
