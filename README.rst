EaR³T (Education and Research 3D Radiative Transfer Toolbox)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://discordapp.com/api/guilds/681619528945500252/widget.png?style=shield
   :target: https://discord.gg/ntqsguwaWv

|

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

    * - Multi-Angle (downward-looking from space)

      - Sunrise to Sunset (downward-looking from space)

    * - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/multi-angle_space.gif

      - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/sunrise-sunset_space.gif

    * - Multi-Angle (upward-looking from ground)

      - Sunrise to Sunset (upward-looking from ground)

    * - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/multi-angle_ground.gif

      - .. image:: https://github.com/hong-chen/er3t/blob/master/docs/assets/sunrise-sunset_ground.gif

|

**How to cite:**

* `Chen et al., 2023 <https://doi.org/10.5194/amt-16-1971-2023>`_

   Chen, H., Schmidt, S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold, G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech., 16, 1971–2000, doi:10.5194/amt-16-1971-2023, 2023.

|

Please `join us on Discord <https://discord.gg/ntqsguwaWv>`_ for the latest information and community support.

|
|


============
Dependencies
============

**1. Python packages** (we recommend using `Anaconda <https://www.anaconda.com/>`_ Python)


    ::

        conda env create -f er3t-env.yml
        conda activate er3t

    * A `Python package version reference list <https://discord.com/channels/681619528945500252/1004090233412923544/1014015720302059561>`_
      (available to Mac and Linux users) is provided for diagnosing dependency version conflicts.

|

**2. Install** `MCARaTS <https://sites.google.com/site/mcarats>`_ **through the** `official installation guide <https://sites.google.com/site/mcarats/mcarats-users-guide-version-0-10/2-installation>`_ (or a step-by-step `installation guide (unofficial) <https://discord.com/channels/681619528945500252/1004090233412923544/1004093265986986104>`_ by Hong Chen)

    * After installation, please specify environment variable ``MCARATS_V010_EXE``.

      For example, if you are using ``bash`` or ``zsh`` shell, add the following line to the shell source file
      (e.g., ``~/.bashrc`` for ``bash`` or ``~/.zshrc`` for ``zsh``):

      ::

        export MCARATS_V010_EXE="/system/path/to/mcarats-0.10.4/src/mcarats"

    * When the installation processes are complete,
      ``er3t.rtm.mca`` can be used to perform IPA/3D radiance/irradiance simulation (details see ``examples/00_er3t_mca.py``).

|

**3. (optional) Install** `libRadtran <http://www.libradtran.org/>`_ **through the** `official installation guide <http://www.libradtran.org/doku.php?id=download>`_ (or a step-by-step `installation guide (unofficial) <https://discord.com/channels/681619528945500252/1004090233412923544/1004479494343622789>`_ by Hong Chen)

    * After installation, please specify environment variable ``LIBRADTRAN_V2_DIR`` for the directory that contains compiled libRadtran (the directory should contain ``bin``, ``lib``, ``src`` etc.).

      For example, if you are using ``bash`` or ``zsh`` shell, add the following line to the shell source file
      (e.g., ``~/.bashrc`` for ``bash`` or ``~/.zshrc`` for ``zsh``):

      ::

        export LIBRADTRAN_V2_DIR="/system/path/to/libradtran/v2.0.1"

    * When the installation processes are complete,
      ``er3t.rtm.lrt`` can be used to perform IPA radiance/irradiance simulation (details see ``examples/00_er3t_lrt.py``).

|

**4. (optional) Install** `SHDOM <https://coloradolinux.com/shdom/>`_

    **Unavailable yet (under development)**


|
|

==============
How to Install
==============

**You will need to have the dependencies installed first.**

**1. From Github (most up-to-date)**


a) Open a terminal, type in the following

::

    git clone https://github.com/hong-chen/er3t.git


b) Under newly cloned ``er3t``, where it contains ``install.sh``, type in the following

::

    bash install.sh


|

**2. From Public Release (fully tested)**

a) Download the latest release from `here <https://github.com/hong-chen/er3t/releases/latest>`_;


b) Unzip or untar the file after download;


3) Under the unzipped directory ``er3t``, where it contains ``install.sh``, type in the following

::

    bash install.sh

|

    If ``install.sh`` fails to download the data from Google Drive for any reason, you can download the required data
    manually from `here <https://drive.google.com/file/d/1KKpLR7IyqJ4gS6xCxc7f1hwUfUMJksVL/view?usp=sharing>`_.

    After you download the file (``er3t-data.tar.gz``), put it under ``er3t`` directory where it contains ``install.sh``,
    then run the command ``bash install.sh`` through a terminal again.


|
|

==========
How to Use
==========

We provide various examples extend from simple demo to complex research project under `examples <https://github.com/hong-chen/er3t/tree/dev/examples>`_.
``examples/00_er3t_mca.py`` and ``examples/00_er3t_lrt.py`` can be used to perform test runs.

A detailed instruction can be found `here <https://github.com/hong-chen/er3t/tree/dev/examples#readme>`_.


|
|


================
Acknowledgements
================

* The absorption database ``er3t/data/abs/abs_16g.h5`` was created by `Coddington et al. (2008) <https://doi.org/10.1029/2008JD010089>`_ using correlated-k method.

    Coddington, O., Schmidt, K. S., Pilewskie, P., Gore, W. J., Bergstrom, R., Roman, M., Redemann, J.,
    Russell, P. B., Liu, J., and Schaaf, C. C.: Aircraft measurements of spectral surface albedo and its
    consistency with ground based and space-borne observations, J. Geophys. Res., 113, D17209,
    doi:10.1029/2008JD010089, 2008.


|

* MCARaTS is a 3D radiative transfer solver developed by `Iwabuchi (2006) <https://doi.org/10.1175/JAS3755.1>`_
  and `Iwabuchi and Okamura (2017) <https://doi.org/10.1016/j.jqsrt.2017.01.025>`_.

    Iwabuchi, H.: Efficient Monte Carlo methods for radiative transfer modeling, J. Atmos. Sci., 63, 2324-2339,
    doi:10.1175/JAS3755.1, 2006.

    |

    Iwabuchi, H., and Okamura, R.: Multispectral Monte Carlo radiative transfer simulation by using the maximum
    cross-section method, Journal of Quantitative Spectroscopy and Radiative Transfer, 193, 40-46,
    doi:10.1016/j.jqsrt.2017.01.025, 2017.


|

*  libRadtran is a library for radiative transfer developed by `Emde et al. (2016) <https://doi.org/10.5194/gmd-9-1647-2016>`_
   and `Mayer and Kylling (2005) <https://doi.org/10.5194/acp-5-1855-2005>`_.

    Emde, C., Buras-Schnell, R., Kylling, A., Mayer, B., Gasteiger, J., Hamann, U., Kylling, J., Richter, B.,
    Pause, C., Dowling, T., and Bugliaro, L.: The libRadtran software package for radiative transfer
    calculations (version 2.0.1), Geosci. Model Dev., 9, 1647–1672, doi:10.5194/gmd-9-1647-2016, 2016.

    |

    Mayer, B. and Kylling, A.: Technical note: The libRadtran software package for radiative transfer
    calculations - description and examples of use, Atmos. Chem. Phys., 5, 1855–1877,
    doi:10.5194/acp-5-1855-2005, 2005.

|

*  SHDOM is a 3D radiative transfer solver developed by `Evans (1998) <https://doi.org/10.1175/1520-0469(1998)055%3C0429:TSHDOM%3E2.0.CO;2>`_.
   The development of SHDOM by Evans has been discontinued since 2016.

    Evans, K. F.: The spherical harmonics discrete ordinate method for three-dimensional atmospheric
    radiative transfer, J. Atmos. Sci., 55, 429–446, 1998.


|
|


============
Publications
============


So far, the following publications have used EaR³T

#. `Gristey et al., 2023 <https://doi.org/10.5194/amt-2023-7>`_

   Gristey, J. J., Schmidt, K. S., Chen, H., Feldman, D. R., Kindel, B. C., Mauss, J., van den Heever, M.,
   Hakuba, M. Z., and Pilewskie, P.: Angular Sampling of a Monochromatic, Wide-Field-of-View Camera to Augment
   Next-Generation Earth Radiation Budget Satellite Observations, Atmos. Meas. Tech. Discuss. [preprint],
   doi:10.5194/amt-2023-7, in review, 2023.

#. `Chen et al., 2023 <https://doi.org/10.5194/amt-16-1971-2023>`_

   Chen, H., Schmidt, S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold, G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech., 16, 1971–2000, doi:10.5194/amt-16-1971-2023, 2023.

#. `Nataraja et al., 2022 <https://doi.org/10.5194/amt-15-5181-2022>`_

   Nataraja, V., Schmidt, S., Chen, H., Yamaguchi, T., Kazil, J., Feingold, G., Wolf, K., and
   Iwabuchi, H.: Segmentation-Based Multi-Pixel Cloud Optical Thickness Retrieval Using a Convolutional
   Neural Network, Atmos. Meas. Tech., 15, 5181–5205, doi:10.5194/amt-15-5181-2022, 2022.


#. `Gristey et al., 2022 <https://doi.org/10.1029/2022JD036822>`_

   Gristey, J. J., Feingold, G., Glenn, I. B., Schmidt, K. S., and Chen, H.: Influence of Aerosol Embedded
   in Shallow Cumulus Cloud Fields on the Surface Solar Irradiance, Journal of Geophysical Research: Atmospheres,
   127, e2022JD036822, doi:10.1029/2022JD036822, 2022.

#. `Gristey et al., 2020 <https://doi.org/10.1029/2020GL090152>`_

   Gristey, J. J., Feingold, G., Glenn, I. B., Schmidt, K. S., and Chen, H.: On the Relationship Between
   Shallow Cumulus Cloud Field Properties and Surface Solar Irradiance, Geophysical Research Letters, 47,
   e2020GL090152, doi:10.1029/2020GL090152, 2020.

#. `Gristey et al., 2020 <https://doi.org/10.1175/JAS-D-19-0261.1>`_

   Gristey, J. J., Feingold, G., Glenn, I. B., Schmidt, K. S., and Chen, H.: Surface Solar Irradiance in
   Continental Shallow Cumulus Fields: Observations and Large-Eddy Simulation, J. Atmos. Sci., 77, 1065-1080,
   doi:10.1175/JAS-D-19-0261.1, 2020.






|
|



============
Contributors
============

Current and past contributors are:

* `Vikas Nataraja <Vikas.HanasogeNataraja@lasp.colorado.edu>`_ (Dec., 2022 - current)

   - improved the automated process of satellite data download (functions in ``er3t/util/util.py``)

* `Ken Hirata <Ken.Hirata@colorado.edu>`_ (Jan., 2023 - current)

   - implementing the Mie scattering phase function support for aerosols (work in progress)

* `Yu-Wen Chen <Yu-Wen.Chen@colorado.edu>`_ (Apr., 2023 - current)

   - implementing spectroscopy support for OCO-2 (work in progress)

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
