EaR³T (Education and Research 3D Radiative Transfer Toolbox)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://discordapp.com/api/guilds/681619528945500252/widget.png?style=shield
   :target: https://discord.gg/ntqsguwaWv

EaR³T provides high-level interfaces that can automate the process of performing 1D/3D
radiative transfer calculations for measured or modeled cloud/aerosol fields using
publicly available 1D/3D radiative transfer models including MCARaTS (**implemented**),
libRadtran (**implemented**, 1D only), and SHDOM (future).

Applicable area:

* Spaceborne remote sensing;

* Airborne remote sensing;

* 3D cloud and aerosol radiative effects;

* Novel CNN-based cloud retrieval algorithms.


|

**How to cite:**

* `Chen et al., 2022 <https://doi.org/10.5194/amt-2022-143>`_

   Chen, H., Schmidt, S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold,G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech. Discuss. [preprint], doi:10.5194/amt-2022-143, in review, 2022.

|

Please contact `Hong Chen <hong.chen.cu@gmail.com>`_ and/or `Sebastian Schmidt <sebastian.schmidt@lasp.colorado.edu>`_ for the most recent information.

|
|


============
Dependencies
============

**1. Python packages (we recommend using** `Anaconda Python <https://www.anaconda.com/>`_ **)**

    ::

        conda install -c conda-forge gdown
        conda install -c conda-forge cartopy=0.17
        conda install -c conda-forge pyhdf
        conda install -c anaconda netcdf4
        conda install -c anaconda beautifulsoup4

|

**2. Install** `MCARaTS <https://sites.google.com/site/mcarats>`_ **through the** `official installation guide <https://sites.google.com/site/mcarats/mcarats-users-guide-version-0-10/2-installation>`_ (or the `unofficial installation guide <https://discord.com/channels/681619528945500252/1004090233412923544/1004093265986986104>`_ by Hong Chen)

|

**3. Specify environment variable** ``MCARATS_V010_EXE``

    * If you are using ``bash`` shell, add the following line to the shell source file (e.g., ``~/.bashrc`` on Linux or ``~/.bash_profile`` on Mac):

    ::

        export MCARATS_V010_EXE="/path/to/the/compiled/MCARaTS/e.g./mcarats-0.10.4/src/mcarats"


    * If you are using ``csh`` shell, add the following line to the shell source file (e.g., ``~/.cshrc``):

    ::

        setenv MCARATS_V010_EXE "/path/to/the/compiled/MCARaTS/e.g./mcarats-0.10.4/src/mcarats"

|

**4*. Install** `libRadtran <http://www.libradtran.org/>`_ **through the** `official installation guide <http://www.libradtran.org/doku.php?id=download>`_ (or the `unofficial installation guide <https://discord.com/channels/681619528945500252/1004090233412923544/1004479494343622789>`_ by Hong Chen)

    * This installation is optional.

    * If installed, please specify environment variable ``LIBRADTRAN_V2_DIR`` for the directory that contains compiled libRadtran (the directory should contain ``bin``, ``lib``, ``src`` etc.).

    * After installation,

      #. Mie scattering phase function will be supported;

      #. ``er3t.rtm.lrt`` can be used to perform 1D radiance/irradiance simulation (details see ``examples/00_er3t_lrt.py``).

|
|

==============
How to Install
==============

**You will need to have the dependencies installed first.**

**1. From Github**


a) Open a terminal, type in the following

::

    git clone https://github.com/hong-chen/er3t.git


b) Under newly cloned ``er3t``, where it contains ``install.sh``, type in the following

::

    bash install.sh


|

**2. From Public Release**

a) Download the latest release from `here <https://github.com/hong-chen/er3t/releases/latest>`_;


b) Unzip or untar the file after download;


3) Under the unzipped directory ``er3t``, where it contains ``install.sh``, type in the following

::

    bash install.sh

|


**Error Solutions**

If ``install.sh`` failed to download the data from Google Drive due to the following error

::

    Too many users have viewed or downloaded this file recently. Please
    try accessing the file again later. If the file you are trying to
    access is particularly large or is shared with many people, it may
    take up to 24 hours to be able to view or download the file. If you
    still can't access a file after 24 hours, contact your domain
    administrator.

You can download the required data manually from `here <https://drive.google.com/uc?id=1GSN7B3rPX8B9C59IVdYqswFiGas--lJo>`_.

After you download the file (``er3t-data.tar.gz``), put it under ``er3t`` directory where it contains ``install.sh``,
then run the command ``bash install.sh`` through a terminal again.


|
|

==========
How to Use
==========

We provide various examples extend from simple demo to complicate research project under ``examples``.
``examples/00_er3t_mca.py`` and ``examples/00_er3t_lrt.py`` can be used to perform test runs.

Details can be found in ``examples/README.rst``.


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

* MCARaTS is a 3D radiative transfer model developed by `Iwabuchi (2006) <https://doi.org/10.1175/JAS3755.1>`_.

    Iwabuchi, H.: Efficient Monte Carlo methods for radiative transfer modeling, J. Atmos. Sci., 63, 2324-2339,
    doi:10.1175/JAS3755.1, 2006.

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
|

===========
Publications
===========


So far, the following publications have used EaR³T

#. `Chen et al., 2022 <https://doi.org/10.5194/amt-2022-143>`_

   Chen, H., Schmidt, S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold,G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR³T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech. Discuss. [preprint], doi:10.5194/amt-2022-143, in review, 2022.

#. `Nataraja et al., 2022 <https://doi.org/10.5194/amt-2022-45>`_

   Nataraja, V., Schmidt, S., Chen, H., Yamaguchi, T., Kazil, J., Feingold, G., Wolf, K., and
   Iwabuchi, H.: Segmentation-Based Multi-Pixel Cloud Optical Thickness Retrieval Using a Convolutional
   Neural Network, Atmos. Meas. Tech. Discuss. [preprint], doi:10.5194/amt-2022-45,
   in review, 2022.


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
