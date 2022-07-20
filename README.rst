EaR3T (Education and Research 3D Radiative Transfer Toolbox)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EaR3T provides high-level interfaces that can automate the process of running 3D
radiative transfer calculations for measured or modeled cloud/aerosol fields using
publicly available 3D radiative transfer models including MCARaTS (**implemented**),
SHDOM (under development), and MYSTIC (under development).


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

**2. Install** `MCARaTS <https://sites.google.com/site/mcarats/home>`_ **through the** `installation guide <https://sites.google.com/site/mcarats/mcarats-user-s-guide-version-0-10/2-installation>`_

|

**3. Specify path variable** ``MCARATS_V010_EXE``

    * If you are using ``bash`` shell, add the following line to the shell source file (e.g., ``~/.bashrc`` on Linux or ``~/.bash_profile`` on Mac):

    ::

        export MCARATS_V010_EXE="/path/to/the/compiled/MCARaTS/e.g./mcarats-0.10.4/src/mcarats"


    * If you are using ``csh`` shell, add the following line to the shell source file (e.g., ``~/.cshrc``):

    ::

        setenv MCARATS_V010_EXE "/path/to/the/compiled/MCARaTS/e.g./mcarats-0.10.4/src/mcarats"

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

We provide various mini-examples under ``tests/04_pre_mca.py``. After installation, you can use the provided
mini-examples to do test runs.

Details can be found under ``tests/README.rst``


|
|


================
Acknowledgements
================

* The absorption database ``er3t/data/abs/abs_16g.h5`` was created by `Coddington et al. (2008) <https://doi.org/10.1029/2008JD010089>`_ using correlated-k method.

    Coddington, O., Schmidt, K. S., Pilewskie, P., Gore, W. J., Bergstrom, R. W., Román, M., Redemann, J., Russell, P. B., Liu, J.,
    and Schaaf, C. C. (2008), Aircraft measurements of spectral surface albedo and its consistency with ground-based and
    space-borne observations, J. Geophys. Res., 113, D17209, doi:10.1029/2008JD010089.


|

* MCARaTS is a 3D radiative transfer model developed by `Iwabuchi (2006) <https://doi.org/10.1175/JAS3755.1>`_.

    Iwabuchi, H., 2006: Efficient Monte Carlo Methods for Radiative Transfer Modeling. J. Atmos. Sci., 63, 2324-2339, doi:10.1175/JAS3755.1.





|
|

===========
How to Cite
===========

#. `Chen et al., 2022 [in review] <https://doi.org/10.5194/amt-2022-143>`_

   Chen, H., Schmidt, S., Massie, S. T., Nataraja, V., Norgren, M. S., Gristey, J. J., Feingold,G.,
   Holz, R. E., and Iwabuchi, H.: The Education and Research 3D Radiative Transfer Toolbox (EaR3T) -
   Towards the Mitigation of 3D Bias in Airborne and Spaceborne Passive Imagery Cloud Retrievals,
   Atmos. Meas. Tech. Discuss. [preprint], ht<span>tps:</span>//doi.org/10.5194/amt-2022-143, in review, 2022.


Please contact `Hong Chen <hong.chen.cu@gmail.com>`_ and/or `Sebastian Schmidt <sebastian.schmidt@lasp.colorado.edu>`_ for the most recent information.

So far, the following publications have used EaR3T

#. `Nataraja et al., 2022 [in review] <https://doi.org/10.5194/amt-2022-45>`_

   Nataraja, V., Schmidt, S., Chen, H., Yamaguchi, T., Kazil, J., Feingold, G., Wolf, K., and
   Iwabuchi, H.: Segmentation-Based Multi-Pixel Cloud Optical Thickness Retrieval Using a Convolutional
   Neural Network, Atmos. Meas. Tech. Discuss. [preprint], https://doi.org/10.5194/amt-2022-45,
   in review, 2022.


#. `Gristey et al., 2022 <https://doi.org/10.1029/2022JD036822>`_

   Gristey, J. J., Feingold, G., Glenn, I. B., Schmidt, K. S., and Chen, H.: Influence of Aerosol Embedded
   in Shallow Cumulus Cloud Fields on the Surface Solar Irradiance, Journal of Geophysical Research: Atmospheres,
   127, e2022JD036822, https://doi.org/10.1029/2022JD036822, 2022.

#. `Gristey et al., 2020 <https://doi.org/10.1029/2020GL090152>`_

   Gristey, J. J., Feingold, G., Glenn, I. B., Schmidt, K. S., and Chen, H.: On the Relationship Between
   Shallow Cumulus Cloud Field Properties and Surface Solar Irradiance, Geophysical Research Letters, 47,
   e2020GL090152, https://doi.org/10.1029/2020GL090152, 2020.

#. `Gristey et al., 2020 <https://doi.org/10.1175/JAS-D-19-0261.1>`_

   Gristey, J. J., Feingold, G., Glenn, I. B., Schmidt, K. S., and Chen, H.: Surface Solar Irradiance in
   Continental Shallow Cumulus Fields: Observations and Large-Eddy Simulation, J. Atmos. Sci., 77, 1065-1080,
   https://doi.org/10.1175/JAS-D-19-0261.1, 2020.






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
