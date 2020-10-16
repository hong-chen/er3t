ER3T (Education and Research 3D Radiative Transfer Toolbox)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ER3T provides high-level interfaces that can automate the process of running 3D
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
        conda install -c conda-forge cartopy
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

**2. From Public Release (will be available soon)**

a) Download the latest release from `here <https://github.com/hong-chen/er3t/releases/latest>`_;


b) Unzip or untar the file after download;


3) Under the unzipped directory ``er3t``, where it contains ``install.sh``, type in the following

::

    bash install.sh


|
|

==========
How to Use
==========

We provide various mini-examples under ``tests/04_pre_mca.py``.

Details can be found under ``tests/README.rst``


|
|


===========
How to Cite
===========

Please contact `Hong Chen <hong.chen.cu@gmail.com>`_ and/or `Sebastian Schmidt <sebastian.schmidt@lasp.colorado.edu>`_ for the most recent information.

So far, two publications have used ER3T.

#. Gristey et al., 2019

   Gristey, J. J. et al.: Surface solar irradiance in continental shallow cumulus cloud fields: observations and Large Eddy Simulation, J. Atmospheric Sci., 2019

#. Gristey et al., 2020

   Gristey, J. J. et al.: On the relationship between shallow cumulus cloud field properties and surface solar irradiance, J. Atmospheric Sci., in review, 2020






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
