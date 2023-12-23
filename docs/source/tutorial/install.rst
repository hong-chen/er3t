============
Installation
============

Dependencies
------------

1. Install ``conda`` Python package manager (pick **one** from the following installers)

    * `Anaconda <https://www.anaconda.com/>`_ (comprehensive, more popular);

    * `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (compact, more system-friendly).

2. Set up `NASA EARTHDATA <https://urs.earthdata.nasa.gov>`_

    * Register an account at https://urs.earthdata.nasa.gov.

    * Create a ``.netrc`` file under your home directory, e.g., type in ``touch ~/.netrc``.

    * Assume you have registered with an username ``abc`` and a password ``123``, in the ``~/.netrc`` file, put in

      .. code-block:: bash

         machine urs.earthdata.nasa.gov
         login abc
         password 123

    * Request a token (instructions `here <https://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens/>`_)
      for your EARTHDATA account and assign it to environment variable ``EARTHDATA_TOKEN``.

      For example, if you are using ``bash`` shell, add the following line to the shell source file
      (e.g., ``~/.bashrc``):

      .. code-block:: bash

         export EARTHDATA_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


3. Install `MCARaTS <https://sites.google.com/site/mcarats>`_ through the `official installation guide <https://sites.google.com/site/mcarats/mcarats-users-guide-version-0-10/2-installation>`__ (or a step-by-step `informal installation guide <https://discord.com/channels/681619528945500252/1004090233412923544/1004093265986986104>`__)

    * After installation, please specify environment variable ``MCARATS_V010_EXE``.

      For example, if you are using ``bash`` shell, add the following line to the shell source file
      (e.g., ``~/.bashrc``):

      .. code-block:: bash

         export MCARATS_V010_EXE="/system/path/to/mcarats-0.10.4/src/mcarats"

    * When the installation processes are complete,
      ``er3t.rtm.mca`` can be used to perform IPA/3D radiance/irradiance simulation (details see ``examples/00_er3t_mca.py``).

    .. tip::

       If you encountered any error, please feel free to reach out at `Discord SUPPORT/mcarats <https://discord.com/channels/681619528945500252/1123343304126365837>`__
       for community support.


4. (optional) Install `libRadtran <http://www.libradtran.org/>`_ through the `official installation guide <http://www.libradtran.org/doku.php?id=download>`__ (or a step-by-step `informal installation guide <https://discord.com/channels/681619528945500252/1004090233412923544/1004479494343622789>`__)

    * After installation, please specify environment variable ``LIBRADTRAN_V2_DIR`` for the directory that contains compiled libRadtran (the directory should contain ``bin/``, ``lib/``, ``src/`` etc.).

      For example, if you are using ``bash`` shell, add the following line to the shell source file
      (e.g., ``~/.bashrc``):

      .. code-block:: bash

         export LIBRADTRAN_V2_DIR="/system/path/to/libradtran/v2.0.1"

    * When the installation processes are complete,
      ``er3t.rtm.lrt`` can be used to perform IPA radiance/irradiance simulation (details see ``examples/00_er3t_lrt.py``).

    .. tip::

       If you encountered any error, please feel free to reach out at `Discord SUPPORT/libradtran <https://discord.com/channels/681619528945500252/1123343342730760222>`__
       for community support.

5. (optional) Install `SHDOM <https://coloradolinux.com/shdom/>`_

    **Unavailable yet (under development)**


|

EaR³T Python Package - ``er3t``
-------------------------------

.. warning::

    You will need to have the dependency steps 1 and 2 completed for accessing/processing satellite data.

.. warning::

    You will need to have the dependency steps 1 and 3 completed for running radiative transfer simulations.

1. Open a terminal, type in the following

    .. code-block:: bash

       git clone https://github.com/hong-chen/er3t.git


2. Under newly cloned ``er3t/``, where it contains ``er3t-env.yml``, type in the following

    .. code-block:: bash

       conda env create -f er3t-env.yml
       conda activate er3t

    * A `Python package version reference list <https://discord.com/channels/681619528945500252/1004090233412923544/1014015720302059561>`_
      (available to Mac and Linux users) is provided for diagnosing dependency version conflicts.


3. Under newly cloned ``er3t/``, where it contains ``install.sh``, type in the following

    .. code-block:: bash

       bash install.sh

    * If ``install.sh`` fails to download the data from Google Drive for any reason, you can download the required data
      manually from `here <https://drive.google.com/file/d/1KKpLR7IyqJ4gS6xCxc7f1hwUfUMJksVL/view?usp=sharing>`_.

      After you download the file (``er3t-data.tar.gz``), put it under ``er3t/``, where it contains ``install.sh``,
      then run the command ``bash install.sh`` through a terminal again.

.. tip::

    If you encountered any error, please feel free to reach out at `Discord SUPPORT/installation <https://discord.com/channels/681619528945500252/1123343093417119754>`__
    for community support.
