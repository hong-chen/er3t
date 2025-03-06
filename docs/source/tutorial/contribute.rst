=======================
Contribution Guidelines
=======================

.. warning::

   Incomplete, under development ...

Documentation
-------------

The documentation is written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
format (also known as ``.rst`` files) and compiled using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ software.
The most up-to-date documentation source code is available at `er3t/gh-pages <https://github.com/hong-chen/er3t/tree/gh-pages>`_
branch and hosted by `Read the Docs <https://readthedocs.org>`_.


Once new changes are pushed to `er3t/gh-pages`_, `Read the Docs`_ will automatically compile the documentation on the
cloud server, where the changes can be reflected on this documentation site shortly if compilation process is successful.


Contributors and Contributions
------------------------------

Current and past contributors are:

* `Vikas Nataraja <Vikas.HanasogeNataraja@lasp.colorado.edu>`_ (Dec., 2022 - current)

   - improved the automated process of satellite data download (functions in ``er3t/util/util.py``)

   - added support for MODIS 35 product (functions in ``er3t/util/modis.py``)

   - implemented command line tool for satellite data download (``bin/sdown``)

* `Yu-Wen Chen <Yu-Wen.Chen@colorado.edu>`_ (Apr., 2023 - current)

   - added support for MODIS 04 product (functions in ``er3t/util/modis.py``)

   - implementing spectroscopy support for OCO-2 (work in progress, functions in ``er3t/pre/abs/abs_oco.py``)

* `Ken Hirata <Ken.Hirata@colorado.edu>`_ (Jan., 2023 - current)

   - contributed to the theoretical development of CPU multithreading optimization (functions in ``er3t/rtm/mca/mca_run.py``)

   - implementing the Mie scattering phase function support for aerosols (work in progress)

