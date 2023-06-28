Usage
=====

Due to the lack of funding support, only limited documentation has been developed.
Despite the funding difficulties, we provide various examples extend from simple demo to complex research
project under `<root>/examples <https://github.com/hong-chen/er3t/tree/dev/examples>`_ and hope users can learn
the usage of EaR³T from the provided examples.


Examples (`code status <https://discord.com/channels/681619528945500252/1004090233412923544/1017575066139103293>`_)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    The example code is under active development. Please check for the latest code status before use.

To run the example codes, auxiliary data is required.
To download the data, under the example directory (``<root>/examples``) where it contains ``install-examples.sh``,
type in the following

    ::

      bash install-examples.sh

    * If ``install-examples.sh`` fails to download the data from Google Drive for any reason, you can download the required data manually
      from `here <https://drive.google.com/file/d/1Oov75VffmuQSljxjoOS6q6egmfT6CmkI/view?usp=share_link>`_.

      After you download the file (``er3t-data-examples.tar.gz``), put it under ``<root>/examples`` directory where
      it contains ``install-examples.sh``, then run the command ``bash install-examples.sh`` through a terminal again.

Codes ``00`` (e.g., ``00_er3t_mca.py``) can be used to perform test runs.

Codes ``01`` to ``04`` can be used to reproduce scientific results discussed in
`Chen et al. (2024) <https://doi.org/10.5194/amt-16-1971-2023>`_.

The figure results for each example are provided in ``<root>/examples/check`` directory (available after installing auxiliary data)
for validation.
