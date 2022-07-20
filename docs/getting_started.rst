.. _getting-started:

Getting started
###############

.. highlight: bash

Installing ``pymbar``
=====================

This documentation covers ``pymbar`` 4.  For the previous versions of pymbar, see: `pymbar 3.0.7 <https://pymbar.readthedocs.io/en/3.0./>`_.

conda (recommended)
-------------------

The easiest way to install the ``pymbar`` release is via `conda <http://conda.pydata.org>`_:

.. code-block:: console

   $ conda install -c conda-forge pymbar

You can also install ``pymbar`` from the `Python package index <https://pypi.python.org/pypi/pymbar>`_ using ``pip``:

.. code-block:: console

   $ pip install pymbar

Development version
-------------------

The development version can be installed directly from GitHub via ``pip``: 

.. code-block:: console

   $ pip install git+https://github.com/choderalab/pymbar.git

In beta testing, this is way to download pymbar 4.

Running the tests
=================

Running the tests is a great way to verify that everything is working.

The test suite uses `pytest <https://docs.pytest.org/>`_, in addition to `statsmodels <http://statsmodels.sourceforge.net/>`_ and `pytables <http://www.pytables.org/>`_, which you can install via ``conda``:

.. code-block:: console

   $ conda install pytest statsmodels


You can then run the tests from within the `pymbar` directory with:

.. code-block:: console
		
   $ pytest -v pymbar
