.. _getting-started:

Getting started
###############

.. highlight: bash

Installing ``pymbar``
=====================

conda (recommended)
-------------------

This documentation covers ``pymbar`` 3.
This is a long term support branch of ``pymbar``.
The easiest way to install the LTS version of ``pymbar`` release is via `conda <http://conda.pydata.org>`_:

.. code-block:: console

   $ conda install -c conda-forge "pymbar<4"

pip (pypi)
----------

Note: We are currently not releasing new versions of the LTS branch on ``PyPI``.

Development version
-------------------

The development version can be installed directly from github via ``pip``:

.. code-block:: console

   $ pip install git+https://github.com/choderalab/pymbar.git@pymbar-3-lts

Running the tests
=================
Running the tests is a great way to verify that everything is working.
The test suite uses `pytest <https://pytest.readthedocs.org/en/latest/>`_, in addition to `statsmodels <http://statsmodels.sourceforge.net/>`_ and `pytables <http://www.pytables.org/>`_, which you can install via ``conda``:

::
   $ conda install pytest "statsmodels<0.13" pytables

You can then run the tests with:

.. code-block:: console

   $ pytest -vv pymbar
