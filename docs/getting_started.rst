.. _getting-started:

Getting started
###############

.. highlight: bash

Installing ``pymbar``
=====================

conda (recommended)
-------------------

The easiest way to install the ``pymbar`` release is via `conda <http://conda.pydata.org>`_:

::
   $ conda install -c omnia pymbar

pip
---

You can also install ``pymbar`` from the `Python package index <https://pypi.python.org/pypi/pymbar>`_ using ``pip``:

::
   $ pip install pymbar

Development version
-------------------

The development version can be installed directly from github via ``pip``:

::
   $ pip install git+https://github.com/choderalab/pymbar.git

Running the tests
=================
Running the tests is a great way to verify that everything is working.
The test suite uses `nose <https://nose.readthedocs.org/en/latest/>`_, in addition to `statsmodels <http://statsmodels.sourceforge.net/>`_ and `pytables <http://www.pytables.org/>`_, which you can install via ``conda``:

::
   $ conda install nose statsmodels pytables

You can then run the tests with:

::
   $ nosetests -vv pymbar
