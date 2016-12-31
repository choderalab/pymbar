.. _getting-started:

Getting started
###############

Installing ``pymbar``
=====================

conda (recommended)
-------------------

The easiest way to install the ``pymbar`` release is via `conda <http://conda.pydata.org>`_:

  $ conda install -c omnia pymbar

pip
---

You can also install ``pymbar`` from the `Python package index <https://pypi.python.org/pypi/pymbar>`_ using ``pip``:

  $ pip install pymbar

Development version
-------------------

The development version can be installed directly from github via ``pip``:

  $ pip install git+https://github.com/choderalab/pymbar.git

Running the tests
=================
Running the tests is a great way to verify that everything is working. The test
suite uses `nose <https://nose.readthedocs.org/en/latest/>`_, which you can pick
up via ``conda`` or ``pip`` if you don't already have it. ::

  $ conda install --yes nose

or

  $ pip install nose

You can then run the quick tests with:

  $ nosetests -vv pymbar

If you want to run the full suite of tests with large test datasets, you can grab the latest version with:

  $ git clone git clone https://github.com/choderalab/pymbar-datasets.git $HOME/pymbar_data

and then run the full test suite with:

$ nosetests -vv pymbar
