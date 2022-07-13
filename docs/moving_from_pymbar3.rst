.. _moving_from_pymbar3:

Moving from ``pymbar3``
###############

.. highlight: bash

=====================

Pymbar v4.0 contains several changes to improve the API longer
term. This, however, breaks the API used in 3.x and previous versions.

The main changes include:

  * Making various estimators return dictionaries, not tuples, making it easier to return optional information requested at call time. 
  * Standardizing on snake_case for function names. 
  * Making the built-in solvers work more like ``scipy`` solvers.

---------------------
Snake Care 
---------------------

Previous version of pymbar had mixed cases in functions. We have standardized on snake case, and tried to make the operator names behave better.  Specific changes include:

  * `` 

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
   $ pytest -v pymbar
