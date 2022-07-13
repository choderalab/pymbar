.. _strategies_for_solution:

Getting started
###############

.. highlight: bash

Approaching the MBAR euqations
=====================

``scipy`` solutions
-------------------
The MBAR equations can be formulated in a number of ways.  They are a
set of coupled, implicit equations for the free energy values
satisfying the equations, up to an overall constant, which ``pymbar``
removes by setting the first free energy to zero to 0.

By rearrangement, this set of self-consistent equations can be written
as simultaneous roots to $K$ equations.  This set of roots also turns
out to be the Jacobian of single maximum likelihood function of all
the free energies. We then can find the MBAR solutions by maximizing
this likelihood.

Given this formulation, we can simply write out the MBAR equations as


Built-in solutions
------------------

``pymbar`` also includes an adaptive solver designed directly for
MBAR.  It calculates every step both the self-consistent formula
presented in Shirts et al., and takes a Newton-Raphson increment.  In
both cases, it calculates the gradient of each step, and selects the
move that makes the magnitude of the gradient (i.e. the dot product of
the gradient with itself) smallest. Far from the solution, the
self-consistent iteration tends have the smaller gradient, while
closer to the solution, the Newton-Raphson step tends to have the
smallest gradient.  T

Constructing solver protocols
-------------------


The solutions for the bootstrapped data should be relatively close to 

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
