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
  * Making the built-in solvers work to have an interface closer to like ``scipy`` solvers. 

---------------------
Snake Care 
---------------------

Previous version of pymbar had mixed cases in functions. We have standardized on snake case, and tried to make the operator names behave better.  Specific changes include:

  * ``getFreeEnergyDifferences`` is now ``get_free_energy_differences``
  * ``computeExpectations`` is now ``compute_expectations``
  * ``computeMultipleExpectations`` is now ``compute_multiple_expectations``
  * ``computePerturbedFreeEnergies`` is now ``compute_perturbed_free_energies``
  * ``computeEntropyAndEnthalpy`` is now ``compute_entropy_and_enthalpy``
  * ``timseries.StatisticalInefficiency`` is now ``timeseries.statistical_inefficiency``
  * ``timeseries.statistical_inefficiency_multiple`` is now ``timeseries.statistical_inefficiency_multiple``  

    Additionally, the 
    
---------------------
More consistent return information 

Previously, different pymbar functions returned different information
as tuples. This became problematic when different functions returned
different types of information or different numbers of results.  We
have thus consolidated on an API where all functions return a
dictionary.


As an example of both API changes of API, a short bit of code that would load in data and
calculate free energies, instead of beinh

::

   
Would now be:

::  

---------------------
Free enerfy surfaces
---------------------

Previously, ``pymbar`` had a method ``PMF`` that estimated a free
energy from a series of umbrella samples.  The term PMF (potential of
mean force) is somewhat of an ambiguous term, as the potential of mean
force has some dependence on the coordinate system in which the mean
force is calculated. Since ``pymbar`` does not calculate free energies
by integration of mean force, this caused some comfusion. To be more
clear, we now have renamed the class ``FES``, for "free energy
surface".


The inclusion of a PMF function also created some confusion where some
authors referred to MBAR as a method to calculate a free energy
surface.  MBAR can only be used to take biased samples an estimate the
unbiased weight of each sample. In order to calculate a free energy
surface, one must also find a way to take the set of discrete weighted
samples and calculate a continous potential of mean force.  Shirts and
Ferguson, J. Chem. Theory Comput. 2020 further discusses the
separation of these two distinct tasks in the construction of free
energy surfaces. The pymbar code more cleanly separates the
calculation of biasing weights associated with umbrella samples, and
the estimation of the free energy surface.

For more information on the options for computing free energy surfaces
with the code, please see: (calculating free energy surfaces with
pymbar).

--------------------
Acceleration
--------------------

Previous version of `pymbar` include acceleration using explict C++
inner loops.  The C++ interface has become out of date. ``pymbar``
optimization routines are now accelerated with `jax`. This provides
approximately a 2$\times$ speed up when performed on most CPUs, and
additional acceleration when a GPU can be detected (pymbar does not
install the appropriate GPU libraries). ``jax`` will be installed when
``pymbar`` in installed can be installed via conda, but ``pymbar``
will function with or without jax.


Running the tests
=================
Running the tests is a great way to verify that everything is working.
The test suite uses `nose <https://nose.readthedocs.org/en/latest/>`_, in addition to `statsmodels <http://statsmodels.sourceforge.net/>`_ and `pytables <http://www.pytables.org/>`_, which you can install via ``conda``:

::
   $ conda install nose statsmodels pytables

You can then run the tests with:

::
   $ pytest -v pymbar
