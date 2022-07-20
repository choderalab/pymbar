.. _moving_from_pymbar3:

Moving from ``pymbar`` version 3
################################

Pymbar v4.0 contains several changes to improve the API longer
term. This, however, breaks the API used in 3.x and previous versions.

The main changes include:

    * Making various estimators return dictionaries, not tuples, making it easier to return optional information requested at call time. 
    * Standardizing on snake_case for function names. 
    * Making the built-in solvers work to have an interface closer to like ``scipy`` solvers. 

----------
Snake case
----------

Previous version of pymbar had mixed cases in functions. We have
standardized on snake case, and tried to make the method names that do
similar things more consistent.  Specific changes include:

    * ``getFreeEnergyDifferences`` is now ``compute_free_energy_differences``
    * ``computeExpectations`` is now ``compute_expectations``
    * ``computeMultipleExpectations`` is now ``compute_multiple_expectations``
    * ``computePerturbedFreeEnergies`` is now ``compute_perturbed_free_energies``
    * ``computeEntropyAndEnthalpy`` is now ``compute_entropy_and_enthalpy``

In the submodule `timeseries`:

    * ``statisticalInefficiency`` is now ``statistical_inefficiency``
    * ``statisticalInefficiencyMltiple`` is now ``statistical_inefficiency_multiple``  
    * ``integratedAutocorrelationTime`` is now ``integrated_autocorrelation_time``
    * ``normalizedFluctuationCorrelationFunction`` is now ``normalized_fluctuation_correlation_function``
    * ``normalizedFluctuationCorrelationFunctionMultiple`` is now ``normalized_fluctuation_correlation_function_multiple``
    * ``subsampleCorrelatedData`` is now ``subsample_correlated_data``
    * ``detectEquilibration`` is now ``detect_equilibration``
    * ``statisticalInefficiency_fft`` is now ``statistical_inefficiency_fft``
    * ``detectEquilibration_binary_search`` is now ``detect_equilibration_binary_search``

Additionally, the other estimators such as the Bennett Acceptance
Ratio and exponential averaging/Zwanzig equation have different, more
consistent, call signatures.  All other estimators are now in the
``other_estimators`` module.

    * ``BAR`` is now ``bar``
    * ``EXP`` is now ``exp``
    * ``EXPGauss``  is now ``exp_gauss``
    * ``PMF`` is now ``FEP`` and is greatly expanded (see :ref:`fes_with_pymbar`).   

------------------------------------
More consistent return functionality
------------------------------------

Previously, different pymbar functions returned different information
as tuples. This became problematic when different functions returned
different types of information or different numbers of results. We
have thus consolidated on an API where all functions return a
dictionary.

As an example of both API changes of API, a short bit of code that
would load in data and calculate free energies, instead of being

.. code-block:: python
    :caption: Example of initializing ``MBAR`` in 3.0.5

    mbar = MBAR(u_kn, N_k)
    results, errors = mbar.getFreeEnergyDifferences()                                                                 
    print(results[0])
    print(errors[0]) 

    
Would now be written as:

.. code-block:: python
    :caption: Example of initializing ``MBAR`` in 4.0

    mbar = MBAR(u_kn, N_k)
    results = mbar.compute_free_energy_differences()
    print(results['Delta_f'])
    print(results['dDelta_f'])


Other estimators including ``bar`` and ``exp`` also use a dictionary for return data.

The ``pymbar.timeseries`` submodule return patterns have *not* changed
in 4.0, however, and one should refer to the individual function
documentations for these return patterns.

.. code-block:: python
		
   results = bar(w_F, w_R)        
   print(f'Free energy difference is {results['Delta_f']:.3f} +- {results['Delta_f']:.3f} kT')


   and:

.. code-block:: python
		
   results = exp(w_F)
   print(f"Forward free energy difference is {results['Delta_f']:.3f} +- {results['dDelta_f']:.3f} kT)
   results = exp(w_R)
   print(f"Reverse free energy difference is {results['Delta_f']:.3f} +- {results['dDelta_f']:.3f} kT)

-----------------
Simulation output
-----------------

Previously, ``pymbar`` send all messages to standard out when verbose
was set to ``True``.  ``pymbar`` now uses the logging module to output
this information.  If you wish to set messages, even if the verbose is
set to ``True``, you will need to turn on logging for your script by
importing the logging module, and adding the lines:

.. code-block:: python
  :caption: Enabling logging in ``pybmar``
	    
  import logging
  import sys
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)


``pymbar`` generally uses the logging levels ``info`` for information
that previously was set to standard out.  Note that for a given method
to produce extensive information, even with logging, the verbose flag
still needs to be set to true.

--------------------
Free energy surfaces
--------------------

Previously, ``pymbar`` had a method ``PMF`` that estimated a free
energy from a series of umbrella samples using a histogram
approach. This was sematically problematin in two ways. First, the
term PMF (potential of mean force) is somewhat of an ambiguous term,
as the potential of mean force has some dependence on the coordinate
system in which the mean force is calculated. Since ``pymbar`` does
not calculate free energies by integration of mean force, this caused
some comfusion. To be clearer, we now have renamed the class
``FES``, for "free energy surface".

The inclusion of a PMF function also created some confusion where some
authors referred to MBAR as a method to calculate a free energy
surface.  MBAR can only be used to take biased samples an estimate the
unbiased weight of each sample. In order to calculate a free energy
surface, one must also find a way to take the set of discrete weighted
samples and calculate a continous potential of mean force: see Shirts
and Ferguson :cite:`shirts_fes_2020` for a further discussion of the
separation of these two distinct tasks in the construction of free
energy surfaces. The pymbar code more cleanly separates the
calculation of biasing weights associated with umbrella samples, and
the estimation of the free energy surface.

For more information on the options for computing free energy surfaces
with the code, please see: :ref:`fes_with_pymbar`. 

------------
Acceleration
------------

Previous version of ``pymbar`` include acceleration using explict C++
inner loops.  The C++ interface has become out of date. ``pymbar``
optimization routines are now accelerated with ``jax``. This provides
approximately a 2x speed up when performed on most CPUs, and
additional acceleration when a GPU can be detected (pymbar does not
install the appropriate GPU libraries). ``jax`` will be installed when
``pymbar`` in installed via conda, but ``pymbar`` will function with
or without ``jax`` installed if there are issues with the JAX configuration.

-------------
Other changes
-------------

Additional changes not affecting the API:
  * Removed legacy `old_mbar.py` support.
  * Moved testing framework to pytest, added significant numbers of tests.
  * Improved code linting using `black`l
  * Bootstrapping for errors in free energies and expectations is now supported; see :ref:`strategies_for_solution` for more information.
  * Added a `bar_overlap` function to find overlap when using just `bar`
  * Fixed an error in computing expectations of small numbers.
  * Improved automated adaptive choice of samplers; see :ref:`strategies_for_solution` for more information.
  * Many instances of code cleanup.
  * Improved docstring documentation.

