.. currentmodule:: pymbar.timeseries

The :mod:`timeseries` module: :module:`pymbar.timeseries`:
=========================================================

The :mod:`pymbar.timeseries` module contains tools for dealing with timeseries data.
The `MBAR <http://www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio>`_ method is only applicable to uncorrelated samples from probability distributions, so we provide a number of tools that can be used to decorrelate simulation data.

Automatically identifying the equilibrated production region
------------------------------------------------------------

Most simulations start from initial conditions that are highly unrepresentative of equilibrated samples that occur late in the simulation.
We can improve our estimates by discarding these initial regions to "equilibration" (also known as "burn-in").
We recommend a simple scheme described in Ref. :cite:`chodera:jctc:2016:automatic-equilibration-detection`, which identifies the production region as the final contiguous region containing the *largest number of uncorrelated samples.
This scheme is implemented in the :func:`detectEquilibration` method:

.. python

   from pymbar import timeseries
   [t0, g, Neff_max] = timeseries.detectEquilibration(A_t) # compute indices of uncorrelated timeseries
   A_t_equil = A_t[t0:]
   indices = timeseries.subsampleCorrelatedData(A_t_equil, g=g)
   A_n = A_t_equil[indices]

In this example, the :func:`detectEquilibration` method is used on the correlated timeseries ``A_t`` to identify the sample index corresponding to the beginning of the production region, ``t_0``, the statistical inefficiency of the production region ``[t0:]``, ``g``, and the effective number of uncorrelated samples in the production region, ``Neff_max``.
The production (equilibrated) region of the timeseries is extracted as ``A_t_equil`` and then subsampled using the :func:`subsampleCorrelatedData` method with the provided statistical inefficiency ``g``.
Finally, the decorrelated samples are stored in ``A_n``.

Note that, by default, the statistical inefficiency is computed for every time origin in a call to :func:`detectEquilibration`, which can be slow.
If your dataset is more than a few hundred samples, you may want to evaluate only every ``nskip`` samples as potential time origins.
This may result in discarding slightly more data than strictly necessary, but may not have a significant impact if the timeseries is long.

.. python

   nskip = 10 # only try every 10 samples for time origins
   [t0, g, Neff_max] = timeseries.detectEquilibration(A_t, nskip=nskip)

Subsampling timeseries data
---------------------------

If there is no need to discard the initial transient to equilibration, the :func:`subsampleCorrelatedData` method can be used directly to identify an effectively uncorrelated subset of data.

.. python

   from pymbar import timeseries
   indices = timeseries.subsampleCorrelatedData(A_t_equil)
   A_n = A_t_equil[indices]

Here, the statistical inefficiency ``g`` is computed automatically.

Other utility timeseries functions
----------------------------------

A number of other useful functions for computing autocorrelation functions from one or more timeseries sampled from the same process are also provided.

.. automodule:: pymbar.timeseries
