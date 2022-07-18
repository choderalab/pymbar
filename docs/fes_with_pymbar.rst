.. _fes_with_pymbar:

Free energy surfaces with pymbar
#################################

Installing ``pymbar``
=====================

-------------------
Free energy surfces
-------------------

``pymbar`` can be used to estimate free energy surfaces using samples
from *K* biased simulations.  It is important to note that MBAR itself
is not enough to generate a free energy surface.  MBAR takes a set of
samples from *K* different states, and can compute the weight that
should be given to each sample in in the unbiased state, i.e. the
state in which one desires to compute the free energy surface. Thus,
there can be no MBAR estimator of the free energy surface; that would
consist only in a set of weighted delta functions.  This is done by
initializing the ``pymbar.FES`` class, which takes :math:`u_{kn}` and :math:`N_k`
matrices and passes them to MBAR.

The second step that needs to be carried out is to determine the best
approximation of the continuous function that the samples are
estimated from. ``pymbar.FES`` supports several methods to estimate
this continuous function.  ``generate_fes``, given an initialized MBAR
object, a set of points, the energies at that point, and a method,
generates an object that contains the FES information.  The current
options are ``histogram``, ``kde``, and spline.  ``histogram`` behaves
as one might expect, creating a free energy surface as a histogram,
and refer to ``FES.rst`` for additional information. ``kde`` creates a
kernel density approximation, using the
``sklearn.neighbors.KernelDensity function``, and parameters can be
passed to that function using the ``kde_parameters`` keyword.
Finally, the ``spline`` method uses a maximum likelhood approach to
calculate the spline most consistent with the input data, using the
formalism presented in Shirts et al. :cite:`shirts_fes_2020`.  The ``spline``
functionality includes the ability to perform Monte Carlo sampling in
the spline parameters to generate confidence intervals for the points
in the spline curve.

``histogram`` and ``kde`` methods can generate multidimesional free
energy surfaces, while ``splines`` for now is limited to a single free
energy surface.

The method `get_fes` return values of the free energy surface at the
specified coordinates, and when available, returns the uncertainties
in the values as well.

.. autoclass:: pymbar.FES
