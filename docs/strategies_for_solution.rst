.. _strategies_for_solution:

Getting started
###############

.. highlight: bash

Approaching the MBAR euqations
=====================

``scipy`` solutions
-------------------

The multistate reweighting approach to calculate free energies can be
formulated in several ways.  The multistate reweighting equations are
a set of coupled, implicit equations for the free energies of $K$
states, given samples from these $K$ states. If one can calculate the
energies of each of the $K$ states, for each sample, then one can
solve for the $K$ free energies satisfying the equations. The
solutions are unique only up to an overall constant, which ``pymbar``
removes by setting the first free energy to zero to 0, leaving $K-1$
free energies.

By rearrangement, this set of self-consistent equations can be written
as simultaneous roots to $K$ equations.  This set of roots also turns
out to be the Jacobian of single maximum likelihood function of all
the free energies.  We then can find the MBAR solutions by either
maximization/minimization techiques, or by root finding.

Because the second derivative of the likelihood is always negative,
there is only one possivle solution. However, if there is poor
overlap, it is not uncommon that some of the optimal $f_k$s could be
in extremely flat region of solution space, and therefore have
significant round-off erros resulting in slow or no convergence to the
solution, and low overlap can also lead to underflow and overflow
leading to crashed solutions.

-----------------
``scipy`` solvers
-----------------

``pymbar`` is set up to use the ``scipy.optimize.minimize`` and
``scipy.optimize.roots`` functionality to perform this
minimization. We use only the gradient-based methods, as the
analytical gradient-based optimization is obtainable from the MBAR
equations.  Available ``scipy.optimize.minimize`` methods include
"L-BFGS-B", "dogleg", "CG", "BFGS", "Newton-CG", "TNC", "trust-ncg",
"trust-krylov", "trust-exact", and "SLSQP". and
``scipy.optimize.roots`` options are ``hybr`` and ``lm``. Methods that
take a Hessian ("dogleg", "Newton-CG", "trust-ncg", "trust-krylov",
"trust-exact") are passed the analytical Hessian.  Options can be
passed to each of these methods through the ``MBAR`` object
initialization interface.

------------------
Built-in solutions
------------------

In addition to the ``scipy`` solcers, ``pymbar`` also includes an
adaptive solver designed directly for MBAR.  At every step, it
calculates both the next iteration of the self-consistent iterative
formula presented in Shirts et al., and takes a Newton-Raphson in.  In
both cases, it calculates the gradient of each step, and selects the
move that makes the magnitude of the gradient (i.e. the dot product of
the gradient with itself) smallest. Far from the solution, the
self-consistent iteration tends have the smaller gradient, while
closer to the solution, the Newton-Raphson step tends to have the
smallest gradient.

-----------------------------
Constructing solver protocols
-----------------------------


-----------------------------
Calculating uncertainties
-----------------------------

The MBAR equations contain analytical estimates of uncertainties.
These are essentially, however, the functional form is bit more
complicated, since they include modifications for error propagation
with implicit equations.

For free energies.


In some cases, to peform additional error analysis, one might need access to


-----------------------------
Bootstrapped uncertaintes
-----------------------------


The solutions for bootstrapped data should be relatively close to the
solutions for the original data set.  Therefore, when solving the
iterative equations, we start iteration from the point 

Note that users have complete control over the solver sequence for
bootstrapped solutions, using the same API as for solvers of the
original solution.

The bootstrapped solution.

---------------


------------------------
Calculating expectations
------------------------

::


Running the tests
=================
Running the tests is a great way to verify that everything is working.
The test suite uses `nose <https://nose.readthedocs.org/en/latest/>`_, in addition to `statsmodels <http://statsmodels.sourceforge.net/>`_ and `pytables <http://www.pytables.org/>`_, which you can install via ``conda``:

::
   $ conda install nose statsmodels pytables

You can then run the tests with:

::
   $ pytest -v pymbar
