##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2010-2014 University of Virginia, Memorial Sloan-Kettering Cancer Center
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with pymbar. If not, see <http://www.gnu.org/licenses/>.
##############################################################################
"""
Please reference the following if you use this code in your research:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177

This module contains implementations of

* BAR - bidirectional estimator for free energy differences / Bennett acceptance ratio estimator

"""

#=============================================================================================
# TODO
# * Fix computeBAR and computeEXP to be BAR() and EXP() to make them easier to find.
# * Make functions that don't need to be exported (like logsum) private by prefixing an underscore.
# * Make asymptotic covariance matrix computation more robust to over/underflow.
# * Double-check correspondence of comments to equation numbers once manuscript has been finalized.
# * Change self.nonzero_N_k_indices to self.states_with_samples
#=============================================================================================


__authors__ = "Michael R. Shirts and John D. Chodera."
__license__ = "LGPL 2.1"

#=============================================================================================
# IMPORTS
#=============================================================================================
import numpy
import numpy.linalg
from pymbar.utils import _logsum, ParameterError, ConvergenceError, BoundsError
from pymbar.exp import EXP



def BARzero(w_F, w_R, DeltaF):
    """Bennett acceptance ratio function to be zeroed to solve for BAR.

    Parameters
    ----------
    w_F : np.ndarray
        w_F[t] is the forward work value from snapshot t.
        t = 0...(T_F-1)  Length T_F is deduced from vector.
    w_R : np.ndarray
        w_R[t] is the reverse work value from snapshot t.
        t = 0...(T_R-1)  Length T_R is deduced from vector.
    DeltaF : float
        Our current guess

    Returns
    -------
    fzero : float
        a variable that is zeroed when DeltaF satisfies BAR.

    Examples
    --------
    Compute free energy difference between two specified samples of work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> DeltaF = BARzero(w_F, w_R, 0.0)

    """

    numpy.seterr(over='raise')  # raise exceptions to overflows
    w_F = numpy.array(w_F, numpy.float64)
    w_R = numpy.array(w_R, numpy.float64)
    DeltaF = float(DeltaF)

    # Recommended stable implementation of BAR.

    # Determine number of forward and reverse work values provided.
    T_F = float(w_F.size)  # number of forward work values
    T_R = float(w_R.size)  # number of reverse work values

    # Compute log ratio of forward and reverse counts.
    M = numpy.log(T_F / T_R)

    # Compute log numerator.
    # log f(W) = - log [1 + exp((M + W - DeltaF))]
    #          = - log ( exp[+maxarg] [exp[-maxarg] + exp[(M + W - DeltaF) - maxarg]] )
    #          = - maxarg - log[exp[-maxarg] + (T_F/T_R) exp[(M + W - DeltaF) - maxarg]]
    # where maxarg = max( (M + W - DeltaF) )

    exp_arg_F = (M + w_F - DeltaF)
    max_arg_F = numpy.choose(numpy.greater(0.0, exp_arg_F), (0.0, exp_arg_F))
    try:
        log_f_F = - max_arg_F - numpy.log(numpy.exp(-max_arg_F) + numpy.exp(exp_arg_F - max_arg_F))
    except:
        # give up; if there's overflow, return zero
        print "The input data results in overflow in BAR"
        return numpy.nan
    log_numer = _logsum(log_f_F) - numpy.log(T_F)

    # Compute log_denominator.
    # log_denom = log < f(-W) exp[-W] >_R
    # NOTE: log [f(-W) exp(-W)] = log f(-W) - W
    exp_arg_R = (M - w_R - DeltaF)
    max_arg_R = numpy.choose(numpy.greater(0.0, exp_arg_R), (0.0, exp_arg_R))
    try:
        log_f_R = - max_arg_R - numpy.log(numpy.exp(-max_arg_R) + numpy.exp(exp_arg_R - max_arg_R)) - w_R
    except:
        print "The input data results in overflow in BAR"
        return numpy.nan
    log_denom = _logsum(log_f_R) - numpy.log(T_R)

    # This function must be zeroed to find a root
    fzero = DeltaF - (log_denom - log_numer)

    numpy.seterr(over='warn')  # return options to standard settings so we don't disturb other functionality.
    return fzero


def BAR(w_F, w_R, DeltaF=0.0, compute_uncertainty=True, maximum_iterations=500, relative_tolerance=1.0e-11, verbose=False, method='false-position', iterated_solution=True):
    """Compute free energy difference using the Bennett acceptance ratio (BAR) method.

    Parameters
    ----------
    w_F : numpy.ndarray
        w_F[t] is the forward work value from snapshot t.
        t = 0...(T_F-1)  Length T_F is deduced from vector.
    w_R : np.ndarray
        w_R[t] is the reverse work value from snapshot t.
        t = 0...(T_R-1)  Length T_R is deduced from vector.
    DeltaF : float, optional, default=0.0
        DeltaF can be set to initialize the free energy difference with a guess
    compute_uncertainty : bool, optional, default=True
        if False, only the free energy is returned
    maximum_iterations : int, optional, default=500
        can be set to limit the maximum number of iterations performed
    relative_tolerance : float, optional, default=1E-11
        can be set to determine the relative tolerance convergence criteria (defailt 1.0e-11)
    verbose : bool
        should be set to True if verbse debug output is desired (default False)
    method : str, optional, defualt='false-position'
        choice of method to solve BAR nonlinear equations, one of 'self-consistent-iteration' or 'false-position' (default: 'false-position')
    iterated_solution : bool, optional, default=True
        whether to fully solve the optimized BAR equation to consistency, or to stop after one step, to be 
        equivalent to transition matrix sampling.

    Returns
    -------
    DeltaF : float
        Free energy difference
    dDeltaF : float
     Estimated standard deviation of free energy difference

    References
    ----------

    [1] Shirts MR, Bair E, Hooker G, and Pande VS. Equilibrium free energies from nonequilibrium
    measurements using maximum-likelihood methods. PRL 91(14):140601, 2003.

    Notes
    -----
    The false position method is used to solve the implicit equation.

    Examples
    --------
    Compute free energy difference between two specified samples of work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> [DeltaF, dDeltaF] = BAR(w_F, w_R)
    >>> print 'Free energy difference is %.3f +- %.3f kT' % (DeltaF, dDeltaF)
    Free energy difference is 1.088 +- 0.050 kT

    Test various other schemes.

    >>> [DeltaF, dDeltaF] = BAR(w_F, w_R, method='self-consistent-iteration')
    >>> [DeltaF, dDeltaF] = BAR(w_F, w_R, method='false-position')
    >>> [DeltaF, dDeltaF] = BAR(w_F, w_R, method='bisection')

    """

    # if computing nonoptimized, one step value, we set the max-iterations
    # to 1, and the method to 'self-consistent-iteration'
    if not iterated_solution:
        maximum_iterations = 1
        method = 'self-consistent-iteration'
        DeltaF_initial = DeltaF

    if method == 'self-consistent-iteration':
        nfunc = 0

    if method == 'bisection' or method == 'false-position':
        UpperB = EXP(w_F)[0]
        LowerB = -EXP(w_R)[0]

        FUpperB = BARzero(w_F, w_R, UpperB)
        FLowerB = BARzero(w_F, w_R, LowerB)
        nfunc = 2

        if (numpy.isnan(FUpperB) or numpy.isnan(FLowerB)):
            # this data set is returning NAN -- will likely not work.  Return 0, print a warning:
            print "Warning: BAR is likely to be inaccurate because of poor overlap. Improve the sampling, or decrease the spacing betweeen states.  For now, guessing that the free energy difference is 0 with no uncertainty."
            if compute_uncertainty:
                return [0.0, 0.0]
            else:
                return 0.0

        while FUpperB * FLowerB > 0:
            # if they have the same sign, they do not bracket.  Widen the bracket until they have opposite signs.
            # There may be a better way to do this, and the above bracket should rarely fail.
            if verbose:
                print 'Initial brackets did not actually bracket, widening them'
            FAve = (UpperB + LowerB) / 2
            UpperB = UpperB - max(abs(UpperB - FAve), 0.1)
            LowerB = LowerB + max(abs(LowerB - FAve), 0.1)
            FUpperB = BARzero(w_F, w_R, UpperB)
            FLowerB = BARzero(w_F, w_R, LowerB)
            nfunc += 2

    # Iterate to convergence or until maximum number of iterations has been exceeded.

    for iteration in range(maximum_iterations):

        DeltaF_old = DeltaF

        if method == 'false-position':
            # Predict the new value
            if (LowerB == 0.0) and (UpperB == 0.0):
                DeltaF = 0.0
                FNew = 0.0
            else:
                DeltaF = UpperB - FUpperB * (UpperB - LowerB) / (FUpperB - FLowerB)
                FNew = BARzero(w_F, w_R, DeltaF)
            nfunc += 1

            if FNew == 0:
                # Convergence is achieved.
                if verbose:
                    print "Convergence achieved."
                relative_change = 10 ^ (-15)
                break

        if method == 'bisection':
            # Predict the new value
            DeltaF = (UpperB + LowerB) / 2
            FNew = BARzero(w_F, w_R, DeltaF)
            nfunc += 1

        if method == 'self-consistent-iteration':
            DeltaF = -BARzero(w_F, w_R, DeltaF) + DeltaF
            nfunc += 1

        # Check for convergence.
        if (DeltaF == 0.0):
            # The free energy difference appears to be zero -- return.
            if verbose:
                print "The free energy difference appears to be zero."
            if compute_uncertainty:
                return [0.0, 0.0]
            else:
                return 0.0

        if iterated_solution:
            relative_change = abs((DeltaF - DeltaF_old) / DeltaF)
            if verbose:
                print "relative_change = %12.3f" % relative_change

            if ((iteration > 0) and (relative_change < relative_tolerance)):
                # Convergence is achieved.
                if verbose:
                    print "Convergence achieved."
                break

        if method == 'false-position' or method == 'bisection':
            if FUpperB * FNew < 0:
                # these two now bracket the root
                LowerB = DeltaF
                FLowerB = FNew
            elif FLowerB * FNew <= 0:
                # these two now bracket the root
                UpperB = DeltaF
                FUpperB = FNew
            else:
                message = 'WARNING: Cannot determine bound on free energy'
                raise BoundsError(message)

        if verbose:
            print "iteration %5d : DeltaF = %16.3f" % (iteration, DeltaF)

    # Report convergence, or warn user if not achieved.
    if iterated_solution:
        if iteration < maximum_iterations:
            if verbose:
                print 'Converged to tolerance of %e in %d iterations (%d function evaluations)' % (relative_change, iteration, nfunc)
        else:
            message = 'WARNING: Did not converge to within specified tolerance. max_delta = %f, TOLERANCE = %f, MAX_ITS = %d' % (relative_change, relative_tolerance, maximum_iterations)
            raise ConvergenceError(message)

    if compute_uncertainty:
        # Compute asymptotic variance estimate using Eq. 10a of Bennett, 1976 (except with n_1<f>_1^2 in
        # the second denominator, it is an error in the original
        # NOTE: The numerical stability of this computation may need to be improved.

        # Determine number of forward and reverse work values provided.
        T_F = float(w_F.size)  # number of forward work values
        T_R = float(w_R.size)  # number of reverse work values
        # Compute log ratio of forward and reverse counts.
        M = numpy.log(T_F / T_R)

        if iterated_solution:
            C = M - DeltaF
        else:
            C = M - DeltaF_initial

        fF = 1 / (1 + numpy.exp(w_F + C))
        fR = 1 / (1 + numpy.exp(w_R - C))

        afF2 = (numpy.average(fF)) ** 2
        afR2 = (numpy.average(fR)) ** 2

        vfF = numpy.var(fF) / T_F
        vfR = numpy.var(fR) / T_R

        variance = vfF / afF2 + vfR / afR2

        dDeltaF = numpy.sqrt(variance)
        if verbose:
            print "DeltaF = %8.3f +- %8.3f" % (DeltaF, dDeltaF)
        return (DeltaF, dDeltaF)
    else:
        if verbose:
            print "DeltaF = %8.3f" % (DeltaF)
        return DeltaF

#=============================================================================================
# For compatibility with 2.0.1-beta
#=============================================================================================

deprecation_warning = """
Warning
-------
This method name is deprecated, and provided for backward-compatibility only.
It may be removed in future versions.
"""

def computeBARzero(*args, **kwargs):
    return BARzero(*args, **kwargs)
computeBARzero.__doc__ = BARzero.__doc__ + deprecation_warning

def computeBAR(*args, **kwargs):
    return BAR(*args, **kwargs)
computeBAR.__doc__ = BAR.__doc__ + deprecation_warning

def _compatibilityDoctests():
    """
    Backwards-compatibility doctests.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> DeltaF = BARzero(w_F, w_R, 0.0)
    >>> [DeltaF, dDeltaF] = computeBAR(w_F, w_R)
    """
    pass
