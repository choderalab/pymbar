##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2016-2017 University of Colorado Boulder
# Copyright 2010-2017 Memorial Sloan-Kettering Cancer Center
# Portions of this software are Copyright 2010-2016 University of Virginia
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################
"""
Please reference the following if you use this code in your research:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177

This module contains implementations of

* bar - bidirectional estimator for free energy differences / Bennett acceptance ratio estimator
* exp - unidirectional estimator for free energy differences based on Zwanzig relation / exponential averaging
* exp_gauss - unidirectional estimator for free energy differences based on Zwanzig relation / exponential averaging, assuming the distribution is Gaussian.
"""

# =============================================================================================
# TODO
# * Fix computeBAR and computeEXP to be bar() and exp() to make them easier to find.
# * Make functions that don't need to be exported (like logsum) private by prefixing an underscore.
# * Make asymptotic covariance matrix computation more robust to over/underflow.
# * Double-check correspondence of comments to equation numbers once manuscript has been finalized.
# * Change self.nonzero_N_k_indices to self.states_with_samples
# =============================================================================================

__authors__ = "Michael R. Shirts and John D. Chodera."
__license__ = "MIT"

# =============================================================================================
# IMPORTS
# =============================================================================================
import logging
import numpy as np
from pymbar.utils import ParameterError, ConvergenceError, BoundsError, logsumexp

logger = logging.getLogger(__name__)


def bar_zero(w_F, w_R, DeltaF):
    """A function that when zeroed is equivalent to the solution of
    the Bennett acceptance ratio.

    from http://journals.aps.org/prl/pdf/10.1103/PhysRevLett.91.140601

        D_F = M + w_F - Delta F
        D_R = M + w_R - Delta F

    we want:

        \\sum_N_F (1+exp(D_F))^-1 = \\sum N_R N_R <(1+exp(-D_R))^-1>
        ln \\sum N_F (1+exp(D_F))^-1>_F = \\ln \\sum N_R exp((1+exp(-D_R))^(-1)>_R
        ln \\sum N_F (1+exp(D_F))^-1>_F - \\ln \\sum N_R exp((1+exp(-D_R))^(-1)>_R = 0

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
        a variable that is zeroed when DeltaF satisfies bar.

    Examples
    --------
    Compute free energy difference between two specified samples of work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> DeltaF = bar_zero(w_F, w_R, 0.0)

    """

    np.seterr(over="raise")  # raise exceptions to overflows
    w_F = np.array(w_F, np.float64)
    w_R = np.array(w_R, np.float64)
    DeltaF = float(DeltaF)

    # Recommended stable implementation of bar.

    # Determine number of forward and reverse work values provided.
    T_F = float(w_F.size)  # number of forward work values
    T_R = float(w_R.size)  # number of reverse work values

    # Compute log ratio of forward and reverse counts.
    M = np.log(T_F / T_R)

    # Compute log numerator. We have to watch out for overflows.  We
    # do this by making sure that 1+exp(x) doesn't overflow, choosing
    # to always exponentiate a negative number.

    # log f(W) = - log [1 + exp((M + W - DeltaF))]
    #          = - log ( exp[+maxarg] [exp[-maxarg] + exp[(M + W - DeltaF) - maxarg]] )
    #          = - maxarg - log(exp[-maxarg] + exp[(M + W - DeltaF) - maxarg])
    # where maxarg = max((M + W - DeltaF), 0)

    exp_arg_F = M + w_F - DeltaF
    # use boolean logic to zero out the ones that are less than 0, but not if greater than zero.
    max_arg_F = np.choose(np.less(0.0, exp_arg_F), (0.0, exp_arg_F))
    try:
        log_f_F = -max_arg_F - np.log(np.exp(-max_arg_F) + np.exp(exp_arg_F - max_arg_F))
    except ParameterError:
        # give up; if there's overflow, return zero
        logger.warning("The input data results in overflow in bar")
        return np.nan
    log_numer = logsumexp(log_f_F)

    # Compute log_denominator.
    # log f(R) = - log [1 + exp(-(M + W - DeltaF))]
    #          = - log ( exp[+maxarg] [exp[-maxarg] + exp[(M + W - DeltaF) - maxarg]] )
    #          = - maxarg - log[exp[-maxarg] + (T_F/T_R) exp[(M + W - DeltaF) - maxarg]]
    # where maxarg = max( -(M + W - DeltaF), 0)

    exp_arg_R = -(M - w_R - DeltaF)
    # use boolean logic to zero out the ones that are less than 0, but not if greater than zero.
    max_arg_R = np.choose(np.less(0.0, exp_arg_R), (0.0, exp_arg_R))
    try:
        log_f_R = -max_arg_R - np.log(np.exp(-max_arg_R) + np.exp(exp_arg_R - max_arg_R))
    except ParameterError:
        logger.info("The input data results in overflow in bar")
        return np.nan
    log_denom = logsumexp(log_f_R)

    # This function must be zeroed to find a root
    fzero = log_numer - log_denom

    np.seterr(
        over="warn"
    )  # return options to standard settings so we don't disturb other functionality.
    return fzero


def bar(
    w_F,
    w_R,
    DeltaF=0.0,
    compute_uncertainty=True,
    uncertainty_method="BAR",
    maximum_iterations=500,
    relative_tolerance=1.0e-12,
    verbose=False,
    method="false-position",
    iterated_solution=True,
):
    """Compute free energy difference using the Bennett acceptance ratio (BAR) method.

    Parameters
    ----------
    w_F : np.ndarray
        w_F[t] is the forward work value from snapshot t.
        t = 0...(T_F-1)  Length T_F is deduced from vector.
    w_R : np.ndarray
        w_R[t] is the reverse work value from snapshot t.
        t = 0...(T_R-1)  Length T_R is deduced from vector.
    DeltaF : float, optional, default=0.0
        DeltaF can be set to initialize the free energy difference with a guess
    compute_uncertainty : bool, optional, default=True
        if False, only the free energy is returned
    uncertainty_method : string, optional, default=''BAR''
        There are two possible uncertainty estimates for BAR.  One agrees with MBAR for two states exactly,
        and is indicated by "MBAR". The other estimator, which is the one originally derived for BAR, only
        agrees with MBAR in the limit of good overlap, and is designated 'BAR'
        See code comments below for derivations of the two methods.
    maximum_iterations : int, optional, default=500
        Can be set to limit the maximum number of iterations performed
    relative_tolerance : float, optional, default=1E-11
        Can be set to determine the relative tolerance convergence criteria (defailt 1.0e-11)
    verbose : bool
        Should be set to True if verbse debug output is desired (default False)
    method: str, optional, defualt='false-position'
        Choice of method to solve bar nonlinear equations: one of 'bisection', 'self-consistent-iteration' or 'false-position' (default : 'false-position').
    iterated_solution: bool, optional, default=True
        whether to fully solve the optimized bar equation to consistency, or to stop after one step, to be
        equivalent to transition matrix sampling.

    Returns
    -------
    dict
        'Delta_f' : float
            Free energy difference
        'dDelta_f' : float
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
    >>> results = bar(w_F, w_R)
    >>> print('Free energy difference is {:.3f} +- {:.3f} kT'.format(results['Delta_f'], results['dDelta_f']))
    Free energy difference is 1.088 +- 0.050 kT

    Test completion of various other schemes.

    >>> results = bar(w_F, w_R, method='self-consistent-iteration')
    >>> results = bar(w_F, w_R, method='false-position')
    >>> results = bar(w_F, w_R, method='bisection')

    """

    result_vals = dict()
    # if computing nonoptimized, one step value, we set the max-iterations
    # to 1, and the method to 'self-consistent-iteration'

    if not iterated_solution:
        maximum_iterations = 1
        method = "self-consistent-iteration"
        DeltaF_initial = DeltaF

    if method not in ["self-consistent-iteration", "false-position", "bisection"]:
        raise ParameterError("method {:d} is not defined for bar".format(method))

    if uncertainty_method not in ["BAR", "MBAR"]:
        raise ParameterError(
            "uncertainty_method {:d} is not defined for bar".format(uncertainty_method)
        )

    if method == "self-consistent-iteration":
        nfunc = 0

    if method == "bisection" or method == "false-position":
        UpperB = exp(w_F)["Delta_f"]
        LowerB = -exp(w_R)["Delta_f"]

        FUpperB = bar_zero(w_F, w_R, UpperB)
        FLowerB = bar_zero(w_F, w_R, LowerB)
        nfunc = 2

        if np.isnan(FUpperB) or np.isnan(FLowerB):
            # this data set is returning NAN -- will likely not work.  Return 0, print a warning:
            # consider returning more information about failure
            logger.warning(
                "BAR is likely to be inaccurate because of poor overlap. Improve the sampling, or decrease the spacing betweeen states.  For now, guessing that the free energy difference is 0 with no uncertainty."
            )
            if compute_uncertainty:
                result_vals["Delta_f"] = 0.0
                result_vals["dDelta_f"] = 0.0
                return result_vals

            else:
                result_vals["Delta_f"] = 0.0
                return result_vals

        while FUpperB * FLowerB > 0:
            # if they have the same sign, they do not bracket.  Widen the bracket until they have opposite signs.
            # There may be a better way to do this, and the above bracket should rarely fail.
            if verbose:
                logger.info("Initial brackets did not actually bracket, widening them")
            FAve = (UpperB + LowerB) / 2
            UpperB = UpperB - max(abs(UpperB - FAve), 0.1)
            LowerB = LowerB + max(abs(LowerB - FAve), 0.1)
            FUpperB = bar_zero(w_F, w_R, UpperB)
            FLowerB = bar_zero(w_F, w_R, LowerB)
            nfunc += 2

    # Iterate to convergence or until maximum number of iterations has been exceeded.

    for iteration in range(maximum_iterations + 1):

        DeltaF_old = DeltaF

        if method == "false-position":
            # Predict the new value
            if (LowerB == 0.0) and (UpperB == 0.0):
                DeltaF = 0.0
                FNew = 0.0
            else:
                DeltaF = UpperB - FUpperB * (UpperB - LowerB) / (FUpperB - FLowerB)
                FNew = bar_zero(w_F, w_R, DeltaF)
            nfunc += 1

            if FNew == 0:
                # Convergence is achieved.
                if verbose:
                    logger.info("Convergence achieved.")
                relative_change = 10 ** (-15)
                break

        if method == "bisection":
            # Predict the new value
            DeltaF = (UpperB + LowerB) / 2
            FNew = bar_zero(w_F, w_R, DeltaF)
            nfunc += 1

        if method == "self-consistent-iteration":
            DeltaF = -bar_zero(w_F, w_R, DeltaF) + DeltaF
            nfunc += 1

        # Check for convergence.
        if DeltaF == 0.0:
            # The free energy difference appears to be zero -- return.
            if verbose:
                logger.info("The free energy difference appears to be zero.")
            break

        if iterated_solution:
            relative_change = abs((DeltaF - DeltaF_old) / DeltaF)
            if verbose:
                logger.info("relative_change = {:12.3f}".format(relative_change))

            if (iteration > 0) and (relative_change < relative_tolerance):
                # Convergence is achieved.
                if verbose:
                    logger.info("Convergence achieved.")
                break

        if method == "false-position" or method == "bisection":
            if FUpperB * FNew < 0:
                # these two now bracket the root
                LowerB = DeltaF
                FLowerB = FNew
            elif FLowerB * FNew <= 0:
                # these two now bracket the root
                UpperB = DeltaF
                FUpperB = FNew
            else:
                message = "WARNING: Cannot determine bound on free energy"
                raise BoundsError(message)

        if verbose:
            logger.info("iteration {:5d}: DeltaF = {:16.3f}".format(iteration, DeltaF))

    # Report convergence, or warn user if not achieved.
    if iterated_solution:
        if iteration < maximum_iterations:
            if verbose:
                logger.info(
                    "Converged to tolerance of {:e} in {:d} iterations ({:d} function evaluations)".format(
                        relative_change, iteration, nfunc
                    )
                )
        else:
            message = "WARNING: Did not converge to within specified tolerance. max_delta = {:f}, TOLERANCE = {:f}, MAX_ITS = {:d}".format(
                relative_change, relative_tolerance, maximum_iterations
            )
            raise ConvergenceError(message)

    if compute_uncertainty:

        #############
        # Compute asymptotic variance estimate using Eq. 10a of Bennett,
        # 1976 (except with n_1<f>_1^2 in the second denominator, it is
        # an error in the original.
        #
        # NOTE: The 'BAR' and 'MBAR' estimators
        # do not agree for poor overlap. This is not because of
        # numerical precision, but because they are fundamentally
        # different estimators. For poor overlap, 'MBAR' diverges high,
        # and 'BAR' diverges by being too low. In situations they are
        # noticeably from each other, they are also pretty different
        # from the true answer (obtained by calculating the standard
        # deviation over lots of realizations).
        #
        # First, we examine the 'BAR' equation. Rederive from Bennett, substituting (8) into (7)
        #
        # (8)    -> W = [q0/n0 exp(-U1) + q1/n1 exp(-U0)]^-1
        #             <(W exp(-U1))^2 >_0         <(W exp(-U0))^2 >_1
        # (7)    -> -----------------------  +   -----------------------   - 1/n0 - 1/n1
        #            n_0 [<(W exp(-U1)>_0]^2      n_1 [<(W exp(-U0)>_1]^2
        #
        #     Const cancels out of top and bottom.   Wexp(-U0) = [q0/n0 exp(-(U1-U0)) + q1/n1]^-1
        #                                                      =  n1/q1 [n1/n0 q0/q1 exp(-(U1-U0)) + 1]^-1
        #                                                      =  n1/q1 [exp (M+(F1-F0)-(U1-U0)+1)^-1]
        #                                                      =  n1/q1 f(x)
        #                                            Wexp(-U1) = [q0/n0 + q1/n1 exp(-(U0-U1))]^-1
        #                                                      =  n0/q0 [1 + n0/n1 q1/q0 exp(-(U0-U1))]^-1
        #                                                      =  n0/q0 [1 + exp(-M+[F0-F1)-(U0-U1))]^-1
        #                                                      =  n0/q0 f(-x)
        #
        #
        #           <(W exp(-U1))^2 >_0          <(W exp(-U0))^2 >_1
        #  (7) -> -----------------------   +  -----------------------   - 1/n0 - 1/n1
        #         n_0 [<(W exp(-U1)>_0]^2      n_1 [<(W exp(-U0)>_1]^2
        #
        #            <[n0/q0 f(-x)]^2>_0        <[n1/q1 f(x)]^2>_1
        #         -----------------------  +  ------------------------   -1/n0 -1/n1
        #           n_0 <n0/q0 f(-x)>_0^2      n_1 <n1/q1 f(x)>_1^2
        #
        #        1      <[f(-x)]^2>_0                 1        <[f(x)]^2>_1
        #        -  [-----------------------  - 1]  + -  [------------------------  - 1]
        #        n0      <f(-x)>_0^2                  n1      n_1<f(x)>_1^2
        #
        # where f = the fermi function, 1/(1+exp(-x))
        #
        # This formula the 'BAR' equation works for works for free
        # energies (F0-F1) that don't satisfy the bar equation.  The
        # 'MBAR' equation, detailed below, only works for free energies
        # that satisfy the equation.
        #
        #
        # Now, let's look at the MBAR version of the uncertainty.  This
        # is written (from Shirts and Chodera, JPC, 129, 124105, Equation E9) as
        #
        #       [ n0<f(x)f(-x)>_0 + n1<f(x)f(-x)_1 ]^-1 - n0^-1 - n1^-1
        #
        #       we note the f(-x) + f(x)  = 1, and change this to:
        #
        #       [ n0<(1-f(-x)f(-x)>_0 + n1<f(x)(1-f(x))_1 ]^-1 - n0^-1 - n1^-1
        #
        #       [ n0<f(-x)-f(-x)^2)>_0 + n1<f(x)-f(x)^2)_1 ]^-1 - n0^-1 - n1^-1
        #
        #                                         1                                         1     1
        #       --------------------------------------------------------------------    -  --- - ---
        #          n0 <f(-x)>_0 - n0 <[f(-x)]^2>_0 + n1 <f(x)>_1 + n1 <[f(x)]^2>_1          n0    n1
        #
        #
        # Removing the factor of - (T_F + T_R)/(T_F*T_R)) from both, we compare:
        #
        #           <[f(-x)]^2>_0          <[f(x)]^2>_1
        #       [------------------]  + [---------------]
        #          n0 <f(-x)>_0^2          n1 <f(x)>_1^2
        #
        #                                         1
        #       --------------------------------------------------------------------
        #          n0 <f(-x)>_0 - n0 <[f(-x)]^2>_0 + n1 <f(x)>_1 + n1 <[f(x)]^2>_1
        #
        # denote: <f(-x)>_0 = afF
        #         <f(-x)^2>_0 = afF2
        #         <f(x)>_1 = afR
        #         <f(x)^2>_1 = afF2
        #
        # Then we can look at both of these as:
        #
        # variance_bar = (afF2/afF**2)/T_F + (afR2/afR**2)/T_R
        # variance_MBAR = 1/(afF*T_F - afF2*T_F + afR*T_R - afR2*T_R)
        #
        # Rearranging:
        #
        # variance_bar = (afF2/afF**2)/T_F + (afR2/afR**2)/T_R
        # variance_MBAR = 1/(afF*T_F + afR*T_R - (afF2*T_F +  afR2*T_R))
        #
        # # check the steps below?  Not quite sure.
        # variance_bar = (afF2/afF**2) + (afR2/afR**2)  = (afF2 + afR2)/afR**2
        # variance_MBAR = 1/(afF + afR - (afF2 +  afR2)) = 1/(2*afR-(afF2+afR2))
        #
        # Definitely not the same.  Now, the reason that they both work
        # for high overlap is still not clear. We will determine the
        # difference at some point.
        #
        # see https://github.com/choderalab/pymbar/issues/281 for more information.
        #
        # Now implement the two computations.
        ###############

        # Determine number of forward and reverse work values provided.
        T_F = float(w_F.size)  # number of forward work values
        T_R = float(w_R.size)  # number of reverse work values

        # Compute log ratio of forward and reverse counts.
        M = np.log(T_F / T_R)

        if iterated_solution:
            C = M - DeltaF
        else:
            C = M - DeltaF_initial

        # In theory, overflow handling should not be needed now, because we use numlogexp or a custom routine?

        # fF = 1 / (1 + np.exp(w_F + C)), but we need to handle overflows
        exp_arg_F = w_F + C
        max_arg_F = np.max(exp_arg_F)
        log_fF = -np.log(np.exp(-max_arg_F) + np.exp(exp_arg_F - max_arg_F))
        afF = np.exp(logsumexp(log_fF) - max_arg_F) / T_F

        # fR = 1 / (1 + np.exp(w_R - C)), but we need to handle overflows
        exp_arg_R = w_R - C
        max_arg_R = np.max(exp_arg_R)
        log_fR = -np.log(np.exp(-max_arg_R) + np.exp(exp_arg_R - max_arg_R))
        afR = np.exp(logsumexp(log_fR) - max_arg_R) / T_R

        afF2 = np.exp(logsumexp(2 * log_fF) - 2 * max_arg_F) / T_F
        afR2 = np.exp(logsumexp(2 * log_fR) - 2 * max_arg_R) / T_R

        nrat = (T_F + T_R) / (T_F * T_R)  # same for both methods

        if uncertainty_method == "BAR":
            variance = (afF2 / afF**2) / T_F + (afR2 / afR**2) / T_R - nrat
            dDeltaF = np.sqrt(variance)
        elif uncertainty_method == "MBAR":
            # OR equivalently
            vartemp = (afF - afF2) * T_F + (afR - afR2) * T_R
            dDeltaF = np.sqrt(1.0 / vartemp - nrat)
        else:
            message = "ERROR: bar uncertainty method {:s} is not defined".format(
                uncertainty_method
            )
            raise ParameterError(message)

        if verbose:
            logger.info("DeltaF = {:8.3f} +- {:8.3f}".format(DeltaF, dDeltaF))
        result_vals["Delta_f"] = DeltaF
        result_vals["dDelta_f"] = dDeltaF
        return result_vals

    else:
        if verbose:
            logger.info("DeltaF = {:8.3f}".format(DeltaF))
        result_vals["Delta_f"] = DeltaF
        return result_vals


def bar_overlap(w_F, w_R):
    """Compute overlap between forward and backward ensembles (using MBAR definition of overlap)

    Parameters
    ----------
    w_F : np.ndarray
        w_F[t] is the forward work value from snapshot t.
        t = 0...(T_F-1)  Length T_F is deduced from vector.
    w_R : np.ndarray
        w_R[t] is the reverse work value from snapshot t.
        t = 0...(T_R-1)  Length T_R is deduced from vector.

    Returns
    -------
    overlap : float
        The overlap: 0 denotes no overlap, 1 denotes complete overlap
    """
    from pymbar import MBAR

    N_k = np.array([len(w_F), len(w_R)])
    N = N_k.sum()
    u_kn = np.zeros([2, N])
    u_kn[1, 0 : N_k[0]] = w_F[:]
    u_kn[0, N_k[0] : N] = w_R[:]
    mbar = MBAR(u_kn, N_k)

    # Check to make sure u_kn has been correctly formed
    results = bar(w_F, w_R)
    bar_df = results["Delta_f"]
    bar_ddf = results["dDelta_f"]

    assert np.isclose(
        mbar.f_k[1] - mbar.f_k[0], bar_df
    ), f"BAR: {bar_df} +- {bar_ddF} | MBAR: {mbar.f_k[1] - mbar.f_k[0]}"

    return mbar.compute_overlap()["scalar"]


def exp(w_F, compute_uncertainty=True, is_timeseries=False):
    """Estimate free energy difference using one-sided (unidirectional) exponential averaging (EXP).

    Parameters
    ----------
    w_F : np.ndarray, float
        w_F[t] is the forward work value from snapshot t.  t = 0...(T-1)  Length T is deduced from vector.
    compute_uncertainty : bool, optional, default=True
        if False, will disable computation of the statistical uncertainty (default: True)
    is_timeseries : bool, default=False
        if True, correlation in data is corrected for by estimation of statisitcal inefficiency (default: False)
        Use this option if you are providing correlated timeseries data and have not subsampled the data to produce uncorrelated samples.

    Returns
    -------
    'Delta_f' : float
        Free energy difference
    'dDelta_f' : float
        Estimated standard deviation of free energy difference

    Notes
    -----
    If you are prodividing correlated timeseries data, be sure to set the 'timeseries' flag to True

    Examples
    --------

    Compute the free energy difference given a sample of forward work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> results = exp(w_F)
    >>> print('Forward free energy difference is {:.3f} +- {:.3f} kT'.format(results['Delta_f'], results['dDelta_f']))
    Forward free energy difference is 1.088 +- 0.076 kT
    >>> results = exp(w_R)
    >>> print('Reverse free energy difference is {:.3f} +- {:.3f} kT'.format(results['Delta_f'], results['dDelta_f']))
    Reverse free energy difference is -1.073 +- 0.082 kT

    """

    result_vals = dict()

    # Get number of work measurements.
    T = float(np.size(w_F))  # number of work measurements

    # Estimate free energy difference by exponential averaging using DeltaF = - log < exp(-w_F) >
    DeltaF = -(logsumexp(-w_F) - np.log(T))

    if compute_uncertainty:
        # Compute x_i = np.exp(-w_F_i - max_arg)
        max_arg = np.max(-w_F)  # maximum argument
        x = np.exp(-w_F - max_arg)

        # Compute E[x] = <x> and dx
        Ex = x.mean()

        # Compute effective number of uncorrelated samples.
        g = 1.0  # statistical inefficiency
        if is_timeseries:
            # Estimate statistical inefficiency of x timeseries.
            from pymbar import timeseries

            g = timeseries.statistical_inefficiency(x, x)

        # Estimate standard error of E[x].
        dx = np.std(x) / np.sqrt(T / g)

        # dDeltaF = <x>^-1 dx
        dDeltaF = dx / Ex

        # Return estimate of free energy difference and uncertainty.
        result_vals["Delta_f"] = DeltaF
        result_vals["dDelta_f"] = dDeltaF
    else:
        result_vals["Delta_f"] = DeltaF

    return result_vals


def exp_gauss(w_F, compute_uncertainty=True, is_timeseries=False):
    """Estimate free energy difference using gaussian approximation to one-sided (unidirectional) exponential averaging.

    Parameters
    ----------
    w_F : np.ndarray, float
        w_F[t] is the forward work value from snapshot t.  t = 0...(T-1)  Length T is deduced from vector.
    compute_uncertainty : bool, optional, default=True
        if False, will disable computation of the statistical uncertainty (default: True)
    is_timeseries : bool, default=False
        if True, correlation in data is corrected for by estimation of statisitcal inefficiency (default: False)
        Use this option if you are providing correlated timeseries data and have not subsampled the data to
        produce uncorrelated samples.

    Returns
    -------
    Results dictionary with keys:
        'Delta_f' : float
            Free energy difference between the two states
        'dDelta_f' : float
            Estimated standard deviation of free energy difference between the two states

    Notes
    -----
    If you are providing correlated timeseries data, be sure to set the 'timeseries' flag to ``True``

    Examples
    --------
    Compute the free energy difference given a sample of forward work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> results = exp_gauss(w_F)
    >>> print('Forward Gaussian approximated free energy difference is {:.3f} +- {:.3f} kT'.format(results['Delta_f'], results['dDelta_f']))
    Forward Gaussian approximated free energy difference is 1.049 +- 0.089 kT
    >>> results = exp_gauss(w_R)
    >>> print('Reverse Gaussian approximated free energy difference is {:.3f} +- {:.3f} kT'.format(results['Delta_f'], results['dDelta_f']))
    Reverse Gaussian approximated free energy difference is -1.073 +- 0.080 kT

    """

    # Get number of work measurements.
    T = float(np.size(w_F))  # number of work measurements

    var = np.var(w_F)
    # Estimate free energy difference by Gaussian approximation, dG = <U> - 0.5*var(U)
    DeltaF = np.average(w_F) - 0.5 * var

    result_vals = dict()
    if compute_uncertainty:
        # Compute effective number of uncorrelated samples.
        g = 1.0  # statistical inefficiency
        T_eff = T
        if is_timeseries:
            # Estimate statistical inefficiency of x timeseries.
            from pymbar import timeseries

            g = timeseries.statistical_inefficiency(w_F, w_F)

            T_eff = T / g
        # Estimate standard error of E[x].
        dx2 = var / T_eff + 0.5 * var * var / (T_eff - 1)
        dDeltaF = np.sqrt(dx2)

        # Return estimate of free energy difference and uncertainty.
        result_vals["Delta_f"] = DeltaF
        result_vals["dDelta_f"] = dDeltaF
    else:
        result_vals["Delta_f"] = DeltaF
    return result_vals
