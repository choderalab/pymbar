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
__license__ = "MIT"

#=============================================================================================
# IMPORTS
#=============================================================================================
import numpy as np
import numpy.linalg
from pymbar.utils import ParameterError, ConvergenceError, BoundsError, logsumexp
from pymbar.exp import EXP

def BARzero(w_F, w_R, DeltaF):
    """A function that when zeroed is equivalent to the solution of
    the Bennett acceptance ratio.

    from http://journals.aps.org/prl/pdf/10.1103/PhysRevLett.91.140601
    D_F = M + w_F - Delta F
    D_R = M + w_R - Delta F

    we want:
    \sum_N_F (1+exp(D_F))^-1 = \sum N_R N_R <(1+exp(-D_R))^-1>
    ln \sum N_F (1+exp(D_F))^-1>_F = \ln \sum N_R exp((1+exp(-D_R))^(-1)>_R
    ln \sum N_F (1+exp(D_F))^-1>_F - \ln \sum N_R exp((1+exp(-D_R))^(-1)>_R = 0

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

    np.seterr(over='raise')  # raise exceptions to overflows
    w_F = np.array(w_F, np.float64)
    w_R = np.array(w_R, np.float64)
    DeltaF = float(DeltaF)

    # Recommended stable implementation of BAR.

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

    exp_arg_F = (M + w_F - DeltaF)
    # use boolean logic to zero out the ones that are less than 0, but not if greater than zero.
    max_arg_F = np.choose(np.less(0.0, exp_arg_F), (0.0, exp_arg_F))
    try:
        log_f_F = - max_arg_F - np.log(np.exp(-max_arg_F) + np.exp(exp_arg_F - max_arg_F))
    except:
        # give up; if there's overflow, return zero
        print("The input data results in overflow in BAR")
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
        log_f_R = - max_arg_R - np.log(np.exp(-max_arg_R) + np.exp(exp_arg_R - max_arg_R))
    except:
        print("The input data results in overflow in BAR")
        return np.nan
    log_denom = logsumexp(log_f_R)

    # This function must be zeroed to find a root
    fzero = log_numer - log_denom

    np.seterr(over='warn')  # return options to standard settings so we don't disturb other functionality.
    return fzero


def BAR(w_F, w_R, DeltaF=0.0, compute_uncertainty=True, uncertainty_method='BAR',maximum_iterations=500, relative_tolerance=1.0e-12, verbose=False, method='false-position', iterated_solution=True, return_dict=False):
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
    uncertainty_method: string, optional, default=BAR
        There are two possible uncertainty estimates for BAR.  One agrees with MBAR for two states exactly;
        The other only agrees with MBAR in the limit of good overlap. See below.
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
    return_dict : bool, default False
        If true, returns are a dict, else they are a tuple

    Returns
    -------
    'Delta_f' : float
        Free energy difference
        If return_dict, key is 'Delta_f'
    'dDelta_f': float
        Estimated standard deviation of free energy difference
        If return_dict, key is 'dDelta_f'


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
    >>> results = BAR(w_F, w_R, return_dict=True)
    >>> print('Free energy difference is {:.3f} +- {:.3f} kT'.format(results['Delta_f'], results['dDelta_f']))
    Free energy difference is 1.088 +- 0.050 kT

    Test completion of various other schemes.

    >>> results = BAR(w_F, w_R, method='self-consistent-iteration', return_dict=True)
    >>> results = BAR(w_F, w_R, method='false-position', return_dict=True)
    >>> results = BAR(w_F, w_R, method='bisection', return_dict=True)

    """

    result_vals = dict()
    # if computing nonoptimized, one step value, we set the max-iterations
    # to 1, and the method to 'self-consistent-iteration'

    if not iterated_solution:
        maximum_iterations = 1
        method = 'self-consistent-iteration'
        DeltaF_initial = DeltaF

    if method == 'self-consistent-iteration':
        nfunc = 0

    if method == 'bisection' or method == 'false-position':
        UpperB = EXP(w_F, return_dict=True)['Delta_f']
        LowerB = -EXP(w_R, return_dict=True)['Delta_f']

        FUpperB = BARzero(w_F, w_R, UpperB)
        FLowerB = BARzero(w_F, w_R, LowerB)
        nfunc = 2

        if (np.isnan(FUpperB) or np.isnan(FLowerB)):
            # this data set is returning NAN -- will likely not work.  Return 0, print a warning:
            # consider returning more information about failure
            print("Warning: BAR is likely to be inaccurate because of poor overlap. Improve the sampling, or decrease the spacing betweeen states.  For now, guessing that the free energy difference is 0 with no uncertainty.")
            if compute_uncertainty:
                result_vals['Delta_f'] = 0.0 
                result_vals['dDelta_f'] = 0.0
                if return_dict:
                    return result_vals
                return 0.0, 0.0
            else:
                result_vals['Delta_f'] = 0.0
                if return_dict:
                    return result_vals
                return 0.0

        while FUpperB * FLowerB > 0:
            # if they have the same sign, they do not bracket.  Widen the bracket until they have opposite signs.
            # There may be a better way to do this, and the above bracket should rarely fail.
            if verbose:
                print('Initial brackets did not actually bracket, widening them')
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
                    print('Convergence achieved.')
                relative_change = 10 ** (-15)
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
                print('The free energy difference appears to be zero.')
            break

        if iterated_solution:
            relative_change = abs((DeltaF - DeltaF_old) / DeltaF)
            if verbose:
                print("relative_change = {:12.3f}".format(relative_change))

            if ((iteration > 0) and (relative_change < relative_tolerance)):
                # Convergence is achieved.
                if verbose:
                    print("Convergence achieved.")
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
            print("iteration {:5d}: DeltaF = {:16.3f}".format(iteration, DeltaF))

    # Report convergence, or warn user if not achieved.
    if iterated_solution:
        if iteration < maximum_iterations:
            if verbose:
                print('Converged to tolerance of {:e} in {:d} iterations ({:d} function evaluations)'.format(relative_change, iteration, nfunc))
        else:
            message = 'WARNING: Did not converge to within specified tolerance. max_delta = {:f}, TOLERANCE = {:f}, MAX_ITS = %d'.format(relative_change, relative_tolerance, maximum_iterations)
            raise ConvergenceError(message)

    if compute_uncertainty:

        '''

        Compute asymptotic variance estimate using Eq. 10a of Bennett,
        1976 (except with n_1<f>_1^2 in the second denominator, it is
        an error in the original NOTE: The 'BAR' and 'MBAR' estimators
        do not agree for poor overlap. This is not because of
        numerical precision, but because they are fundamentally
        different estimators. For poor overlap, 'MBAR' diverges high,
        and 'BAR' diverges by being too low. In situations they are
        noticeably from each other, they are also pretty different
        from the true answer (obtained by calculating the standard
        deviation over lots of realizations).

        First, we examine the 'BAR' equation. Rederive from Bennett, substituting (8) into (7)

        (8)    -> W = [q0/n0 exp(-U1) + q1/n1 exp(-U0)]^-1
                    <(W exp(-U1))^2 >_0         <(W exp(-U0))^2 >_1
        (7)    -> -----------------------  +   -----------------------   - 1/n0 - 1/n1
                   n_0 [<(W exp(-U1)>_0]^2      n_1 [<(W exp(-U0)>_1]^2

            Const cancels out of top and bottom.   Wexp(-U0) = [q0/n0 exp(-(U1-U0)) + q1/n1]^-1
                                                             =  n1/q1 [n1/n0 q0/q1 exp(-(U1-U0)) + 1]^-1
                                                             =  n1/q1 [exp (M+(F1-F0)-(U1-U0)+1)^-1]
                                                             =  n1/q1 f(x)
                                                   Wexp(-U1) = [q0/n0 + q1/n1 exp(-(U0-U1))]^-1
                                                             =  n0/q0 [1 + n0/n1 q1/q0 exp(-(U0-U1))]^-1
                                                             =  n0/q0 [1 + exp(-M+[F0-F1)-(U0-U1))]^-1
                                                             =  n0/q0 f(-x)


                  <(W exp(-U1))^2 >_0          <(W exp(-U0))^2 >_1
         (7) -> -----------------------   +  -----------------------   - 1/n0 - 1/n1
                n_0 [<(W exp(-U1)>_0]^2      n_1 [<(W exp(-U0)>_1]^2

                   <[n0/q0 f(-x)]^2>_0        <[n1/q1 f(x)]^2>_1
                -----------------------  +  ------------------------   -1/n0 -1/n1
                  n_0 <n0/q0 f(-x)>_0^2      n_1 <n1/q1 f(x)>_1^2

               1      <[f(-x)]^2>_0                 1        <[f(x)]^2>_1
               -  [-----------------------  - 1]  + -  [------------------------  - 1]
               n0      <f(-x)>_0^2                  n1      n_1<f(x)>_1^2

        where f = the fermi function, 1/(1+exp(-x))

        This formula the 'BAR' equation works for works for free
        energies (F0-F1) that don't satisfy the BAR equation.  The
        'MBAR' equation, detailed below, only works for free energies
        that satisfy the equation.


        Now, let's look at the MBAR version of the uncertainty.  This
        is written (from Shirts and Chodera, JPC, 129, 124105, Equation E9) as

              [ n0<f(x)f(-x)>_0 + n1<f(x)f(-x)_1 ]^-1 - n0^-1 - n1^-1

              we note the f(-x) + f(x)  = 1, and change this to:

              [ n0<(1-f(-x)f(-x)>_0 + n1<f(x)(1-f(x))_1 ]^-1 - n0^-1 - n1^-1

              [ n0<f(-x)-f(-x)^2)>_0 + n1<f(x)-f(x)^2)_1 ]^-1 - n0^-1 - n1^-1

                                                1                                         1     1
              --------------------------------------------------------------------    -  --- - ---
                 n0 <f(-x)>_0 - n0 <[f(-x)]^2>_0 + n1 <f(x)>_1 + n1 <[f(x)]^2>_1          n0    n1


        Removing the factor of - (T_F + T_R)/(T_F*T_R)) from both, we compare:

                  <[f(-x)]^2>_0          <[f(x)]^2>_1
              [------------------]  + [---------------]
                 n0 <f(-x)>_0^2          n1 <f(x)>_1^2

                                                1
              --------------------------------------------------------------------
                 n0 <f(-x)>_0 - n0 <[f(-x)]^2>_0 + n1 <f(x)>_1 + n1 <[f(x)]^2>_1

        denote: <f(-x)>_0 = afF
                <f(-x)^2>_0 = afF2
                <f(x)>_1 = afR
                <f(x)^2>_1 = afF2

        Then we can look at both of these as:

        variance_BAR = (afF2/afF**2)/T_F + (afR2/afR**2)/T_R
        variance_MBAR = 1/(afF*T_F - afF2*T_F + afR*T_R - afR2*T_R)

        Rearranging:

        variance_BAR = (afF2/afF**2)/T_F + (afR2/afR**2)/T_R
        variance_MBAR = 1/(afF*T_F + afR*T_R - (afF2*T_F +  afR2*T_R))

        # check the steps below?  Not quite sure.
        variance_BAR = (afF2/afF**2) + (afR2/afR**2)  = (afF2 + afR2)/afR**2
        variance_MBAR = 1/(afF + afR - (afF2 +  afR2)) = 1/(2*afR-(afF2+afR2))

        Definitely not the same.  Now, the reason that they both work
        for high overlap is still not clear. We will determine the
        difference at some point.

        see https://github.com/choderalab/pymbar/issues/281 for more information.

        Now implement the two computations.
        '''

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
        exp_arg_F = (w_F + C)
        max_arg_F  = np.max(exp_arg_F)
        log_fF = - np.log(np.exp(-max_arg_F) + np.exp(exp_arg_F - max_arg_F))
        afF  = np.exp(logsumexp(log_fF)-max_arg_F)/T_F

        # fR = 1 / (1 + np.exp(w_R - C)), but we need to handle overflows
        exp_arg_R = (w_R - C)
        max_arg_R  = np.max(exp_arg_R)
        log_fR = - np.log(np.exp(-max_arg_R) + np.exp(exp_arg_R - max_arg_R))
        afR = np.exp(logsumexp(log_fR)-max_arg_R)/T_R

        afF2 = np.exp(logsumexp(2*log_fF)-2*max_arg_F)/T_F
        afR2 = np.exp(logsumexp(2*log_fR)-2*max_arg_R)/T_R

        nrat = (T_F + T_R)/(T_F * T_R)   # same for both methods

        if uncertainty_method == 'BAR':
            variance = (afF2/afF**2)/T_F + (afR2/afR**2)/T_R - nrat
            dDeltaF = np.sqrt(variance)
        elif uncertainty_method == 'MBAR':
            # OR equivalently
            vartemp = ((afF - afF2)*T_F + (afR - afR2)*T_R)
            dDeltaF = np.sqrt(1.0/vartemp - nrat)
        else:
            message = 'ERROR: BAR uncertainty method {:s} is not defined'.format(uncertainty_method)
            raise ParameterError(message)

        if verbose:
            print("DeltaF = {:8.3f} +- {:8.3f}".format(DeltaF, dDeltaF))
        result_vals['Delta_f'] = DeltaF
        result_vals['dDelta_f'] = dDeltaF
        if return_dict:
            return result_vals
        return DeltaF, dDeltaF
    else:
        if verbose:
            print("DeltaF = {:8.3f}".format(DeltaF))
        result_vals['Delta_f'] = DeltaF
        if return_dict:
            return result_vals
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
