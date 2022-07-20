##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2016-2020 University of Colorado Boulder
# Copyright 2010-2020 Memorial Sloan-Kettering Cancer Center
# Portions of this software are Copyright (c) 2010-2016 University of Virginia
# Portions of this software are Copyright (c) 2006-2007 The Regents of the University of California.  All Rights Reserved.
# Portions of this software are Copyright (c) 2007-2008 Stanford University and Columbia University.
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
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
A module for extracting uncorrelated samples from correlated timeseries data.

This module provides various tools that allow one to examine the correlation functions and
integrated autocorrelation times in correlated timeseries data, compute statistical inefficiencies,
and automatically extract uncorrelated samples for data analysis.

Please reference the following if you use this code in your research:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008
http://dx.doi.org/10.1063/1.2978177

[2] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
histogram analysis method for the analysis of simulated and parallel tempering simulations.
JCTC 3(1):26-41, 2007.

"""

# =============================================================================================
# TODO
# * Implement unit tests that generate timeseries with various levels of Gaussian correlation to test all methods.
# * Add Zwanzig procedure for estimating statistical uncertainties in correlation functions
# (by making Gaussian process assumptions).
# =============================================================================================


__authors__ = "Michael R. Shirts and John D. Chodera."
__license__ = "MIT"


# =============================================================================================
# IMPORTS
# =============================================================================================
import logging
import math
import numpy as np
from pymbar.utils import ParameterError

# =============================================================================================
# Issue warning on import.
# =============================================================================================

logger = logging.getLogger(__name__)
LongWarning = (
    "Warning on use of the timeseries module: If the inherent timescales of the system "
    "are long compared to those being analyzed, this statistical inefficiency may be an underestimate.  "
    "The estimate presumes the use of many statistically independent samples.  "
    "Tests should be performed to assess whether this condition is satisfied.   "
    "Be cautious in the interpretation of the data."
)
logger.warning(LongWarning)
# sys.stderr.write(LongWarning + '\n')

# =============================================================================================
# METHODS
# =============================================================================================


def statistical_inefficiency(A_n, B_n=None, fast=False, mintime=3, fft=False):
    """Compute the (cross) statistical inefficiency of (two) timeseries.

    Parameters
    ----------
    A_n : np.ndarray, float
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    B_n : np.ndarray, float, optional, default=None
        B_n[n] is nth value of timeseries B.  Length is deduced from vector.
        If supplied, the cross-correlation of timeseries A and B will be estimated instead of the
        autocorrelation of timeseries A.
    fast : bool, optional, default=False
        f True, will use faster (but less accurate) method to estimate correlation
        time, described in Ref. [1] (default: False).  This is ignored
        when B_n=None and fft=True.
    mintime : int, optional, default=3
        minimum amount of correlation function to compute (default: 3)
        The algorithm terminates after computing the correlation time out to mintime when the
        correlation function first goes negative.  Note that this time may need to be increased
        if there is a strong initial negative peak in the correlation function.
    fft : bool, optional, default=False
        If fft=True and B_n=None, then use the fft based approach, as
        implemented in statistical_inefficiency_fft().

    Returns
    -------
    g : np.ndarray,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.

    Notes
    -----
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    The fast method described in Ref [1] is used to compute g.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
    histogram analysis method for the analysis of simulated and parallel tempering simulations.
    JCTC 3(1):26-41, 2007.

    Examples
    --------

    Compute statistical inefficiency of timeseries data with known correlation time.

    >>> from pymbar.testsystems import correlated_timeseries_example
    >>> A_n = correlated_timeseries_example(N=100000, tau=5.0)
    >>> g = statistical_inefficiency(A_n, fast=True)

    """

    # Create numpy copies of input arguments.
    A_n = np.array(A_n)

    if fft and B_n is None:
        return statistical_inefficiency_fft(A_n, mintime=mintime)

    if B_n is not None:
        B_n = np.array(B_n)
    else:
        B_n = np.array(A_n)

    # Get the length of the timeseries.
    N = A_n.size

    # Be sure A_n and B_n have the same dimensions.
    if A_n.shape != B_n.shape:
        raise ParameterError("A_n and B_n must have same dimensions.")

    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0

    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()

    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B

    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean()  # standard estimator to ensure C(0) = 1

    # Trap the case where this covariance is zero, and we cannot proceed.
    if sigma2_AB == 0:
        raise ParameterError(
            "Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency"
        )

    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while t < N - 1:

        # compute normalized fluctuation correlation function at time t
        C = np.sum(dA_n[0 : (N - t)] * dB_n[t:N] + dB_n[0 : (N - t)] * dA_n[t:N]) / (
            2.0 * float(N - t) * sigma2_AB
        )
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break

        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t) / float(N)) * float(increment)

        # Increment t and the amount by which we increment t.
        t += increment

        # Increase the interval if "fast mode" is on.
        if fast:
            increment += 1

    # g must be at least unity
    if g < 1.0:
        g = 1.0

    # Return the computed statistical inefficiency.
    return g


# =============================================================================================


def statistical_inefficiency_multiple(A_kn, fast=False, return_correlation_function=False):
    """Estimate the statistical inefficiency from multiple stationary timeseries (of potentially differing lengths).

    Parameters
    ----------
    A_kn : list of np.ndarrays
        A_kn[k] is the kth timeseries, and A_kn[k][n] is nth value of timeseries k.  Length is deduced from arrays.

    fast : bool, optional, default=False
        f True, will use faster (but less accurate) method to estimate correlation
        time, described in Ref. [1] (default: False)
    return_correlation_function : bool, optional, default=False
        if True, will also return estimates of normalized fluctuation correlation function that were computed (default: False)

    Returns
    -------
    g : np.ndarray,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.
    Ct : list (of tuples)
        Ct[n] = (t, C) with time t and normalized correlation function estimate C is returned as well if return_correlation_function is set to True

    Notes
    -----
    The autocorrelation of the timeseries is used to compute the statistical inefficiency.
    The normalized fluctuation autocorrelation function is computed by averaging the unnormalized raw correlation functions.
    The fast method described in Ref [1] is used to compute g.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.

    Examples
    --------

    Estimate statistical efficiency from multiple timeseries of different lengths.

    >>> from pymbar import testsystems
    >>> N_k = [1000, 2000, 3000, 4000, 5000]
    >>> tau = 5.0 # exponential relaxation time
    >>> A_kn = [ testsystems.correlated_timeseries_example(N=N, tau=tau) for N in N_k ]
    >>> g = statistical_inefficiency_multiple(A_kn)

    Also return the values of the normalized fluctuation autocorrelation function that were computed.

    >>> [g, Ct] = statistical_inefficiency_multiple(A_kn, return_correlation_function=True)

    """

    # Convert A_kn into a list of arrays if it is not in this form already.
    if type(A_kn) == np.ndarray:
        A_kn_list = list()
        if A_kn.ndim == 1:
            A_kn_list.append(A_kn.copy())
        else:
            K, N = A_kn.shape
            for k in range(K):
                A_kn_list.append(A_kn[k, :].copy())
        A_kn = A_kn_list

    # Determine number of timeseries.
    K = len(A_kn)

    # Get the length of each timeseries.
    N_k = np.zeros([K], np.int32)
    for k in range(K):
        N_k[k] = A_kn[k].size

    # Compute average timeseries length.
    Navg = np.array(N_k, np.float64).mean()

    # Determine total number of samples.
    N = np.sum(N_k)

    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0

    # Compute sample mean.
    mu = 0.0
    for k in range(K):
        mu += np.sum(A_kn[k])
    mu /= float(N)

    # Construct and store fluctuation timeseries.
    dA_kn = list()
    for k in range(K):
        dA_n = A_kn[k] - mu
        dA_kn.append(dA_n.copy())

    # Compute sample variance from mean of squared fluctuations, to ensure that C(0) = 1.
    sigma2 = 0.0
    for k in range(K):
        sigma2 += np.sum(dA_kn[k] ** 2)
    sigma2 /= float(N)

    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0

    # Initialize storage for correlation function.
    Ct = (
        list()
    )  # Ct[n] is a tuple (t, C) of the time lag t and estimate of normalized fluctuation correlation function C

    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while t < N_k.max() - 1:
        # compute unnormalized correlation function
        numerator = 0.0
        denominator = 0.0
        for k in range(K):
            if t >= N_k[k]:
                continue  # skip trajectory if lag time t is greater than its length
            dA_n = dA_kn[k]  # retrieve trajectory
            x = dA_n[0 : (N_k[k] - t)] * dA_n[t : N_k[k]]
            numerator += np.sum(x)  # accumulate contribution from trajectory k
            denominator += float(x.size)  # count how many overlapping time segments we've included

        C = numerator / denominator

        # compute normalized fluctuation correlation function at time t
        C = C / sigma2
        # logger.info("C[{:5d}] = {:16f} ({:16f} / {:16f})".format(t, C, numerator, denominator))

        # Store estimate of correlation function.
        Ct.append((t, C))

        # Terminate if the correlation function has crossed zero.
        # Note that we've added a hack (t > 10) condition to avoid terminating too early in correlation functions that have a strong negative peak at
        if (C <= 0.0) and (t > 10):
            break

        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t) / Navg) * float(increment)

        # Increment t and the amount by which we increment t.
        t += increment

        # Increase the interval if "fast mode" is on.
        if fast:
            increment += 1

    # g must be at least unity
    if g < 1.0:
        g = 1.0

    # Return statistical inefficency and correlation function estimate, if requested.
    if return_correlation_function:
        return g, Ct

    # Return the computed statistical inefficiency.
    return g


# =============================================================================================


def integrated_autocorrelation_time(A_n, B_n=None, fast=False, mintime=3):
    """Estimate the integrated autocorrelation time.

    See Also
    --------
    statisticalInefficiency

    """

    g = statistical_inefficiency(A_n, B_n, fast, mintime)
    tau = (g - 1.0) / 2.0
    return tau


# =============================================================================================


def integrated_autocorrelation_timeMultiple(A_kn, fast=False):
    """Estimate the integrated autocorrelation time from multiple timeseries.

    See Also
    --------
    statistical_inefficiency_multiple

    """

    g = statistical_inefficiency_multiple(A_kn, fast, False)
    tau = (g - 1.0) / 2.0
    return tau


# =============================================================================================


def normalized_fluctuation_correlation_function(A_n, B_n=None, N_max=None, norm=True):
    """Compute the normalized fluctuation (cross) correlation function of (two) stationary timeseries.

    C(t) = (<A(t) B(t)> - <A><B>) / (<AB> - <A><B>)

    This may be useful in diagnosing odd time-correlations in timeseries data.

    Parameters
    ----------
    A_n : np.ndarray
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    B_n : np.ndarray
        B_n[n] is nth value of timeseries B.  Length is deduced from vector.
    N_max : int, default=None
        if specified, will only compute correlation function out to time lag of N_max
    norm: bool, optional, default=True
        if False will return the unnormalized correlation function D(t) = <A(t) B(t)>

    Returns
    -------
    C_n : np.ndarray
        C_n[n] is the normalized fluctuation auto- or cross-correlation function for timeseries A(t) and B(t).

    Notes
    -----
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    This procedure may be slow.
    The statistical error in C_n[n] will grow with increasing n.  No effort is made here to estimate the uncertainty.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
    histogram analysis method for the analysis of simulated and parallel tempering simulations.
    JCTC 3(1):26-41, 2007.

    Examples
    --------

    Estimate normalized fluctuation correlation function.

    >>> from pymbar import testsystems
    >>> A_t = testsystems.correlated_timeseries_example(N=10000, tau=5.0)
    >>> C_t = normalized_fluctuation_correlation_function(A_t, N_max=25)

    """

    # If B_n is not specified, set it to be identical to A_n.
    if B_n is None:
        B_n = A_n

    # Create np copies of input arguments.
    A_n = np.array(A_n)
    B_n = np.array(B_n)

    # Get the length of the timeseries.
    N = A_n.size

    # Set maximum time to compute correlation functon for.
    if (not N_max) or (N_max > N - 1):
        N_max = N - 1

    # Be sure A_n and B_n have the same dimensions.
    if A_n.shape != B_n.shape:
        raise ParameterError("A_n and B_n must have same dimensions.")

    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0

    # Compute means and variance.
    mu_A = A_n.mean()
    mu_B = B_n.mean()

    # Make temporary copies at high precision with means subtracted off.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B

    # sigma2_AB = sum((A_n-mu_A) * (B_n-mu_B)) / (float(N)-1.0) # unbiased estimator
    sigma2_AB = (dA_n * dB_n).mean()  # standard estimator to ensure C(0) = 1
    if sigma2_AB == 0:
        raise ParameterError(
            "Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency"
        )

    # allocate storage for normalized fluctuation correlation function
    C_n = np.zeros([N_max + 1], np.float64)

    # Compute normalized correlation function.
    t = 0
    for t in range(0, N_max + 1):
        # compute normalized fluctuation correlation function at time t
        C_n[t] = np.sum(dA_n[0 : (N - t)] * dB_n[t:N] + dB_n[0 : (N - t)] * dA_n[t:N]) / (
            2.0 * float(N - t) * sigma2_AB
        )

    # Return the computed correlation function
    if norm:
        return C_n
    else:
        return C_n * sigma2_AB + mu_A * mu_B


# =============================================================================================


def normalized_fluctuation_correlation_function_multiple(
    A_kn, B_kn=None, N_max=None, norm=True, truncate=False
):
    """Compute the normalized fluctuation (cross) correlation function of (two) timeseries from multiple timeseries samples.

    C(t) = (<A(t) B(t)> - <A><B>) / (<AB> - <A><B>)
    This may be useful in diagnosing odd time-correlations in timeseries data.

    Parameters
    ----------
    A_kn : Python list of numpy arrays
        A_kn[k] is the kth timeseries, and A_kn[k][n] is nth value of timeseries k.  Length is deduced from arrays.
    B_kn : Python list of numpy arrays
        B_kn[k] is the kth timeseries, and B_kn[k][n] is nth value of timeseries k.  B_kn[k] must have same length as A_kn[k]
    N_max : int, optional, default=None
        if specified, will only compute correlation function out to time lag of N_max
    norm: bool, optional, default=True
        if False, will return unnormalized D(t) = <A(t) B(t)>
    truncate: bool, optional, default=False
        if True, will stop calculating the correlation function when it goes below 0

    Returns
    -------
    C_n[n] : np.ndarray
        The normalized fluctuation auto- or cross-correlation function for timeseries A(t) and B(t).

    Notes
    -----
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    This procedure may be slow.
    The statistical error in C_n[n] will grow with increasing n.  No effort is made here to estimate the uncertainty.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
    histogram analysis method for the analysis of simulated and parallel tempering simulations.
    JCTC 3(1):26-41, 2007.

    Examples
    --------

    Estimate a portion of the normalized fluctuation autocorrelation function from multiple timeseries of different length.

    >>> from pymbar import testsystems
    >>> N_k = [1000, 2000, 3000, 4000, 5000]
    >>> tau = 5.0 # exponential relaxation time
    >>> A_kn = [ testsystems.correlated_timeseries_example(N=N, tau=tau) for N in N_k ]
    >>> C_n = normalized_fluctuation_correlation_function_multiple(A_kn, N_max=25)

    """

    # If B_kn is not specified, define it to be identical with A_kn.
    if B_kn is None:
        B_kn = A_kn

    # TODO: Change this to support other iterable types, like sets.
    # Make sure A_kn and B_kn are both lists
    if (type(A_kn) is not list) or (type(B_kn) is not list):
        raise ParameterError("A_kn and B_kn must each be a list of numpy arrays.")

    # Ensure the same number of timeseries are stored in A_kn and B_kn.
    if len(A_kn) != len(B_kn):
        raise ParameterError(
            "A_kn and B_kn must contain corresponding timeseries -- different numbers of timeseries detected in each."
        )

    # Determine number of timeseries stored.
    K = len(A_kn)

    # Ensure both observable trajectories in each timeseries are of the same length.
    for k in range(K):
        A_n = A_kn[k]
        B_n = B_kn[k]
        if A_n.size != B_n.size:
            raise ParameterError(
                "A_kn and B_kn must contain corresponding timeseries -- lack of correspondence in timeseries lenghts detected."
            )

    # Get the length of each timeseries.
    N_k = np.zeros([K], np.int32)
    for k in range(K):
        N_k[k] = A_kn[k].size

    # Determine total number of samples.
    N = np.sum(N_k)

    # Set maximum time to compute correlation functon for.
    if (not N_max) or (N_max > max(N_k) - 1):
        N_max = max(N_k) - 1

    # Compute means.
    mu_A = 0.0
    mu_B = 0.0
    for k in range(K):
        mu_A += np.sum(A_kn[k])
        mu_B += np.sum(B_kn[k])
    mu_A /= float(N)
    mu_B /= float(N)

    # Compute fluctuation timeseries.
    dA_kn = list()
    dB_kn = list()
    for k in range(K):
        dA_n = A_kn[k] - mu_A
        dB_n = B_kn[k] - mu_B
        dA_kn.append(dA_n)
        dB_kn.append(dB_n)

    # Compute covariance.
    sigma2_AB = 0.0
    for k in range(K):
        sigma2_AB += np.sum(dA_kn[k] * dB_kn[k])
    sigma2_AB /= float(N)

    # allocate storage for normalized fluctuation correlation function
    C_n = np.zeros([N_max + 1], np.float64)

    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 0
    negative = False
    for t in range(0, N_max + 1):
        # compute unnormalized correlation function
        numerator = 0.0
        denominator = 0.0
        for k in range(K):
            if t >= N_k[k]:
                continue  # skip this trajectory if t is longer than the timeseries
            numerator += np.sum(dA_kn[k][0 : (N_k[k] - t)] * dB_kn[k][t : N_k[k]])
            denominator += float(N_k[k] - t)
            if truncate and numerator < 0:
                negative = True
        C = numerator / denominator

        # compute normalized fluctuation correlation function at time t
        C /= sigma2_AB

        # Store correlation function.
        C_n[t] = C

        if negative:
            break

    # Return the computed fluctuation correlation function.
    if norm:
        return C_n[:t]
    else:
        return C_n[:t] * sigma2_AB + mu_A * mu_B


# =============================================================================================


def subsample_correlated_data(A_t, g=None, fast=False, conservative=False, verbose=False):
    """Determine the indices of an uncorrelated subsample of the data.

    Parameters
    ----------
    A_t : np.ndarray
        A_t[t] is the t-th value of timeseries A(t).  Length is deduced from vector.
    g : float, optional
        if provided, the statistical inefficiency g is used to subsample the timeseries -- otherwise it will be computed (default: None)
    fast : bool, optional, default=False
        fast can be set to True to give a less accurate but very quick estimate (default: False)
    conservative : bool, optional, default=False
        if set to True, uniformly-spaced indices are chosen with interval ceil(g), where
        g is the statistical inefficiency.  Otherwise, indices are chosen non-uniformly with interval of
        approximately g in order to end up with approximately T/g total indices
    verbose : bool, optional, default=False
        if True, some output is printed

    Returns
    -------
    indices : list of int
        the indices of an uncorrelated subsample of the data

    Notes
    -----
    The statistical inefficiency is computed with the function computeStatisticalInefficiency().

    Examples
    --------

    Subsample a correlated timeseries to extract an effectively uncorrelated dataset.

    >>> from pymbar import testsystems
    >>> A_t = testsystems.correlated_timeseries_example(N=10000, tau=5.0) # generate a test correlated timeseries
    >>> indices = subsample_correlated_data(A_t) # compute indices of uncorrelated timeseries
    >>> A_n = A_t[indices] # extract uncorrelated samples

    Extract uncorrelated samples from multiple timeseries data from the same process.

    >>> # Generate multiple correlated timeseries data of different lengths.
    >>> T_k = [1000, 2000, 3000, 4000, 5000]
    >>> K = len(T_k) # number of timeseries
    >>> tau = 5.0 # exponential relaxation time
    >>> A_kt = [ testsystems.correlated_timeseries_example(N=T, tau=tau) for T in T_k ] # A_kt[k] is correlated timeseries k
    >>> # Estimate statistical inefficiency from all timeseries data.
    >>> g = statistical_inefficiency_multiple(A_kt)
    >>> # Count number of uncorrelated samples in each timeseries.
    >>> N_k = np.array([ len(subsample_correlated_data(A_t, g=g)) for A_t in A_kt ]) # N_k[k] is the number of uncorrelated samples in timeseries k
    >>> N = N_k.sum() # total number of uncorrelated samples
    >>> # Subsample all trajectories to produce uncorrelated samples
    >>> A_kn = [ A_t[subsample_correlated_data(A_t, g=g)] for A_t in A_kt ] # A_kn[k] is uncorrelated subset of trajectory A_kt[t]
    >>> # Concatenate data into one timeseries.
    >>> A_n = np.zeros([N], np.float32) # A_n[n] is nth sample in concatenated set of uncorrelated samples
    >>> A_n[0:N_k[0]] = A_kn[0]
    >>> for k in range(1,K): A_n[N_k[0:k].sum():N_k[0:k+1].sum()] = A_kn[k]

    """

    # Create np copy of arrays.
    A_t = np.array(A_t)

    # Get the length of the timeseries.
    T = A_t.size

    # Compute the statistical inefficiency for the timeseries.
    if not g:
        if verbose:
            logger.info("Computing statistical inefficiency...")
        g = statistical_inefficiency(A_t, A_t, fast=fast)
        if verbose:
            logger.info("g = {:f}".format(g))

    if conservative:
        # Round g up to determine the stride we can use to pick out regularly-spaced uncorrelated samples.
        stride = int(math.ceil(g))
        if verbose:
            logger.info("conservative subsampling: using stride of {:d}".format(stride))

        # Assemble list of indices of uncorrelated snapshots.
        indices = range(0, T, stride)
    else:
        # Choose indices as floor(n*g), with n = 0,1,2,..., until we run out of data.
        indices = []
        n = 0
        while int(round(n * g)) < T:
            t = int(round(n * g))
            # ensure we don't sample the same point twice
            if (n == 0) or (t != indices[n - 1]):
                indices.append(t)
            n += 1
        if verbose:
            logger.info("standard subsampling: using average stride of {:f}".format(g))

    # Number of samples in subsampled timeseries.
    N = len(indices)

    if verbose:
        logger.info(
            "The resulting subsampled set has {:d} samples (original timeseries had {:d}).".format(
                N, T
            )
        )

    # Return the list of indices of uncorrelated snapshots.
    return indices


def detect_equilibration(A_t, fast=True, nskip=1):
    """Automatically detect equilibrated region of a dataset using a heuristic that maximizes number of effectively uncorrelated samples.

    Parameters
    ----------
    A_t : np.ndarray
        timeseries
    nskip : int, optional, default=1
        number of samples to sparsify data by in order to speed equilibration detection

    Returns
    -------
    t : int
        start of equilibrated data
    g : float
        statistical inefficiency of equilibrated data
    Neff_max : float
        number of uncorrelated samples

    Notes
    -----
    If your input consists of some period of equilibration followed by
    a constant sequence, this function treats the trailing constant sequence
    as having Neff = 1.

    Examples
    --------

    Determine start of equilibrated data for a correlated timeseries.

    >>> from pymbar import testsystems
    >>> A_t = testsystems.correlated_timeseries_example(N=1000, tau=5.0) # generate a test correlated timeseries
    >>> [t, g, Neff_max] = detect_equilibration(A_t) # compute indices of uncorrelated timeseries

    Determine start of equilibrated data for a correlated timeseries with a shift.

    >>> from pymbar import testsystems
    >>> A_t = testsystems.correlated_timeseries_example(N=1000, tau=5.0) + 2.0 # generate a test correlated timeseries
    >>> B_t = testsystems.correlated_timeseries_example(N=10000, tau=5.0) # generate a test correlated timeseries
    >>> C_t = np.concatenate([A_t, B_t])
    >>> [t, g, Neff_max] = detect_equilibration(C_t, nskip=50) # compute indices of uncorrelated timeseries

    """
    # TODO: Consider implementing a binary search for Neff_max.

    T = A_t.size

    # Special case if timeseries is constant.
    if A_t.std() == 0.0:
        return 0, 1, 1  # Changed from Neff=N to Neff=1 after issue #122

    g_t = np.ones([T - 1], np.float32)
    Neff_t = np.ones([T - 1], np.float32)
    for t in range(0, T - 1, nskip):
        try:
            g_t[t] = statistical_inefficiency(A_t[t:T], fast=fast)
        except ParameterError:  # Fix for issue https://github.com/choderalab/pymbar/issues/122
            g_t[t] = T - t + 1
        Neff_t[t] = (T - t + 1) / g_t[t]
    Neff_max = Neff_t.max()
    t = Neff_t.argmax()
    g = g_t[t]

    return t, g, Neff_max


def statistical_inefficiency_fft(A_n, mintime=3):
    """Compute the (cross) statistical inefficiency of (two) timeseries.

    Parameters
    ----------
    A_n : np.ndarray, float
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    mintime : int, optional, default=3
        minimum amount of correlation function to compute (default: 3)
        The algorithm terminates after computing the correlation time out to mintime when the
        correlation function first goes negative.  Note that this time may need to be increased
        if there is a strong initial negative peak in the correlation function.

    Returns
    -------
    g : np.ndarray,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.

    Notes
    -----
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    The fast method described in Ref [1] is used to compute g.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.

    """
    try:
        import statsmodels.api as sm
    except ImportError as err:
        err.args = (
            err.args[0]
            + "\n You need to install statsmodels to use the FFT based correlation function.",
        )
        raise

    # Create np copies of input arguments.
    A_n = np.array(A_n)

    # Get the length of the timeseries.
    N = A_n.size

    # The "ubiased" kwarg deprecated in favor of "adjusted"
    C_t = sm.tsa.stattools.acf(A_n, fft=True, adjusted=True, nlags=N)
    t_grid = np.arange(N).astype("float")
    g_t = 2.0 * C_t * (1.0 - t_grid / float(N))

    try:
        ind = np.where((C_t <= 0) & (t_grid > mintime))[0][0]
    except IndexError:
        ind = N

    g = 1.0 + g_t[1:ind].sum()
    g = max(1.0, g)

    return g  # , g_t, C_t


def detect_equilibration_binary_search(A_t, bs_nodes=10):
    """Automatically detect equilibrated region of a dataset using a heuristic that maximizes number of effectively uncorrelated samples.

    Parameters
    ----------
    A_t : np.ndarray
        timeseries

    bs_nodes : int > 4
        number of geometrically distributed binary search nodes

    Returns
    -------
    t : int
        start of equilibrated data
    g : float
        statistical inefficiency of equilibrated data
    Neff_max : float
        number of uncorrelated samples

    Notes
    -----
    Finds the discard region (t) by a binary search on the range of
    possible lagtime values, with logarithmic spacings.  This will give
    a local maximum.  The global maximum is not guaranteed, but will
    likely be found if the N_eff[t] varies smoothly.

    """
    assert bs_nodes > 4, "Number of nodes for binary search must be > 4"
    T = A_t.size

    # Special case if timeseries is constant.
    if A_t.std() == 0.0:
        return 0, 1, T

    start = 1
    end = T - 1
    n_grid = min(bs_nodes, T)

    while True:
        time_grid = np.unique(
            (10 ** np.linspace(np.log10(start), np.log10(end), n_grid)).round().astype("int")
        )
        g_t = np.ones(time_grid.size)
        Neff_t = np.ones(time_grid.size)

        for k, t in enumerate(time_grid):
            if t < T - 1:
                g_t[k] = statistical_inefficiency_fft(A_t[t:])
                Neff_t[k] = (T - t + 1) / g_t[k]

        Neff_max = Neff_t.max()
        k = Neff_t.argmax()
        t = time_grid[k]
        g = g_t[k]

        if end - start < 4:
            break

        if k == 0:
            start = time_grid[0]
            end = time_grid[1]
        elif k == time_grid.size - 1:
            start = time_grid[-2]
            end = time_grid[-1]
        else:
            start = time_grid[k - 1]
            end = time_grid[k + 1]

    return t, g, Neff_max
