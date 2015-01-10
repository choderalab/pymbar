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

* EXP - unidirectional estimator for free energy differences based on Zwanzig relation / exponential averaging
"""

#=============================================================================================
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
import numpy as np
from pymbar.utils import logsumexp

#=============================================================================================
# One-sided exponential averaging (EXP).
#=============================================================================================

def EXP(w_F, compute_uncertainty=True, is_timeseries=False):
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
    DeltaF : float
        DeltaF is the free energy difference between the two states.
    dDeltaF : float
        dDeltaF is the uncertainty, and is only returned if compute_uncertainty is set to True

    Notes
    -----
    If you are prodividing correlated timeseries data, be sure to set the 'timeseries' flag to True

    Examples
    --------

    Compute the free energy difference given a sample of forward work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> [DeltaF, dDeltaF] = EXP(w_F)
    >>> print('Forward free energy difference is %.3f +- %.3f kT' % (DeltaF, dDeltaF))
    Forward free energy difference is 1.088 +- 0.076 kT
    >>> [DeltaF, dDeltaF] = EXP(w_R)
    >>> print('Reverse free energy difference is %.3f +- %.3f kT' % (DeltaF, dDeltaF))
    Reverse free energy difference is -1.073 +- 0.082 kT

    """

    # Get number of work measurements.
    T = float(np.size(w_F))  # number of work measurements

    # Estimate free energy difference by exponential averaging using DeltaF = - log < exp(-w_F) >
    DeltaF = - (logsumexp(- w_F) - np.log(T))

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
            import timeseries
            g = timeseries.statisticalInefficiency(x, x)

        # Estimate standard error of E[x].
        dx = np.std(x) / np.sqrt(T / g)

        # dDeltaF = <x>^-1 dx
        dDeltaF = (dx / Ex)

        # Return estimate of free energy difference and uncertainty.
        return (DeltaF, dDeltaF)
    else:
        return DeltaF

#=============================================================================================
# Gaussian approximation to exponential averaging (Gauss).
#=============================================================================================

def EXPGauss(w_F, compute_uncertainty=True, is_timeseries=False):
    """Estimate free energy difference using gaussian approximation to one-sided (unidirectional) exponential averaging.

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
    DeltaF : float
        DeltaF is the free energy difference between the two states.
    dDeltaF : float
        dDeltaF is the uncertainty, and is only returned if compute_uncertainty is set to True

    Notes
    -----
    If you are prodividing correlated timeseries data, be sure to set the 'timeseries' flag to True

    Examples
    --------
    Compute the free energy difference given a sample of forward work values.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> [DeltaF, dDeltaF] = EXPGauss(w_F)
    >>> print('Forward Gaussian approximated free energy difference is %.3f +- %.3f kT' % (DeltaF, dDeltaF))
    Forward Gaussian approximated free energy difference is 1.049 +- 0.089 kT
    >>> [DeltaF, dDeltaF] = EXPGauss(w_R)
    >>> print('Reverse Gaussian approximated free energy difference is %.3f +- %.3f kT' % (DeltaF, dDeltaF))
    Reverse Gaussian approximated free energy difference is -1.073 +- 0.080 kT

    """

    # Get number of work measurements.
    T = float(np.size(w_F))  # number of work measurements

    var = np.var(w_F)
    # Estimate free energy difference by Gaussian approximation, dG = <U> - 0.5*var(U)
    DeltaF = np.average(w_F) - 0.5 * var

    if compute_uncertainty:
        # Compute effective number of uncorrelated samples.
        g = 1.0  # statistical inefficiency
        T_eff = T
        if is_timeseries:
            # Estimate statistical inefficiency of x timeseries.
            import timeseries
            g = timeseries.statisticalInefficiency(w_F, w_F)

            T_eff = T / g
        # Estimate standard error of E[x].
        dx2 = var / T_eff + 0.5 * var * var / (T_eff - 1)
        dDeltaF = np.sqrt(dx2)

        # Return estimate of free energy difference and uncertainty.
        return (DeltaF, dDeltaF)
    else:
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

def computeEXP(*args, **kwargs):
    return EXP(*args, **kwargs)
computeEXP.__doc__ = EXP.__doc__ + deprecation_warning

def computeEXPGauss(*args, **kwargs):
    return EXPGauss(*args, **kwargs)
computeEXPGauss.__doc__ = EXPGauss.__doc__ + deprecation_warning

def _compatibilityDoctests():
    """
    Backwards-compatibility doctests.

    >>> from pymbar import testsystems
    >>> [w_F, w_R] = testsystems.gaussian_work_example(mu_F=None, DeltaF=1.0, seed=0)
    >>> [DeltaF, dDeltaF] = computeEXP(w_F)
    >>> [DeltaF, dDeltaF] = computeEXPGauss(w_F)
    """
    pass
