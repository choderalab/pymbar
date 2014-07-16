"""Test MBAR by performing statistical tests on a set of of 1D exponential
distributions, the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
from pymbar.testsystems import exponential_distributions
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK

rates = np.array([1.0, 2.0, 3.0])  # Rates, e.g. Lambda
N_k = np.array([50, 60, 70])

def test_analytical_exponential():
    """Exponential Distribution Test: generate test object and calculate analytical results."""
    test = exponential_distributions.ExponentialTestCase(rates)
    mu = test.analytical_means()
    variance = test.analytical_variances()
    f_k = test.analytical_free_energies()

def test_exponential_samples():
    """Exponential Distribution Test:  draw samples via test object."""
    test = exponential_distributions.ExponentialTestCase(rates)

    x_n, u_kn, N_k = test.sample([5, 6, 7], mode='u_kn')
    x_n, u_kn, N_k = test.sample([5, 5, 5], mode='u_kn')
    x_n, u_kn, N_k = test.sample([1, 1, 1], mode='u_kn')

    x_kn, u_kln, N_k = test.sample([5, 6, 7], mode='u_kln')
    x_kn, u_kln, N_k = test.sample([5, 5, 5], mode='u_kln')
    x_kn, u_kln, N_k = test.sample([1, 1, 1], mode='u_kln')

def test_exponential_mbar_free_energies():
    """Exponential Distribution Test: can MBAR calculate correct free energy differences?"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    eq(N_k, N_k_output)

    mbar = MBAR(u_kn, N_k)
    fe, fe_sigma = mbar.getFreeEnergyDifferences()
    fe, fe_sigma = fe[0,1:], fe_sigma[0,1:]

    fe0 = test.analytical_free_energies()
    fe0 = fe0[1:] - fe0[0]

    z = (fe - fe0) / fe_sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_exponential_mbar_xkn():
    """Exponential Distribution Test: can MBAR calculate E(x_n)?"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')

    eq(N_k, N_k_output)

    mbar = MBAR(u_kn, N_k)

    mu, sigma = mbar.computeExpectations(x_n)
    mu0 = test.analytical_means()

    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_exponential_mbar_xkn_squared():
    """Exponential Distribution Test: can MBAR calculate E(x_n^2)"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')

    eq(N_k, N_k_output)

    mbar = MBAR(u_kn, N_k)

    mu, sigma = mbar.computeExpectations(x_n ** 2)
    mu0 = test.analytical_x_squared()

    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)
