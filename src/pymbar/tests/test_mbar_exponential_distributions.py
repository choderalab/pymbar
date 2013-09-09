"""Test MBAR by performing statistical tests on a set of of 1D exponential
distributions, the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
from pymbar.testsystems import exponential_distributions
from pymbar.utils import ensure_type, convert_ukn_to_uijn, convert_xn_to_x_kn
from pymbar.utils_for_testing import eq

seed = None

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
    x_n, u_kn, origin = test.sample(np.array([5, 6, 7]))
    x_n, u_kn, origin = test.sample(np.array([5, 5, 5]))
    x_n, u_kn, origin = test.sample(np.array([1., 1, 1.]))
    
def test_shape_conversion_exponential():
    """Exponential Distribution Test: convert shape from u_kn to u_ijn."""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(np.array([5, 6, 7]))
    u_ijn, N_k = convert_ukn_to_uijn(u_kn)
    x_n, u_kn, origin = test.sample(np.array([5, 5, 5]))
    u_ijn, N_k = convert_ukn_to_uijn(u_kn)
    x_n, u_kn, origin = test.sample(np.array([1., 1, 1.]))
    u_ijn, N_k = convert_ukn_to_uijn(u_kn)
    
def test_exponential_mbar_free_energies():
    """Exponential Distribution Test: can MBAR calculate correct free energy differences?"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)
    
    eq(N_k, N_k_output.values)

    mbar = MBAR(u_ijn.values, N_k)
    fe, fe_sigma = mbar.getFreeEnergyDifferences()
    fe, fe_sigma = fe[0], fe_sigma[0]

    fe0 = test.analytical_free_energies()

    z = (fe - fe0) / fe_sigma
    z = z[1:]  # First component is undetermined.
    eq(z / 3.0, np.zeros(len(z)), decimal=0)


def test_exponential_mbar_xkn():
    """Exponential Distribution Test: can MBAR calculate E(x_kn)?"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)
    
    eq(N_k, N_k_output.values)

    mbar = MBAR(u_ijn.values, N_k)
    
    x_kn = convert_xn_to_x_kn(x_n)

    x_kn = x_kn.values  # Convert to numpy for MBAR
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0

    mu, sigma = mbar.computeExpectations(x_kn)
    mu0 = test.analytical_means()
    
    z = (mu0 - mu) / sigma
    eq(z / 3.0, np.zeros(len(z)), decimal=0)


def test_exponential_mbar_xkn_squared():
    """Exponential Distribution Test: can MBAR calculate E(x_kn^2)"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)
    
    eq(N_k, N_k_output.values)

    mbar = MBAR(u_ijn.values, N_k)
    
    x_kn = convert_xn_to_x_kn(x_n) ** 2.0

    x_kn = x_kn.values  # Convert to numpy for MBAR
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0

    mu, sigma = mbar.computeExpectations(x_kn)
    mu0 = test.analytical_x_squared()
    
    z = (mu0 - mu) / sigma
    eq(z / 3.0, np.zeros(len(z)), decimal=0)
