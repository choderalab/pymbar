"""Test MBAR by performing statistical tests on a set of of 1D exponential
distributions, the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
import pymbar.mbar
from pymbar.testsystems import exponential_distributions
from pymbar.utils import ensure_type, convert_ukn_to_uijn, convert_xn_to_x_kn
from pymbar.utils_for_testing import eq
from pymbar.old.mbar import MBAR as MBAR1  # Import mbar 1.0 for some reference calculations

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK

rates = np.array([1.0, 2.0, 3.0])  # Rates, e.g. Lambda
N_k = np.array([50, 60, 70]) * 500
N_k_even = 50 * np.ones(3)
        
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
        
def test_exponential_mbar_free_energies():
    """Exponential Distribution Test: can MBAR calculate correct free energy differences?"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)
    fe, fe_sigma = mbar.get_free_energy_differences()
    fe, fe_sigma = fe[0], fe_sigma[0]

    fe0 = test.analytical_free_energies()

    eq(fe, fe0, decimal=1)

def test_exponential_mbar_xkn():
    """Exponential Distribution Test: can MBAR calculate E(x_kn)?"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)

    mu, sigma, theta = mbar.compute_expectation(x_n.values)
    mu0 = test.analytical_means()

    z = (mu0 - mu) / sigma    
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_exponential_mbar_xkn_squared():
    """Exponential Distribution Test: can MBAR calculate E(x_kn^2)"""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k)
    
    mbar = MBAR(u_kn.values, N_k)

    mu, sigma, theta = mbar.compute_expectation(x_n.values ** 2.0)
    mu0 = test.analytical_x_squared()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_unnormalized_log_weights_against_mbar1():
    """Exponential Distribution Test: Compare unnormalized log weights against pymbar 1.0 results."""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn.values, N_k_even)
    w = pymbar.mbar.compute_unnormalized_log_weights(mbar.u_kn, mbar.N_k, mbar.f_k)
    w_kn = np.array(np.split(w, mbar.n_states))  # THIS IS A HACK that assumes specific ordering, fix later!

    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)    
    mbar1 = MBAR1(u_ijn.values, N_k_even)
    bias = np.zeros((mbar.n_states, N_k_even[0]))  # pymbar1.0 requires a "biasing" potential as input.  
    w1_kn = mbar1._computeUnnormalizedLogWeights(bias)

    x_n[:] = w[:]  # Using x_n as a pandas template for converting shape.
    w_kn_converted = convert_xn_to_x_kn(x_n)
        
    eq(w_kn, w1_kn)  # Test manual conversion by np.split
    eq(w_kn_converted.values, w1_kn)  # Test the conversion function that I wrote


def test_expectation_against_mbar1():
    """Exponential Distribution Test: Compare expectations and uncertainties against pymbar 1.0 results."""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn.values, N_k_even)
    mu, sigma, theta = mbar.compute_expectation(x_n.values)
    
    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)    
    mbar1 = MBAR1(u_ijn.values, N_k_even)
    
    x_kn = convert_xn_to_x_kn(x_n)

    x_kn = x_kn.values  # Convert to numpy for MBAR
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0
        
    mu1, sigma1, theta1 = mbar1.computeExpectations(x_kn, return_theta=True)
    
    eq(mu1, mu)
    eq(sigma1, sigma)
    eq(theta1, theta)


def test_expectation_against_mbar1():
    """Exponential Distribution Test: Compare expectation differences and uncertainties against pymbar 1.0 results."""
    test = exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn.values, N_k_even)
    mu, sigma, theta = mbar.compute_expectation_difference(x_n.values)
    
    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)    
    mbar1 = MBAR1(u_ijn.values, N_k_even)
    
    x_kn = convert_xn_to_x_kn(x_n)

    x_kn = x_kn.values  # Convert to numpy for MBAR
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0
        
    mu1, sigma1, theta1 = mbar1.computeExpectations(x_kn, return_theta=True, output="differences")
    
    eq(mu1, mu)
    eq(sigma1, sigma)
    eq(theta1, theta)
