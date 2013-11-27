"""Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators.
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
import pymbar.mbar
from pymbar.testsystems import harmonic_oscillators
from pymbar.utils import ensure_type, convert_ukn_to_uijn, convert_An_to_Akn
from pymbar.utils_for_testing import eq
from pymbar.old.mbar import MBAR as MBAR1  # Import mbar 1.0 for some reference calculations

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK

O_k = np.array([1.0, 2.0, 3.0])
k_k = np.array([1.0, 1.5, 2.0])
beta_k = np.array([1.0, 1.1, 1.2])

N_k = np.array([50, 60, 70]) * 500
N_k_even = 50 * np.ones(3)

num_samples_per_state = 100
num_states = 20

O_k_random = np.random.normal(size=num_states)
k_k_random = np.random.uniform(1.0, 2.0, size=num_states)
beta_k_random = np.random.uniform(1.0, 2.0, size=num_states)
N_k_random = np.ones(num_states) * num_samples_per_state

        
def test_analytical_harmonic_oscillators():
    """Harmonic Oscillators Test: generate test object and calculate analytical results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    mu = test.analytical_means()
    variance = test.analytical_variances()
    f_k = test.analytical_free_energies()

def test_harmonic_oscillators_samples():
    """Harmonic Oscillators Test:  draw samples via test object."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(np.array([5, 6, 7]))
    x_n, u_kn, origin = test.sample(np.array([5, 5, 5]))
    x_n, u_kn, origin = test.sample(np.array([1., 1, 1.]))
        
def test_harmonic_oscillators_mbar_free_energies_empty_state():
    """Harmonic Oscillators Test: can MBAR calculate correct free energy differences with empty states?"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    
    N_k = np.array([140000, 0, 150000])
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)
    fe, fe_sigma = mbar.get_free_energy_differences()
    fe, fe_sigma = fe[0], fe_sigma[0]
    
    initial_f_k = 0.1 * np.ones(len(O_k))
    mbar = MBAR(u_kn, N_k, initial_f_k=initial_f_k)
    fe2, fe_sigma2 = mbar.get_free_energy_differences()
    fe2, fe_sigma2 = fe2[0], fe_sigma2[0]

    fe0 = test.analytical_free_energies()

    eq(fe, fe0, decimal=2)
    eq(fe2, fe0, decimal=2)
    eq(fe2, fe, decimal=2)

def test_harmonic_oscillators_mbar_free_energies():
    """Harmonic Oscillators Test: can MBAR calculate correct free energy differences?"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)
    fe, fe_sigma = mbar.get_free_energy_differences()
    fe, fe_sigma = fe[0], fe_sigma[0]
    
    initial_f_k = 0.1 * np.ones(len(O_k))
    mbar = MBAR(u_kn, N_k, initial_f_k=initial_f_k)
    fe2, fe_sigma2 = mbar.get_free_energy_differences()
    fe2, fe_sigma2 = fe2[0], fe_sigma2[0]

    fe0 = test.analytical_free_energies()

    eq(fe, fe0, decimal=1)
    eq(fe2, fe0, decimal=1)
    eq(fe2, fe, decimal=1)

def test_exponential_mbar_xkn():
    """Harmonic Oscillators Test: can MBAR calculate E(x_kn)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)

    mu, sigma, theta = mbar.compute_expectation(x_n)
    mu0 = test.analytical_means()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar_xkn_squared():
    """Harmonic Oscillators Test: can MBAR calculate E(x_kn^2)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)
    
    mu, sigma, theta = mbar.compute_expectation(x_n ** 2.0)
    mu0 = test.analytical_x_squared()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_unnormalized_log_weights_against_mbar1():
    """Harmonic Oscillators Test: Compare unnormalized log weights against pymbar 1.0 results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn, N_k_even)
    w = pymbar.mbar.compute_unnormalized_log_weights(mbar.u_kn, mbar.N_k, mbar.f_k)
    w_kn = np.array(np.split(w, mbar.n_states))

    u_ijn = convert_ukn_to_uijn(u_kn, N_k_even) 
    mbar1 = MBAR1(u_ijn, N_k_even)
    bias = np.zeros((mbar.n_states, N_k_even[0]))  # pymbar1.0 requires a "biasing" potential as input.  
    w1_kn = mbar1._computeUnnormalizedLogWeights(bias)
    
    x_n[:] = w[:]  # Using x_n as a pandas template for converting shape.
    w_kn_converted = convert_An_to_Akn(x_n, N_k_even)
    
    eq(w_kn, w1_kn)  # Test manual conversion by np.split
    eq(w_kn_converted, w1_kn)  # Test the conversion function that I wrote

def test_unnormalized_log_weights_against_mbar1_random():
    """Harmonic Oscillators Test: Compare unnormalized log weights against pymbar 1.0 results for %d states and %d samples per state with randomly selected parameters.""" % (num_states, num_samples_per_state)
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k_random, k_k_random, beta_k_random)
    x_n, u_kn, origin = test.sample(N_k_random)

    mbar = MBAR(u_kn, N_k_random)
    w = pymbar.mbar.compute_unnormalized_log_weights(mbar.u_kn, mbar.N_k, mbar.f_k)
    w_kn = np.array(np.split(w, mbar.n_states))  # THIS IS A HACK that assumes specific ordering, fix later!

    u_ijn = convert_ukn_to_uijn(u_kn, N_k_random)
    mbar1 = MBAR1(u_ijn, N_k_random)
    bias = np.zeros((mbar.n_states, N_k_random[0]))  # pymbar1.0 requires a "biasing" potential as input.  
    w1_kn = mbar1._computeUnnormalizedLogWeights(bias)
    
    x_n[:] = w[:]  # Using x_n as a pandas template for converting shape.
    w_kn_converted = convert_An_to_Akn(x_n, N_k_random)
    
    eq(w_kn, w1_kn, decimal=4)  # Test manual conversion by np.split
    eq(w_kn_converted, w1_kn, decimal=4)  # Test the conversion function that I wrote
    
def test_log_weights_against_mbar1_random():
    """Harmonic Oscillators Test: Compare log weights against pymbar 1.0 results for %d states and %d samples per state with randomly selected parameters.""" % (num_states, num_samples_per_state)
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k_random, k_k_random, beta_k_random)
    x_n, u_kn, origin = test.sample(N_k_random)

    mbar = MBAR(u_kn, N_k_random)
    weights, f_k = pymbar.mbar.compute_log_weights(mbar.f_k, mbar.N_k, mbar.u_kn)
    
    u_ijn = convert_ukn_to_uijn(u_kn, N_k_random)
    mbar1 = MBAR1(u_ijn, N_k_random)
    weights1 = mbar1.Log_W_nk 

    eq(weights1, weights)

def test_expectation_against_mbar1():
    """Harmonic Oscillators Test: Compare expectations and uncertainties against pymbar 1.0 results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn, N_k_even)
    mu, sigma, theta = mbar.compute_expectation(x_n)
    
    u_ijn = convert_ukn_to_uijn(u_kn, N_k_even) 
    mbar1 = MBAR1(u_ijn, N_k_even)
    
    x_kn = convert_An_to_Akn(x_n, N_k_even)
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0
        
    mu1, sigma1, theta1= mbar1.computeExpectations(x_kn, return_theta=True)
    
    eq(mu1, mu)
    eq(sigma1, sigma)
    eq(theta1, theta)

def test_expectation_difference_against_mbar1():
    """Harmonic Oscillators Test: Compare expectation difference and uncertainties against pymbar 1.0 results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn, N_k_even)
    mu, sigma, theta = mbar.compute_expectation_difference(x_n)
    
    u_ijn = convert_ukn_to_uijn(u_kn, N_k_even)
    mbar1 = MBAR1(u_ijn, N_k_even)
    
    x_kn = convert_An_to_Akn(x_n, N_k_even)
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0
        
    mu1, sigma1, theta1= mbar1.computeExpectations(x_kn, return_theta=True, output="differences")
    
    eq(mu1, mu)
    eq(sigma1, sigma)
    eq(theta1, theta)

