"""Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators.
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators
from pymbar.utils import ensure_type, convert_ukn_to_uijn, convert_xn_to_x_kn
from pymbar.utils_for_testing import eq
from pymbar.old.mbar import MBAR as MBAR1  # Import mbar 1.0 for some reference calculations

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK

O_k = np.array([1.0, 2.0, 3.0])
k_k = np.array([1.0, 1.5, 2.0])
beta_k = np.array([1.0, 1.1, 1.2])

N_k = np.array([50, 60, 70])
N_k_even = 50 * np.ones(3)
        
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
        
def test_harmonic_oscillators_mbar_free_energies():
    """Harmonic Oscillators Test: can MBAR calculate correct free energy differences?"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)
    fe, fe_sigma = mbar.get_free_energy_differences()
    fe, fe_sigma = fe[0], fe_sigma[0]

    fe0 = test.analytical_free_energies()

    z = (fe - fe0) / fe_sigma
    z = z[1:]  # First component is undetermined.
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar_xkn():
    """Harmonic Oscillators Test: can MBAR calculate E(x_kn)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)

    mu, sigma = mbar.compute_expectation(x_n.values)
    mu0 = test.analytical_means()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar_xkn_squared():
    """Harmonic Oscillators Test: can MBAR calculate E(x_kn^2)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)
    
    mu, sigma = mbar.compute_expectation(x_n.values ** 2.0)
    mu0 = test.analytical_x_squared()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_unnormalized_log_weights_against_mbar1():
    """Harmonic Oscillators Test: Compare unnormalized log weights against pymbar 1.0 results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn.values, N_k_even)
    w = mbar._compute_unnormalized_log_weights()
    w_kn = np.array(np.split(w, mbar.n_states))

    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)    
    mbar1 = MBAR1(u_ijn.values, N_k_even)
    bias = np.zeros((mbar.n_states, N_k_even[0]))  # pymbar1.0 requires a "biasing" potential as input.  
    w1_kn = mbar1._computeUnnormalizedLogWeights(bias)
    
    eq(w_kn, w1_kn)


def test_expectation_against_mbar1():
    """Harmonic Oscillators Test: Compare expectations and uncertainties against pymbar 1.0 results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k_even)

    mbar = MBAR(u_kn.values, N_k_even)
    mu, sigma, theta = mbar.compute_expectation(x_n.values, return_theta=True)
    
    u_ijn, N_k_output = convert_ukn_to_uijn(u_kn)    
    mbar1 = MBAR1(u_ijn.values, N_k_even)
    
    x_kn = convert_xn_to_x_kn(x_n)

    x_kn = x_kn.values  # Convert to numpy for MBAR
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0
        
    mu1, sigma1, theta1= mbar1.computeExpectations(x_kn, return_theta=True)
    
    eq(mu1, mu)
    eq(sigma1, sigma)
    eq(theta1, theta)

