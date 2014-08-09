"""Test MBAR by performing statistical tests on a set of model systems
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK
N_k = np.array([50, 60, 70])

def generate_ho():    
    O_k = np.array([1.0, 2.0, 3.0])
    k_k = np.array([1.0, 1.5, 2.0])
    return "Harmonic Oscillators", harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)

def generate_exp():
    rates = np.array([1.0, 2.0, 3.0])  # Rates, e.g. Lambda
    return "Exponentials", exponential_distributions.ExponentialTestCase(rates)

system_generators = [generate_ho, generate_exp]

def test_analytical():
    """Generate test objects and calculate analytical results."""
    for system_generator in system_generators:
        name, test = system_generator()
        mu = test.analytical_means()
        variance = test.analytical_variances()
        f_k = test.analytical_free_energies()

def test_sample():
    """Draw samples via test object."""
    
    for system_generator in system_generators:
        name, test = system_generator()
        print(name)

        x_n, u_kn, N_k = test.sample([5, 6, 7], mode='u_kn')
        x_n, u_kn, N_k = test.sample([5, 5, 5], mode='u_kn')
        x_n, u_kn, N_k = test.sample([1, 1, 1], mode='u_kn')

        x_kn, u_kln, N_k = test.sample([5, 6, 7], mode='u_kln')
        x_kn, u_kln, N_k = test.sample([5, 5, 5], mode='u_kln')
        x_kn, u_kln, N_k = test.sample([1, 1, 1], mode='u_kln')

def test_mbar_freee_energies():
    """Can MBAR calculate correct free energy differences?"""    
    for system_generator in system_generators:
        name, test = system_generator()
        print(name)
        x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')

        eq(N_k, N_k_output)

        mbar = MBAR(u_kn, N_k)
        fe, fe_sigma = mbar.getFreeEnergyDifferences()
        fe, fe_sigma = fe[0,1:], fe_sigma[0,1:]

        fe0 = test.analytical_free_energies()
        fe0 = fe0[1:] - fe0[0]

        z = (fe - fe0) / fe_sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_expectation_xkn():
    """Can MBAR calculate E(x_n)??"""
    for system_generator in system_generators:
        name, test = system_generator()
        print(name) 
        x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')

        eq(N_k, N_k_output)

        mbar = MBAR(u_kn, N_k)

        mu, sigma = mbar.computeExpectations(x_n)
        mu0 = test.analytical_means()

        z = (mu0 - mu) / sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_harmonic_oscillators_mbar_xkn_squared():
    """Can MBAR calculate E(x_n^2)??"""
    for system_generator in system_generators:
        name, test = system_generator()    
        print(name)        
        x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')

        eq(N_k, N_k_output)

        mbar = MBAR(u_kn, N_k)

        mu, sigma = mbar.computeExpectations(x_n ** 2)
        mu0 = test.analytical_x_squared()

        z = (mu0 - mu) / sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_general_expectations():
    """Can MBAR calculate general expectations."""
    for system_generator in system_generators:
        name, test = system_generator()    
        print(name)        
        x_kn, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
        mbar = MBAR(u_kn, N_k)
        A_in = np.array([x_kn, x_kn ** 2, x_kn ** 3])
        u_n = u_kn[:2, :]
        state_list = np.array([[0, 0], [1, 0], [2, 0], [2, 1]],int)
        [A_i, d2A_ij] = mbar.computeGeneralExpectations(A_in, u_n, state_list)
