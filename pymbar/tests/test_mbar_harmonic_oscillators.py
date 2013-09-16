"""Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators.
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq
#from pymbar.old import MBAR as MBAR1  # Import mbar 1.0 for some reference calculations

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK

O_k = np.array([1.0, 2.0, 3.0])
k_k = np.array([1.0, 1.5, 2.0])
N_k = np.array([50, 60, 70])
        
def test_analytical_harmonic_oscillators():
    """Harmonic Oscillators Test: generate test object and calculate analytical results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    mu = test.analytical_means()
    variance = test.analytical_variances()
    f_k = test.analytical_free_energies()

def test_harmonic_oscillators_samples():
    """Harmonic Oscillators Test:  draw samples via test object."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, origin = test.sample(np.array([5, 6, 7]))
    x_n, u_kn, origin = test.sample(np.array([5, 5, 5]))
    x_n, u_kn, origin = test.sample(np.array([1., 1, 1.]))
        
def test_harmonic_oscillators_mbar_free_energies():
    """Harmonic Oscillators Test: can MBAR calculate correct free energy differences?"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)
    fe, fe_sigma = mbar.get_free_energy_differences()
    fe, fe_sigma = fe[0], fe_sigma[0]

    fe0 = test.analytical_free_energies()

    z = (fe - fe0) / fe_sigma
    z = z[1:]  # First component is undetermined.
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar__xkn():
    """Harmonic Oscillators Test: can MBAR calculate E(x_kn)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)

    mu, sigma = mbar.compute_expectation(x_n.values)
    mu0 = test.analytical_means()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar_xkn_squared():
    """Harmonic Oscillators Test: can MBAR calculate E(x_kn^2)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn.values, N_k)
    
    mu, sigma = mbar.compute_expectation(x_n.values ** 2.0)
    mu0 = test.analytical_x_squared()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)
