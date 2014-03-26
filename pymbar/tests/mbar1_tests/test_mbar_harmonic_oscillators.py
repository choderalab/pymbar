"""Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators.
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar.old.mbar import MBAR
from pymbar.testsystems import harmonic_oscillators
from pymbar.utils import ensure_type, convert_ukn_to_uijn, convert_An_to_Akn
from pymbar.utils_for_testing import eq

z_scale_factor = 3.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK

O_k = np.array([1.0, 2.0, 3.0])
k_k = np.array([1.0, 1.5, 2.0])
beta_k = np.array([1.0, 1.1, 1.2])

N_k = np.array([50, 60, 70])
        
def test_analytical_harmonic_oscillators():
    """Harmonic Oscillators Test (pymbar 1.0): generate test object and calculate analytical results."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    mu = test.analytical_means()
    variance = test.analytical_variances()
    f_k = test.analytical_free_energies()

def test_harmonic_oscillators_samples():
    """Harmonic Oscillators Test (pymbar 1.0):  draw samples via test object."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(np.array([5, 6, 7]))
    x_n, u_kn, origin = test.sample(np.array([5, 5, 5]))
    x_n, u_kn, origin = test.sample(np.array([1., 1, 1.]))
    
def test_shape_conversion_harmonic_oscillators():
    """Harmonic Oscillators Test (pymbar 1.0): convert shape from u_kn to u_ijn."""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    
    N_k = np.array([5, 6, 7])
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)
    
    N_k = np.array([5, 5, 5])
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)
    
    N_k = np.array([1., 1, 1.])
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)
    
def test_harmonic_oscillators_mbar_free_energies():
    """Harmonic Oscillators Test (pymbar 1.0): can MBAR calculate correct free energy differences?"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)

    mbar = MBAR(u_ijn, N_k)
    fe, fe_sigma = mbar.getFreeEnergyDifferences()
    fe, fe_sigma = fe[0], fe_sigma[0]

    fe0 = test.analytical_free_energies()

    z = (fe - fe0) / fe_sigma
    z = z[1:]  # First component is undetermined.
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar__xkn():
    """Harmonic Oscillators Test (pymbar 1.0): can MBAR calculate E(x_kn)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)

    mbar = MBAR(u_ijn, N_k)
    
    x_kn = convert_An_to_Akn(x_n, N_k)
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0

    mu, sigma = mbar.computeExpectations(x_kn)
    mu0 = test.analytical_means()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_exponential_mbar_xkn_squared():
    """Harmonic Oscillators Test (pymbar 1.0): can MBAR calculate E(x_kn^2)??"""
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)

    mbar = MBAR(u_ijn, N_k)
    
    x_kn = convert_An_to_Akn(x_n, N_k) ** 2.
    x_kn[np.isnan(x_kn)] = 0.0  # Convert nans to 0.0

    mu, sigma = mbar.computeExpectations(x_kn)
    mu0 = test.analytical_x_squared()
    
    z = (mu0 - mu) / sigma
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)
