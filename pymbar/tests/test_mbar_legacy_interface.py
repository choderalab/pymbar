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

O_k = np.array([1.0, 2.0, 3.0])
k_k = np.array([1.0, 1.5, 2.0])
beta_k = np.array([1.0, 1.1, 1.2])

N_k = np.array([50, 60, 70]) * 500

def test_mbar_2D_input():
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)
    weights = mbar.getWeights()

    u_ijn = convert_ukn_to_uijn(u_kn, N_k)    
    mbar1 = MBAR1(u_ijn, N_k)
    weights1 = mbar1.getWeights()
    
    mbar2 = MBAR(u_ijn, N_k)
    weights2 = mbar2.getWeights()
        
    eq(weights, weights1)
    eq(weights, weights2)
    eq(weights1, weights2)


def test_getWeights():
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)
    weights = mbar.getWeights()

    u_ijn = convert_ukn_to_uijn(u_kn, N_k)    
    mbar1 = MBAR1(u_ijn, N_k)
    weights1 = mbar1.getWeights()
    
    eq(weights, weights1)

def test_getFreeEnergyDifferences():
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)

    mbar = MBAR(u_kn, N_k)
    dFE, covFE = mbar.getFreeEnergyDifferences()

    u_ijn = convert_ukn_to_uijn(u_kn, N_k)    
    mbar1 = MBAR1(u_ijn, N_k)
    dFE1, covFE1 = mbar1.getFreeEnergyDifferences()
    
    eq(dFE, dFE1)

def test_computeExpectations_2D_averages():
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)
    
    x_kn = convert_An_to_Akn(x_n, N_k)

    mbar = MBAR(u_kn, N_k)
    mu, sigma, theta = mbar.computeExpectations(x_kn)
    
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)
    mbar1 = MBAR1(u_ijn, N_k)
        
    mu1, sigma1, theta1= mbar1.computeExpectations(x_kn, return_theta=True)

    eq(mu, mu1)
    eq(sigma, sigma1)
    eq(theta, theta1)

def test_computeExpectations_2D_differences():
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k, beta_k)
    x_n, u_kn, origin = test.sample(N_k)
    
    x_kn = convert_An_to_Akn(x_n, N_k)

    mbar = MBAR(u_kn, N_k)
    mu, sigma, theta = mbar.computeExpectations(x_kn, output="differences")  # The input is 2D
    
    u_ijn = convert_ukn_to_uijn(u_kn, N_k)    
    mbar1 = MBAR1(u_ijn, N_k)
        
    mu1, sigma1, theta1= mbar1.computeExpectations(x_kn, return_theta=True, output="differences")

    eq(mu, mu1)
    eq(sigma, sigma1)
    eq(theta, theta1)
