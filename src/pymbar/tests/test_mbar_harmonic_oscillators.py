"""Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators, for which
the true free energy differences can be computed analytically.

A number of replications of an experiment in which i.i.d. samples are drawn from a set of
K harmonic oscillators are produced.  For each replicate, we estimate the dimensionless free
energy differences and mean-square displacements (an observable), as well as their uncertainties.

For a 1D harmonic oscillator, the potential is given by
V(x;K) = (K/2) * (x-x_0)**2
where K denotes the spring constant.

The equilibrium distribution is given analytically by
p(x;beta,K) = sqrt[(beta K) / (2 pi)] exp[-beta K (x-x_0)**2 / 2]
The dimensionless free energy is therefore
f(beta,K) = - (1/2) * ln[ (2 pi) / (beta K) ]
"""

DECIMAL_PLACES = 1  # Controls the # of decimal places for test success.  Adjust as necessary.

import numpy as np
from pymbar import testsystems, MBAR
import pymbar.testsystems.harmonic_oscillator_reference as ref
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq

seed = None
        
def test_01_analytical_harmonic_oscillator():
    """Test that we can generate analytical data."""
    analytical = ref.AnalyticalHarmonicOscillator(ref.beta, ref.K_k, ref.O_k)
        
def test_02_harmonic_oscillators_example():
    """Test that we can generate example data."""
    x_kn, u_kln, N_k = testsystems.harmonic_oscillators_example(ref.N_k, ref.O_k, ref.K_k * ref.beta, seed=seed)
    eq(N_k, ref.N_k)

def test_03_MBAR_setup():
    """Test that we can construct MBAR object."""
    x_kn, u_kln, N_k = testsystems.harmonic_oscillators_example(ref.N_k, ref.O_k, ref.K_k * ref.beta, seed=seed)
    mbar = MBAR(u_kln, N_k, method = 'adaptive', relative_tolerance=1.0e-10, verbose=False)

class TestHarmonicOscillator(object):
    def setup(self):
        self.beta, self.K_k, self.O_k, self.N_k = ref.beta, ref.K_k, ref.O_k, ref.N_k

        self.analytical = ref.AnalyticalHarmonicOscillator(self.beta, self.K_k, self.O_k)
        self.x_kn, self.u_kln, self.N_k = testsystems.harmonic_oscillators_example(self.N_k, self.O_k, self.K_k * self.beta, seed=seed)
        
        self.K = len(self.K_k)
        self.N_max = self.N_k.max()
        
        self.mbar = MBAR(self.u_kln, self.N_k, method = 'adaptive', relative_tolerance=1.0e-10, verbose=False)
        
    def test_FE_difference(self):
        """Test FE differences between states."""
        (Delta_f_ij_estimated, dDelta_f_ij_estimated) = self.mbar.getFreeEnergyDifferences()
        FE = Delta_f_ij_estimated[0] - Delta_f_ij_estimated[0,0]
        FE0 = self.analytical.f_k - self.analytical.f_k[0]
        eq(FE, FE0, decimal=DECIMAL_PLACES)

    def test_displacement(self):
        A_kn = np.zeros([self.K, self.K, self.N_max], dtype = np.float64)
        for k in xrange(0, self.K):
            for l in xrange(0, self.K):
                A_kn[k, l, 0:self.N_k[k]] = (self.x_kn[k, 0:self.N_k[k]] - self.O_k[l])

        (A_k_estimated, dA_k_estimated) = self.mbar.computeExpectations(A_kn)

        eq(A_k_estimated, self.analytical.displacement, decimal=DECIMAL_PLACES)
        
    def test_displacement_squared(self):
        A_kn = np.zeros([self.K, self.K, self.N_max], dtype = np.float64)
        for k in xrange(0, self.K):
            for l in xrange(0, self.K):
                A_kn[k,l,0:self.N_k[k]] = (self.x_kn[k,0:self.N_k[k]] - self.O_k[l]) ** 2.

        (A_k_estimated, dA_k_estimated) = self.mbar.computeExpectations(A_kn)
        eq(A_k_estimated, self.analytical.displacement_squared, decimal=DECIMAL_PLACES)
                
    def test_position(self):
        A_kn = np.zeros([self.K, self.N_max], dtype = np.float64)
        for k in xrange(0, self.K):
            A_kn[k, 0:self.N_k[k]] = self.x_kn[k, 0:self.N_k[k]]

        (A_k_estimated, dA_k_estimated) = self.mbar.computeExpectations(A_kn)
        eq(A_k_estimated, self.analytical.position, decimal=DECIMAL_PLACES)
    
    def test_position_squared(self):
        A_kn = np.zeros([self.K, self.N_max], dtype = np.float64)
        for k in xrange(0, self.K):
            A_kn[k, 0:self.N_k[k]] = self.x_kn[k, 0:self.N_k[k]] ** 2. 

        (A_k_estimated, dA_k_estimated) = self.mbar.computeExpectations(A_kn)
        eq(A_k_estimated, self.analytical.position_squared, decimal=DECIMAL_PLACES)
