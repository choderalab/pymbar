"""Test BAR by performing statistical tests on a set of model systems
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import BAR, BARoverlap, MBAR
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq

precision = 5 # the precision for systems that do have analytical results that should be matched.
z_scale_factor = 12.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK
#0.5 is rounded to 1, so this says they must be within 3.0 sigma
N_k = np.array([1000, 500])

def generate_ho(O_k = np.array([1.0, 2.0]), K_k = np.array([0.5, 1.0])):
    return "Harmonic Oscillators", harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, K_k)

def generate_exp(rates = np.array([1.0, 2.0])): # Rates, e.g. Lambda
    return "Exponentials", exponential_distributions.ExponentialTestCase(rates)

def convert_to_differences(x_ij,dx_ij,xa):
    xa_ij = np.array(np.matrix(xa) - np.matrix(xa).transpose())

    # add ones to the diagonal of the uncertainties, because they are zero
    for i in range(len(N_k)):
        dx_ij[i,i] += 1
    z = (x_ij - xa_ij) / dx_ij
    for i in range(len(N_k)):
        z[i,i] = x_ij[i,i]-xa_ij[i,i]  # these terms should be zero; so we only throw an error if they aren't
    return z

system_generators = [generate_ho, generate_exp]
observables = ['position', 'position^2', 'RMS deviation', 'potential energy']

def test_bar_free_energies():

    """Can BAR calculate moderately correct free energy differences?"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)

        i = 0 ; j = 1 ; i_indices = np.arange(0, N_k[0]) ; j_indices = np.arange(N_k[0], N_k[0]+N_k[1])
        w_f = u_kn[j,i_indices] - u_kn[i,i_indices]
        w_r = u_kn[i,j_indices] - u_kn[j,j_indices]
        fe, fe_sigma = BAR(w_f, w_r)

        # Compare with analytical
        fe0 = test.analytical_free_energies()
        fe0 = fe0[1] - fe0[0]

        z = (fe - fe0) / fe_sigma
        eq(z / z_scale_factor, 0*z, decimal=0)

        # Compare with MBAR
        mbar = MBAR(u_kn, N_k)
        results = mbar.getFreeEnergyDifferences(return_dict=True)
        fe_t, dfe_t = mbar.getFreeEnergyDifferences(return_dict=False)

        eq(fe, fe_t[0,1])
        # absolute(fe_sigma - dfe_t[0,1]) <= (rtol * absolute(dfe_t[0,1]))
        np.testing.assert_allclose(fe_sigma, dfe_t[0,1], rtol=0.01)


def test_bar_computeOverlap():

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)

        i = 0 ; j = 1 ; i_indices = np.arange(0, N_k[0]) ; j_indices = np.arange(N_k[0], N_k[0]+N_k[1])
        w_f = u_kn[j,i_indices] - u_kn[i,i_indices]
        w_r = u_kn[i,j_indices] - u_kn[j,j_indices]

        # Compute overlap
        bar_overlap = BARoverlap(w_f, w_r)

        # Compare with MBAR
        mbar = MBAR(u_kn, N_k)
        mbar_overlap = mbar.computeOverlap()['scalar']

        eq(bar_overlap, mbar_overlap)
