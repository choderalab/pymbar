"""Test MBAR by performing statistical tests on a set of model systems
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq

precision = 8 # the precision for systems that do have analytical results that should be matched.
z_scale_factor = 12.0  # Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK
#0.5 is rounded to 1, so this says they must be within 3.0 sigma
N_k = np.array([1000, 500, 0, 800])

def generate_ho(O_k = np.array([1.0, 2.0, 3.0, 4.0]), K_k = np.array([0.5, 1.0, 1.5, 2.0])):
    return "Harmonic Oscillators", harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, K_k)

def generate_exp(rates = np.array([1.0, 2.0, 3.0, 4.0])): # Rates, e.g. Lambda
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

def test_analytical():
    """Generate test objects and calculate analytical results."""
    for system_generator in system_generators:
        name, test = system_generator()
        mu = test.analytical_means()
        variance = test.analytical_variances()
        f_k = test.analytical_free_energies()
        for observable in observables:
            A_k = test.analytical_observable(observable = observable)
        s_k = test.analytical_entropies()

def test_sample():
    """Draw samples via test object."""

    for system_generator in system_generators:
        name, test = system_generator()
        print(name)

        x_n, u_kn, N_k, s_n = test.sample([5, 6, 7, 8], mode='u_kn')
        x_n, u_kn, N_k, s_n = test.sample([5, 5, 5, 5], mode='u_kn')
        x_n, u_kn, N_k, s_n = test.sample([1, 1, 1, 1], mode='u_kn')
        x_n, u_kn, N_k, s_n = test.sample([10, 0, 8, 0], mode='u_kn')

        x_kn, u_kln, N_k = test.sample([5, 6, 7, 8], mode='u_kln')
        x_kn, u_kln, N_k = test.sample([5, 5, 5, 5], mode='u_kln')
        x_kn, u_kln, N_k = test.sample([1, 1, 1, 1], mode='u_kln')
        x_kn, u_kln, N_k = test.sample([10, 0, 8, 0], mode='u_kln')

def test_mbar_free_energies():

    """Can MBAR calculate moderately correct free energy differences?"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)

        fe, fe_sigma, Theta_ij = mbar.getFreeEnergyDifferences()
        fe, fe_sigma = fe[0,1:], fe_sigma[0,1:]

        fe0 = test.analytical_free_energies()
        fe0 = fe0[1:] - fe0[0]

        z = (fe - fe0) / fe_sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_computeExpectations_position_averages():

    """Can MBAR calculate E(x_n)??"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        mu, sigma = mbar.computeExpectations(x_n)
        mu0 = test.analytical_observable(observable = 'position')

        z = (mu0 - mu) / sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_computeExpectations_position_differences():

    """Can MBAR calculate E(x_n)??"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        mu_ij, sigma_ij = mbar.computeExpectations(x_n, output = 'differences')
        mu0 = test.analytical_observable(observable = 'position')
        z = convert_to_differences(mu_ij, sigma_ij, mu0)
        eq(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)

def test_mbar_computeExpectations_position2():

    """Can MBAR calculate E(x_n^2)??"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        mu, sigma = mbar.computeExpectations(x_n ** 2)
        mu0 = test.analytical_observable(observable = 'position^2')

        z = (mu0 - mu) / sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_computeExpectations_potential():

    """Can MBAR calculate E(u_kn)??"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        mu, sigma = mbar.computeExpectations(u_kn, state_dependent = True)
        mu0 = test.analytical_observable(observable = 'potential energy')
        print(mu)
        print(mu0)
        z = (mu0 - mu) / sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_computeMultipleExpectations():

    """Can MBAR calculate E(u_kn)??"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        A = np.zeros([2,len(x_n)])
        A[0,:] = x_n
        A[1,:] = x_n**2
        state = 1
        mu, sigma, covariances = mbar.computeMultipleExpectations(A,u_kn[state,:])
        mu0 = test.analytical_observable(observable = 'position')[state]
        mu1 = test.analytical_observable(observable = 'position^2')[state]
        z = (mu0 - mu[0]) / sigma[0]
        eq(z / z_scale_factor, 0*z, decimal=0)
        z = (mu1 - mu[1]) / sigma[1]
        eq(z / z_scale_factor, 0*z, decimal=0)

def test_mbar_computeEntropyAndEnthalpy():

    """Can MBAR calculate f_k, <u_k> and s_k ??"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        f_ij, df_ij, u_ij, du_ij, s_ij, ds_ij = mbar.computeEntropyAndEnthalpy(u_kn)

        fa = test.analytical_free_energies()
        ua = test.analytical_observable('potential energy')
        sa = test.analytical_entropies()

        fa_ij = np.array(np.matrix(fa) - np.matrix(fa).transpose())
        ua_ij = np.array(np.matrix(ua) - np.matrix(ua).transpose())
        sa_ij = np.array(np.matrix(sa) - np.matrix(sa).transpose())

        z = convert_to_differences(f_ij,df_ij,fa)
        eq(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)
        z = convert_to_differences(u_ij,du_ij,ua)
        eq(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)
        z = convert_to_differences(s_ij,ds_ij,sa)
        eq(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)

def test_mbar_computeEffectiveSampleNumber():
    """ testing computeEffectiveSampleNumber """

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)

        # one mathematical effective sample numbers should be between N_k and sum_k N_k
        N_eff = mbar.computeEffectiveSampleNumber()
        sumN = np.sum(N_k)
        assert all(N_eff > N_k)
        assert all(N_eff < sumN)

def test_mbar_computeOverlap():

    # tests with identical states, which gives analytical results.

    d = len(N_k)
    even_O_k = 2.0*np.ones(d)
    even_K_k = 0.5*np.ones(d)
    even_N_k = 100*np.ones(d)
    name, test = generate_ho(O_k = even_O_k, K_k = even_K_k)
    x_n, u_kn, N_k_output, s_n = test.sample(even_N_k, mode='u_kn')
    mbar = MBAR(u_kn, even_N_k)

    overlap_scalar, eigenval, O = mbar.computeOverlap()

    reference_matrix = np.matrix((1.0/d)*np.ones([d,d]))
    reference_eigenvalues = np.zeros(d)
    reference_eigenvalues[0] = 1.0
    reference_scalar = np.float64(1.0)

    eq(O, reference_matrix, decimal=precision)
    eq(eigenval, reference_eigenvalues, decimal=precision)
    eq(overlap_scalar, reference_scalar, decimal=precision)

    # test of more straightforward examples
    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        mbar = MBAR(u_kn, N_k)
        overlap_scalar, eigenval, O = mbar.computeOverlap()

        # rows of matrix should sum to one
        sumrows = np.array(np.sum(O,axis=1))
        eq(sumrows, np.ones(np.shape(sumrows)), decimal=precision)
        eq(eigenval[0], np.float64(1.0), decimal=precision)

def test_mbar_getWeights():

    """ testing getWeights """

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        mbar = MBAR(u_kn, N_k)
        # rows should be equal to zero
        W = mbar.getWeights()
        sumrows = np.sum(W,axis=0)
        eq(sumrows, np.ones(len(sumrows)), decimal=precision)

def test_mbar_computePerturbedFreeEnergeies():

    """ testing computePerturbedFreeEnergies """

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        numN = np.sum(N_k[:2])
        mbar = MBAR(u_kn[:2,:numN], N_k[:2])  # only do MBAR with the first and last set
        fe, fe_sigma = mbar.computePerturbedFreeEnergies(u_kn[2:,:numN])
        fe, fe_sigma = fe[0,1:], fe_sigma[0,1:]

        print(fe, fe_sigma)
        fe0 = test.analytical_free_energies()[2:]
        fe0 = fe0[1:] - fe0[0]

        z = (fe - fe0) / fe_sigma
        eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_computePMF():

    """ testing computePMF """

    name, test = generate_ho()
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
    mbar = MBAR(u_kn,N_k)
    #do a 1d PMF of the potential in the 3rd state:
    refstate = 2
    dx = 0.25
    xmin = test.O_k[refstate] - 1
    xmax = test.O_k[refstate] + 1
    within_bounds = (x_n >= xmin) & (x_n < xmax)
    bin_centers = dx*np.arange(np.int(xmin/dx),np.int(xmax/dx)) + dx/2
    bin_n = np.zeros(len(x_n),int)
    bin_n[within_bounds] = 1 + np.floor((x_n[within_bounds]-xmin)/dx)
    # 0 is reserved for samples outside the domain.  We will ignore this state
    range = np.max(bin_n)+1
    f_i, df_i = mbar.computePMF(u_kn[refstate,:], bin_n, range, uncertainties = 'from-specified', pmf_reference = 1)
    f0_i = 0.5*test.K_k[refstate]*(bin_centers-test.O_k[refstate])**2
    f_i, df_i = f_i[2:], df_i[2:] # first state is ignored, second is zero, with zero uncertainty
    normf0_i = f0_i[1:] - f0_i[0] # normalize to first state
    z = (f_i - normf0_i) / df_i
    eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)

def test_mbar_computeExpectationsInner():

    """Can MBAR calculate general expectations inner code (note: this just tests completion)"""

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
        eq(N_k, N_k_output)
        mbar = MBAR(u_kn, N_k)
        A_in = np.array([x_n, x_n ** 2, x_n ** 3])
        u_n = u_kn[:2,:]
        state_map = np.array([[0,0],[1,0],[2,0],[2,1]],int)
        [A_i, d2A_ij] = mbar.computeExpectationsInner(A_in, u_n, state_map)
