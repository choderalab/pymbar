import numpy as np
import pymbar
import warnings
from pymbar.utils_for_testing import eq, suppress_derivative_warnings_for_tests
import scipy.misc
from nose import SkipTest


def load_oscillators(n_states, n_samples, provide_test=False):
    name = "%dx%d oscillators" % (n_states, n_samples)
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
    returns = [name, u_kn, N_k_output, s_n]
    if provide_test:
        returns.append(test)
    return returns


def load_exponentials(n_states, n_samples, provide_test=False):
    name = "%dx%d exponentials" % (n_states, n_samples)
    rates = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
    returns = [name, u_kn, N_k_output, s_n]
    if provide_test:
        returns.append(test)
    return returns


def _test(data_generator):
    name, U, N_k, s_n = data_generator()
    print(name)
    mbar = pymbar.MBAR(U, N_k)
    eq(pymbar.mbar_solvers.mbar_gradient(U, N_k, mbar.f_k), np.zeros(N_k.shape), decimal=8)
    eq(np.exp(mbar.Log_W_nk).sum(0), np.ones(len(N_k)), decimal=10)
    eq(np.exp(mbar.Log_W_nk).dot(N_k), np.ones(U.shape[1]), decimal=10)
    eq(pymbar.mbar_solvers.self_consistent_update(U, N_k, mbar.f_k), mbar.f_k, decimal=10)

    # Test against old MBAR code.
    with suppress_derivative_warnings_for_tests():
        mbar0 = pymbar.old_mbar.MBAR(U, N_k)
    eq(mbar.f_k, mbar0.f_k, decimal=8)
    eq(np.exp(mbar.Log_W_nk), np.exp(mbar0.Log_W_nk), decimal=5)


def test_100x100_oscillators():
    data_generator = lambda : load_oscillators(100, 100)
    _test(data_generator)


def test_200x50_oscillators():
    data_generator = lambda : load_oscillators(200, 50)
    _test(data_generator)


def test_200x50_exponentials():
    data_generator = lambda : load_exponentials(200, 50)
    _test(data_generator)


def test_protocols():
    '''
    Test that free energy is moderately equal to analytical solution, independent of solver protocols
    '''

    # Importing the hacky fix to asert that free energies are moderately correct
    from pymbar.tests.test_mbar import z_scale_factor
    name, u_kn, N_k, s_n, test = load_oscillators(50, 100, provide_test=True)
    fa = test.analytical_free_energies()
    fa = fa[1:] - fa[0]
    with suppress_derivative_warnings_for_tests():
        # scipy.optimize.minimize methods, same ones that are checked for in mbar_solvers.py
        # subsampling_protocols = ['adaptive', 'L-BFGS-B', 'dogleg', 'CG', 'BFGS', 'Newton-CG', 'TNC', 'trust-ncg', 'SLSQP']
        # scipy.optimize.root methods. Omitting methods which do not use the Jacobian. Adding the custom adaptive protocol.
        solver_protocols = ['adaptive', 'hybr', 'lm', 'L-BFGS-B', 'dogleg', 'CG', 'BFGS', 'Newton-CG', 'TNC',
                            'trust-ncg', 'SLSQP']
        for solver_protocol in solver_protocols:
            # Solve MBAR with zeros for initial weights
            mbar = pymbar.MBAR(u_kn, N_k, solver_protocol=({'method': solver_protocol},))
            # Solve MBAR with the correct f_k used for the inital weights
            mbar = pymbar.MBAR(u_kn, N_k, initial_f_k=mbar.f_k, solver_protocol=({'method': solver_protocol},))
            results = mbar.getFreeEnergyDifferences(return_dict=True)
            fe = results['Delta_f'][0,1:]
            fe_sigma = results['dDelta_f'][0,1:]
            z = (fe - fa) / fe_sigma
            eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)
