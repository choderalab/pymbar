import numpy as np
import pymbar
from pymbar.utils_for_testing import eq, suppress_derivative_warnings_for_tests

def load_oscillators(n_states, n_samples):
    name = "%dx%d oscillators" % (n_states, n_samples)
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output, s_n


def load_exponentials(n_states, n_samples):
    name = "%dx%d exponentials" % (n_states, n_samples)
    rates = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output, s_n


def _test(data_generator):
    name, U, N_k, s_n = data_generator()
    print(name)
    mbar = pymbar.MBAR(U, N_k)
    results1 = mbar.getFreeEnergyDifferences(uncertainty_method="svd", return_dict=True)
    fij1_t, dfij1_t = mbar.getFreeEnergyDifferences(uncertainty_method="svd", return_dict=False)
    results2 = mbar.getFreeEnergyDifferences(uncertainty_method="svd-ew", return_dict=True)
    fij1 = results1['Delta_f']
    dfij1 = results1['dDelta_f']
    fij2 = results2['Delta_f']
    dfij2 = results2['dDelta_f']

    # Check to make sure the returns from with and w/o dict are the same
    eq(fij1, fij1_t)
    eq(dfij1, dfij1_t)

    eq(pymbar.mbar_solvers.mbar_gradient(U, N_k, mbar.f_k), np.zeros(N_k.shape), decimal=8)
    eq(np.exp(mbar.Log_W_nk).sum(0), np.ones(len(N_k)), decimal=10)
    eq(np.exp(mbar.Log_W_nk).dot(N_k), np.ones(U.shape[1]), decimal=10)
    eq(pymbar.mbar_solvers.self_consistent_update(U, N_k, mbar.f_k), mbar.f_k, decimal=10)

    # Test against old MBAR code.
    with suppress_derivative_warnings_for_tests():
        mbar0 = pymbar.old_mbar.MBAR(U, N_k)
    fij0, dfij0 = mbar0.getFreeEnergyDifferences(uncertainty_method="svd")
    eq(mbar.f_k, mbar0.f_k, decimal=8)
    eq(np.exp(mbar.Log_W_nk), np.exp(mbar0.Log_W_nk), decimal=5)

    eq(fij0, fij1, decimal=8)
    eq(dfij0, dfij1, decimal=8)

    eq(fij0, fij2, decimal=8)
    eq(dfij0, dfij2, decimal=8)


def test_100x100_oscillators():
    data_generator = lambda: load_oscillators(100, 100)
    _test(data_generator)


def test_200x50_oscillators():
    data_generator = lambda: load_oscillators(200, 50)
    _test(data_generator)


def test_200x50_exponentials():
    data_generator = lambda: load_exponentials(200, 50)
    _test(data_generator)
