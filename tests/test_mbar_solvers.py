import numpy as np
import pymbar
from pymbar.utils_for_testing import eq
import scipy.misc
from nose import SkipTest

def test_logsumexp():
    a = np.random.normal(size=(200, 500, 5))

    for axis in range(a.ndim):
        ans_ne = pymbar.mbar_solvers.logsumexp(a, axis=axis)
        ans_no_ne = pymbar.mbar_solvers.logsumexp(a, axis=axis, use_numexpr=False)
        ans_scipy = scipy.misc.logsumexp(a, axis=axis)
        eq(ans_ne, ans_no_ne)
        eq(ans_ne, ans_scipy)

def test_logsumexp_b():
    a = np.random.normal(size=(200, 500, 5))
    b = np.random.normal(size=(200, 500, 5)) ** 2.

    for axis in range(a.ndim):
        ans_ne = pymbar.mbar_solvers.logsumexp(a, b=b, axis=axis)
        ans_no_ne = pymbar.mbar_solvers.logsumexp(a, b=b, axis=axis, use_numexpr=False)
        ans_scipy = scipy.misc.logsumexp(a, b=b, axis=axis)
        eq(ans_ne, ans_no_ne)
        eq(ans_ne, ans_scipy)


def load_oscillators(n_states, n_samples):
    name = "%dx%d oscillators" % (n_states, n_samples)
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output

def load_exponentials(n_states, n_samples):
    name = "%dx%d exponentials" % (n_states, n_samples)
    rates = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output

def _test(data_generator):
    try:
        name, U, N_k = data_generator()
    except IOError as exc:
        raise(SkipTest("Cannot load dataset; skipping test.  Try downloading pymbar-datasets GitHub repository and setting PYMBAR-DATASETS environment variable.  Error was '%s'" % exc))
    except ImportError as exc:
        raise(SkipTest("Error importing pytables to load dataset; skipping test. Error was '%s'" % exc))
    print(name)
    mbar = pymbar.MBAR(U, N_k)
    eq(pymbar.mbar_solvers.mbar_gradient(U, N_k, mbar.f_k), np.zeros(N_k.shape), decimal=8)
    eq(np.exp(mbar.Log_W_nk).sum(0), np.ones(len(N_k)), decimal=10)
    eq(np.exp(mbar.Log_W_nk).dot(N_k), np.ones(U.shape[1]), decimal=10)
    eq(pymbar.mbar_solvers.self_consistent_update(U, N_k, mbar.f_k), mbar.f_k, decimal=10)

    # Test against old MBAR code.
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

def test_gas():
    data_generator = pymbar.testsystems.pymbar_datasets.load_gas_data
    _test(data_generator)

def test_8proteins():
    data_generator = pymbar.testsystems.pymbar_datasets.load_8proteins_data
    _test(data_generator)
