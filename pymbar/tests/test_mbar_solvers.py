import numpy as np
import pymbar
from pymbar.utils_for_testing import eq
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
    try:
        name, U, N_k, s_n = data_generator()
    except IOError as exc:
        raise(SkipTest("Cannot load dataset; skipping test.  Try downloading pymbar-datasets GitHub repository and setting PYMBAR-DATASETS environment variable.  Error was '%s'" % exc))
    except ImportError as exc:
        raise(SkipTest("Error importing pytables to load external dataset; skipping test. Error was '%s'" % exc))
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

def test_k69():
    data_generator = pymbar.testsystems.pymbar_datasets.load_k69_data
    _test(data_generator)


def test_subsampling():
    name, u_kn, N_k, s_n = load_exponentials(5, 20000)
    mbar = pymbar.MBAR(u_kn, N_k)
    u_kn_sub, N_k_sub = pymbar.mbar_solvers.subsample_data(u_kn, N_k, s_n, 2)
    mbar_sub = pymbar.MBAR(u_kn_sub, N_k_sub)
    eq(mbar.f_k, mbar_sub.f_k, decimal=2)
    
def test_protocols():
    '''Test that free energy is moderatley equal to analytical solution, independent of solver protocols'''
    #Supress the warnings when jacobian and Hessian information is not used in a specific solver
    import warnings
    warnings.filterwarnings('ignore', '.*does not use the jacobian.*')
    warnings.filterwarnings('ignore', '.*does not use Hessian.*')
    from pymbar.tests.test_mbar import z_scale_factor # Importing the hacky fix to asert that free energies are moderatley correct
    name, u_kn, N_k, s_n, test = load_oscillators(50, 100, provide_test=True)
    fa = test.analytical_free_energies()
    fa = fa[1:] - fa[0]

    #scipy.optimize.minimize methods, same ones that are checked for in mbar_solvers.py
    subsampling_protocols = ["L-BFGS-B", "dogleg", "CG", "BFGS", "Newton-CG", "TNC", "trust-ncg", "SLSQP"] 
    solver_protocols = ['hybr', 'lm'] #scipy.optimize.root methods. Omitting methods which do not use the Jacobian
    for subsampling_protocol in subsampling_protocols:
        for solver_protocol in solver_protocols:
            #Solve MBAR with zeros for initial weights
            mbar = pymbar.MBAR(u_kn, N_k, subsampling_protocol=({'method':subsampling_protocol},), solver_protocol=({'method':solver_protocol},))
            #Solve MBAR with the correct f_k used for the inital weights 
            mbar = pymbar.MBAR(u_kn, N_k, initial_f_k=mbar.f_k, subsampling_protocol=({'method':subsampling_protocol},), solver_protocol=({'method':solver_protocol},))
            fe, fe_sigma, Theta_ij = mbar.getFreeEnergyDifferences()
            fe, fe_sigma = fe[0,1:], fe_sigma[0,1:]
            z = (fe - fa) / fe_sigma
            eq(z / z_scale_factor, np.zeros(len(z)), decimal=0)
    #Clear warning filters
    warnings.resetwarnings()
