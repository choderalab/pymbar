import pandas as pd
import numpy as np
import pymbar
from pymbar.testsystems.pymbar_datasets import load_gas_data, load_8proteins_data
import time

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


n_states = 50
#name, u_kn0, N_k0, s_n = load_exponentials(n_states, 500)
name, u_kn0, N_k0, s_n = load_8proteins_data()
#name, u_kn0, N_k0, s_n = load_gas_data()
n_states = len(N_k0)

solver_protocol0 = None
solver_protocol1 = None

u_kn, N_k = pymbar.mbar_solvers.get_representative_sample(u_kn0, N_k0, s_n, subsampling=10)
#u_kn, N_k = pymbar.mbar_solvers.get_representative_sample_old(u_kn0, N_k0, s_n)
%time fsub1, asub1 = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, np.zeros_like(N_k), solver_protocol=solver_protocol0)
%time f0, a0 = pymbar.mbar_solvers.solve_mbar(u_kn0, N_k0, fsub1, solver_protocol=solver_protocol1)

%time f1, a1 = pymbar.mbar_solvers.solve_mbar(u_kn0, N_k0, np.zeros_like(N_k), solver_protocol=solver_protocol1)
