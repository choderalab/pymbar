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


#n_states = 50
#name, u_kn0, N_k0, s_n = load_exponentials(n_states, 500)
name, u_kn0, N_k0, s_n = load_8proteins_data()
n_states = len(N_k0)

n_states = len(N_k0)
#u_kn, N_k = pymbar.mbar_solvers.get_two_sample_representation(u_kn0, N_k0, s_n)
u_kn, N_k = pymbar.mbar_solvers.get_representative_sample(u_kn0, N_k0, s_n, n_choose=200)

#solver_protocol = [dict(method="L-BFGS-B"), dict(method="hybr")]
#solver_protocol = [dict(method="L-BFGS-B"), dict(method="adaptive")]
solver_protocol = None
#solver_protocol = [dict(method="adaptive")]
#solver_protocol = [dict(method="hybr")]

%time fsub, asub = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, np.zeros_like(N_k), solver_protocol=solver_protocol)
%time f0, a0 = pymbar.mbar_solvers.solve_mbar(u_kn0, N_k0, fsub, solver_protocol=solver_protocol)
%time f1, a1 = pymbar.mbar_solvers.solve_mbar(u_kn0, N_k0, np.zeros_like(N_k), solver_protocol=solver_protocol)
