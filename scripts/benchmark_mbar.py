import pandas as pd
import numpy as np
import pymbar
from pymbar.testsystems.pymbar_datasets import load_gas_data, load_8proteins_data, load_k69_data
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


solver_protocol = None
mbar_gens = {"new":lambda u_kn, N_k, x_kindices: pymbar.MBAR(u_kn, N_k, x_kindices=x_kindices)}
#mbar_gens = {"old":lambda u_kn, N_k, x_kindices: pymbar.old_mbar.MBAR(u_kn, N_k)}
#mbar_gens = {"new":lambda u_kn, N_k, x_kindices: pymbar.MBAR(u_kn, N_k, x_kindices=x_kindices), "old":lambda u_kn, N_k, x_kindices: pymbar.old_mbar.MBAR(u_kn, N_k)}
systems = [lambda : load_exponentials(25, 100), lambda : load_exponentials(100, 100), lambda : load_exponentials(250, 250),
lambda : load_oscillators(25, 100), lambda : load_oscillators(100, 100), lambda : load_oscillators(250, 250),
load_gas_data, load_8proteins_data, load_k69_data]

timedata = []
for version, mbar_gen in mbar_gens.items():
    for sysgen in systems:
        name, u_kn, N_k, s_n = sysgen()
        time0 = time.time()
        mbar = mbar_gen(u_kn, N_k, s_n)
        dt = time.time() - time0
        wsum = np.linalg.norm(np.exp(mbar.Log_W_nk).sum(0) - 1.0)
        wdot = np.linalg.norm(np.exp(mbar.Log_W_nk).dot(N_k) - 1.0)
        obj, grad = pymbar.mbar_solvers.mbar_objective_and_gradient(u_kn, N_k, mbar.f_k)
        grad_norm = np.linalg.norm(grad)
        timedata.append([name, version, dt, grad_norm, wsum, wdot])


timedata = pd.DataFrame(timedata, columns=["name", "version", "time", "|grad|", "|W.sum(0) - 1|", "|W.dot(N_k) - 1|"])
print timedata.to_string(float_format=lambda x: "%.3g" % x)
