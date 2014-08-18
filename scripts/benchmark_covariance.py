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


mbar_gens = {"new":lambda u_kn, N_k: pymbar.MBAR(u_kn, N_k)}
systems = [lambda : load_exponentials(25, 100), lambda : load_exponentials(100, 100), lambda : load_exponentials(250, 250),
lambda : load_oscillators(25, 100), lambda : load_oscillators(100, 100), lambda : load_oscillators(250, 250),
lambda : load_oscillators(500, 100), lambda : load_oscillators(1000, 50), lambda : load_oscillators(2000, 20), lambda : load_oscillators(4000, 10), 
lambda : load_exponentials(500, 100), lambda : load_exponentials(1000, 50), lambda : load_exponentials(2000, 20), lambda : load_oscillators(4000, 10), 
load_gas_data, load_8proteins_data]

timedata = []
for version, mbar_gen in mbar_gens.items():
    for sysgen in systems:
        name, u_kn, N_k, s_n = sysgen()
        K, N = u_kn.shape
        mbar = mbar_gen(u_kn, N_k)
        time0 = time.time()
        fij, dfij = mbar.getFreeEnergyDifferences(uncertainty_method="svd-ew-kab")
        dt =  time.time() - time0
        timedata.append([name, K, N, dt])


timedata = pd.DataFrame(timedata, columns=["name", "K", "N", "time"])
print timedata.to_string(float_format=lambda x: "%.3g" % x)
