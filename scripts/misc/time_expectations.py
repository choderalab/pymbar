import numpy as np
import pymbar
import time

n_states, n_samples = 400, 10
name, testsystem, x_n, u_kn, N_k, s_n = pymbar.testsystems.HarmonicOscillatorsTestCase.evenly_spaced_oscillators(n_states, n_samples)

# Saving the results in case you want to use them in an old version of pymbar that doesn't have the helper function to generate data.
np.savez("u_kn.npz", u_kn)
np.savez("N_k.npz", N_k)
np.savez("s_n.npz", s_n)

u_kn = np.load("/home/kyleb/src/kyleabeauchamp/pymbar/u_kn.npz")["arr_0"]
N_k = np.load("/home/kyleb/src/kyleabeauchamp/pymbar/N_k.npz")["arr_0"]
s_n = np.load("/home/kyleb/src/kyleabeauchamp/pymbar/s_n.npz")["arr_0"]

mbar = pymbar.MBAR(u_kn, N_k, s_n)
mbar0 = pymbar.old_mbar.MBAR(u_kn, N_k, initial_f_k=mbar.f_k)  # use initial guess to speed this up

N = u_kn.shape[1]
x = np.random.normal(size=(N))
x0 = np.random.normal(size=(N))
x1 = np.random.normal(size=(N))
x2 = np.random.normal(size=(N))

%time fe, fe_sigma = mbar.computePerturbedFreeEnergies(np.array([x0, x1, x2]))
%time fe0, fe_sigma0 = mbar0.computePerturbedFreeEnergies(np.array([x0, x1, x2]))

fe - fe0
fe_sigma - fe_sigma0

%time A, dA = mbar.computeExpectations(x, compute_uncertainty=False)
%time A = mbar0.computeExpectations(x, compute_uncertainty=False)



