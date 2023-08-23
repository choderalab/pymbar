"""
Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators, for which
the true free energy differences can be computed analytically.

A number of replications of an experiment in which i.i.d. samples are drawn from a set of
K harmonic oscillators are produced.  For each replicate, we estimate the dimensionless free
energy differences and mean-square displacements (an observable), as well as their uncertainties.

For a 1D harmonic oscillator, the potential is given by
  V(x;K) = (K/2) * (x-x_0)**2
where K denotes the spring constant.

The equilibrium distribution is given analytically by
  p(x;beta,K) = sqrt[(beta K) / (2 pi)] exp[-beta K (x-x_0)**2 / 2]
The dimensionless free energy is therefore
  f(beta,K) = - (1/2) * ln[ (2 pi) / (beta K) ]

"""

# =============================================================================================
# IMPORTS
# =============================================================================================

import sys
import numpy as np
from pymbar import testsystems, exp, exp_gauss, bar, MBAR, FES
from pymbar.utils import ParameterError

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# =============================================================================================
# HELPER FUNCTIONS
# =============================================================================================


def stddev_away(namex, errorx, dx):
    if dx > 0:
        print(f"{namex} differs by {errorx / dx:.3f} standard deviations from analytical")
    else:
        print(f"{namex} differs by an undefined number of standard deviations")


def get_analytical(beta, K, O, observables):
    # For a harmonic oscillator with spring constant K,
    # x ~ Normal(x_0, sigma^2), where sigma = 1/sqrt(beta K)

    # Compute the absolute dimensionless free energies of each oscillator analytically.
    # f = - ln(sqrt((2 pi)/(beta K)) )
    print("Computing dimensionless free energies analytically...")

    sigma = (beta * K) ** -0.5
    f_k_analytical = -np.log(np.sqrt(2 * np.pi) * sigma)

    Delta_f_ij_analytical = f_k_analytical - np.vstack(f_k_analytical)

    A_k_analytical = dict()
    A_ij_analytical = dict()

    for observe in observables:
        if observe == "RMS displacement":
            # mean square displacement
            A_k_analytical[observe] = sigma
        if observe == "potential energy":
            # By equipartition
            A_k_analytical[observe] = 1 / (2 * beta) * np.ones(len(K), float)
        if observe == "position":
            # observable is the position
            A_k_analytical[observe] = O
        if observe == "position^2":
            # observable is the position^2
            A_k_analytical[observe] = (1 + beta * K * O**2) / (beta * K)

        A_ij_analytical[observe] = A_k_analytical[observe] - np.vstack(A_k_analytical[observe])

    return f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical


# =============================================================================================
# PARAMETERS
# =============================================================================================


K_k = np.array([25, 16, 9, 4, 1, 1])  # spring constants for each state
O_k = np.array([0, 1, 2, 3, 4, 5])  # offsets for spring constants
# number of samples from each state (can be zero for some states)
N_k = 10 * np.array([1000, 1000, 1000, 1000, 0, 1000])
Nk_ne_zero = N_k != 0
beta = 1.0  # inverse temperature for all simulations
K_extra = np.array([20, 12, 6, 2, 1])
O_extra = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
observables = ["position", "position^2", "potential energy", "RMS displacement"]

seed = None
# Uncomment the following line to seed the random number generated to
# produce reproducible output.
seed = 0
np.random.seed(seed)

# =============================================================================================
# MAIN
# =============================================================================================

# Determine number of simulations.
K = np.size(N_k)
if np.shape(K_k) != np.shape(N_k):
    msg = f"K_k ({np.shape(K_k):d}) and N_k ({np.shape(N_k):d}) must have same dimensions."
    raise ParameterError(msg)
if np.shape(O_k) != np.shape(N_k):
    msg = f"O_k ({np.shape(K_k):d}) and N_k ({np.shape(N_k):d}) must have same dimensions."
    raise ParameterError(msg)

# Determine maximum number of samples to be drawn for any state.
N_max = np.max(N_k)

f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical = get_analytical(
    beta, K_k, O_k, observables
)

print(f"This script will draw samples from {K:d} harmonic oscillators.")
print("The harmonic oscillators have equilibrium positions:", O_k)
print("and spring constants:", K_k)
print(
    "and the following number of samples will be drawn from each",
    "(can be zero if no samples drawn):",
    N_k,
)


# =============================================================================================
# Generate independent data samples from K one-dimensional harmonic oscillators centered at q = 0.
# =============================================================================================

print("generating samples...")
randomsample = testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(
    O_k=O_k, K_k=K_k, beta=beta
)
x_kn, u_kln, N_k = randomsample.sample(N_k, mode="u_kln", seed=seed)

# get the unreduced energies
U_kln = u_kln / beta

# =============================================================================================
# Estimate free energies and expectations.
# =============================================================================================

print("======================================")
print("      Initializing MBAR               ")
print("======================================")

# Estimate free energies from simulation using MBAR.
print("Estimating relative free energies from simulation (this may take a while)...")

# Initialize the MBAR class, determining the free energies.
mbar = MBAR(u_kln, N_k, relative_tolerance=1.0e-10, verbose=True)
# Get matrix of dimensionless free energy differences and uncertainty estimate.

print("=============================================")
print("      Testing compute_free_energy_differences       ")
print("=============================================")

results = mbar.compute_free_energy_differences()
Delta_f_ij_estimated = results["Delta_f"]
dDelta_f_ij_estimated = results["dDelta_f"]

# Compute error from analytical free energy differences.
Delta_f_ij_error = Delta_f_ij_estimated - Delta_f_ij_analytical

print("Error in free energies is:")
print(Delta_f_ij_error)
print("Uncertainty in free energies is:")
print(dDelta_f_ij_estimated)

print("Standard deviations away is:")
# mathematical manipulation to avoid dividing by zero errors; we don't care
# about the diagnonals, since they are identically zero.
df_ij_mod = dDelta_f_ij_estimated + np.identity(K)
stdevs = np.abs(Delta_f_ij_error / df_ij_mod)
for k in range(K):
    stdevs[k, k] = 0
print(stdevs)

print("==============================================")
print("             Testing computeBAR               ")
print("==============================================")

nonzero_indices = np.arange(K)[Nk_ne_zero]
Knon = len(nonzero_indices)
for i in range(Knon - 1):
    k = nonzero_indices[i]
    k1 = nonzero_indices[i + 1]
    w_F = u_kln[k, k1, 0 : N_k[k]] - u_kln[k, k, 0 : N_k[k]]  # forward work
    w_R = u_kln[k1, k, 0 : N_k[k1]] - u_kln[k1, k1, 0 : N_k[k1]]  # reverse work
    results = bar(w_F, w_R)
    df_bar = results["Delta_f"]
    ddf_bar = results["dDelta_f"]
    bar_analytical = f_k_analytical[k1] - f_k_analytical[k]
    bar_error = bar_analytical - df_bar
    print(
        f"BAR estimator for reduced free energy from states {k:d} to {k1:d} is {df_bar:f} +/- {ddf_bar:f}"
    )
    stddev_away("BAR estimator", bar_error, ddf_bar)

print("==============================================")
print("             Testing EXP               ")
print("==============================================")

print("EXP forward free energy")
for k in range(K - 1):
    if N_k[k] != 0:
        # forward work
        w_F = u_kln[k, k + 1, 0 : N_k[k]] - u_kln[k, k, 0 : N_k[k]]
        results = exp(w_F)
        df_exp = results["Delta_f"]
        ddf_exp = results["dDelta_f"]
        exp_analytical = f_k_analytical[k + 1] - f_k_analytical[k]
        exp_error = exp_analytical - df_exp
        print(f"df from states {k:d} to {k + 1:d} is {df_exp:f} +/- {ddf_exp:f}")
        stddev_away("df", exp_error, ddf_exp)

print("EXP reverse free energy")
for k in range(1, K):
    if N_k[k] != 0:
        w_R = u_kln[k, k - 1, 0 : N_k[k]] - u_kln[k, k, 0 : N_k[k]]  # reverse work
        results = exp(w_R)
        df_exp = -results["Delta_f"]
        ddf_exp = results["dDelta_f"]
        exp_analytical = f_k_analytical[k] - f_k_analytical[k - 1]
        exp_error = exp_analytical - df_exp
        print(f"df from states {k:d} to {k - 1:d} is {df_exp:f} +/- {ddf_exp:f}")
        stddev_away("df", exp_error, ddf_exp)

print("==============================================")
print("             Testing computeGauss               ")
print("==============================================")

print("Gaussian forward estimate")
for k in range(K - 1):
    if N_k[k] != 0:
        w_F = u_kln[k, k + 1, 0 : N_k[k]] - u_kln[k, k, 0 : N_k[k]]  # forward work
        results = exp_gauss(w_F)
        df_gauss = results["Delta_f"]
        ddf_gauss = results["dDelta_f"]
        gauss_analytical = f_k_analytical[k + 1] - f_k_analytical[k]
        gauss_error = gauss_analytical - df_gauss
        print(
            f"df for reduced free energy from states {k:d} to {k + 1:d} is {df_gauss:f} +/- {ddf_gauss:f}"
        )
        stddev_away("df", gauss_error, ddf_gauss)

print("Gaussian reverse estimate")
for k in range(1, K):
    if N_k[k] != 0:
        # reverse work
        w_R = u_kln[k, k - 1, 0 : N_k[k]] - u_kln[k, k, 0 : N_k[k]]
        results = exp_gauss(w_R)
        df_gauss = results["Delta_f"]
        ddf_gauss = results["dDelta_f"]
        gauss_analytical = f_k_analytical[k] - f_k_analytical[k - 1]
        gauss_error = gauss_analytical - df_gauss
        print(
            f"df for reduced free energy from states {k:d} to {k - 1:d} is {df_gauss:f} +/- {ddf_gauss:f}"
        )
        stddev_away("df", gauss_error, ddf_gauss)

print("======================================")
print("      Testing compute_expectations")
print("======================================")

A_kn_all = dict()
A_k_estimated_all = dict()
A_kl_estimated_all = dict()
N = np.sum(N_k)

for observe in observables:
    print("============================================")
    print(f"      Testing observable '{observe}'")
    print("============================================")

    if observe == "RMS displacement":
        state_dependent = True
        A_kn = np.zeros([K, N], dtype=np.float64)
        n = 0
        for k in range(K):
            for nk in range(N_k[k]):
                # observable is the squared displacement
                A_kn[:, n] = (x_kn[k, nk] - O_k[:]) ** 2
                n += 1

    # observable is the potential energy, a 3D array since the
    # potential energy is a function of the thermodynamic state
    elif observe == "potential energy":
        state_dependent = True
        A_kn = np.zeros([K, N], dtype=np.float64)
        n = 0
        for k in range(0, K):
            for nk in range(0, N_k[k]):
                A_kn[:, n] = U_kln[k, :, nk]
                n += 1

    # observable for estimation is the position
    elif observe == "position":
        state_dependent = False
        A_kn = np.zeros([K, N_max], dtype=np.float64)
        for k in range(0, K):
            A_kn[k, 0 : N_k[k]] = x_kn[k, 0 : N_k[k]]

    # observable for estimation is the position^2
    elif observe == "position^2":
        state_dependent = False
        A_kn = np.zeros([K, N_max], dtype=np.float64)
        for k in range(0, K):
            A_kn[k, 0 : N_k[k]] = x_kn[k, 0 : N_k[k]] ** 2

    results = mbar.compute_expectations(A_kn, state_dependent=state_dependent)
    A_k_estimated = results["mu"]
    dA_k_estimated = results["sigma"]

    # need to additionally transform to get the square root
    if observe == "RMS displacement":
        A_k_estimated = np.sqrt(A_k_estimated)
        # Compute error from analytical observable estimate.
        dA_k_estimated = dA_k_estimated / (2 * A_k_estimated)

    As_k_estimated = np.zeros([K], np.float64)
    dAs_k_estimated = np.zeros([K], np.float64)

    # 'standard' expectation averages - not defined if no samples
    nonzeros = np.arange(K)[Nk_ne_zero]

    totaln = 0
    for k in nonzeros:
        if (observe == "position") or (observe == "position^2"):
            As_k_estimated[k] = np.average(A_kn[k, 0 : N_k[k]])
            dAs_k_estimated[k] = np.sqrt(np.var(A_kn[k, 0 : N_k[k]]) / (N_k[k] - 1))
        elif (observe == "RMS displacement") or (observe == "potential energy"):
            totalp = totaln + N_k[k]
            As_k_estimated[k] = np.average(A_kn[k, totaln:totalp])
            dAs_k_estimated[k] = np.sqrt(np.var(A_kn[k, totaln:totalp]) / (N_k[k] - 1))
            totaln = totalp
            if observe == "RMS displacement":
                As_k_estimated[k] = np.sqrt(As_k_estimated[k])
                dAs_k_estimated[k] = dAs_k_estimated[k] / (2 * As_k_estimated[k])

    A_k_error = A_k_estimated - A_k_analytical[observe]
    As_k_error = As_k_estimated - A_k_analytical[observe]

    print("------------------------------")
    print("Now testing 'averages' mode")
    print("------------------------------")

    print(f"Analytical estimator of {observe} is")
    print(A_k_analytical[observe])

    print(f"MBAR estimator of the {observe} is")
    print(A_k_estimated)

    print("MBAR estimators differ by X standard deviations")
    stdevs = np.abs(A_k_error / dA_k_estimated)
    print(stdevs)

    print(f"Standard estimator of {observe} is (states with samples):")
    print(As_k_estimated[Nk_ne_zero])

    print("Standard estimators differ by X standard deviations (states with samples)")
    stdevs = np.abs(As_k_error[Nk_ne_zero] / dAs_k_estimated[Nk_ne_zero])
    print(stdevs)

    results = mbar.compute_expectations(
        A_kn, state_dependent=state_dependent, output="differences"
    )
    A_kl_estimated = results["mu"]
    dA_kl_estimated = results["sigma"]

    print("------------------------------")
    print("Now testing 'differences' mode")
    print("------------------------------")

    if (
        "RMS displacement" != observe
    ):  # can't test this, because we're actually computing the expectation of
        # the mean square displacement, and so the differences are <a_i^2> - <a_j^2>,
        # not sqrt<a_i>^2 - sqrt<a_j>^2
        A_kl_analytical = A_k_analytical[observe] - np.vstack(A_k_analytical[observe])
        A_kl_error = A_kl_estimated - A_kl_analytical

        print(f"Analytical estimator of differences of {observe} is")
        print(A_kl_analytical)

        print(f"MBAR estimator of the differences of {observe} is")
        print(A_kl_estimated)

        print("MBAR estimators differ by X standard deviations")
        stdevs = np.abs(A_kl_error / (dA_kl_estimated + np.identity(K)))
        for k in range(K):
            stdevs[k, k] = 0
        print(stdevs)

    # save up the A_k for use in compute_multiple_expectations
    A_kn_all[observe] = A_kn
    A_k_estimated_all[observe] = A_k_estimated
    A_kl_estimated_all[observe] = A_kl_estimated

print("=============================================")
print("      Testing compute_multiple_expectations")
print("=============================================")

# have to exclude the potential and RMS displacemet for now, not functions
# of a single state
observables_single = ["position", "position^2"]

A_ikn = np.zeros([len(observables_single), K, N_k.max()], np.float64)
for i, observe in enumerate(observables_single):
    A_ikn[i, :, :] = A_kn_all[observe]
for i in range(K):
    results = mbar.compute_multiple_expectations(A_ikn, u_kln[:, i, :], compute_covariance=True)
    A_i = results["mu"]
    dA_ij = results["sigma"]
    Ca_ij = results["covariances"]
    print(f"Averages for state {i:d}")
    print(A_i)
    print(f"Uncertainties for state {i:d}")
    print(dA_ij)
    print(f"Correlation matrix between observables for state {i:d}")
    print(Ca_ij)

print("============================================")
print("      Testing compute_entropy_and_enthalpy")
print("============================================")

results = mbar.compute_entropy_and_enthalpy(u_kn=u_kln, verbose=True)
Delta_f_ij = results["Delta_f"]
dDelta_f_ij = results["dDelta_f"]
Delta_u_ij = results["Delta_u"]
dDelta_u_ij = results["dDelta_u"]
Delta_s_ij = results["Delta_s"]
dDelta_s_ij = results["dDelta_s"]

print("Free energies")
print(Delta_f_ij)
print(dDelta_f_ij)
diffs1 = Delta_f_ij - Delta_f_ij_estimated
print(
    f"maximum difference between values computed here and in computeFreeEnergies is {np.max(diffs1):g}"
)
if np.max(np.abs(diffs1)) > 1.0e-10:
    print("Difference in values from computeFreeEnergies")
    print(diffs1)
diffs2 = dDelta_f_ij - dDelta_f_ij_estimated
print(
    f"maximum difference between uncertainties computed here and in computeFreeEnergies is {np.max(diffs2):g}"
)
if np.max(np.abs(diffs2)) > 1.0e-10:
    print("Difference in expectations from computeFreeEnergies")
    print(diffs2)

print("Energies")
print(Delta_u_ij)
print(dDelta_u_ij)
U_k = A_k_estimated_all["potential energy"]
expectations = U_k - np.vstack(U_k)
diffs1 = Delta_u_ij - expectations
print(
    f"maximum difference between values computed here and in compute_expectations is {np.max(diffs1):g}"
)
if np.max(np.abs(diffs1)) > 1.0e-10:
    print("Difference in values from compute_expectations")
    print(diffs1)

print("Entropies")
print(Delta_s_ij)
print(dDelta_s_ij)

# analytical entropy estimate
s_k_analytical = 0.5 / beta - f_k_analytical
Delta_s_ij_analytical = s_k_analytical - np.vstack(s_k_analytical)

Delta_s_ij_error = Delta_s_ij_analytical - Delta_s_ij
print("Error in entropies is:")
print(Delta_f_ij_error)

print("Standard deviations away is:")
# mathematical manipulation to avoid dividing by zero errors; we don't care
# about the diagnonals, since they are identically zero.
ds_ij_mod = dDelta_s_ij + np.identity(K)
stdevs = np.abs(Delta_s_ij_error / ds_ij_mod)
for k in range(K):
    stdevs[k, k] = 0
print(stdevs)

print("============================================")
print("      Testing compute_perturbed_free_energies")
print("============================================")

L = np.size(K_extra)
f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical = get_analytical(
    beta, K_extra, O_extra, observables
)

if np.size(O_extra) != np.size(K_extra):
    raise ParameterError(
        f"O_extra ({np.shape(K_k):d}) and K_extra ({np.shape(N_k):d}) must have the same dimensions."
    )

unew_kln = np.zeros([K, L, np.max(N_k)], np.float64)
for k in range(K):
    for l in range(L):
        unew_kln[k, l, 0 : N_k[k]] = (K_extra[l] / 2.0) * (x_kn[k, 0 : N_k[k]] - O_extra[l]) ** 2

results = mbar.compute_perturbed_free_energies(unew_kln)
Delta_f_ij_estimated = results["Delta_f"]
dDelta_f_ij_estimated = results["dDelta_f"]

Delta_f_ij_error = Delta_f_ij_estimated - Delta_f_ij_analytical

print("Error in free energies is:")
print(Delta_f_ij_error)

print("Standard deviations away is:")
# mathematical manipulation to avoid dividing by zero errors; we don't care
# about the diagnonals, since they are identically zero.
df_ij_mod = dDelta_f_ij_estimated + np.identity(L)
stdevs = np.abs(Delta_f_ij_error / df_ij_mod)
for l in range(L):
    stdevs[l, l] = 0
print(stdevs)

print("============================================")
print("      Testing compute_expectation (new states)")
print("============================================")

nth = 3
# test the nth "extra" states, O_extra[nth] & K_extra[nth]
for observe in observables:
    print("============================================")
    print(f"      Testing observable '{observe}'")
    print("============================================")

    if observe == "RMS displacement":
        state_dependent = True
        A_kn = np.zeros([K, 1, N_max], dtype=np.float64)
        for k in range(0, K):
            # observable is the squared displacement
            A_kn[k, 0, 0 : N_k[k]] = (x_kn[k, 0 : N_k[k]] - O_extra[nth]) ** 2

    # observable is the potential energy, a 3D array since the potential energy is a function of
    # thermodynamic state
    elif observe == "potential energy":
        state_dependent = True
        A_kn = unew_kln[:, [nth], :] / beta

    # position and position^2 can use the same observables
    # observable for estimation is the position
    elif observe == "position":
        state_dependent = False
        A_kn = A_kn_all["position"]

    elif observe == "position^2":
        state_dependent = False
        A_kn = A_kn_all["position^2"]

    A_k_estimated, dA_k_estimated
    results = mbar.compute_expectations(
        A_kn, unew_kln[:, [nth], :], state_dependent=state_dependent
    )
    A_k_estimated = results["mu"]
    dA_k_estimated = results["sigma"]
    # need to additionally transform to get the square root
    if observe == "RMS displacement":
        A_k_estimated = np.sqrt(A_k_estimated)
        dA_k_estimated = dA_k_estimated / (2 * A_k_estimated)

    A_k_error = A_k_estimated - A_k_analytical[observe][nth]

    print(f"Analytical estimator of {observe} is")
    print(A_k_analytical[observe][nth])

    print(f"MBAR estimator of the {observe} is")
    print(A_k_estimated)

    print("MBAR estimators differ by X standard deviations")
    stdevs = np.abs(A_k_error / dA_k_estimated)
    print(stdevs)

print("============================================")
print("      Testing compute_overlap")
print("============================================")

results = mbar.compute_overlap()
O = results["scalar"]
O_i = results["eigenvalues"]
O_ij = results["matrix"]

print("Overlap matrix output")
print(O_ij)

for k in range(K):
    print(f"Sum of row {k:d} is {np.sum(O_ij[k, :]):f} (should be 1),", end=" ")
    if np.abs(np.sum(O_ij[k, :]) - 1) < 1.0e-10:
        print("looks like it is.")
    else:
        print("but it's not.")

print("Eigenvalues of overlap matrix:")
print(O_i)


print("Overlap scalar measure: (1-lambda_2)")
print(O)

print("============================================")
print("    Testing compute_effective_sample_number")
print("============================================")

N_eff = mbar.compute_effective_sample_number(verbose=True)
print("Effective Sample number")
print(N_eff)
print("Compare stanadrd estimate of <x> with the MBAR estimate of <x>")
print("We should have that with MBAR, err_MBAR = sqrt(N_k/N_eff)*err_standard,")
print("so standard (scaled) results should be very close to MBAR results.")
print("No standard estimate exists for states that are not sampled.")
A_kn = x_kn
results = mbar.compute_expectations(A_kn)
val_mbar = results["mu"]
err_mbar = results["sigma"]
err_standard = np.zeros([K], dtype=np.float64)
err_scaled = np.zeros([K], dtype=np.float64)

for k in range(K):
    if N_k[k] != 0:
        # use position
        err_standard[k] = np.std(A_kn[k, 0 : N_k[k]]) / np.sqrt(N_k[k] - 1)
        err_scaled[k] = np.std(A_kn[k, 0 : N_k[k]]) / np.sqrt(N_eff[k] - 1)

print("                   ", end=" ")
for k in range(K):
    print(f"     {k:d}    ", end=" ")
print("")
print("MBAR             :", end=" ")
print(err_mbar)
print("standard         :", end=" ")
print(err_standard)
print("sqrt N_k/N_eff   :", end=" ")
print(np.sqrt(N_k / N_eff))
print("Standard (scaled):", end=" ")
print(err_standard * np.sqrt(N_k / N_eff))

print("============================================")
print("      Testing free energy surface functions   ")
print("============================================")

# For 2-D, The equilibrium distribution is given analytically by
#   p(x;beta,K) = sqrt[(beta K) / (2 pi)] exp[-beta K [(x-mu)^2] / 2]
#
# The dimensionless free energy is therefore
#   f(beta,K) = - (1/2) * ln[ (2 pi) / (beta K) ]
#
# In this problem, we are investigating the sum of two Gaussians, once
# centered at 0, and others centered at grid points.
#
#   V(x;K) = (K0/2) * [(x-x_0)^2]
#
# For 1-D, The equilibrium distribution is given analytically by
#   p(x;beta,K) = 1/N exp[-beta (K0 [x^2] / 2  + KU [(x-mu)^2] / 2)]
#   Where N is the normalization constant.
#
# The dimensionless free energy is the integral of this, and can be computed as:
#   f(beta,K)           = - ln [ (2*np.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
#   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))


def generate_fes_data(
    ndim=1, nbinsperdim=15, nsamples=1000, K0=20.0, Ku=100.0, gridscale=0.2, xrange=((-3, 3),)
):
    x0 = np.zeros([ndim], np.float64)  # center of base potential
    numbrellas = 1
    nperdim = np.zeros([ndim], int)
    for d in range(ndim):
        nperdim[d] = xrange[d][1] - xrange[d][0] + 1
        numbrellas *= nperdim[d]

    print(f"There are a total of {numbrellas:d} umbrellas.")

    # Enumerate umbrella centers, and compute the analytical free energy of
    # that umbrella
    print("Constructing umbrellas...")
    ksum = (Ku + K0) / beta
    kprod = (Ku * K0) / (beta * beta)
    f_k_analytical = np.zeros(numbrellas, np.float64)
    # xu_i[i,:] is the center of umbrella i
    xu_i = np.zeros([numbrellas, ndim], np.float64)

    dp = np.zeros(ndim, int)
    dp[0] = 1
    for d in range(1, ndim):
        dp[d] = nperdim[d] * dp[d - 1]

    umbrella_zero = 0
    for i in range(numbrellas):
        center = []
        for d in range(ndim):
            val = gridscale * ((int(i // dp[d])) % nperdim[d] + xrange[d][0])
            center.append(val)
        center = np.array(center)
        xu_i[i, :] = center
        mu2 = np.dot(center, center)
        f_k_analytical[i] = np.log(
            (ndim * np.pi / ksum) ** (3.0 / 2.0) * np.exp(-kprod * mu2 / (2.0 * ksum))
        )
        # assumes that we have one state that is at the zero.
        if np.all(center == 0.0):
            umbrella_zero = i
        i += 1
        f_k_analytical -= f_k_analytical[umbrella_zero]

    print(f"Generating {nsamples:d} samples for each of {numbrellas:d} umbrellas...")
    x_n = np.zeros([numbrellas * nsamples, ndim], np.float64)

    for i in range(numbrellas):
        for dim in range(ndim):
            # Compute mu and sigma for this dimension for sampling from V0(x) + Vu(x).
            # Product of Gaussians: N(x ; a, A) N(x ; b, B) = N(a ; b , A+B) x N(x ; c, C) where
            # C = 1/(1/A + 1/B)
            # c = C(a/A+b/B)
            # A = 1/K0, B = 1/Ku
            sigma = 1.0 / (K0 + Ku)
            mu = sigma * (x0[dim] * K0 + xu_i[i, dim] * Ku)
            # Generate normal deviates for this dimension.
            x_n[i * nsamples : (i + 1) * nsamples, dim] = np.random.normal(
                mu, np.sqrt(sigma), [nsamples]
            )

    u_kn = np.zeros([numbrellas, nsamples * numbrellas], np.float64)
    # Compute reduced potential due to V0.
    u_n = beta * (K0 / 2) * np.sum((x_n[:, :] - x0) ** 2, axis=1)
    for k in range(numbrellas):
        # reduced potential due to umbrella k
        uu = beta * (Ku / 2) * np.sum((x_n[:, :] - xu_i[k, :]) ** 2, axis=1)
        u_kn[k, :] = u_n + uu

    return u_kn, u_n, x_n, f_k_analytical


nbinsperdim = 15
gridscale = 0.2
nsamples = 1000
ndim = 1
K0 = 20.0
Ku = 100.0
print("============================================")
print("      Test 1: 1D free energy profile   ")
print("============================================")

xrange = [[-3, 3]]
ndim = 1
u_kn, u_n, x_n, f_k_analytical = generate_fes_data(
    K0=K0,
    Ku=Ku,
    ndim=ndim,
    nbinsperdim=nbinsperdim,
    nsamples=nsamples,
    gridscale=gridscale,
    xrange=xrange,
)
numbrellas = (np.shape(u_kn))[0]
N_k = nsamples * np.ones([numbrellas], int)
print("Solving for free energies of state ...")
mbar = MBAR(u_kn, N_k)

# Histogram bins are indexed using the scheme:
# index = 1 + np.floor((x[0] - xmin)/dx) + nbins*np.floor((x[1] - xmin)/dy)
# index = 0 is reserved for samples outside of the allowed domain
xmin = gridscale * (np.min(xrange[0][0]) - 1 / 2.0)
xmax = gridscale * (np.max(xrange[0][1]) + 1 / 2.0)
dx = (xmax - xmin) / nbinsperdim
nbins = 1 + nbinsperdim**ndim
bin_edges = np.linspace(xmin, xmax, nbins)  # list of bin edges.
bin_centers = np.zeros([nbins, ndim], np.float64)

ibin = 1
fes_analytical = np.zeros([nbins], np.float64)
minmu2 = 1000000
zeroindex = 0
# construct the bins and the fes
for i in range(nbinsperdim):
    xbin = xmin + dx * (i + 0.5)
    bin_centers[ibin, 0] = xbin
    mu2 = xbin * xbin
    if mu2 < minmu2:
        minmu2 = mu2
        zeroindex = ibin
    fes_analytical[ibin] = K0 * mu2 / 2.0
    ibin += 1
fzero = fes_analytical[zeroindex]
fes_analytical -= fzero
fes_analytical[0] = 0

bin_n = np.zeros([numbrellas * nsamples], int)
# Determine indices of those within bounds.
within_bounds = (x_n[:, 0] >= xmin) & (x_n[:, 0] < xmax)
# Determine states for these.
bin_n[within_bounds] = 1 + np.floor((x_n[within_bounds, 0] - xmin) / dx)
# Determine indices of bins that are not empty.
bin_counts = np.zeros([nbins], int)
for i in range(nbins):
    bin_counts[i] = (bin_n == i).sum()

# Compute fre energy profile, first with histograms
print("Solving for free energies of state to initialize free energy profile...")
mbar_options = dict()
mbar_options["verbose"] = False
fes = FES(u_kn, N_k, mbar_options=mbar_options)
print("Computing free energy profile ...")
histogram_parameters = dict()
histogram_parameters["bin_edges"] = bin_edges
fes.generate_fes(u_n, x_n, histogram_parameters=histogram_parameters)
results = fes.get_fes(
    bin_centers[:, 0],
    reference_point="from-specified",
    fes_reference=0.0,
    uncertainty_method="analytical",
)
f_ih = results["f_i"]
df_ih = results["df_i"]

# now estimate the PDF with a kde
kde_parameters = dict()
kde_parameters["bandwidth"] = dx / 3.0
fes.generate_fes(u_n, x_n, fes_type="kde", n_bootstraps=20, kde_parameters=kde_parameters)
results_kde = fes.get_fes(
    bin_centers,
    reference_point="from-specified",
    fes_reference=0.0,
    uncertainty_method="bootstrap",
)
f_ik = results_kde["f_i"]
df_ik = results_kde["df_i"]

# Show free energy and uncertainty of each occupied bin relative to lowest
# free energy

print("1D free energy profile:")
print(f"{bin_counts[0]:d} counts out of {numbrellas * nsamples:d} counts not in any bin")
print(
    f"{'bin':>8s} {'x':>6s} {'N':>8s} {'true':>10s}"
    f"{'f_hist':>10s} {'err_hist':>10s} {'df_hist':>10s} {'sig_hist':>8s}"
    f"{'f_kde':>10s} {'err_kde':>10s} {'df_kde':>10s} {'sig_kde':>8s}"
)

for i in range(1, nbins):
    error_h = fes_analytical[i] - f_ih[i]
    error_k = fes_analytical[i] - f_ik[i]
    if df_ih[i] > 0:
        stdevs_h = np.abs(error_h) / df_ih[i]
    else:
        stdevs_h = 0

    if df_ik[i] > 0:
        stdevs_k = np.abs(error_k) / df_ik[i]
    else:
        stdevs_k = 0

    print(
        f"{i:>8d} {bin_centers[i, 0]:>6.2f} {bin_counts[i]:>8d} {fes_analytical[i]:>10.3f}"
        f"{f_ih[i]:>10.3f} {error_h:>10.3f} {df_ih[i]:>10.3f} {stdevs_h:>8.2f}"
        f"{f_ik[i]:>10.3f} {error_k:>10.3f} {df_ik[i]:>10.3f} {stdevs_k:>8.2f}"
    )

print("============================================")
print("      Test 2: 2D free energy surface  ")
print("============================================")

xrange = [[-3, 3], [-3, 3]]
ndim = 2
nsamples = 500
u_kn, u_n, x_n, f_k_analytical = generate_fes_data(
    K0=K0,
    Ku=Ku,
    ndim=ndim,
    nbinsperdim=nbinsperdim,
    nsamples=nsamples,
    gridscale=gridscale,
    xrange=xrange,
)
numbrellas = np.shape(u_kn)[0]
N_k = nsamples * np.ones([numbrellas], int)
print("Solving for free energies of state ...")
mbar = MBAR(u_kn, N_k)

# The dimensionless free energy is the integral of this, and can be computed as:
#   f(beta,K)           = - ln [ (2*np.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
#   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))
# for computing harmonic samples

# Can compare the free energies computed with MBAR if desired with
# f_k_analytical

# Histogram bins are indexed using the scheme:
# index = 1 + np.floor((x[0] - xmin)/dx) + nbins*np.floor((x[1] - xmin)/dy)
# index = 0 is reserved for samples outside of the allowed domain

xmin = gridscale * (np.min(xrange[0][0]) - 1 / 2.0)
xmax = gridscale * (np.max(xrange[0][1]) + 1 / 2.0)
ymin = gridscale * (np.min(xrange[1][0]) - 1 / 2.0)
ymax = gridscale * (np.max(xrange[1][1]) + 1 / 2.0)
dx = (xmax - xmin) / nbinsperdim
dy = (ymax - ymin) / nbinsperdim
nbins = 1 + nbinsperdim**ndim
bin_centers = np.zeros([nbins, ndim], np.float64)

ibin = 1  # first reserved for something outside.
fes_analytical = np.zeros([nbins], np.float64)
minmu2 = 1000000
zeroindex = 0
# construct the bins and the fes
for i in range(nbinsperdim):
    xbin = xmin + dx * (i + 0.5)
    for j in range(nbinsperdim):
        # Determine (x,y) of bin center.
        ybin = ymin + dy * (j + 0.5)
        bin_centers[ibin, 0] = xbin
        bin_centers[ibin, 1] = ybin
        mu2 = xbin * xbin + ybin * ybin
        if mu2 < minmu2:
            minmu2 = mu2
            zeroindex = ibin
        fes_analytical[ibin] = K0 * mu2 / 2.0
        ibin += 1
fzero = fes_analytical[zeroindex]
fes_analytical -= fzero

bin_n = np.zeros([numbrellas * nsamples], int)
# Determine indices of those within bounds.
within_bounds = (x_n[:, 0] >= xmin) & (x_n[:, 0] < xmax) & (x_n[:, 1] >= ymin) & (x_n[:, 1] < ymax)
# Determine states for these.
xgrid = (x_n[within_bounds, 0] - xmin) / dx
ygrid = (x_n[within_bounds, 1] - ymin) / dy
bin_n[within_bounds] = 1 + xgrid.astype(int) + nbinsperdim * ygrid.astype(int)

# Determine indices of bins that are not empty.
bin_counts = np.zeros([nbins], int)
for i in range(nbins):
    bin_counts[i] = (bin_n == i).sum()

# Compute free energy surface, first using histograms weighted with MBAR
print("Computing free energy surface ...")
fes = FES(u_kn, N_k)

# for 2D bins, we input a list of the bin edges in each dimension.
histogram_parameters["bin_edges"] = [
    np.linspace(xmin, xmax, nbinsperdim + 1),
    np.linspace(ymin, ymax, nbinsperdim + 1),
]  # list of histogram edges.
fes.generate_fes(u_n, x_n, fes_type="histogram", histogram_parameters=histogram_parameters)
delta = 0.0001  # to break ties in things being too close.

results = fes.get_fes(
    bin_centers + delta,
    reference_point="from-specified",
    fes_reference=[0, 0],
    uncertainty_method="analytical",
)
f_i = results["f_i"]
df_i = results["df_i"]

# now generate the kernel density estimate
kde_parameters["bandwidth"] = 0.5 * dx
fes.generate_fes(u_n, x_n, fes_type="kde", kde_parameters=kde_parameters)
results_kde = fes.get_fes(bin_centers, reference_point="from-specified", fes_reference=[0, 0])
f_ik = results_kde["f_i"]

# Show free energy and uncertainty of each occupied bin relative to lowest
# free energy
print("2D FES:")

print(f"{bin_counts[0]:d} counts out of {numbrellas * nsamples:d} counts not in any bin")
print("Uncertainties only calculated for histogram methods")
print(
    f"{'bin':>8s} {'x':>6s} {'y':>6s} {'N':>8s} {'f_hist':>10s} {'f_kde':>10s} {'true':>10s} {'err_hist':>10s} {'err_kde':>10s} {'df':>10s} {'sigmas':>8s}"
)
for i in range(1, nbins):
    if i == zeroindex:
        stdevs = 0
        df_i[0] = 0
    else:
        error = fes_analytical[i] - f_i[i]
        if df_i[i] > 0:
            stdevs = np.abs(error) / df_i[i]
        else:
            stdevs = np.nan
    print(
        f"{i:>8d} "
        f"{bin_centers[i, 0]:>6.2f} "
        f"{bin_centers[i, 1]:>6.2f} "
        f"{bin_counts[i]:>8d} "
        f"{f_i[i]:>10.3f} "
        f"{f_ik[i]:>10.3f} "
        f"{fes_analytical[i]:>10.3f} "
        f"{error:>10.3f} "
        f"{fes_analytical[i]-f_ik[i]:>10.3f} "
        f"{df_i[i]:>10.3f} "
        f"{stdevs:>8.2f}"
    )
