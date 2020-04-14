"""
Example illustrating the application of MBAR to compute a 1D PMF from an umbrella sampling simulation.

The data represents an umbrella sampling simulation for the chi torsion of
a valine sidechain in lysozyme L99A with benzene bound in the cavity.

Reference:

    D. L. Mobley, A. P. Graves, J. D. Chodera, A. C. McReynolds, B. K. Shoichet and K. A. Dill,
    "Predicting absolute ligand binding free energies to a simple model site,"
    Journal of Molecular Biology 371(4):1118-1134 (2007).
    http://dx.doi.org/10.1016/j.jmb.2007.06.002


"""

import numpy as np

import pymbar  # multistate Bennett acceptance ratio
from pymbar import timeseries  # timeseries analysis

# Constants.
kB = 1.381e-23 * 6.022e23 / 1000.0  # Boltzmann constant in kJ/mol/K

temperature = 300  # assume a single temperature -- can be overridden with data from center.dat

# Parameters
K = 26  # number of umbrellas
N_max = 501  # maximum number of snapshots/simulation
T_k = np.ones(K, float) * temperature  # inital temperatures are all equal
beta = 1.0 / (kB * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
chi_min = -180.0  # min for PMF
chi_max = +180.0  # max for PMF
nbins = 36  # number of bins for 1D PMF

# Allocate storage for simulation data
# N_k[k] is the number of snapshots from umbrella simulation k
N_k = np.zeros([K], dtype=int)
# K_k[k] is the spring constant (in kJ/mol/deg**2) for umbrella simulation k
K_k = np.zeros([K])
# chi0_k[k] is the spring center location (in deg) for umbrella simulation k
chi0_k = np.zeros([K])
# chi_kn[k,n] is the torsion angle (in deg) for snapshot n from umbrella simulation k
chi_kn = np.zeros([K, N_max])
# u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
u_kn = np.zeros([K, N_max])
g_k = np.zeros([K])

# Read in umbrella spring constants and centers.
with open("data/centers.dat") as infile:
    lines = infile.readlines()

for k in range(K):
    # Parse line k.
    line = lines[k]
    tokens = line.split()
    chi0_k[k] = float(tokens[0])  # spring center location (in deg)
    # spring constant (read in kJ/mol/rad**2, converted to kJ/mol/deg**2)
    K_k[k] = float(tokens[1]) * (np.pi / 180) ** 2
    if len(tokens) > 2:
        T_k[k] = float(tokens[2])  # temperature the kth simulation was run at.

beta_k = 1.0 / (kB * T_k)  # beta factor for the different temperatures
different_temperatures = True
if min(T_k) == max(T_k):
    # if all the temperatures are the same, then we don't have to read in energies.
    different_temperatures = False

# Read the simulation data
for k in range(K):
    # Read torsion angle data.
    filename = f"data/prod{k:d}_dihed.xvg"
    print(f"Reading {filename}...")
    n = 0
    with open(filename, "r") as infile:
        for line in infile:
            if line[0] != "#" and line[0] != "@":
                tokens = line.split()
                chi = float(tokens[1])  # torsion angle
                # wrap chi_kn to be within [-180,+180)
                while chi < -180.0:
                    chi += 360.0
                while chi >= +180.0:
                    chi -= 360.0
                chi_kn[k, n] = chi
                n += 1
    N_k[k] = n

    if different_temperatures:  # if different temperatures are specified the metadata file,
        # then we need the energies to compute the PMF
        # Read energies
        filename = f"data/prod{k:d}_energies.xvg"
        print(f"Reading {filename}...")
        n = 0
        with open(filename, "r") as infile:
            for line in infile:
                if line[0] != "#" and line[0] != "@":
                    tokens = line.split()
                    # reduced potential energy without umbrella restraint
                    u_kn[k, n] = beta_k[k] * (float(tokens[2]) - float(tokens[1]))
                    n += 1

    # Compute correlation times for potential energy and chi
    # timeseries.  If the temperatures differ, use energies to determine samples; otherwise, use the cosine of chi

    if different_temperatures:
        g_k[k] = timeseries.statistical_inefficiency(u_kn[k, :], u_kn[k, 0 : N_k[k]])
        print(f"Correlation time for set {k:5d} is {g_k[k]:10.3f}")
        indices = timeseries.subsample_correlated_data(u_kn[k, 0 : N_k[k]])
    else:
        chi_radians = chi_kn[k, 0 : N_k[k]] / (180.0 / np.pi)
        g_cos = timeseries.statistical_inefficiency(np.cos(chi_radians))
        g_sin = timeseries.statistical_inefficiency(np.sin(chi_radians))
        print(f"g_cos = {g_cos:.1f} | g_sin = {g_sin:.1f}")
        g_k[k] = max(g_cos, g_sin)
        print(f"Correlation time for set {k:5d} is {g_k[k]:10.3f}")
        indices = timeseries.subsample_correlated_data(chi_radians, g=g_k[k])
    # Subsample data.
    N_k[k] = len(indices)
    u_kn[k, 0 : N_k[k]] = u_kn[k, indices]
    chi_kn[k, 0 : N_k[k]] = chi_kn[k, indices]

N_max = np.max(N_k)  # shorten the array size
# u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
u_kln = np.zeros([K, K, N_max])

# Set zero of u_kn -- this is arbitrary.
u_kn -= u_kn.min()

# compute bin centers
bin_center_i = np.zeros([nbins])
bin_edges = np.linspace(chi_min, chi_max, nbins + 1)
for i in range(nbins):
    bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

N = np.sum(N_k)
chi_n = pymbar.utils.kn_to_n(chi_kn, N_k=N_k)

# Evaluate reduced energies in all umbrellas
print("Evaluating reduced potential energies...")
for k in range(K):
    for n in range(N_k[k]):
        # Compute minimum-image torsion deviation from umbrella center l
        dchi = chi_kn[k, n] - chi0_k
        for l in range(K):
            if abs(dchi[l]) > 180.0:
                dchi[l] = 360.0 - abs(dchi[l])

        # Compute energy of snapshot n from simulation k in umbrella potential l
        u_kln[k, :, n] = u_kn[k, n] + beta_k[k] * (K_k / 2.0) * dchi ** 2

# initialize PMF with the data collected
pmf = pymbar.PMF(u_kln, N_k, verbose=True)
# Compute PMF in unbiased potential (in units of kT).
histogram_parameters = {}
histogram_parameters["bin_edges"] = [bin_edges]
pmf.generate_pmf(u_kn, chi_n, pmf_type="histogram", histogram_parameters=histogram_parameters)
results = pmf.get_pmf(bin_center_i, reference_point="from-lowest")
center_f_i = results["f_i"]
center_df_i = results["df_i"]

# Write out PMF
print("PMF (in units of kT), from histogramming")
print(f"{'bin':>8s} {'f':>8s} {'df':>8s}")
for i in range(nbins):
    print(f"{bin_center_i[i]:8.1f} {center_f_i[i]:8.3f} {center_df_i[i]:8.3f}")

# NOW KDE:
kde_parameters = {}
kde_parameters["bandwidth"] = 0.5 * ((chi_max - chi_min) / nbins)
pmf.generate_pmf(u_kn, chi_n, pmf_type="kde", kde_parameters=kde_parameters)
results = pmf.get_pmf(bin_center_i, reference_point="from-lowest")
# Write out PMF for KDE
center_f_i = results["f_i"]
print("")
print("PMF (in units of kT), from KDE")
print(f"{'bin':>8s} {'f':>8s}")
for i in range(nbins):
    print(f"{bin_center_i[i]:8.1f} {center_f_i[i]:8.3f}")
