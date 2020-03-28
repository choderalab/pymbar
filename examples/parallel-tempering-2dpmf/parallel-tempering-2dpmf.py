#!/usr/bin/python

"""
Estimate 2D potential of mean force for alanine dipeptide parallel tempering data using MBAR.

PROTOCOL

* Potential energies and (phi, psi) torsions from parallel tempering simulation are read in by temperature
* Replica trajectories of potential energies and torsions are reconstructed to reflect their true temporal
  correlation, and then subsampled to produce statistically independent samples, collecting them again by temperature
* The `pymbar` class is initialized to compute the dimensionless free energies at each temperature using MBAR
* The torsions are binned into sequentially labeled bins in two dimensions
* The relative free energies and uncertainties of these torsion bins at the temperature of interest is estimated
* The 2D PMF is written out

REFERENCES

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008. http://dx.doi.org/10.1063/1.2978177
"""

# ===================================================================================================
# IMPORTS
# ===================================================================================================
from pathlib import Path

import numpy as np
import pymbar  # for MBAR analysis

from pymbar import timeseries  # for timeseries analysis

# ===================================================================================================
# CONSTANTS
# ===================================================================================================

kB = 1.3806503 * 6.0221415 / 4184.0  # Boltzmann constant in kcal/mol/K

# ===================================================================================================
# PARAMETERS
# ===================================================================================================

DATA_DIRECTORY = Path("data/")  # directory containing the parallel tempering data
# file containing temperatures in K
temperature_list_filename = DATA_DIRECTORY / "temperatures"
free_energies_filename = "f_k.out"
# file containing total energies (in kcal/mol) for each temperature and snapshot
potential_energies_filename = DATA_DIRECTORY / "energies" / "potential-energies"
trajectory_segment_length = 20  # number of snapshots in each contiguous trajectory segment
niterations = 500  # number of iterations to use
target_temperature = 302  # target temperature for 2D PMF (in K)
nbins_per_torsion = 10  # number of bins per torsion dimension

# ===================================================================================================
# SUBROUTINES
# ===================================================================================================


def read_file(filename):
    """Read contents of the specified file.

    Parameters:
    -----------
    filename : str
        The name of the file to be read

    Returns
    -------
    lines : list of str
        The contents of the file, split by line
    """
    with open(filename, "r") as f:
        return f.readlines()


# ===================================================================================================
# MAIN
# ===================================================================================================

# ===================================================================================================
# Read temperatures
# ===================================================================================================

# Read list of temperatures.
lines = read_file(temperature_list_filename)
# Construct list of temperatures
temperatures = lines[0].split()
# Create array of temperatures
K = len(temperatures)
temperature_k = np.zeros([K])  # temperature_k[k] is temperature of temperature index k in K
for k in range(K):
    temperature_k[k] = float(temperatures[k])
# Compute inverse temperatures
beta_k = (kB * temperature_k) ** (-1)

# Define other constants
T = trajectory_segment_length * niterations  # total number of snapshots per temperature

# ===================================================================================================
# Read potential eneriges
# ===================================================================================================

print("Reading potential energies...")
# U_kn[k,t] is the potential energy (in kcal/mol) for snapshot t of temperature index k
U_kt = np.zeros([K, T])
lines = read_file(potential_energies_filename)
print(f"{len(lines):d} lines read, processing {T:d} snapshots")
for t in range(T):
    # Get line containing the energies for snapshot t of trajectory segment n
    line = lines[t]
    # Extract energy values from text
    elements = line.split()
    for k in range(K):
        U_kt[k, t] = float(elements[k])

# ===================================================================================================
# Read phi, psi trajectories
# ===================================================================================================

print("Reading phi, psi trajectories...")
# phi_kt[k,n,t] is phi angle (in degrees) for snapshot t of temperature k
phi_kt = np.zeros([K, T])
# psi_kt[k,n,t] is psi angle (in degrees) for snapshot t of temperature k
psi_kt = np.zeros([K, T])
for k in range(K):
    phi_filename = DATA_DIRECTORY / "backbone-torsions" / f"{k:d}.phi"
    psi_filename = DATA_DIRECTORY / "backbone-torsions" / f"{k:d}.psi"
    phi_lines = read_file(phi_filename)
    psi_lines = read_file(psi_filename)
    print(f"k = {k:d}, {len(phi_lines):d} phi lines read, {len(psi_lines):d} psi lines read")
    for t in range(T):
        # Extract phi and psi
        phi_kt[k, t] = float(phi_lines[t])
        psi_kt[k, t] = float(psi_lines[t])

# ===================================================================================================
# Read replica indices
# ===================================================================================================

print("Reading replica indices...")
filename = DATA_DIRECTORY / "replica-indices"
lines = read_file(filename)
# replica_ki[i,k] is the replica index of temperature k for iteration i
replica_ik = np.zeros([niterations, K], np.int32)
for i in range(niterations):
    elements = lines[i].split()
    for k in range(K):
        replica_ik[i, k] = int(elements[k])
print(f"Replica indices for {niterations:d} iterations processed.")

# ===================================================================================================
# Permute data by replica and subsample to generate an uncorrelated subset of data by temperature
# ===================================================================================================

assume_uncorrelated = False
if assume_uncorrelated:
    # DEBUG - use all data, assuming it is uncorrelated
    print("Using all data, assuming it is uncorrelated...")
    U_kn = U_kt.copy()
    phi_kn = phi_kt.copy()
    psi_kn = psi_kt.copy()
    N_k = np.zeros([K], np.int32)
    N_k[:] = T
    N_max = T
else:
    # Permute data by replica
    print("Permuting data by replica...")
    U_kt_replica = U_kt.copy()
    phi_kt_replica = psi_kt.copy()
    psi_kt_replica = psi_kt.copy()
    for iteration in range(niterations):
        # Determine which snapshot indices are associated with this iteration
        snapshot_indices = iteration * trajectory_segment_length + np.arange(
            0, trajectory_segment_length
        )
        for k in range(K):
            # Determine which replica generated the data from temperature k at this iteration
            replica_index = replica_ik[iteration, k]
            # Reconstruct portion of replica trajectory.
            U_kt_replica[replica_index, snapshot_indices] = U_kt[k, snapshot_indices]
            phi_kt_replica[replica_index, snapshot_indices] = phi_kt[k, snapshot_indices]
            psi_kt_replica[replica_index, snapshot_indices] = psi_kt[k, snapshot_indices]
    # Estimate the statistical inefficiency of the simulation by analyzing the timeseries of interest.
    # We use the max of cos and sin of the phi and psi timeseries because they are periodic angles.
    # The  ## TODO: ???
    print("Computing statistical inefficiencies...")
    g_cosphi = timeseries.statistical_inefficiency_multiple(np.cos(phi_kt_replica * np.pi / 180.0))
    print(f"g_cos(phi) = {g_cosphi:.1f}")
    g_sinphi = timeseries.statistical_inefficiency_multiple(np.sin(phi_kt_replica * np.pi / 180.0))
    print(f"g_sin(phi) = {g_sinphi:.1f}")
    g_cospsi = timeseries.statistical_inefficiency_multiple(np.cos(psi_kt_replica * np.pi / 180.0))
    print(f"g_cos(psi) = {g_cospsi:.1f}")
    g_sinpsi = timeseries.statistical_inefficiency_multiple(np.sin(psi_kt_replica * np.pi / 180.0))
    print(f"g_sin(psi) = {g_sinpsi:.1f}")
    # Subsample data with maximum of all correlation times.
    print("Subsampling data...")
    g = np.max(np.array([g_cosphi, g_sinphi, g_cospsi, g_sinpsi]))
    indices = timeseries.subsample_correlated_data(U_kt[k, :], g=g)
    print(f"Using g = {g:.1f} to obtain {len(indices):d} uncorrelated samples per temperature")
    N_max = int(np.ceil(T / g))  # max number of samples per temperature
    U_kn = np.zeros([K, N_max])
    phi_kn = np.zeros([K, N_max])
    psi_kn = np.zeros([K, N_max])
    N_k = N_max * np.ones([K], dtype=int)
    for k in range(K):
        U_kn[k, :] = U_kt[k, indices]
        phi_kn[k, :] = phi_kt[k, indices]
        psi_kn[k, :] = psi_kt[k, indices]
    print(f"{N_max:d} uncorrelated samples per temperature")

# ===================================================================================================
# Generate a list of indices of all configurations in kn-indexing
# ===================================================================================================

# Create a list of indices of all configurations in kn-indexing.
mask_kn = np.zeros([K, N_max], dtype=np.bool)
for k in range(K):
    mask_kn[k, 0 : N_k[k]] = True
# Create a list from this mask.
indices = np.where(mask_kn)

# ===================================================================================================
# Compute reduced potential energy of all snapshots at all temperatures
# ===================================================================================================

print("Computing reduced potential energies...")
# u_kln[k,l,n] is reduced potential energy of trajectory segment n of temperature k evaluated at temperature l
u_kln = np.zeros([K, K, N_max])
for k in range(K):
    for l in range(K):
        u_kln[k, l, 0 : N_k[k]] = beta_k[l] * U_kn[k, 0 : N_k[k]]

# ===================================================================================================
# Bin torsions into histogram bins for PMF calculation
# ===================================================================================================

# Here, we bin the (phi,psi) samples into bins in a 2D histogram.
# We assign indices 0...(nbins-1) to the bins, even though the histograms are in two dimensions.
# All bins must have at least one sample in them.
# This strategy scales to an arbitrary number of dimensions.

print("Binning torsions...")
# Determine torsion bin size (in degrees)
torsion_min = -180.0
torsion_max = +180.0
dx = (torsion_max - torsion_min) / float(nbins_per_torsion)
# Assign torsion bins
# bin_kn[k,n] is the index of which histogram bin sample n from temperature index k belongs to
bin_kn = np.zeros([K, N_max], dtype=int)
nbins = 0
bin_nonzero = 0
bin_counts = []
bin_centers = []  # bin_centers[i] is a (phi,psi) tuple that gives the center of bin i
count_nonzero = []
centers_nonzero = []

for i in range(nbins_per_torsion):
    for j in range(nbins_per_torsion):
        # Determine (phi,psi) of bin center.
        phi = torsion_min + dx * (i + 0.5)
        psi = torsion_min + dx * (j + 0.5)

        # Determine which configurations lie in this bin.
        in_bin = (
            (phi - dx / 2 <= phi_kn[indices])
            & (phi_kn[indices] < phi + dx / 2)
            & (psi - dx / 2 <= psi_kn[indices])
            & (psi_kn[indices] < psi + dx / 2)
        )
        # Count number of configurations in this bin.
        bin_count = in_bin.sum()
        # Generate list of indices in bin.
        # set bin indices of both dimensions
        if bin_count > 0:
            count_nonzero.append(bin_count)
            centers_nonzero.append((phi, psi))
            bin_nonzero += 1

print(f"{bin_nonzero:d} bins were populated:")
for i in range(bin_nonzero):
    print(
        f"bin {i:5d} ({centers_nonzero[i][0]:6.1f}, {centers_nonzero[i][1]:6.1f}) {count_nonzero[i]:12d} conformations"
    )

x_n = np.zeros([np.sum(N_k), 2])  # the configurations

Ntot = 0  
for k in range(K):
    for n in range(N_k[k]):
        x_n[Ntot, 0] = phi_kn[k, n]
        x_n[Ntot, 1] = psi_kn[k, n]
        Ntot += 1

bin_edges = []
for i in range(2):
    bin_edges.append(np.linspace(torsion_min, torsion_max, nbins_per_torsion + 1))

# Initialize PMF with data collected
pmf = pymbar.PMF(u_kln, N_k)

# ===================================================================================================
# Compute PMF at the desired temperature.
# ===================================================================================================

print("Computing potential of mean force...")

# Compute reduced potential energies at the temperaure of interest
target_beta = 1.0 / (kB * target_temperature)
u_kn = target_beta * U_kn
# Compute PMF at this temperature, returning dimensionless free energies and uncertainties.
# f_i[i] is the dimensionless free energy of bin i (in kT) at the temperature of interest
# df_i[i,j] is an estimate of the covariance in the estimate of (f_i[i] - f_j[j], with reference
# the lowest free energy state.
# Compute PMF in unbiased potential (in units of kT).
histogram_parameters = {}
histogram_parameters["bin_edges"] = bin_edges
pmf.generate_pmf(u_kn, x_n, pmf_type="histogram", histogram_parameters=histogram_parameters)
results = pmf.get_pmf(np.array(centers_nonzero), uncertainties="from-lowest")
f_i = results["f_i"]
df_i = results["df_i"]

# Show free energy and uncertainty of each occupied bin relative to lowest free energy
print("2D PMF")
print()
print(f"{'bin':>8s} {'phi':>6s} {'psi':>6s} {'N':>8s} {'f':>10s} {'df':>10s}")

for i in range(bin_nonzero):
    print(
        f"{i:>8d} {centers_nonzero[i][0]:>6.1f} {centers_nonzero[i][1]:>6.1f} {count_nonzero[i]:>8d} {f_i[i]:>10.3f} {df_i[i]:>10.3f}"
    )
