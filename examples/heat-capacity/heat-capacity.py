"""
todo -- simplify the total energy read in, the kinetic energy read-in, temperature read-in
"""
# =========================================================
# IMPORTS
# =========================================================

from optparse import OptionParser
from pathlib import Path

import numpy as np

import pymbar  # for MBAR analysis
from pymbar import timeseries  # for timeseries analysis

# ===================================================================================================
# INPUT PARAMETERS
# ===================================================================================================
def parse_cli():
    parser = OptionParser()

    parser.add_option(
        "-d",
        "--directory",
        dest="simulation",
        default="energydata",
        help="the directory of the energies we care about",
    )
    parser.add_option(
        "-b",
        "--n_bootstraps",
        dest="n_boots",
        type=int,
        default=0,
        help="Number of bootstrap samples taken",
    )
    parser.add_option(
        "-s",
        "--spacing",
        dest="num_intermediates",
        type="int",
        default=200,
        help="Number of intermediate simulations used to calculate finite differences (default 200)",
    )
    parser.add_option(
        "-f",
        "--finitedifftype",
        dest="dertype",
        default="temperature",
        help='the type of finite difference energy, choice is "temperature" or "beta" [default = %default]',
    )
    parser.add_option(
        "-r",
        "--randomseed",
        dest="rseed",
        type=int,
        default=None,
        help="random seed for bootstraping [default = %default]",
    )
    return parser.parse_args()


options, args = parse_cli()
simulation = options.simulation
n_boots = options.n_boots
num_intermediates = options.num_intermediates
dertype = options.dertype
rseed = options.rseed

# ========================================================
# CONSTANTS
# ========================================================

kB = 0.008314462  # Boltzmann constant (Gas constant) in kJ/(mol*K)
TE_COL_NUM = 11  # The column number of the total energy in ener_box#.output

num_temps = 16  # Last TEMP # + 1 (start counting at 1)
NUM_ITERATIONS = 1000  # The number of energies to be taken and analyzed, starting from the last
# Extra data will be ignored

if dertype == "temperature":  # if the temperatures are equally spaced
    types = "var", "dT", "ddT"
elif dertype == "beta":  # if the inverse temperatures are equally spaced.
    types = "var", "dbeta", "ddbeta"
else:
    sys.exit("type of finite difference not recognized must be 'beta' or 'temperature'")

NTYPES = len(types)

np.random.seed(rseed)  # seed the random numbers

###########################################################
#      For Cv vs T
#                _____
# Cv            /     \                <-- what we expect the graph to look like
#  ____________/       \____________
#                  T
############################################################

# =========================================================
# SUBROUTINES
# =========================================================


def read_total_energies(pathname, colnum):
    """Reads in the TEMP#/ener_box#.output file and parses it, returning an array of energies

    Parameters
    ----------
    pathname : str
        the path to the folder of the simulation
	colnum : int
        column the energy is found in

    Returns
    -------
    array-like
    """
    pathname = Path(pathname)
    print(f"--Reading total energies from {pathname}/...")

    # Initialize Return variables
    E_kn = np.zeros([num_temps, NUM_ITERATIONS])

    # Read files
    for k in range(num_temps):
        # Construct each TEMP#/ener_box#.output name and read in the file
        filename = pathname / f"TEMP{k}" / f"ener_box{k}.output"

        with open(filename) as infile:
            lines = infile.readlines()
            n_lines = len(lines)

        # Initialize arrays for E
        E_from_file = np.zeros(NUM_ITERATIONS)

        # Parse lines in each file
        for n in range(NUM_ITERATIONS):
            # Count down (the 2 is for index purposes(1) and to not use the double-counted last line (1))
            m = n_lines - 2 - n
            elements = lines[m].split()
            E_from_file[n] = float(elements[colnum])

        # Add in the E's for each timestep (n) at this temperature (k)
        E_kn[k] = E_from_file
    return E_kn


def read_simulation_temps(pathname, num_temps):
    """Reads in the various temperatures from each TEMP#/simul.output file by knowing
        beforehand the total number of temperatures (parameter at top)

    Parameters
    ----------
    pathname : str
    num_temps : int

    """
    pathname = Path(pathname)
    print(f"--Reading temperatures from {pathname}/...")

    # Initialize return variable
    temps_from_file = np.zeros(num_temps)

    for k in range(num_temps):
        filename = pathname / f"TEMP{k}" / f"simul{k}.output"
        with open(filename) as f:
            for line in f:
                if line[0:11] == "Temperature":
                    vals = line.split(":")
                    break
            temps_from_file[k] = float(vals[1])

    return temps_from_file


def print_results(string, E, dE, Cv, dCv, types):

    print(string)
    print("Temperature    dA        <E> +/- d<E>  ", end=" ")
    for t in types:
        print(f"    Cv +/- dCv ({t})", end=" ")
    print()
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    for k in range(originalK, K):
        print(
            f"{temp_k[k]:8.3f} {mbar.f_k[k] / beta_k[k]:8.3f} {E[k]:9.3f} +/- {dE[k]:5.3f}",
            end=" ",
        )
        for i in range(len(types)):
            if Cv[k, i, 0] < -100000.0:
                print("         N/A          ", end=" ")
            else:
                print(f"    {Cv[k, i, 0]:7.4f} +/- {dCv[k, i]:6.4f}", end=" ")
        print()


# ========================================================================
# MAIN
# ========================================================================

# ------------------------------------------------------------------------
# Read Data From File
# ------------------------------------------------------------------------

print()
print("Preparing data:")
T_from_file = read_simulation_temps(simulation, num_temps)
E_from_file = read_total_energies(simulation, TE_COL_NUM)
K = len(T_from_file)
N_k = np.zeros(K, dtype=int)
g = np.zeros(K)

for k in range(K):  # subsample the energies
    g[k] = timeseries.statistical_inefficiency(E_from_file[k])
    # indices of uncorrelated samples
    indices = np.array(timeseries.subsample_correlated_data(E_from_file[k], g=g[k]))
    N_k[k] = len(indices)  # number of uncorrelated samples
    E_from_file[k, 0 : N_k[k]] = E_from_file[k, indices]

# ------------------------------------------------------------------------
# Insert Intermediate T's and corresponding blank U's and E's
# ------------------------------------------------------------------------
temp_k = T_from_file
minT = T_from_file[0]
maxT = T_from_file[len(T_from_file) - 1]
# beta = 1/(k*BT)
# T = 1/(kB*beta)
if dertype == "temperature":
    minv = minT
    maxv = maxT
elif dertype == "beta":  # actually going in the opposite direction as beta for logistical reasons
    minv = 1 / (kB * minT)
    maxv = 1 / (kB * maxT)
delta = (maxv - minv) / (num_intermediates - 1)
originalK = len(temp_k)

print("--Adding intermediate temperatures...")

val_k = []
currentv = minv
if dertype == "temperature":
    # Loop, inserting equally spaced T's at which we are interested in the properties
    while currentv <= maxv:
        val_k = np.append(val_k, currentv)
        currentv = currentv + delta
    temp_k = np.concatenate((temp_k, np.array(val_k)))
elif dertype == "beta":
    # Loop, inserting equally spaced T's at which we are interested in the properties
    while currentv >= maxv:
        val_k = np.append(val_k, currentv)
        currentv = currentv + delta
    temp_k = np.concatenate((temp_k, (1 / (kB * np.array(val_k)))))

# Update number of states
K = len(temp_k)
# Loop, inserting E's into blank matrix (leaving blanks only where new Ts are inserted)

# Number of samples (n) for each state (k) = number of iterations/energies
Nall_k = np.zeros([K], dtype=int)
E_kn_files = np.zeros([K, NUM_ITERATIONS])

for k in range(originalK):
    E_kn_files[k, 0 : N_k[k]] = E_from_file[k, 0 : N_k[k]]
    Nall_k[k] = N_k[k]

# ------------------------------------------------------------------------
# Compute inverse temperatures
# ------------------------------------------------------------------------
beta_k = 1 / (kB * temp_k)

# ------------------------------------------------------------------------
# Compute reduced potential energies
# ------------------------------------------------------------------------

print("--Computing reduced energies...")

# u_kln is reduced pot. ener. of segment n of temp k evaluated at temp l
u_kn = np.zeros([K, np.sum(N_k)])
# u_kln is reduced pot. ener. of segment n of temp k evaluated at temp l
E_kn_samp = np.zeros([K, NUM_ITERATIONS])
# we add +1 to the bootstrap number, as the zeroth bootstrap sample is the original
n_boots_work = n_boots + 1

allCv_expect = np.zeros([K, NTYPES, n_boots_work])
dCv_expect = np.zeros([K, NTYPES])
allE_expect = np.zeros([K, n_boots_work])
allE2_expect = np.zeros([K, n_boots_work])
dE_expect = np.zeros([K])


for n in range(n_boots_work):
    if n > 0:
        print(f"Bootstrap: {n:d}/{n_boots:d}")

    for k in range(K):
        # resample the results:
        if Nall_k[k] > 0:
            if n == 0:  # don't randomize the first one
                booti = np.array(range(N_k[k]))
            else:
                booti = np.random.randint(Nall_k[k], size=Nall_k[k])
            E_kn_samp[k, 0 : Nall_k[k]] = E_kn_files[k, booti]

    for l in range(K):
        nsum = 0
        for k in range(K):
            u_kn[l, nsum : nsum + Nall_k[k]] = beta_k[l] * E_kn_samp[k, 0 : Nall_k[k]]
            nsum = nsum + Nall_k[k]
    # ------------------------------------------------------------------------
    # Initialize MBAR
    # ------------------------------------------------------------------------

    # Initialize MBAR with Newton-Raphson
    if n == 0:  # only print this information the first time
        print()
        print("Initializing MBAR:")
        print(f"--K = number of Temperatures with data = {originalK:d}")
        print(f"--L = number of total Temperatures = {K:d}")
        print(f"--N = number of Energies per Temperature = {np.max(Nall_k):d}")

    if n == 0:
        initial_f_k = None  # start from zero
    else:
        initial_f_k = mbar.f_k  # start from the previous final free energies to speed convergence

    mbar = pymbar.MBAR(
        u_kn, Nall_k, verbose=False, relative_tolerance=1e-12, initial_f_k=initial_f_k
    )

    # ------------------------------------------------------------------------
    # Compute Expectations for E_kt and E2_kt as E_expect and E2_expect
    # ------------------------------------------------------------------------

    print("")
    print("Computing Expectations for E...")
    E_kn = u_kn.copy()
    for k in range(K):
        # get the 'unreduced' potential -- we can't take differences of reduced potentials
        # because the beta is different; math is much more confusing with derivatives of the reduced potentials.
        E_kn[k, :] *= beta_k[k] ** (-1)
    results = mbar.compute_expectations(E_kn, state_dependent=True)
    E_expect = results["mu"]
    dE_expect = results["sigma"]
    allE_expect[:, n] = E_expect[:]

    # expectations for the differences, which we need for numerical derivatives
    results = mbar.compute_expectations(E_kn, output="differences", state_dependent=True)
    DeltaE_expect = results["mu"]
    dDeltaE_expect = results["sigma"]
    print("Computing Expectations for E^2...")

    results = mbar.compute_expectations(E_kn ** 2, state_dependent=True)
    E2_expect = results["mu"]
    dE2_expect = results["sigma"]
    allE2_expect[:, n] = E2_expect[:]

    results = mbar.compute_free_energy_differences()
    df_ij = results["Delta_f"]
    ddf_ij = results["dDelta_f"]

    # ------------------------------------------------------------------------
    # Compute Cv for NVT simulations as <E^2> - <E>^2 / (RT^2)
    # ------------------------------------------------------------------------

    if n == 0:
        print()
        print("Computing Heat Capacity as ( <E^2> - <E>^2 ) / ( R*T^2 ) and as d<E>/dT")

    # Problem is that we don't have a good uncertainty estimate for the variance.
    # Try a silly trick: but it doesn't work super well.
    # An estimator of the variance of the standard estimator of th evariance is
    # var(sigma^2) = (sigma^4)*[2/(n-1)+kurt/n]. If we assume the kurtosis is low
    # (which it will be for sufficiently many samples), then we can say that
    # d(sigma^2) = sigma^2 sqrt[2/(n-1)].
    # However, dE_expect**2 is already an estimator of sigma^2/(n-1)
    # Cv = sigma^2/kT^2, so d(Cv) = d(sigma^2)/kT^2 = sigma^2*[sqrt(2/(n-1)]/kT^2
    # we just need an estimate of n-1, but we can try to get that by var(dE)/dE_expect**2
    # it's within 50% or so, but that's not good enough.

    allCv_expect[:, 0, n] = (E2_expect - (E_expect * E_expect)) / (kB * temp_k ** 2)

    ####################################
    # C_v by fluctuation formula
    ####################################

    # Cv = (A - B^2) / (kT^2)
    # d2(Cv) = [1/(kT^2)]^2 [(dCv/dA)^2*d2A + 2*dCv*(dCv/dA)*(dCv/dB)*dAdB + (dCv/dB)^2*d2B]
    # = [1/(kT^2)]^2 [d2A - 4*B*dAdB + 4*B^2*d2B]
    # But this formula is not working for uncertainies!

    if n == 0:
        # sigma^2 / (sigma^2/n) = effective number of samples
        N_eff = (E2_expect - (E_expect * E_expect)) / dE_expect ** 2
        dCv_expect[:, 0] = allCv_expect[:, 0, n] * np.sqrt(2 / N_eff)

    # only loop over the points that will be plotted, not the ones that
    for i in range(originalK, K):

        # Now, calculae heat capacity by T-differences
        im = i - 1
        ip = i + 1
        if i == originalK:
            im = originalK
        if i == K - 1:
            ip = i

        ####################################
        # C_v by first derivative of energy
        ####################################

        if dertype == "temperature":  # temperature derivative
            # C_v = d<E>/dT
            allCv_expect[i, 1, n] = (DeltaE_expect[im, ip]) / (temp_k[ip] - temp_k[im])
            if n == 0:
                dCv_expect[i, 1] = (dDeltaE_expect[im, ip]) / (temp_k[ip] - temp_k[im])
        elif dertype == "beta":  # beta derivative
            # Cv = d<E>/dT = dbeta/dT d<E>/beta = - kB*T(-2) d<E>/dbeta  = - kB beta^2 d<E>/dbeta
            allCv_expect[i, 1, n] = (
                kB * beta_k[i] ** 2 * (DeltaE_expect[ip, im]) / (beta_k[ip] - beta_k[im])
            )
            if n == 0:
                dCv_expect[i, 1] = (
                    -kB * beta_k[i] ** 2 * (dDeltaE_expect[ip, im]) / (beta_k[ip] - beta_k[im])
                )

        #########################################
        # C_v by second derivative of free energy
        #########################################

        if dertype == "temperature":
            # C_v = d<E>/dT = d/dT k_B T^2 df/dT = 2*T*df/dT + T^2*d^2f/dT^2

            if (i == originalK) or (i == K - 1):
                # We can't calculate this, set a number that will be printed as NAN
                allCv_expect[i, 2, n] = -10000000.0
            else:
                allCv_expect[i, 2, n] = (
                    kB
                    * temp_k[i]
                    * (
                        2 * df_ij[ip, im] / (temp_k[ip] - temp_k[im])
                        + temp_k[i]
                        * (df_ij[ip, i] - df_ij[i, im])
                        / ((temp_k[ip] - temp_k[im]) / (ip - im)) ** 2
                    )
                )

            if n == 0:
                # Previous work to calculate the uncertainty commented out, should be cleaned up eventually
                # all_Cv_expect[i,2,n] = kB*temp_k[i]*(2*df_ij[ip,i]+df_ij[i,im]/(temp_k[ip]-temp_k[im]) + temp_k[i]*(df_ij[ip,i]-df_ij[i,im])/(temp_k[ip]-temp_k[i])**2)
                # all_Cv_expect[i,2,n] = kB*([2*temp_k[i]/(temp_k[ip]-temp_k[im]) + temp_k[i]**2/(temp_k[ip]-temp_k[i])**2]*df_ij[ip,i] + [2*temp_k[i]/(temp_k[ip]-temp_k[im]) - temp_k[i]**2/(temp_k[ip]-temp_k[i])**2]) df_ij[i,im]
                # all_Cv_expect[i,2,n] = kB*(A df_ij[ip,i] + B df_ij[i,im]
                A = (
                    2 * temp_k[i] / (temp_k[ip] - temp_k[im])
                    + 4 * temp_k[i] ** 2 / (temp_k[ip] - temp_k[im]) ** 2
                )
                B = (
                    2 * temp_k[i] / (temp_k[ip] - temp_k[im])
                    + 4 * temp_k[i] ** 2 / (temp_k[ip] - temp_k[im]) ** 2
                )
                # dCv_expect[i,2,n] = kB* [(A ddf_ij[ip,i])**2 + (B sdf_ij[i,im])**2 + 2*A*B*cov(df_ij[ip,i],df_ij[i,im])
                # This isn't it either: need to figure out that last term.
                dCv_expect[i, 2] = kB * ((A * ddf_ij[ip, i]) ** 2 + (B * ddf_ij[i, im]) ** 2)
                # Would need to add function computing covariance of DDG, (A-B)-(C-D)

        elif dertype == "beta":
            # if beta is evenly spaced, rather than t, we can do 2nd derivative in beta
            # C_v = d<E>/dT = d/dT (df/dbeta) = dbeta/dT d/dbeta (df/dbeta) = -k_b beta^2 df^2/d^2beta
            if (i == originalK) or (i == K - 1):
                # Flag as N/A -- we don't try to compute at the endpoints for now
                allCv_expect[i, 2, n] = -10000000.0
            else:
                allCv_expect[i, 2, n] = (
                    kB
                    * beta_k[i] ** 2
                    * (df_ij[ip, i] - df_ij[i, im])
                    / ((beta_k[ip] - beta_k[im]) / (ip - im)) ** 2
                )
            if n == 0:
                dCv_expect[i, 2] = (
                    kB
                    * (beta_k[i]) ** 2
                    * (ddf_ij[ip, i] - ddf_ij[i, im])
                    / ((beta_k[ip] - beta_k[im]) / (ip - im)) ** 2
                )
                # also wrong, need to be fixed.

    if n == 0:
        print(
            "WARNING: only the first derivative (dT) analytic error estimates can currently be trusted."
        )
        print("They are the only ones reasonably close to bootstrap, within 10-15% at all T.")
        print()
        print_results(
            "Analytic Error Estimates", E_expect, dE_expect, allCv_expect, dCv_expect, types
        )

if n_boots > 0:
    Cv_boot = np.zeros([K, NTYPES])
    dCv_boot = np.zeros([K, NTYPES])
    dE_boot = np.zeros([K])

    for k in range(K):
        for i in range(NTYPES):
            # for these averages, don't include the first one, because it's the non-bootstrapped one.
            Cv_boot[k, i] = np.mean(allCv_expect[k, i, 1:n_boots_work])
            dCv_boot[k, i] = np.std(allCv_expect[k, i, 1:n_boots_work])
            dE_boot[k] = np.std(allE_expect[k, 1:n_boots_work])
    print_results(
        "Bootstrap Error Estimates", allE_expect[:, 0], dE_boot, allCv_expect, dCv_boot, types
    )
