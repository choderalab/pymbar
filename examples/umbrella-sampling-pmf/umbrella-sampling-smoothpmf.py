"""
Example illustrating the application of MBAR to compute a 1D PMF from an umbrella sampling simulation.

The data represents an umbrella sampling simulation for the chi torsion of a valine sidechain in lysozyme L99A with benzene bound in the cavity.

Reference:

    [1] M. R. Shirts and Andrew L. Ferguson,
    "Statistically optimal continuous potentials of mean force from
    umbrella sampling and multistate reweighting" https://arxiv.org/abs/2001.01170

    [2] D. L. Mobley, A. P. Graves, J. D. Chodera, A. C. McReynolds, B. K. Shoichet and K. A. Dill,
    "Predicting absolute ligand binding free energies to a simple model site,"
    Journal of Molecular Biology 371(4):1118-1134 (2007).
    http://dx.doi.org/10.1016/j.jmb.2007.06.002
"""
import copy
from timeit import default_timer as timer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # numerical array library

import pymbar  # multistate Bennett acceptance ratio
from pymbar import timeseries  # timeseries analysis

kB = 1.381e-23 * 6.022e23 / 1000.0  # Boltzmann constant in kJ/mol/K
nplot = 1200
# set minimizer options to display.
optimize_options = {"disp": True, "tol": 10 ** (-8)}
# histogram is self explanatory.  'kde' is a kernel density approximation. Currently it uses a
# Gaussian kernel, but this can be adjusted in the kde_parameters section below.

methods = ["unbiased-ml", "histogram", "kde", "biased-ml"]
# mc_methods = ['unbiased-map'] # which methods to run MCMC sampling on (much slower).
mc_methods = []  # which methods to run MCMC sampling on (much slower).
# The code supports arbitrary powers of of B-splines (that are supported by scipy
# Just replace '3' with the desired degree below. 1-5 suggested.
spline_degree = 3
nspline = 16  # number of spline knots used for the fit.
nbootstraps = 2  # should increase to ~50 for good statistics
mc_iterations = 50000  # could take a while.
smoothness_scalefac = 0.01
fig_suffix = "test1"  # figure suffix for identifiability of the output!

colors = {}
descriptions = {}
colors["histogram"] = "k-"
colors["kde"] = "k:"
colors["biased-ml"] = "g-"
colors["biased-map"] = "g--"
colors["unbiased-ml"] = "b-"
colors["unbiased-map"] = "b--"
descriptions["histogram"] = "Histogram"
descriptions["kde"] = "Kernel density (Gaussian)"
descriptions["unbiased-ml"] = "Unbiased state maximum likelihood"
descriptions["unbiased-map"] = "Unbiased state maximum a posteriori"
descriptions["simple"] = "vFEP"
descriptions["biased-ml"] = "biased states maximum likelihood"
descriptions["biased-map"] = "biased states maximum a posteriori"

optimization_algorithm = "Newton-CG"  # other good options are 'L-BFGS-B' and 'Custom-NR'
# optimization_algorithm = 'Custom-NR'  #other good options are 'L-BFGS-B' and 'Custom-NR'
# below - information to load the data.
temperature = 300  # assume a single temperature -- can be overridden with data from center.dat
# Parameters
K = 26  # number of umbrellas
N_max = 501  # maximum number of snapshots/simulation
T_k = np.ones(K, float) * temperature  # inital temperatures are all equal
beta = 1.0 / (kB * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
chi_min = -180.0  # min for PMF
chi_max = +180.0  # max for PMF
# number of bins for 1D PMF. Note, does not have to correspond to the number of umbrellas at all.
nbins = 30

# Allocate storage for simulation data
N_k = np.zeros([K], np.int32)  # N_k[k] is the number of snapshots from umbrella simulation k
# K_k[k] is the spring constant (in kJ/mol/deg**2) for umbrella simulation k
K_k = np.zeros([K], np.float64)
# chi0_k[k] is the spring center location (in deg) for umbrella simulation k
chi0_k = np.zeros([K], np.float64)
# chi_kn[k,n] is the torsion angle (in deg) for snapshot n from umbrella simulation k
chi_kn = np.zeros([K, N_max], np.float64)
# u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
u_kn = np.zeros([K, N_max], np.float64)
g_k = np.zeros([K], np.float32)

# Read in umbrella spring constants and centers.
with open("data/centers.dat") as infile:
    lines = infile.readlines()
for k in range(K):
    # Parse line k.
    line = lines[k]
    tokens = line.split()
    chi0_k[k] = float(tokens[0])  # spring center locatiomn (in deg)
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
    with open(filename) as infile:
        lines = infile.readlines()

    # Parse data.
    n = 0
    for line in lines:
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
        with open(filename) as infile:
            lines = infile.readlines()

        # Parse data.
        n = 0
        for line in lines:
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
        # g_k[k] = max(g_cos, g_sin)  #TODO: switch?
        g_k[k] = 1
        print(f"Correlation time for set {k:5d} is {g_k[k]:10.3f}")
        indices = timeseries.subsample_correlated_data(chi_radians, g=g_k[k])
    # Subsample data.
    N_k[k] = len(indices)
    u_kn[k, 0 : N_k[k]] = u_kn[k, indices]
    chi_kn[k, 0 : N_k[k]] = chi_kn[k, indices]

N_max = np.max(N_k)  # shorten the array size
# u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
u_kln = np.zeros([K, K, N_max], np.float64)

# Set zero of u_kn -- this is arbitrary.
u_kn -= u_kn.min()

# Construct torsion bins
# compute bin centers

bin_center_i = np.zeros([nbins], np.float64)
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

# initialize PMF with the data collected.
basepmf = pymbar.PMF(u_kln, N_k, verbose=True)


def bias_potential(x, k):
    """Define the bias potentials needed for umbrella sampling"""
    dchi = x - chi0_k[k]
    # vectorize the conditional
    i = np.fabs(dchi) > 180.0
    dchi = i * (360.0 - np.fabs(dchi)) + (1 - i) * dchi
    return beta_k[k] * (K_k[k] / 2.0) * dchi ** 2


def deltag(c, scalef=1, n=nspline):
    """bias on the smoothness, including periodicity. Normalization is indepednent of parameters, so we ignore.
    consider periodicity later!!
    """
    cdiff = np.diff(c)
    logp = -scalef / n * (np.sum(cdiff ** 2))
    return logp


def ddeltag(c, scalef=1, n=nspline):
    r"""derivative of the log prior above

    The logprior is \sum_{i}^{C-1} - scalef*(c_{i+1} - c_{i})^2
    this is unnormalized.  However, the normalization is independent of the values of the parameters, it only
    depends on the hyperparameter a, so we can omit it in the normalization.

    However, we fix c[1] to be zero.  So we are actually minimizing -scalef (c1)^2 + \sum_{i=1}^{C-1} - scalef*(c_{i+1} - c_{i})^2
    So the derivative is  -2*scalef*[c[1]-c[2],c[1]+2*c[2]-c[3],c[2]+2*c[3]-c[4], . . ., c[C-1]-c[C]]

    Finally, for the minimization, we are the first coefficient to zero, and it's not allowed to move.
    so we shift everything over by
    """
    cdiff = np.diff(c)
    lenc = len(c)
    dlogp = np.zeros(lenc)
    dlogp[0 : lenc - 1] = cdiff
    dlogp[1:lenc] -= cdiff
    # c[0] only occurs in the first two entries. We ignore the 0th (no derivative) and in the second, it's: 2a*(c[0]-2c[1]+c[2]), so
    # settting it zero is equal to ignoring it.
    dlogp = (2 * scalef / n) * dlogp
    return dlogp[1:]


def dddeltag(c, scalef=1, n=nspline):
    r"""
    Hessian of the log prior above

    The logprior is \sum_{i}^{C-1} - scalef*(c_{i+1} - c_{i})^2
    this is unnormalized.  However, the normalization is independent of the values of the parameters, it only
    depends on the hyperparameter a, so we can omit it in the normalization
    The derivative is -2*scalef*[c[1]-c[2],c[1]+2*c[2]-c[3],c[2]+2*c[3]-c[4], . . ., c[C-1]-c[C]]
    so the hessian willl be a constant matrix.  Will have -2 down diagonal, 1 on off diagonal, except for
    first and last rows
    """
    cdiff = np.diff(c)
    lenc = len(c)
    ddlogp = np.zeros([lenc, lenc])
    np.fill_diagonal(ddlogp, -2.0)
    np.fill_diagonal(ddlogp[1:], 1.0)
    np.fill_diagonal(ddlogp[:, 1:], 1.0)
    ddlogp[0, 0] = -1
    ddlogp[lenc - 1, lenc - 1] = -1
    ddlogp = (2 * scalef / n) * ddlogp
    # the first variable is set to zero in the MAP.
    return ddlogp[1:, 1:]


times = {}  # keep track of time elaped each method takes

xplot = np.linspace(chi_min, chi_max, nplot)  # number of points we are plotting
f_i_kde = None  # We check later if these have been defined or not
# the data we used initially to parameterize points, from the KDE
xstart = np.linspace(chi_min, chi_max, nbins * 3)

pmfs = {}
for methodfull in methods:

    # create a fresh copy of the initialized pmf object. Operate on that within the loop.
    # do the deepcopy here since there seem to be issues if it's done after data is added
    # For example, the scikit-learn kde object fails to deepopy.

    pmfs[methodfull] = copy.deepcopy(basepmf)
    pmf = pmfs[methodfull]
    start = timer()
    if "-" in methodfull:
        method, tominimize = methodfull.split("-")
    else:
        method = methodfull

    if method == "histogram":
        histogram_parameters = {}
        histogram_parameters["bin_edges"] = [bin_edges]
        pmf.generate_pmf(
            u_kn,
            chi_n,
            pmf_type="histogram",
            histogram_parameters=histogram_parameters,
            nbootstraps=nbootstraps,
        )

    if method == "kde":

        kde_parameters = {}
        # set the sigma for the spline.
        kde_parameters["bandwidth"] = 0.5 * ((chi_max - chi_min) / nbins)
        pmf.generate_pmf(
            u_kn, chi_n, pmf_type="kde", kde_parameters=kde_parameters, nbootstraps=nbootstraps
        )

        # save this for initializing other types
        results = pmf.get_pmf(xstart, reference_point="from-lowest")
        f_i_kde = results["f_i"]  # kde results

    if method in ["unbiased", "biased", "simple"]:

        spline_parameters = {}
        if method == "unbiased":
            spline_parameters["spline_weights"] = "unbiasedstate"
        elif method == "biased":
            spline_parameters["spline_weights"] = "biasedstates"
        elif method == "simple":
            spline_parameters["spline_weights"] = "simplesum"

        spline_parameters["nspline"] = nspline
        spline_parameters["spline_initialize"] = "explicit"

        # need to initialize: use KDE results for now (assumes KDE exists)
        spline_parameters["xinit"] = xstart
        if f_i_kde is not None:
            spline_parameters["yinit"] = f_i_kde
        else:
            spline_parameters["yinit"] = np.zeros(len(xstart))

        spline_parameters["xrange"] = [chi_min, chi_max]
        # introduce klocal to force K to use local definition of K, otherwise would use global value of k.
        spline_parameters["fkbias"] = [
            (lambda x, klocal=k: bias_potential(x, klocal)) for k in range(K)
        ]

        spline_parameters["kdegree"] = spline_degree
        spline_parameters["optimization_algorithm"] = optimization_algorithm
        spline_parameters["optimize_options"] = optimize_options

        if tominimize == "map":
            spline_parameters["objective"] = "map"
            spline_parameters["map_data"] = {}
            spline_parameters["map_data"]["logprior"] = lambda x: deltag(
                x, scalef=smoothness_scalefac
            )
            spline_parameters["map_data"]["dlogprior"] = lambda x: ddeltag(
                x, scalef=smoothness_scalefac
            )
            spline_parameters["map_data"]["ddlogprior"] = lambda x: dddeltag(
                x, scalef=smoothness_scalefac
            )
        else:
            spline_parameters["objective"] = "ml"
            spline_parameters["map_data"] = None

        pmf.generate_pmf(
            u_kn,
            chi_n,
            pmf_type="spline",
            spline_parameters=spline_parameters,
            nbootstraps=nbootstraps,
        )

    end = timer()
    times[methodfull] = end - start

    yout = {}
    yerr = {}
    print(f"PMF (in units of kT) for {methodfull}")
    print(f"{'bin':>8s} {'f':>8s} {'df':>8s}")
    results = pmf.get_pmf(bin_center_i, reference_point="from-lowest")
    for i in range(nbins):
        if results["df_i"] is not None:
            print(f"{bin_center_i[i]:8.1f} {results['f_i'][i]:8.1f} {results['df_i'][i]:8.1f}")
        else:
            print(f"{bin_center_i[i]:8.1f} {results['f_i'][i]:8.1f}")

    results = pmf.get_pmf(xplot, reference_point="from-lowest")
    yout[methodfull] = results["f_i"]
    yerr[methodfull] = results["df_i"]
    if len(xplot) <= nbins:
        errorevery = 1
    else:
        errorevery = int(np.floor(len(xplot) / nbins))

    if method == "histogram":
        # handle histogram differently
        perbin = nplot // nbins
        # get the errors in the rigtt place
        indices = np.arange(0, nplot, perbin) + int(perbin // 2)
        plt.errorbar(
            xplot[indices],
            yout[method][indices],
            yerr=yerr[method][indices],
            fmt="none",
            ecolor="k",
            elinewidth=1.0,
            capsize=3,
        )
        plt.plot(xplot, yout[method], colors[method], label=descriptions[method])
    else:
        plt.errorbar(
            xplot,
            yout[methodfull],
            yerr=yerr[methodfull],
            errorevery=errorevery,
            label=descriptions[methodfull],
            fmt=colors[methodfull],
            elinewidth=0.8,
            capsize=3,
        )

        if "-ml" in methodfull:
            aic = pmf.get_information_criteria(type="AIC")
            bic = pmf.get_information_criteria(type="BIC")
            print(f"AIC for {method} with {nspline:d} splines is: {aic:f}")
            print(f"BIC for {method} with {nspline:d} splines is: {bic:f}")

plt.xlim([chi_min, chi_max])
plt.ylim([0, 20])
plt.xlabel("Torsion angle (degrees)")
plt.ylabel(r"PMF (units of $k_BT$)")
plt.legend(fontsize="x-small")
plt.title("Comparison of PMFs")
plt.savefig(f"compare_pmf_{fig_suffix}.pdf")
plt.clf()

# now perform MC sampling in parameter space
pltname = [
    "bayes_posterior_histogram",
    "bayesian_95percent",
    "bayesian_1sigma",
    "parameter_time_series",
]
for method in mc_methods:
    pmf = pmfs[method]
    mc_parameters = {
        "niterations": mc_iterations,
        "fraction_change": 0.05,
        "sample_every": 10,
        "logprior": lambda x: deltag(x, scalef=smoothness_scalefac),
        "print_every": 50,
    }

    pmf.sample_parameter_distribution(chi_n, mc_parameters=mc_parameters, decorrelate=True)

    mc_results = pmf.get_mc_data()

    plt.figure(1)
    plt.hist(mc_results["logposteriors"], label=descriptions[method])

    # plot maximum likelihood as well
    method_ml = method.replace("map", "ml")
    pmf_ml = pmfs[method_ml]
    results_ml = pmf_ml.get_pmf(xplot, reference_point="from-lowest")

    plt.figure(2)
    plt.xlim([chi_min, chi_max])
    ci_results = pmf.get_confidence_intervals(xplot, 2.5, 97.5, reference="zero")
    ylow = ci_results["plow"]
    yhigh = ci_results["phigh"]
    plt.plot(xplot, ci_results["values"], colors[method], label=descriptions[method])
    plt.plot(xplot, results_ml["f_i"], colors[method_ml], label=descriptions[method_ml])
    plt.fill_between(xplot, ylow, yhigh, color=colors[method][0], alpha=0.3)
    plt.title("PMF with 95% confidence intervals")
    plt.xlabel("Torsion angle (degrees)")
    plt.ylabel(r"PMF (units of $k_BT$)")

    plt.figure(3)
    plt.xlim([chi_min, chi_max])
    ci_results = pmf.get_confidence_intervals(xplot, 16, 84)
    plt.plot(xplot, ci_results["values"], colors[method], label=descriptions[method])
    plt.plot(xplot, results_ml["f_i"], colors[method_ml], label=descriptions[method_ml])
    ylow = ci_results["plow"]
    yhigh = ci_results["phigh"]
    plt.fill_between(xplot, ylow, yhigh, color=colors[method][0], alpha=0.3)
    plt.xlabel("Torsion angle (degrees)")
    plt.ylabel(r"PMF (units of $k_BT$)")
    plt.title("PMF (in units of kT) with 1 sigma percent confidence intervals")

    # plot the timeseries of the parameters to check for equilibration
    plt.figure(4)
    samples = mc_results["samples"]
    lp, lt = np.shape(samples)
    for p in range(lp):
        plt.plot(np.arange(lt), samples[p, :], label=f"{p:d}_{method}")
    plt.title("Spline parameter time series")

    # print text results
    ci_results = pmf.get_confidence_intervals(bin_center_i, 16, 84)
    df = (ci_results["phigh"] - ci_results["plow"]) / 2
    print("PMF (in units of kT) with 1 sigma errors from posterior sampling")
    for i in range(nbins):
        print(f"{bin_center_i[i]:8.1f} {ci_results['values'][i]:8.1f} {df[i]:8.1f}")

    for i in range(len(pltname)):
        plt.figure(i + 1)
        plt.legend(fontsize="x-small")
        plt.savefig(f"{pltname[i]}_{fig_suffix}.pdf")
