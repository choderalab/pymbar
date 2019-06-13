# Example illustrating the application of MBAR to compute a 1D PMF from an umbrella sampling simulation.
#
# The data represents an umbrella sampling simulation for the chi torsion of a valine sidechain in lysozyme L99A with benzene bound in the cavity.
# 
# REFERENCE
# 
# D. L. Mobley, A. P. Graves, J. D. Chodera, A. C. McReynolds, B. K. Shoichet and K. A. Dill, "Predicting absolute ligand binding free energies to a simple model site," Journal of Molecular Biology 371(4):1118-1134 (2007).
# http://dx.doi.org/10.1016/j.jmb.2007.06.002

from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import copy
nplot = 1000
# set minimizer options to display. Apprently does not exist for BFGS. Probably don't need to set eps.
optimize_options = {'disp':True, 'tol':10**(-3)}
#methods = ['histogram','kde','kl-1','sumkl-1','simple-1','weighted-1','kl-3','sumkl-3','simple-3','weighted-3']
methods = ['histogram','kde','kl-3','weighted-3']
mc_methods = ['kl-3','weighted-3']
optimization_algorithm = 'L-BFGS-B'
#optimization_algorithm = 'Custom-NR'
#optimization_algorithm = 'Newton-CG'
colors = dict()
colors['histogram'] = 'k-'
colors['kde'] = 'k:'
colors['kl-1'] = 'g-'
colors['kl-3'] = 'm-'
colors['kl-5'] = 'c-'
colors['sumkl-1'] = 'g--'
colors['sumkl-3'] = 'm--'
colors['sumkl-5'] = 'c--'
colors['simple-1'] = 'g-.'
colors['simple-3'] = 'm-.'
colors['simple-5'] = 'c-.'
colors['weighted-1'] = 'g:'
colors['weighted-3'] = 'm:'
colors['weighted-5'] = 'c:'

# example illustrating the application of MBAR to compute a 1D PMF from an umbrella sampling simulation.
#
# The data represents an umbrella sampling simulation for the chi torsion of a valine sidechain in lysozyme L99A with benzene bound in the cavity.
# 
# REFERENCE
# 
# D. L. Mobley, A. P. Graves, J. D. Chodera, A. C. McReynolds, B. K. Shoichet and K. A. Dill, "Predicting absolute ligand binding free energies to a simple model site," Journal of Molecular Biology 371(4):1118-1134 (2007).
# http://dx.doi.org/10.1016/j.jmb.2007.06.002
import pdb
from timeit import default_timer as timer
import numpy as np # numerical array library
import pymbar # multistate Bennett acceptance ratio
from pymbar import timeseries # timeseries analysis
from pymbar import PMF
# Constants.
kB = 1.381e-23 * 6.022e23 / 1000.0 # Boltzmann constant in kJ/mol/K


temperature = 300 # assume a single temperature -- can be overridden with data from center.dat 
# Parameters
K = 26 # number of umbrellas
N_max = 501 # maximum number of snapshots/simulation
T_k = np.ones(K,float)*temperature # inital temperatures are all equal 
beta = 1.0 / (kB * temperature) # inverse temperature of simulations (in 1/(kJ/mol))
chi_min = -180.0 # min for PMF
chi_max = +180.0 # max for PMF
nbins = 40 # number of bins for 1D PMF. Note, does not have to correspond to the number of umbrellas at all.
nsplines = 15
nbootstraps = 0
mc_iterations = 1000

# Allocate storage for simulation data
N_k = np.zeros([K], np.int32) # N_k[k] is the number of snapshots from umbrella simulation k
K_k = np.zeros([K], np.float64) # K_k[k] is the spring constant (in kJ/mol/deg**2) for umbrella simulation k
chi0_k = np.zeros([K], np.float64) # chi0_k[k] is the spring center location (in deg) for umbrella simulation k
chi_kn = np.zeros([K,N_max], np.float64) # chi_kn[k,n] is the torsion angle (in deg) for snapshot n from umbrella simulation k
u_kn = np.zeros([K,N_max], np.float64) # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
g_k = np.zeros([K],np.float32);

# Read in umbrella spring constants and centers.
infile = open('data/centers.dat', 'r')
lines = infile.readlines()
infile.close()
for k in range(K):
    # Parse line k.
    line = lines[k]
    tokens = line.split()
    chi0_k[k] = float(tokens[0]) # spring center locatiomn (in deg)
    K_k[k] = float(tokens[1]) * (np.pi/180)**2 # spring constant (read in kJ/mol/rad**2, converted to kJ/mol/deg**2)    
    if len(tokens) > 2:
        T_k[k] = float(tokens[2])  # temperature the kth simulation was run at.

beta_k = 1.0/(kB*T_k)   # beta factor for the different temperatures
DifferentTemperatures = True
if (min(T_k) == max(T_k)):
    DifferentTemperatures = False            # if all the temperatures are the same, then we don't have to read in energies.
# Read the simulation data
for k in range(K):
    # Read torsion angle data.
    filename = 'data/prod%d_dihed.xvg' % k
    print("Reading {:s}...".format(filename))
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
    # Parse data.
    n = 0
    for line in lines:
        if line[0] != '#' and line[0] != '@':
            tokens = line.split()
            chi = float(tokens[1]) # torsion angle
            # wrap chi_kn to be within [-180,+180)
            while(chi < -180.0):
                chi += 360.0
            while(chi >= +180.0):
                chi -= 360.0
            chi_kn[k,n] = chi
            
            n += 1
    N_k[k] = n

    if (DifferentTemperatures):  # if different temperatures are specified the metadata file, 
                                 # then we need the energies to compute the PMF
        # Read energies
        filename = 'data/prod%d_energies.xvg' % k
        print("Reading {:s}...".format(filename))
        infile = open(filename, 'r')
        lines = infile.readlines()
        infile.close()
        # Parse data.
        n = 0
        for line in lines:
            if line[0] != '#' and line[0] != '@':
                tokens = line.split()            
                u_kn[k,n] = beta_k[k] * (float(tokens[2]) - float(tokens[1])) # reduced potential energy without umbrella restraint
                n += 1

    # Compute correlation times for potential energy and chi
    # timeseries.  If the temperatures differ, use energies to determine samples; otherwise, use the cosine of chi
            
    if (DifferentTemperatures):        
        g_k[k] = timeseries.statisticalInefficiency(u_kn[k,:], u_kn[k,0:N_k[k]])
        print("Correlation time for set {:5d} is {:10.3f}".format(k,g_k[k]))
        indices = timeseries.subsampleCorrelatedData(u_kn[k,0:N_k[k]])
    else:
        chi_radians = chi_kn[k,0:N_k[k]]/(180.0/np.pi)
        g_cos = timeseries.statisticalInefficiency(np.cos(chi_radians))
        g_sin = timeseries.statisticalInefficiency(np.sin(chi_radians))
        print("g_cos = {:.1f} | g_sin = {:.1f}".format(g_cos, g_sin))
        g_k[k] = max(g_cos, g_sin)
        print("Correlation time for set {:5d} is {:10.3f}".format(k,g_k[k]))
        indices = timeseries.subsampleCorrelatedData(chi_radians, g=g_k[k]) 
    # Subsample data.
    N_k[k] = len(indices)
    u_kn[k,0:N_k[k]] = u_kn[k,indices]
    chi_kn[k,0:N_k[k]] = chi_kn[k,indices]

N_max = np.max(N_k) # shorten the array size
u_kln = np.zeros([K,K,N_max], np.float64) # u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l

# Set zero of u_kn -- this is arbitrary.
u_kn -= u_kn.min()

# Construct torsion bins
# compute bin centers

bin_center_i = np.zeros([nbins], np.float64)
bin_edges = np.linspace(chi_min,chi_max,nbins+1)
for i in range(nbins):
    bin_center_i[i] = 0.5*(bin_edges[i] + bin_edges[i+1])

N = np.sum(N_k)
x_n = np.zeros(N,np.int32)
chi_n = pymbar.utils.kn_to_n(chi_kn, N_k = N_k)

ntot = 0
for k in range(K):
    for n in range(N_k[k]):
        # Compute bin assignment.
        x_n[ntot] = chi_kn[k,n]
        ntot +=1

# Evaluate reduced energies in all umbrellas
print("Evaluating reduced potential energies...")
for k in range(K):
    for n in range(N_k[k]):
        # Compute minimum-image torsion deviation from umbrella center l
        dchi = chi_kn[k,n] - chi0_k
        for l in range(K):
            if (abs(dchi[l]) > 180.0):
                dchi[l] = 360.0 - abs(dchi[l])

        # Compute energy of snapshot n from simulation k in umbrella potential l
        u_kln[k,:,n] = u_kn[k,n] + beta_k[k] * (K_k/2.0) * dchi**2

# initialize PMF with the data collected.
basepmf = pymbar.PMF(u_kln, N_k, verbose = True)

# define the bias potentials needed for umbrella sampling.
def bias_potential(x,k):
    dchi = x - chi0_k[k]
    # vectorize the conditional
    i = np.fabs(dchi) > 180.0
    dchi = i*(360.0 - np.fabs(dchi)) + (1-i)*dchi
    return beta_k[k]* (K_k[k] /2.0) * dchi**2

times = dict() # keep track of time elaped each method takes

xplot = np.linspace(chi_min,chi_max,nplot)  # number of points we are plotting
f_i_kde = None # We check later if these have been defined or not
xstart = np.linspace(chi_min,chi_max,nsplines*2) # the data we used initially to parameterize the splines.

pmfs = dict()
for method in methods:
    start = timer()

    # create a fresh copy of the initialized pmf object. Operate on that within the loop.
    # do the deepcopy here since there seem to be issues if it's done after data is added 
    # For example, the scikit-learn kde object fails to deepopy.

    pmfs[method] = copy.deepcopy(basepmf)
    pmf = pmfs[method]

    if method == 'histogram':

        histogram_parameters = dict()
        histogram_parameters['bin_edges'] = [bin_edges]
        pmf.generatePMF(u_kn, chi_n, pmf_type = 'histogram', histogram_parameters=histogram_parameters, nbootstraps=nbootstraps)

    if method == 'kde':

        kde_parameters = dict()
        kde_parameters['bandwidth'] = 0.5*((chi_max-chi_min)/nbins)
        pmf.generatePMF(u_kn, chi_n, pmf_type = 'kde', kde_parameters=kde_parameters, nbootstraps=nbootstraps)

        # save this for initializing other types
        results = pmf.getPMF(xstart, uncertainties = 'from-lowest')
        f_i_kde = results['f_i']  # kde results

    if method[:2] == 'kl' or method[:5] == 'sumkl' or  method[:8] == 'weighted' or method[:6] == 'simple':
        spline_parameters = dict()
        if method[:2] == 'kl':
            spline_parameters['spline_weights'] = 'kldivergence'
        if method[:5] == 'sumkl':
            spline_parameters['spline_weights'] = 'sumkldivergence'
        if method[:8] == 'weighted':
            spline_parameters['spline_weights'] = 'weightedsum'
        if method[:6] == 'simple':
            spline_parameters['spline_weights'] = 'simplesum'

        spline_parameters['nspline'] = nsplines 
        spline_parameters['spline_initialize'] = 'explicit'

        # need to initialize: use KDE results for now (assumes KDE exists)
        spline_parameters['xinit'] = xstart
        if f_i_kde is not None:
            spline_parameters['yinit'] = f_i_kde
        else:
            spline_parameters['yinit'] = np.zeros(len(xstart))

        spline_parameters['xrange'] = [chi_min,chi_max]

        spline_parameters['fkbias'] = [(lambda x, klocal=k: bias_potential(x,klocal)) for k in range(K)]  # introduce klocal to force K to use local definition of K, otherwise would use global value of k.

        spline_parameters['kdegree'] = int(method[-1])
        spline_parameters['optimization_algorithm'] = optimization_algorithm
        spline_parameters['optimize_options'] = optimize_options
        pmf.generatePMF(u_kn, chi_n, pmf_type = 'spline', spline_parameters=spline_parameters, nbootstraps=nbootstraps)

        if method in ['kl-3','weighted-3']:
            print(pmf.bspline.c)

    end = timer()
    times[method] = end-start

    yout = dict()
    yerr = dict()
    print("PMF (in units of kT) for {:s}".format(method))
    print("{:8s} {:8s} {:8s}".format('bin', 'f', 'df'))
    results = pmf.getPMF(bin_center_i, uncertainties = 'from-lowest')
    for i in range(nbins):
        if results['df_i'] is not None:
            print("{:8.1f} {:8.1f} {:8.1f}".format(bin_center_i[i], results['f_i'][i], results['df_i'][i]))
        else:
            print("{:8.1f} {:8.1f}".format(bin_center_i[i], results['f_i'][i]))

    results = pmf.getPMF(xplot, uncertainties = 'from-lowest')
    yout[method] = results['f_i']
    yerr[method] = results['df_i']
    if len(xplot) <= 50:
        errorevery = 1
    else:
        errorevery = np.floor(len(xplot)/50)

    plt.errorbar(xplot,yout[method],yerr=yerr[method],errorevery=errorevery,label=method,fmt=colors[method])

    if method not in ['histogram','kde']:
        print("AIC for {:s} with {:d} nsplines is: {:f}".format(method, nsplines, pmf.getInformationCriteria(type='AIC')))
        print("BIC for {:s} with {:d} nsplines is: {:f}".format(method, nsplines, pmf.getInformationCriteria(type='BIC')))

plt.xlim([chi_min,chi_max])
plt.legend()
plt.savefig('compare_pmf_{:d}.pdf'.format(nsplines))

for method in methods:
    print("time for method {:s} is {:2f} s".format(method,times[method]))

#import this for defining the prior
from scipy.stats import multivariate_normal
def deltag(c,scalef=500,n=nsplines):
    # we want to impose a smoothness prior. So we want all differences to be chosen from a Gaussian distribution.
    # looking at the preliminary results, something changing by 15 over 60 degrees is common.  So we want this to have 
    # reasonable probability. 60 degrees is 1/6 of the range.  So our possible rate of change is 15/(1/6) = 90 over the range
    # The amount changed per spline coefficient will be roughly 90/nspline.  We want this degree of curvature to have relatively 
    # little penalty, so we'll make this ~sigma/6.  So sigma/6 = 90/nsplines, sigma \approx 500/nsplines
    # we twiddled around so that the maximum likelihood was within the range . . . 
    cdiff = np.diff(c)
    logp = multivariate_normal.logpdf([cdiff],mean=None,cov=(scalef/n)*np.eye(len(cdiff))) # could be made more efficient, not worth it.
    return logp

for method in mc_methods:

    pmf = pmfs[method]
    print(pmf.bspline.c)
    if mc_iterations == 0:
        break
    mc_parameters = {"niterations":mc_iterations, "fraction_change":0.05, "sample_every": 10, 
                 "logprior": lambda x: deltag(x),"print_every":500}
    
    pmf.sampleParameterDistribution(chi_n, mc_parameters = mc_parameters, decorrelate = True) 

    mc_results = pmf.getMCData()

    plt.figure(1)
    plt.hist(mc_results['logposteriors'])
    plt.savefig('bayes_posterior_histogram_n{:d}.pdf'.format(nsplines))

    plt.figure(2)
    CI_results = pmf.getConfidenceIntervals(xplot,2.5,97.5,reference='zero')
    ylow = CI_results['plow']
    yhigh = CI_results['phigh']
    plt.plot(xplot,CI_results['values'],colors[method],label=method)
    plt.fill_between(xplot,ylow,yhigh,color=colors[method][0],alpha=0.3)
    plt.title('95 percent confidence intervals')
    plt.legend()
    plt.savefig('bayesian_95p_n{:d}.pdf'.format(nsplines))

    plt.figure(3)
    CI_results = pmf.getConfidenceIntervals(xplot,16,84)
    plt.clf()
    plt.plot(xplot,CI_results['values'],colors[method],label=method)
    ylow = CI_results['plow']
    yhigh = CI_results['phigh']
    plt.fill_between(xplot,ylow,yhigh,color=colors[method][0],alpha=0.3)
    plt.title('1 sigma percent confidence intervals')
    plt.savefig('bayesian_1sigma_n{:d}.pdf'.format(nsplines))

    CI_results = pmf.getConfidenceIntervals(bin_center_i,16,84)
    df = (CI_results['phigh']-CI_results['plow'])/2
    print("PMF (in units of kT) with 1 sigma errors from posterior sampling for {:s}".format(method))
    for i in range(nbins):
        print("{:8.1f} {:8.1f} {:8.1f}".format(bin_center_i[i], CI_results['values'][i], df[i]))

    plt.figure(4)
    plt.clf()
    samples = mc_results['samples']
    [lp,lt] = np.shape(samples)
    for p in range(lp):
        plt.plot(np.arange(lt),samples[p,:],label="{:d}_{:s}".format(p,method))
        plt.savefig("parameter_time_series_n{:d}.pdf".format(nsplines))



