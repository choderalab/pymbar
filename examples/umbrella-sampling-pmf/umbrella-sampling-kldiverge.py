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
nplot = 1000
# set minimizer options to display. Apprently does not exist for BFGS. Probably don't need to set eps.
options = {'disp':True, 'eps':10**(-4), 'gtol':10**(-3)}
#methods = ['histogram', 'kde', 'kl-scipy', 'sumkl-newton-1', 'kl-newton-1', 'kl-newton-3', 
#           'sumkl-newton-3', 'sumkl-scipy', 'vFEP-scipy']
#methods = ['histogram','kde','kl-newton-1','sumkl-newton-3','kl-newton-3','kl-scipy-3','sumkl-scipy-3','vFEP-scipy-3']
methods = ['histogram','kde','vFEP-scipy-3']

colors = dict()
colors['histogram'] = 'k:'
colors['sumkl-scipy-3'] = 'k-'
colors['kde'] = 'm-'
colors['vFEP-scipy-3'] = 'b-'
colors['sumkl-newton-1'] = 'g--'
colors['sumkl-newton-2'] = 'r--'
colors['sumkl-newton-3'] = 'c--'
colors['sumkl-newton-4'] = 'm--'
colors['sumkl-newton-5'] = 'y--'
colors['kl-scipy-3'] = 'k-'
colors['sumkl-scipy-3'] = 'k-'
colors['kl-newton-1'] = 'g-'
colors['kl-newton-2'] = 'r-'
colors['kl-newton-3'] = 'c-'
colors['kl-newton-4'] = 'm-'
colors['kl-newton-5'] = 'y-'

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
nsplines = 40

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

# Initialize histogram PMF for comparison:
#initialize PMF with the data collected
pmf = pymbar.PMF(u_kln, N_k, verbose = True)

# define the bias potentials needed for some method
def bias_potential(x,k):
    dchi = x - chi0_k[k]
    # vectorize the conditional
    i = np.fabs(dchi) > 180.0
    dchi = i*(360.0 - np.fabs(dchi)) + (1-i)*dchi
    return beta_k[k]* (K_k[k] /2.0) * dchi**2

times = dict() # keep track of times each method takes

xplot = np.linspace(chi_min,chi_max,nplot)
for method in methods:
    start = timer()

    if method == 'histogram':

        histogram_parameters = dict()
        histogram_parameters['bin_edges'] = [bin_edges]
        pmf.generatePMF(u_kn, chi_n, pmf_type = 'histogram', histogram_parameters=histogram_parameters)

    if method == 'kde':

        kde_parameters = dict()
        kde_parameters['bandwidth'] = 0.5*((chi_max-chi_min)/nbins)
        pmf.generatePMF(u_kn, chi_n, pmf_type = 'kde', kde_parameters=kde_parameters)

        # save this for initializing other types
        xstart = np.linspace(chi_min,chi_max,nsplines*2)
        results = pmf.getPMF(xstart, uncertainties = 'from-lowest')
        f_i_kde = results['f_i']  # kde results

    if 'newton' in method or 'scipy' in method:

        spline_parameters = dict()
        spline_parameters['nspline'] = nsplines
        spline_parameters['spline_initialize'] = 'explicit'
        # need to initialize newton
        spline_parameters['xinit'] = xstart
        spline_parameters['yinit'] = f_i_kde
        spline_parameters['xrange'] = [chi_min,chi_max]

        if method[:2] == 'kl':
            spline_parameters['spline_weights'] = 'kldivergence'
        if method[:5] == 'sumkl':
            spline_parameters['spline_weights'] = 'sumkldivergence'
        if method[:4] == 'vFEP':
            spline_parameters['spline_weights'] = 'vFEP'
            
        spline_parameters['fkbias'] = [(lambda x, klocal=k: bias_potential(x,klocal)) for k in range(K)]  # introduce klocal to force K to use local definition of K, otherwise would use global value of k.

        if 'newton' in method:

            spline_parameters['optimization_type'] = 'newton'
            newton_parameters = dict()
            kdegree = int(method[-1])
            newton_parameters['kdegree'] = kdegree
            newton_parameters['gtol'] = 1e-10

            spline_parameters['newton_parameters'] = newton_parameters 


        if 'scipy' in method:

            spline_parameters['optimization_type'] = 'scipy'
            scipy_parameters = dict()
            kdegree = int(method[-1])
            scipy_parameters['kdegree'] = kdegree
            spline_parameters['scipy_parameters'] = scipy_parameters

        pmf.generatePMF(u_kn, chi_n, pmf_type = 'spline', spline_parameters=spline_parameters)

    end = timer()
    times[method] = end-start

    print("PMF (in units of kT) for {:s}".format(method))
    print("{:8s} {:8s} {:8s}".format('bin', 'f', 'df'))
    results = pmf.getPMF(bin_center_i, uncertainties = 'from-lowest')
    for i in range(nbins):
        print("{:8.1f} {:8.1f}".format(bin_center_i[i], results['f_i'][i]))

    results = pmf.getPMF(xplot, uncertainties = 'from-lowest')
    yout = results['f_i']
    plt.plot(xplot,yout,colors[method],label=method)

plt.xlim([chi_min,chi_max])
plt.legend()
plt.savefig('compare_pmf_{:d}.pdf'.format(nsplines))

for method in methods:
    print("time for method {:s} is {:2f} s".format(method,times[method]))

