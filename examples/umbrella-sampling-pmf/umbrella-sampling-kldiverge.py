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
#methods = ['kl-newton-1','kl-newton-3', 'sumkl-newton-1','sumkl-newton-3','kldiverge','sumkldiverge','vFEP']
methods = ['kde','kl-newton-1','kl-newton-3']

colors = dict()
colors['sumkldiverge'] = 'k-'
colors['kde'] = 'm-'
colors['vFEP'] = 'b-'
colors['sumkl-newton-1'] = 'g--'
colors['sumkl-newton-2'] = 'r--'
colors['sumkl-newton-3'] = 'c--'
colors['sumkl-newton-4'] = 'm--'
colors['sumkl-newton-5'] = 'y--'
colors['kldiverge'] = 'k-'
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
# Compute PMF in unbiased potential (in units of kT).
histogram_parameters = dict()
histogram_parameters['bin_edges'] = [bin_edges]
pmf.generatePMF(u_kn, chi_n, pmf_type = 'histogram', histogram_parameters=histogram_parameters)
results = pmf.getPMF(bin_center_i, uncertainties = 'from-lowest')
f_i = results['f_i']
df_i = results['df_i']

# NOW KDE:
kde_parameters = dict()
kde_parameters['bandwidth'] = 0.5*((chi_max-chi_min)/nbins)
pmf.generatePMF(u_kn, chi_n, pmf_type = 'kde', kde_parameters=kde_parameters)
results = pmf.getPMF(bin_center_i, uncertainties = 'from-lowest')
f_ik = results['f_i']

# Write out PMF
print("PMF (in units of kT)")
print("{:8s} {:8s} {:8s}".format('bin', 'f', 'df'))
for i in range(nbins):
    print("{:8.1f} {:8.1f} {:8.1f}".format(bin_center_i[i], f_i[i],f_ik[i], df_i[i]))

# get mbar ready
mbar = pmf.getMBAR()

# compute KL divergence to the empirical distribution for the trial distribution F
# convert angles to a single array.

################
from scipy.interpolate import BSpline, make_interp_spline, make_lsq_spline
from scipy.interpolate import interp1d
from scipy.integrate import quad, romb, romberg, quadrature
from scipy.optimize import minimize

smoother = 'spline'

nspline = nbins # set number of spline points. Set to the same number of bins for now.
verbose = True

# define the bias functions
def fbias(k,x):
    dchi = x - chi0_k[k]
    # vectorize the conditional
    i = np.fabs(dchi) > 180.0
    dchi = i*(360.0 - np.fabs(dchi)) + (1-i)*dchi
    return beta_k[k] * (K_k[k]/2.0) * dchi**2

# define functions that can change each iteration
if smoother == 'spline':
    xstart = np.linspace(chi_min,chi_max,nspline)
    def trialf(t):
        return interp1d(xstart, t, kind='cubic')
    tstart = 0*xstart
    for i in range(nspline): 
        # start with nearest PMF value. Hopefully helps convergence?
        tstart[i] = f_i[np.argmin(np.abs(bin_center_i-xstart[i]))]  

if smoother == 'periodic':  # doesn't work very well not using.
    # vary the magnitude, phase, and period
    def trialf(t):
        def interperiod(x):
            y = np.zeros(np.size(x))
            for i in range(nperiod):
                t[i+2*nperiod] = t[i+2*nperiod]%(360.0) # recenter the offsets, all in range
                y += t[i]*np.cos(t[i+nperiod]*x+t[i+2*nperiod])
            return y    
        return interperiod

    tstart = np.zeros(3*nperiod)
    # initial values of ampliudes, period, and phase
    tstart[0:nperiod] = 0
    d = (chi_max - chi_min)/(nperiod+1)
    tstart[nperiod:2*nperiod] = (2*np.pi/(chi_max-chi_min))
    tstart[2*nperiod:3*nperiod] = np.linspace(chi_min+d/2,chi_max-d/2,nperiod)
    
def kldiverge(t,ft,x_n,w_n,xrange):

    # define the function f based on the current parameters t
    t -= t[0] # set a reference state, may make the minization faster by removing degenerate solutions
    feval = ft(t) # current value of the PMF
    # define the exponential of f based on the current parameters t
    expf = lambda x: np.exp(-feval(x))
    pE = np.dot(w_n,feval(x_n))
    pF = np.log(quad(expf,xrange[0],xrange[1])[0])  #value 0 is the value of quad
    kl = pE + pF 
    print(kl, t, pE, pF)
    return kl

def sumkldiverge(t,ft,x_n,K,w_kn,fbias,xrange):
    t -= t[0] # set a reference state, may make the minization faster by removing degenerate solutions
    feval = ft(t)  # the current value of the PMF
    fx = feval(x_n)  # only need to evaluate this over all points outside(?)      
    kl = 0
    # figure out the bias
    for k in range(K):
        # what is the biasing function for this state
        # define the exponential of f based on the current parameters t.
        expf = lambda x: np.exp(-feval(x)-fbias(k,x))
        pE = np.dot(w_kn[:,k],fx+fbias(k,x_n))
        pF = np.log(quad(expf,xrange[0],xrange[1])[0])  #value 0 is the value of quad
        kl += (pE + pF)
    if verbose:
        print (kl,t)
    return kl

def vFEP(t,ft,x_kn,K,N_k,fbias,xrange):
    t -= t[0] # set a reference state, may make the minization faster by removing degenerate solutions
    feval = ft(t)  # the current value of the PMF
    kl = 0
    # figure out the bias
    for k in range(K):
        x_n = x_kn[k,0:N_k[k]]
        # what is the biasing function for this state
        # define the exponential of f based on the current parameters t.
        expf = lambda x: np.exp(-feval(x)-fbias(k,x))
        pE = np.sum(feval(x_n)+fbias(k,x_n))/N_k[k]
        pF = np.log(quad(expf,xrange[0],xrange[1])[0])  #0 is the value of quad
        kl += (pE + pF)
    if verbose:
        print(kl,t)
    return kl

x = np.linspace(chi_min,chi_max,nplot)
#plot the bin centers
plt.plot(bin_center_i,f_i,'rx',label='histogram')

isPeriodic = True # can use this for some spline functions.

times = dict()  # keep track of times that each methods take.

def calculate_f(w_n, x_n, b, method, K, xrange):

    # let's compute the value of the current function just to be careful.
    f = np.dot(w_n,b(x_n))
    pF = np.zeros(K)
    if 'sumkl' in method:
        for k in range(K):
            # what is the biasing function for this state?
            # define the biasing function 
            # define the exponential of f based on the current parameters t.
            expf = lambda x,k: np.exp(-b(x)-fbias(k,x))
            # compute the partition function
            pF[k] = quad(expf,xrange[0],xrange[1],args=(k))[0]  
            # subtract the free energy (add log partition function)
        f += np.sum(np.log(pF))

    else: # just KL divergence of the unbiased potential
        expf = lambda x: np.exp(-b(x))
        pF[0] = quad(expf,xrange[0],xrange[1])[0]  #0 is the value of quad
        # subtract the free energy (add log partition function)
        f += np.log(pF[0]) 

    print("function value to minimize: {:f}".format(f))

    return f, expf, pF  # return the value and the needed data (eventually move to class)

def calculate_g(w_n, x_n, b, db_c, expf, pF, method, K, xrangei):

    ##### COMPUTE THE GRADIENT #######  
    # The gradient of the function is \sum_n [\sum_k W_k(x_n)] dF(phi(x_n))/dtheta_i - \sum_k <dF/dtheta>_k 
    # 
    # where <O>_k = \int O(xi) exp(-F(xi) - u_k(xi)) dxi / \int exp(-F(xi) - u_k(xi)) dxi  

    g = np.zeros(nspline-1)
    for i in range(1,nspline):
        # compute the weighted sum of functions.
        g[i-1] += np.dot(w_n,db_c[i](x_n))

    # now the second part of the gradient.

    if 'sumkl' in method:
        gkquad = np.zeros([nspline-1,K])
        for k in range(K):
            for i in range(nspline-1):
                # Boltzmann weighted derivative with each biasing function
                dexpf = lambda x,k: db_c[i+1](x)*expf(x,k)
                # now compute the expectation of each derivative
                pE = quad(dexpf,xrangei[i+1,0],xrangei[i+1,1],args=(k))[0]
                # normalize the expectation
                gkquad[i,k] = pE/pF[k] 
        g -= np.sum(gkquad,axis=1)
        pE = 0

    else: # just doing a single one.
        gkquad = 0
        pE = np.zeros(nspline-1)
        for i in range(nspline-1):
            # Boltzmann weighted derivative
            dexpf = lambda x: db_c[i+1](x)*expf(x)
            # now compute the expectation of each derivative
            pE[i] = quad(dexpf,xrangei[i+1,0],xrangei[i+1,1])[0]
            # normalize the expetation.
            pE[i] /= pF[0]
            g[i] -= pE[i]

    dg2 = np.dot(g,g)
    print("gradient norm: {:.10f}".format(dg2))
    return g, dg2, gkquad, pE

def calculate_h(w_n, x_n, b, db_c, expf, pF, pE, method, K, Ki, xrangeij):

    # now, compute the Hessian.  First, the first order components
    h = np.zeros([nspline-1,nspline-1])
    if 'sumkl' in method:
        for k in range(K):
            h += -np.outer(gkquad[:,k],gkquad[:,k])
    else:
        h = -np.outer(pE,pE)
 
    # works for both sum and non-sum 
        
    if 'sumkl' in method:
        for i in range(nspline-1):
            for j in range(0,i+1):
                if np.abs(i-j) <= kdegree:
                    for k in range(Ki):
                        dexpf = lambda x,k: db_c[i+1](x)*db_c[j+1](x)*expf(x,k)
                        # now compute the expectation of each derivative
                        pE = quad(dexpf,xrangeij[i+1,j+1,0],xrangeij[i+1,j+1,1],args=(k))[0]
                        h[i,j] += pE/pF[k]
    else:
        for i in range(nspline-1):
            for j in range(0,i+1):
                if np.abs(i-j) <= kdegree:
                    for k in range(Ki):
                        dexpf = lambda x,k: db_c[i+1](x)*db_c[j+1](x)*expf(x)
                        # now compute the expectation of each derivative
                        pE = quad(dexpf,xrangeij[i+1,j+1,0],xrangeij[i+1,j+1,1],args=(k))[0]
                        h[i,j] += pE/pF[k]
  
    for i in range(nspline-1):  
        for j in range(i+1,nspline-1):
            h[i,j] = h[j,i]

    return h

def print_kde(pmf,x):
    results = pmf.getPMF(x, uncertainties = 'from-lowest')
    return results['f_i']

for method in methods:

    if 'newton' in method:
        kdegree = int(method[-1])

        start = timer() 
        x_kn = chi_kn
        xrange = [chi_min,chi_max]

        tol = 1e-10
        # first, construct a least squares cubic spline in the free energies to start with, set 2nd derivs zero.
        # we assume this is decent.

        # t has to be of size nsplines + kdegree + 1
        t = np.zeros(nspline+kdegree+1)
        t[0:kdegree] = xrange[0]
        t[kdegree:nspline+1] = np.linspace(xrange[0], xrange[1], num=nspline+1-kdegree, endpoint=True)
        t[nspline+1:nspline+kdegree+1] = xrange[1]

        # come up with an initial starting fit
        # This will be an overfit if the number of splines is too big.  
        # We'll have to interpolate in some other numbers to set an initial one.

        if nbins < 2*nspline:
            noverfit = int(np.round(nbins/2))
            tinit = np.zeros(noverfit+kdegree+1)
            tinit[0:kdegree] = xrange[0]
            tinit[kdegree:noverfit+1] = np.linspace(xrange[0], xrange[1], num=noverfit+1-kdegree, endpoint=True)
            tinit[noverfit+1:noverfit+kdegree+1] = xrange[1]
            binit = make_lsq_spline(bin_center_i, f_i, tinit, k=kdegree)
            xinit = np.linspace(xrange[0],xrange[1],num=2*nspline)
            yinit = binit(xinit)
        else:
            xinit = bin_center_i
            yinit = f_i
        
        # initial fit
        bfirst = make_lsq_spline(xinit, yinit, t, k=kdegree)
        xi = bfirst.c  # the bspline coefficients are the variables we care about.
        xold = xi.copy()

        # The function is \sum_n [\sum_k W_k(x_n)] F(phi(x_n)) + \sum_k ln \int exp(-F(xi) - u_k(xi)) dxi  
        # if we assume bsplines are of the form f(x) = a*b_i(x), then 
        # dF/dtheta is simply the basis function that has support over that region of space  

        b = bfirst
        # we now need the derivative of the function WRT the coefficients. Doesn't change knots or degree.
        # A vector function that is 
        db_c = list()
        for i in range(nspline):
            dc = np.zeros(nspline)
            dc[i] = 1.0
            db_c.append(BSpline(b.t,dc,b.k))
        # OK, we've defined the derivatives.  

        # we need the points the function is evaluated at.
        # define the x_n 
        x_n = np.zeros(np.sum(N_k))
        for k in range(K):
            nsum = np.sum(N_k[0:k])
            x_n[nsum:nsum + N_k[k]] = x_kn[k,0:N_k[k]]
            
        # we need these numbers for computing the function.
        if 'sumkl' in method:
            w_kn = np.exp(mbar.Log_W_nk) # normalized weights 
            w_n = np.sum(w_kn, axis=1) # sum along the w_kn
            Ki = K # the K we iterate over; makes it possible to do both in the same
        else:
            # if just kl divergence, we want the unnormalized probability
            log_w_n = mbar._computeUnnormalizedLogWeights(pymbar.utils.kn_to_n(u_kn, N_k = N_k))
            w_n = np.exp(log_w_n)
            w_n = w_n/np.sum(w_n)
            Ki = 1 # the K we iterate over; makes it possible to do both sum and nonsum in the same.

        # We also construct integration ranges for the derivatives, since no point in integrating when 
        # the function is zero.
        xrangei = np.zeros([nspline,2])
        for i in range(0,nspline):
            xrangei[i,0] = t[i]
            xrangei[i,1] = t[i+kdegree+1]

        # set integration ranges for derivative products; saves time on integration.
        xrangeij = np.zeros([nspline,nspline,2])
        for i in range(0,nspline):
            for j in range(0,nspline):
                xrangeij[i,j,0] = np.max([xrangei[i,0],xrangei[j,0]])
                xrangeij[i,j,1] = np.min([xrangei[i,1],xrangei[j,1]])

        dg2 = tol + 1.0
        firsttime = True

        while dg2 > tol: # until we reach the tolerance.

            f, expf, pF = calculate_f(w_n, x_n, b, method, K, xrange)

            # we need some error handling: if we stepped too far, we should go back
            if not firsttime:
                count = 0
                while f >= fold*(1.0+np.sqrt(tol)) and count < 5:   # we went too far!  Pull back.
                    f = fold
                    #let's not step as far:
                    dx = 0.5*dx
                    xi[1:nspline] = xold[1:nspline] - dx # step back half of dx.
                    xold = xi.copy()
                    print(xi)
                    b = BSpline(b.t,xi,b.k)
                    f, expf, pF = calculate_f(w_n, x_n, b, method, K, xrange)
                    count += 1
            else:
                firsttime = False
            fold = f
            xold = xi.copy()

            g, dg2, gkquad, pE = calculate_g(w_n, x_n, b, db_c, expf, pF, method, K, xrangei)

            h = calculate_h(w_n, x_n, b, db_c, expf, pF, pE, method, K, Ki, xrangeij)

            # now find the new point.
            # x_n+1 = x_n - f''(x_n)^-1 f'(x_n) 
            # which we solve more stably as:
            # x_n - x_n+1 = f''(x_n)^-1 f'(x_n)
            # f''(x_n)(x_n-x_n+1) = f'(x_n)  
            # solution is dx = x_n-x_n+1

            dx = np.linalg.lstsq(h,g)[0]
            xi[1:nspline] = xold[1:nspline] - dx
            b = BSpline(b.t,xi,b.k)

        pmf_final = b
        end = timer()
        times[method] = end-start

    elif method == 'kldiverge':
        start = timer()
        # inputs to minimize are:
        # the function that we are to minimize
        # the x values we have samples at
        # the weights at the samples
        # the domain of the function

        # Compute unnormalized log weights for the given reduced potential u_kn.
        log_w_n = mbar._computeUnnormalizedLogWeights(pymbar.utils.kn_to_n(u_kn, N_k = N_k))
        w_n = np.exp(log_w_n)
        w_n = w_n/np.sum(w_n)
        
        result = minimize(kldiverge,tstart,args=(trialf,chi_n,w_n,[chi_min,chi_max]),options=options)
        pmf_final = trialf(result.x)
        end = timer()
        times[method] = end-start

    elif method == 'sumkldiverge':
        start = timer()
        w_kn = np.exp(mbar.Log_W_nk) # normalized weights 

        # inputs to kldivergence in minimize are:
        # the function that we are computing the kldivergence of
        # the x values we have samples at
        # the number of umbrellas
        # the weights at the samples
        # the umbrella restraints strengths
        # the umbrella restraints centers
        # the domain of the function  

        result = minimize(sumkldiverge,tstart,args=(trialf,chi_n,K,w_kn,fbias,[chi_min,chi_max]),options=options)
        pmf_final = trialf(result.x)
        end = timer()
        times[method] = end-start

    elif method == 'vFEP':
        start = timer()
        result = minimize(vFEP,tstart,args=(trialf,chi_kn,K,N_k,fbias,[chi_min,chi_max]),options=options)
        pmf_final = trialf(result.x)
        end = timer()
        times[method] = end-start

    elif method == 'kde':
        start = timer()
        pmf_final = lambda x: print_kde(pmf,x)
        end = timer()
        times[method] = end-start

    # write out the PMF for this distribution
    yout = pmf_final(x)
    ymin = np.min(yout)
    yout -= ymin

    print("PMF (in units of kT) for {:s}".format(method))
    print("{:8s} {:8s} {:8s}".format('bin', 'f', 'df'))
    for i in range(nbins):
        print("{:8.1f} {:8.1f}".format(bin_center_i[i], float(pmf_final(bin_center_i[i])-ymin)))
    plt.plot(x,yout,colors[method],label=method)

plt.xlim([chi_min,chi_max])
plt.legend()
plt.savefig('compare_pmf_{:d}.pdf'.format(nspline))
for method in methods:
    print("time for method {:s} is {:2f} s".format(method,times[method]))
