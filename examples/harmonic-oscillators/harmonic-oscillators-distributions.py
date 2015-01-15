#!/usr/bin/python

#=============================================================================================
# Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators, for which
# the true free energy differences can be computed analytically.
#
# A number of replications of an experiment in which i.i.d. samples are drawn from a set of
# K harmonic oscillators are produced.  For each replicate, we estimate the dimensionless free
# energy differences and mean-square displacements (an observable), as well as their uncertainties.
#
# For a 1D harmonic oscillator, the potential is given by
#   V(x;K) = (K/2) * (x-x_0)**2
# where K denotes the spring constant.
#
# The equilibrium distribution is given analytically by
#   p(x;beta,K) = sqrt[(beta K) / (2 pi)] exp[-beta K (x-x_0)**2 / 2]
# The dimensionless free energy is therefore
#   f(beta,K) = - (1/2) * ln[ (2 pi) / (beta K) ]
#
#=============================================================================================

#=============================================================================================

#=============================================================================================
# IMPORTS
#=============================================================================================
import numpy
from pymbar import testsystems, MBAR, confidenceintervals
#=============================================================================================
# PARAMETERS
#=============================================================================================

K_k = numpy.array([25, 16, 9, 4, 1, 1]) # spring constants for each state
O_k = numpy.array([0, 1, 2, 3, 4, 5]) # offsets for spring constants
N_k = numpy.array([2000, 2000, 2000, 2000, 2000, 0]) # number of samples from each state (can be zero for some states)
beta = 1.0 # inverse temperature for all simulations
nreplicates = 200 # number of replicates of experiment for testing uncertainty estimate

generateplots = True
if (generateplots):
  try:
    import matplotlib.pyplot as plt
  except:
    print "Can't import matplotlib, will not produce graphs."
    generateplots = False

observe = 'position^2' # the observable, one of 'mean square displacement','position', or 'potential energy'

# Uncomment the following line to seed the random number generated to produce reproducible output.
numpy.random.seed(0)

#=============================================================================================
# MAIN
#=============================================================================================

# Determine number of simulations.
K = numpy.size(N_k)
if numpy.shape(K_k) != numpy.shape(N_k): raise "K_k and N_k must have same dimensions."

# Determine maximum number of samples to be drawn for any state.
N_max = numpy.max(N_k)

# Compute widths of sampled distributions.
# For a harmonic oscillator with spring constant K,
# x ~ Normal(x_0, sigma^2), where sigma = 1/sqrt(beta K)
sigma_k = (beta * K_k)**-0.5
print "Gaussian widths:"
print sigma_k

# Compute the absolute dimensionless free energies of each oscillator analytically.
# f = - ln(sqrt((2 pi)/(beta K)) )
print 'Computing dimensionless free energies analytically...'
f_k_analytical = - numpy.log(numpy.sqrt(2 * numpy.pi) * sigma_k )

# Compute true free energy differences.
Deltaf_ij_analytical = numpy.zeros([K,K], dtype = numpy.float64)
for i in range(0,K):
  for j in range(0,K):
    Deltaf_ij_analytical[i,j] = f_k_analytical[j] - f_k_analytical[i]

# Compute ensemble averages analytically 
if observe == 'RMS displacement':
  A_k_analytical = sigma_k              # mean square displacement
elif observe == 'potential energy':
  A_k_analytical = 1/(2*beta)*numpy.ones([K],float)  # By eqipartition
elif observe == 'position': 
  A_k_analytical  = O_k                 # observable is the position
elif observe == 'position^2': 
  A_k_analytical  = (1+ beta*K_k*O_k**2)/(beta*K_k)                 # observable is the position^2
else:
  raise "Observable %s not known." % observe

# DEBUG info
print "This script will perform %d replicates of an experiment where samples are drawn from %d harmonic oscillators." % (nreplicates, K)
print "The harmonic oscillators have equilibrium positions"
print O_k
print "and spring constants"
print K_k
print "and the following number of samples will be drawn from each (can be zero if no samples drawn):"
print N_k
print ""

# Conduct a number of replicates of the same experiment
replicates_observable = [] # storage for one hash for each replicate
replicates_standobservable = [] # storage for one hash for each replicate
replicates_df = [] # storage for one hash for each replicate
replicates_fdf = [] # storage for one hash for final observable

for replicate_index in range(0,nreplicates):
  print "Performing replicate %d / %d" % (replicate_index+1, nreplicates)

  # Initialize a hash to store data for this replicate.
  replicate_df = { }
  replicate_fdf = { }
  replicate_bar = { }
  replicate_observable = { }
  replicate_standobservable = { }

  #=============================================================================================
  # Generate independent data samples from K one-dimensional harmonic oscillators centered at q = 0.
  #=============================================================================================
  
  randomsample = testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k=O_k, K_k=K_k, beta=beta)
  [x_kn,u_kln,N_k] = randomsample.sample(N_k,mode='u_kln')

  # get the unreduced energies
  U_kln = u_kln/beta

  #=============================================================================================
  # Estimate free energies and expectations.
  #=============================================================================================

  # Initialize the MBAR class, determining the free energies.
  mbar = MBAR(u_kln, N_k, relative_tolerance=1.0e-10,verbose=False) # use fast Newton-Raphson solver
  (Deltaf_ij_estimated, dDeltaf_ij_estimated, _theta) = mbar.getFreeEnergyDifferences()
  
  # Compute error from analytical free energy differences.
  Deltaf_ij_error = Deltaf_ij_estimated - Deltaf_ij_analytical

  # Estimate the expectation of the mean-squared displacement at each condition.
  if observe == 'RMS displacement':
    A_kn = numpy.zeros([K,K,N_max], dtype = numpy.float64);
    for k in range(0,K):
      for l in range(0,K):
        A_kn[k,l,0:N_k[k]] = (x_kn[k,0:N_k[k]] - O_k[l])**2 # observable is the squared displacement

  # observable is the potential energy, a 3D array since the potential energy is a function of 
  # thermodynamic state
  elif observe == 'potential energy':
    A_kn = U_kln

  # observable for estimation is the position
  elif observe == 'position': 
    A_kn = numpy.zeros([K,N_max], dtype = numpy.float64)
    for k in range(0,K):
      A_kn[k,0:N_k[k]] = x_kn[k,0:N_k[k]]  

  elif observe == 'position^2': 
    A_kn = numpy.zeros([K,N_max], dtype = numpy.float64)
    for k in range(0,K):
      A_kn[k,0:N_k[k]] = x_kn[k,0:N_k[k]]**2   

  (A_k_estimated, dA_k_estimated) = mbar.computeExpectations(A_kn)
  As_k_estimated = numpy.zeros([K],numpy.float64)
  dAs_k_estimated = numpy.zeros([K],numpy.float64)

  # 'standard' expectation averages

  ifzero = numpy.array(N_k != 0)

  for k in range(K):
    if (ifzero[k]):
      if (observe == 'position') or (observe == 'position^2'):
        As_k_estimated[k] = numpy.average(A_kn[k,0:N_k[k]])
        dAs_k_estimated[k]  = numpy.sqrt(numpy.var(A_kn[k,0:N_k[k]])/(N_k[k]-1))
      elif (observe == 'RMS displacement' ) or (observe == 'potential energy'):
        As_k_estimated[k] = numpy.average(A_kn[k,k,0:N_k[k]])
        dAs_k_estimated[k]  = numpy.sqrt(numpy.var(A_kn[k,k,0:N_k[k]])/(N_k[k]-1))

  print A_k_estimated
  print dA_k_estimated

  # need to additionally transform to get the square root
  if observe == 'RMS displacement':
    A_k_estimated = numpy.sqrt(A_k_estimated)
    As_k_estimated = numpy.sqrt(As_k_estimated)    

    # Compute error from analytical observable estimate.
    dA_k_estimated = dA_k_estimated/(2*A_k_estimated)
    dAs_k_estimated = dAs_k_estimated/(2*As_k_estimated)

  A_k_error = A_k_estimated - A_k_analytical
  As_k_error = As_k_estimated - A_k_analytical

  #=============================================================================================
  # Store data for this replicate.
  #=============================================================================================  
  replicate_df['estimated'] = Deltaf_ij_estimated.copy()
  replicate_df['destimated'] = numpy.asarray(dDeltaf_ij_estimated.copy())
  replicate_df['error'] = Deltaf_ij_error.copy()
  replicates_df.append(replicate_df)

  replicate_observable['estimated'] = A_k_estimated.copy()
  replicate_observable['destimated'] = numpy.asarray(dA_k_estimated.copy())
  replicate_observable['error'] = A_k_error.copy()
  replicates_observable.append(replicate_observable)

  replicate_standobservable['estimated'] = As_k_estimated[ifzero].copy()
  replicate_standobservable['destimated'] = dAs_k_estimated[ifzero].copy()
  replicate_standobservable['error'] = As_k_error[ifzero].copy()
  replicates_standobservable.append(replicate_standobservable)


# compute the probability distribution of all states
print "Free energies"
# compute anderson/darling statistics

D = confidenceintervals.AndersonDarling(replicates_df,K)
print "Anderson-Darling Metrics (see README.md)"
print D
if (generateplots):
  confidenceintervals.QQPlot(replicates_df,K,title='Q-Q plots of free energy differences',filename="QQdf.pdf")
(alpha_fij,Pobs_fij,Plow_fij,Phigh_fij,dPobs_fij,Pnorm_fij) = confidenceintervals.generateConfidenceIntervals(replicates_df,K)

print "Standard ensemble averaged observables"
(alpha_Ai,Pobs_Ai,Plow_Ai,Phigh_Ai,dPobs_Ai,Pnorm_Ai) = confidenceintervals.generateConfidenceIntervals(replicates_standobservable,numpy.sum(ifzero))
D = confidenceintervals.AndersonDarling(replicates_standobservable,numpy.sum(ifzero))
print "Anderson-Darling Metrics (see README.md)"
print D
if (generateplots):
  confidenceintervals.QQPlot(replicates_standobservable,numpy.sum(ifzero),title='Q-Q plots of ensemble averaged observables \n with standard error estimates', filename="QQstandardobserve.pdf")

print "MBAR ensemble averaged observables"
(alpha_Ai,Pobs_Ai,Plow_Ai,Phigh_Ai,dPobs_Ai,Pnorm_Ai) = confidenceintervals.generateConfidenceIntervals(replicates_observable,K)
D = confidenceintervals.AndersonDarling(replicates_observable,K)
print "Anderson-Darling Metrics (see README.md)"
print D
if (generateplots):
  confidenceintervals.QQPlot(replicates_observable,K,title='Q-Q plots of ensemble averaged observables using MBAR', filename="QQMBARobserve.pdf")


if (generateplots):
    override = {
      'family'              : 'sans-serif',
      'verticalalignment'   : 'bottom',
      'horizontalalignment' : 'center',
      'weight'              : 'bold',
      'size'                : 30
      }
    
    formatstrings = ['b-','g-','c-','y-','r-','m-']

if (generateplots):    
    plt.figure(1);
    plt.axis([0.0, 4.0, 0.0, 1.0])
    plt.plot(alpha_fij,Pnorm_fij,'k-',label="Normal")

for k in range(1,K):
   replicates_fdf = []
   for replicate_ij in replicates_df:
     replicate = {}
     replicate['estimated'] = replicate_ij['estimated'][0,k]
     replicate['destimated'] = replicate_ij['destimated'][0,k]
     replicate['error'] = replicate_ij['error'][0,k]
     replicates_fdf.append(replicate)
   # compute the distribution of the end states only
   print ""
   print " ==== State %d alone with MBAR ===== " %(k)   
   (alpha_f,Pobs_f,Plow_f,Phigh_f,dPobs_f,Pnorm_f) = confidenceintervals.generateConfidenceIntervals(replicates_fdf,K);
   label = 'State %d' % k
   if (generateplots):
     plt.plot(alpha_f,Pobs_f,formatstrings[k-1],label=label)

if (generateplots):
  plt.title('Cumulative Probabilty vs. Normal Distribution',size=24)
  plt.xlabel('Standard Deviations',size = 18)
  plt.ylabel('Cumulative Probability',size= 18)
  plt.legend(loc=4)
  plt.savefig('cumulative_probability_comparison_curves.pdf')

if (generateplots):
  plt.figure(2);
  plt.axis([0.0, 4.0, 0.0, 1.0])
  plt.plot(alpha_fij,Pnorm_fij,'k-',label="Normal")

