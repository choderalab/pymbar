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
# IMPORTS
#=============================================================================================
import sys
import numpy
from pymbar import testsystems, EXP, EXPGauss, BAR, MBAR

#=============================================================================================
# HELPER FUNCTIONS
#=============================================================================================

def stddev_away(namex,errorx,dx):

  if dx > 0:
    print "%s differs by %.3f standard deviations from analytical" % (namex,errorx/dx)
  else:
    print "%s differs by an undefined number of standard deviations" % (namex)

def GetAnalytical(beta,K,O,observables):

  # For a harmonic oscillator with spring constant K,
  # x ~ Normal(x_0, sigma^2), where sigma = 1/sqrt(beta K)

  # Compute the absolute dimensionless free energies of each oscillator analytically.
  # f = - ln(sqrt((2 pi)/(beta K)) )
  print 'Computing dimensionless free energies analytically...'

  sigma = (beta * K)**-0.5
  f_k_analytical = - numpy.log(numpy.sqrt(2 * numpy.pi) * sigma )

  Delta_f_ij_analytical = numpy.matrix(f_k_analytical) - numpy.matrix(f_k_analytical).transpose()

  A_k_analytical = dict()
  A_ij_analytical = dict()

  for observe in observables:
    if observe == 'RMS displacement':
      A_k_analytical[observe] = sigma                           # mean square displacement
    if observe == 'potential energy':
      A_k_analytical[observe] = 1/(2*beta)*numpy.ones(len(K),float)  # By equipartition
    if observe == 'position':  
      A_k_analytical[observe] = O                                       # observable is the position
    if observe == 'position^2':    
      A_k_analytical[observe]  = (1+ beta*K*O**2)/(beta*K)        # observable is the position^2

    A_ij_analytical[observe] = A_k_analytical[observe] - numpy.transpose(numpy.matrix(A_k_analytical[observe]))

  return f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical

#=============================================================================================
# PARAMETERS
#=============================================================================================

K_k = numpy.array([25, 16, 9, 4, 1, 1]) # spring constants for each state
O_k = numpy.array([0, 1, 2, 3, 4, 5]) # offsets for spring constants
N_k = 10*numpy.array([1000, 1000, 1000, 1000, 0, 1000]) # number of samples from each state (can be zero for some states)
Nk_ne_zero = (N_k!=0)
beta = 1.0 # inverse temperature for all simulations
K_extra = numpy.array([20, 12, 6, 2, 1]) 
O_extra = numpy.array([ 0.5, 1.5, 2.5, 3.5, 4.5])
observables = ['position','position^2','potential energy','RMS displacement']

seed = None
# Uncomment the following line to seed the random number generated to produce reproducible output.
seed = 0
numpy.random.seed(seed)

#=============================================================================================
# MAIN
#=============================================================================================

# Determine number of simulations.
K = numpy.size(N_k)
if numpy.shape(K_k) != numpy.shape(N_k): raise "K_k and N_k must have same dimensions."
if numpy.shape(O_k) != numpy.shape(N_k): raise "O_k and N_k must have same dimensions."

# Determine maximum number of samples to be drawn for any state.
N_max = numpy.max(N_k)

(f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical) = GetAnalytical(beta,K_k,O_k,observables)

print "This script will draw samples from %d harmonic oscillators." % (K)
print "The harmonic oscillators have equilibrium positions"
print O_k
print "and spring constants"
print K_k
print "and the following number of samples will be drawn from each (can be zero if no samples drawn):"
print N_k
print ""

#=============================================================================================
# Generate independent data samples from K one-dimensional harmonic oscillators centered at q = 0.
#=============================================================================================
  
print 'generating samples...'
randomsample = testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k=O_k, K_k=K_k, beta=beta)
[x_kn,u_kln,N_k] = randomsample.sample(N_k,mode='u_kln')

# get the unreduced energies
U_kln = u_kln/beta

#=============================================================================================
# Estimate free energies and expectations.
#=============================================================================================

print "======================================"
print "      Initializing MBAR               "
print "======================================"

# Estimate free energies from simulation using MBAR.
print "Estimating relative free energies from simulation (this may take a while)..."

# Initialize the MBAR class, determining the free energies.
mbar = MBAR(u_kln, N_k, relative_tolerance=1.0e-10, verbose=True)
# Get matrix of dimensionless free energy differences and uncertainty estimate.

print "============================================="
print "      Testing getFreeEnergyDifferences       "
print "============================================="

(Delta_f_ij_estimated, dDelta_f_ij_estimated, _Theta_ij) = mbar.getFreeEnergyDifferences()

# Compute error from analytical free energy differences.
Delta_f_ij_error = Delta_f_ij_estimated - Delta_f_ij_analytical

print "Error in free energies is:"
print Delta_f_ij_error
print "Uncertainty in free energies is:"
print dDelta_f_ij_estimated

print "Standard deviations away is:"
# mathematical manipulation to avoid dividing by zero errors; we don't care
# about the diagnonals, since they are identically zero.
df_ij_mod = dDelta_f_ij_estimated + numpy.identity(K)
stdevs = numpy.abs(Delta_f_ij_error/df_ij_mod)
for k in range(K):
  stdevs[k,k] = 0;
print stdevs

print "=============================================="
print "             Testing computeBAR               "
print "=============================================="

nonzero_indices = numpy.array(range(K))[Nk_ne_zero]
Knon = len(nonzero_indices)
for i in range(Knon-1):
  k = nonzero_indices[i]
  k1 = nonzero_indices[i+1]
  w_F = u_kln[k, k1, 0:N_k[k]]   - u_kln[k, k, 0:N_k[k]]       # forward work                                  
  w_R = u_kln[k1, k, 0:N_k[k1]] - u_kln[k1, k1, 0:N_k[k1]]    # reverse work                                  
  (df_bar,ddf_bar) = BAR(w_F, w_R)
  bar_analytical = (f_k_analytical[k1]-f_k_analytical[k]) 
  bar_error = bar_analytical - df_bar
  print "BAR estimator for reduced free energy from states %d to %d is %f +/- %f" % (k,k1,df_bar,ddf_bar) 
  stddev_away("BAR estimator",bar_error,ddf_bar)

print "=============================================="
print "             Testing computeEXP               "
print "=============================================="

print "EXP forward free energy"
for k in range(K-1):
  if N_k[k] != 0:
    w_F = u_kln[k, k+1, 0:N_k[k]]   - u_kln[k, k, 0:N_k[k]]       # forward work                                  
    (df_exp,ddf_exp) = EXP(w_F)
    exp_analytical = (f_k_analytical[k+1]-f_k_analytical[k]) 
    exp_error = exp_analytical - df_exp
    print "df from states %d to %d is %f +/- %f" % (k,k+1,df_exp,ddf_exp) 
    stddev_away("df",exp_error,ddf_exp)

print "EXP reverse free energy"
for k in range(1,K):
  if N_k[k] != 0:
    w_R = u_kln[k, k-1, 0:N_k[k]] - u_kln[k, k, 0:N_k[k]]         # reverse work                                  
    (df_exp,ddf_exp) = EXP(w_R)
    df_exp = -df_exp
    exp_analytical = (f_k_analytical[k]-f_k_analytical[k-1]) 
    exp_error = exp_analytical - df_exp
    print "df from states %d to %d is %f +/- %f" % (k,k-1,df_exp,ddf_exp) 
    stddev_away("df",exp_error,ddf_exp)

print "=============================================="
print "             Testing computeGauss               "
print "=============================================="

print "Gaussian forward estimate"
for k in range(K-1):
  if N_k[k] != 0:
    w_F = u_kln[k, k+1, 0:N_k[k]]   - u_kln[k, k, 0:N_k[k]]       # forward work                                  
    (df_gauss,ddf_gauss) = EXPGauss(w_F)
    gauss_analytical = (f_k_analytical[k+1]-f_k_analytical[k]) 
    gauss_error = gauss_analytical - df_gauss
    print "df for reduced free energy from states %d to %d is %f +/- %f" % (k,k+1,df_gauss,ddf_gauss) 
    stddev_away("df",gauss_error,ddf_gauss)

print "Gaussian reverse estimate"
for k in range(1,K):
  if N_k[k] != 0:
    w_R = u_kln[k, k-1, 0:N_k[k]] - u_kln[k, k, 0:N_k[k]]         # reverse work                                  
    (df_gauss,ddf_gauss) = EXPGauss(w_R)
    df_gauss = df_gauss
    gauss_analytical = (f_k_analytical[k]-f_k_analytical[k-1]) 
    gauss_error = gauss_analytical - df_gauss
    print "df for reduced free energy from states %d to %d is %f +/- %f" % (k,k-1,df_gauss,ddf_gauss) 
    stddev_away("df",gauss_error,ddf_gauss)
    
print "======================================"
print "      Testing computeExpectations"
print "======================================"

A_kn_all = dict()
A_k_estimated_all = dict()
A_kl_estimated_all = dict()
N = numpy.sum(N_k)

for observe in observables:
  print "============================================"
  print "      Testing observable %s" % (observe)
  print "============================================"

  if observe == 'RMS displacement':
    state_dependent = True
    A_kn = numpy.zeros([K,N], dtype = numpy.float64);
    n = 0
    for k in range(0,K):
      for nk in range(0,N_k[k]):
        A_kn[:,n] = (x_kn[k,nk] - O_k[:])**2 # observable is the squared displacement
        n += 1

  # observable is the potential energy, a 3D array since the potential energy is a function of 
  # thermodynamic state
  elif observe == 'potential energy':
    state_dependent = True
    A_kn = numpy.zeros([K,N], dtype = numpy.float64);
    n = 0
    for k in range(0,K):
      for nk in range(0,N_k[k]):
        A_kn[:,n] = U_kln[k,:,nk]
        n += 1
  
  # observable for estimation is the position
  elif observe == 'position':
    state_dependent = False
    A_kn = numpy.zeros([K,N_max], dtype = numpy.float64)
    for k in range(0,K):
      A_kn[k,0:N_k[k]] = x_kn[k,0:N_k[k]]

  # observable for estimation is the position^2
  elif observe == 'position^2':
    state_dependent = False
    A_kn = numpy.zeros([K,N_max], dtype = numpy.float64)
    for k in range(0,K):
      A_kn[k,0:N_k[k]] = x_kn[k,0:N_k[k]]**2

  (A_k_estimated, dA_k_estimated) = mbar.computeExpectations(A_kn, state_dependent = state_dependent)

  # need to additionally transform to get the square root
  if observe == 'RMS displacement':
    A_k_estimated = numpy.sqrt(A_k_estimated)
    # Compute error from analytical observable estimate.
    dA_k_estimated = dA_k_estimated/(2*A_k_estimated)

  As_k_estimated = numpy.zeros([K],numpy.float64)
  dAs_k_estimated = numpy.zeros([K],numpy.float64)

  # 'standard' expectation averages - not defined if no samples
  nonzeros = numpy.arange(K)[Nk_ne_zero]

  totaln = 0
  for k in nonzeros:
    if (observe == 'position') or (observe == 'position^2'):
      As_k_estimated[k] = numpy.average(A_kn[k,0:N_k[k]])
      dAs_k_estimated[k]  = numpy.sqrt(numpy.var(A_kn[k,0:N_k[k]])/(N_k[k]-1))
    elif (observe == 'RMS displacement' ) or (observe == 'potential energy'):
      totalp = totaln + N_k[k]
      As_k_estimated[k] = numpy.average(A_kn[k,totaln:totalp])
      dAs_k_estimated[k]  = numpy.sqrt(numpy.var(A_kn[k,totaln:totalp])/(N_k[k]-1))
      totaln = totalp
      if observe == 'RMS displacement':
        As_k_estimated[k] = numpy.sqrt(As_k_estimated[k])    
        dAs_k_estimated[k] = dAs_k_estimated[k]/(2*As_k_estimated[k])

  A_k_error = A_k_estimated - A_k_analytical[observe]
  As_k_error = As_k_estimated - A_k_analytical[observe]

  print "------------------------------"
  print "Now testing 'averages' mode"
  print "------------------------------"

  print "Analytical estimator of %s is" % (observe)
  print A_k_analytical[observe]

  print "MBAR estimator of the %s is" % (observe)
  print A_k_estimated

  print "MBAR estimators differ by X standard deviations"
  stdevs = numpy.abs(A_k_error/dA_k_estimated)
  print stdevs

  print "Standard estimator of %s is (states with samples):" % (observe)
  print As_k_estimated[Nk_ne_zero]

  print "Standard estimators differ by X standard deviations (states with samples)"
  stdevs = numpy.abs(As_k_error[Nk_ne_zero]/dAs_k_estimated[Nk_ne_zero])
  print stdevs

  (A_kl_estimated, dA_kl_estimated) = mbar.computeExpectations(A_kn, state_dependent = state_dependent, output = 'differences')

  print "------------------------------"
  print "Now testing 'differences' mode"
  print "------------------------------"

  if 'RMS displacement' != observe: # can't test this, because we're actually computing the expectation of
                                    # the mean square displacement, and so the differences are <a_i^2> - <a_j^2>,
                                    # not sqrt<a_i>^2 - sqrt<a_j>^2
    A_kl_analytical = numpy.matrix(A_k_analytical[observe]) - numpy.matrix(A_k_analytical[observe]).transpose()
    A_kl_error = A_kl_estimated - A_kl_analytical
    
    print "Analytical estimator of differences of %s is" % (observe)
    print A_kl_analytical
    
    print "MBAR estimator of the differences of %s is" % (observe)
    print A_kl_estimated
    
    print "MBAR estimators differ by X standard deviations"
    stdevs = numpy.abs(A_kl_error/(dA_kl_estimated+numpy.identity(K)))
    for k in range(K):
      stdevs[k,k] = 0;
    print stdevs

  # save up the A_k for use in computeMultipleExpectations
  A_kn_all[observe] = A_kn
  A_k_estimated_all[observe] = A_k_estimated
  A_kl_estimated_all[observe] = A_kl_estimated

print "============================================="
print "      Testing computeMultipleExpectations"
print "============================================="

# have to exclude the potential and RMS displacemet for now, not functions of a single state
observables_single = ['position','position^2']  

A_ikn = numpy.zeros([len(observables_single), K, N_k.max()], numpy.float64)
for i,observe in enumerate(observables_single):
  A_ikn[i,:,:] = A_kn_all[observe]
for i in range(K):
  [A_i,dA_ij,Ca_ij] = mbar.computeMultipleExpectations(A_ikn, u_kln[:,i,:], compute_covariance=True)
  print "Averages for state %d" % (i)
  print A_i
  print "Uncertainties for state %d" % (i)
  print dA_ij
  print "Correlation matrix between observables for state %d" % (i)
  print Ca_ij

print "============================================"
print "      Testing computeEntropyAndEnthalpy"
print "============================================"

(Delta_f_ij, dDelta_f_ij, Delta_u_ij, dDelta_u_ij, Delta_s_ij, dDelta_s_ij) = mbar.computeEntropyAndEnthalpy(u_kn = u_kln, verbose = True)
print "Free energies"
print Delta_f_ij
print dDelta_f_ij
diffs1 = Delta_f_ij - Delta_f_ij_estimated
print "maximum difference between values computed here and in computeFreeEnergies is %g" % (numpy.max(diffs1))
if (numpy.max(numpy.abs(diffs1)) > 1.0e-10):
  print "Difference in values from computeFreeEnergies"
  print diffs1
diffs2 = dDelta_f_ij - dDelta_f_ij_estimated
print "maximum difference between uncertainties computed here and in computeFreeEnergies is %g" % (numpy.max(diffs2))
if (numpy.max(numpy.abs(diffs2)) > 1.0e-10):
  print "Difference in expectations from computeFreeEnergies"
  print diffs2

print "Energies"
print Delta_u_ij
print dDelta_u_ij
U_k = numpy.matrix(A_k_estimated_all['potential energy'])
expectations = U_k - U_k.transpose()
diffs1 = Delta_u_ij - expectations
print "maximum difference between values computed here and in computeExpectations is %g" % (numpy.max(diffs1))
if (numpy.max(numpy.abs(diffs1)) > 1.0e-10):
  print "Difference in values from computeExpectations"
  print diffs1

print "Entropies"
print Delta_s_ij
print dDelta_s_ij

#analytical entropy estimate
s_k_analytical = numpy.matrix(0.5 / beta - f_k_analytical)
Delta_s_ij_analytical = s_k_analytical - s_k_analytical.transpose()

Delta_s_ij_error = Delta_s_ij_analytical - Delta_s_ij
print "Error in entropies is:"
print Delta_f_ij_error

print "Standard deviations away is:"
# mathematical manipulation to avoid dividing by zero errors; we don't care 
# about the diagnonals, since they are identically zero.
ds_ij_mod = dDelta_s_ij + numpy.identity(K)
stdevs = numpy.abs(Delta_s_ij_error/ds_ij_mod)
for k in range(K):
  stdevs[k,k] = 0;
print stdevs

print "============================================"
print "      Testing computePerturbedFreeEnergies"
print "============================================"

L = numpy.size(K_extra)
(f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical) = GetAnalytical(beta,K_extra,O_extra,observables)

if numpy.size(O_extra) != numpy.size(K_extra):
  raise "O_extra and K_extra mut have the same dimensions."

unew_kln = numpy.zeros([K,L,numpy.max(N_k)],numpy.float64)
for k in range(K):
    for l in range(L):
      unew_kln[k,l,0:N_k[k]] = (K_extra[l]/2.0) * (x_kn[k,0:N_k[k]]-O_extra[l])**2

(Delta_f_ij_estimated, dDelta_f_ij_estimated) = mbar.computePerturbedFreeEnergies(unew_kln)

Delta_f_ij_error = Delta_f_ij_estimated - Delta_f_ij_analytical

print "Error in free energies is:"
print Delta_f_ij_error

print "Standard deviations away is:"
# mathematical manipulation to avoid dividing by zero errors; we don't care
# about the diagnonals, since they are identically zero.
df_ij_mod = dDelta_f_ij_estimated + numpy.identity(L)
stdevs = numpy.abs(Delta_f_ij_error/df_ij_mod)
for l in range(L):
  stdevs[l,l] = 0;
print stdevs

print "============================================"
print "      Testing computeExpectation (new states)  "
print "============================================"

nth = 3
# test the nth "extra" states, O_extra[nth] & K_extra[nth]
for observe in observables:
  print "============================================"
  print "      Testing observable %s" % (observe)
  print "============================================"

  if observe == 'RMS displacement':
    state_dependent = True
    A_kn = numpy.zeros([K,1,N_max], dtype = numpy.float64);
    for k in range(0,K):
      A_kn[k,0,0:N_k[k]] = (x_kn[k,0:N_k[k]] - O_extra[nth])**2 # observable is the squared displacement

  # observable is the potential energy, a 3D array since the potential energy is a function of 
  # thermodynamic state
  elif observe == 'potential energy':
    state_dependent = True
    A_kn = unew_kln[:,[nth],:]/beta

  # position and position^2 can use the same observables  
  # observable for estimation is the position
  elif observe == 'position': 
    state_dependent = False
    A_kn = A_kn_all['position']

  elif observe == 'position^2': 
    state_dependent = False
    A_kn = A_kn_all['position^2']

  A_k_estimated, dA_k_estimated = mbar.computeExpectations(A_kn,unew_kln[:,[nth],:],state_dependent=state_dependent)

  # need to additionally transform to get the square root
  if observe == 'RMS displacement':
    A_k_estimated = numpy.sqrt(A_k_estimated)
    dA_k_estimated = dA_k_estimated/(2*A_k_estimated)

  A_k_error = A_k_estimated - A_k_analytical[observe][nth]

  print "Analytical estimator of %s is" % (observe)
  print A_k_analytical[observe][nth]

  print "MBAR estimator of the %s is" % (observe)
  print A_k_estimated

  print "MBAR estimators differ by X standard deviations"
  stdevs = numpy.abs(A_k_error/dA_k_estimated)
  print stdevs

print "============================================"
print "      Testing computeOverlap   "
print "============================================"

O, O_i, O_ij = mbar.computeOverlap()

print "Overlap matrix output"
print O_ij

for k in range(K):
  print "Sum of row %d is %f (should be 1)," % (k,numpy.sum(O_ij[k,:])),
  if (numpy.abs(numpy.sum(O_ij[k,:])-1)<1.0e-10):
    print "looks like it is."
  else:
    print "but it's not."

print "Overlap eigenvalue output"
print O_i


print "Overlap scalar output"
print O

print "============================================"
print "    Testing computeEffectiveSampleNumber    "
print "============================================"

N_eff = mbar.computeEffectiveSampleNumber(verbose = True)
print "Effective Sample number"
print N_eff
print "Compare stanadrd estimate of <x> with the MBAR estimate of <x>"
print "We should have that with MBAR, err_MBAR = sqrt(N_k/N_eff)*err_standard,"
print "so standard (scaled) results should be very close to MBAR results."
print "No standard estimate exists for states that are not sampled."
A_kn = x_kn
(val_mbar, err_mbar) = mbar.computeExpectations(A_kn)
err_standard = numpy.zeros([K],dtype = numpy.float64)
err_scaled = numpy.zeros([K],dtype = numpy.float64)

for k in range(K):
  if N_k[k] != 0:
    # use position
    err_standard[k] = numpy.std(A_kn[k,0:N_k[k]])/numpy.sqrt(N_k[k]-1)
    err_scaled[k] = numpy.std(A_kn[k,0:N_k[k]])/numpy.sqrt(N_eff[k]-1)

print "                    ",
for k in range(K):
  print "       %d   " %(k),
print ""
print "MBAR             :",
print err_mbar
print "standard         :",
print err_standard
print "sqrt N_k/N_eff   :",
print numpy.sqrt(N_k/N_eff)
print "Standard (scaled):",
print err_standard * numpy.sqrt(N_k/N_eff)

print "============================================"
print "      Testing computePMF   "
print "============================================"

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
#   f(beta,K)           = - ln [ (2*numpy.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
#   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))

def generate_pmf_data(ndim=1, nbinsperdim=15, nsamples = 1000, K0=20.0, Ku = 100.0, gridscale=0.2, xrange = [[-3,3]]):

  x0 = numpy.zeros([ndim], numpy.float64) # center of base potential
  numbrellas = 1
  nperdim = numpy.zeros([ndim],int)
  for d in range(ndim):
    nperdim[d] = xrange[d][1] - xrange[d][0] + 1
    numbrellas *= nperdim[d]

  print "There are a total of %d umbrellas." % numbrellas

  # Enumerate umbrella centers, and compute the analytical free energy of that umbrella
  print "Constructing umbrellas..."
  ksum = (Ku+K0)/beta
  kprod = (Ku*K0)/(beta*beta)
  f_k_analytical = numpy.zeros(numbrellas, numpy.float64);
  xu_i = numpy.zeros([numbrellas, ndim], numpy.float64) # xu_i[i,:] is the center of umbrella i

  dp = numpy.zeros(ndim,int)
  dp[0] = 1
  for d in range(1,ndim):
    dp[d] = nperdim[d]*dp[d-1]

  umbrella_zero = 0
  for i in range(numbrellas):
    center = []
    for d in range(ndim):
      val = gridscale*((i/dp[d]) % nperdim[d] + xrange[d][0])
      center.append(val)
    center = numpy.array(center)
    xu_i[i,:] = center
    mu2 = numpy.dot(center,center)
    f_k_analytical[i] = numpy.log((ndim*numpy.pi/ksum)**(3/2) *numpy.exp(-kprod*mu2/(2*ksum)))
    if numpy.all(center==0.0):  # assumes that we have one state that is at the zero.
      umbrella_zero = i
    i += 1
    f_k_analytical -= f_k_analytical[umbrella_zero]

  print "Generating %d samples for each of %d umbrellas..." % (nsamples, numbrellas)
  x_n = numpy.zeros([numbrellas * nsamples, ndim], numpy.float64)

  for i in range(numbrellas):
    for dim in range(ndim):
      # Compute mu and sigma for this dimension for sampling from V0(x) + Vu(x).
      # Product of Gaussians: N(x ; a, A) N(x ; b, B) = N(a ; b , A+B) x N(x ; c, C) where
      # C = 1/(1/A + 1/B)
      # c = C(a/A+b/B)
      # A = 1/K0, B = 1/Ku
      sigma = 1.0 / (K0 + Ku)
      mu = sigma * (x0[dim]*K0 + xu_i[i,dim]*Ku)
      # Generate normal deviates for this dimension.
      x_n[i*nsamples:(i+1)*nsamples,dim] = numpy.random.normal(mu, numpy.sqrt(sigma), [nsamples])

  u_kn = numpy.zeros([numbrellas, nsamples*numbrellas], numpy.float64)
  # Compute reduced potential due to V0.
  u_n = beta*(K0/2)*numpy.sum((x_n[:,:] - x0)**2, axis=1)
  for k in range(numbrellas):
    uu = beta*(Ku/2)*numpy.sum((x_n[:,:] - xu_i[k,:])**2, axis=1) # reduced potential due to umbrella k
    u_kn[k,:] = u_n + uu

  return u_kn, u_n, x_n, f_k_analytical

nbinsperdim = 15
gridscale = 0.2
nsamples = 1000
ndim = 1
K0 = 20.0
Ku = 100.0
print "============================================"
print "      Test 1: 1D PMF   "
print "============================================"

xrange = [[-3,3]]
ndim = 1

u_kn, u_n, x_n, f_k_analytical = generate_pmf_data(K0 = K0, Ku = Ku, ndim=ndim, nbinsperdim = nbinsperdim, nsamples = nsamples, gridscale = gridscale, xrange=xrange)
numbrellas = (numpy.shape(u_kn))[0]
N_k = nsamples*numpy.ones([numbrellas], int)
print "Solving for free energies of state ..."
mbar = MBAR(u_kn, N_k)

# Histogram bins are indexed using the scheme:
# index = 1 + numpy.floor((x[0] - xmin)/dx) + nbins*numpy.floor((x[1] - xmin)/dy)
# index = 0 is reserved for samples outside of the allowed domain
xmin = gridscale*(numpy.min(xrange[0][0])-1/2.0)
xmax = gridscale*(numpy.max(xrange[0][1])+1/2.0)
dx = (xmax-xmin)/nbinsperdim
nbins = 1 + nbinsperdim**ndim
bin_centers = numpy.zeros([nbins,ndim],numpy.float64)

ibin = 1;
pmf_analytical = numpy.zeros([nbins],numpy.float64)
minmu2 = 1000000;
zeroindex = 0;
# construct the bins and the pmf
for i in range(nbinsperdim):
  xbin = xmin + dx * (i + 0.5)
  bin_centers[ibin,0] = xbin
  mu2 = xbin*xbin
  if (mu2 < minmu2):
    minmu2 = mu2;
    zeroindex = ibin
  pmf_analytical[ibin] = K0*mu2/2.0
  ibin += 1
fzero = pmf_analytical[zeroindex]
pmf_analytical -= fzero
pmf_analytical[0] = 0

bin_n = numpy.zeros([numbrellas*nsamples], int)
# Determine indices of those within bounds.
within_bounds = (x_n[:,0] >= xmin) & (x_n[:,0] < xmax)
# Determine states for these.
bin_n[within_bounds] = 1 + numpy.floor((x_n[within_bounds,0]-xmin)/dx)
# Determine indices of bins that are not empty.
bin_counts = numpy.zeros([nbins], int)
for i in range(nbins):
  bin_counts[i] = (bin_n == i).sum()

# Compute PMF.
print "Computing PMF ..."
[f_i, df_i] = mbar.computePMF(u_n, bin_n, nbins, uncertainties = 'from-specified', pmf_reference = zeroindex)
# Show free energy and uncertainty of each occupied bin relative to lowest free energy

print "1D PMF:"
print "%d counts out of %d counts not in any bin" % (bin_counts[0],numbrellas*nsamples)
print "%8s %6s %8s %10s %10s %10s %10s %8s" % ('bin', 'x', 'N', 'f', 'true','error','df','sigmas')
for i in range(1,nbins):
   if (i == zeroindex):
     stdevs = 0 
     df_i[0] = 0
   else:
     error = pmf_analytical[i]-f_i[i]
     stdevs = numpy.abs(error)/df_i[i]
   print '%8d %6.2f %8d %10.3f %10.3f %10.3f %10.3f %8.2f' % (i, bin_centers[i,0], bin_counts[i], f_i[i], pmf_analytical[i], error, df_i[i], stdevs)

print "============================================"
print "      Test 2: 2D PMF   "
print "============================================"

xrange = [[-3,3],[-3,3]]
ndim = 2
nsamples = 300
u_kn, u_n, x_n, f_k_analytical = generate_pmf_data(K0 = K0, Ku = Ku, ndim=ndim, nbinsperdim = nbinsperdim, nsamples = nsamples, gridscale = gridscale, xrange=xrange)
numbrellas = (numpy.shape(u_kn))[0]
N_k = nsamples*numpy.ones([numbrellas], int)
print "Solving for free energies of state ..."
mbar = MBAR(u_kn, N_k)

# The dimensionless free energy is the integral of this, and can be computed as:
#   f(beta,K)           = - ln [ (2*numpy.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
#   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))
# for computing harmonic samples

#Can compare the free energies computed with MBAR if desired with f_k_analytical

# Histogram bins are indexed using the scheme:
# index = 1 + numpy.floor((x[0] - xmin)/dx) + nbins*numpy.floor((x[1] - xmin)/dy)
# index = 0 is reserved for samples outside of the allowed domain

xmin = gridscale*(numpy.min(xrange[0][0])-1/2.0)
xmax = gridscale*(numpy.max(xrange[0][1])+1/2.0)
ymin = gridscale*(numpy.min(xrange[1][0])-1/2.0)
ymax = gridscale*(numpy.max(xrange[1][1])+1/2.0)
dx = (xmax-xmin)/nbinsperdim
dy = (ymax-ymin)/nbinsperdim
nbins = 1 + nbinsperdim**ndim
bin_centers = numpy.zeros([nbins,ndim],numpy.float64)

ibin = 1;
pmf_analytical = numpy.zeros([nbins],numpy.float64)
minmu2 = 1000000;
zeroindex = 0;
# construct the bins and the pmf
for i in range(nbinsperdim):
  xbin = xmin + dx * (i + 0.5)
  for j in range(nbinsperdim):
    # Determine (x,y) of bin center.
    ybin = ymin + dy * (j + 0.5)
    bin_centers[ibin,0] = xbin
    bin_centers[ibin,1] = ybin
    mu2 = xbin*xbin+ybin*ybin
    if (mu2 < minmu2):
      minmu2 = mu2;
      zeroindex = ibin
    pmf_analytical[ibin] = K0*mu2/2.0 
    ibin += 1
fzero = pmf_analytical[zeroindex]
pmf_analytical -= fzero
pmf_analytical[0] = 0

bin_n = numpy.zeros([numbrellas * nsamples], int)
# Determine indices of those within bounds.
within_bounds = (x_n[:,0] >= xmin) & (x_n[:,0] < xmax) & (x_n[:,1] >= ymin) & (x_n[:,1] < ymax)
# Determine states for these.
bin_n[within_bounds] = 1 + numpy.floor((x_n[within_bounds,0]-xmin)/dx) + nbinsperdim*numpy.floor((x_n[within_bounds,1]-ymin)/dy)

# Determine indices of bins that are not empty.
bin_counts = numpy.zeros([nbins], int)
for i in range(nbins):                             
  bin_counts[i] = (bin_n == i).sum()

# Compute PMF.          
print "Computing PMF ..."
[f_i, df_i] = mbar.computePMF(u_n, bin_n, nbins, uncertainties = 'from-specified', pmf_reference = zeroindex)
# Show free energy and uncertainty of each occupied bin relative to lowest free energy
print "2D PMF:"

print "%d counts out of %d counts not in any bin" % (bin_counts[0],numbrellas*nsamples)
print "%8s %6s %6s %8s %10s %10s %10s %10s %8s" % ('bin', 'x', 'y', 'N', 'f', 'true','error','df','sigmas')
for i in range(1,nbins):
   if (i == zeroindex):
     stdevs = 0
     df_i[0] = 0
   else:
     error = pmf_analytical[i]-f_i[i]
     stdevs = numpy.abs(error)/df_i[i]
   print '%8d %6.2f %6.2f %8d %10.3f %10.3f %10.3f %10.3f %8.2f' % (i, bin_centers[i,0], bin_centers[i,1] , bin_counts[i], f_i[i], pmf_analytical[i], error, df_i[i], stdevs)

#=============================================================================================
# TERMINATE
#=============================================================================================

# Signal successful execution.
sys.exit(0)

