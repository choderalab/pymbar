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
from __future__ import print_function
import sys
import numpy as np
from pymbar import testsystems, EXP, EXPGauss, BAR, MBAR
from pymbar.utils import ParameterError

#=============================================================================================
# HELPER FUNCTIONS
#=============================================================================================

def stddev_away(namex,errorx,dx):

  if dx > 0:
    print("%s differs by %.3f standard deviations from analytical" % (namex,errorx/dx))
  else:
    print("%s differs by an undefined number of standard deviations" % (namex))

def GetAnalytical(beta,K,O,observables):

  # For a harmonic oscillator with spring constant K,
  # x ~ Normal(x_0, sigma^2), where sigma = 1/sqrt(beta K)

  # Compute the absolute dimensionless free energies of each oscillator analytically.
  # f = - ln(sqrt((2 pi)/(beta K)) )
  print('Computing dimensionless free energies analytically...')

  sigma = (beta * K)**-0.5
  f_k_analytical = - np.log(np.sqrt(2 * np.pi) * sigma )

  Delta_f_ij_analytical = np.matrix(f_k_analytical) - np.matrix(f_k_analytical).transpose()

  A_k_analytical = dict()
  A_ij_analytical = dict()

  for observe in observables:
    if observe == 'RMS displacement':
      A_k_analytical[observe] = sigma                           # mean square displacement
    if observe == 'potential energy':
      A_k_analytical[observe] = 1/(2*beta)*np.ones(len(K),float)  # By equipartition
    if observe == 'position':  
      A_k_analytical[observe] = O                                       # observable is the position
    if observe == 'position^2':    
      A_k_analytical[observe]  = (1+ beta*K*O**2)/(beta*K)        # observable is the position^2

    A_ij_analytical[observe] = A_k_analytical[observe] - np.transpose(np.matrix(A_k_analytical[observe]))

  return f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical

#=============================================================================================
# PARAMETERS
#=============================================================================================

copies = 2
K_k = copies*[2.5,1.6,9,4,1,1]
K_k = np.array(K_k) # spring constants for each state
O_i = [0,1,2,3,4,5]
O_k = np.array(copies*O_i) # offsets for spring constants
O_k = np.array(O_k) 
for c in range(copies):
  O_k[len(O_i)*c:len(O_i)*(c+1)] += c*len(O_i)*np.ones(len(O_i),int)

N_k = copies*[1000, 1000, 1000, 1000, 0, 1000]
N_k = 200*np.array(N_k) # number of samples from each state (can be zero for some states)
Nk_ne_zero = (N_k!=0)
beta = 1.0 # inverse temperature for all simulations
K_extra = np.array([20, 12, 6, 2, 1]) 
O_extra = np.array([ 0.5, 1.5, 2.5, 3.5, 4.5])
observables = ['position','position^2','potential energy','RMS displacement']

seed = None
# Uncomment the following line to seed the random number generated to produce reproducible output.
seed = 0
np.random.seed(seed)

#=============================================================================================
# MAIN
#=============================================================================================

# Determine number of simulations.
K = np.size(N_k)
if np.shape(K_k) != np.shape(N_k): 
  raise ParameterError("K_k (%d) and N_k (%d) must have same dimensions." % (np.shape(K_k), np.shape(N_k)))
if np.shape(O_k) != np.shape(N_k): 
  raise ParameterError("O_k (%d) and N_k (%d) must have same dimensions." % (np.shape(K_k), np.shape(N_k)))

# Determine maximum number of samples to be drawn for any state.
N_max = np.max(N_k)

(f_k_analytical, Delta_f_ij_analytical, A_k_analytical, A_ij_analytical) = GetAnalytical(beta,K_k,O_k,observables)

print("This script will draw samples from %d harmonic oscillators." % (K))
print("The harmonic oscillators have equilibrium positions")
print(O_k)
print("and spring constants")
print(K_k)
print("and the following number of samples will be drawn from each (can be zero if no samples drawn):")
print(N_k)
print("")

#=============================================================================================
# Generate independent data samples from K one-dimensional harmonic oscillators centered at q = 0.
#=============================================================================================
  
print('generating samples...')
randomsample = testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k=O_k, K_k=K_k, beta=beta)
[x_kn,u_kn,N_k,s_n] = randomsample.sample(N_k,mode='u_kn')

# get the unreduced energies
U_kn = u_kn/beta

#=============================================================================================
# Estimate free energies and expectations.
#=============================================================================================

print("======================================")
print("      Initializing MBAR               ")
print("======================================")

# Estimate free energies from simulation using MBAR.
print("Estimating relative free energies from simulation (this may take a while)...")

# Initialize the MBAR class, determining the free energies.
mbar = MBAR(u_kn, N_k, relative_tolerance=1.0e-10, verbose=True)
# Get matrix of dimensionless free energy differences and uncertainty estimate.

print("=============================================")
print("      Testing getFreeEnergyDifferences       ")
print("=============================================")
results = mbar.getFreeEnergyDifferences(return_dict=True)
Delta_f_ij_estimated = results['Delta_f']
dDelta_f_ij_estimated = results['dDelta_f']

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
stdevs = np.abs(Delta_f_ij_error/df_ij_mod)
for k in range(K):
  stdevs[k,k] = 0
print(stdevs)

exit()
