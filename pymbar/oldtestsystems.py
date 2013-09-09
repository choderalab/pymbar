#!/usr/bin/env python

"""
Test systems for pymbar.

"""

#=============================================================================================
# COPYRIGHT NOTICE
#
# Written by John D. Chodera <jchodera@gmail.com> and Michael R. Shirts <mrshirts@gmail.com>.
#
# Copyright (c) 2006-2007 The Regents of the University of California.  All Rights Reserved.
# Portions of this software are Copyright (c) 20010-2012 University of California and University of Virginia
#
# This program is free software; you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#=============================================================================================

#=============================================================================================
# TODO
#=============================================================================================

#=============================================================================================
# VERSION CONTROL INFORMATION
#=============================================================================================

__version__ = "$Revision: $ $Date: $"
# $Date: 2009-11-03 21:43:35 -0600 (Tue, 03 Nov 2009) $
# $Revision: 87 $
# $LastChangedBy: mrshirts $
# $HeadURL: https://simtk.org/svn/pymbar/trunk/pymbar/testsystems.py $
# $Id: MBAR.py 87 2009-11-04 03:43:35Z mrshirts $

#=============================================================================================
# IMPORTS
#=============================================================================================

import math
import numpy
import numpy.random
import numpy.linalg

#=============================================================================================
# Exception class.
#=============================================================================================

class ParameterError(Exception):
  """
  An error in the input parameters has been detected.

  """
  pass

#=============================================================================================
# Correlated timeseries
#=============================================================================================

def generateCorrelatedTimeseries(N=10000, tau=5.0, seed=None):
  """
  Generate synthetic timeseries data with known correlation time using bivariate Gaussian
  process described by Janke (Eq. 41 of Ref. [1]).

  OPTIONAL ARGUMENTS
    N (int) - length (in number of samples) of timeseries to generate
    tau (float) - correlation time (in number of samples) for timeseries
    seed (int) - specify the random number seed

  NOTES
    As noted in Eq. 45-46 of Ref. [1], the true integrated autocorrelation time will be given by

    tau_int = (1/2) coth(1 / 2 tau) = (1/2) (1+rho)/(1-rho)

    which, for tau >> 1, is approximated by

    tau_int = tau + 1/(12 tau) + O(1/tau^3)

    So for tau >> 1, tau_int is approximately the given exponential tau.

  REFERENCES
    [1] Janke W. Statistical analysis of simulations: Data correlations and error estimation.
    In 'Quantum Simulations of Complex Many-Body Systems: From Theory to Algorithms'.
    NIC Series, VOl. 10, pages 423-445, 2002.

  EXAMPLES

  Generate a timeseries of length 10000 with correlation time of 10.

  >>> A_t = generateCorrelatedTimeseries(N=10000, tau=10.0)

  Generate an uncorrelated timeseries of length 1000.

  >>> A_t = generateCorrelatedTimeseries(N=1000, tau=1.0)

  Generate a correlated timeseries with correlation time longer than the length.

  >>> A_t = generateCorrelatedTimeseries(N=1000, tau=2000.0)
    
  """

  # Set random number generator into a known state for reproducibility.                                                       
  if seed is not None:
    numpy.random.seed(seed)

  # Compute correlation coefficient rho, 0 <= rho < 1.
  rho = math.exp(-1.0 / tau)
  sigma = math.sqrt(1.0 - rho*rho)

  # Generate uncorrelated Gaussian variates.
  e_n = numpy.random.randn(N)

  # Restore random number generator state.
  if seed is not None:
    numpy.random.set_state(state)
    
  # Generate correlated signal from uncorrelated Gaussian variates using correlation coefficient.
  # NOTE: This will be slow.
  # TODO: Can we speed this up using vector operations?
  A_n = numpy.zeros([N], numpy.float32)
  A_n[0] = e_n[0]
  for n in range(1,N):
    A_n[n] = rho * A_n[n-1] + sigma * e_n[n]

  return A_n

#=============================================================================================
# Gaussian work distributions.
#=============================================================================================

def GaussianWorkSample(N_F=200, N_R=200, mu_F=2.0, DeltaF=None, sigma_F=1.0, seed=None):
  """
  Generate samples from forward and reverse Gaussian work distributions.

  OPTIONAL ARGUMENTS
    N_F (int) - number of forward measurements (default: 20)
    N_R (int) - number of reverse measurements (default: 20)
    mu_F (float) - mean of forward work distribution
    DeltaF (float) - the free energy difference, which can be specified instead of mu_F (default: None)
    sigma_F (float) - variance of the forward work distribution (default: 1.0)
    seed (any hashable object) - random number generator seed for reproducible results, or None (default: None)
      old state is restored after call
      
  RETURNS
    w_F (numpy.array of numpy.float64) - forward work values
    w_R (numpy.array of numpy.float64) - reversework values    

  NOTES
    By the Crooks fluctuation theorem (CFT), the forward and backward work distributions are related by

    P_R(-w) = P_F(w) \exp[DeltaF - w]

    If the forward distribution is Gaussian with mean \mu_F and std dev \sigma_F, then

    P_F(w) = (2 \pi)^{-1/2} \sigma_F^{-1} \exp[-(w - \mu_F)^2 / (2 \sigma_F^2)]

    With some algebra, we then find the corresponding mean and std dev of the reverse distribution are

    \mu_R = - \mu_F + \sigma_F^2
    \sigma_R = \sigma_F \exp[\mu_F - \sigma_F^2 / 2 + \Delta F]

    where all quantities are in reduced units (e.g. divided by kT).

    Note that \mu_F and \Delta F are not independent!  By the Zwanzig relation,

    E_F[exp(-w)] = \int dw \exp(-w) P_F(w) = \exp[-\Delta F]

    which, with some integration, gives

    \Delta F = \mu_F + \sigma_F^2/2 

    which can be used to determine either \mu_F or \DeltaF.

  EXAMPLES

  Generate work values with default parameters.
  
  >>> [w_F, w_R] = GaussianWorkSample()

  Generate 50 forward work values and 70 reverse work values.

  >>> [w_F, w_R] = GaussianWorkSample(N_F=50, N_R=70)

  Generate work values specifying the work distribution parameters.

  >>> [w_F, w_R] = GaussianWorkSample(mu_F=3.0, sigma_F=2.0)

  Generate work values specifying the work distribution parameters, specifying free energy difference instead of mu_F.

  >>> [w_F, w_R] = GaussianWorkSample(mu_F=None, DeltaF=3.0, sigma_F=2.0)

  Generate work values with known seed to ensure reproducibility for testing.

  >>> [w_F, w_R] = GaussianWorkSample(seed=0)

  """

  # Make sure either mu_F or DeltaF, but not both, are specified.
  if (mu_F is not None) and (DeltaF is not None):
    raise ParameterError("mu_F and DeltaF are not independent, and cannot both be specified; one must be set to None.")
  if (mu_F is None) and (DeltaF is None):
    raise ParameterError("Either mu_F or DeltaF must be specified.")
  if (mu_F is None):
    mu_F = DeltaF + sigma_F**2/2.0
  if (DeltaF is None):
    DeltaF = mu_F - sigma_F**2/2.0
  
  # Set random number generator into a known state for reproducibility.
  if seed is not None:
    state = numpy.random.get_state()
    numpy.random.seed(seed)

  # Determine mean and variance of reverse work distribution by Crooks fluctuation theorem (CFT).
  mu_R = - mu_F + sigma_F**2
  sigma_R = sigma_F * math.exp(mu_F - sigma_F**2/2.0 - DeltaF)

  # Draw samples from forward and reverse distributions.
  w_F = numpy.random.randn(N_F) * sigma_F + mu_F
  w_R = numpy.random.randn(N_R) * sigma_R + mu_R
  
  # Restore random number generator state.
  if seed is not None:
    numpy.random.set_state(state)

  return [w_F, w_R]

#=============================================================================================
# Gaussian work distributions.
#=============================================================================================

def HarmonicOscillatorsSample(N_k=[100, 100, 100], O_k = [0, 1, 2], K_k = [1, 1, 1], seed=None):
  """
  Generate samples from 1D harmonic oscillators with specified relative spacing (in units of std devs).

  OPTIONAL ARGUMENTS
    N_k (list or numpy.array of nstates) - number of samples per state
    O_k (list or numpy.array of floats) - offsets of the harmonic oscillators in dimensionless units
    K_k (list or numpy.array of floats) - force constants of harmonic oscillators in dimensionless units
    seed (int) - random number seed for reproducibility, default none
    
    N_k,O_k,and K_k must have the same length.

  RETURNS
    x_kn (numpy.array of nstates x nsamples) - 1D harmonic oscillator positions
    u_kln (numpy.array of nstates x nstates x nsamples) - reduced potential
    N_k (numpy.array of nstates) - number of samples per state

  EXAMPLES

  Generate energy samples with default parameters.
  
  >>> [x_kn, u_kln, N_k] = HarmonicOscillatorsSample()

  Specify number of samples, specify the states of the harmonic oscillators

  >>> [x_kn, u_kln, N_k] = HarmonicOscillatorsSample(N_k=[10, 20, 30, 40, 50], O_k=[0, 1, 2, 3, 4], K_k=[1, 2, 4, 8, 16]) 

  """
  
  # Promote to array.
  N_k = numpy.array(N_k)
  O_k = numpy.array(O_k)
  K_k = numpy.array(K_k);

  # Determine maximum number of samples.
  Nmax = N_k.max()

  # Determine number of states.
  K = N_k.size
  
  # check to make sure that the number of states is consistent between the arrays
  if O_k.size != K:
    raise "O_k and N_k mut have the same dimensions."

  if K_k.size != K:
    raise "K_k and N_k mut have the same dimensions."

  # initialize seed
  numpy.random.seed(seed)

  # calculate the standard deviation induced by the spring constants.
  sigma_k = (K_k)**-0.5
  
  # generate space to store the energies
  u_kln  = numpy.zeros([K, K, Nmax], numpy.float64)

  # Generate position samples.
  x_kn = numpy.zeros([K, Nmax], numpy.float64)
  for k in range(K):
    x_kn[k,0:N_k[k]] = numpy.random.normal(O_k[k], sigma_k[k], N_k[k])
    for l in range(K):
      u_kln[k,l,0:N_k[k]] = (K_k[l]/2.0) * (x_kn[k,0:N_k[k]]-O_k[l])**2

  # Return results.
  return [x_kn, u_kln, N_k]

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
  import doctest
  doctest.testmod()

  # Test computeMultipleExpectations.
  [x_kn, u_kln, N_k] = HarmonicOscillatorsSample(N_k=[100, 100, 100, 100, 100], O_k = [0, 1, 2, 3, 4], K_k = [1,1,1,1,1] )
  import pymbar
  K = len(N_k)
  mbar = pymbar.MBAR(u_kln, N_k)
  A_ikn = numpy.zeros([2, K, N_k.max()], numpy.float64)
  A_ikn[0,:,:] = x_kn[:,:]
  A_ikn[1,:,:] = x_kn[:,:]**2
  for i in range(K):
    [A_i, d2A_ij] = mbar.computeMultipleExpectations(A_ikn, u_kln[:,i,:])
    print (i, A_i)







