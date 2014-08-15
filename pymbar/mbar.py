##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2010-2014 University of Virginia, Memorial Sloan-Kettering Cancer Center
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with pymbar. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

"""
A module implementing the multistate Bennett acceptance ratio (MBAR) method for the analysis
of equilibrium samples from multiple arbitrary thermodynamic states in computing equilibrium
expectations, free energy differences, potentials of mean force, and entropy and enthalpy contributions.

Please reference the following if you use this code in your research:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177

This module contains implementations of

* MBAR - multistate Bennett acceptance ratio estimator

"""

#=========================================================================
# COPYRIGHT NOTICE
#
# Written by John D. Chodera <jchodera@gmail.com> and Michael R. Shirts <mrshirts@gmail.com>.
#
# Copyright (c) 2006-2007 The Regents of the University of California.  All Rights Reserved.
# Portions of this software are Copyright (c) 2007-2008 Stanford University and Columbia University.
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
#=========================================================================

import math
import numpy as np
import numpy.linalg as linalg
from pymbar import mbar_solvers
from pymbar.utils import _logsum, kln_to_kn, kn_to_n, ParameterError

#=========================================================================
# MBAR class definition
#=========================================================================


class MBAR:

    """Multistate Bennett acceptance ratio method (MBAR) for the analysis of multiple equilibrium samples.

    Notes
    -----    
    Note that this method assumes the data are uncorrelated.
    Correlated data must be subsampled to extract uncorrelated (effectively independent) samples (see example below).

    References
    ----------

    [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
    J. Chem. Phys. 129:124105, 2008
    http://dx.doi.org/10.1063/1.2978177
    """
    #=========================================================================

    def __init__(self, u_kn, N_k, maximum_iterations=10000, relative_tolerance=1.0e-7, verbose=False, initial_f_k=None, solver_protocol=None, initialize='zeros', x_kindices=None, **kwargs):
        """Initialize multistate Bennett acceptance ratio (MBAR) on a set of simulation data.

        Upon initialization, the dimensionless free energies for all states are computed.
        This may take anywhere from seconds to minutes, depending upon the quantity of data.
        After initialization, the computed free energies may be obtained by a call to 'getFreeEnergies()', or
        free energies or expectation at any state of interest can be computed by calls to 'computeFreeEnergy()' or
        'computeExpectations()'.

        ----------
        u_kn : np.ndarray, float, shape=(K, N_max)
            u_kn[k,n] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.
            u_kln: np.ndarray, float, shape (K, L, N_max)
               if the simulation is in form u_kln[k,l,n] it is converted to u_kn format

        u_kn = [ u_1(x_1) u_1(x_2) u_1(x_3) . . . u_1(x_n)
                 u_2(x_1) u_2(x_2) u_2(x_3) . . . u_2(x_n)
                                .  .  .
                 u_k(x_1) u_k(x_2) u_k(x_3) . . . u_k(x_n)]

        N_k :  np.ndarray, int, shape=(K)
            N_k[k] is the number of uncorrelated snapshots sampled from state k.
            Some may be zero, indicating that there are no samples from that state.

        We assume that the states are ordered such that the first N_k
        are from the first state, the 2nd N_k the second state, and so
        forth. This only becomes important for BAR -- MBAR does not
        care which samples are from which state.  We should eventually
        allow this assumption to be overwritten by parameters passed
        from above, once u_kln is phased out.

        maximum_iterations : int, optional
            Set to limit the maximum number of iterations performed (default 1000)
        relative_tolerance : float, optional
            Set to determine the relative tolerance convergence criteria (default 1.0e-6)
        verbosity : bool, optional
            Set to True if verbose debug output is desired (default False)
        initial_f_k : np.ndarray, float, shape=(K), optional
            Set to the initial dimensionless free energies to use as a 
            guess (default None, which sets all f_k = 0)
        method : list(dict), optional, default=None
            List of dictionaries to define a sequence of solver algorithms
            and options used to estimate the dimensionless free energies.
            See `pymbar.mbar_solvers.solve_mbar()` for details.  If None,
            use the developers best guess at an appropriate algorithm.
        initialize : string, optional
            If equal to 'BAR', use BAR between the pairwise state to
            initialize the free energies.  Eventually, should specify a path;
            for now, it just does it zipping up the states.
            (default: 'zeros', unless specific values are passed in.)
        x_kindices
            Which state is each x from?  Usually doesn't matter, but does for BAR. We assume the samples
            are in K order (the first N_k[0] samples are from the 0th state, the next N_k[1] samples from 
            the 1st state, and so forth.

        Notes
        -----
        The reduced potential energy u_kn[k,n] = u_k(x_{ln}), where the reduced potential energy u_l(x) is defined (as in the text) by:
        u_k(x) = beta_k [ U_k(x) + p_k V(x) + mu_k' n(x) ]
        where
        beta_k = 1/(kB T_k) is the inverse temperature of condition k, where kB is Boltzmann's constant
        U_k(x) is the potential energy function for state k
        p_k is the pressure at state k (if an isobaric ensemble is specified)
        V(x) is the volume of configuration x
        mu_k is the M-vector of chemical potentials for the various species, if a (semi)grand ensemble is specified, and ' denotes transpose
        n(x) is the M-vector of numbers of the various molecular species for configuration x, corresponding to the chemical potential components of mu_m.
        x_n indicates that the samples are from k different simulations of the n states. These simulations need only be a subset of the k states.
        The configurations x_ln must be uncorrelated.  This can be ensured by subsampling a correlated timeseries with a period larger than the statistical inefficiency,
        which can be estimated from the potential energy timeseries {u_k(x_ln)}_{n=1}^{N_k} using the provided utility function 'statisticalInefficiency()'.
        See the help for this function for more information.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_kn, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)

        """
        for key, val in kwargs.items():
            print("Warning: parameter %s=%s is unrecognized and unused." % (key, val))

        # Store local copies of necessary data.
        # N_k[k] is the number of samples from state k, some of which might be zero.
        self.N_k = np.array(N_k, dtype=np.int32)
        self.N = np.sum(self.N_k)

        # Get dimensions of reduced potential energy matrix, and convert to KxN form if needed.
        if len(np.shape(u_kn)) == 3:
            self.K = np.shape(u_kn)[1]  # need to set self.K, and it's the second index
            u_kn = kln_to_kn(u_kn, N_k=self.N_k)

        # u_kn[k,n] is the reduced potential energy of sample n evaluated at state k
        self.u_kn = np.array(u_kn, dtype=np.float64)

        [K, N] = np.shape(u_kn)

        if verbose:
            print "K (total states) = %d, total samples = %d" % (K, N)

        if np.sum(N_k) != N:
            raise ParameterError(
                'The sum of all N_k must equal the total number of samples (length of second dimension of u_kn.')

        # Store local copies of other data
        self.K = K  # number of thermodynamic states energies are evaluated at
        # N = \sum_{k=1}^K N_k is the total number of samples
        self.N = N  # maximum number of configurations

        # if not defined, identify from which state each sample comes from.
        if x_kindices is not None:
            self.x_kindices = x_kindices
        else:
            self.x_kindices = np.array(N, dtype=np.int32)
            Nsum = 0
            for k in range(K):
                self.x_kindices[Nsum:Nsum+N_k[k]] = k
                Nsum += N_k[k]
            
        # verbosity level -- if True, will print extra debug information
        self.verbose = verbose

        # perform consistency checks on the data.

        # if, for any set of data, all reduced potential energies are the same,
        # they are probably the same state.  We check to within
        # relative_tolerance.

        self.samestates = []
        if self.verbose:
            for k in range(K):
                for l in range(k):
                    diffsum = 0
                    uzero = u_kn[k, :] - u_kn[l, :]
                    diffsum += np.dot(uzero, uzero)
                    if (diffsum < relative_tolerance):
                        self.samestates.append([k, l])
                        self.samestates.append([l, k])
                        print ''
                        print 'Warning: states %d and %d have the same energies on the dataset.' % (l, k)
                        print 'They are therefore likely to to be the same thermodynamic state.  This can occasionally cause'
                        print 'numerical problems with computing the covariance of their energy difference, which must be'
                        print 'identically zero in any case. Consider combining them into a single state.'
                        print ''

        # Print number of samples from each state.
        if self.verbose:
            print "N_k = "
            print N_k

        # Determine list of k indices for which N_k != 0
        self.states_with_samples = np.where(self.N_k != 0)[0]
        self.states_with_samples = self.states_with_samples.astype(np.int32)

        # Number of states with samples.
        self.K_nonzero = self.states_with_samples.size
        if verbose:
            print "There are %d states with samples." % self.K_nonzero

        # Initialize estimate of relative dimensionless free energy of each state to zero.
        # Note that f_k[0] will be constrained to be zero throughout.
        # this is default
        self.f_k = np.zeros([self.K], dtype=np.float64)

        # If an initial guess of the relative dimensionless free energies is
        # specified, start with that.
        if initial_f_k != None:
            if self.verbose:
                print "Initializing f_k with provided initial guess."
            # Cast to np array.
            initial_f_k = np.array(initial_f_k, dtype=np.float64)
            # Check shape
            if initial_f_k.shape != self.f_k.shape:
                raise ParameterError(
                    "initial_f_k must be a %d-dimensional np array." % self.K)
            # Initialize f_k with provided guess.
            self.f_k = initial_f_k
            if self.verbose:
                print self.f_k
            # Shift all free energies such that f_0 = 0.
            self.f_k[:] = self.f_k[:] - self.f_k[0]
        else:
            # Initialize estimate of relative dimensionless free energies.
            self._initializeFreeEnergies(verbose, method=initialize)

            if self.verbose:
                print "Initial dimensionless free energies with method %s" % (initialize)
                print "f_k = "
                print self.f_k
        
        self.solve_mbar(solver_protocol=solver_protocol)
        
        # Print final dimensionless free energies.
        if self.verbose:
            print "Final dimensionless free energies"
            print "f_k = "
            print self.f_k

        if self.verbose:
            print "MBAR initialization complete."
        return

    def solve_mbar(self, solver_protocol=None, start_small=True):
        """Solve for free energies of states with samples, then calculate for empty states."""

        if start_small:
            s_n = np.unique(self.x_kindices, return_inverse=True)[1]
            u_kn, N_k = mbar_solvers.get_representative_sample(self.u_kn[self.states_with_samples], self.N_k[self.states_with_samples], s_n, subsampling=5, rescale=True)
            #u_kn, N_k = mbar_solvers.get_two_sample_representation(self.u_kn[self.states_with_samples], self.N_k[self.states_with_samples], self.x_kindices)
            f_k_nonzero, all_results = mbar_solvers.solve_mbar(u_kn, N_k, self.f_k[self.states_with_samples], solver_protocol=[dict(method="L-BFGS-B", fast=True), dict(method="adaptive")])
            self.f_k[self.states_with_samples] = f_k_nonzero

        f_k_nonzero, all_results = mbar_solvers.solve_mbar(self.u_kn[self.states_with_samples], self.N_k[self.states_with_samples], self.f_k[self.states_with_samples], solver_protocol)
        self.f_k[self.states_with_samples] = f_k_nonzero
        # Recompute all free energies because those from states with zero samples are not correctly computed by Newton-Raphson.
        # and store the log weights
        if self.verbose:
            print "Recomputing all free energies and log weights for storage"

        self.f_k[self.N_k <= 0] = mbar_solvers.self_consistent_update(self.u_kn, self.N_k, self.f_k)[self.N_k <= 0]
        self.Log_W_nk = np.log(mbar_solvers.mbar_W_nk(self.u_kn, self.N_k, self.f_k))


    #=========================================================================
    def getWeights(self):
        """Retrieve the weight matrix W_nk from the MBAR algorithm.

        Necessary because they are stored internally as log weights.

        Returns
        -------
        weights : np.ndarray, float, shape=(N, K)
            NxK matrix of weights in the MBAR covariance and averaging formulas

        """

        return np.exp(self.Log_W_nk)

    #=========================================================================
    def getFreeEnergyDifferences(self, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Get the dimensionless free energy differences and uncertainties among all thermodynamic states.


        Parameters
        ----------
        compute_uncertainty : bool, optional
            If False, the uncertainties will not be computed (default: True)
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method,
            or None to use default.  See help for computeAsymptoticCovarianceMatrix()
            for more information on various methods. (default: svd)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude
            than this number (default: 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences.

        Returns
        -------
        Deltaf_ij :L np.ndarray, float, shape=(K, K)
            Deltaf_ij[i,j] is the estimated free energy difference
        dDeltaf_ij :L np.ndarray, float, shape=(K, K)
            dDeltaf_ij[i,j] is the estimated statistical uncertainty 
            (one standard deviation) in Deltaf_ij[i,j]

        Notes
        -----
        Computation of the covariance matrix may take some time for large K.

        The reported statistical uncertainty should, in the asymptotic limit, reflect one standard deviation for the normal distribution of the estimate.
        The true free energy difference should fall within the interval [-df, +df] centered on the estimate 68% of the time, and within
        the interval [-2 df, +2 df] centered on the estimate 95% of the time.
        This will break down in cases where the number of samples is not large enough to reach the asymptotic normal limit.

        See Section III of Reference [1].

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_kn, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> [Deltaf_ij, dDeltaf_ij] = mbar.getFreeEnergyDifferences()

        """

        # Compute free energy differences.
        f_i = np.matrix(self.f_k)
        Deltaf_ij = f_i - f_i.transpose()

        # zero out numerical error for thermodynamically identical states
        self._zerosamestates(Deltaf_ij)

        returns = []
        returns.append(np.array(Deltaf_ij))

        if compute_uncertainty or return_theta:
            # Compute asymptotic covariance matrix.
            Theta_ij = self._computeAsymptoticCovarianceMatrix(
                np.exp(self.Log_W_nk), self.N_k, method=uncertainty_method)

        if compute_uncertainty:
            # compute the covariance component without doing the double loop.
            # d2DeltaF = Theta_ij[i,i] + Theta_ij[j,j] - 2.0 * Theta_ij[i,j]

            diag = Theta_ij.diagonal()
            d2DeltaF = diag + diag.transpose() - 2 * Theta_ij

            # zero out numerical error for thermodynamically identical states
            self._zerosamestates(d2DeltaF)

            # check for any numbers below zero.
            if (np.any(d2DeltaF < 0.0)):
                if(np.any(d2DeltaF) < warning_cutoff):
                    # Hmm.  Will this print correctly?
                    print "A squared uncertainty is negative.  d2DeltaF = %e" % d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)]
                else:
                    d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)] = 0.0

            # take the square root of the entries of the matrix
            dDeltaf_ij = np.sqrt(d2DeltaF)

            # Return matrix of free energy differences and uncertainties.
            returns.append(np.array(dDeltaf_ij))

        if (return_theta):
            returns.append(np.array(Theta_ij))

        return returns

    #=========================================================================
    def computeGeneralExpectations(self, A_in, u_ln, state_list, compute_uncertainty=True,
                                   uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the expectations of multiple observables of phase space functions on multiple states.

        Compute the expectations of multiple observables of phase
        space functions.  [A_0(x),A_1(x),...,A_i(x)] along with the
        covariances of their estimates at multiple states.

        We calculate all observables at all states that are specified by the choice of state list.
        Generally, it will be called in specific conditions.

        Parameters
        ----------
        A_in : np.ndarray, float, shape=(I, N)
            A_in[i,n] = A_i(x_n), the value of phase observable i for configuration n
        u_ln : np.ndarray, float, shape=(L, N)
            u_n[l,n] is the reduced potential of configuration n at state l
            if u_ln = None, we use self.u_kn

        state_list : np.ndarray, int, shape (NS,2), where NS is the
                     total number of states of interest.  it will be
                     of the form [[0,0],[1,1],[2,1]] which would
                     indicate we want to output the properties of
                     three observables total: the first property A[0]
                     at the 0th state, the 2nd property A[1] at the
                     1th state, and the 2nd property A[1] at the 2nd
                     state.  This allows us to tailor this to a number of different situations.

        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences.

        Returns
        -------

        A_i : np.ndarray, float, shape = (I)
            A_i[i] is the estimate for the expectation of A_state_list[i](x) at the state specified by u_n[state_list[i],:]

        d2A_ik : np.ndarray, float, shape = (I, J)
            d2A_ij[i,j] is the COVARIANCE in the estimates of observables A_i and A_j (as determined by the state list)
            (* not the square root *)

        General cases this will be used for.
            single observable, multiple states (replacement for computeExpectations)
            multiple observables, single state (replacement for computeMultipleExpectations)
            diagonal cases of multiple states, single states.

        Examples
        --------

        update this example to be more general
        >>> from pymbar import testsystems
        >>> [x_kn, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> A_in = np.array([x_kn,x_kn**2,x_kn**3])
        >>> u_n = u_kn[:2,:]
        >>> state_list = np.array([[0,0],[1,0],[2,0],[2,1]],int)
        >>> [A_i, d2A_ij] = mbar.computeGeneralExpectations(A_in, u_n, state_list)

        """

        # Retrieve N and K for convenience.
        S = len(state_list) # number of computed expectations examined
        K = self.K

        N = self.N  # N is total number of samples

        # make observables all positive
        A_list = np.unique(state_list[:,0])
        I = len(A_list) # number of observables used
        A_min = np.zeros([I], dtype=np.float64)

        for i in A_list:  # only need to zero the ones we will use.  May be some repetition here.
            A_min[i] = np.min(A_in[i, :]) #find the minimum
            A_in[i, :] = A_in[i,:] - (A_min[i] - 1)  #all values now positive so that we can work in logarithmic scale

        # Augment W_nk, N_k, and c_k for q_A(x) for the observables, with one
        # row for the specified state and I rows for the observable at that
        # state.
        # log weight matrix
        sizea = K + 2*S # augmented size
        Log_W_nk = np.zeros([N, sizea], np.float64) # log weight matrix
        N_k = np.zeros([sizea], np.int32)  # counts
        f_k = np.zeros([sizea], np.float64)  # free energies

        # Fill in first section of matrix with existing q_k(x) from states.
        Log_W_nk[:, 0:K] = self.Log_W_nk
        N_k[0:K] = self.N_k
        f_k[0:K] = self.f_k

        # Compute row of W matrix for the extra states corresponding to u_ln according to the state list
        for s in range(S):
            l = state_list[s,1]
            la = K+s  #l augmented
            Log_W_nk[:, la] = self._computeUnnormalizedLogWeights(u_ln[l,:])
            f_k[la] = -_logsum(Log_W_nk[:, la])
            Log_W_nk[:, la] += f_k[l]

        # Compute the remaining rows/columns of W_nk and c_k for the
        # observables.
        for s in range(S):
            sa = K+S+s  # augmented s
            i = state_list[s,0]
            l = state_list[s,1]
            Log_W_nk[:, sa] = np.log(A_in[i, :]) + Log_W_nk[:, K+l]
            f_k[sa] = -_logsum(Log_W_nk[:, sa])
            Log_W_nk[:, sa] += f_k[sa]    # normalize this row

        # Compute estimates.
        A_i = np.zeros([S], np.float64)
        for s in range(S):
            A_i[s] = np.exp(-f_k[K + S + s])

        if compute_uncertainty or return_theta:
            # Compute augmented asymptotic covariance matrix.
            W_nk = np.exp(Log_W_nk)
            Theta_ij = self._computeAsymptoticCovarianceMatrix(
                W_nk, N_k, method=uncertainty_method)

        if compute_uncertainty:
            # Compute estimates of statistical covariance
            # these variances will be the same whether or not we subtract a different constant from each A_i
            # todo: vectorize
            # compute the covariance component without doing the double loop
            d2A_ij = np.zeros([S, S], np.float64)
            for i in range(S):
                si = K+S+i
                li = K+state_list[i,1]
                for j in range(S):
                    sj = K+S+j
                    lj = K+state_list[j,1]
                    d2A_ij[i, j] = A_i[i] * A_i[j] * (
                        Theta_ij[si, sj] - Theta_ij[si, li] - Theta_ij[lj, sj] + Theta_ij[li, lj])

        # Now that covariances are computed, add the constants back to A_i that
        # were required to enforce positivity
        for s in range(S):
            A_i[s] += (A_min[state_list[s,0]] - 1)

        # these values may be used outside the routine, so copy back.
        for i in A_list:
            A_in[i, :] = A_in[i,:] + (A_min[i] - 1)

        returns = []
        returns.append(A_i)

        if compute_uncertainty:
            returns.append(d2A_ij)

        if return_theta:
            returns.append(Theta_ij)

        # Return expectations and uncertainties.
        return returns

    #=========================================================================

    def computeExpectations(self, A_n, output='averages', compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False, useGeneral = False, state_dependent = False):
        """Compute the expectation of an observable of a phase space function.

        Compute the expectation of an observable of phase space
        function A(x) at all states where potentials are generated,
        including states for which no samples were drawn.

        We assume observables are not function of the state.  u is not
        an observable -- it changes depending on the state. u_k is an
        observable; the energy of state k does not depend on the
        state. To compute the estimators of the energy at all K
        states, use . . .

        Parameters
        ----------
        A_n : np.ndarray, float
            A_n (N_max np float64 array) - A_n[n] = A(x_n)
        output : string, optional
            Either output averages, and uncertainties, or output a matrix of differences, with uncertainties.
        compute_uncertainty : bool, optional
            If False, the uncertainties will not be computed (default: True)
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method,
            or None to use default See help for computeAsymptoticCovarianceMatrix()
            for more information on various methods. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences.
        useGeneral: bool, whether to use the GeneralExpectations formalism = False,
        state_dependent: bool, whether the expectations are state-dependent.

        Returns
        -------
        A : np.ndarray, float
            if output is 'averages'
            A_i  (K np float64 array) -  A_i[i] is the estimate for the expectation of A(x) for state i.
            if output is 'differences'
        dA : np.ndarray, float
            dA_i  (K np float64 array) - dA_i[i] is uncertainty estimate (one standard deviation) for A_i[i]
            or
            dA_ij (K np float64 array) - dA_ij[i,j] is uncertainty estimate (one standard deviation) for the difference in A beteen i and j

        Notes
        -----

        The reported statistical uncertainty should, in the asymptotic limit,
        reflect one standard deviation for the normal distribution of the estimate.
        The true expectation should fall within the interval [-dA, +dA] centered on the estimate 68% of the time, and within
        the interval [-2 dA, +2 dA] centered on the estimate 95% of the time.
        This will break down in cases where the number of samples is not large enough to reach the asymptotic normal limit.
        This 'breakdown' can be exacerbated by the computation of observables like indicator functions for histograms that are sparsely populated.

        References
        ----------

        See Section IV of [1].

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_n, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> A_n = x_n
        >>> (A_ij, dA_ij) = mbar.computeExpectations(A_n)
        >>> A_n = u_kn[0,:]
        >>> (A_ij, dA_ij) = mbar.computeExpectations(A_n, output='differences')
        """

        dims = len(np.shape(A_n))

        # Retrieve N and K for convenience.
        N = self.N
        K = self.K

        if dims == 3:
            print "expecting dim=1 or dim=2"
            return None

        if (useGeneral):

            state_list = np.zeros([K,2],int)
            if (state_dependent):
                for k in range(K):
                    state_list[k,0] = k
                    state_list[k,1] = k
                A_in = A_n
            else:
                A_in = np.zeros([1,N], dtype=np.float64)
                if dims == 2:
                    A_n = kn_to_n(A_n, N_k=self.N_k)
                A_in[0,:] = A_n

                for k in range(K):
                    state_list[k,0] = 0
                    state_list[k,1] = k

            general_results = self.computeGeneralExpectations(A_in, self.u_kn, state_list,
                                                              compute_uncertainty=compute_uncertainty,
                                                              uncertainty_method=uncertainty_method,
                                                              warning_cutoff=warning_cutoff,
                                                              return_theta=return_theta)
            returns = []

            if output == 'averages':
                # Return expectations and uncertainties.
                returns.append(general_results[0])

                if compute_uncertainty:
                    indices = np.eye(K,dtype=bool)
                    returns.append(np.sqrt(general_results[1][indices]))

            if output == 'differences':
                A_im = np.matrix(general_results[0])
                A_ij = A_im - A_im.transpose()

                returns.append(np.array(A_ij))
                if compute_uncertainty:
                    return np.sqrt(general_results[1])

            if return_theta:
                returns.append(general_results[2])

        else:  # current style

            if dims == 2: #convert to 1xN shape
                A_n = kn_to_n(A_n, N_k=self.N_k)
                # Convert to np array.
                A_n = np.array(A_n, np.float64)

            # Augment W_nk, N_k, and c_k for q_A(x) for the observable, with one
            # extra row/column for each state (Eq. 13 of [1]).
            # log of weight matrix
            Log_W_nk = np.zeros([N, K * 2], np.float64)
            N_k = np.zeros([K * 2], np.int32)  # counts
            # "free energies" of the new states
            f_k = np.zeros([K], np.float64)

            # Fill in first half of matrix with existing q_k(x) from states.
            Log_W_nk[:, 0:K] = self.Log_W_nk
            N_k[0:K] = self.N_k

            # Make A_n all positive so we can operate logarithmically for
            # robustness
            A_i = np.zeros([K], np.float64)
            A_min = np.min(A_n)
            A_n = A_n - (A_min - 1)

            # Compute the remaining rows/columns of W_nk and the rows c_k for the
            # observables.

            for l in range(K):

                # this works because all A_n are now positive;
                Log_W_nk[:, K + l] = np.log(A_n) + self.Log_W_nk[:, l]
                # we took the min at the beginning.
                f_k[l] = -_logsum(Log_W_nk[:, K + l])
                Log_W_nk[:, K + l] += f_k[l]              # normalize the row
                A_i[l] = np.exp(-f_k[l])

            if compute_uncertainty or return_theta:
                # Compute augmented asymptotic covariance matrix.
                Theta_ij = self._computeAsymptoticCovarianceMatrix(
                    np.exp(Log_W_nk), N_k, method=uncertainty_method)

            returns = []

            if output == 'averages':

                if compute_uncertainty:
                    # Compute uncertainties.
                    dA_i = np.zeros([K], np.float64)
                # just the diagonals
                    for k in range(0, K):
                        dA_i[k] = np.abs(A_i[k]) * np.sqrt(
                            Theta_ij[K + k, K + k] + Theta_ij[k, k] - 2.0 * Theta_ij[k, K + k])

                # add back minima now now that uncertainties are computed.
                    A_i += (A_min - 1)

                # Return expectations and uncertainties.
                returns.append(np.array(A_i))

                if compute_uncertainty:
                    returns.append(np.array(dA_i))

            if output == 'differences':
                # Return differences of expectations and uncertainties.

                # compute expectation differences
                A_im = np.matrix(A_i)
                A_ij = A_im - A_im.transpose()

                returns.append(np.array(A_ij))

                # todo - vectorize the differences!  Faster and less likely to give errors.

                if compute_uncertainty:
                    dA_ij = np.zeros([K, K], dtype=np.float64)

                    for i in range(0, K):
                        for j in range(0, K):
                            try:
                                dA_ij[i, j] = np.sqrt(
                                    + A_i[i] * Theta_ij[i, i] * A_i[i]
                                      - A_i[i] * Theta_ij[i, j] * A_i[j]
                                      - A_i[i] * Theta_ij[i, K + i] * A_i[i]
                                      + A_i[i] * Theta_ij[i, K + j] * A_i[j]
                                      - A_i[j] * Theta_ij[j, i] * A_i[i]
                                      + A_i[j] * Theta_ij[j, j] * A_i[j]
                                      + A_i[j] * Theta_ij[j, K + i] * A_i[i]
                                      - A_i[j] * Theta_ij[j, K + j] * A_i[j]
                                      - A_i[i] * Theta_ij[K + i, i] * A_i[i]
                                      + A_i[i] * Theta_ij[K + i, j] * A_i[j]
                                      + A_i[i] * Theta_ij[K + i, K + i] * A_i[i]
                                      - A_i[i] * Theta_ij[K + i, K + j] * A_i[j]
                                      + A_i[j] * Theta_ij[K + j, i] * A_i[i]
                                      - A_i[j] * Theta_ij[K + j, j] * A_i[j]
                                      - A_i[j] * Theta_ij[K + j, K + i] * A_i[i]
                                      + A_i[j] * Theta_ij[K + j, K + j] * A_i[j]
                                      )
                            except:
                                dA_ij[i, j] = 0.0

                    returns.append(dA_ij)

                if return_theta:
                    returns.append(Theta_ij)

        return returns


    #=========================================================================
    def computeMultipleExpectations(self, A_in, u_n, compute_uncertainty=True,
                                    uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the expectations of multiple observables of phase space functions.

        Compute the expectations of multiple observables of phase space functions.
        [A_0(x),A_1(x),...,A_i(x)] along with the covariances of their estimates.  The state is specified by
        the choice of u_n, which is the energy of the n samples evaluated at a the chosen state.

        Parameters
        ----------
        A_in : np.ndarray, float, shape=(I, k, N)
            A_in[i,n] = A_i(x_n), the value of phase observable i for configuration n at state of interest
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential of configuration n at the state of interest
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences.

        Returns
        -------

        A_i : np.ndarray, float, shape=(I)
            A_i[i] is the estimate for the expectation of A_i(x) at the state specified by u_kn
        d2A_ij : np.ndarray, float, shape=(I, I)
            d2A_ij[i,j] is the COVARIANCE in the estimates of A_i[i] and A_i[j],
            not the square root of the covariance

        Notes
        -----
        Not fully tested!

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_kn, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> A_in = np.array([x_kn,x_kn**2,x_kn**3])
        >>> u_n = u_kn[0,:]
        >>> [A_i, d2A_ij] = mbar.computeMultipleExpectations(A_in, u_kn)

        """

        # Retrieve N and K for convenience.
        I = A_in.shape[0]  # number of observables
        K = self.K
        N = self.N  # N is total number of samples

        if len(np.shape(A_in)) == 3:
            A_in_old = A_in.copy()
            A_in = np.zeros([I, N], np.float64)
            for i in range(I):
                A_in[i,:] = kn_to_n(A_in_old[i, :, :], N_k=self.N_k)

        A_min = np.zeros([I], dtype=np.float64)
        for i in range(I):
            A_min[i] = np.min(A_in[i, :]) #find the minimum
            A_in[i, :] -= (A_min[i]-1)  #all now values will be positive so that we can work in logarithmic scale

        if len(np.shape(u_n)) == 2:
            u_n = kn_to_n(u_n, N_k=self.N_k)

        # Augment W_nk, N_k, and c_k for q_A(x) for the observables, with one
        # row for the specified state and I rows for the observable at that
        # state.
        # log weight matrix
        Log_W_nk = np.zeros([N, K + 1 + I], np.float64)
        W_nk = np.zeros([N, K + 1 + I], np.float64)  # weight matrix
        N_k = np.zeros([K + 1 + I], np.int32)  # counts
        f_k = np.zeros([K + 1 + I], np.float64)  # free energies

        # Fill in first section of matrix with existing q_k(x) from states.
        Log_W_nk[:, 0:K] = self.Log_W_nk
        W_nk[:, 0:K] = np.exp(self.Log_W_nk)
        N_k[0:K] = self.N_k
        f_k[0:K] = self.f_k

        # Compute row of W matrix for the extra state corresponding to u_kn.
        Log_W_nk[:, K]  = self._computeUnnormalizedLogWeights(u_n)
        f_k[K] = -_logsum(Log_W_nk[:, K])
        Log_W_nk[:, K] += f_k[K]

        # Compute the remaining rows/columns of W_nk and c_k for the
        # observables.
        for i in range(I):
            Log_W_nk[:, K+1+i] = np.log(A_in[i, :]) + Log_W_nk[:, K]
            f_k[K + 1 + i] = -_logsum(Log_W_nk[:, K + 1 + i])
            Log_W_nk[:, K + 1 + i] += f_k[K + 1 + i]    # normalize this row

        # Compute estimates.
        A_i = np.zeros([I], np.float64)
        for i in range(I):
            A_i[i] = np.exp(-f_k[K + 1 + i])

        if compute_uncertainty or return_theta:
            # Compute augmented asymptotic covariance matrix.
            W_nk = np.exp(Log_W_nk)
            Theta_ij = self._computeAsymptoticCovarianceMatrix(
                W_nk, N_k, method=uncertainty_method)

        if compute_uncertainty:
            # Compute estimates of statistical covariance
            # these variances will be the same whether or not we subtract a different constant from each A_i
            # todo: vectorize
            # compute the covariance component without doing the double loop
            d2A_ij = np.zeros([I, I], np.float64)
            for i in range(I):
                for j in range(I):
                    d2A_ij[i, j] = A_i[i] * A_i[j] * (Theta_ij[K + 1 + i, K + 1 + j] - Theta_ij[
                                                      K + 1 + i, K] - Theta_ij[K, K + 1 + j] + Theta_ij[K, K])

        # Now that covariances are computed, add the constants back to A_i that
        # were required to enforce positivity
        A_i = A_i + (A_min - 1)

        returns = []
        returns.append(A_i)

        if compute_uncertainty:
            returns.append(d2A_ij)

        if return_theta:
            returns.append(Theta_ij)

        # Return expectations and uncertainties.
        return returns

    #=========================================================================
    def computeOverlap(self, output='scalar'):
        """Compute estimate of overlap matrix between the states.

        Returns
        -------
        O : np.ndarray, float, shape=(K, K)
            estimated state overlap matrix: O[i,j] is an estimate
            of the probability of observing a sample from state i in state j

        Parameters
        ----------
        output : string, optional
            One of 'scalar', 'matrix', 'eigenvalues', 'all', specifying
        what measure of overlap to return

        Notes
        -----

        W.T * W \approx \int (p_i p_j /\sum_k N_k p_k)^2 \sum_k N_k p_k dq^N
                      = \int (p_i p_j /\sum_k N_k p_k) dq^N

        Multiplying elementwise by N_i, the elements of row i give the probability
        for a sample from state i being observed in state j.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_kn, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> O_ij = mbar.computeOverlap()
        """

        W = np.matrix(self.getWeights(), np.float64)
        O = np.multiply(self.N_k, W.T * W)
        (eigenval, eigevec) = linalg.eig(O)
        # sort in descending order
        eigenval = np.sort(eigenval)[::-1]
        overlap_scalar = 1 - eigenval[1]
        if (output == 'scalar'):
            return overlap_scalar
        elif (output == 'eigenvalues'):
            return eigenval
        elif (output == 'matrix'):
            return O
        elif (output == 'all'):
            return overlap_scalar, eigenval, O

    #=========================================================================
    def computePerturbedExpectation(self, u_n, A_n, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the expectation of an observable of phase space function A(x) for a single new state.

        Parameters
        ----------
        u_n : np.ndarray, float, shape=(K, N_max)
            u_n[n] = u(x_n) - the energy of the new state at all N samples previously sampled.
        A_n : np.ndarray, float, shape=(K, N_max)
            A_n[n] = A(x_n) - the phase space function of the new state at all N samples previously sampled.  If this does NOT depend on state (e.g. position), it's simply the value of the observation.  If it DOES depend on the current state, then the observables from the previous states need to be reevaluated at THIS state.
        compute_uncertainty : bool, optional
            If False, the uncertainties will not be computed (default: True)
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences.


        Returns
        -------
        A : float
            A is the estimate for the expectation of A(x) for the specified state
        dA : float
            dA is uncertainty estimate for A

        Notes
        -----
        See Section IV of [1].
        # Compute estimators and uncertainty.
        #A = sum(W_n[:,K] * A_n[:]) # Eq. 15 of [1]
        #dA = abs(A) * np.sqrt(Theta_ij[K,K] + Theta_ij[K+1,K+1] - 2.0 * Theta_ij[K,K+1]) # Eq. 16 of [1]
        """

        if len(np.shape(u_n)) == 2:
            u_n = kn_to_n(u_n, N_k=self.N_k)

        if len(np.shape(A_n)) == 2:
            A_n = kn_to_n(A_n, N_k=self.N_k)

        # Convert to np matrix.
        A_n = np.array(A_n, dtype=np.float64)

        # Retrieve N and K for convenience.
        N = self.N
        K = self.K

        # Make A_k all positive so we can operate logarithmically for
        # robustness
        A_min = np.min(A_n)
        A_n = A_n - (A_min - 1)

        # Augment W_nk, N_k, and c_k for q_A(x) for the observable, with one
        # extra row/column for the specified state (Eq. 13 of [1]).
        # weight matrix
        Log_W_nk = np.zeros([N, K + 2], dtype=np.float64)
        N_k = np.zeros([K + 2], dtype=np.int32)  # counts
        f_k = np.zeros([K + 2], dtype=np.float64)  # free energies

        # Fill in first K states with existing q_k(x) from states.
        Log_W_nk[:, 0:K] = self.Log_W_nk
        N_k[0:K] = self.N_k

        # compute the free energy of the additional state
        log_w_n = self._computeUnnormalizedLogWeights(u_n)
        # Compute free energies
        f_k[K] = -_logsum(log_w_n)
        Log_W_nk[:, K] = log_w_n + f_k[K]

        # compute the observable at this state
        Log_W_nk[:, K + 1] = np.log(A_n) + Log_W_nk[:, K]
        f_k[K + 1] = -_logsum(Log_W_nk[:, K + 1])
        Log_W_nk[:, K + 1] += f_k[K + 1]              # normalize the row
        A = np.exp(-f_k[K + 1])

        if (compute_uncertainty or return_theta):
            # Compute augmented asymptotic covariance matrix.
            Theta_ij = self._computeAsymptoticCovarianceMatrix(
                np.exp(Log_W_nk), N_k, method=uncertainty_method)

        if (compute_uncertainty):
            dA = np.abs(A) * np.sqrt(
                Theta_ij[K + 1, K + 1] + Theta_ij[K, K] - 2.0 * Theta_ij[K, K + 1])  # Eq. 16 of [1]

        # shift answers back with the offset now that variances are computed
        A += (A_min - 1)

        returns = []
        returns.append(A)

        if (compute_uncertainty):
            returns.append(dA)

        if (return_theta):
            returns.append(Theta_ij)

        # Return expectations and uncertainties.
        return returns

    #=========================================================================
    def computePerturbedFreeEnergies(self, u_ln, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the free energies for a new set of states.

        Here, we desire the free energy differences among a set of new states, as well as the uncertainty estimates in these differences.

        Parameters
        ----------
        u_ln : np.ndarray, float, shape=(L, Nmax)
            u_ln[l,n] is the reduced potential energy of uncorrelated
            configuration n evaluated at new state k.  Can be completely indepednent of the original number of states.
        compute_uncertainty : bool, optional
            If False, the uncertainties will not be computed (default: True)
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)

        Returns
        -------
        Deltaf_ij : np.ndarray, float, shape=(L, L)
            Deltaf_ij[i,j] = f_j - f_i, the dimensionless free energy difference between new states i and j
        dDeltaf_ij : np.ndarray, float, shape=(L, L)
            dDeltaf_ij[i,j] is the estimated statistical uncertainty in Deltaf_ij[i,j]

        Examples
        --------
        >>> from pymbar import testsystems
        >>> [x_n, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> [Deltaf_ij, dDeltaf_ij] = mbar.computePerturbedFreeEnergies(u_kn)
        """

        # Convert to np matrix.
        u_ln = np.array(u_ln, dtype=np.float64)

        # Get the dimensions of the matrix of reduced potential energies, and convert if necessary
        if len(np.shape(u_ln)) == 3:
            u_ln = kln_to_kn(u_ln, N_k=self.N_k)

        [L, N] = u_ln.shape

        # Check dimensions.
        if (N < self.N):
            raise "There seems to be too few samples in u_kn. You must evaluate at the new potential with all of the samples used originally."

        # Retrieve N and K for convenience.
        N = self.N
        K = self.K

        # Augment W_nk, N_k, and c_k for the new states.
        W_nk = np.zeros([N, K + L], dtype=np.float64)  # weight matrix
        N_k = np.zeros([K + L], dtype=np.int32)  # counts
        f_k = np.zeros([K + L], dtype=np.float64)  # free energies

        # Fill in first half of matrix with existing q_k(x) from states.
        W_nk[:, 0:K] = np.exp(self.Log_W_nk)
        N_k[0:K] = self.N_k
        f_k[0:K] = self.f_k

        # Compute normalized weights.
        for l in range(0, L):
            # Compute unnormalized log weights.
            log_w_n = self._computeUnnormalizedLogWeights(u_ln[l, :])
            # Compute free energies
            f_k[K + l] = - _logsum(log_w_n)
            # Store normalized weights.  Keep in exponential not log form
            # because we will not store W_nk
            W_nk[:, K + l] = np.exp(log_w_n + f_k[K + l])

        if (compute_uncertainty or return_theta):
            # Compute augmented asymptotic covariance matrix.
            Theta_ij = self._computeAsymptoticCovarianceMatrix(
                W_nk, N_k, method = uncertainty_method)

        # Compute matrix of free energy differences between states and
        # associated uncertainties.
        # makes matrix operations easier to recast
        f_k = np.matrix(f_k[K:K + L])

        Deltaf_ij = f_k - f_k.transpose()

        returns = []
        returns.append(Deltaf_ij)

        if (compute_uncertainty):
            diag = Theta_ij.diagonal()
            dii = diag[0, K:K + L]
            d2DeltaF = dii + dii.transpose() - 2 * Theta_ij[K:K + L, K:K + L]

            # check for any numbers below zero.
            if (np.any(d2DeltaF < 0.0)):
                if(np.any(d2DeltaF) < warning_cutoff):
                    print "A squared uncertainty is negative.  d2DeltaF = %e" % d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)]
                else:
                    d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)] = 0.0

            # take the square root of entries of the matrix
            dDeltaf_ij = np.sqrt(d2DeltaF)

            returns.append(dDeltaf_ij)

        if (return_theta):
            returns.append(Theta_ij)

        # Return matrix of free energy differences and uncertainties.
        return returns

    def computeEntropyAndEnthalpy(self, uncertainty_method=None, verbose=False, warning_cutoff=1.0e-10):
        """Decompose free energy differences into enthalpy and entropy differences.

        Compute the decomposition of the free energy difference between
        states 1 and N into reduced free energy differences, reduced potential
        (enthalpy) differences, and reduced entropy (S/k) differences.

        Parameters
        ----------
        uncertainty_method : string , optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)

        Returns
        -------
        Delta_f_ij : np.ndarray, float, shape=(K, K)
            Delta_f_ij[i,j] is the dimensionless free energy difference f_j - f_i
        dDelta_f_ij : np.ndarray, float, shape=(K, K)
            uncertainty in Delta_f_ij
        Delta_u_ij : np.ndarray, float, shape=(K, K)
            Delta_u_ij[i,j] is the reduced potential energy difference u_j - u_i
        dDelta_u_ij : np.ndarray, float, shape=(K, K)
            uncertainty in Delta_f_ij
        Delta_s_ij : np.ndarray, float, shape=(K, K)
            Delta_s_ij[i,j] is the reduced entropy difference S/k between states i and j (s_j - s_i)
        dDelta_s_ij : np.ndarray, float, shape=(K, K)
            uncertainty in Delta_s_ij

        Notes
        -----
        This method is EXPERIMENTAL and should be used at your own risk.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_n, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> [Delta_f_ij, dDelta_f_ij, Delta_u_ij, dDelta_u_ij, Delta_s_ij, dDelta_s_ij] = mbar.computeEntropyAndEnthalpy()

        """

        if verbose:
            print "Computing average energy and entropy by MBAR."

        # Retrieve N and K for convenience.
        N = self.N
        K = self.K

        # Augment W_nk, N_k, and c_k for q_A(x) for the potential energies,
        # with one extra row/column for each state.
        # weight matrix
        Log_W_nk = np.zeros([N, K * 2], dtype=np.float64)
        N_k = np.zeros([K * 2], dtype=np.int32)  # counts
        # "free energies" of average states
        f_k = np.zeros(K, dtype=np.float64)

        # Fill in first half of matrix with existing q_k(x) from states.
        Log_W_nk[:, 0:K] = self.Log_W_nk
        N_k[0:K] = self.N_k

        # Compute the remaining rows/columns of W_nk and c_k for the potential
        # energy observable.

        u_min = self.u_kn.min()
        u_i = np.zeros([K], dtype=np.float64)
        for l in range(0, K):
            u_kn = self.u_kn[l, :] - (u_min-1)  # all positive now!  Subtracting off arbitrary constants doesn't affect results
            # since they are all differences.
            # Compute unnormalized weights.
            # A(x_n) exp[f_{k} - q_{k}(x_n)] / \sum_{k'=1}^K N_{k'} exp[f_{k'} - q_{k'}(x_n)]
            # harden for over/underflow with logarithms

            Log_W_nk[:, K + l] = np.log(u_kn) + self.Log_W_nk[:, l]

            f_k[l] = -_logsum(Log_W_nk[:, K + l])
            Log_W_nk[:, K + l] += f_k[l]              # normalize the row
            u_i[l] = np.exp(-f_k[l])

            # print "MBAR u_i[%d]: %10.5f,%10.5f" % (l,u_i[l]+u_min, u_i[l])

        # Compute augmented asymptotic covariance matrix.
        W_nk = np.exp(Log_W_nk)
        Theta_ij = self._computeAsymptoticCovarianceMatrix(
            W_nk, N_k, method=uncertainty_method)

        # Compute estimators and uncertainties.
        dDelta_f_ij = np.zeros([K, K], dtype=np.float64)
        dDelta_u_ij = np.zeros([K, K], dtype=np.float64)
        dDelta_s_ij = np.zeros([K, K], dtype=np.float64)

        # Compute reduced free energy difference.
        f_k = np.matrix(self.f_k)
        Delta_f_ij = f_k - f_k.transpose()

        # Compute reduced enthalpy difference.
        u_k = np.matrix(u_i)
        Delta_u_ij = u_k - u_k.transpose()

        # Compute reduced entropy difference
        s_k = u_k - f_k
        Delta_s_ij = s_k - s_k.transpose()

        # compute uncertainty matrix in free energies:
        # d2DeltaF = Theta_ij[i,i] + Theta_ij[j,j] - 2.0 * Theta_ij[i,j]

        diag = Theta_ij.diagonal()
        dii = diag[0:K, 0:K]
        d2DeltaF = dii + dii.transpose() - 2 * Theta_ij[0:K, 0:K]

        # check for any numbers below zero.
        if (np.any(d2DeltaF < 0.0)):
            if(np.any(d2DeltaF) < warning_cutoff):
                # Hmm.  Will this print correctly?
                print "A squared uncertainty is negative.  d2DeltaF = %e" % d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)]
            else:
                d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)] = 0.0

        # take the square root of the entries of matrix
        dDelta_f_ij = np.sqrt(d2DeltaF)
        # TODO -- vectorize this calculation for entropy and enthalpy!

        for i in range(0, K):
            for j in range(0, K):
                try:
                    dDelta_u_ij[i, j] = math.sqrt(
                        + u_i[i] * Theta_ij[i, i] * u_i[i] - u_i[i] * Theta_ij[i, j] * u_i[j] - u_i[
                            i] * Theta_ij[i, K + i] * u_i[i] + u_i[i] * Theta_ij[i, K + j] * u_i[j]
                        - u_i[j] * Theta_ij[j, i] * u_i[i] + u_i[j] * Theta_ij[j, j] * u_i[j] + u_i[
                            j] * Theta_ij[j, K + i] * u_i[i] - u_i[j] * Theta_ij[j, K + j] * u_i[j]
                        - u_i[i] * Theta_ij[K + i, i] * u_i[i] + u_i[i] * Theta_ij[K + i, j] * u_i[
                            j] + u_i[i] * Theta_ij[K + i, K + i] * u_i[i] - u_i[i] * Theta_ij[K + i, K + j] * u_i[j]
                        + u_i[j] * Theta_ij[K + j, i] * u_i[i] - u_i[j] * Theta_ij[K + j, j] * u_i[
                            j] - u_i[j] * Theta_ij[K + j, K + i] * u_i[i] + u_i[j] * Theta_ij[K + j, K + j] * u_i[j]
                    )
                except:
                    dDelta_u_ij[i, j] = 0.0

                # Compute reduced entropy difference.
                try:
                    dDelta_s_ij[i, j] = math.sqrt(
                        + (u_i[i] - 1) * Theta_ij[i, i] * (u_i[i] - 1) + (u_i[i] - 1) * Theta_ij[i, j] * (-u_i[j] + 1) + (
                            u_i[i] - 1) * Theta_ij[i, K + i] * (-u_i[i]) + (u_i[i] - 1) * Theta_ij[i, K + j] * u_i[j]
                        + (-u_i[j] + 1) * Theta_ij[j, i] * (u_i[i] - 1) + (-u_i[j] + 1) * Theta_ij[j, j] * (-u_i[j] + 1) +
                        (-u_i[j] + 1) * Theta_ij[j, K + i] * (-u_i[i]) +
                        (-u_i[j] + 1) * Theta_ij[j, K + j] * u_i[j]
                        + (-u_i[i]) * Theta_ij[K + i, i] * (u_i[i] - 1) + (-u_i[i]) * Theta_ij[K + i, j] * (-u_i[j] + 1) +
                        (-u_i[i]) * Theta_ij[K + i, K + i] * (-u_i[i]) +
                        (-u_i[i]) * Theta_ij[K + i, K + j] * u_i[j]
                        + u_i[j] * Theta_ij[K + j, i] * (u_i[i] - 1) + u_i[j] * Theta_ij[K + j, j] * (-u_i[j] + 1) + u_i[
                            j] * Theta_ij[K + j, K + i] * (-u_i[i]) + u_i[j] * Theta_ij[K + j, K + j] * u_i[j]
                    )
                except:
                    dDelta_s_ij[i, j] = 0.0

        # Return expectations and uncertainties.
        return (Delta_f_ij, dDelta_f_ij, Delta_u_ij, dDelta_u_ij, Delta_s_ij, dDelta_s_ij)
    #=========================================================================

    def computePMF(self, u_n, bin_n, nbins, uncertainties='from-lowest', pmf_reference=None):
        """Compute the free energy of occupying a number of bins.

        This implementation computes the expectation of an indicator-function observable for each bin.

        Parameters
        ----------
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential energy of snapshot n of state k for which the PMF is to be computed.
        bin_n : np.ndarray, float, shape=(N)
            bin_n[n] is the bin index of snapshot n of state k.  bin_n can assume a value in range(0,nbins)
        nbins : int
            The number of bins
        uncertainties : string, optional
            Method for reporting uncertainties (default: 'from-lowest')
            'from-lowest' - the uncertainties in the free energy difference with lowest point on PMF are reported
            'from-reference' - same as from lowest, but from a user specified point
            'from-normalization' - the normalization \sum_i p_i = 1 is used to determine uncertainties spread out through the PMF
            'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        Returns
        -------
        f_i : np.ndarray, float, shape=(K)
            f_i[i] is the dimensionless free energy of state i, relative to the state of lowest free energy
        df_i : np.ndarray, float, shape=(K)
            df_i[i] is the uncertainty in the difference of f_i with respect to the state of lowest free energy

        Notes
        -----
        All bins must have some samples in them from at least one of the states -- this will not work if bin_n.sum(0) == 0. Empty bins should be removed before calling computePMF().
        This method works by computing the free energy of localizing the system to each bin for the given potential by aggregating the log weights for the given potential.
        To estimate uncertainties, the NxK weight matrix W_nk is augmented to be Nx(K+nbins) in order to accomodate the normalized weights of states where
        the potential is given by u_kn within each bin and infinite potential outside the bin.  The uncertainties with respect to the bin of lowest free energy
        are then computed in the standard way.

        WARNING
        This method is EXPERIMENTAL and should be used at your own risk.

        Examples
        --------

        >>> # Generate some test data
        >>> from pymbar import testsystems
        >>> [x_n, u_kn, N_k] = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> # Initialize MBAR on data.
        >>> mbar = MBAR(u_kn, N_k)
        >>> # Select the potential we want to compute the PMF for (here, condition 0).
        >>> u_n = u_kn[0, :]
        >>> # Sort into nbins equally-populated bins
        >>> nbins = 10 # number of equally-populated bins to use
        >>> import numpy as np
        >>> N_tot = N_k.sum()
        >>> x_n_sorted = np.sort(x_n) # unroll to n-indices
        >>> bins = np.append(x_n_sorted[0::(N_tot/nbins)], x_n_sorted.max()+0.1)
        >>> bin_widths = bins[1:] - bins[0:-1]
        >>> bin_n = np.zeros(x_n.shape, np.int32)
        >>> bin_n = np.digitize(x_n, bins) - 1
        >>> # Compute PMF for these unequally-sized bins.
        >>> [f_i, df_i] = mbar.computePMF(u_n, bin_n, nbins)
        >>> # If we want to correct for unequally-spaced bins to get a PMF on uniform measure
        >>> f_i_corrected = f_i - np.log(bin_widths)

        """

        # Verify that no PMF bins are empty -- we can't deal with empty bins,
        # because the free energy is infinite.
        for i in range(nbins):
            if np.sum(bin_n == i) == 0:
                raise ParameterError(
                    "At least one bin in provided bin_n argument has no samples.  All bins must have samples for free energies to be finite.  Adjust bin sizes or eliminate empty bins to ensure at least one sample per bin.")
        K = self.K

        # Compute unnormalized log weights for the given reduced potential
        # u_n.
        log_w_n = self._computeUnnormalizedLogWeights(u_n)

        if len(np.shape(u_n)) == 2:
            u_n = kn_to_n(u_n, N_k = self.N_k)

        if len(np.shape(bin_n)) == 2:
            bin_n = kn_to_n(bin_n, N_k = self.N_k)

        # Compute the free energies for these states.
        f_i = np.zeros([nbins], np.float64)
        df_i = np.zeros([nbins], np.float64)
        for i in range(nbins):
            # Get linear n-indices of samples that fall in this bin.
            indices = np.where(bin_n == i)

            # Compute dimensionless free energy of occupying state i.
            f_i[i] = - _logsum(log_w_n[indices])

        # Compute uncertainties by forming matrix of W_nk.
        N_k = np.zeros([self.K + nbins], np.int32)
        N_k[0:K] = self.N_k
        W_nk = np.zeros([self.N, self.K + nbins], np.float64)
        W_nk[:, 0:K] = np.exp(self.Log_W_nk)
        for i in range(nbins):
            # Get indices of samples that fall in this bin.
            indices = np.where(bin_n == i)

            # Compute normalized weights for this state.
            W_nk[indices, K + i] = np.exp(log_w_n[indices] + f_i[i])

        # Compute asymptotic covariance matrix using specified method.
        Theta_ij = self._computeAsymptoticCovarianceMatrix(W_nk, N_k)

        if (uncertainties == 'from-lowest') or (uncertainties == 'from-specified'):
            # Report uncertainties in free energy difference from lowest point
            # on PMF.

            if (uncertainties == 'from-lowest'):
                # Determine bin index with lowest free energy.
                j = f_i.argmin()
            elif (uncertainties == 'from-specified'):
                if pmf_reference == None:
                    raise ParameterError(
                        "no reference state specified for PMF using uncertainties = from-reference")
                else:
                    j = pmf_reference
            # Compute uncertainties with respect to difference in free energy
            # from this state j.
            for i in range(nbins):
                df_i[i] = math.sqrt(
                    Theta_ij[K + i, K + i] + Theta_ij[K + j, K + j] - 2.0 * Theta_ij[K + i, K + j])

            # Shift free energies so that state j has zero free energy.
            f_i -= f_i[j]

            # Return dimensionless free energy and uncertainty.
            return (f_i, df_i)

        elif (uncertainties == 'all-differences'):
            # Report uncertainties in all free energy differences.

            diag = Theta_ij.diagonal()
            dii = diag[K, K + nbins]
            d2f_ij = dii + \
                dii.transpose() - 2 * Theta_ij[K:K + nbins, K:K + nbins]

            # unsquare uncertainties
            df_ij = np.sqrt(d2f_ij)

            # Return dimensionless free energy and uncertainty.
            return (f_i, df_ij)

        elif (uncertainties == 'from-normalization'):
            # Determine uncertainties from normalization that \sum_i p_i = 1.

            # Compute bin probabilities p_i
            p_i = np.exp(-f_i - _logsum(-f_i))

            # todo -- eliminate triple loop over nbins!
            # Compute uncertainties in bin probabilities.
            d2p_i = np.zeros([nbins], np.float64)
            for k in range(nbins):
                for i in range(nbins):
                    for j in range(nbins):
                        delta_ik = 1.0 * (i == k)
                        delta_jk = 1.0 * (j == k)
                        d2p_i[k] += p_i[k] * (p_i[i] - delta_ik) * p_i[
                            k] * (p_i[j] - delta_jk) * Theta_ij[K + i, K + j]

            # Transform from d2p_i to df_i
            d2f_i = d2p_i / p_i ** 2
            df_i = np.sqrt(d2f_i)

            # return free energy and uncertainty
            return (f_i, df_i)

        else:
            raise "Uncertainty method '%s' not recognized." % uncertainties

        return

    #=========================================================================
    def computePMF_states(self, u_n, bin_n, nbins):
        """Compute the free energy of occupying a number of bins.

        This implementation defines each bin as a separate thermodynamic state.

        Parameters
        ----------
        u_n : np.ndarray, float, shape=(K, N)
            u_n[n] is the reduced potential energy of snapshot n for which the PMF is to be computed.
        bin_n : np.ndarray, int, shape=(N)
            bin_n[n] is the bin index of snapshot n.  bin_n can assume a value in range(0,nbins)
        nbins : int
            The number of bins
        fmax : float, optional
            The maximum value of the free energy, used for an empty bin (default: 1000)

        Returns
        -------
        f_i : np.ndarray, float, shape=(K)
            f_i[i] is the dimensionless free energy of state i, relative to the state of lowest free energy
        d2f_ij : np.ndarray, float, shape=(K)
            d2f_ij[i,j] is the uncertainty in the difference of (f_i - f_j)

        Notes
        -----
        All bins must have some samples in them from at least one of the states -- this will not work if bin_n.sum(0) == 0. Empty bins should be removed before calling computePMF().
        This method works by computing the free energy of localizing the system to each bin for the given potential by aggregating the log weights for the given potential.
        To estimate uncertainties, the NxK weight matrix W_nk is augmented to be Nx(K+nbins) in order to accomodate the normalized weights of states where
        the potential is given by u_kn within each bin and infinite potential outside the bin.  The uncertainties with respect to the bin of lowest free energy
        are then computed in the standard way.

        WARNING!
        This method is EXPERIMENTAL and should be used at your own risk.

        """

        # Verify that no PMF bins are empty -- we can't deal with empty bins,
        # because the free energy is infinite.
        for i in range(nbins):
            if np.sum(bin_n == i) == 0:
                raise ParameterError(
                    "At least one bin in provided bin_n argument has no samples.  All bins must have samples for free energies to be finite.  Adjust bin sizes or eliminate empty bins to ensure at least one sample per bin.")

        K = self.K

        # Unroll to n-indices
        log_w_n = self._computeUnnormalizedLogWeights(u_n)

        # Compute the free energies for these states.
        f_i = np.zeros([nbins], np.float64)
        for i in range(nbins):
            # Get linear n-indices of samples that fall in this bin.
            indices = np.where(bin_n[self.indices] == i)[0]

            # Sanity check.
            if (len(indices) == 0):
                raise "WARNING: bin %d has no samples -- all bins must have at least one sample." % i

            # Compute dimensionless free energy of occupying state i.
            f_i[i] = - _logsum(log_w_n[indices])

        # Shift so that f_i.min() = 0
        f_i_min = f_i.min()
        f_i -= f_i.min()

        if self.verbose:
            print "bins f_i = "
            print f_i

        # Compute uncertainties by forming matrix of W_nk.
        if self.verbose:
            print "Forming W_nk matrix..."
        N_k = np.zeros([self.K + nbins], np.int32)
        N_k[0:K] = self.N_k
        W_nk = np.zeros([self.N, self.K + nbins], np.float64)
        W_nk[:, 0:K] = np.exp(self.Log_W_nk)
        for i in range(nbins):
            # Get indices of samples that fall in this bin.
            indices = np.where(bin_n[self.indices] == i)[0]

            if self.verbose:
                print "bin %5d count = %10d" % (i, len(indices))

            # Compute normalized weights for this state.
            W_nk[indices, K + i] = np.exp(
                log_w_n[indices] + f_i[i] + f_i_min)

        # Compute asymptotic covariance matrix using specified method.
        Theta_ij = self._computeAsymptoticCovarianceMatrix(W_nk, N_k)

        # Compute uncertainties with respect to difference in free energy from
        # this state j.
        diag = Theta_ij.diagonal()
        dii = diag[0, K:K + nbins]
        d2f_ij = dii + dii.transpose() - 2 * Theta_ij[K:K + nbins, K:K + nbins]

        # Return dimensionless free energy and uncertainty.
        return (f_i, d2f_ij)

    #=========================================================================
    # PRIVATE METHODS - INTERFACES ARE NOT EXPORTED
    #=========================================================================

    def _pseudoinverse(self, A, tol=1.0e-10):
        """
        Compute the Moore-Penrose pseudoinverse.

        REQUIRED ARGUMENTS
          A (np KxK matrix) - the square matrix whose pseudoinverse is to be computed

        RETURN VALUES
          Ainv (np KxK matrix) - the pseudoinverse

        OPTIONAL VALUES
          tol - the tolerance (relative to largest magnitude singlular value) below which singular values are to not be include in forming pseudoinverse (default: 1.0e-10)

        NOTES
          This implementation is provided because the 'pinv' function of np is broken in the version we were using.

        TODO
          Can we get rid of this and use np.linalg.pinv instead?

        """

        # DEBUG
        # TODO: Should we use pinv, or _pseudoinverse?
        # return np.linalg.pinv(A)

        # Get size
        [M, N] = A.shape
        if N != M:
            raise "pseudoinverse can only be computed for square matrices: dimensions were %d x %d" % (
                M, N)

        # Make sure A contains no nan.
        if(np.any(np.isnan(A))):
            print "attempted to compute pseudoinverse of A ="
            print A
            raise ParameterError("A contains nan.")

        # DEBUG
        diagonal_loading = False
        if diagonal_loading:
            # Modify matrix by diagonal loading.
            eigs = linalg.eigvalsh(A)
            most_negative_eigenvalue = eigs.min()
            if (most_negative_eigenvalue < 0.0):
                print "most negative eigenvalue = %e" % most_negative_eigenvalue
                # Choose loading value.
                gamma = -most_negative_eigenvalue * 1.05
                # Modify Theta by diagonal loading
                A += gamma * np.eye(A.shape[0])

        # Compute SVD of A.
        [U, S, Vt] = linalg.svd(A)

        # Compute pseudoinverse by taking square root of nonzero singular
        # values.
        Ainv = np.matrix(np.zeros([M, M], dtype=np.float64))
        for k in range(M):
            if (abs(S[k]) > tol * abs(S[0])):
                Ainv += (1.0/S[k]) * np.outer(U[:, k], Vt[k, :]).T

        return Ainv
    #=========================================================================

    def _zerosamestates(self, A):
        """
        zeros out states that should be identical

        REQUIRED ARGUMENTS

        A: the matrix whose entries are to be zeroed.

        """

        for pair in self.samestates:
            A[pair[0], pair[1]] = 0
            A[pair[1], pair[0]] = 0

    #=========================================================================
    def _computeAsymptoticCovarianceMatrix(self, W, N_k, method=None):
        """
        Compute estimate of the asymptotic covariance matrix.

        REQUIRED ARGUMENTS
          W (np.array of np.float of dimension [N,K]) - matrix of normalized weights (see Eq. 9 of [1]) - W[n,k] is the weight of snapshot n (n = 1..N) in state k
                                          Note that sum(W(:,k)) = 1 for any k = 1..K, and sum(N_k(:) .* W(n,:)) = 1 for any n.
          N_k (np.array of np.int32 of dimension [K]) - N_k[k] is the number of samples from state K

        RETURN VALUES
          Theta (KxK np float64 array) - asymptotic covariance matrix (see Eq. 8 of [1])

        OPTIONAL ARGUMENTS
          method (string) - if not None, specified method is used to compute asymptotic covariance method:
                            method must be one of ['generalized-inverse', 'svd', 'svd-ew', 'inverse', 'tan-HGH', 'tan', 'approximate']
                            If None is specified, 'svd-ew' is used.

        NOTES

        The computational costs of the various 'method' arguments varies:

          'generalized-inverse' currently requires computation of the pseudoinverse of an NxN matrix (where N is the total number of samples)
          'svd' computes the generalized inverse using the singular value decomposition -- this should be efficient yet accurate (faster)
          'svd-ev' is the same as 'svd', but uses the eigenvalue decomposition of W'W to bypass the need to perform an SVD (fastest)
          'inverse' only requires standard inversion of a KxK matrix (where K is the number of states), but requires all K states to be different
          'approximate' only requires multiplication of KxN and NxK matrices, but is an approximate underestimate of the uncertainty
          'tan' uses a simplified form that requires two pseudoinversions, but can be unstable
          'tan-HGH' makes weaker assumptions on 'tan' but can occasionally be unstable

        REFERENCE
          See Section II and Appendix D of [1].

        """

        # Set 'svd-ew' as default if uncertainty method specified as None.
        if method == None:
            method = 'svd-ew'

        # Get dimensions of weight matrix.
        [N, K] = W.shape

        # Check dimensions
        if(K != N_k.size):
            raise ParameterError(
                'W must be NxK, where N_k is a K-dimensional array.')
        if(np.sum(N_k) != N):
            raise ParameterError('W must be NxK, where N = sum_k N_k.')

        # Check to make sure the weight matrix W is properly normalized.
        tolerance = 1.0e-4  # tolerance for checking equality of sums

        column_sums = np.sum(W, axis=0)
        badcolumns = (np.abs(column_sums - 1) > tolerance)
        if np.any(badcolumns):
            which_badcolumns = np.arange(K)[badcolumns]
            firstbad = which_badcolumns[0]
            raise ParameterError(
                'Warning: Should have \sum_n W_nk = 1.  Actual column sum for state %d was %f. %d other columns have similar problems' %
                (firstbad, column_sums[firstbad], np.sum(badcolumns)))

        row_sums = np.sum(W * N_k, axis=1)
        badrows = (np.abs(row_sums - 1) > tolerance)
        if np.any(badrows):
            which_badrows = np.arange(N)[badrows]
            firstbad = which_badrows[0]
            raise ParameterError(
                'Warning: Should have \sum_k N_k W_nk = 1.  Actual row sum for sample %d was %f. %d other rows have similar problems' %
                (firstbad, row_sums[firstbad], np.sum(badrows)))

        # Compute estimate of asymptotic covariance matrix using specified
        # method.
        if method == 'generalized-inverse':
            # Use generalized inverse (Eq. 8 of [1]) -- most general
            # Theta = W' (I - W N W')^+ W

            # Construct matrices
            # Diagonal N_k matrix.
            Ndiag = np.matrix(np.diag(N_k), dtype=np.float64)
            W = np.matrix(W, dtype=np.float64)
            I = np.identity(N, dtype=np.float64)

            # Compute covariance
            Theta = W.T * self._pseudoinverse(I - W * Ndiag * W.T) * W

        elif method == 'inverse':
            # Use standard inverse method (Eq. D8 of [1]) -- only applicable if all K states are different
            # Theta = [(W'W)^-1 - N + 1 1'/N]^-1

            # Construct matrices
            # Diagonal N_k matrix.
            Ndiag = np.matrix(np.diag(N_k), dtype=np.float64)
            W = np.matrix(W, dtype=np.float64)
            I = np.identity(N, dtype=np.float64)
            # matrix of ones, times 1/N
            O = np.ones([K, K], dtype=np.float64) / float(N)

            # Make sure W is nonsingular.
            if (abs(linalg.det(W.T * W)) < tolerance):
                print "Warning: W'W appears to be singular, yet 'inverse' method of uncertainty estimation requires W contain no duplicate states."

            # Compute covariance
            Theta = ((W.T * W).I - Ndiag + O).I

        elif method == 'approximate':
            # Use fast approximate expression from Kong et al. -- this underestimates the true covariance, but may be a good approximation in some cases and requires no matrix inversions
            # Theta = P'P

            # Construct matrices
            W = np.matrix(W, dtype=np.float64)

            # Compute covariance
            Theta = W.T * W

        elif method == 'svd':
            # Use singular value decomposition based approach given in supplementary material to efficiently compute uncertainty
            # See Appendix D.1, Eq. D4 in [1].

            # Construct matrices
            Ndiag = np.matrix(np.diag(N_k), dtype=np.float64)
            W = np.matrix(W, dtype=np.float64)
            I = np.identity(K, dtype=np.float64)

            # Compute SVD of W
            [U, S, Vt] = linalg.svd(W)
            Sigma = np.matrix(np.diag(S))
            V = np.matrix(Vt).T

            # Compute covariance
            Theta = V * Sigma * self._pseudoinverse(
                I - Sigma * V.T * Ndiag * V * Sigma) * Sigma * V.T

        elif method == 'svd-ew':
            # Use singular value decomposition based approach given in supplementary material to efficiently compute uncertainty
            # The eigenvalue decomposition of W'W is used to forego computing the SVD.
            # See Appendix D.1, Eqs. D4 and D5 of [1].

            # Construct matrices
            Ndiag = np.matrix(np.diag(N_k), dtype=np.float64)
            W = np.matrix(W, dtype=np.float64)
            I = np.identity(K, dtype=np.float64)

            # Compute singular values and right singular vectors of W without using SVD
            # Instead, we compute eigenvalues and eigenvectors of W'W.
            # Note W'W = (U S V')'(U S V') = V S' U' U S V' = V (S'S) V'
            [S2, V] = linalg.eigh(W.T * W)
            # Set any slightly negative eigenvalues to zero.
            S2[np.where(S2 < 0.0)] = 0.0
            # Form matrix of singular values Sigma, and V.
            Sigma = np.matrix(np.diag(np.sqrt(S2)))
            V = np.matrix(V)

            # Compute covariance
            Theta = V * Sigma * self._pseudoinverse(
                I - Sigma * V.T * Ndiag * V * Sigma) * Sigma * V.T

        elif method == 'tan-HGH':
            # Use method suggested by Zhiqiang Tan without further simplification.
            # TODO: There may be a problem here -- double-check this.

            [N, K] = W.shape

            # Estimate O matrix from W'W.
            W = np.matrix(W, dtype=np.float64)
            O = W.T * W

            # Assemble the Lambda matrix.
            Lambda = np.matrix(np.diag(N_k), dtype=np.float64)

            # Identity matrix.
            I = np.matrix(np.eye(K), dtype=np.float64)

            # Compute H and G matrices.
            H = O * Lambda - I
            G = O - O * Lambda * O

            # Compute pseudoinverse of H
            Hinv = self._pseudoinverse(H)

            # Compute estimate of asymptotic covariance.
            Theta = Hinv * G * Hinv.T

        elif method == 'tan':
            # Use method suggested by Zhiqiang Tan.

            # Estimate O matrix from W'W.
            W = np.matrix(W, dtype=np.float64)
            O = W.T * W

            # Assemble the Lambda matrix.
            Lambda = np.matrix(np.diag(N_k), dtype=np.float64)

            # Compute covariance.
            Oinv = self._pseudoinverse(O)
            Theta = self._pseudoinverse(Oinv - Lambda)

        else:
            # Raise an exception.
            raise ParameterError('Method ' + method + ' unrecognized.')

        return Theta
    #=========================================================================

    def _initializeFreeEnergies(self, verbose=False, method='zeros'):
        """
        Compute an initial guess at the relative free energies.

        OPTIONAL ARGUMENTS
        verbose (boolean) - If True, will print debug information (default: False)
        method (string) - Method for initializing guess at free energies.
        'zeros' - all free energies are initially set to zero
        'mean-reduced-potential' - the mean reduced potential is used

        """

        if (method == 'zeros'):
            # Use zeros for initial free energies.
            if verbose:
                print "Initializing free energies to zero."
            self.f_k[:] = 0.0
        elif (method == 'mean-reduced-potential'):
            # Compute initial guess at free energies from the mean reduced
            # potential from each state
            if verbose:
                print "Initializing free energies with mean reduced potential for each state."
            means = np.zeros([self.K], float)
            for k in self.states_with_samples:
                means[k] = self.u_kn[k, 0:self.N_k[k]].mean()
            if (np.max(np.abs(means)) < 0.000001):
                print "Warning: All mean reduced potentials are close to zero. If you are using energy differences in the u_kln matrix, then the mean reduced potentials will be zero, and this is expected behavoir."
            self.f_k = means
        elif (method == 'BAR'):
            # For now, make a simple list of those states with samples.
            initialization_order = np.where(self.N_k > 0)[0]
            # Initialize all f_k to zero.
            self.f_k[:] = 0.0
            # Initialize the rest
            for index in range(0, np.size(initialization_order) - 1):
                k = initialization_order[index]
                l = initialization_order[index + 1]
                # forward work
                # here, we actually need to distinguish which states are which
                w_F = (
                    self.u_kn[l,self.x_kindices==k] - self.u_kn[k,self.x_kindices==k])
                    #self.u_kln[k, l, 0:self.N_k[k]] - self.u_kln[k, k, 0:self.N_k[k]])
                    # reverse work
                w_R = (
                    self.u_kn[k,self.x_kindices==l] - self.u_kn[l,self.x_kindices==l])
                    #self.u_kln[l, k, 0:self.N_k[l]] - self.u_kln[l, l, 0:self.N_k[l]])

                if (len(w_F) > 0 and len(w_R) > 0):
                    # BAR solution doesn't need to be incredibly accurate to
                    # kickstart NR.
                    import pymbar.bar
                    self.f_k[l] = self.f_k[k] + pymbar.bar.BAR(
                        w_F, w_R, relative_tolerance=0.000001, verbose=False, compute_uncertainty=False)
                else:
                    # no states observed, so we don't need to initialize this free energy anyway, as
                    # the solution is noniterative.
                    self.f_k[l] = 0

        else:
            # The specified method is not implemented.
            raise ParameterError('Method ' + method + ' unrecognized.')

        # Shift all free energies such that f_0 = 0.
        self.f_k[:] = self.f_k[:] - self.f_k[0]

        return


    def _computeUnnormalizedLogWeights(self, u_n):
        """
        Return unnormalized log weights.

        REQUIRED ARGUMENTS
          u_n (N np float64 array) - reduced potential energies at single state

        OPTIONAL ARGUMENTS

        RETURN VALUES
          log_w_n (N array) - unnormalized log weights of each of a number of states

        REFERENCE
          'log weights' here refers to \log [ \sum_{k=1}^K N_k exp[f_k - (u_k(x_n) - u(x_n)] ]
        """
        return -1. * mbar_solvers.logsumexp(self.f_k + u_n[:, np.newaxis] - self.u_kn.T, b=self.N_k, axis=1)
