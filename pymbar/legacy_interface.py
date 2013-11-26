# Copyright 2012 pymbar developers
#
# This file is part of pymbar
#
# pymbar is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymbar is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pymbar. If not, see http://www.gnu.org/licenses/.

"""
A pymbar1.0 compatible interface for pymbar2.0
"""

import numpy as np

from pymbar.utils import deprecated, convert_uijn_to_ukn, ensure_type, convert_Akn_to_An

import logging
logger = logging.getLogger(__name__)


class LegacyMBARMixin(object):
    """This class provides wrappers for pymbar1.0 member functions."""


    @deprecated()
    def getWeights(self):
        """Retrieve the weight matrix W_nk from the MBAR algorithm.
        
        Necessary because they are stored internally as log weights.

        Returns
        -------
        weights : np.ndarray, float, shape=(N, K)
            NxK matrix of weights in the MBAR covariance and averaging formulas

        """

        return np.exp(self.Log_W_nk)

    
    @deprecated()
    def getFreeEnergyDifferences(self, compute_uncertainty=True, uncertainty_method='svd-ew', warning_cutoff=1.0e-10, return_theta=False):
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

        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> [Deltaf_ij, dDeltaf_ij] = mbar.getFreeEnergyDifferences()

        """
        return self.get_free_energy_differences(compute_uncertainty=compute_uncertainty, uncertainty_method=uncertainty_method, warning_cutoff=warning_cutoff, return_theta=return_theta)

    @deprecated()
    def computeExpectations(self, A_kn, output='averages', compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the expectation of an observable of phase space function.
         
        Compute the expectation of an observable of phase space function 
        A(x) at all K states, including states for which no samples 
        were drawn. A may be a function of the state k.

        Parameters
        ----------
        A : np.ndarray, float          
            Two possibilities, depending on if the observable is a function of the state or not.
            either: not dependent on the state
            A_kn (KxN_max numpy float64 array) - A_kn[k,n] = A(x_kn)
            or: dependent on state
            A_kn (KxKxN_max numpy float64 array) - A_kn[k,l,n] = A(x_kn)
            where the 2nd dimension is the observable as a function of the state
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

        Returns
        -------
        A : np.ndarray, float
            if output is 'averages'
            A_i  (K numpy float64 array) -  A_i[k] is the estimate for the expectation of A(x) for state k.
            if output is 'differences'
            A_ij (K numpy float64 array) -  A_ij[i,j] is the difference in the estimates for the expectation of A(x).
        dA : np.ndarray, float
            dA_i  (K numpy float64 array) - dA_i[k] is uncertainty estimate (one standard deviation) for A_k[k]
            or
            dA_ij (K numpy float64 array) - dA_ij[i,j] is uncertainty estimate (one standard deviation) for the difference in A beteen i and j

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

        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> A_kn = x_kn
        >>> (A_ij, dA_ij) = mbar.computeExpectations(A_kn)
        >>> A_kn = u_kln
        >>> (A_ij, dA_ij) = mbar.computeExpectations(A_kn, output='differences')
        """
        
        input_rank = np.rank(A_kn)
        
        if input_rank == 2:
            A_kn = ensure_type(A_kn, 'float', 2, 'A_kn', shape=(self.n_states, self.N_max))
        elif input_rank == 3:
            A_kn = ensure_type(A_kn, 'float', 3, 'A_kn', shape=(self.n_states, self.n_states, self.N_max))
        else:
            raise(Exception("A_kn must have rank 2 or 3!"))

        if input_rank == 2:
            A_n = convert_Akn_to_An(A_kn, self.N_k)            
            if output == "differences":
                return self.compute_expectation_difference(A_n, compute_uncertainty=compute_uncertainty, uncertainty_method=uncertainty_method, warning_cutoff=warning_cutoff)
            elif output == "averages":
                return self.compute_expectation(A_n, compute_uncertainty=compute_uncertainty, uncertainty_method=uncertainty_method, warning_cutoff=warning_cutoff)
            else:
                raise(Exception("output must be either 'differences' or 'averages'"))
        
        elif input_rank == 3:
            A = convert_uijn_to_ukn(A_kn, self.N_k)
            # To do: write 3D expectation code.
            raise(Exception("Not implemented!"))


    @deprecated()
    def computeMultipleExpectations(self, A_ikn, u_kn, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the expectations of multiple observables of phase space functions.
        
        Compute the expectations of multiple observables of phase space functions.
        [A_0(x),A_1(x),...,A_n(x)] at single specified state, 
        along with the covariances of their estimates.  The state is specified by
        the choice of u_kn, which is the energy of the kxn samples evaluated at the chosen state.  Note that
        these variables A should NOT be functions of the state!


        Parameters
        ----------
        A_ikn : np.ndarray, float, shape=(I, k, N_max)       
            A_ikn[i,k,n] = A_i(x_kn), the value of phase observable i for configuration n at state k
        u_kn : np.ndarray, float, shape=(K, N_max)
            u_kn[k,n] is the reduced potential of configuration n gathered from state k, at the state of interest
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

        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> A_ikn = numpy.array([x_kn,x_kn**2,x_kn**3])
        >>> u_kn = u_kln[:,0,:]
        >>> [A_i, d2A_ij] = mbar.computeMultipleExpectations(A_ikn, u_kn)

        """

        raise(Exception("Not Implemented!"))

    @deprecated()
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

        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> O_ij = mbar.computeOverlap()
        """

        raise(Exception("Not Implemented!"))

    
    @deprecated()    
    def computePerturbedExpectation(self, u_kn, A_kn, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the expectation of an observable of phase space function A(x) for a single new state.

        Parameters
        ----------
        u_kn : np.ndarray, float, shape=(K, N_max)
            u_kn[k,n] = u(x_kn) - the energy of the new state at all N samples previously sampled.
        A_kn : np.ndarray, float, shape=(K, N_max)
            A_kn[k,n] = A(x_kn) - the phase space function of the new state at all N samples previously sampled.  If this does NOT depend on state (e.g. position), it's simply the value of the observation.  If it DOES depend on the current state, then the observables from the previous states need to be reevaluated at THIS state.
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
        #A = sum(W_nk[:,K] * A_n[:]) # Eq. 15 of [1]
        #dA = abs(A) * numpy.sqrt(Theta_ij[K,K] + Theta_ij[K+1,K+1] - 2.0 * Theta_ij[K,K+1]) # Eq. 16 of [1]
        """
        raise(Exception("Not Implemented!"))


    @deprecated()
    def computePerturbedFreeEnergies(self, u_kln, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10, return_theta=False):
        """Compute the free energies for a new set of states.
        
        Here, we desire the free energy differences among a set of new states, as well as the uncertainty estimates in these differences.

        Parameters
        ----------
        u_kln : np.ndarray, float, shape=(K, L, Nmax)
            u_kln[k,l,n] is the reduced potential energy of uncorrelated
            configuration n sampled from state k, evaluated at new state l.
            L need not be the same as K.
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
        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> [Deltaf_ij, dDeltaf_ij] = mbar.computePerturbedFreeEnergies(u_kln)
        """
        raise(Exception("Not Implemented!"))

    
    @deprecated()
    def computeEntropyAndEnthalpy(self, uncertainty_method=None, verbose=False, warning_cutoff=1.0e-10):
        """
        Compute the decomposition of the free energy difference between states 1 and N into reduced free energy differences, reduced potential (enthalpy) differences, and reduced entropy (S/k) differences.

        OPTINAL ARUMENTS
          uncertainty_method (string) - choice of method used to compute asymptotic covariance method, or None to use default
                            See help for computeAsymptoticCovarianceMatrix() for more information on various methods. (default: None)
          warning_cutoff (float) - warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
        RETURN VALUES
          Delta_f_ij (KxK numpy float matrix) - Delta_f_ij[i,j] is the dimensionless free energy difference f_j - f_i
          dDelta_f_ij (KxK numpy float matrix) - uncertainty in Delta_f_ij
          Delta_u_ij (KxK numpy float matrix) - Delta_u_ij[i,j] is the reduced potential energy difference u_j - u_i
          dDelta_u_ij (KxK numpy float matrix) - uncertainty in Delta_f_ij
          Delta_s_ij (KxK numpy float matrix) - Delta_s_ij[i,j] is the reduced entropy difference S/k between states i and j (s_j - s_i)
          dDelta_s_ij (KxK numpy float matrix) - uncertainty in Delta_s_ij

        WARNING
          This method is EXPERIMENTAL and should be used at your own risk.

        TEST

        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> [Delta_f_ij, dDelta_f_ij, Delta_u_ij, dDelta_u_ij, Delta_s_ij, dDelta_s_ij] = mbar.computeEntropyAndEnthalpy()

        """
        raise(Exception("Not Implemented!"))


    @deprecated()
    def computePMF(self, u_kn, bin_kn, nbins, uncertainties='from-lowest', pmf_reference=None):
        """
        Compute the free energy of occupying a number of bins.
        This implementation computes the expectation of an indicator-function observable for each bin.

        REQUIRED ARGUMENTS
          u_kn[k,n] is the reduced potential energy of snapshot n of state k for which the PMF is to be computed.
          bin_kn[k,n] is the bin index of snapshot n of state k.  bin_kn can assume a value in range(0,nbins)
          nbins is the number of bins

        OPTIONAL ARGUMENTS
          uncertainties (string) - choose method for reporting uncertainties (default: 'from-lowest')
            'from-lowest' - the uncertainties in the free energy difference with lowest point on PMF are reported
            'from-reference' - same as from lowest, but from a user specified point
            'from-normalization' - the normalization \sum_i p_i = 1 is used to determine uncertainties spread out through the PMF
            'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        RETURN VALUES
          f_i[i], i = 0..nbins - the dimensionless free energy of state i, relative to the state of lowest free energy
          df_i[i] is the uncertainty in the difference of f_i with respect to the state of lowest free energy

        NOTES
          All bins must have some samples in them from at least one of the states -- this will not work if bin_kn.sum(0) == 0. Empty bins should be removed before calling computePMF().
          This method works by computing the free energy of localizing the system to each bin for the given potential by aggregating the log weights for the given potential.
          To estimate uncertainties, the NxK weight matrix W_nk is augmented to be Nx(K+nbins) in order to accomodate the normalized weights of states where
          the potential is given by u_kn within each bin and infinite potential outside the bin.  The uncertainties with respect to the bin of lowest free energy
          are then computed in the standard way.

        WARNING
          This method is EXPERIMENTAL and should be used at your own risk.

        TEST

        >>> from pymbar.old import oldtestsystems
        >>> [x_kn, u_kln, N_k] = oldtestsystems.HarmonicOscillatorsSample(N_k=[100,100,100])
        >>> mbar = MBAR(u_kln, N_k)
        >>> u_kn = u_kln[0,:,:]
        >>> xmin = x_kn.min()
        >>> xmax = x_kn.max()
        >>> nbins = 10
        >>> dx = (xmax - xmin) * 1.00001 / float(nbins)
        >>> bin_kn = numpy.array((x_kn - xmin) / dx, numpy.int32)
        >>> [f_i, df_i] = mbar.computePMF(u_kn, bin_kn, nbins)

        """
        raise(Exception("Not Implemented!"))


    @deprecated()
    def computePMF_states(self, u_kn, bin_kn, nbins):
        """
        Compute the free energy of occupying a number of bins.
        This implementation defines each bin as a separate thermodynamic state.

        REQUIRED ARGUMENTS
          u_kn[k,n] is the reduced potential energy of snapshot n of state k for which the PMF is to be computed.
          bin_kn[k,n] is the bin index of snapshot n of state k.  bin_kn can assume a value in range(0,nbins)
          nbins is the number of bins

        OPTIONAL ARGUMENTS
          fmax is the maximum value of the free energy, used for an empty bin (default: 1000)

        RETURN VALUES
          f_i[i], i = 0..nbins - the dimensionless free energy of state i, relative to the state of lowest free energy
          d2f_ij[i,j] is the uncertainty in the difference of (f_i - f_j)

        NOTES
          All bins must have some samples in them from at least one of the states -- this will not work if bin_kn.sum(0) == 0. Empty bins should be removed before calling computePMF().
          This method works by computing the free energy of localizing the system to each bin for the given potential by aggregating the log weights for the given potential.
          To estimate uncertainties, the NxK weight matrix W_nk is augmented to be Nx(K+nbins) in order to accomodate the normalized weights of states where
          the potential is given by u_kn within each bin and infinite potential outside the bin.  The uncertainties with respect to the bin of lowest free energy
          are then computed in the standard way.

        WARNING
          This method is EXPERIMENTAL and should be used at your own risk.

        """
        raise(Exception("Not Implemented!"))
