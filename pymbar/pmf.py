##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2017 University of Colorado Boulder
# Copyright 2010-2017 Memorial Sloan-Kettering Cancer Center
# Portions of this software are Copyright (c) 2010-2016 University of Virginia
# Portions of this software are Copyright (c) 2006-2007 The Regents of the University of California.  All Rights Reserved.
# Portions of this software are Copyright (c) 2007-2008 Stanford University and Columbia University.
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################

"""
A module implementing calculation of potentials of mean force from biased simulations.

"""

import math
import numpy as np
import numpy.linalg as linalg
import pymbar
from pymbar import mbar_solvers
from pymbar.utils import kln_to_kn, kn_to_n, ParameterError, DataError, logsumexp, check_w_normalized

DEFAULT_SOLVER_PROTOCOL = mbar_solvers.DEFAULT_SOLVER_PROTOCOL

# =========================================================================
# PMF class definition
# =========================================================================


class PMF:
    """

    generating potentials of mean force with statistics.

    Notes
    -----
    Note that this method assumes the data are uncorrelated.

    Correlated data must be subsampled to extract uncorrelated (effectively independent) samples.

    References
    ----------

    [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
    J. Chem. Phys. 129:124105, 2008
    http://dx.doi.org/10.1063/1.2978177

    [2] Some paper.

    """
    # =========================================================================

    def __init__(self, u_kn, N_k, mbar_options = None, **kwargs):

        """Initialize multistate Bennett acceptance ratio (MBAR) on a set of simulation data.

        Upon initialization, the dimensionless free energies for all states are computed.
        This may take anywhere from seconds to minutes, depending upon the quantity of data.
        After initialization, the computed free energies may be obtained by a call to :func:`getFreeEnergyDifferences`,
        or expectation at any state of interest can be computed by calls to :func:`computeExpectations`.

        Parameters
        ----------
        u_kn : np.ndarray, float, shape=(K, N_max)
            ``u_kn[k,n]`` is the reduced potential energy of uncorrelated
            configuration n evaluated at state ``k``.

        N_k :  np.ndarray, int, shape=(K)
            ``N_k[k]`` is the number of uncorrelated snapshots sampled from state ``k``.
            Some may be zero, indicating that there are no samples from that state.

            We assume that the states are ordered such that the first ``N_k``
            are from the first state, the 2nd ``N_k`` the second state, and so
            forth. This only becomes important for BAR -- MBAR does not
            care which samples are from which state.  We should eventually
            allow this assumption to be overwritten by parameters passed
            from above, once ``u_kln`` is phased out.


        mbar_options: dictionary, with the following options supported by mbar (see MBAR documentation)
    
            maximum_iterations : int, optional
            relative_tolerance : float, optional
            verbosity : bool, optional
            initial_f_k : np.ndarray, float, shape=(K), optional
            solver_protocol : list(dict) or None, optional, default=None
            mbar_initialize : 'zeros' or 'BAR', optional, Default: 'zeros'
            x_indices : 

        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> pmf(u_kn, N_k)

        """
        for key, val in kwargs.items():
            print("Warning: parameter {}={} is unrecognized and unused.".format(key, val))

        # Store local copies of necessary data.
        # N_k[k] is the number of samples from state k, some of which might be zero.
        self.N_k = np.array(N_k, dtype=np.int64)
        self.N = np.sum(self.N_k)

        # u_kn[k,n] is the reduced potential energy of sample n evaluated at state k
        self.u_kn = np.array(u_kn, dtype=np.float64)

        K, N = np.shape(u_kn)

        if mbar_options:
            if mbar_options['verbose']:
                print("K (total states) = %d, total samples = %d" % (K, N))

        if np.sum(self.N_k) != N:
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
            self.x_kindices = np.arange(N, dtype=np.int64)
            Nsum = 0
            for k in range(K):
                self.x_kindices[Nsum:Nsum+self.N_k[k]] = k
                Nsum += self.N_k[k]

        # verbosity level -- if True, will print extra debug information
        self.verbose = verbose

        if mbar_options==None:
            pmf_mbar = MBAR(u_kn, N_k)
        else:
            # if the dictionary does not define the option, add it in
            required_mbar_options = ('maximum_iterations','relative_tolerance','verbose','initial_f_k'
                                     'solver_protocol','initialize','x_kindices')
            for o in required_mbar_options:
                if o not in mbar_options:
                    mbar_options[o] = None

            # reset the options that might be none to the default value
            if mbar_options['maximum_iterations'] == None:
                mbar_options['maximum_iterations'] = 10000
            if mbar_options['relative_tolerance'] == None:
                mbar_options['relative_toleratance'] = 1.0e-7
            if mbar_options['initialize'] == None:
                mbar_options['initialize'] = 'zeros'

            pmf_mbar = MBAR(u_kn, N_k, 
                            maximum_iterations = mbar_options['maximum_iterations'],
                            relative_tolerance = mbar_options['relative_tolerance'],
                            verbose = mbar_options['verbose'],
                            initial_f_k = mbar_options['initial_f_k'],
                            solver_protocol = mbar_options['solver_protocol'],
                            initialize = mbar_options['zeros'],
                            x_indices = mbar_options['x_indices'])

            self.mbar = pmf_mbar

    def getPMF(self, x, uncertainties = 'from-lowest', pmf_reference = None):

        """
        Returns values of the PMF at the specified x points.

        uncertainties : string, optional
            Method for reporting uncertainties (default: 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on PMF are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \sum_i p_i = 1 is used to determine uncertainties spread out through the PMF
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        pmf_reference : int, optional
            the reference state that is zeroed when uncertainty = 'from-specified'
        
        Returns
        -------
        result_vals : dictionary

        Possible keys in the result_vals dictionary:

        'f_i' : np.ndarray, float, shape=(K)
            result_vals['f_i'][i] is the dimensionless free energy of state i, relative to the state of lowest free energy
        'df_i' : np.ndarray, float, shape=(K)
            result_vals['df_i'][i] is the uncertainty in the difference of f_i with respect to the state of lowest free energy

        """

        if self.pmf_type == None:
            ParameterError('pmf_type has not been set!')

        # create dictionary to return results
        result_vals = dict()

        if self.pmf_type == 'histogram':
            # figure out which bins the samples are in. Clearly a faster way to do this.
            x_indices = np.zeros(np.len(x),int) 
            for i, xi in enumerate(x):
                for j in range(nbins-1):
                    if x > bins[j]:
                        x_indices[i] = j
                        continue

            # now we know what bins to calculate values for.
    
            # Compute uncertainties by forming matrix of W_nk.
            N_k = np.zeros([self.K + nbins], np.int64)
            N_k[0:K] = self.N_k
            W_nk = np.zeros([self.N, self.K + nbins], np.float64)
            W_nk[:, 0:K] = np.exp(self.mbar.Log_W_nk)
            for i in range(nbins):
                # Get indices of samples that fall in this bin.
                indices = np.where(bin_n == i)

                # Compute normalized weights for this state.
                W_nk[indices, K + i] = np.exp(log_w_n[indices] + self.f_i[i])

             # Compute asymptotic covariance matrix using specified method.
            Theta_ij = self.mbar._computeAsymptoticCovarianceMatrix(W_nk, N_k)


            if (uncertainties == 'from-lowest') or (uncertainties == 'from-specified'):
                # Report uncertainties in free energy difference from a given point
                # on PMF.

                if (uncertainties == 'from-lowest'):
                    # Determine bin index with lowest free energy.
                    j = self.mbar.f_i.argmin()
                elif (uncertainties == 'from-specified'):
                    if pmf_reference == None:
                        raise ParameterError(
                             "no reference state specified for PMF using uncertainties = from-specified")
                    else:
                        j = pmf_reference
                # Compute uncertainties with respect to difference in free energy
                # from this state j.
                for i in range(nbins):
                    df_i[i] = math.sqrt(
                    Theta_ij[K + i, K + i] + Theta_ij[K + j, K + j] - 2.0 * Theta_ij[K + i, K + j])

                # Shift free energies so that state j has zero free energy.
                f_i -= f_i[j]

                fx_vals = np.zeros(len(x_indices))
                dfx_vals = np.zeros(len(x_indices))
                for i in range(x_indices):
                    fx_vals = f_[x_indices]
                    dfx_vals = df[x_indices]

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_i'] = dfx_vals

            elif (uncertainties == 'all-differences'):
                # Report uncertainties in all free energy differences.

                diag = Theta_ij.diagonal()
                dii = diag[K, K + nbins]
                d2f_ij = dii + dii.transpose() - 2 * Theta_ij[K:K + nbins, K:K + nbins]

                # unsquare uncertainties
                df_ij = np.sqrt(d2f_ij)

                fx_vals = np.zeros(len(x_indices))
                dfx_vals = np.zeros(len(x_indices),len(x_indices))
                for i in range(x_indices):
                    fx_vals = f_i[x_indices]
                    for j in range(x_indices):
                        dfx_vals = df_ij[x_indices,x_indices]

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_ij'] = dfx_vals

            elif (uncertainties == 'from-normalization'):
                # Determine uncertainties from normalization that \sum_i p_i = 1.

                # Compute bin probabilities p_i
                p_i = np.exp(-f_i - logsumexp(-f_i))

                # todo -- eliminate triple loop over nbins!
                # Compute uncertainties in bin probabilities.
                d2p_i = np.zeros([nbins], np.float64)
                for k in range(nbins):
                    for i in range(nbins):
                        for j in range(nbins):
                            delta_ik = 1.0 * (i == k)
                            delta_jk = 1.0 * (j == k)
                            d2p_i[k] += p_i[k] * (p_i[i] - delta_ik) * p_i[k] * (p_i[j] - delta_jk) * Theta_ij[K + i, K + j]

                # Transform from d2p_i to df_i
                d2f_i = d2p_i / p_i ** 2
                df_i = np.sqrt(d2f_i)

                fx_vals = np.zeros(len(x_indices))
                dfx_vals = np.zeros(len(x_indices))
                for i in range(x_indices):
                    fx_vals = f_i[x_indices]
                    dfx_vals = df_i[x_indices]

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_i'] = dfx_vals

            
        return result_vals


    def generatePMF(self, pmf_type = 'histogram', histogram_parameters = None, uncertainties='from-lowest', pmf_reference=None):

        """
        With the initialized PMF, compute a PMF using the options.

        This implementation computes the expectation of an indicator-function observable for each bin.

        Parameters
        ----------

        pmf_type: string
             options = 'histogram'
        
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential energy of snapshot n of state k for which the PMF is to be computed.

        histogram_parameters:
            - bin_n : np.ndarray, float, shape=(N)
                 bin_n[n] is the bin index of snapshot n of state k.  bin_n can assume a value in range(0,nbins)
            - nbins : int
                 The number of bins

        Notes
        -----
        * pmf_type = 'histogram':
            * All bins must have some samples in them from at least one of the states -- this will not work if bin_n.sum(0) == 0. Empty bins should be removed before calling computePMF().
            * This method works by computing the free energy of localizing the system to each bin for the given potential by aggregating the log weights for the given potential.
            * To estimate uncertainties, the NxK weight matrix W_nk is augmented to be Nx(K+nbins) in order to accomodate the normalized weights of states . . . (?)
            * the potential is given by u_n within each bin and infinite potential outside the bin.  The uncertainties with respect to the bin of lowest free energy are then computed in the standard way.

        Examples
        --------

        >>> # Generate some test data
        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> # Select the potential we want to compute the PMF for (here, condition 0).
        >>> u_n = u_kn[0, :]
        >>> # Sort into nbins equally-populated bins
        >>> nbins = 10 # number of equally-populated bins to use
        >>> import numpy as np
        >>> N_tot = N_k.sum()
        >>> x_n_sorted = np.sort(x_n) # unroll to n-indices
        >>> bins = np.append(x_n_sorted[0::int(N_tot/nbins)], x_n_sorted.max()+0.1)
        >>> bin_widths = bins[1:] - bins[0:-1]
        >>> bin_n = np.zeros(x_n.shape, np.int64)
        >>> bin_n = np.digitize(x_n, bins) - 1
        >>> # Compute PMF for these unequally-sized bins.
        >>> pmf.generatePMF(u_n, bin_n, nbins)
        >>> results = pmf.getPMF()x)
        >>> f_i = results['f_i']
        >>> # If we want to correct for unequally-spaced bins to get a PMF on uniform measure
        >>> mbar = pmf.getMBAR()
        >>> f_i_corrected = mbar['f_i'] - np.log(bin_widths)

        """
        self.pmf_type = pmf_type

        if self.pmf_type == 'histogram':
            if histogram_parameters['nbins'] == None:
                ParameterError('histogram_parameters[\'nbins\'] cannot be undefined with pmf_type = histogram')
            else:
                if histogram_paramters['nbins'] < 0:
                    ParameterError('histogram_parameters[\'nbins\'] must be positive')

            # Verify that no PMF bins are empty -- we can't deal with empty bins,
            # because the free energy is infinite.
            if histogram_parameters['bins_n'] == None:    
                ParameterError('histogram_parameters[\'nbins\'] cannot be undefined with pmf_type = histogram')
                
            for i in range(nbins):
                if np.sum(bin_n == i) == 0:
                    raise ParameterError(
                "At least one bin in provided bin_n argument has no samples.  All bins must have samples for free energies to be finite.  Adjust bin sizes or eliminate empty bins to ensure at least one sample per bin.")
            K = self.K

            # Compute unnormalized log weights for the given reduced potential
            # u_n.
            log_w_n = self.mbar._computeUnnormalizedLogWeights(u_n)

            # Compute the free energies for these states.
            f_i = np.zeros([nbins], np.float64)
            df_i = np.zeros([nbins], np.float64)
            for i in range(nbins):
                # Get linear n-indices of samples that fall in this bin.
                indices = np.where(bin_n == i)

                # Sanity check.
                if (len(indices) == 0):
                    raise DataError("WARNING: bin %d has no samples -- all bins must have at least one sample." % i)

                # Compute dimensionless free energy of occupying state i.
                f_i[i] = - pymbar.mbar.logsumexp(log_w_n[indices])

        else:
            raise ParameterError("Uncertainty method '%s' not recognized." % uncertainties)



    def getMBAR(self):
        """return the MBAR object being used by the PMF  

           Parameters: None

           Returns: MBAR object
        """

        return self.mbar

    #=====================================================================



    #=========================================================================
    # PRIVATE METHODS - INTERFACES ARE NOT EXPORTED
    #=========================================================================

    def _ErrorOfDifferences(self, cov, warning_cutoff=1.0e-10):
        """
        inputs:
        cov is the covariance matrix of A

        returns the statistical error matrix of A_i - A_j
        """

        diag = np.matrix(cov.diagonal())
        d2 = diag + diag.transpose() - 2 * cov

        # Cast warning_cutoff to compare a negative number
        cutoff = -abs(warning_cutoff)

        # check for any numbers below zero.
        if np.any(d2 < 0.0):
            if np.any(d2 < cutoff):
                print("A squared uncertainty is negative. Largest Magnitude = {0:f}".format(
                    abs(np.min(d2[d2 < cutoff]))))
            else:
                d2[np.logical_and(0 > d2, d2 > cutoff)] = 0.0
        return np.sqrt(np.array(d2))

    def _pseudoinverse(self, A, tol=1.0e-10):
        """Compute the Moore-Penrose pseudoinverse, wraps np.linalg.pinv

        REQUIRED ARGUMENTS
          A (np KxK matrix) - the square matrix whose pseudoinverse is to be computed

        RETURN VALUES
          Ainv (np KxK matrix) - the pseudoinverse

        OPTIONAL VALUES
          tol - the tolerance (relative to largest magnitude singlular value) below which singular values are to not be include in forming pseudoinverse (default: 1.0e-10)

        NOTES
          In previous versions of pymbar / Numpy, we wrote our own pseudoinverse
          because of a bug in Numpy.

        """

        return np.linalg.pinv(A, rcond=tol)

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
        """Compute estimate of the asymptotic covariance matrix.

        Parameters
        ----------
        W : np.ndarray, shape=(N, K), dtype='float'
            The normalized weight matrix for snapshots and states.
            W[n, k] is the weight of snapshot n in state k.
        N_k : np.ndarray, shape=(K), dtype='int'
            N_k[k] is the number of samples from state k.
        method : string, optional, default=None
            Method used to compute the asymptotic covariance matrix.
            Must be either "approximate", "svd", or "svd-ew".  If None,
            defaults to "svd-ew".

        Returns
        -------
        Theta: np.ndarray, shape=(K, K), dtype='float'
            Asymptotic covariance matrix

        Notes
        -----
        The computational costs of the various 'method' arguments varies:
          'svd' computes the generalized inverse using the singular value decomposition -- this should be efficient yet accurate (faster)
          'svd-ew' is the same as 'svd', but uses the eigenvalue decomposition of W'W to bypass the need to perform an SVD (fastest)
          'approximate' only requires multiplication of KxN and NxK matrices, but is an approximate underestimate of the uncertainty.

        svd and svd-ew are described in appendix D of Shirts, 2007 JCP, while
        "approximate" in Section 4 of Kong, 2003. J. R. Statist. Soc. B.

        We currently recommend 'svd-ew'.
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

        check_w_normalized(W, N_k)

        # Compute estimate of asymptotic covariance matrix using specified method.
        if method == 'approximate':
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
            [U, S, Vt] = linalg.svd(W, full_matrices=False)  # False Avoids O(N^2) memory allocation by only calculting the active subspace of U.
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
                print("Initializing free energies to zero.")
            self.f_k[:] = 0.0
        elif (method == 'mean-reduced-potential'):
            # Compute initial guess at free energies from the mean reduced
            # potential from each state
            if verbose:
                print("Initializing free energies with mean reduced potential for each state.")
            means = np.zeros([self.K], float)
            for k in self.states_with_samples:
                means[k] = self.u_kn[k, 0:self.N_k[k]].mean()
            if (np.max(np.abs(means)) < 0.000001):
                print("Warning: All mean reduced potentials are close to zero. If you are using energy differences in the u_kln matrix, then the mean reduced potentials will be zero, and this is expected behavoir.")
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
        return -1. * logsumexp(self.f_k + u_n[:, np.newaxis] - self.u_kn.T, b=self.N_k, axis=1)
