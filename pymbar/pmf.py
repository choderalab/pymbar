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
import itertools as it
import numpy as np
import numpy.linalg as linalg
import pymbar
from pymbar import mbar_solvers
from pymbar.utils import kln_to_kn, kn_to_n, ParameterError, DataError, logsumexp, check_w_normalized

import pdb
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

    def __init__(self, u_kn, N_k, verbose = False, mbar_options = None, **kwargs):

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
        
        # for now, still need to convert from 3 to 2 dim
        # Get dimensions of reduced potential energy matrix, and convert to KxN form if needed.
        if len(np.shape(u_kn)) == 3:
            self.K = np.shape(u_kn)[1]  # need to set self.K, and it's the second index
            u_kn = kln_to_kn(u_kn, N_k=self.N_k)

        # u_kn[k,n] is the reduced potential energy of sample n evaluated at state k
        self.u_kn = np.array(u_kn, dtype=np.float64)

        K, N = np.shape(u_kn)

        if np.sum(self.N_k) != N:
            raise ParameterError(
                'The sum of all N_k must equal the total number of samples (length of second dimension of u_kn.')

        # Store local copies of other data
        self.K = K  # number of thermodynamic states energies are evaluated at
        # N = \sum_{k=1}^K N_k is the total number of samples
        self.N = N  # maximum number of configurations

        # verbosity level -- if True, will print extra debug information
        self.verbose = verbose

        if mbar_options == None:
            pmf_mbar = pymbar.MBAR(u_kn, N_k)
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

            pmf_mbar = pymbar.MBAR(u_kn, N_k, 
                                   maximum_iterations = mbar_options['maximum_iterations'],
                                   relative_tolerance = mbar_options['relative_tolerance'],
                                   verbose = mbar_options['verbose'],
                                   initial_f_k = mbar_options['initial_f_k'],
                                   solver_protocol = mbar_options['solver_protocol'],
                                   initialize = mbar_options['zeros'],
                                   x_indices = mbar_options['x_indices'])

        self.mbar = pmf_mbar

        if self.verbose:
            print("PMF initialized")

    def generatePMF(self, u_n, pmf_type = 'histogram', histogram_parameters = None, uncertainties='from-lowest', pmf_reference=None):

        """
        With the initialized PMF, compute a PMF using the options.

        This implementation computes the expectation of an indicator-function observable for each bin.

        Parameters
        ----------

        pmf_type: string
             options = 'histogram'
        
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential energy of snapshot n of state k for which the PMF is to be computed. 
            Often, it will be one of the indicies of u_kn, used in initializing the PMF object, but we want
            to allow more generality.

        histogram_parameters:
            - bin_n : np.ndarray, float, shape=(N,D)
                 bin_n[n,k] is the bin index of snapshot n of state k.  
                 bin_n is an length-d array with a value in range(0,nbins) for each dimension.

            - bin_edges : list of D np.ndarraym each shape Nd+1
                 The bin edges. Compatible with `bin_edges` output of np.histogram.  

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
        >>> pmf.generatePMF(u_n, bin_edges, bin_n)
        >>> results = pmf.getPMF(x)
        >>> f_i = results['f_i']
        >>> # If we want to correct for unequally-spaced bins to get a PMF on uniform measure
        >>> mbar = pmf.getMBAR()
        >>> f_i_corrected = mbar['f_i'] - np.log(bin_widths)

        """
        self.pmf_type = pmf_type

        if len(np.shape(u_n)) == 2:
            u_n = pymbar.mbar.kn_to_n(u_n, N_k = self.N_k)

        self.u_n = u_n

        if self.pmf_type == 'histogram':
            if 'bin_edges' not in histogram_parameters:
                ParameterError('histogram_parameters[\'bin_edges\'] cannot be undefined with pmf_type = histogram')

            # Verify that no PMF bins are empty -- we can't deal with empty bins,
            # because the free energy is infinite.
            if 'bin_n' not in histogram_parameters:    
                ParameterError('histogram_parameters[\'nbins\'] cannot be undefined with pmf_type = histogram')

            self.histogram_parameters = histogram_parameters
            bin_n = histogram_parameters['bin_n']
            self.bins = histogram_parameters['bin_edges']

            # need to determine the shape of bin_n, as we need to:
            #    1. collapse it into one dimension.
            #    2. remove the zeros.
            # First, determine the number of dimensions of the histogram. This can be determined 
            # by the shape of bin_edges

            dims = len(self.bins)

            # now we need to loop over the bin_n and identify and label the nonzero bins.
            # make sure this works with 1D still

            nonzero_n = 0
            nonzero_bins = list()
            states = np.shape(bin_n)[0] 
            Nmax = np.shape(bin_n)[1]
            bins_counted = np.zeros([states,Nmax],dtype=int)

            for k in range(states):
                for n in range(Nmax):
                    if dims == 1:
                        ind2 = bin_n[k,n]
                    else:
                        ind2 = tuple(bin_n[k,n])
                    if ind2 not in nonzero_bins:
                        nonzero_n += 1
                        nonzero_bins.append(ind2)
                    bins_counted[k,n] = nonzero_bins.index(ind2)

            bin_n = pymbar.mbar.kn_to_n(bins_counted, N_k = self.N_k)

            self.bin_n = bin_n 
            self.nbins = np.int(np.max(self.bin_n)+1)  # the total number of nonzero bins

            K = self.mbar.K

            # Compute unnormalized log weights for the given reduced potential u_n.
            log_w_n = self.mbar._computeUnnormalizedLogWeights(self.u_n)

            # Compute the free energies for these histogram states with samples
            f_i = np.zeros([self.nbins], np.float64)
            df_i = np.zeros([self.nbins], np.float64)
            for i in range(self.nbins):
                # Get linear n-indices of samples that fall in this bin.
                indices = np.where(self.bin_n == i)

                # Sanity check.
                if (len(indices) == 0):
                    raise DataError("WARNING: bin %d has no samples -- all bins must have at least one sample." % i)

                # Compute dimensionless free energy of occupying state i.
                f_i[i] = - pymbar.mbar.logsumexp(log_w_n[indices])

            self.f = f_i    
            # now assign back the free energy from the count_only bins to all of the bins. 
            corner_vectors = list()
            returnsize = list()
            for d in range(dims):
                maxv = len(self.bins[d])-1
                corner_vectors.append(np.arange(0,maxv))
                returnsize.append(maxv)
            gridpoints = it.product(*corner_vectors)
            fbin = np.zeros(np.array(returnsize),int)
            for g in gridpoints:
                if g in nonzero_bins:
                    fbin[g] = nonzero_bins.index(g)  
                else: 
                    fbin[g] = -1
            self.fbin = fbin

        else:
            raise ParameterError("pmf_type '%s' not recognized." % pmf_type)


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

        K = self.mbar.K  # number of states

        # create dictionary to return results
        result_vals = dict()

        if self.pmf_type == 'histogram':
            # figure out which bins the samples are in. Clearly a faster way to do this.

            dims = len(self.bins)
            if dims == 1:
                # what index does each x fall into?
                diff_edge = np.abs(np.floor(np.subtract.outer(x,self.bins[0]))) # how far is it above each bin edge?
                loc_indices = diff_edge.argmin(axis=1)
            else:
                loc_indices = np.zeros([len(x),dims],dtype=int)
                # what index does each x fall into?
                for d in range(dims):
                    diff_edge = np.abs(np.floor(np.subtract.outer(x[:,d],self.bins[d]))) # how far is it above each bin edge?
                    loc_indices[:,d] = diff_edge.argmin(axis=1)

            # Compute uncertainties by forming matrix of W_nk.
            N_k = np.zeros([self.K + self.nbins], np.int64)
            N_k[0:K] = self.N_k
            W_nk = np.zeros([self.N, self.K + self.nbins], np.float64)
            W_nk[:, 0:K] = np.exp(self.mbar.Log_W_nk)

            log_w_n = self.mbar._computeUnnormalizedLogWeights(self.u_n)
            for i in range(self.nbins):
                # Get indices of samples that fall in this bin.
                indices = np.where(self.bin_n == i)

                # Compute normalized weights for this state.
                W_nk[indices, K + i] = np.exp(log_w_n[indices] + self.f[i])

            # Compute asymptotic covariance matrix using specified method.
            Theta_ij = self.mbar._computeAsymptoticCovarianceMatrix(W_nk, N_k)

            f_i = np.zeros([len(x)], np.float64)
            df_i = np.zeros([len(x)], np.float64)

            if (uncertainties == 'from-lowest') or (uncertainties == 'from-specified'):
                # Report uncertainties in free energy difference from a given point
                # on PMF.

                if (uncertainties == 'from-lowest'):
                    # Determine bin index with lowest free energy.
                    j = self.f.argmin()
                elif (uncertainties == 'from-specified'):
                    if pmf_reference == None:
                        raise ParameterError(
                             "no reference state specified for PMF using uncertainties = from-specified")
                    else:
                        j = pmf_reference
                # Compute uncertainties with respect to difference in free energy
                # from this state j.
                for i in range(self.nbins):
                    df_i[i] = math.sqrt(
                    Theta_ij[K + i, K + i] + Theta_ij[K + j, K + j] - 2.0 * Theta_ij[K + i, K + j])

                # Shift free energies so that state j has zero free energy.
                f_i = self.f - self.f[j]

                fx_vals = np.zeros(len(x))
                dfx_vals = np.zeros(len(x))

                for i,l in enumerate(loc_indices):
                    if dims == 1:
                        vol_e = self.fbin[l]
                    else:
                        vol_e = self.fbin[tuple(l)]
                    if vol_e >= 0:
                        fx_vals[i] = f_i[vol_e]
                        dfx_vals[i] = df_i[vol_e]
                    else:
                        fx_vals[i] = np.nan
                        dfx_vals[i] = np.nan

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_i'] = dfx_vals

            elif (uncertainties == 'all-differences'):
                # Report uncertainties in all free energy differences.

                pdb.set_trace()
                diag = Theta_ij.diagonal()
                dii = diag[K, K + self.nbins]  # appears broken?  Not used?
                d2f_ij = dii + dii.transpose() - 2 * Theta_ij[K:K + self.nbins, K:K + self.nbins]

                # unsquare uncertainties
                df_ij = np.sqrt(d2f_ij)

                fx_vals = np.zeros(len(x))
                dfx_vals = np.zeros([len(x),len(x)])

                vol_es = list()
                for i,l in enumerate(loc_indices):
                    if dims == 1:
                        vol_e = self.fbin[l]
                    else:
                        vol_e = self.fbin[tuple(l)]
                    if vol_e >= 0:
                        fx_vals[i] = f_i[vol_e]
                    else:
                        fx_vals[i] = np.nan
                    vol_es.append(vol_e)

                for i,vi in enumerate(vol_es):
                    for j,vj in enumerate(vol_es):
                        dfx_vals[i,j] = df_ij[vi,vj]

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_ij'] = dfx_vals

            elif (uncertainties == 'from-normalization'):
                # Determine uncertainties from normalization that \sum_i p_i = 1.

                # Compute bin probabilities p_i
                p_i = np.exp(-self.fbin - logsumexp(-self.fbin))

                # todo -- eliminate triple loop over nbins!
                # Compute uncertainties in bin probabilities.
                d2p_i = np.zeros([self.nbins], np.float64)
                for k in range(nbins):
                    for i in range(self.nbins):
                        for j in range(self.nbins):
                            delta_ik = 1.0 * (i == k)
                            delta_jk = 1.0 * (j == k)
                            d2p_i[k] += p_i[k] * (p_i[i] - delta_ik) * p_i[k] * (p_i[j] - delta_jk) * Theta_ij[K + i, K + j]

                # Transform from d2p_i to df_i
                d2f_i = d2p_i / p_i ** 2
                df_i = np.sqrt(d2f_i)


                fx_vals = np.zeros(len(x))
                dfx_vals = np.zeros(len(x))

                for i,l in enumerate(loc_indices):
                    if dims == 1:
                        vol_e = self.fbin[l]
                    else:
                        vol_e = self.fbin[tuple(l)]
                    if vol_e >= 0:
                        fx_vals[i] = f_i[vol_e]
                        dfx_vals[i] = df_i[vol_e]
                    else:
                        fx_vals[i] = np.nan
                        dfx_vals[i] = np.nan

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_i'] = dfx_vals

            
        return result_vals


    def getMBAR(self):
        """return the MBAR object being used by the PMF  

           Parameters: None

           Returns: MBAR object
        """

        return self.mbar

