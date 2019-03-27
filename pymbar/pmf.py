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

        """Initialize a potential of mean force calculation by performing
        multistate Bennett acceptance ratio (MBAR) on a set of
        simulation data from umbrella sampling at K states.

        Upon initialization, the dimensionless free energies for all
        states are computed.  This may take anywhere from seconds to
        minutes, depending upon the quantity of data.

        This creates an internal mbar object that is used to create
        the potential of means force.

        Methods are: 

           generatePMF: given an intialized MBAR object, a set of points, 
                        the desired energies at that point, and a method, generate 
                        an object that contains the PMF information.

           getPMF: given coordinates, generate the PMF at each coordinate (and uncertainty)

           getMBAR: return the underlying mbar object.

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
            initialize : 'zeros' or 'BAR', optional, Default: 'zeros'
            x_kindices : which state index each sample is from.

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
            required_mbar_options = ('maximum_iterations','relative_tolerance','verbose','initial_f_k',
                                     'solver_protocol','initialize','x_kindices')
            for o in required_mbar_options:
                if o not in mbar_options:
                    mbar_options[o] = None

            # reset the options that might be none to the default value
            if mbar_options['maximum_iterations'] == None:
                mbar_options['maximum_iterations'] = 10000
            if mbar_options['relative_tolerance'] == None:
                mbar_options['relative_tolerance'] = 1.0e-7
            if mbar_options['initialize'] == None:
                mbar_options['initialize'] = 'zeros'

            pmf_mbar = pymbar.MBAR(u_kn, N_k, 
                                   maximum_iterations = mbar_options['maximum_iterations'],
                                   relative_tolerance = mbar_options['relative_tolerance'],
                                   verbose = mbar_options['verbose'],
                                   initial_f_k = mbar_options['initial_f_k'],
                                   solver_protocol = mbar_options['solver_protocol'],
                                   initialize = mbar_options['initialize'],
                                   x_kindices = mbar_options['x_kindices'])

        self.mbar = pmf_mbar

        if self.verbose:
            print("PMF initialized")

    def generatePMF(self, u_n, x_n, pmf_type = 'histogram', histogram_parameters = None, kde_parameters = None, uncertainties='from-lowest'):

        """
        Given an intialized MBAR object, a set of points, 
        the desired energies at that point, and a method, generate 
        an object that contains the PMF information.
        
        Parameters
        ----------

        pmf_type: string
             options = 'histogram', 'kde'
        
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential energy of snapshot n of state for which the PMF is to be computed. 
            Often, it will be one of the states in of u_kn, used in initializing the PMF object, but we want
            to allow more generality.

        x_n : np.ndarray, float, shape=(N,D)
            x_n[n] is the d-dimensional coordinates of the samples, where D is the reduced dimensional space.
            Currently, not used for pmf_type='histogram', which takes bin counts, but it probably should be used in the future,
            with the bin count determined within the program.

        histogram_parameters:
            - bin_n : np.ndarray, float, shape=(N,K) or (N)
                 If 1D, bin_n is an length-d array with a value in range(0,nbins).
                 If 2D, bin_n is an length-d x k array x K states with a value in range(0,nbins) for each dimension.
                 We do not currently support passing in array of bins in the shape K x Nmax
                 If a sample is out of the grid (out of min, max in bin edges in that direction), its value is set to -1 in that dimension.

            - bin_edges: list of ndim np.ndarray, each array shaped ndum+1
                 The bin edges. Compatible with `bin_edges` output of np.histogram.  

        kde_parameters:
            - all the parameters from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html).
              Defaults will be used if nothing changed.


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
        >>> from pymbar import PMF
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn',seed=0)
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
        >>> pmf = PMF(u_kn,N_k)
        >>> histogram_parameters = dict()
        >>> histogram_parameters['bin_edges'] = [bins] 
        >>> histogram_parameters['bin_n'] = bin_n
        >>> pmf.generatePMF(u_n, x_n=None, pmf_type='histogram', histogram_parameters = histogram_parameters)
        >>> results = pmf.getPMF(x_n)
        >>> f_i = results['f_i']
        >>> for i,x_n in enumerate(x_n):
        >>> print(x_n,f_i[i])
        >>> mbar = pmf.getMBAR()
        >>> print(mbar.f_k)
        >>> print(N_k) 

        """
        self.pmf_type = pmf_type

        # eventually, we just want the desired energy of each sample.  For now, we allow conversion 
        # from older 2d format (K,Nmax instead of N); this is data SAMPLED from each k, not the energy at different K.
        if len(np.shape(u_n)) == 2:
            u_n = pymbar.mbar.kn_to_n(u_n, N_k = self.N_k)

        self.u_n = u_n
        
        # Compute unnormalized log weights for the given reduced potential u_n, needed for all methods.
        log_w_n = self.mbar._computeUnnormalizedLogWeights(self.u_n)


        K = self.mbar.K  # number of states
        
        if self.pmf_type == 'histogram':
            if 'bin_edges' not in histogram_parameters:
                ParameterError('histogram_parameters[\'bin_edges\'] cannot be undefined with pmf_type = histogram')

            if 'bin_n' not in histogram_parameters:    
                ParameterError('histogram_parameters[\'nbins\'] cannot be undefined with pmf_type = histogram')

            self.histogram_parameters = histogram_parameters
            bin_n = histogram_parameters['bin_n']
            self.bins = histogram_parameters['bin_edges']

            # First, determine the number of dimensions of the histogram. This can be determined 
            # by the shape of bin_edges
            dims = len(self.bins)

            self.dims = dims  # store the dimensionality for checking later.

            # now we need to loop over the bin_n and identify and label the nonzero bins.
            if len(np.shape(bin_n)) > dims:
                ParameterError("bin_n must be in the format of N_total x (bin dimensions). It must not be in the form K states x N_max x (bin_dimensions).") 

            nonzero_bins = list() # a list of the bins with at least one sample in them.
            nonzero_bins_index = np.zeros(self.N,dtype=int) # for each sample, the index of the nonzero_bins element it belongs to.

            for n in range(self.N):
                if np.any(bin_n[n] < 0):
                    nonzero_bins_index[n] = -1
                    continue  # this sample is out of grid
                if dims == 1:
                    ind2 = bin_n[n]  # which bin sample n is in
                else:
                    ind2 = tuple(bin_n[n]) # which bin (labeled N-d) sample n is in
                if ind2 not in nonzero_bins:
                    nonzero_bins.append(ind2)  # this bin has a sample.  Add it to the list 
                nonzero_bins_index[n] = nonzero_bins.index(ind2)  # the index of the nonzero bins

            self.bin_n = nonzero_bins_index  # store the index of the nonzero bins for each sample. 
            self.nbins = np.int(np.max(self.bin_n))+1  # the total number of nonzero bins

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

            self.f = f_i  # store the free energies for this bin  

            # now assign back the free energy from the sample_only bins to all of the bins. 

            # rebuild the graph from the edges.
            corner_vectors = list()
            returnsize = list()
            for d in range(dims):
                maxv = len(self.bins[d])-1
                corner_vectors.append(np.arange(0,maxv))
                returnsize.append(maxv)
            gridpoints = it.product(*corner_vectors)  # iterator giving all bin locations in N dimensions.

            fbin_index = np.zeros(np.array(returnsize),int) # index in self.f where the free energy for this gridpoint is stored
            for g in gridpoints:
                if g in nonzero_bins:
                    fbin_index[g] = nonzero_bins.index(g)
                else: 
                    fbin_index[g] = -1  # no free energy for this index, since there are no points.
            self.fbin_index = fbin_index


        elif pmf_type == 'kde':
            from sklearn.neighbors.kde import KernelDensity
            kde = KernelDensity()
            kde_defaults = kde.get_params()  # get the default params to set them.

            for k in kde_defaults:
                if k in kde_parameters:
                    kde_defaults[k] = kde_parameters[k]

            # make sure we didn't pass any arguments that DON'T belong here
            for k in kde_parameters:
                if k not in kde_defaults:
                    raise "Warning: {:s} is not a parameter in KernelDensity".format(k)

            kde.set_params(**kde_defaults) 

            kde.fit(x_n, sample_weight = np.exp(log_w_n))
            self.kde = kde

        else:

            raise ParameterError("pmf_type '%s' not recognized." % pmf_type)


    def getPMF(self, x, uncertainties = 'from-lowest', pmf_reference = None):

        """
        Returns values of the PMF at the specified x points.

        Parameters
        ----------

        x: numpy:ndarray of D dimensions, where D is the dimensionality of the PMF defined.
        
        uncertainties : string, optional
            Method for reporting uncertainties (default: 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on PMF are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \sum_i p_i = 1 is used to determine uncertainties spread out through the PMF
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        pmf_reference : an N-d point specifying the reference state. Ignored except with uncertainty method 'from_specified''
        
        Returns
        -------
        result_vals : dictionary

        keys in the result_vals dictionary:

        'f_i' : np.ndarray, float, shape=(K)
            result_vals['f_i'][i] is the dimensionless free energy of the x_i point, relative to the reference point
        'df_i' : np.ndarray, float, shape=(K)
            result_vals['df_i'][i] is the uncertainty in the difference of x_i with respect to the reference point

        """

        if len(np.shape(x)) == 1:
            coorddim = 1
        else:
            coorddim = np.shape(x)[1]
        if self.dims != coorddim:
            raise DataError('coordinates have inconsistent dimension with the PMF.')

        if uncertainties is 'from-specified' and pmf_reference is None:
            raise ParameterError(
                "No reference state specified for PMF using uncertainties = from-specified")

        if self.pmf_type == None:
            raise ParameterError('pmf_type has not been set!')

        K = self.mbar.K  # number of states

        # create dictionary to return results
        result_vals = dict()

        if self.pmf_type == 'histogram':

            # figure out which bins the values are in.
            dims = len(self.bins)

            if dims == 1:
                # what gridpoint does each x fall into?
                loc_indices = np.digitize(x,self.bins[0])-1 # -1 and nbinsperdim are out of range
            else:
                loc_indices = np.zeros([len(x),dims],dtype=int)
                for d in range(dims):
                    loc_indices[:,d] = np.digitize(x[:,d],self.bins[d])-1  # -1 and nbinsperdim are out of range

            # figure out which grid point the pmf_reference is at
            if pmf_reference is not None:        
                if dims == 1:
                    pmf_reference = [pmf_reference] # make it a list for reduced code duplication.
                pmf_ref_grid = np.zeros([dims],dtype=int)                
                for d in range(dims):
                    pmf_ref_grid[d] = np.digitize(pmf_reference[d],self.bins[d])-1  # -1 and nbins_per_dim are out of range
                    if pmf_ref_grid[d] == -1 or pmf_ref_grid[d] == len(self.bins[d]):
                        raise ParameterError("Specified reference point coordinate {:f} in dim {:d} grid point is out of the defined free energy region [{:f},{:f}]".format(pmf_ref_grid[d],np.min(bins[d]),np.max(bins[d])))


            # Compute uncertainties in free energy at each gridpoint by forming matrix of W_nk.
            N_k = np.zeros([self.K + self.nbins], np.int64)
            N_k[0:K] = self.N_k
            W_nk = np.zeros([self.N, self.K + self.nbins], np.float64)
            W_nk[:, 0:K] = np.exp(self.mbar.Log_W_nk)

            log_w_n = self.mbar._computeUnnormalizedLogWeights(self.u_n)
            for i in range(self.nbins): # loop over the nonzero bins, internal list numbering
                # Get indices of samples that fall in this bin.
                indices = np.where(self.bin_n == i) 

                # Compute normalized weights for this state.
                W_nk[indices, K + i] = np.exp(log_w_n[indices] + self.f[i])

            # Compute asymptotic covariance matrix using specified method.
            Theta_ij = self.mbar._computeAsymptoticCovarianceMatrix(W_nk, N_k)

            df_i = np.zeros(len(self.f), np.float64)

            if uncertainties == 'from-lowest' or uncertainties == 'from-specified' or uncertainties == 'all-differences':
                # Report uncertainties in free energy difference from a given point
                # on PMF.

                if (uncertainties == 'from-lowest'):
                    # Determine bin index with lowest free energy.
                    j = self.f.argmin()
                elif (uncertainties == 'from-specified'):
                    j = self.fbin_index[tuple(pmf_ref_grid)]

                # Compute uncertainties with respect to difference in free energy
                # from this state j.
                for i in range(self.nbins):
                    df_i[i] = math.sqrt(
                    Theta_ij[K + i, K + i] + Theta_ij[K + j, K + j] - 2.0 * Theta_ij[K + i, K + j])

                # Shift free energies so that state j has zero free energy.
                f_i = self.f - self.f[j]

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

            # figure out how many grid points in each direction
            maxp = np.zeros(dims,int)
            for d in range(dims):
                maxp[d] = len(self.bins[d])

            for i,l in enumerate(loc_indices):  
                # Must be a way to list comprehend this?
                if np.any(l < 0):  # out of index below
                    fx_vals[i] = np.nan
                    dfx_vals[i] = np.nan
                    continue
                if np.any(l > maxp): # out of index above
                    fx_vals[i] = np.nan
                    dfx_vals[i] = np.nan
                    continue

                if dims == 1:
                    findex = self.fbin_index[l]  
                else:
                    findex = self.fbin_index[tuple(l)] 
                if findex >= 0:
                    fx_vals[i] = f_i[findex]
                    dfx_vals[i] = df_i[findex]
                else:
                    fx_vals[i] = np.nan
                    dfx_vals[i] = np.nan

                # Return dimensionless free energy and uncertainty.
                result_vals['f_i'] = fx_vals
                result_vals['df_i'] = dfx_vals

            if uncertainties == 'all-differences':
                # Report uncertainties in all free energy differences as well.

                diag = Theta_ij.diagonal()
                dii = diag[K, K + self.nbins]  # appears broken?  Not used?
                d2f_ij = dii + dii.transpose() - 2 * Theta_ij[K:K + self.nbins, K:K + self.nbins]

                # unsquare uncertainties
                df_ij = np.sqrt(d2f_ij)

                dfxij_vals = np.zeros([len(x),len(x)])

                findexs = list()
                for i, l in enumerate(loc_indices):
                    if dims == 1:
                        findex = self.fbin_index[l]
                    else:
                        findex = self.fbin_index[tuple(l)]
                    findexs.append(findex)

                for i, vi in enumerate(findexs):
                    for j,vj in enumerate(findexs):
                        if vi != -1 and vj != 1:
                            dfxij_vals[i,j] = df_ij[vi,vj]
                        else:
                            dfxij_vals[i,j] = np.nan

                # Return dimensionless free energy and uncertainty.
                result_vals['df_ij'] = dfxij_vals

        elif self.pmf_type == 'kde':
            f_i = -self.kde.score_samples(x)
            if uncertainties == 'from_lowest':
                fmin = np.min(f_i)
                f_i =- fmin
            elif uncertainties == 'from-specified':
                fmin = -self.kde.score_samples(np.array(pmf_reference).reshape(1, -1))
                f_i = f_i - fmin
            # uncertainites "from normalization" reference is applied, since the density is normalized.
            result_vals['f_i'] = f_i
            # no error method yet. Maybe write a bootstrap class? 

        else:
            raise ParameterError('pmf_type {:s} is not defined!'.format(pmf_type))

        return result_vals


    def getMBAR(self):
        """return the MBAR object being used by the PMF  

           Parameters: None

           Returns: MBAR object
        """
        if self.mbar is not None:
           return self.mbar
        else:
           raise DataError('MBAR in the PMF object is not initialized, cannot return it.')

    def getKDE(self):
        """ return the KernelDensity object of it exists.

            Parameters: None
     
            Returns: sklearn KernelDensity object
        """

        if self.pmf_type == 'kde':
            return self.kde
        else:
            raise ParameterError('Can\'t return the KernelDensity object because pmf_type != kde')
