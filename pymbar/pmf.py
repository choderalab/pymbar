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
             options = 'histogram', 'kde', 'max_likelihood'
        
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential energy of snapshot n of state for which the PMF is to be computed. 
            Often, it will be one of the states in of u_kn, used in initializing the PMF object, but we want
            to allow more generality.

        x_n : np.ndarray, float, shape=(N,D)
            x_n[n] is the d-dimensional coordinates of the samples, where D is the reduced dimensional space.

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

        maxlikelihood_parameters:      
            - power - power of spline to use. 
            - 'kl-divergence': 
            - minimization_parameters - dictionary parameters to pass to the minimizer. 
            - fbiaw: array of functions that return the Kth bias potential for each function

        Notes
        -----
        * pmf_type = 'histogram':
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
        >>> # Compute PMF for these unequally-sized bins.
        >>> pmf = PMF(u_kn,N_k)
        >>> histogram_parameters = dict()
        >>> histogram_parameters['bin_edges'] = [bins] 
        >>> pmf.generatePMF(u_n, x_n, pmf_type='histogram', histogram_parameters = histogram_parameters)
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
                raise ParameterError('histogram_parameters[\'bin_edges\'] cannot be undefined with pmf_type = histogram')

            self.bins = histogram_parameters['bin_edges']

            # First, determine the number of dimensions of the histogram. This can be determined 
            # by the shape of bin_edges
            dims = len(self.bins)
            self.dims = dims  # store the dimensionality for checking later.

            # now create the bins from the data.
            if len(np.shape(x_n))==1:  # it's a 1D array, instead of a Nx1 array.  Reshape.
                x_n = x_n.reshape(-1,1)

            bin_n = np.zeros(x_n.shape, np.int64)

            for d in range(dims):
                bin_n[:,d] = np.digitize(x_n[:,d], self.bins[d])-1 # bins returns 0 as out of bin.  We want to use -1 as out of bin
            self.bin_n = bin_n

            self.histogram_parameters = histogram_parameters


            # now we need to loop over the bin_n and identify and label the nonzero bins.

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

            # reshape data if needed.
            if len(np.shape(x_n))==1:  # it's a 1D array, instead of a Nx1 array.  Reshape.
                x_n = x_n.reshape(-1,1)

            kde.fit(x_n, sample_weight = np.exp(log_w_n))

            self.kde = kde

        elif pmf_type == 'max_likelihood':
            optimize_options = ml_params['optimize_parameters'] 
            if ml_param['fit_type'] == 'bspline': 
                kdegree = ml_params['spline_degree']
                nspline = ml_params['nspline']
                tol = ml_params['gtol']
                
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
                
                # we need a fast initialization.  Use KDE with nsplines points.
                init_pmf = self.copy()
                kde_parameters['bandwidth'] = 0.5*(xrange[1]-xrange[0])/nsplines
                init_pmf.generatePMF(u_kn, x_n, pmf_type='kde', kde_parameters = kde_parameters)
                centers = np.linspace(xrange[0],xrange[1],nspline)
                results = init_pmf.getPMF(np.linspace(xrange[0],xrange[1],centers))
                xinit = centers
                yinit = results['f_i']
        
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

                # same for the next execution. Not sure if best time to save it.
                self.bspline_derivatives = db_c
                
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
                    # if just kl divergence, we want the normalized probability
                    log_w_n = mbar._computeUnnormalizedLogWeights(pymbar.utils.kn_to_n(u_kn, N_k = mbar.N_k))
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

                    f, expf, pF = self._bspline_calculate_f(w_n, x_n, b, method, K, xrange)

                    # we need some error handling: if we stepped too far, we should go back
                    if not firsttime:
                        count = 0
                        while f >= fold*(1.0+np.sqrt(tol)) and count < 5:   # we went too far!  Pull back.
                            f = fold
                            # let's not step as far:
                            dx = 0.5*dx
                            xi[1:nspline] = xold[1:nspline] - dx # step back half of dx.
                            xold = xi.copy()
                            print(xi)
                            b = BSpline(b.t,xi,b.k)
                            f, expf, pF = self._bspline_calculate_f(w_n, x_n, b, method, K, xrange)
                            count += 1
                    else:
                        firsttime = False
                    fold = f
                    xold = xi.copy()

                    g, dg2, gkquad, pE = self._bspline_calculate_g(w_n, x_n, nspline, b, db_c, expf, pF, method, K, xrangei)

                    h = self._bspline_calculate_h(w_n, x_n, nspline, kdegree, b, db_c, expf, pF, pE, method, K, Ki, xrangeij)
                
                    # now find the new point.
                    # x_n+1 = x_n - f''(x_n)^-1 f'(x_n) 
                    # which we solve more stably as:
                    # x_n - x_n+1 = f''(x_n)^-1 f'(x_n)
                    # f''(x_n)(x_n-x_n+1) = f'(x_n)  
                    # solution is dx = x_n-x_n+1

                    dx = np.linalg.lstsq(h,g)[0]
                    xi[1:nspline] = xold[1:nspline] - dx
                    b = BSpline(b.t,xi,b.k)

                self.pmf_function = pmf_final
                # at this point, we should have a set of spline parameters.

            elif ml_param['fit_type'] == 'kldiverge':
                w_n = np.exp(log_w_n)
                w_n = w_n/np.sum(w_n) # nomalize the weighs
                result = minimize(self._kldiverge,tstart,args=(trialf,x_n,w_n,xrange),options=optimize_options)
                self.pmf_function  = trialf(result.x)

            elif ml_paramts['fit_type'] = 'vFEP'


        else:
            raise ParameterError('pmf_type {:s} is not defined!'.format(pmf_type))

        if self.timings:  # we put the timings outside, since the switch / common stuff is really low.
            end = timer()
            results_vals['timing'] = end-start 

        return result_vals


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

        if len(np.shape(x)) <= 1: # if it's zero, it's a scalar.
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

            # if it's not an array, make it one.
            x = np.array(x)

            if len(np.shape(x)) <=1:  # it's a 1D array, instead of a Nx1 array.  Reshape.
                x = x.reshape(-1,1)
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

        elif self.pmf_type == 'max_likelihood':

            f_i = self.pmf_function(x)

            if uncertainties == 'from_lowest':
                fmin = np.min(f_i)
                f_i =- fmin
            elif uncertainties == 'from-specified':
                fmin = -self.kde.score_samples(np.array(pmf_reference).reshape(1, -1))
                f_i = f_i - fmin

            # uncertainites "from normalization" reference is applied, since the density is normalized.
            result_vals['f_i'] = f_i

            # no error method yet. Maybe write a bootstrap class? 

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
        """ return the KernelDensity object if it exists.

            Parameters: None
     
            Returns: sklearn KernelDensity object
        """

        if self.pmf_type == 'kde':
            return self.kde
        else:
            raise ParameterError('Can\'t return the KernelDensity object because pmf_type != kde')


    def _kldiverge(t,ft,x_n,K,w_n,xrange):

        """ The function that determines the KL divergence over the weighted points

            Parameters: t:        vector of parameters 
                        ft:       spline object
                        x_n:      coordinates of the samples.
                        K:        Number of states from MBAR
                        w_n:      weights of each point from MBAR
                        xrange:   low and high points of the range of interest
            Returns: 

        """

        t -= t[0] # set a reference state, may make the minization faster by removing degenerate solutions
        # define the function f based on the current parameters t
        feval = ft(t) # current value of the PMF
        # define the exponential of f based on the current parameters t
        expf = lambda x: np.exp(-feval(x))
        pE = np.dot(w_n,feval(x_n))
        pF = np.log(quad(expf,xrange[0],xrange[1])[0]) #value 0 is the value of quadrature
        kl = pE + pF 

    return kl

    def _sumkldiverge(t,ft,x_n,K,w_kn,fbias,xrange):

        t -= t[0] # set a reference state, may make the minization faster by removing degenerate solutions
        feval = ft(t)  # the current value of the PMF
        fx = feval(x_n)  # only need to evaluate this over all points outside(?)      
        kl = 0
        # figure out the bias at each point
        for k in range(K):
        # define the exponential of f based on the current parameters t.
            expf = lambda x: np.exp(-feval(x)-fbias[k](x))
            pE = np.dot(w_kn[:,k],fx+fbias(k,x_n))
            pF = np.log(quad(expf,xrange[0],xrange[1])[0])  #value 0 is the value of quad
            kl += (pE + pF)
        return kl

    def _vFEP(t,ft,x_kn,K,N_k,fbias,xrange):
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
        return kl    

    def _bspline_calculate_f(w_n, x_n, b, method, K, xrange):

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

        #print("function value to minimize: {:f}".format(f))

        return f, expf, pF  # return the value and the needed data (eventually move to class)


    def _bspline_calculate_g(w_n, x_n, nspline, b, db_c, expf, pF, method, K, xrangei):

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

    def _bspline_calculate_h(w_n, x_n, nspline, kdegree, b, db_c, expf, pF, pE, method, K, Ki, xrangeij):

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
