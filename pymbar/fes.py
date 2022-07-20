##############################################################################
# pymbar: A Python Library for MBAR (FES module)
#
# Copyright 2019 University of Colorado Boulder
#
# Authors: Michael Shirts
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
A module implementing calculation of free energy surfaces (profiles) from biased simulations.
"""

import logging
import math
import numpy as np
import pymbar

from pymbar.utils import kln_to_kn, ParameterError, DataError, logsumexp
from pymbar import timeseries

# bunch of imports needed for doing newton optimization of B-splines
from scipy.interpolate import BSpline, make_lsq_spline

# imports needed for scipy minimizations
from scipy.integrate import quad
from scipy.optimize import minimize

from timeit import default_timer as timer  # may remove timing?

logger = logging.getLogger(__name__)

# =========================================================================
# FES class definition
# =========================================================================


class FES:
    """

    Methods for generating free energy surfaces (profile) with statistical uncertainties.

    Notes
    -----
    Note that this method assumes the data are uncorrelated.

    Correlated data must be subsampled to extract uncorrelated (effectively independent) samples.

    References
    ----------

    [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
    J. Chem. Phys. 129:124105, 2008
    http://dx.doi.org/10.1063/1.2978177

    [2] Shirts MR and Ferguson AF. Statistically optimal continuous
    free energy surfaces from umbrella sampling and multistate
    reweighting
    https://arxiv.org/abs/2001.01170

    """

    # =========================================================================

    def __init__(self, u_kn, N_k, verbose=False, mbar_options=None, timings=True, **kwargs):
        """Initialize a free energy surface calculation by performing
        multistate Bennett acceptance ratio (MBAR) on a set of
        simulation data from umbrella sampling at K states.

        Upon initialization, the dimensionless free energies for all
        states are computed.  This may take anywhere from seconds to
        minutes, depending upon the quantity of data.

        This also creates an internal mbar object that is used to create
        the free energy surface.

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
            forth. This only becomes important for bar -- MBAR does not
            care which samples are from which state.  We should eventually
            allow this assumption to be overwritten by parameters passed
            from above, once ``u_kln`` is phased out.

        mbar_options : dict
            The following options supported by mbar (see MBAR documentation)

            maximum_iterations : int, optional
            relative_tolerance : float, optional
            verbosity : bool, optional
            initial_f_k : np.ndarray, float, shape=(K), optional
            solver_protocol : list(dict) or None, optional, default=None
            initialize : 'zeros' or 'BAR', optional, Default: 'zeros'
            x_kindices : array of ints, shape=(K), which state index each sample is from.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> fes = FES(u_kn, N_k)

        """

        for key, val in kwargs.items():
            logging.warning(
                "Warning: parameter {:s}={:s} is unrecognized and unused.".format(key, val)
            )

        # Store local copies of necessary data.
        # N_k[k] is the number of samples from state k, some of which might be
        # zero.
        self.N_k = np.array(N_k, dtype=np.int64)
        self.N = np.sum(self.N_k)

        # for now, still need to convert from 3 to 2 dim
        # Get dimensions of reduced potential energy matrix, and convert to KxN
        # form if needed.
        if len(np.shape(u_kn)) == 3:
            # need to set self.K, and it's the second index
            self.K = np.shape(u_kn)[1]
            u_kn = kln_to_kn(u_kn, N_k=self.N_k)

        # u_kn[k,n] is the reduced potential energy of sample n evaluated at
        # state k
        self.u_kn = np.array(u_kn, dtype=np.float64)

        K, N = np.shape(u_kn)

        if np.sum(self.N_k) != N:
            raise ParameterError(
                "The sum of all N_k must equal the total number of samples (length of second dimension of u_kn."
            )

        # Store local copies of other data
        self.K = K  # number of thermodynamic states energies are evaluated at
        # N = \sum_{k=1}^K N_k is the total number of samples
        self.N = N  # maximum number of configurations

        # verbosity level -- if True, will print extra debug information
        self.verbose = verbose

        if timings:
            self.timings = True

        if mbar_options == None:
            fes_mbar = pymbar.MBAR(u_kn, N_k)
        else:
            # if the dictionary does not define the option, add it in
            required_mbar_options = (
                "maximum_iterations",
                "relative_tolerance",
                "verbose",
                "initial_f_k",
                "solver_protocol",
                "initialize",
                "x_kindices",
            )
            for o in required_mbar_options:
                if o not in mbar_options:
                    mbar_options[o] = None

            # reset the options that might be none to the default value
            if mbar_options["maximum_iterations"] is None:
                mbar_options["maximum_iterations"] = 10000
            if mbar_options["relative_tolerance"] is None:
                mbar_options["relative_tolerance"] = 1.0e-7
            if mbar_options["initialize"] is None:
                mbar_options["initialize"] = "zeros"

            fes_mbar = pymbar.MBAR(
                u_kn,
                N_k,
                maximum_iterations=mbar_options["maximum_iterations"],
                relative_tolerance=mbar_options["relative_tolerance"],
                verbose=mbar_options["verbose"],
                initial_f_k=mbar_options["initial_f_k"],
                solver_protocol=mbar_options["solver_protocol"],
                initialize=mbar_options["initialize"],
                x_kindices=mbar_options["x_kindices"],
            )

        self.mbar = fes_mbar

        # TODO: eliminate this call - it's causing problems
        # with deepcopy, needed in some cases, since you can't
        # copy the np.random module
        # self._random = np.random
        # self._seed = None

        if self.verbose:
            logger.info("FES initialized")

    # TODO: see above about not storing np.random
    # @property
    # def seed(self):
    #    return self._seed
    #
    # def reset_random(self):
    #    self._random = np.random
    #    self._seed = None

    def generate_fes(
        self,
        u_n,
        x_n,
        fes_type="histogram",
        histogram_parameters=None,
        kde_parameters=None,
        spline_parameters=None,
        n_bootstraps=0,
        seed=-1,
    ):

        """
        Given an intialized MBAR object, a set of points,
        the desired energies at that point, and a method, generate
        an object that contains the FES information.

        Parameters
        ----------
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential energy of snapshot n of state for which the FES is to be computed.
            Often, it will be one of the states in of u_kn, used in initializing the FES object, but we want
            to allow more generality.

        x_n : np.ndarray, float, shape=(N,D)
            x_n[n] is the d-dimensional coordinates of the samples, where D is the reduced dimensional space.

        fes_type : str
             options = 'histogram', 'kde', 'spline'

        histogram_parameters: dictionary

            Input dictionary with the following keys:

                 bin_edges: list of ndim np.ndarray, each array shaped ndum+1
                     The bin edges. Compatible with `bin_edges` output of np.histogram.

                 kde_parameters
                     all the parameters from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html). Defaults will be used if nothing changed.

                 spline_parameters
                      'spline_weights'
                           which type of fit to use:
                           'biasedstates' - sum of log likelihood over all weighted states
                           'unbiasedstate' - log likelihood of the single unbiased state
                           'simplesum' - sum of log likelihoods from the biased simulation. Essentially equivalent to vFEP (York et al.)
                      'optimization_algorithm':
                           'Custom-NR' - a custom Newton-Raphson that is particularly fast for close data, but can fail
                           'Newton-CG' - scipy Newton-CG, only Hessian based method that works correctly because of data ordering.
                           '         ' - scipy gradient based methods that work, but are generally slower (CG, BFGS, L-LBFGS-B, TNC, SLSQP)
                      'fkbias' : array of functions
                           Return the Kth bias potential for each function

                      'nspline' : int
                           Number of spline points

                      'kdegree' : int
                           Degree of the spline.  Default is cubic ('3')

                      'objective' : string
                           'ml','map' # whether to fit the maximum likelihood or the maximum a posteriori

        n_bootstraps : int, 0 or > 1, Default: 0
            Number of bootstraps to create an uncertainty estimate. If 0, no bootstrapping is done. Required if
            one uses uncertainty_method = 'bootstrap' in get_fes

        seed : int, Default = -1
            Set the randomization seed. Settting should get the
            randomization (assuming the same calls are made in the
            same order) to return the same numbers.  This is local to
            this class and will not change any other random objects.

        Returns
        -------
        dict, optional
            if 'timings' is set to True in __init__, returns the time taken to generate the FES

        Notes
        -----
        * fes_type = 'histogram':
            * This method works by computing the free energy of localizing the system to each bin for the given potential by aggregating the log weights for the given potential.
            * To estimate uncertainties, the NxK weight matrix W_nk is augmented to be Nx(K+nbins) in order to accomodate the normalized weights of states . . .
            * the potential is given by u_n within each bin and infinite potential outside the bin.  The uncertainties with respect to the bin of lowest free energy are then computed in the standard way.

        Examples
        --------

        >>> # Generate some test data
        >>> from pymbar import testsystems
        >>> from pymbar import FES
        >>> x_n, u_kn, N_k, s_n = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn',seed=0)
        >>> # Select the potential we want to compute the FES for (here, condition 0).
        >>> u_n = u_kn[0, :]
        >>> # Sort into nbins equally-populated bins
        >>> nbins = 10 # number of equally-populated bins to use
        >>> import numpy as np
        >>> N_tot = N_k.sum()
        >>> x_n_sorted = np.sort(x_n) # unroll to n-indices
        >>> bins = np.append(x_n_sorted[0::int(N_tot/nbins)], x_n_sorted.max()+0.1)
        >>> bin_widths = bins[1:] - bins[0:-1]
        >>> # Compute FES for these unequally-sized bins.
        >>> fes = FES(u_kn, N_k)
        >>> histogram_parameters = dict()
        >>> histogram_parameters['bin_edges'] = [bins]
        >>> _ = fes.generate_fes(u_n, x_n, fes_type='histogram', histogram_parameters = histogram_parameters)
        >>> results = fes.get_fes(x_n)
        >>> f_i = results['f_i']
        >>> for i, x_n in enumerate(x_n):  # doctest: +SKIP
        >>>     print(x_n, f_i[i])  # doctest: +SKIP
        >>> mbar = fes.get_mbar()
        >>> print(mbar.f_k)  # doctest: +SKIP
        >>> print(N_k)  # doctest: +SKIP

        """

        result_vals = dict()  # for results we may want to return.

        self.fes_type = fes_type

        # eventually, we just want the desired energy of each sample.  For now, we allow conversion
        # from older 2d format (K,Nmax instead of N); this is data SAMPLED from
        # each k, not the energy at different K.
        if len(np.shape(u_n)) == 2:
            u_n = pymbar.mbar.kn_to_n(u_n, N_k=self.N_k)

        self.u_n = u_n

        if seed >= 0:
            np.random.seed(seed)

        # TODO: see above about storing np.random
        # if seed >= 0:
        #    # Set a better seeded random state
        #    self._random = np.random.RandomState(seed=seed)
        #    self._seed = seed

        # we need to save this for calculating uncertainties.
        if not np.issubdtype(type(n_bootstraps), np.integer) or n_bootstraps == 1:
            raise ValueError(
                f"n_bootstraps must be an integer of 0 or >=2, it was set to {n_bootstraps}"
            )
        self.n_bootstraps = n_bootstraps

        if self.timings:
            start = timer()

        self.fes_function = list()

        # set some variables before bootstrapping loop.

        self.mc_data = None  # we have not sampled MC data yet.

        if self.fes_type == "histogram":
            self._setup_fes_histogram(histogram_parameters)

        elif fes_type == "kde":
            self._setup_fes_kde(kde_parameters)

        elif fes_type == "spline":
            self._setup_fes_spline(spline_parameters)

        else:
            raise ParameterError("fes_type {:s} is not defined!".format(fes_type))

        N_k = self.mbar.N_k
        K = self.mbar.K
        N = np.sum(N_k)

        for b in range(n_bootstraps + 1):  # generate bootstrap samples.
            # we bootstrap from each simulation separately.
            if b == 0:  # the default
                bootstrap_indices = np.arange(0, N)
                mbar = self.mbar
                x_nb = x_n
            else:
                index = 0
                for k in range(K):
                    # TODO: address issue with storinig np.random in self
                    bootstrap_indices[index : index + N_k[k]] = index + np.random.randint(
                        0, N_k[k], size=N_k[k]
                    )
                    index += N_k[k]
                    # recompute MBAR.
                    mbar = pymbar.MBAR(
                        self.u_kn[:, bootstrap_indices], self.N_k, initial_f_k=self.mbar.f_k
                    )
                    x_nb = x_n[bootstrap_indices]

            # Compute unnormalized log weights for the given reduced potential
            # u_n, needed for all methods.
            log_w_nb = mbar._computeUnnormalizedLogWeights(self.u_n[bootstrap_indices])
            # calculate a few other things used for multiple methods
            max_log_w_nb = np.max(log_w_nb)  # we need to solve underflow.
            w_nb = np.exp(log_w_nb - max_log_w_nb)
            w_nb = w_nb / np.sum(w_nb)  # normalize the weights
            # normalized weights for all states.
            w_knb = np.exp(mbar.Log_W_nk)

            if b == 0:
                self.w_n = w_nb
                self.w_kn = w_knb

            if self.fes_type == "histogram":
                # not clear if need to pass both w_nb and log_w_nb, but saves some processing
                self._generate_fes_histogram(b, x_nb, w_nb, log_w_nb)

            elif self.fes_type == "kde":
                self._generate_fes_kde(b, x_nb, w_nb)

            elif self.fes_type == "spline":
                self._generate_fes_spline(b, x_nb, w_nb)

        # we put the timings outside, since the switch / common stuff is really
        # low.
        if self.timings:
            end = timer()
            result_vals["timing"] = end - start

        return result_vals  # should we return results under some other conditions?

    def _setup_fes_histogram(self, histogram_parameters):

        """
        Does initial processsing of histogram_parameters

        Parameters
        ----------
        histogram_parameters : dict()
            A options:values dictonary for parameters to create the FES using histogramss

        Returns
        -------
        None

        Internally, creates histogram_data object and histogram_datas
        to store information generated by generate_fes

        """

        if "bin_edges" not in histogram_parameters:
            raise ParameterError(
                "histogram_parameters['bin_edges'] cannot be undefined with fes_type = histogram"
            )

        # code expects that the bin edges consist in an list of
        # arrays. But for 1D, we should be able to just input a single array.
        if len(np.shape(histogram_parameters["bin_edges"])) == 1:
            histogram_parameters["bin_edges"] = [histogram_parameters["bin_edges"]]

        self.histogram_parameters = histogram_parameters

        self.histogram_data = None
        if self.n_bootstraps > 0:
            self.histogram_datas = list()
        else:
            self.histogram_datas = None

    def _generate_fes_histogram(self, b, x_n, w_nb, log_w_nb):

        """
        Parameters
        ----------
        b : int
            the bootstrap sample this is on (for non-bootstrap methods, will be 0)

        x_n : ndarray, length self.N
            The data point set used in this bootstrap

        log_w_n : np.ndarray, float, shape=(self.N)

            Normalized log weights for each sample for the state in which we want the FES
            (usually, the unbiased state).  Doing it outside the loop to avoid redoing it each time.

        Returns
        -------
        None

        Doesn't return, rather adds histogram_data (for b==0) to self, or for b>0, adds
        histogram_data to the array self.histogram_data for further processing by get_fes.

        """

        # store the data that will be regenerated each time.
        # We will not try to regenerate the bin locations each time,
        # as that would make it hard to calculate uncertainties.
        # We will just recalculate the populations of each bin.

        histogram_parameters = self.histogram_parameters
        bins = histogram_parameters["bin_edges"]
        dims = len(bins)

        histogram_data = {}
        histogram_data["dims"] = dims  # store the dimensionality for checking later.
        histogram_data["bins"] = bins  # save for other functions.

        # create the bins from the data.
        # it's a 1D array, instead of a Nx1 array.  Reshape.

        if len(np.shape(x_n)) == 1:
            x_n = x_n.reshape(-1, 1)

        bin_n = np.zeros(x_n.shape, int)
        bin_length = np.zeros(dims, int)
        for d in range(dims):
            bin_length[d] = len(bins[d])
            # bins returns 0 as out of bin.  We want to use -1 as out
            # of bin instead.
            bin_n[:, d] = np.digitize(x_n[:, d], bins[d]) - 1

        histogram_data["bin_n"] = bin_n  # bin counts in each bin

        # number each of the bins with samples with an integer
        # Assign each sample this integer label (in addition to the tuple label)

        nonzero_bins = list()
        bin_label = {}
        sample_label = np.zeros(self.N, int)

        for n in range(self.N):
            bin = tuple(bin_n[n])  # which bin (labeled N-D) sample n is in
            if np.any(bin_n[n] < 0):  # this sample is out of grid
                sample_label[n] = -1
            else:
                # how do we label the bins? if N-dimensional:
                # bins[0] + bins[1]*bin_length[1]**1 + bins[2]*bin_length[2]**2
                sample_label[n] = int(
                    np.sum([bin_n[n][d] * bin_length[d] ** d for d in range(dims)])
                )
            if bin not in nonzero_bins:
                nonzero_bins.append(bin)
                bin_label[bin] = sample_label[n]
        histogram_data["nonzero_bins"] = nonzero_bins
        histogram_data["sample_label"] = sample_label

        # problem with bins above:
        #
        # all over bootstraps, not all nonzero bins will occur, as in some bootstraps,
        # some labels won't appear.
        # However, if they appear in the list of nonzero bins, they will have the same labels.
        # that may be OK, since we are only really interested in
        # uncertainties for the ones that have nonzero samples in the original list.

        # Need to come up with an order for the labels for all bootstraps, so free energies
        # are always assigned.

        if b == 0:
            bin_order = {}
            i = 0
            for bv in bin_label.values():
                if bv not in bin_order:
                    bin_order[bv] = i
                    i += 1
            histogram_data["bin_order"] = bin_order
            histogram_data["bin_label"] = bin_label
        else:
            bin_order = self.histogram_data["bin_order"]

        # Compute the free energies for the histogram bins
        # with samples. We cannot calculate free energes
        # for bins w/o samples.

        f_i = np.zeros(len(bin_label), np.float64)

        for i, label in enumerate(bin_label.values()):
            # Get linear n-indices of samples that fall in this bin.
            indices = np.where(sample_label == label)

            # Sanity check.
            if len(indices) == 0:
                raise DataError(
                    f"WARNING: bin {i} has no samples -- all bins must have at least one sample."
                )

            # Compute dimensionless free energy of occupying state i.
            f_i[bin_order[label]] = -logsumexp(log_w_nb[indices])

        # store the free energies for this bin
        histogram_data["f"] = f_i

        if b == 0:
            self.histogram_data = histogram_data
        else:
            self.histogram_datas.append(histogram_data)

    def _setup_fes_kde(self, kde_parameters):

        """
        Does initial processsing of kde_parameters

        Parameters
        ----------
        kde_parameters : dict()
            A options:values dictonary for parameters to create the FES using the kernel density approach.
            Parameters are passsed on to sklearn KernelDensity.

        Returns
        -------
        None

        Internally, creates kde object and kdes list of kde ibjects
        to store information generated by generate_fes_kde

        """

        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            raise ImportError(
                "Cannot use 'kde' type FES without the scikit-learn module. Could not import sklearn"
            )

        kde = KernelDensity()
        # get the default params to set them.
        kde_defaults = kde.get_params()

        for k in kde_defaults:
            if k in kde_parameters:
                kde_defaults[k] = kde_parameters[k]

        # make sure we didn't pass any arguments that DON'T belong here
        for k in kde_parameters:
            if k not in kde_defaults:
                raise ParameterError("Warning: {:s} is not a parameter in KernelDensity".format(k))
        kde.set_params(**kde_defaults)

        self.kde_parameters = kde_parameters
        if self.n_bootstraps > 0:
            self.kdes = list()
        else:
            self.kdes = None

        self.kde = kde

    def _generate_fes_kde(self, b, x_n, w_n):

        """
        Given an fes object with the kde data set up, determine
        the information necessary to define a FES using a kernel density approximation

        Parameters
        ----------

        b : int
            Which bootstrap this is: b==0 is the initial value with the "untouched" data.

        x_n : np.ndarray, float, shape=(N,D)
            x_n[n] is the d-dimensional coordinates of the samples, where D is the reduced dimensional space.

        w_n : np.ndarray, float, shape=(sself.N)
            Weights for each sample for the state in which we want the FES (usually, the unbiased state)

        Returns
        -------
        None

        Data is stored in ``self.kde`` or ``self.kdes`` (for bootstrap replicates).

        """

        # reshape data if needed.
        # it's a 1D array, instead of a Nx1 array.  Reshape.
        if len(np.shape(x_n)) == 1:
            x_n = x_n.reshape(-1, 1)

        # TODO: figure out if this should be called each bootstrap run or not (shouldn't cost?)
        # basically, just need to have KernelDensity defined here.
        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            raise ImportError(
                "Cannot use 'kde' type FES without the scikit-learn module. Could not import sklearn"
            )

        if b > 0:
            kde = KernelDensity()  # need to create a new one so won't get refit
            # Will take a refactor to get this correct for pylint
            params = self.kde.get_params()  # pylint: disable=access-member-before-definition
            kde.set_params(**params)
        else:
            kde = self.kde
        kde.fit(x_n, sample_weight=self.w_n)

        if b > 0:
            self.kdes.append(kde)

    def _setup_fes_spline(self, spline_parameters):

        """
        Does initial processsing of spline_parameters

        Parameters
        ----------
        spline_parameters : dict()
            A options:values dictonary for parameters to create the FES using the spline approach.
            Parameters are explained in docstring of

        Returns
        -------
        None

        Internally, creates spline_data object and spline_datas list of spline_Data objects
        to store information generated by generate_fes_spline

        """

        if "objective" not in spline_parameters:
            spline_parameters["objective"] = "ml"  # default

        objective = spline_parameters["objective"]

        if objective not in ["ml", "map"]:
            raise ParameterError(
                "objective may only be 'ml' or 'map': you have selected {:s}".format(objective)
            )

        if (
            objective == "ml"
        ):  # we are doing maximum likelihood minimization, shouldn't be any prior defined
            if "map_data" in spline_parameters:
                if spline_parameters["map_data"] is not None:
                    raise ParameterError(
                        "if 'objective' is 'ml' then 'map_data' structure containing priors should not be included"
                    )
            # Fill them in with Nones, so the code logic can proceed
            spline_parameters["map_data"] = {}
            spline_parameters["map_data"]["logprior"] = None
            spline_parameters["map_data"]["dlogprior"] = None
            spline_parameters["map_data"]["ddlogprior"] = None

        elif objective == "map":
            if "map_data" not in spline_parameters:
                raise ParameterError(
                    "if 'objective' is 'map' you must include 'map_data' structure"
                )
            elif spline_parameters["map_data"] == None:
                raise ParameterError("MAP data must be defined if objective is MAP")
            else:
                map_data = spline_parameters["map_data"]
                if map_data["logprior"] == None:
                    raise ParameterError("log prior must be included if objective is MAP")
                if map_data["dlogprior"] == None:
                    raise ParameterError("d(log prior) must be included if objective is MAP")
                if map_data["ddlogprior"] == None:
                    raise ParameterError("d^2(log prior) must be included if objective is MAP")

        # we need to intialize our function starting point

        if spline_parameters["optimization_algorithm"] != "Custom-NR":
            # we are passing it on to scipy

            if "optimize_options" not in spline_parameters:
                spline_parameters["optimize_options"] = {
                    "disp": True,
                    "ftol": 10 ** (-7),
                    "xtol": 10 ** (-7),
                }

            if "tol" in spline_parameters["optimize_options"]:
                # scipy doesn't like 'tol' within options
                spline_parameters["scipy_tol"] = spline_parameters["optimize_options"]["tol"]
                spline_parameters["optimize_options"].pop("tol", None)
            else:
                spline_parameters["scipy_tol"] = None  # this is just the default anyway.

            if spline_parameters["optimization_algorithm"] not in [
                "Newton-CG",
                "CG",
                "BFGS",
                "L-BFGS-B",
                "TNC",
                "SLSQP",
            ]:
                raise ParameterError(
                    "Optimization method {:s} is not supported".format(
                        spline_parameters["optimization_algorithm"]
                    )
                )
        else:
            if "optimize_options" not in spline_parameters:
                spline_parameters["optimize_options"] = dict()
            if "gtol" not in spline_parameters["optimize_options"]:
                spline_parameters["optimize_options"]["tol"] = 10 ** (-7)

        self.spline_parameters = spline_parameters

        xinit, yinit = self._get_initial_spline_points()

        self.spline_data = self._get_initial_spline(xinit, yinit)

        if self.n_bootstraps > 0:
            self.fes_functions = list()
        else:
            self.fes_functions = None

    def _get_initial_spline_points(self):

        """
        Uses information from spline_parameters to construct initial
        points to create a spline frmo which to start the minimization.

        Parameters
        ----------
        None

        Returns
        -------
        xinit : ndarray, float
            x-values of spline to be fit for start of minimizaton
        yinit : ndarray, float, shape
            y-values of spline to be fit for start of minimizaton

        """

        spline_parameters = self.spline_parameters
        nspline = spline_parameters["nspline"]
        kdegree = spline_parameters["kdegree"]
        xrange = spline_parameters["xrange"]

        if spline_parameters["spline_initialize"] == "bias_free_energies":
            initvals = self.mbar.f_k
            # initialize to the bias free energies
            if "bias_centers" in spline_parameters:  # if we are provided bias center, use them
                bias_centers = spline_parameters["bias_centers"]
                sort_indices = np.argsort(bias_centers)
                K = self.mbar.K
                if K < 2 * nspline:
                    noverfit = int(np.round(K / 2))
                    tinit = np.zeros(noverfit + kdegree + 1)
                    tinit[0:kdegree] = xrange[0]
                    tinit[kdegree : noverfit + 1] = np.linspace(
                        xrange[0], xrange[1], num=noverfit + 1 - kdegree, endpoint=True
                    )
                    tinit[noverfit + 1 : noverfit + kdegree + 1] = xrange[1]
                    # problem: bin centers might not actually be sorted.
                    binit = make_lsq_spline(
                        bias_centers[sort_indices], initvals[sort_indices], tinit, k=kdegree
                    )
                    xinit = np.linspace(xrange[0], xrange[1], num=2 * nspline)
                    yinit = binit(xinit)
                else:
                    xinit = bias_centers[sort_indices]
                    yinit = initvals[sort_indices]
            else:
                # assume equally spaced bias scenters
                xinit = np.linspace(xrange[0], xrange[1], self.mbar.K + 1)[1:-1]
                yinit = initvals

        elif spline_parameters["spline_initialize"] == "explicit":
            if "xinit" in spline_parameters:
                xinit = spline_parameters["xinit"]
            else:
                raise ParameterError(
                    "spline_initialize set as explicit, but no xinit array specified"
                )
            if "yinit" in spline_parameters:
                yinit = spline_parameters["yinit"]
            else:
                raise ParameterError(
                    "spline_initialize set as explicit, but no yinit array specified"
                )
        elif spline_parameters["spline_initialize"] == "zeros":  # initialize to zero
            xinit = np.linspace(xrange[0], xrange[1], nspline + kdegree)
            yinit = np.zeros(len(xinit))
        else:
            spline_initialization = spline_parameters["spline_initialize"]
            raise ParameterError(f"Initialization type {spline_initialization} not recognized")

        return xinit, yinit

    def _get_initial_spline(self, xinit, yinit):

        """
        Uses information from spline_parameters to construct initial
        points to create a spline frmo which to start the minimization.

        Parameters
        ----------
        xinit : ndarray, float:
            x-values of spline to fit
        yinit : ndarray, float:
            y-values of spline to fit


        Returns
        -------
        spline_data : dict() of information used in optimizing the spline parameters

        """

        spline_data = {}

        spline_parameters = self.spline_parameters

        kdegree = spline_parameters["kdegree"]  # degree of the spline
        nspline = spline_parameters["nspline"]  # number of spline points.
        xrange = spline_parameters["xrange"]

        # this is kind of duplicated: figure out how to simplify.
        # t has to be of size nsplines + kdegree + 1

        t = np.zeros(nspline + kdegree + 1)
        t[0:kdegree] = xrange[0]
        t[kdegree : nspline + 1] = np.linspace(
            xrange[0], xrange[1], num=nspline + 1 - kdegree, endpoint=True
        )
        t[nspline + 1 : nspline + kdegree + 1] = xrange[1]

        # initial fit function
        # problem: bin centers might not actually be sorted. Should data
        # getting to here be already sorted?

        sort_indices = np.argsort(xinit)
        b = make_lsq_spline(xinit[sort_indices], yinit[sort_indices], t, k=kdegree)
        # one, since the FES is only determined up to a constant.
        b.c = b.c - b.c[0]  # We zero out the first
        # the bspline coefficients are the variables we care about.
        xi = b.c[1:]
        xold = xi.copy()

        # The function is \sum_n F(phi(x_n)) + \sum_k ln \int exp(-F(xi) - u_k(xi)) dxi
        # if we assume bsplines are of the form f(x) = a*b_i(x), then
        # dF/dtheta is simply the basis function that has support over that
        # region of space

        # we now need the derivative of the function WRT the coefficients. Doesn't change knots or degree.
        # A vector function that is
        db_c = list()
        for i in range(nspline):
            dc = np.zeros(nspline)
            dc[i] = 1.0
            db_c.append(BSpline(b.t, dc, b.k))
            # OK, we've defined the derivatives.

        # same for the next execution. Not sure if best time to save it.

        # We also construct integration ranges for the derivatives, since no point in integrating when
        # the function is zero.
        xrangei = np.zeros([nspline, 2])
        for i in range(0, nspline):
            xrangei[i, 0] = t[i]
            xrangei[i, 1] = t[i + kdegree + 1]

        # set integration ranges for derivative products; saves time on
        # integration.
        xrangeij = np.zeros([nspline, nspline, 2])
        for i in range(0, nspline):
            for j in range(0, nspline):
                xrangeij[i, j, 0] = np.max([xrangei[i, 0], xrangei[j, 0]])
                xrangeij[i, j, 1] = np.min([xrangei[i, 1], xrangei[j, 1]])

        spline_data["initial_coefficients"] = xi
        spline_data["bspline_derivatives"] = db_c
        spline_data["bspline"] = b
        spline_data["xrangei"] = xrangei
        spline_data["xrangeij"] = xrangeij

        return spline_data

    def _generate_fes_spline(self, b, x_n, w_n):

        """
        Given an fes object with the spline set up, determine
        the information necessary to define a FES.

        Parameters
        ----------

        b : int

            Which bootstrap this is: b==0 is the initial value with the "untouched" data.

        x_n : np.ndarray, float, shape=(N,D)
            x_n[n] is the d-dimensional coordinates of the samples, where D is the reduced dimensional space.

        w_n : np.ndarray, float, shape=(sself.N)

            Weights for each sample for the state in which we want the FES (usually, the unbiased state)

        Returns
        -------
        None

        Data is stored in self.fes_function or self.fes_functions (for bootstrap replicates).

        """

        if b == 0:
            xi = self.spline_data["initial_coefficients"].copy()
        else:
            xi = self.spline_data["first_coefficients"].copy()

        spline_parameters = self.spline_parameters
        spline_data = self.spline_data
        func = self._bspline_calculate_f
        grad = self._bspline_calculate_g
        hess = self._bspline_calculate_h

        if spline_parameters["optimization_algorithm"] != "Custom-NR":
            if spline_parameters["optimization_algorithm"] == "Newton-CG":
                hessian = hess
            else:
                hessian = None

            spline_args = (x_n, w_n)

            results = minimize(
                func,
                xi,
                args=spline_args,
                method=spline_parameters["optimization_algorithm"],
                jac=grad,
                tol=spline_parameters["scipy_tol"],
                hess=hess,
                options=spline_parameters["optimize_options"],
            )
            bspline = self._val_to_spline(results["x"], form="log")  # TODO: where is this saved?
            savexi = results["x"]
        else:
            if "gtol" in spline_parameters["optimize_options"]:
                tol = spline_parameters["optimize_options"]["gtol"]
            elif "tol" in spline_parameters["optimize_options"]:
                tol = spline_parameters["optimize_options"]["tol"]

            # should come up with better way to make sure it passes the first time.
            dg = tol * 1e10
            firsttime = True

            while dg > tol:  # until we reach the tolerance.

                f = func(xi, *spline_args)

                # we need some error handling: if we stepped too far, we should go back
                # still not great error handling.  Requires something
                # close.

                if firsttime:
                    firsttime = False
                else:
                    count = 0
                    # we went too far!  Pull back.
                    # pylint: disable=used-before-assignment,undefined-variable
                    while (f >= fold * (1.1) and count < 5) or (np.isinf(f)):
                        f = fold
                        # let's not step as far:
                        dx = 0.9 * dx
                        # step back 90% of dx
                        xi = xold - dx
                        xold = xi.copy()
                        f = func(xi, *spline_args)
                        count += 1

                fold = f
                xold = xi.copy()

                g = grad(xi, *spline_args)
                h = hess(xi, *spline_args)

                # now find the new point.
                # x_n+1 = x_n - f''(x_n)^-1 f'(x_n)
                # which we solve more stably as:
                # x_n - x_n+1 = f''(x_n)^-1 f'(x_n)
                # f''(x_n)(x_n-x_n+1) = f'(x_n)
                # solution is dx = x_n-x_n+1

                dx = np.linalg.lstsq(h, g, rcond=None)[0]
                xi = xold - dx
                if spline_parameters["optimize_options"]["disp"]:
                    dg = np.sqrt(np.dot(g, g))
                    logger.info("f = {:.10f}. gradient norm = {:.10f}".format(f, np.sqrt(dg)))
            bspline = self._val_to_spline(xi, form="log")
            savexi = xi

        if b == 0:
            nparameters = len(savexi)
            minus_log_likelihood = func(savexi, *spline_args)
            self.spline_data["first_coefficients"] = savexi

            # now store BIC and AIC
            results = self._calculate_information_criteria(
                nparameters, minus_log_likelihood, self.N
            )
            self.spline_data["aic"] = results["aic"]
            self.spline_data["bic"] = results["bic"]

        if b == 0:
            self.fes_function = bspline
        else:
            self.fes_functions.append(bspline)

    @staticmethod
    def _calculate_information_criteria(nparameters, minus_log_likelihood, N):

        """
        Calculate and store various informaton criterias

        Parameters
        ----------

        nparameters : int, number of parameters in the model

        minus_log_likelihood : float, minus the log likelihood of the model.

        N : int, number of samples

        Results
        -------

        results : dictionary
            Keys are "aic" (Akaike information critera) and "bic" (Bayesian information criteria)

        """
        results = {}
        # calculate the AIC
        # formula is: 2(number of parameters) - 2 * loglikelihood
        results["aic"] = 2 * nparameters + 2 * minus_log_likelihood

        # calculate the BIC
        # formula is: ln(number of data points) * (number of parameters) - 2 ln (likelihood at maximum)
        results["bic"] = 2 * np.log(N) * nparameters + 2 * minus_log_likelihood

        # potential problems: we don't compute the full log likelihood currently since we
        # exclude the potential energy part - will have to see what this is in reference to.
        # this shouldn't be too much of a problem, since we are interested in choosing BETWEEN models.

        return results

    def get_information_criteria(self, type="akaike"):
        """
        returns the Akaike or Bayesian Informatiton Criteria for the model if it exists.

        Parameters
        ----------

        type : string
           either 'Akaike' (or 'akaike' or 'aic') or 'Bayesian' (or 'bayesian' or 'bic')

        Returns
        -------

        float :
           value of information criteria

        """

        if self.fes_type != "spline":
            raise ParameterError(
                "Information criteria currently only defined for spline approaches, you are currently using {:s}".format(
                    type
                )
            )
        if type in ["akaike", "Akaike", "AIC", "aic"]:
            return self.spline_data["aic"]
        elif type in ["bayesian", "Bayesian", "BIC", "bic"]:
            return self.spline_data["bic"]
        else:
            raise ParameterError("Information criteria of type '{:s}' not defined".format(type))

    def get_fes(
        self, x, reference_point="from-lowest", fes_reference=None, uncertainty_method=None
    ):
        """
        Returns values of the FES at the specified x points.

        Parameters
        ----------

        x : numpy.ndarray of D dimensions, where D is the dimensionality of the FES defined.

        reference_point : str, optional
            Method for reporting values and uncertainties (default : 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on FES are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \\sum_i p_i = 1 is used to determine uncertainties spread out through the FES
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        uncertainty_method : str, optional
            Method for computing uncertainties (default: None)

        fes_reference:
            an N-d point specifying the reference state. Ignored except with uncertainty method ``from_specified``

        Returns
        -------
        dict
            'f_i' : np.ndarray, float, shape=(K)
                result_vals['f_i'][i] is the dimensionless free energy of the x_i point, relative to the reference point
            'df_i' : np.ndarray, float, shape=(K)
                result_vals['df_i'][i] is the uncertainty in the difference of x_i with respect to the reference point
                Only included if uncertainty_method is not None

        """

        # if it's not an array, make it one.
        x = np.array(x)

        # it's a 1D array, instead of a Nx1 array.  Reshape.
        if len(np.shape(x)) <= 1:
            x = x.reshape(-1, 1)

        if reference_point == "from-specified" and fes_reference is None:
            logger.info(
                "No reference state specified for FES, using uncertainty_method = from-specified"
            )

        if self.fes_type == "histogram":
            result_vals = self._get_fes_histogram(
                x, reference_point, fes_reference, uncertainty_method
            )

        elif self.fes_type == "kde":
            # TODO: check dimensionality here
            result_vals = self._get_fes_kde(x, reference_point, fes_reference, uncertainty_method)

        elif self.fes_type == "spline":
            result_vals = self._get_fes_spline(
                x, reference_point, fes_reference, uncertainty_method
            )
        else:
            raise ParameterError("fes_type {self.fes_type} is not supported")

        return result_vals

    def get_mbar(self):
        """return the MBAR object being used by the FES

        Returns
        -------
           MBAR object
        """
        if self.mbar is not None:
            return self.mbar
        else:
            raise DataError("MBAR in the FES object is not initialized, cannot return it.")

    def get_kde(self):
        """return the KernelDensity object if it exists.

        Returns
        -------
        sklearn KernelDensity object
        """

        if self.fes_type == "kde":
            if self.kde != None:
                return self.kde
            else:
                raise ParameterError(
                    "Can't return the KernelDensity object because kde not yet defined"
                )
        else:
            raise ParameterError("Can't return the KernelDensity object because fes_type != kde")

    def _get_fes_histogram(
        self, x, reference_point="from-lowest", fes_reference=None, uncertainty_method=None
    ):
        """
        Returns values of the FES at the specified x points for histogram FESs.

        Parameters
        ----------

        x : numpy.ndarray of D dimensions, where D is the dimensionality of the FES defined.

        reference_point : str, optional
            Method for reporting values and uncertainties (default : 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on FES are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \\sum_i p_i = 1 is used to determine uncertainties spread out through the FES
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        uncertainty_method : str, optional
            Method for computing uncertainties (default: None)

        fes_reference:
            an N-d point specifying the reference state. Ignored except with uncertainty method ``from_specified``

        Returns
        -------
        dict
            'f_i' : np.ndarray, float, shape=(K)
                result_vals['f_i'][i] is the dimensionless free energy of the x_i point, relative to the reference point
            'df_i' : np.ndarray, float, shape=(K)
                result_vals['df_i'][i] is the uncertainty in the difference of x_i with respect to the reference point
                Only included if uncertainty_method is not None

        """

        if np.shape(x)[1] != self.histogram_data["dims"]:
            raise DataError(
                "query coordinates have inconsistent dimension with the data the FES is fit to."
            )

        if (
            uncertainty_method not in ["bootstrap", "analytical"]
            and uncertainty_method is not None
        ):
            raise ParameterError(f"Uncertainty_method {uncertainty_method} is not a valid option")

        if uncertainty_method == "bootstrap":
            if self.histogram_datas is None:
                raise ParameterError(
                    "Can't calculate uncertainties via bootstrap if bootstrapping was not performed when running get_fes"
                )
            else:
                n_bootstraps = len(self.histogram_datas)

        # set up structure for return data
        result_vals = {}

        # retrive data for each bootstrap
        histogram_data = self.histogram_data
        histogram_datas = self.histogram_datas

        bins = histogram_data["bins"]
        dims = histogram_data["dims"]
        bin_order = histogram_data["bin_order"]
        nbins = len(bin_order)

        # figure out which bins the values are in.
        if dims == 1:
            # what gridpoint does each x fall into?
            # -1 and nbinsperdim are out of range
            loc_indices = np.digitize(x, bins[0]) - 1
        else:
            loc_indices = np.zeros([len(x), dims], dtype=int)
            for d in range(dims):
                # -1 and nbinsperdim are out of range
                loc_indices[:, d] = np.digitize(x[:, d], bins[d]) - 1

        # figure out which grid point the fes_reference is at
        if reference_point == "from-specified":
            if fes_reference is not None:
                if dims == 1:
                    # make it a list for reduced code duplication.
                    fes_reference = [fes_reference]
                fes_ref_grid = np.zeros([dims], dtype=int)
                for d in range(dims):
                    # -1 and nbins_per_dim are out of range
                    fes_ref_grid[d] = np.digitize(fes_reference[d], bins[d]) - 1
                    if fes_ref_grid[d] == -1 or fes_ref_grid[d] == len(bins[d]):
                        raise ParameterError(
                            "Specified reference point coordinate {:f} in dim {:d} grid point is out of the FES region [{:f},{:f}]".format(
                                fes_ref_grid[d], d, np.min(bins[d]), np.max(bins[d])
                            )
                        )
            else:
                raise ParameterError("Specified reference point for FES not given")

        if reference_point in ["from-lowest", "from-specified", "all-differences"]:

            if reference_point == "from-lowest":
                # Determine free energy with lowest free energy to serve as reference point
                j = histogram_data["f"].argmin()
            elif reference_point == "from-specified":
                # find the label of this bin
                ref_bin_label = histogram_data["bin_label"][tuple(fes_ref_grid)]
                # then find the invariant free energy index of this bin
                j = bin_order[ref_bin_label]
            elif reference_point == "all-differences":
                raise ParameterError(
                    "reference point method of 'all-differences' is not yet supported for histogram "
                    "FES types (not implemented)"
                )
            f_i = histogram_data["f"] - histogram_data["f"][j]

            # now calculate uncertainty for these reference_method approaches.
            df_i = np.zeros(len(histogram_data["f"]), np.float64)

            if uncertainty_method == "analytical":
                K = self.mbar.K
                # Compute uncertainties in free energy at each gridpoint by
                # forming matrix of W_nk.
                N_k = np.zeros([K + nbins], np.int64)
                N_k[0:K] = self.mbar.N_k
                W_nk = np.zeros([self.mbar.N, K + nbins], np.float64)
                W_nk[:, 0:K] = np.exp(self.mbar.Log_W_nk)

                log_w_n = self.mbar._computeUnnormalizedLogWeights(self.u_n)

                # loop over the nonzero bins
                for label in histogram_data["bin_label"].values():
                    # Get indices of samples that fall in this bin.
                    indices = np.where(histogram_data["sample_label"] == label)

                    # Compute normalized weights for this state.
                    flabel = bin_order[label]
                    W_nk[indices, K + flabel] = np.exp(
                        log_w_n[indices] + self.histogram_data["f"][flabel]
                    )

                # Compute asymptotic covariance matrix using specified
                # method.
                Theta_ij = self.mbar._computeAsymptoticCovarianceMatrix(W_nk, N_k)

                # Compute uncertainties with respect to difference in free energy
                # from this state j.
                for i in range(nbins):
                    df_i[i] = math.sqrt(
                        Theta_ij[K + i, K + i]
                        + Theta_ij[K + j, K + j]
                        - 2.0 * Theta_ij[K + i, K + j]
                    )

            elif uncertainty_method == "bootstrap":
                fall = np.zeros([len(histogram_data["f"]), n_bootstraps])
                for b in range(n_bootstraps):
                    h = histogram_datas[b]  # just to make this shorter
                    fall[:, b] = h["f"] - h["f"][j]
                df_i = np.std(fall, axis=1)

        elif reference_point == "from-normalization":
            # Determine uncertainties from normalization that \sum_i p_i = 1.
            # need to reimplement this . . . maybe.

            # # Compute bin probabilities p_i
            # p_i = np.exp(-self.fbin - logsumexp(-self.fbin))
            #
            # # todo -- eliminate triple loop over nbins!
            # # Compute uncertainties in bin probabilities.
            # d2p_i = np.zeros([nbins], np.float64)
            # for k in range(nbins):
            #     for i in range(nbins):
            #         for j in range(nbins):
            #             delta_ik = 1.0 * (i == k)
            #             delta_jk = 1.0 * (j == k)
            #             d2p_i[k] += p_i[k] * (p_i[i] - delta_ik) * p_i[k] * (
            #                 p_i[j] - delta_jk) * Theta_ij[K + i, K + j]
            #
            # # Transform from d2p_i to df_i
            # d2f_i = d2p_i / p_i ** 2
            # df_i = np.sqrt(d2f_i)

            raise ParameterError(
                "uncertainty_method 'from-normalization' is not currently supported for histograms"
            )

        fx_vals = np.zeros(len(x))
        dfx_vals = np.zeros(len(x))

        # figure out how many grid points in each direction
        maxp = np.zeros(dims, int)
        for d in range(dims):
            maxp[d] = len(bins[d])

        for i, l in enumerate(loc_indices):
            # Must be a way to list comprehend this?
            if np.any(l < 0):  # out of index below
                fx_vals[i] = np.nan
                dfx_vals[i] = np.nan
                continue
            if np.any(l >= maxp - 1):  # out of index above
                fx_vals[i] = np.nan
                dfx_vals[i] = np.nan
                continue

            bin_label = histogram_data["bin_label"][tuple(l)]
            if bin_label >= 0:
                fx_vals[i] = f_i[bin_order[bin_label]]
                dfx_vals[i] = df_i[bin_order[bin_label]]
            else:
                fx_vals[i] = np.nan
                dfx_vals[i] = np.nan

        # Return dimensionless free energy and uncertainty.
        result_vals["f_i"] = fx_vals
        if uncertainty_method is not None:
            result_vals["df_i"] = dfx_vals

        if reference_point == "all-differences":
            if uncertainty_method == "analytical":
                # Report uncertainties in all free energy differences as
                # well.
                diag = Theta_ij.diagonal()
                dii = diag[K, K + nbins]  # appears broken?  Not used?
                d2f_ij = dii + dii.transpose() - 2 * Theta_ij[K : K + nbins, K : K + nbins]

                # unsquare uncertainties
                df_ij = np.sqrt(d2f_ij)

                dfxij_vals = np.zeros([len(x), len(x)])

                bin_is = ()
                for i, l in enumerate(loc_indices):
                    bin_i = histogram_data["bin_order"][self.histogram_data["bin_label"][tuple(l)]]
                bin_is.append(bin_i)

                for i, vi in bin_is:
                    for j, vj in bin_i:
                        if vi != -1 and vj != 1:
                            dfxij_vals[i, j] = df_ij[vi, vj]
                        else:
                            dfxij_vals[i, j] = np.nan

            elif uncertainty_method == "bootstrap":  # TODO: check this is working!
                dfxij_vals = np.zeros([len(histogram_data["f"]), len(histogram_data["f"])])
                fall = np.zeros([len(histogram_data["f"]), len(histogram_data["f"]), n_bootstraps])
                for b in range(n_bootstraps):
                    h = histogram_datas[b]
                    for i in range(nbins):
                        fall[i, j, b] = (
                            histogram_datas[b]["f"] - histogram_datas[b]["f"].transpose()
                        )
                dfxij_vals = np.std(fall, axis=2)
            if uncertainty_method is not None:
                # Return dimensionless free energy and uncertainty.
                result_vals["df_ij"] = dfxij_vals

        return result_vals

    def _get_fes_kde(
        self, x, reference_point="from-normalization", fes_reference=None, uncertainty_method=None
    ):
        """

        Returns values of the FES at the specified x points for kde FESs.

        Parameters
        ----------

        x : numpy.ndarray of D dimensions, where D is the dimensionality of the FES defined.

        reference_point : str, optional
            Method for reporting values and uncertainties (default : 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on FES are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \\sum_i p_i = 1 is used to determine uncertainties spread out through the FES
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        uncertainty_method : str, optional
            Method for computing uncertainties (default : None)

        fes_reference:
            an N-d point specifying the reference state. Ignored except with uncertainty method ``from_specified``

        Returns
        -------
        dict
            'f_i' : np.ndarray, float, shape=(K)
                result_vals['f_i'][i] is the dimensionless free energy of the x_i point, relative to the reference point
            'df_i' : np.ndarray, float, shape=(K)
                result_vals['df_i'][i] is the uncertainty in the difference of x_i with respect to the reference point
                Only included if uncertainty_method is not None

        """

        if np.shape(x)[1] != np.shape(self.kde.sample())[1]:
            raise DataError(
                "query coordinates have inconsistent dimension with the data the FES is fit to."
            )

        result_vals = {}
        f_i = -self.kde.score_samples(x)  # gives the LOG density, which is what we want.

        if reference_point == "from-lowest":
            fmin = np.min(f_i)
            f_i = f_i - fmin
        elif reference_point == "from-specified":
            fmin = -self.kde.score_samples(np.array(fes_reference).reshape(1, -1))
            f_i = f_i - fmin
        elif reference_point == "from-normalization":
            # uncertainites "from normalization" reference is already applied, since
            # the density is normalized.
            pass
        else:
            raise ParameterError(
                f"reference point choice {reference_point} for kde is unavailable"
            )

        result_vals["f_i"] = f_i

        # now calculate bootstrap uncertainties

        if uncertainty_method is None:
            df_i = None

        elif uncertainty_method == "bootstrap":

            if self.kdes is None:
                raise ParameterError(
                    f"Cannot calculate bootstrap error of boostrap KDE's not determined"
                )
            else:
                n_bootstraps = len(self.kdes)

            fall = np.zeros([len(x), n_bootstraps])
            for b in range(n_bootstraps):
                fall[:, b] = -self.kdes[b].score_samples(x) - fmin
            df_i = np.std(fall, axis=1)
        else:
            raise ParameterError(
                f"Uncertainty method {uncertainty_method} for kde is not implemented"
            )

        result_vals["df_i"] = df_i

        return result_vals

    def _get_fes_spline(
        self, x, reference_point="from_lowest", fes_reference=0.0, uncertainty_method=None
    ):
        """

        Returns values of the FES at the specified x points for spline FESs.

        Parameters
        ----------

        x : numpy.ndarray of D dimensions, where D is the dimensionality of the FES defined.

        reference_point : str, optional
            Method for reporting values and uncertainties (default : 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on FES are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \\sum_i p_i = 1 is used to determine uncertainties spread out through the FES
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        uncertainty_method : str, optional
            Method for computing uncertainties (default : None)

        fes_reference:
            an N-d point specifying the reference state. Ignored except with uncertainty method ``from_specified``

        Returns
        -------
        dict
            'f_i' : np.ndarray, float, shape=(K)
                result_vals['f_i'][i] is the dimensionless free energy of the x_i point, relative to the reference point
            'df_i' : np.ndarray, float, shape=(K)
                result_vals['df_i'][i] is the uncertainty in the difference of x_i with respect to the reference point
                Only included if uncertainty_method is not None

        """

        if np.shape(x)[1] != 1:
            raise DataError("splines FES only supported in 1D")

        result_vals = {}
        # for splines now, should only be 1D.  x is passed in as a 2D array, need to covert
        # back to 1D array before being put in fes_function (which will preserve shape)
        x = x[:, 0]

        # before being put here, needs to be converted back
        f_i = self.fes_function(x)

        if reference_point == "from-lowest":
            fmin = np.min(f_i)
            f_i = f_i - fmin

        elif reference_point == "from-specified":
            fmin = -self.fes_function(np.array(fes_reference).reshape(1, -1))
            f_i = f_i - fmin

        else:
            raise ParameterError(
                f"reference point {reference_point} not implemented for spline fes"
            )

        if uncertainty_method is None:
            df_i = None

        if uncertainty_method == "bootstrap":
            if self.fes_functions is None:
                raise ParameterError(
                    "Cannot calculate via uncertainties error if boostrapping was not peformed running get_fes"
                )
            else:
                n_bootstraps = len(self.fes_functions)

            dim_breakdown = [d for d in x.shape] + [n_bootstraps]
            fall = np.zeros(dim_breakdown)
            for b in range(n_bootstraps):
                fall[:, b] = self.fes_functions[b](x) - fmin
            df_i = np.std(fall, axis=-1)

        # uncertainites "from normalization" reference is applied, since
        # the density is normalized.
        result_vals["f_i"] = f_i
        result_vals["df_i"] = df_i

        return result_vals

    def sample_parameter_distribution(
        self, x_n, mc_parameters=None, decorrelate=True, verbose=True
    ):
        """
        Samples the valus of the spline parameters with MC.

        Parameters
        ----------
        x_n : numpy.ndarray of D dimensions
           D is the dimensionality of the FES defined.

        mc_parameters: dictionary
            niteratons : int
                 number of iterations of the Monte Carlo procedure
            fraction_change : float
                 which fraction of the range of input parameters is used to make new MC moves.
            sample_every : int
                 the frequency in steps at which the MC timeseries is saved.
            print_every : int
                 the frequency in steps aat which the MC timeseries is saved to log.
            logprior : function,
                 the function of parameters, the Function must take a single argument, an array the size of parameters
                 in in the same order as a used by the internal functions.

        decorrelate : boolean
            Whether to decorrelate the time series of output points

        verbose : boolean
            Whether to print high levels of information to the logger

        Returns
        -------
        None

        """

        # determine the range of the bspline at the start of the
        # process: changes are made as fractions of this

        if self.fes_type != "spline":
            ParameterError("Sampling of posterior is only supported for spline type")

        spline_parameters = self.spline_parameters

        if spline_parameters is None:
            ParameterError("Must specify spline_parameters to sample the distributions")

        spline_weights = spline_parameters["spline_weights"]

        # need the x-range for all methods, since we need to
        xrange = spline_parameters["xrange"]
        # numerically integrate over this range

        if self.fes_function is None:
            ParameterError(
                "Need to generate an initial splined FES using generate_fes before performing MCMC sampling"
            )

        if mc_parameters is None:
            logger.info("Using default MC parameters")
            mc_parameters = dict()

        if "niterations" not in mc_parameters:
            mc_parameters["niterations"] = 5000
        if "fraction_change" not in mc_parameters:
            mc_parameters["fraction_change"] = 0.01
        if "sample_every" not in mc_parameters:
            mc_parameters["sample_every"] = 50
        if "print_every" not in mc_parameters:
            mc_parameters["print_every"] = 1000
        if "logprior" not in mc_parameters:
            mc_parameters["logprior"] = lambda x: 0

        niterations = mc_parameters["niterations"]
        fraction_change = mc_parameters["fraction_change"]
        sample_every = mc_parameters["sample_every"]
        print_every = mc_parameters["print_every"]
        logprior = mc_parameters["logprior"]

        # create a new copy of the spline for MC sampling.
        self.mc_data = dict()

        # we would like to make this below a copy, but BSpline doesn't have copy.

        self.mc_data["bspline"] = self.fes_function
        bspline = self.mc_data["bspline"]

        # ensure normalization of spline
        def prob(x):
            return np.exp(-bspline(x))

        norm = self._integrate(prob, xrange[0], xrange[1])
        bspline.c = bspline.c + np.log(norm)

        # make a copy of the original spline to preserve it.
        self.mc_data["original_spline"] = BSpline(bspline.t, bspline.c, bspline.k)

        # this might not work as well for probability
        c = bspline.c
        crange = np.max(c) - np.min(c)
        dc = fraction_change * crange

        self.mc_data["naccept"] = 0
        csamples = np.zeros([len(c), int(niterations) // int(sample_every)])
        logposteriors = np.zeros(int(niterations) // int(sample_every))
        self.mc_data["first_step"] = True

        for n in range(niterations):
            results = self._MC_step(x_n, self.w_n, dc, xrange, spline_weights, logprior)
            if n % sample_every == 0:
                csamples[:, n // sample_every] = results["c"]
                logposteriors[n // sample_every] = results["logposterior"]
            if n % print_every == 0 and verbose:
                logger.info(
                    "MC Step {:d} of {:d}".format(n, niterations),
                    str(results["logposterior"]),
                    str(bspline.c),
                )

        # We now have a distribution of samples of parameters sampled according
        # to the posterior.

        # decorrelate the data
        t_mc = 0
        g_mc = None

        if verbose:
            logger.info("Done MC sampling")

        if decorrelate:
            t_mc, g_mc, Neff = timeseries.detect_equilibration(logposteriors)
            logger.info(
                "First equilibration sample is {:d} of {:d}".format(t_mc, len(logposteriors))
            )
            equil_logp = logposteriors[t_mc:]
            g_mc = timeseries.statistical_inefficiency(equil_logp)
            if verbose:
                logger.info("Statistical inefficiency of log posterior is {:.3g}".format(g_mc))
            g_c = np.zeros(len(c))
            for nc in range(len(c)):
                g_c[nc] = timeseries.statistical_inefficiency(csamples[nc, t_mc:])
            if verbose:
                logger.info("Time series for spline parameters are : {:s}".format(str(g_c)))
            maxgc = np.max(g_c)
            meangc = np.mean(g_c)
            guse = g_mc  # doesn't affect the distribution that much
            indices = timeseries.subsample_correlated_data(equil_logp, g=guse)
            logposteriors = equil_logp[indices]
            csamples = (csamples[:, t_mc:])[:, indices]
            if verbose:
                logger.info("samples after decorrelation : {:d}".format(np.shape(csamples)[1]))

        self.mc_data["samples"] = csamples
        self.mc_data["logposteriors"] = logposteriors
        self.mc_data["mc_parameters"] = mc_parameters
        self.mc_data["acceptance_ratio"] = self.mc_data["naccept"] / niterations
        if verbose:
            logger.info("Acceptance rate : {:5.3f}".format(self.mc_data["acceptance_ratio"]))
        self.mc_data["nequil"] = t_mc  # the start of the "equilibrated" data set
        self.mc_data["g_logposterior"] = g_mc  # statistical efficiency of the log posterior
        self.mc_data["g_parameters"] = g_c  # statistical efficiency of the parametere
        self.mc_data["g"] = guse  # statistical efficiency used for subsampling

    def get_confidence_intervals(self, xplot, plow, phigh, reference="zero"):

        """
        Parameters
        ----------
        xplot:
            data points we want to plot at
        plow:
            lowest percentile
        phigh:
            highest percentile

        Returns
        -------

        Dictionary of results.  Contains:
            plow : ndarray of float
                len(xplot) value of the parameter at plow percentile of the distribution at each x in xplot.
            phigh : ndarray of float
                value of the parameter at phigh percentile of the distribution at each x in xplot.
            median : ndarray of float
                value of the parameter at the median of the distribution at each x in xplot.
            values : ndarray of float
                shape [niterations//sample_every, len(xplot)] of the FES saved during the MCMC sampling at each input value of xplot.

        """
        if self.mc_data is None:
            raise DataError("No MC sampling has been done, cannot construct confidence intervals")

        # determine confidence intervals
        nplot = len(xplot)  # number of data points to plot.
        nsamples = len(self.mc_data["logposteriors"])
        samplevals = np.zeros([nplot, nsamples])

        csamples = self.mc_data["samples"]
        base_spline = self.mc_data["original_spline"]

        yvals = base_spline(xplot)

        for n in range(nsamples):
            # create a sample spline
            pcurve = BSpline(base_spline.t, csamples[:, n], base_spline.k)
            samplevals[:, n] = pcurve(xplot)

        # now determine the Bayesian confidence intervals.

        ylows = np.zeros(len(xplot))
        yhighs = np.zeros(len(xplot))
        ymedians = np.zeros(len(xplot))
        for n in range(len(xplot)):
            ylows[n] = np.percentile(samplevals[n, :], plow)
            yhighs[n] = np.percentile(samplevals[n, :], phigh)
            ymedians[n] = np.percentile(samplevals[n, :], 50)

        return_vals = dict()

        if reference == "zero":
            ref = np.min(yvals)
        elif reference == None:
            ref = 0
        else:
            raise ParameterError("{:s} is not a valid value for 'reference'")

        return_vals["plow"] = ylows - ref
        return_vals["phigh"] = yhighs - ref
        return_vals["median"] = ymedians - ref
        return_vals["values"] = yvals - ref

        return return_vals

    def get_mc_data(self):

        """convenience function to retrieve MC data

        Parameters
        ----------
        None

        Returns
        -------
        dict
            samples : samples of the parameters with size [len(parameters) times niterations/sample_every]
            logposteriors : log posteriors (which might be defined with respect to some reference) as a time series size [# points]
            mc_parameters : dictionary of parameters that were run with (see definitons in sample_parameter_distrbution)
            acceptance_ratio : overall acceptance ratio of the MC chain
            nequil : the start of the "equilibrated" data set (i.e. nequil-1 is the number that werer thrown out)
            g_logposterior : statistical efficiency of the log posterior
            g_parameters : statistical efficiency of the parametere
            g : statistical efficiency used for subsampling

        """

        if self.mc_data is None:
            raise DataError("No MC sampling has been done, cannot construct confidence intervals")
        else:
            return self.mc_data

    def _get_MC_loglikelihood(self, x_n, w_n, spline_weights, spline, xrange):

        """
        Parameters
        ----------

        x_n : np.ndarray, float, shape=(N,D)
            x_n[n] is the d-dimensional coordinates of the samples, where D is the reduced dimensional space.

        w_n : np.ndarray, float, shape=(sself.N)

            Weights for each sample for the state in which we want the FES (usually, the unbiased state)

        spline_weights : string
            which type of fit to the likelihood to use (see `generate_fes` options)

        spline : function of ndarray argument
            function current value of the spline for which likelihood is being calculated

        xrange : float, shape=(2)
           range over which the FES is defined (defined in spline_parameters ["xrange"]

        Returns
        -------

        loglikelihood : float
           log-likelihood of this spline

        """

        N = self.N
        K = self.K

        if spline_weights in ["simplesum", "biasedstates"]:
            loglikelihood = 0

            def splinek(x, kf):
                return spline(x) + self.spline_parameters["fkbias"][kf](x)

            def expk(x, kf):
                return np.exp(-splinek(x, kf))

            for k in range(K):
                x_kn = x_n[self.mbar.x_kindices == k]

                normalize = np.log(self._integrate(expk, xrange[0], xrange[1], args=(k)))
                if spline_weights == "simplesum":
                    loglikelihood += (N / K) * np.mean(splinek(x_kn, k))
                    loglikelihood += (N / K) * normalize
                elif spline_weights == "biasedstates":
                    loglikelihood += np.sum(splinek(x_kn, k))
                    loglikelihood += self.N_k[k] * normalize

        elif spline_weights == "unbiasedstate":
            loglikelihood = N * np.dot(w_n, spline(x_n))
            # no need to add normalization, should be normalized.

        return loglikelihood

    def _MC_step(self, x_n, w_n, stepsize, xrange, spline_weights, logprior):

        """sample over the posterior space of the FES as splined.

        Parameters
        ----------
        x_n:
            samples from the biased distribution
        w_n:
            weights of each sample.
        stepsize:
            sigma of the normal distribution used to propose steps
        xrange:
            Range the probility distribution is defined o er.
        spline_weights:
            Type of weighting used for maximum likelihood for splines.  See class
                        definition for description of types.
        logprior:
            function describing the prior of the parameters. Default is uniform.

        Outputs
        -------
        dict
            * 'c' : the value of the spline constants (len nsplines - we always assume normalized
            * 'logposterior' : the current value of the logposterior.

        Notes
        -----
        Modifies several saved variables saved in the structure.x

        """

        mc_data = self.mc_data  # keep it shorter
        bspline = self.mc_data["bspline"]

        if self.mc_data["first_step"]:
            c = bspline.c
            mc_data["previous_logposterior"] = self._get_MC_loglikelihood(
                x_n,
                w_n,
                self.spline_parameters["spline_weights"],
                bspline,
                self.spline_parameters["xrange"],
            ) - logprior(c)
            cold = bspline.c
            mc_data["first_step"] = True
            # create an extra one we can carry around
            mc_data["newspline"] = BSpline(bspline.t, bspline.c, bspline.k)

        mc_data["cold"] = bspline.c
        psize = len(mc_data["cold"])
        rchange = stepsize * np.random.normal()
        cnew = mc_data["cold"].copy()
        ci = np.random.randint(psize)
        cnew[ci] += rchange
        mc_data["newspline"].c = cnew

        # determine the change in the integral
        def prob(x):
            return np.exp(-mc_data["newspline"](x))

        new_integral = self._integrate(prob, xrange[0], xrange[1])

        cnew = cnew + np.log(new_integral)

        mc_data["newspline"].c = cnew  # this spline should now be normalized.

        # now calculate the change in log likelihood
        loglikelihood = self._get_MC_loglikelihood(
            x_n, w_n, spline_weights, mc_data["newspline"], xrange
        )

        newlogposterior = loglikelihood - logprior(cnew)
        dlogposterior = newlogposterior - (mc_data["previous_logposterior"])
        accept = False
        if dlogposterior <= 0:
            accept = True
        if dlogposterior > 0:
            if np.random.random() < np.exp(-dlogposterior):
                accept = True

        if accept:
            mc_data["bspline"].c = mc_data["newspline"].c
            mc_data["cold"] = bspline.c
            mc_data["previous_logposterior"] = newlogposterior
            mc_data["naccept"] = mc_data["naccept"] + 1
        results = dict()
        results["c"] = mc_data["bspline"].c
        results["logposterior"] = mc_data["previous_logposterior"]
        return results

    def _bspline_calculate_f(self, xi, x_n, w_n):

        """Calculate the maximum likelihood / KL divergence of the FES represented using B-splines.

        Parameters
        ----------

        xi : array of floats size nspline-1
            spline coefficients,
        w_n:
            weights for each sample.
        x_n:
            values of each sample.

        Output
        ------
        float
            function value at spline coefficients xi
        """

        mbar = self.mbar
        K = mbar.K
        N_k = mbar.N_k
        N = self.N

        bloc = self._val_to_spline(xi)  # convert the spline coefficients into a spline object
        spline_weights = self.spline_parameters[
            "spline_weights"
        ]  # how to weight the integrated splines in the final likelihood
        nspline = self.spline_parameters["nspline"]  # number of spline points
        kdegree = self.spline_parameters["kdegree"]  # degree of spline
        xrange = self.spline_parameters["xrange"]  # the range FES is defined over
        fkbias = self.spline_parameters["fkbias"]  # K biasing functions

        if spline_weights in ["simplesum", "biasedstates"]:
            pF = np.zeros(K)
            if spline_weights == "simplesum":
                f = 0
                for k in range(K):
                    f += (N / K) * np.mean(bloc(x_n[mbar.x_kindices == k]))
            elif spline_weights == "biasedstates":
                # multiply by K to get it in the same order of magnitude
                f = np.sum(bloc(x_n))

            if spline_weights == "simplesum":
                integral_scaling = (N / K) * np.ones(K)
            elif spline_weights == "biasedstates":
                integral_scaling = N_k

            expf = list()
            for k in range(K):
                # what is the biasing function for this state?
                # define the biasing function
                # define the exponential of f based on the current parameters
                # t.
                def expfk(x, kf=k):
                    return np.exp(-bloc(x) - fkbias[kf](x))

                # compute the partition function
                pF[k] = self._integrate(expfk, xrange[0], xrange[1], args=(k))
                expf.append(expfk)
            # subtract the free energy (add log partition function)
            f += np.dot(integral_scaling, np.log(pF))

        elif spline_weights == "unbiasedstate":  # just KL divergence of the unbiased potential
            # may need to recast w_n
            f = N * np.dot(w_n, bloc(x_n))

            def expf(x):
                return np.exp(-bloc(x))

            # setting limit to try to eliminate errors: hard time because it
            # goes so small.
            pF = self._integrate(expf, xrange[0], xrange[1])
            # subtract the free energy (add log partition function)
            f += N * np.log(pF)

        self.spline_data["bspline_expf"] = expf
        self.spline_data["bspline_pF"] = pF

        logprior = self.spline_parameters["map_data"]["logprior"]
        if logprior != None:
            # need to add the zero explicitly to the front
            f -= logprior(np.concatenate([[0], xi], axis=None))

        return f

    def _bspline_calculate_g(self, xi, x_n, w_n):
        """Calculate the gradient of the maximum likelihood / KL divergence of the FES represented using B-splines.

        Parameters
        -----------

        xi : array of floats, size nspline-1
            spline coefficients,
        w_n : array of floats
            weights for each sample.
        x_n : array of floats
            values of each sample.

        Output
        ------

        gradient: array of floats, size (nspline-1)

        """
        ##### COMPUTE THE GRADIENT #######
        # The gradient of the function is \sum_n [\sum_k W_k(x_n)] dF(phi(x_n))/dtheta_i - \sum_k <dF/dtheta>_k
        #
        # where <O>_k = \int O(xi) exp(-F(xi) - u_k(xi)) dxi / \int exp(-F(xi)
        # - u_k(xi)) dxi

        mbar = self.mbar
        K = mbar.K
        N_k = mbar.N_k
        N = self.N

        bloc = self._val_to_spline(xi)  # convert the spline coefficients into a spline object
        spline_weights = self.spline_parameters[
            "spline_weights"
        ]  # how to weight the integrated splines in the final likelihood
        nspline = self.spline_parameters["nspline"]  # number of spline points
        kdegree = self.spline_parameters["kdegree"]  # degree of spline
        xrange = self.spline_parameters["xrange"]  # the range FES is defined over
        fkbias = self.spline_parameters["fkbias"]  # K biasing functions
        db_c = self.spline_data[
            "bspline_derivatives"
        ]  # coefficients of the derivatives of the splines
        xrangei = self.spline_data[
            "xrangei"
        ]  # range the ith basis function of the spline is defined over

        pF = np.zeros(K)

        if spline_weights == "simplesum":
            integral_scaling = (N / K) * np.ones(K)
        elif spline_weights == "biasedstates":
            integral_scaling = N_k

        g = np.zeros(nspline - 1)

        for i in range(1, nspline):
            if spline_weights == "simplesum":
                for k in range(K):
                    g[i - 1] += (N / K) * np.mean(db_c[i](x_n[mbar.x_kindices == k]))
            elif spline_weights == "biasedstates":
                g[i - 1] = np.sum(db_c[i](x_n))
            elif spline_weights == "unbiasedstate":
                g[i - 1] = N * np.dot(w_n, db_c[i](x_n))

        # now the second part of the gradient.

        if spline_weights in ["biasedstates", "simplesum"]:
            gkquad = np.zeros([nspline - 1, K])

            def expf(x, k):
                return np.exp(-bloc(x) - fkbias[k](x))

            def dexpf(x, k):
                return db_c[i + 1](x) * expf(x, k)

            for k in range(K):
                # putting this in rather than saving the term so gradient and f
                # can be called independently
                pF[k] = self._integrate(expf, xrange[0], xrange[1], args=(k))

                for i in range(nspline - 1):
                    # Boltzmann weighted derivative with each biasing function
                    # now compute the expectation of each derivative
                    pE = self._integrate(dexpf, xrangei[i + 1, 0], xrangei[i + 1, 1], args=(k))

                    # normalize the expectation
                    gkquad[i, k] = pE / pF[k]
            g -= np.dot(gkquad, integral_scaling)

        elif spline_weights == "unbiasedstate":
            gkquad = 0  # not used here, but saved for Hessian calls.

            def expf(x):
                return np.exp(-bloc(x))

            # 0 is the value of gkquad. Recomputed here to avoid problems
            pF = self._integrate(expf, xrange[0], xrange[1])
            # with other scipy solvers
            pE = np.zeros(nspline - 1)

            def dexpf(x, index):
                return db_c[index + 1](x) * expf(x)

            for i in range(nspline - 1):
                # Boltzmann weighted derivative

                # now compute the expectation of each derivative
                pE[i] = self._integrate(dexpf, xrangei[i + 1, 0], xrangei[i + 1, 1], args=(i,))
                # normalize the expectation.
                pE[i] /= pF
            g -= N * pE

        dlogprior = self.spline_parameters["map_data"]["dlogprior"]
        if dlogprior != None:
            # need to add the zero explicitly to the front
            g -= dlogprior(np.concatenate([[0], xi], axis=None))

        self.spline_data["bspline_gkquad"] = gkquad
        self.spline_data["bspline_pE"] = pE
        return g

    def _bspline_calculate_h(self, xi, x_n, w_n):

        """Calculate the Hessian of the maximum likelihood / KL divergence of the FES represented using B-splines.

        Parameters
        ----------

        xi : array of floats size nspline-1
            spline coefficients
        w_n : array of floats, size number of points
            weights for each sample
        x_n : array of floats, size number of points
            values of each sample

        Output
        ------
        Hessian
            nfloat, size (nspline-1) x (nspline - 1)

        Notes
        -----
        CURRENTLY assumes that the gradient has already been called at
        the current value of the parameters.  Otherwise, it fails.  This means it only
        works for certain algorithms.
        """

        mbar = self.mbar
        K = mbar.K
        N_k = mbar.N_k
        N = self.N

        bloc = self._val_to_spline(xi)  # convert the spline coefficients into a spline object
        spline_weights = self.spline_parameters[
            "spline_weights"
        ]  # how to weight the integrated splines in the final likelihood
        nspline = self.spline_parameters["nspline"]  # number of spline points
        kdegree = self.spline_parameters["kdegree"]  # degree of spline
        fkbias = self.spline_parameters["fkbias"]  # K biasing functions
        db_c = self.spline_data[
            "bspline_derivatives"
        ]  # coefficients of the derivatives of the splines
        xrangeij = self.spline_data[
            "xrangeij"
        ]  # range in x and y the 2d integration of basis functions i and j are defined over.
        expf = self.spline_data["bspline_expf"]
        gkquad = self.spline_data["bspline_gkquad"]
        pF = self.spline_data["bspline_pF"]
        pE = self.spline_data["bspline_pE"]

        if spline_weights == "simplesum":
            integral_scaling = N / K * np.ones(K)
        elif spline_weights == "biasedstates":
            integral_scaling = N_k

        # now, compute the Hessian.  First, the first order components
        h = np.zeros([nspline - 1, nspline - 1])

        if spline_weights in ["simplesum", "biasedstates"]:
            for k in range(K):
                h += -integral_scaling[k] * np.outer(gkquad[:, k], gkquad[:, k])
        elif spline_weights == "unbiasedstate":
            h = -N * np.outer(pE, pE)

        if spline_weights in ["simplesum", "biasedstates"]:
            for i in range(nspline - 1):
                for j in range(0, i + 1):
                    if np.abs(i - j) <= kdegree:

                        def ddexpf(x, k_inner):
                            # Disable the PyLint check here because this is the behavior we want
                            # pylint: disable=cell-var-from-loop
                            return db_c[i + 1](x) * db_c[j + 1](x) * expf[k_inner](x)

                        for k in range(K):
                            # now compute the expectation of each derivative
                            pE = integral_scaling[k] * self._integrate(
                                ddexpf,
                                xrangeij[i + 1, j + 1, 0],
                                xrangeij[i + 1, j + 1, 1],
                                args=(k,),
                            )
                            h[i, j] += pE / pF[k]

        elif spline_weights == "unbiasedstate":

            def ddexpf(x, index_i, index_j):
                return db_c[index_i + 1](x) * db_c[index_j + 1](x) * expf(x)

            for i in range(nspline - 1):
                for j in range(0, i + 1):
                    if np.abs(i - j) <= kdegree:

                        # now compute the expectation of each derivative
                        pE = self._integrate(
                            ddexpf,
                            xrangeij[i + 1, j + 1, 0],
                            xrangeij[i + 1, j + 1, 1],
                            args=(i, j),
                        )
                        h[i, j] += N * pE / pF

        for i in range(nspline - 1):
            for j in range(i + 1, nspline - 1):
                h[i, j] = h[j, i]

        ddlogprior = self.spline_parameters["map_data"]["ddlogprior"]
        if ddlogprior != None:  # add hessian of prior
            # need to add the zero explicitly to the front
            h -= ddlogprior(np.concatenate([[0], xi], axis=None))

        return h

    @staticmethod
    def _integrate(func, xlow, xhigh, args=(), method="quad"):  # could add more ability to specify
        """
        wrapper for integration in case we decide to replace quad with something analytical
        """
        if method == "quad":
            # just use scipy quadrature
            results = quad(func, xlow, xhigh, args)[0]
        else:
            raise ParameterError("integration method {:s} not yet implemented".format(method))
        return results

    def _val_to_spline(self, x, form=None):
        """
        Convert a set of B-spline coefficients into a BSpline object

        Parameters
        ----------
        x : float, array
            the last N-1 coefficients for a bspline; we assume the initial coefficient is set to zero.

        Returns
        -------
        bspline : A bspline object (or function returning -log (bspline) object if we need it)

        """

        template_bspline = self.spline_data["bspline"]
        # create new spline with values
        xnew = np.zeros(len(x) + 1)
        xnew[0] = (template_bspline).c[0]
        xnew[1:] = x
        bspline = BSpline((template_bspline).t, xnew, (template_bspline).k)
        if form == "exp":
            return lambda x: -np.log(bspline(x))
        elif form == "log":
            return bspline
        elif form is None:
            return bspline


########
# Integration notes:
#
# If we wanted to integrate the function, can we do it analytically?
# Let's focus on the likelihood integral, which is what is slowing down
# the MC, which is the slow part.
#
# for k=0, then B_i,0:t = 1 if t_i < x < t_i+i, 0 otherwise
# for k=1, It is a piecewise sum of 2 linear terms, so linear.
# f(x) = \\int exp(ax+b)_{t_i}^{t_i+1) = (1/a) e^b (e^a*t2 - e^a t1)
# for k=2, it is piecewise sum of 3 quadratic terms, which is quadradic
# f(x) = \\int exp(-a(x-b)(x-c))_{t_i)+{t_i+1) = (exp^(1/4 a (b - c)^2) Sqrt[\\pi]] (Erf[1/2 Sqrt[a] (b + c - 2 t1)] -
#    Erf[1/2 Sqrt[a] (b + c - 2 t2)]))/(2 Sqrt[a]), for a > 0, switch for a<0.
#
# for k=3, piecewise sum of cubic terms, which appears hard in general.
#
# Of course, even with linear, we need to be able to integrate with the
# bias functions.  If it's a Gaussian bias, then linear and quadratic should integrate fine.
########
