##############################################################################
# pymbar: A Python Library for MBAR (PMF module)
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
A module implementing calculation of potentials of mean force from biased simulations.

"""

import logging
import math
import itertools as it
import numpy as np
import pymbar

from pymbar import mbar_solvers
from pymbar.utils import kln_to_kn, ParameterError, DataError, logsumexp
from pymbar import timeseries

# bunch of imports needed for doing newton optimization of B-splines
from scipy.interpolate import BSpline, make_lsq_spline

# imports needed for scipy minimizations
from scipy.integrate import quad
from scipy.optimize import minimize

from timeit import default_timer as timer  # may remove timing?

logger = logging.getLogger(__name__)
DEFAULT_SOLVER_PROTOCOL = mbar_solvers.DEFAULT_SOLVER_PROTOCOL

# =========================================================================
# PMF class definition
# =========================================================================


class PMF:
    """

    Methods for generating potentials of mean force with statistical uncertainties.

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
    potentials of mean force from umbrella sampling and multistate
    reweighting
    https://arxiv.org/abs/2001.01170

    """

    # =========================================================================

    def __init__(self, u_kn, N_k, verbose=False, mbar_options=None, timings=True, **kwargs):
        """Initialize a potential of mean force calculation by performing
        multistate Bennett acceptance ratio (MBAR) on a set of
        simulation data from umbrella sampling at K states.

        Upon initialization, the dimensionless free energies for all
        states are computed.  This may take anywhere from seconds to
        minutes, depending upon the quantity of data.

        This creates an internal mbar object that is used to create
        the potential of means force.

        Methods are:

           generate_pmf: given an intialized MBAR object, a set of points,
                        the desired energies at that point, and a method, generate
                        an object that contains the PMF information.

           get_pmf: given coordinates, generate the PMF at each coordinate (and uncertainty)

           get_mbar: return the underlying mbar object.

           get_kde: return the underlying kde object.

           sample_parameter_distribution: Only works for pmf_type =
           'spline'. Sample the space of spline parameters according
           to the likelihood function.

           get_confidence_intervals: if sample_parameter_distribution has
           been called, generates confidence intervals for the curves
           given the posterior distribution.

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

        mbar_options: dict, with the following options supported by mbar (see MBAR documentation)

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

        if mbar_options is None:
            pmf_mbar = pymbar.MBAR(u_kn, N_k)
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

            pmf_mbar = pymbar.MBAR(
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

        self.mbar = pmf_mbar

        self._random = np.random
        self._seed = None

        if self.verbose:
            logger.info("PMF initialized")

    @property
    def seed(self):
        return self._seed

    def reset_random(self):
        self._random = np.random
        self._seed = None

    def generate_pmf(
        self,
        u_n,
        x_n,
        pmf_type="histogram",
        histogram_parameters=None,
        kde_parameters=None,
        spline_parameters=None,
        nbootstraps=0,
        seed=-1,
    ):
        """
        Given an intialized MBAR object, a set of points,
        the desired energies at that point, and a method, generate
        an object that contains the PMF information.

        Parameters
        ----------
        pmf_type: str
             options = 'histogram', 'kde', 'spline'

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

        spline_parameters:
            - 'fit_type': which type of fit to use:
                -- 'biasedstates' - sum of log likelihood over all weighted states
                -- 'unbiasedstate' - log likelihood of the single unbiased state
                -- 'simplesum': sum of log likelihoods from the biased simulation. Essentially equivalent to vFEP (York et al.)
            - 'optimization_algorithm':
                -- 'Custom-NR': a custom Newton-Raphson that is particularly fast for close data, but can fail
                -- 'Newton-CG': scipy Newton-CG, only Hessian based method that works correctly because of data ordering.
                -- '         ': scipy gradient based methods that work, but are generally slower (CG, BFGS, L-LBFGS-B, TNC, SLSQP)
            - 'fkbias': array of functions that return the Kth bias potential for each function
            - 'nspline': number of spline points
            - 'kdegree': degree of the spline.  Default is cubic ('3')
            - 'objective': - 'ml','map' # whether to fit the maximum likelihood or the maximum a posteriori

        nbootstraps : int, 0 or > 1, Default: 0
            Number of bootstraps to create an uncertainty estimate. If 0, no bootstrapping is done.

        seed : int, Default: -1
            Set the randomization seed. This does not ensure true randomization, but should get the randomization
            (assuming the same calls are made in the same order) to return the same numbers.
            This is local to this class and will not change any other random objects.

        Returns
        -------
        dict
        float, optional
            if 'timings' is True, returns the time taken to construct the model.

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
        >>> pmf.generate_pmf(u_n, x_n, pmf_type='histogram', histogram_parameters = histogram_parameters)
        >>> results = pmf.get_pmf(x_n)
        >>> f_i = results['f_i']
        >>> for i,x_n in enumerate(x_n):
        >>> print(x_n,f_i[i])
        >>> mbar = pmf.get_mbar()
        >>> print(mbar.f_k)
        >>> print(N_k)

        """

        result_vals = dict()  # for results we may want to return.

        self.pmf_type = pmf_type

        # eventually, we just want the desired energy of each sample.  For now, we allow conversion
        # from older 2d format (K,Nmax instead of N); this is data SAMPLED from
        # each k, not the energy at different K.
        if len(np.shape(u_n)) == 2:
            u_n = pymbar.mbar.kn_to_n(u_n, N_k=self.N_k)

        self.u_n = u_n

        if seed >= 0:
            # Set a better seeded random state
            self._random = np.random.RandomState(seed=seed)
            self._seed = seed

        # we need to save this for calculating uncertainties.
        if not np.issubdtype(type(nbootstraps), np.integer) or nbootstraps == 1:
            raise ValueError(
                f"nbootstraps must be an integer of 0 or >=2, it was set to {nbootstraps}"
            )
        self.nbootstraps = nbootstraps

        if self.timings:
            start = timer()

        self.pmf_function = list()

        # set some variables before bootstrapping loop.

        N_k = self.mbar.N_k
        K = self.mbar.K
        N = np.sum(N_k)

        self.mc_data = None  # we have not sampled MC data yet.

        if self.pmf_type == "histogram":

            self.histogram_datas = list()

            if "bin_edges" not in histogram_parameters:
                raise ParameterError(
                    "histogram_parameters['bin_edges'] cannot be undefined with pmf_type = histogram"
                )

            bins = histogram_parameters["bin_edges"]

            # First, determine the number of dimensions of the histogram. This can be determined
            # by the shape of bin_edges
            dims = len(bins)
            self.dims = dims  # store the dimensionality for checking later.
            self.histogram_parameters = histogram_parameters

        elif pmf_type == "kde":

            self.kdes = list()

            try:
                from sklearn.neighbors import KernelDensity
            except ImportError:
                raise ImportError(
                    "Cannot use 'kde' type PMF without the scikit-learn module. Could not import sklearn"
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
                    raise ParameterError(
                        "Warning: {:s} is not a parameter in KernelDensity".format(k)
                    )
            kde.set_params(**kde_defaults)

        elif pmf_type == "spline":

            # zero this out so we know if we haven't called it yet.
            self.bspline = None
            self.pmf_functions = list()

            self.spline_parameters = spline_parameters  # save these for later references.

            if "objective" not in spline_parameters:
                spline_parameters["objective"] = "ml"  # default

            objective = spline_parameters["objective"]

            if objective not in ["ml", "map"]:
                raise ParameterError(
                    "objective may only be 'ml' or 'map': you have selected {:s}".format(objective)
                )

            if objective == "ml":  # we are doing maximum likelihood minimization
                logprior = None
                dlogprior = None
                ddlogprior = None

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
                    else:
                        logprior = map_data["logprior"]
                    if map_data["dlogprior"] == None:
                        raise ParameterError("d(log prior) must be included if objective is MAP")
                    else:
                        dlogprior = map_data["dlogprior"]
                    if map_data["ddlogprior"] == None:
                        raise ParameterError("d^2(log prior) must be included if objective is MAP")
                    else:
                        ddlogprior = map_data["ddlogprior"]

            spline_weights = spline_parameters["spline_weights"]

            # need the x-range for all methods, since we need to
            xrange = spline_parameters["xrange"]
            # numerically integrate over this range
            nspline = spline_parameters["nspline"]  # number of spline points.
            kdegree = spline_parameters["kdegree"]  # degree of the spline

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
                    scipy_tol = spline_parameters["optimize_options"]["tol"]
                    spline_parameters["optimize_options"].pop("tol", None)
                else:
                    scipy_tol = None  # this is just the default anyway.
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

            if spline_parameters["spline_initialize"] == "bias_free_energies":

                initivals = self.mbar.f_k
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
                            bias_centers[sort_indices], initivals[sort_indices], tinit, k=kdegree
                        )
                        xinit = np.linspace(xrange[0], xrange[1], num=2 * nspline)
                        yinit = binit(xinit)
                    else:
                        xinit = bias_centers[sort_indices]
                        yinit = initivals[sort_indices]
                else:
                    # assume equally spaced bias scenters
                    xinit = np.linspace(xrange[0], xrange[1], self.mbar.K + 1)[1:-1]
                    yinit = initivals

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

            # first, construct a least squares cubic spline in the free energies to start with, set 2nd derivs zero.
            # we assume this is decent.

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
            # one, since the PMF is only determined up to a constant.
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
            self.bspline_derivatives = db_c
            self.bspline = b
            self.fkbias = spline_parameters["fkbias"]

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

        else:
            raise ParameterError("pmf_type {:s} is not defined!".format(pmf_type))

        for b in range(nbootstraps + 1):  # generate bootstrap samples.
            # we bootstrap from each simulation separately.
            if b == 0:  # the default
                bootstrap_indices = np.arange(0, N)
                mbar = self.mbar
                x_nb = x_n
            else:
                index = 0
                for k in range(K):
                    bootstrap_indices[index : index + N_k[k]] = index + self._random.randint(
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
            log_w_n = mbar._computeUnnormalizedLogWeights(self.u_n[bootstrap_indices])
            # calculate a few other things used for multiple methods
            max_log_w_n = np.max(log_w_n)  # we need to solve underflow.
            self.w_n = np.exp(log_w_n - max_log_w_n)
            self.w_n = self.w_n / np.sum(self.w_n)  # nomalize the weights
            # normalized weights for all states.
            self.w_kn = np.exp(mbar.Log_W_nk)

            if self.pmf_type == "histogram":

                # store the data that will be regenerated each time.
                # We will not try to regenerate the bin locations each time,
                # as that would make it hard to calculate uncertainties.
                # We will just recalculate the populations.
                histogram_data = dict()

                histogram_data["bins"] = bins  # save for other functions.
                # Does not actually vary between bootstraps

                # create the bins from the data.
                # it's a 1D array, instead of a Nx1 array.  Reshape.
                if len(np.shape(x_nb)) == 1:
                    x_nb = x_nb.reshape(-1, 1)

                bin_n = np.zeros(x_nb.shape, np.int64)

                for d in range(dims):
                    # bins returns 0 as out of bin.  We want to use -1 as out
                    # of bin
                    bin_n[:, d] = np.digitize(x_nb[:, d], bins[d]) - 1
                histogram_data["bin_n"] = bin_n  # bin counts

                # now we need to loop over the bin_n and identify and label the
                # nonzero bins.

                # a list of the bins with at least one sample in them.
                nonzero_bins = list()
                # for each sample, the index of the nonzero_bins element it
                # belongs to.
                nonzero_bins_index = np.zeros(self.N, dtype=int)
                for n in range(self.N):
                    if np.any(bin_n[n] < 0):
                        nonzero_bins_index[n] = -1
                        continue  # this sample is out of grid
                    if dims == 1:
                        ind2 = bin_n[n]  # which bin sample n is in
                    else:
                        # which bin (labeled N-d) sample n is in
                        ind2 = tuple(bin_n[n])
                    if ind2 not in nonzero_bins:
                        # this bin has a sample.  Add it to the list
                        nonzero_bins.append(ind2)
                    nonzero_bins_index[n] = nonzero_bins.index(
                        ind2
                    )  # the index of the nonzero bins

                histogram_data["nbins"] = (
                    np.int(np.max(nonzero_bins_index)) + 1
                )  # the total number of nonzero bins
                histogram_data["bin_n"] = nonzero_bins_index

                # Compute the free energies for these histogram states with
                # samples
                f_i = np.zeros([histogram_data["nbins"]], np.float64)
                df_i = np.zeros([histogram_data["nbins"]], np.float64)

                for i in range(histogram_data["nbins"]):
                    # Get linear n-indices of samples that fall in this bin.
                    indices = np.where(histogram_data["bin_n"] == i)

                    # Sanity check.
                    if len(indices) == 0:
                        raise DataError(
                            "WARNING: bin %d has no samples -- all bins must have at least one sample."
                            % i
                        )

                    # Compute dimensionless free energy of occupying state i.
                    f_i[i] = -logsumexp(log_w_n[indices])

                # store the free energies for this bin
                histogram_data["f"] = f_i

                # now assign back the free energy from the sample_only bins to
                # all of the bins.

                # rebuild the graph from the edges.
                corner_vectors = list()
                returnsize = list()
                for d in range(dims):
                    maxv = len(bins[d]) - 1
                    corner_vectors.append(np.arange(0, maxv))
                    returnsize.append(maxv)
                # iterator giving all bin locations in N dimensions.
                gridpoints = it.product(*corner_vectors)

                # index in self.f where the free energy for this gridpoint is
                # stored
                fbin_index = np.zeros(np.array(returnsize), int)
                for g in gridpoints:
                    if g in nonzero_bins:
                        fbin_index[g] = nonzero_bins.index(g)
                    else:
                        # no free energy for this index, since there are no
                        # points.
                        fbin_index[g] = -1

                histogram_data["fbin_index"] = fbin_index
                if b == 0:
                    self.histogram_data = histogram_data
                else:
                    self.histogram_datas.append(histogram_data)

            elif pmf_type == "kde":

                # reshape data if needed.
                # it's a 1D array, instead of a Nx1 array.  Reshape.
                if len(np.shape(x_nb)) == 1:
                    x_nb = x_nb.reshape(-1, 1)

                if b > 0:
                    kde = KernelDensity()  # need to create a new one so won't get refit
                    params = self.kde.get_params()
                    kde.set_params(**params)

                kde.fit(x_nb, sample_weight=self.w_n)

                if b == 0:
                    self.kde = kde
                else:
                    self.kdes.append(kde)

            elif pmf_type == "spline":

                w_n = self.w_n
                if b > 0:
                    xi = savexi
                if spline_parameters["optimization_algorithm"] != "Custom-NR":
                    if spline_parameters["optimization_algorithm"] == "Newton-CG":
                        hess = self._bspline_calculate_h
                    else:
                        hess = None
                    results = minimize(
                        self._bspline_calculate_f,
                        xi,
                        args=(
                            w_n,
                            x_nb,
                            nspline,
                            kdegree,
                            spline_weights,
                            xrange,
                            xrangei,
                            xrangeij,
                            logprior,
                            dlogprior,
                            ddlogprior,
                        ),
                        method=spline_parameters["optimization_algorithm"],
                        jac=self._bspline_calculate_g,
                        tol=scipy_tol,
                        hess=hess,
                        options=spline_parameters["optimize_options"],
                    )
                    self.bspline = self._val_to_spline(results["x"], form="log")
                    savexi = results["x"]
                    f = results["fun"]
                else:
                    if "gtol" in spline_parameters["optimize_options"]:
                        tol = spline_parameters["optimize_options"]["gtol"]
                    elif "tol" in spline_parameters["optimize_options"]:
                        tol = spline_parameters["optimize_options"]["tol"]

                    # should come up with better way to make sure it passes the first time.
                    dg = tol * 1e10
                    firsttime = True

                    while dg > tol:  # until we reach the tolerance.

                        f = self._bspline_calculate_f(
                            xi,
                            w_n,
                            x_n,
                            nspline,
                            kdegree,
                            spline_weights,
                            xrange,
                            xrangei,
                            xrangeij,
                            logprior,
                            dlogprior,
                            ddlogprior,
                        )

                        # we need some error handling: if we stepped too far, we should go back
                        # still not great error handling.  Requires something
                        # close.

                        if not firsttime:
                            count = 0
                            # we went too far!  Pull back.
                            while (f >= fold * (1.1) and count < 5) or (np.isinf(f)):
                                f = fold
                                # let's not step as far:
                                dx = 0.9 * dx
                                xi = xold - dx  # step back 90% of dx
                                xold = xi.copy()
                                f = self._bspline_calculate_f(
                                    xi,
                                    w_n,
                                    x_n,
                                    nspline,
                                    kdegree,
                                    spline_weights,
                                    xrange,
                                    xrangei,
                                    xrangeij,
                                    logprior,
                                    dlogprior,
                                    ddlogprior,
                                )
                                count += 1
                        else:
                            firsttime = False
                        fold = f
                        xold = xi.copy()

                        g = self._bspline_calculate_g(
                            xi,
                            w_n,
                            x_n,
                            nspline,
                            kdegree,
                            spline_weights,
                            xrange,
                            xrangei,
                            xrangeij,
                            logprior,
                            dlogprior,
                            ddlogprior,
                        )
                        h = self._bspline_calculate_h(
                            xi,
                            w_n,
                            x_n,
                            nspline,
                            kdegree,
                            spline_weights,
                            xrange,
                            xrangei,
                            xrangeij,
                            logprior,
                            dlogprior,
                            ddlogprior,
                        )

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
                            logger.info(
                                "f = {:.10f}. gradient norm = {:.10f}".format(f, np.sqrt(dg))
                            )
                    self.bspline = self._val_to_spline(xi, form="log")

                if b == 0:
                    self.pmf_function = self.bspline
                    savexi = xi

                    minusloglikelihood = self._bspline_calculate_f(
                        savexi,
                        w_n,
                        x_n,
                        nspline,
                        kdegree,
                        spline_weights,
                        xrange,
                        xrangei,
                        xrangeij,
                        logprior,
                        dlogprior,
                        ddlogprior,
                    )

                    nparameters = len(savexi)

                    # calculate the AIC
                    # formula is: 2(number of parameters) - 2 * loglikelihood

                    self.aic = 2 * nparameters + 2 * minusloglikelihood

                    # calculate the BIC
                    # formula is: ln(number of data points) * (number of parameters) - 2 ln (likelihood at maximum)

                    self.bic = 2 * np.log(self.N) * len(xi) + 2 * minusloglikelihood

                    # potential problems: we don't compute the full log likelihood currently since we
                    # exclude the potential energy part - will have to see what this is in reference to.
                    # this shouldn't be too much of a problem, since we are interested in choosing BETWEEN models.

                else:
                    self.pmf_functions.append(self.bspline)

        # we put the timings outside, since the switch / common stuff is really
        # low.
        if self.timings:
            end = timer()
            result_vals["timing"] = end - start

        return result_vals  # should we returrn results under some other conditions?

    def get_information_criteria(self, type="akaike"):
        """
        returns the Akaike Informatiton Criteria for the model if it exists.

        Parameters
        ----------

        type: string, Either 'Akaike' or 'Bayesian'

        Output
        ------

        information criteria
        """

        if self.pmf_type != "spline":
            raise ParameterError(
                "Information criteria currently only defined for spline approaches, you are currently using {:s}".format(
                    type
                )
            )
        if type in ["akaike", "Akaike", "AIC", "aic"]:
            return self.aic
        elif type in ["bayesian", "Bayesian", "BIC", "bic"]:
            return self.bic
        else:
            raise ParameterError("Information criteria of type '{:s}' not defined".format(type))

    def get_pmf(self, x, uncertainties="from-lowest", pmf_reference=None):
        """
        Returns values of the PMF at the specified x points.

        Parameters
        ----------

        x: numpy.ndarray of D dimensions, where D is the dimensionality of the PMF defined.

        uncertainties : str, optional
            Method for reporting uncertainties (default: 'from-lowest')

            * 'from-lowest' - the uncertainties in the free energy difference with lowest point on PMF are reported
            * 'from-specified' - same as from lowest, but from a user specified point
            * 'from-normalization' - the normalization \\sum_i p_i = 1 is used to determine uncertainties spread out through the PMF
            * 'all-differences' - the nbins x nbins matrix df_ij of uncertainties in free energy differences is returned instead of df_i

        pmf_reference :
            an N-d point specifying the reference state. Ignored except with uncertainty method ``from_specified``

        Returns
        -------
        dict
            'f_i' : np.ndarray, float, shape=(K)
                result_vals['f_i'][i] is the dimensionless free energy of the x_i point, relative to the reference point
            'df_i' : np.ndarray, float, shape=(K)
                result_vals['df_i'][i] is the uncertainty in the difference of x_i with respect to the reference point

        """

        if len(np.shape(x)) <= 1:  # if it's zero, it's a scalar.
            coorddim = 1
        else:
            coorddim = np.shape(x)[1]

        if self.pmf_type == "histogram":
            if self.dims != coorddim:
                # later, need to put coordinate check on other methods.
                raise DataError("coordinates have inconsistent dimension with the PMF.")

        if uncertainties == "from-specified" and pmf_reference is None:
            raise ParameterError(
                "No reference state specified for PMF using uncertainties = from-specified"
            )

        if self.pmf_type is None:
            raise ParameterError("pmf_type has not been set!")

        K = self.mbar.K  # number of states

        # create dictionary to return results
        result_vals = dict()

        if self.pmf_type == "histogram":

            # figure out which bins the values are in.
            nbins = self.histogram_data["nbins"]
            bins = self.histogram_data["bins"]
            dims = len(bins)

            if dims == 1:
                # what gridpoint does each x fall into?
                # -1 and nbinsperdim are out of range
                loc_indices = np.digitize(x, bins[0]) - 1
            else:
                loc_indices = np.zeros([len(x), dims], dtype=int)
                for d in range(dims):
                    # -1 and nbinsperdim are out of range
                    loc_indices[:, d] = np.digitize(x[:, d], bins[d]) - 1

            # figure out which grid point the pmf_reference is at
            if pmf_reference is not None:
                if dims == 1:
                    # make it a list for reduced code duplication.
                    pmf_reference = [pmf_reference]
                pmf_ref_grid = np.zeros([dims], dtype=int)
                for d in range(dims):
                    # -1 and nbins_per_dim are out of range
                    pmf_ref_grid[d] = np.digitize(pmf_reference[d], bins[d]) - 1
                    if pmf_ref_grid[d] == -1 or pmf_ref_grid[d] == len(bins[d]):
                        raise ParameterError(
                            "Specified reference point coordinate {:f} in dim {:d} grid point is out of the defined free energy region [{:f},{:f}]".format(
                                pmf_ref_grid[d], d, np.min(bins[d]), np.max(bins[d])
                            )
                        )

            if (
                uncertainties == "from-lowest"
                or uncertainties == "from-specified"
                or uncertainties == "all-differences"
            ):
                # Report uncertainties in free energy difference from a given
                # point on PMF.

                df_i = np.zeros(len(self.histogram_data["f"]), np.float64)

                if uncertainties == "from-lowest":
                    # Determine bin index with lowest free energy.
                    j = self.histogram_data["f"].argmin()
                elif uncertainties == "from-specified":
                    j = self.histogram_data["fbin_index"][tuple(pmf_ref_grid)]
                elif uncertainties == "all-differences":
                    raise ParameterError(
                        "Uncertainty method of 'all-differences' is not yet supported for histogram "
                        "PMF types (not implemented)"
                    )

                if self.nbootstraps == 0:
                    # Compute uncertainties in free energy at each gridpoint by
                    # forming matrix of W_nk.
                    N_k = np.zeros([self.K + nbins], np.int64)
                    N_k[0:K] = self.N_k
                    W_nk = np.zeros([self.N, self.K + nbins], np.float64)
                    W_nk[:, 0:K] = np.exp(self.mbar.Log_W_nk)

                    log_w_n = self.mbar._computeUnnormalizedLogWeights(self.u_n)
                    for i in range(nbins):  # loop over the nonzero bins, internal numbering
                        # Get indices of samples that fall in this bin.
                        indices = np.where(self.histogram_data["bin_n"] == i)

                        # Compute normalized weights for this state.
                        W_nk[indices, K + i] = np.exp(
                            log_w_n[indices] + self.histogram_data["f"][i]
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

                else:
                    fall = np.zeros([len(self.histogram_data["f"]), self.nbootstraps])
                    for b in range(self.nbootstraps):
                        fall[:, b] = self.histogram_datas[b]["f"] - self.histogram_datas[b]["f"][j]

                        df_i = np.std(fall, axis=1)
                    # Shift free energies so that state j has zero free energy.
                f_i = self.histogram_data["f"] - self.histogram_data["f"][j]

            elif uncertainties == "from-normalization":
                # Determine uncertainties from normalization that \sum_i p_i = 1.
                # need to reimplement this . . . maybe.
                raise ParameterError(
                    "uncertainty method 'from-normalization' is not currently supported for histograms"
                )

                # Currently unreachable code

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

                if dims == 1:
                    findex = self.histogram_data["fbin_index"][l]
                else:
                    findex = self.histogram_data["fbin_index"][tuple(l)]
                if findex >= 0:
                    fx_vals[i] = f_i[findex]
                    dfx_vals[i] = df_i[findex]
                else:
                    fx_vals[i] = np.nan
                    dfx_vals[i] = np.nan

                # Return dimensionless free energy and uncertainty.
                result_vals["f_i"] = fx_vals
                result_vals["df_i"] = dfx_vals

            if uncertainties == "all-differences":
                if self.nbootstraps == 0:
                    # Report uncertainties in all free energy differences as
                    # well.
                    diag = Theta_ij.diagonal()
                    dii = diag[K, K + nbins]  # appears broken?  Not used?
                    d2f_ij = dii + dii.transpose() - 2 * Theta_ij[K : K + nbins, K : K + nbins]

                    # unsquare uncertainties
                    df_ij = np.sqrt(d2f_ij)

                    dfxij_vals = np.zeros([len(x), len(x)])

                    findexs = list()
                    for i, l in enumerate(loc_indices):
                        if dims == 1:
                            findex = self.histogram_data["fbin_index"][l]
                        else:
                            findex = self.histogram_data["fbin_index"][tuple(l)]
                        findexs.append(findex)

                    for i, vi in enumerate(findexs):
                        for j, vj in enumerate(findexs):
                            if vi != -1 and vj != 1:
                                dfxij_vals[i, j] = df_ij[vi, vj]
                            else:
                                dfxij_vals[i, j] = np.nan
                else:
                    dfxij_vals = np.zeros(
                        [len(self.histogram_data["f"]), len(self.histogram_data["f"])]
                    )
                    fall = np.zeros(
                        [
                            len(self.histogram_data["f"]),
                            len(self.histogram_data["f"]),
                            self.nbootstraps,
                        ]
                    )
                    for b in range(self.nbootstraps):
                        fall[:, b] = (
                            self.histogram_datas[b]["f"] - self.histogram_data[b]["f"].transpose()
                        )
                    dfxij_vals = np.std(fall, axis=2)

                # Return dimensionless free energy and uncertainty.
                result_vals["df_ij"] = dfxij_vals

        elif self.pmf_type == "kde":

            # if it's not an array, make it one.
            x = np.array(x)

            # it's a 1D array, instead of a Nx1 array.  Reshape.
            if len(np.shape(x)) <= 1:
                x = x.reshape(-1, 1)
            f_i = -self.kde.score_samples(x)  # gives the LOG density, which is what we want.

            if uncertainties == "from-lowest":
                fmin = np.min(f_i)
                f_i = f_i - fmin

            elif uncertainties == "from-specified":
                fmin = -self.kde.score_samples(np.array(pmf_reference).reshape(1, -1))
                f_i = f_i - fmin
            else:
                raise ParameterError(f"Uncertainty method {uncertainties} for kde is unavailable")

            if self.nbootstraps == 0:
                df_i = None
            else:
                fall = np.zeros([len(x), self.nbootstraps])
                for b in range(self.nbootstraps):
                    fall[:, b] = -self.kdes[b].score_samples(x) - fmin
                df_i = np.std(fall, axis=1)

            # uncertainites "from normalization" reference is applied, since
            # the density is normalized.
            result_vals["f_i"] = f_i
            result_vals["df_i"] = df_i

        elif self.pmf_type == "spline":

            f_i = self.pmf_function(x)

            if uncertainties == "from-lowest":
                fmin = np.min(f_i)
                f_i = f_i - fmin

            elif uncertainties == "from-specified":
                fmin = -self.pmf_function(np.array(pmf_reference).reshape(1, -1))
                f_i = f_i - fmin
            if self.nbootstraps == 0:
                df_i = None
            else:
                dim_breakdown = [d for d in x.shape] + [self.nbootstraps]
                fall = np.zeros(dim_breakdown)
                for b in range(self.nbootstraps):
                    fall[:, :, b] = self.pmf_functions[b](x) - fmin
                df_i = np.std(fall, axis=-1)

            # uncertainites "from normalization" reference is applied, since
            # the density is normalized.
            result_vals["f_i"] = f_i
            result_vals["df_i"] = df_i
            # no error method yet. Maybe write a bootstrap class?

        return result_vals

    def get_mbar(self):
        """return the MBAR object being used by the PMF


        Returns
        -------
           MBAR object
        """
        if self.mbar is not None:
            return self.mbar
        else:
            raise DataError("MBAR in the PMF object is not initialized, cannot return it.")

    def get_kde(self):
        """ return the KernelDensity object if it exists.

        Returns
        -------
        sklearn KernelDensity object
        """

        if self.pmf_type == "kde":
            if self.kde != None:
                return self.kde
            else:
                raise ParameterError(
                    "Can't return the KernelDensity object because kde not yet defined"
                )
        else:
            raise ParameterError("Can't return the KernelDensity object because pmf_type != kde")

    def sample_parameter_distribution(
        self, x_n, mc_parameters=None, decorrelate=True, verbose=True,
    ):

        # determine the range of the bspline at the start of the
        # process: changes are made as fractions of this

        spline_parameters = self.spline_parameters

        pmf_type = self.pmf_type

        if pmf_type != "spline":
            ParameterError("Keyword 'pmf_type' must be spline")

        K = self.mbar.K
        spline_weights = spline_parameters["spline_weights"]

        if spline_parameters is None:
            ParameterError("Must specify spline_parameters to sample the distributions")

        spline_weights = spline_parameters["spline_weights"]

        w_n = self.w_n

        # need the x-range for all methods, since we need to
        xrange = spline_parameters["xrange"]
        # numerically integrate over this range

        if pmf_type != "spline":
            ParameterError("Sampling of posterior is only supported for spline type")

        if self.bspline is None:
            ParameterError(
                "Need to generate a splined PMF using GeneratePMF before performing MCMC sampling"
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

        # ensure normalization of spline
        def prob(x):
            return np.exp(-self.bspline(x))

        norm = self._integrate(spline_parameters["spline_weights"], prob, xrange[0], xrange[1])
        self.bspline.c = self.bspline.c + np.log(norm)

        self.mc_data = dict()
        # make a copy of the original spline to preserve it.
        self.mc_data["original_spline"] = BSpline(self.bspline.t, self.bspline.c, self.bspline.k)

        # this might not work as well for probability
        c = self.bspline.c
        crange = np.max(c) - np.min(c)
        dc = fraction_change * crange

        self.naccept = 0
        csamples = np.zeros([len(c), int(niterations) // int(sample_every)])
        logposteriors = np.zeros(int(niterations) // int(sample_every))
        self.first_step = True

        w_n = self.w_n
        for n in range(niterations):
            results = self._MCStep(x_n, w_n, dc, xrange, spline_weights, logprior)
            if n % sample_every == 0:
                csamples[:, n // sample_every] = results["c"]
                logposteriors[n // sample_every] = results["logposterior"]
            if n % print_every == 0 and verbose:
                print(
                    "MC Step {:d} of {:d}".format(n, niterations),
                    str(results["logposterior"]),
                    str(self.bspline.c),
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
                logger.info("Time series for spline parameters are: {:s}".format(str(g_c)))
            maxgc = np.max(g_c)
            meangc = np.mean(g_c)
            guse = g_mc  # doesn't affect the distribution that much
            indices = timeseries.subsample_correlated_data(equil_logp, g=guse)
            logposteriors = equil_logp[indices]
            csamples = (csamples[:, t_mc:])[:, indices]
            if verbose:
                logger.info("samples after decorrelation: {:d}".format(np.shape(csamples)[1]))

        self.mc_data["samples"] = csamples
        self.mc_data["logposteriors"] = logposteriors
        self.mc_data["mc_parameters"] = mc_parameters
        self.mc_data["acceptance ratio"] = self.naccept / niterations
        if verbose:
            logger.info("Acceptance rate: {:5.3f}".format(self.mc_data["acceptance ratio"]))
        self.mc_data["nequil"] = t_mc  # the start of the "equilibrated" data set
        self.mc_data["g_logposterior"] = g_mc  # statistical efficiency of the log posterior
        self.mc_data["g_parameters"] = g_c  # statistical efficiency of the parametere
        self.mc_data["g"] = guse  # statistical efficiency used for subsampling

    def get_confidence_intervals(self, xplot, plow, phigh, reference="zero"):
        """
        Parameters
        ----------
        xplot :
            data points we want to plot at
        plow :
            lowest percentile
        phigh :
            highest percentile

        Returns
        -------
        TODO:
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

        """ convenience function to get MC data

        Returns
        -------
        dict
            samples: samples of the parameters with size [# parameters x # points]
            logposteriors: log posteriors (which might be defined with respect to some reference) as a time series size [# points]
            mc_parameters: dictionary of parameters that were run with
            acceptanceatio: acceptance ratio overall of the MC chain
            nequil: the start of the "equilibrated" data set (i.e. nequil-1 is the number that werer thrown out)
            g_logposterior: statistical efficiency of the log posterior
            g_parameters: statistical efficiency of the parametere
            g: statistical efficiency used for subsampling

        """

        if self.mc_data is None:
            raise DataError("No MC sampling has been done, cannot construct confidence intervals")
        else:
            return self.mc_data

    def _get_MC_loglikelihood(self, x_n, w_n, spline_weights, spline, xrange):

        N = self.N
        K = self.K

        if spline_weights in ["simplesum", "biasedstates"]:
            loglikelihood = 0

            for k in range(self.K):
                x_kn = x_n[self.mbar.x_kindices == k]

                def splinek(x, kf=k):
                    return spline(x) + self.fkbias[kf](x)

                def expk(x, kf=k):
                    return np.exp(-splinek(x, kf))

                normalize = np.log(
                    self._integrate(spline_weights, expk, xrange[0], xrange[1], args=(k))
                )
                if spline_weights == "simplesum":
                    loglikelihood += (N / K) * np.mean(splinek(x_kn))
                    loglikelihood += (N / K) * normalize
                elif spline_weights == "biasedstates":
                    loglikelihood += np.sum(splinek(x_kn))
                    loglikelihood += self.N_k[k] * normalize

        elif spline_weights == "unbiasedstate":
            loglikelihood = N * np.dot(w_n, spline(x_n))
            # no need to add normalization, should be normalized.

        return loglikelihood

    def _MCStep(self, x_n, w_n, stepsize, xrange, spline_weights, logprior):

        """ sample over the posterior space of the PMF as splined.

        Parameters
        ----------
        x_n :
            samples from the biased distribution
        w_n :
            weights of each sample.
        stepsize :
            sigma of the normal distribution used to propose steps
        xrange :
            Range the probility distribution is defined o er.
        spline_weights :
            Type of weighting used for maximum likelihood for splines.  See class
                        definition for description of types.
        logprior :
            function describing the prior of the parameters. Default is uniform.

        Outputs
        -------
        dict
            * 'c': the value of the spline constants (len nsplines - we always assume normalized
            * 'logposterior': the current value of the logoposterior.

        Notes
        -----
        Modifies several saved variables saved in the structure.x

        """

        if self.first_step:
            c = self.bspline.c
            self.previous_logposterior = self._get_MC_loglikelihood(
                x_n, w_n, spline_weights, self.bspline, xrange
            ) - logprior(c)
            cold = self.bspline.c
            self.first_step = True
            # create an extra one we can carry around
            self.newspline = BSpline(self.bspline.t, self.bspline.c, self.bspline.k)

        self.cold = self.bspline.c
        psize = len(self.cold)
        rchange = stepsize * self._random.normal()
        cnew = self.cold.copy()
        ci = self._random.randint(psize)
        cnew[ci] += rchange
        self.newspline.c = cnew

        # determine the change in the integral
        def prob(x):
            return np.exp(-self.newspline(x))

        new_integral = self._integrate(spline_weights, prob, xrange[0], xrange[1])

        cnew = cnew + np.log(new_integral)

        self.newspline.c = cnew  # this spline should now be normalized.

        # now calculate the change in log likelihood
        loglikelihood = self._get_MC_loglikelihood(
            x_n, w_n, spline_weights, self.newspline, xrange
        )

        newlogposterior = loglikelihood - logprior(cnew)
        dlogposterior = newlogposterior - (self.previous_logposterior)
        accept = False
        if dlogposterior <= 0:
            accept = True
        if dlogposterior > 0:
            if self._random.random() < np.exp(-dlogposterior):
                accept = True

        if accept:
            self.bspline.c = self.newspline.c
            self.cold = self.bspline.c
            self.previous_logposterior = newlogposterior
            self.naccept = self.naccept + 1
        results = dict()
        results["c"] = self.bspline.c
        results["logposterior"] = self.previous_logposterior
        return results

    def _bspline_calculate_f(
        self,
        xi,
        w_n,
        x_n,
        nspline,
        kdegree,
        spline_weights,
        xrange,
        xrangei,
        xrangeij,
        logprior,
        dlogprior,
        ddlogprior,
    ):

        """ Calculate the maximum likelihood / KL divergence of the PMF represented using B-splines.

        Parameters
        ----------

        xi : array of floats size nspline-1
            spline coefficients,
        w_n :
            weights for each sample.
        x_n :
            values of each sample.
        nspline :
            number of spline points
        kdegree :
            degree of spline
        spline_weights :
            type of spline weighting (i.e. choice of maximum likelihood)
        xrange :
            range the PMF is defined over
        xrangei :
            range the ith basis function of the spline is defined over
        xrangeij :
            range in x and y the 2d integration of basis functions i and j are defined over.
        logprior :
            log of the prior for MAP
        dlogprior :
            d(log prior)/xi for MAP.  Not needed here, but included for consistent arguments
        ddlogprior :
            d^2(log prior)/xi for MAP.  Not needed here, but included for consistent arguments

        Output
        ------
        float
            function value
        """

        K = self.mbar.K
        N_k = self.mbar.N_k
        N = self.N

        bloc = self._val_to_spline(xi)

        if spline_weights in ["simplesum", "biasedstates"]:
            pF = np.zeros(K)
            if spline_weights == "simplesum":
                f = 0
                for k in range(K):
                    f += (N / K) * np.mean(bloc(x_n[self.mbar.x_kindices == k]))
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
                    return np.exp(-bloc(x) - self.fkbias[kf](x))

                # compute the partition function
                pF[k] = self._integrate(spline_weights, expfk, xrange[0], xrange[1], args=(k))
                expf.append(expfk)
            # subtract the free energy (add log partition function)
            f += np.dot(integral_scaling, np.log(pF))

        elif spline_weights == "unbiasedstate":  # just KL divergence of the unbiased potential
            f = N * np.dot(w_n, bloc(x_n))

            def expf(x):
                return np.exp(-bloc(x))

            # setting limit to try to eliminate errors: hard time because it
            # goes so small.
            pF = self._integrate(spline_weights, expf, xrange[0], xrange[1])
            # subtract the free energy (add log partition function)
            f += N * np.log(pF)

        self.bspline_expf = expf
        self.bspline_pF = pF

        # need to add the zero explicitly to the front
        if logprior != None:
            f -= logprior(np.concatenate([[0], xi], axis=None))

        return f

    def _bspline_calculate_g(
        self,
        xi,
        w_n,
        x_n,
        nspline,
        kdegree,
        spline_weights,
        xrange,
        xrangei,
        xrangeij,
        logprior,
        dlogprior,
        ddlogprior,
    ):
        """Calculate the gradient of the maximum likelihood / KL divergence of the PMF represented using B-splines.

        Parameters
        -----------

        xi: spline coefficients, array of floats size nspline-1
        w_n: weights for each sample.
        x_n: values of each sample.
        nspline: number of spline points
        kdegree: degree of spline
        spline_weights: type of spline weighting (i.e. choice of maximum likelihood)
        dlogprior: derivative of logprior with respect to the parameters, for use with MAP estimates
        xrange: range the PMF is defined over
        xrangei: range the ith basis function of the spline is defined over
        xrangeij: range in x and y the 2d integration of basis functions i and j are defined over.

        xrangeij is not used, but used to keep consistent call arguments among f,g,h calls.
        logprior: log of the prior for MAP   Not needed here, but included for consistent arguments
        dlogprior: d(log prior)/xi for MAP.
        ddlogprior: d^2(log prior)/xi for MAP.  Not needed here, but included for consistent arguments

        Output
        ------

        gradient: float, size (nspline-1)

        """
        ##### COMPUTE THE GRADIENT #######
        # The gradient of the function is \sum_n [\sum_k W_k(x_n)] dF(phi(x_n))/dtheta_i - \sum_k <dF/dtheta>_k
        #
        # where <O>_k = \int O(xi) exp(-F(xi) - u_k(xi)) dxi / \int exp(-F(xi)
        # - u_k(xi)) dxi

        K = self.mbar.K
        N_k = self.mbar.N_k
        N = np.sum(N_k)

        db_c = self.bspline_derivatives
        bloc = self._val_to_spline(xi)
        pF = np.zeros(K)

        if spline_weights == "simplesum":
            integral_scaling = (N / K) * np.ones(K)
        elif spline_weights == "biasedstates":
            integral_scaling = N_k

        g = np.zeros(nspline - 1)

        for i in range(1, nspline):
            if spline_weights == "simplesum":
                for k in range(K):
                    g[i - 1] += (N / K) * np.mean(db_c[i](x_n[self.mbar.x_kindices == k]))
            elif spline_weights == "biasedstates":
                g[i - 1] = np.sum(db_c[i](x_n))
            elif spline_weights == "unbiasedstate":
                g[i - 1] = N * np.dot(w_n, db_c[i](x_n))

        # now the second part of the gradient.

        if spline_weights in ["biasedstates", "simplesum"]:
            gkquad = np.zeros([nspline - 1, K])

            def expf(x, k):
                return np.exp(-bloc(x) - self.fkbias[k](x))

            def dexpf(x, k):
                return db_c[i + 1](x) * expf(x, k)

            for k in range(K):
                # putting this in rather than saving the term so gradient and f
                # can be called independently
                pF[k] = self._integrate(spline_weights, expf, xrange[0], xrange[1], args=(k))

                for i in range(nspline - 1):
                    # Boltzmann weighted derivative with each biasing function
                    # now compute the expectation of each derivative
                    pE = self._integrate(
                        spline_weights, dexpf, xrangei[i + 1, 0], xrangei[i + 1, 1], args=(k)
                    )

                    # normalize the expectation
                    gkquad[i, k] = pE / pF[k]
            g -= np.dot(gkquad, integral_scaling)

        elif spline_weights == "unbiasedstate":
            gkquad = 0  # not used here, but saved for Hessian calls.

            def expf(x):
                return np.exp(-bloc(x))

            # 0 is the value of gkquad. Recomputed here to avoid problems
            pF = self._integrate(spline_weights, expf, xrange[0], xrange[1])
            # with other scipy solvers
            pE = np.zeros(nspline - 1)

            for i in range(nspline - 1):
                # Boltzmann weighted derivative
                def dexpf(x):
                    return db_c[i + 1](x) * expf(x)

                # now compute the expectation of each derivative
                pE[i] = self._integrate(
                    spline_weights, dexpf, xrangei[i + 1, 0], xrangei[i + 1, 1]
                )
                # normalize the expectation.
                pE[i] /= pF
            g -= N * pE

        # need to add the zero explicitly to the front
        if dlogprior != None:
            g -= dlogprior(np.concatenate([[0], xi], axis=None))

        self.bspline_gkquad = gkquad
        self.bspline_pE = pE
        return g

    def _bspline_calculate_h(
        self,
        xi,
        w_n,
        x_n,
        nspline,
        kdegree,
        spline_weights,
        xrange,
        xrangei,
        xrangeij,
        logprior,
        dlogprior,
        ddlogprior,
    ):

        """ Calculate the Hessian of the maximum likelihood / KL divergence of the PMF represented using B-splines.

        Parameters
        ----------

        xi: array of floats size nspline-1
            spline coefficients
        w_n:
            weights for each sample
        x_n:
            values of each sample
        nspline :
            number of spline points
        kdegree :
            degree of spline
        spline_weights :
            type of spline weighting (i.e. choice of maximum likelihood)
        xrange :
            range the PMF is defined over
        xrangei :
            range the ith basis function of the spline is defined over
        xrangeij :
            range in x and y the 2d integration of basis functions i and j are defined over.
        logprior :
            log of the prior for MAP  Not needed here, but included for consistent arguments
        dlogprior :
            d(log prior)/xi for MAP.  Not needed here, but included for consistent arguments
        ddlogprior :
            d^2(log prior)/xi for MAP.

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
        K = self.mbar.K
        N_k = self.mbar.N_k
        N = np.sum(N_k)

        expf = self.bspline_expf
        gkquad = self.bspline_gkquad
        pF = self.bspline_pF
        pE = self.bspline_pE
        db_c = self.bspline_derivatives

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

                        def ddexpf(x, k):
                            return db_c[i + 1](x) * db_c[j + 1](x) * expf[k](x)

                        for k in range(K):
                            # now compute the expectation of each derivative
                            pE = integral_scaling[k] * self._integrate(
                                spline_weights,
                                ddexpf,
                                xrangeij[i + 1, j + 1, 0],
                                xrangeij[i + 1, j + 1, 1],
                                args=(k),
                            )
                            h[i, j] += pE / pF[k]

        elif spline_weights == "unbiasedstate":
            for i in range(nspline - 1):
                for j in range(0, i + 1):
                    if np.abs(i - j) <= kdegree:

                        def ddexpf(x):
                            return db_c[i + 1](x) * db_c[j + 1](x) * expf(x)

                        # now compute the expectation of each derivative
                        pE = self._integrate(
                            spline_weights,
                            ddexpf,
                            xrangeij[i + 1, j + 1, 0],
                            xrangeij[i + 1, j + 1, 1],
                        )
                        h[i, j] += N * pE / pF

        for i in range(nspline - 1):
            for j in range(i + 1, nspline - 1):
                h[i, j] = h[j, i]

        # need to add the zero explicitly to the front
        if ddlogprior != None:  # add hessian of prior
            h -= ddlogprior(np.concatenate([[0], xi], axis=None))

        return h

    def _integrate(self, spline_parameters, func, xlow, xhigh, args=(), method="quad"):
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
        x:
            the last N-1 coefficients for a bspline; we assume the initial coefficient is set to zero.

        Returns
        -------
        A bspline object (or function returning -log (bspline) object if we need it)

        """

        # create new spline with values
        xnew = np.zeros(len(x) + 1)
        xnew[0] = (self.bspline).c[0]
        xnew[1:] = x
        bspline = BSpline((self.bspline).t, xnew, (self.bspline).k)
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
