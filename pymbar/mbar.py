##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2017-2022 University of Colorado Boulder
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
###########
pymbar.MBAR
###########

A module implementing the multistate Bennett acceptance ratio (MBAR) method for the analysis
of equilibrium samples from multiple arbitrary thermodynamic states in computing equilibrium
expectations, free energy differences, potentials of mean force, and entropy and enthalpy contributions.

Please reference the following if you use this code in your research:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177

"""

from textwrap import dedent
import logging
import numpy as np
import numpy.linalg as linalg
from pymbar import mbar_solvers
from pymbar.utils import (
    kln_to_kn,
    kn_to_n,
    ParameterError,
    DataError,
    logsumexp,
    check_w_normalized,
)

logger = logging.getLogger(__name__)
DEFAULT_SOLVER_PROTOCOL = mbar_solvers.DEFAULT_SOLVER_PROTOCOL
ROBUST_SOLVER_PROTOCOL = mbar_solvers.ROBUST_SOLVER_PROTOCOL
JAX_SOLVER_PROTOCOL = mbar_solvers.JAX_SOLVER_PROTOCOL
BOOTSTRAP_SOLVER_PROTOCOL = mbar_solvers.BOOTSTRAP_SOLVER_PROTOCOL

# =========================================================================
# MBAR class definition
# =========================================================================
class MBAR:
    """

    Multistate Bennett acceptance ratio method (MBAR) for the analysis of multiple equilibrium samples.

    Notes
    -----
    Note that this method assumes the data are uncorrelated.

    Correlated data must be subsampled to extract uncorrelated (effectively independent) samples.

    References
    ----------

    [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
    J. Chem. Phys. 129:124105, 2008
    http://dx.doi.org/10.1063/1.2978177
    """

    # =========================================================================

    def __init__(
        self,
        u_kn,
        N_k,
        maximum_iterations=10000,
        relative_tolerance=1.0e-7,
        verbose=False,
        initial_f_k=None,
        solver_protocol=None,
        initialize="zeros",
        x_kindices=None,
        n_bootstraps=0,
        bootstrap_solver_protocol=None,
        rseed=None,
    ):
        """Initialize multistate Bennett acceptance ratio (MBAR) on a set of simulation data.

        Upon initialization, the dimensionless free energies for all states are computed.
        This may take anywhere from seconds to minutes, depending upon the quantity of data.
        After initialization, the computed free energies may be obtained by a call to :func:`compute_free_energy_differences`,
        or expectation at any state of interest can be computed by calls to :func:`compute_expectations`.

        Parameters
        ----------
        u_kn : np.ndarray, float, shape=(K, N_max)
            ``u_kn[k,n]`` is the reduced potential energy of uncorrelated
            configuration n evaluated at state ``k``.
        u_kln : np.ndarray, float, shape (K, L, N_max)
            If the simulation is in form ``u_kln[k,l,n]`` it is converted to ``u_kn`` format

            .. code-block:: python

                u_kn = [ u_1(x_1) u_1(x_2) u_1(x_3) . . . u_1(x_n)
                         u_2(x_1) u_2(x_2) u_2(x_3) . . . u_2(x_n)
                                                    . . .
                         u_k(x_1) u_k(x_2) u_k(x_3) . . . u_k(x_n)]

        N_k :  np.ndarray, int, shape=(K)
            ``N_k[k]`` is the number of uncorrelated snapshots sampled from state ``k``.
            Some may be zero, indicating that there are no samples from that state.

            We assume that the states are ordered such that the first ``N_k``
            are from the first state, the 2nd ``N_k`` the second state, and so
            forth. This only becomes important for bar -- MBAR does not
            care which samples are from which state.  We should eventually
            allow this assumption to be overwritten by parameters passed
            from above, once ``u_kln`` is phased out.

        maximum_iterations : int, optional
            Set to limit the maximum number of iterations performed (default 1000)

        relative_tolerance : float, optional
            Set to determine the relative tolerance convergence criteria (default 1.0e-6)

        verbosity : bool, optional
            Set to True if verbose debug output is desired (default False)

        initial_f_k : np.ndarray, float, shape=(K), optional
            Set to the initial dimensionless free energies to use as a
            guess (default None, which sets all f_k = 0)

        solver_protocol : list(dict), string or None, optional, default=None
            List of dictionaries to define a sequence of solver algorithms
            and options used to estimate the dimensionless free energies.
            See `pymbar.mbar_solvers.solve_mbar()` for details.  If None,
            use the developers best guess at an appropriate algorithm.

            if the string is "robust", it tries "L-BFGS-B" and "adaptive",
            with relatively large numbers of iterations.

            if "default" or "none", it will use 'hybr' root solver, followed with
            some rounds of Newton-Raphson if it fails to converge.

            The default will try to solve with an adaptive solver algorithm
            which alternates between self-consistent iteration and
            Newton-Raphson, where the method with the smallest
            gradient is chosen to improve numerical stability.

        initialize : 'zeros' or 'BAR', optional, Default: 'zeros'
            If equal to 'BAR', use bar between the pairwise state to
            initialize the free energies.  Eventually, should specify a path;
            for now, it just does it zipping up the states.

            The 'BAR' option works best when the states are ordered such that adjacent states
            maximize the overlap between states. It is up to the user
            to arrange the states in such an order, or at least close to such an order.
            If you are uncertain what the order of states should be, or if it does not make
            sense to think of states as adjacent, then choose 'zeros'.

        x_kindices : np.ndarray, int, shape=(K), default = [0,1,2,3,4...K]
            Describes which state is each sample *x* is from?  Usually doesn't matter,
            but it does for bar. As a default, we assume the samples are in ``K`` order
            (the first ``N_k[0]`` samples are from the 0th state, the next ``N_k[1]``
            samples from the 1st state, and so forth.

        n_bootstraps: int
            How many bootstrap free energies will be computed? If None, no bootstraps will be computed.
            computing uncertainties with bootstraps is only possible if this is > 0.
            (default: None)

        bootstrap_solver_protocol: list(dict), string or None, optional, default=None
            We usually just do steps of adaptive sampling without. "robust" would be the backup.
            Default: dict(method="adaptive", options=dict(min_sc_iter=0)),

        Notes
        -----
        The reduced potential energy ``u_kn[k,n] = u_k(x_{ln})``, where the reduced potential energy ``u_l(x)`` is
        defined (as in the text) by:
        ``u_k(x) = beta_k [ U_k(x) + p_k V(x) + mu_k' n(x) ]``
        where

        ``beta_k = 1/(k_B T_k)`` is the inverse temperature of condition ``k``, where k_B is Boltzmann's constant

        ``U_k(x)`` is the potential energy function for state ``k``

        ``p_k`` is the pressure at state ``k`` (if an isobaric ensemble is specified)

        ``V(x)`` is the volume of configuration ``x``

        ``mu_k`` is the M-vector of chemical potentials for the various species, if a (semi)grand ensemble is
        specified, and ``'`` denotes transpose

        ``n(x)`` is the M-vector of numbers of the various molecular species for configuration ``x``,
        corresponding to the chemical potential components of ``mu_m``.

        ``x_n`` indicates that the samples are from ``k`` different simulations of the ``n`` states. These simulations
        need only be a subset of the k states.

        The configurations ``x_ln`` must be uncorrelated.  This can be ensured by subsampling a correlated timeseries
        with a period larger than the statistical inefficiency, which can be estimated from the potential energy
        timeseries ``{u_k(x_ln)}_{n=1}^{N_k}`` using the provided utility
        :func:`pymbar.timeseries.statistical_inefficiency`.
        See the help for this function for more information.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)

        """

        # Store local copies of necessary data.
        # N_k[k] is the number of samples from state k, some of which might be zero.
        self.N_k = np.array(N_k, dtype=np.int64)
        self.N = np.sum(self.N_k)

        # Get dimensions of reduced potential energy matrix, and convert to KxN form if needed.
        if len(np.shape(u_kn)) == 3:
            self.K = np.shape(u_kn)[1]  # need to set self.K, and it's the second index
            u_kn = kln_to_kn(u_kn, N_k=self.N_k)

        # u_kn[k,n] is the reduced potential energy of sample n evaluated at state k
        self.u_kn = np.array(u_kn, dtype=np.float64)

        K, N = np.shape(u_kn)

        if verbose:
            logger.info("K (total states) = {:d}, total samples = {:d}".format(K, N))

        if np.sum(self.N_k) != N:
            raise ParameterError(
                "The sum of all N_k must equal the total number of samples (length of second dimension of u_kn."
            )

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
                self.x_kindices[Nsum : Nsum + self.N_k[k]] = k
                Nsum += self.N_k[k]

        # verbosity level -- if True, will print extra debug information
        self.verbose = verbose

        # perform consistency checks on the data.

        self.samestates = []
        if self.verbose:
            # if, for any set of data, all reduced potential energies are the same,
            # they are probably the same state.
            #
            # This can take quite a while, so we do it on just
            # the first few data points.
            #
            # determine if there are less than 50 points to compare energies.
            # If so, use that number instead of 50.
            # (the number 50 is pretty arbitrary)

            maxpoint = 50
            if self.N < maxpoint:
                maxpoint = self.N

            # this could possibly be made faster with np.unique(axis=0,return_indices=True)
            # but not clear if needed.

            # pick random indices
            # indices = np.arange(maxpoint) # can uncomment this if need to remove random choices in testing.
            indices = np.random.choice(np.arange(self.N), maxpoint)
            for k in range(K):
                for l in range(k):
                    diffsum = 0
                    uzero = u_kn[k, indices] - u_kn[l, indices]
                    diffsum += np.dot(uzero, uzero)
                    if diffsum < relative_tolerance:
                        self.samestates.append([k, l])
                        self.samestates.append([l, k])
                        msg = """
                        States {:d} and {:d} have the same energies on the dataset.
                        They are therefore likely to to be the same thermodynamic state. This can occasionally cause
                        numerical problems with computing the covariance of their energy difference, which must be
                        identically zero in any case. Consider combining them into a single state.
                        """.format(
                            l, k
                        )
                        logger.warning(dedent(msg[1:]))

        # Print number of samples from each state.
        if self.verbose:
            logger.info("N_k = ")
            logger.info(self.N_k)

        # Determine list of k indices for which N_k != 0
        self.states_with_samples = np.where(self.N_k != 0)[0]
        self.states_with_samples = self.states_with_samples.astype(np.int64)

        # Number of states with samples.
        self.K_nonzero = self.states_with_samples.size
        if verbose:
            logger.info("There are {:d} states with samples.".format(self.K_nonzero))

        # Initialize estimate of relative dimensionless free energy of each state to zero.
        # Note that f_k[0] will be constrained to be zero throughout.
        # this is default
        self.f_k = np.zeros([self.K], dtype=np.float64)

        # If an initial guess of the relative dimensionless free energies is
        # specified, start with that.
        if initial_f_k is not None:
            if self.verbose:
                logger.info("Initializing f_k with provided initial guess.")
            # Cast to np array.
            initial_f_k = np.array(initial_f_k, dtype=np.float64)
            # Check shape
            if initial_f_k.shape != self.f_k.shape:
                raise ParameterError(
                    "initial_f_k must be a {:d}-dimensional np array.".format(self.K)
                )
            # Initialize f_k with provided guess.
            self.f_k = initial_f_k
            if self.verbose:
                logger.info(self.f_k)
            # Shift all free energies such that f_0 = 0.
            self.f_k[:] = self.f_k[:] - self.f_k[0]
        else:
            # Initialize estimate of relative dimensionless free energies.
            self._initializeFreeEnergies(verbose, method=initialize, f_k_init=initial_f_k)

            if self.verbose:
                logger.info(
                    "Initial dimensionless free energies with method {:s}".format(initialize)
                )
                logger.info("f_k = ")
                logger.info(self.f_k)

        # decide on the protocol to use when solving.  We use similar logic
        # for both the solver protocol and the bootstrap solver protocol

        defaults = [DEFAULT_SOLVER_PROTOCOL, BOOTSTRAP_SOLVER_PROTOCOL]
        robusts = [ROBUST_SOLVER_PROTOCOL, ROBUST_SOLVER_PROTOCOL]
        pnames = ["solver_protocol", "bootstrap_solver_protocol"]
        protocols = {pnames[0]: solver_protocol, pnames[1]: bootstrap_solver_protocol}

        for defl, rob, pname in zip(defaults, robusts, pnames):

            prot = protocols[pname]
            if prot is None or prot == "default":
                prot = defl
            elif prot == "robust":
                prot = rob
            elif prot == "jax":
                prot = JAX_SOLVER_PROTOCOL
            else:
                for solver in prot:
                    if not isinstance(solver, dict):
                        logger.warning(
                            "{pname} is not 'robust','default' or a tuple/list dictionaries, setting to 'default'"
                        )
                        prot = defl

            for solver in prot:
                if "options" not in solver:
                    solver["options"] = dict()
                if "continuation" not in solver:
                    solver["continuation"] = None
                if "maxiter" not in solver["options"]:
                    solver["options"]["maxiter"] = maximum_iterations
                if maximum_iterations > solver["options"]["maxiter"]:
                    solver["options"]["maxiter"] = maximum_iterations
                    logger.info(
                        f"Explicitly overwriting maxiter={solver['options']['maxiter']} with maximum_iterations={maximum_iterations}"
                    )
                if "verbose" not in solver["options"]:
                    # should add in other ways to get information out of the scipy solvers, not just adaptive,
                    # which might involve passing in different combinations of options, and passing out other strings.
                    solver["options"]["verbose"] = self.verbose
            # seems like there should be a better way to do this.
            if pname == "solver_protocol":
                solver_protocol = prot
            elif pname == "bootstrap_solver_protocol":
                bootstrap_solver_protocol = prot

        if rseed != None:
            self.rstate = np.random.get_state()
        else:
            np.random.seed(rseed)

        self.f_k = mbar_solvers.solve_mbar_for_all_states(
            self.u_kn, self.N_k, self.f_k, self.states_with_samples, solver_protocol
        )

        if n_bootstraps > 0:
            self.n_bootstraps = n_bootstraps
            maxfrac = int(max((1, 0.1 * self.n_bootstraps)))

            self.f_k_boots = np.zeros([self.n_bootstraps, self.K])
            allN = int(np.sum(N_k))
            self.bootstrap_rints = np.zeros([self.n_bootstraps, allN], int)
            for b in range(self.n_bootstraps):
                f_k_init = np.array(self.f_k.copy())
                # picking random samples from the same N_k states.
                rints = np.zeros(allN, int)
                for k in range(K):
                    # which of the indices are equal to K
                    k_indices = np.where(self.x_kindices == k)[0]
                    # pick new random ones, selected of these K.
                    new_kindices = k_indices[np.random.randint(int(N_k[k]), size=int(N_k[k]))]
                    rints[k_indices] = new_kindices
                # If we initialized with BAR, then BAR, starting from the provided initial_f_k as well.
                if initialize == "BAR":
                    f_k_init = self._initialize_with_bar(self.u_kn[:, rints], f_k_init=self.f_k)
                self.f_k_boots[b, :] = mbar_solvers.solve_mbar_for_all_states(
                    self.u_kn[:, rints],
                    self.N_k,
                    f_k_init,
                    self.states_with_samples,
                    bootstrap_solver_protocol,
                )
                # save the random integers for computing expectations.
                self.bootstrap_rints[b, :] = rints

                if verbose:
                    if b % maxfrac == 0:
                        logger.info(f"Calculated {b+1:d}/{self.n_bootstraps:d} bootstrap samples")
        elif n_bootstraps < 0:
            logger.warning("n_bootstraps must be an integer >= 0")

        # bootstrapped weight matrices not generated here, but when expectations are needed
        # otherwise, it's too much memory to keep
        self.Log_W_nk = mbar_solvers.mbar_log_W_nk(self.u_kn, self.N_k, self.f_k)

        # Print final dimensionless free energies.
        if self.verbose:
            logger.info("Final dimensionless free energies")
            logger.info("f_k = ")
            logger.info(self.f_k)

        if self.verbose:
            logger.info("MBAR initialization complete.")

    @property
    def W_nk(self):
        """Retrieve the weight matrix W_nk from the MBAR algorithm.

        Necessary because they are stored internally as log weights.

        Returns
        -------
        weights : np.ndarray, float, shape=(N, K)
            NxK matrix of weights in the MBAR covariance and averaging formulas

        """
        return np.exp(self.Log_W_nk)

    # =========================================================================
    def weights(self):
        """Retrieve the weight matrix W_nk from the MBAR algorithm.

        Necessary because they are stored internally as log weights.

        Returns
        -------
        weights : np.ndarray, float, shape=(N, K)
            NxK matrix of weights in the MBAR covariance and averaging formulas

        """

        return self.W_nk

    # =========================================================================
    def compute_effective_sample_number(self, verbose=False):
        """
        Compute the effective sample number of each state;
        essentially, an estimate of how many samples are contributing to the average
        at given state.  See pymbar/examples for a demonstration.

        It also counts the efficiency of the sampling, which is simply the ratio
        of the effective number of samples at a given state to the total number
        of samples collected.  This is printed in verbose output, but is not
        returned for now.

        Returns
        -------
        N_eff : np.ndarray, float, shape=(K)
                estimated number of samples contributing to estimates at each
                state i. An estimate to how many samples collected just at state
                i would result in similar statistical efficiency as the MBAR
                simulation. Valid for both sampled states (in which the weight
                will be greater than N_k[i], and unsampled states.

        Parameters
        ----------
        verbose : print out information about the effective number of samples

        Notes
        -----

        Using Kish (1965) formula (Kish, Leslie (1965). Survey Sampling. New York: Wiley)

        As the weights become more concentrated in fewer observations, the effective sample size shrinks.
        from http://healthcare-economist.com/2013/08/22/effective-sample-size/

        .. code-block:: none

            effective number of samples contributing to averages carried out at state i
                =  (\\sum_{n=1}^N w_in)^2 / \\sum_{n=1}^N w_in^2
                =  (\\sum_{n=1}^N w_in^2)^-1

        the effective sample number is most useful to diagnose when there are only a few samples
        contributing to the averages.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> [x_kn, u_kln, N_k, s_n] = testsystems.HarmonicOscillatorsTestCase().sample()
        >>> mbar = MBAR(u_kln, N_k)
        >>> N_eff = mbar.compute_effective_sample_number()
        """

        N_eff = np.zeros(self.K)
        for k in range(self.K):
            w = np.exp(self.Log_W_nk[:, k])
            N_eff[k] = 1 / np.sum(w**2)
            if verbose:
                logger.info(
                    "Effective number of sample in state {:d} is {:10.3f}".format(k, N_eff[k])
                )
                logger.info(
                    "Efficiency for state {:d} is {:6f}/{:d} = {:10.4f}".format(
                        k, N_eff[k], len(w), N_eff[k] / len(w)
                    )
                )

        return N_eff

    # =========================================================================
    def compute_overlap(self):
        """
        Compute estimate of overlap matrix between the states.

        Parameters
        ----------
        None

        Returns
        -------
        Results dictionary with the following keys:

        'scalar' : np.ndarray, float, shape=(K, K)
            One minus the largest nontrival eigenvalue (largest is 1 or -1)
        'eigenvalues' : np.ndarray, float, shape=(K)
            The sorted (descending) eigenvalues of the overlap matrix.
        'matrix' : np.ndarray, float, shape=(K, K)
            Estimated state overlap matrix : O[i,j] is an estimate
            of the probability of observing a sample from state i in state j

        Notes
        -----

        .. code-block:: none

            W.T * W \\approx \\int (p_i p_j /\\sum_k N_k p_k)^2 \\sum_k N_k p_k dq^N
                = \\int (p_i p_j /\\sum_k N_k p_k) dq^N

        Multiplying elementwise by N_i, the elements of row i give the probability
        for a sample from state i being observed in state j.


        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_kn, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> results = mbar.compute_overlap()

        """

        W = self.weights()
        O = self.N_k * (W.T @ W)
        eigenvals, eigevec = linalg.eig(O)
        # sort in descending order
        eigenvals = np.sort(eigenvals)[::-1]
        overlap_scalar = 1 - eigenvals[1]  # 1 minus the second largest eigenvalue

        results_vals = dict()
        results_vals["scalar"] = overlap_scalar
        results_vals["eigenvalues"] = eigenvals
        results_vals["matrix"] = O

        return results_vals

    # =========================================================================
    def compute_free_energy_differences(
        self,
        compute_uncertainty=True,
        uncertainty_method=None,
        warning_cutoff=1.0e-10,
        return_theta=False,
    ):
        """Compute and return  the dimensionless free energy differences and uncertainties among all thermodynamic states.


        Parameters
        ----------
        compute_uncertainty : bool, optional
            If False, the uncertainties will not be computed (default : True)
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method,
            or None to use default.  See help for _computeAsymptoticCovarianceMatrix()
            for more information on various methods. (default : svd)
            The exception is the "bootstrap" option, which requires n_boostraps being defined upon initialization of MBAR.
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude
            than this number (default : 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences.

        Returns
        -------
        Results dictionary with the following keys:

        'Delta_f' : np.ndarray, float, shape=(K, K)
            Deltaf_ij[i,j] is the estimated free energy difference
        'dDelta_f' : np.ndarray, float, shape=(K, K)
            If compute_uncertainty==True,
            dDeltaf_ij[i,j] is the estimated statistical uncertainty
            (one standard deviation) in Deltaf_ij[i,j].  Otherwise not included.
        'Theta' : np.ndarray, float, shape=(K, K)
            The theta_matrix if return_theta==True, otherwise not included.

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
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> results = mbar.compute_free_energy_differences()

        """

        # Compute free energy differences and force cast back to mutable, normal ndarray (we're done with JAX here)
        # Cannot use asarray because that makes reference to the JAX memory, which NumPy doesn't own and is flagged
        # read-only (.flags(write=0)). Using .array makes a copy of the data to a new memory address, and is thus
        # mutable
        Deltaf_ij = np.array(self.f_k - np.vstack(self.f_k))

        # zero out numerical error for thermodynamically identical states
        self._zerosamestates(Deltaf_ij)

        result_vals = dict()

        result_vals["Delta_f"] = Deltaf_ij

        if uncertainty_method == "bootstrap" and (
            self.n_bootstraps <= 0 or self.n_bootstraps is None
        ):
            raise ParameterError(
                "Cannot request bootstrap sampling of free energy differences without any bootstraps."
            )

        if (compute_uncertainty and uncertainty_method != "bootstrap") or return_theta:
            # Compute asymptotic covariance matrix.
            Theta_ij = self._computeAsymptoticCovarianceMatrix(
                np.exp(self.Log_W_nk), self.N_k, method=uncertainty_method
            )

        if compute_uncertainty:
            if uncertainty_method == "bootstrap":
                diffm = np.zeros([self.n_bootstraps, self.K, self.K])
                for b in range(self.n_bootstraps):
                    # take the matrix of differences of f_ij
                    # Cast to np.ndarray if JAX array. See Deltaf_ij assignment above for details.
                    f = self.f_k_boots[b, :]
                    diffm[b, :, :] = f - np.vstack(f)
                dDeltaf_ij = np.std(diffm, axis=0)
                result_vals["dDelta_f"] = dDeltaf_ij
            else:
                # Cast to np.ndarray if JAX array. See Deltaf_ij assignment above for details.
                dDeltaf_ij = np.array(
                    self._ErrorOfDifferences(Theta_ij, warning_cutoff=warning_cutoff)
                )
                # zero out numerical error for thermodynamically identical states
                self._zerosamestates(dDeltaf_ij)
                # Return matrix of free energy differences and uncertainties.
                dDeltaf_ij = np.array(dDeltaf_ij)
                result_vals["dDelta_f"] = dDeltaf_ij

        if return_theta:
            result_vals["Theta"] = Theta_ij

        return result_vals

    # =========================================================================
    def compute_expectations_inner(
        self,
        A_n,
        u_ln,
        state_map,
        uncertainty_method=None,
        warning_cutoff=1.0e-10,
        return_theta=False,
    ):
        """
        Compute the expectations of multiple observables of phase space functions in multiple states.

        Compute the expectations of multiple observables of phase
        space functions [A_0(x),A_1(x),...,A_i(x)] along with the
        covariances of their estimates at multiple states.

        Intended as an internal function to keep all the optimized and
        robust expectation code in one place, but will leave it
        open to allow for later modifications and uses.

        It calculates all input observables at all states which are
        specified by the list of states in the state list.

        Parameters
        ----------
        A_n : np.ndarray, float, shape=(I, N)
            A_in[i,n] = A_i(x_n), the value of phase observable i for configuration n
        u_ln : np.ndarray, float, shape=(L, N)
            u_ln[l,n] is the reduced potential of configuration n at state l
            if u_ln = None, we use self.u_kn
        state_map : np.ndarray, int, shape (2,NS) or shape(1,NS)
            If state_map has only one dimension where NS is the total number of states we want to simulate things
            a.  The list will be of the form ``[[0,1,2],[0,1,1]]``. This particular example indicates we want to output
            the properties of three observables total: the first property A[0] at the 0th state, the 2nd property
            A[1] at the 1th state, and the 2nd property A[1] at the 2nd state. This allows us to tailor our output to a
            large number of different situations.
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for _computeAsymptoticCovarianceMatrix() for more information on various methods.
            The exception is the "bootstrap" option, which required n_boostraps being defined at initialization of MBAR.
            (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
        return_theta : bool, optional
            Whether or not to return the theta matrix.  Can be useful for complicated differences of observables.

        Returns
        -------
        result_vals : dictionary

        Keys in the result_vals dictionary:

        'observables' : np.ndarray, float, shape = (S)
            results_vals['observables'][i] is the estimate for the expectation of A_state_map[i](x) at the state specified by u_n[state_map[i],:]

        'Theta' : np.ndarray, float, shape = (K+len(state_list), K+len(state_list)) the covariance matrix of log weights.

        'Amin' : np.ndarray, float, shape = (S), needed for reconstructing the covariance one level up.

        'f' : np.ndarray, float, shape = (K+len(state_list)), 'free energies' of the new states (i.e. ln (<A>-Amin+logfactor)) as the log form is easier to work with.

        'bootstrapped_observables' : np.ndarray, float, shape = (n_boostraps,S), array of the observables bootstrapped over random samples.

        'bootstrapped_f' : np.ndarray, float, shape = (n_boostraps,S), free energies 'f' bootstrapped over random samples.


        Notes
        -----

        Situations this will be used for:

        * Multiple observables, single state (called though compute_multiple_expectations)
        * Single observable, multiple states (called through compute_expectations)

            This has two cases : observables that don't change with state, and observables that
            do change with state.
            For example, the set of energies at state k consist in energy function of state
            1 evaluated at state 1, energies of state 2 evaluated at
            state 2, and so forth.

        * Computing only free energies at new states.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> A_n = np.array([x_n,x_n**2,x_n**3])
        >>> u_n = u_kn[:2,:]
        >>> state_map = np.array([[0,0],[1,0],[2,0],[2,1]],int)
        >>> results = mbar.compute_expectations_inner(A_n, u_n, state_map)

        """

        logfactor = 4.0 * np.finfo(np.float64).eps
        # make sure all results are larger than this number.
        # We tried 1 before, but expecations that are all very small (like
        # fraction folded when it is low) cannot be computed accurately.
        # 0 causes warnings in the test with divide by zero, as does 1*eps (though fewer),
        # and even occasionally 2*eps, so we chooose 4*eps
        # Retrieve N and K for convenience.
        mapshape = np.shape(state_map)  # number of computed expectations we desire
        # need to convert to matrix to be able to pick up D=1
        if len(mapshape) < 2:
            # if 1D, it's just a list of states
            state_list = state_map.copy()
            state_map = np.zeros([0, 0], np.float64)
            S = 0
        else:  # if 2D, then it's a list of observables and corresponding states
            state_list = state_map[0, :]
            S = mapshape[1]

        # reshape arrays explicitly into 2d (even if only one state) to make it easy to manipulate
        shapeu = np.shape(u_ln)
        if len(shapeu) == 1:
            u_ln = np.reshape(u_ln, [1, shapeu[0]])

        shapeA = np.shape(A_n)
        if len(shapeA) == 1:
            A_n = np.reshape(A_n, [1, shapeA[0]])

        K = self.K
        N = self.N  # N is total number of samples
        result_vals = dict()  # dictionary we will store uncertainties in

        # make observables all positive, allowing us to take the logarithm, which is
        # required to prevent overflow in some examples.
        # WARNING: one issue to watch for is if one of the energies is extremely
        # low (-10^10 or lower), but most of the energies of interest are much higher.
        # This could lead to roundoff problems.

        L_list = np.unique(state_list)
        NL = len(L_list)  # number of states we need to examine
        if S > 0:
            A_list = np.unique(state_map[1, :])  # what are the unique observables
            A_min = np.zeros([len(A_list)], dtype=np.float64)
        else:
            A_list = np.zeros(0, dtype=int)

        logfactors = np.zeros(len(A_list))
        for i in A_list:
            A_min[i] = np.min(A_n[i, :])  # find the minimum
            logfactors[i] = np.abs(logfactor * A_min[i])
            A_n[i, :] = A_n[i, :] - (
                A_min[i] - logfactors[i]
            )  # all values now positive so that we can work in logarithmic scale

        # Augment W_nk, N_k, and c_k for q_A(x) for the observables, with one
        # row for the specified state and I rows for the observable at that
        # state.
        # log weight matrix
        msize = K + NL + S  # augmented size; all of the states needed to calculate
        # the observables, and the observables themselves.
        Log_W_nk = np.zeros([N, msize], np.float64)  # log weight matrix
        N_k = np.zeros([msize], np.int64)  # counts
        f_k = np.zeros([msize], np.float64)  # free energies

        if uncertainty_method == "bootstrap":
            n_total = self.n_bootstraps + 1
            A_i_bootstrap = np.zeros([self.n_bootstraps, S])
            f_bootstrap = np.zeros([self.n_bootstraps, len(state_list)])
        else:
            n_total = 1

        for n in range(n_total):
            # <A> = A(x_n) exp[f_{k} - q_{k}(x_n)] / \sum_{k'=1}^K N_{k'} exp[f_{k'} - q_{k'}(x_n)]
            # Fill in first section of matrix with existing q_k(x) from states.
            N_k[0:K] = self.N_k
            if n == 0:
                f_k[0:K] = self.f_k
                u_kn = self.u_kn
                Log_W_nk[:, 0:K] = self.Log_W_nk
                ri = np.arange(int(np.sum(self.N_k)))
            else:
                f_k[0:K] = self.f_k_boots[n - 1, :]
                ri = self.bootstrap_rints[n - 1]
                u_kn = self.u_kn[:, ri]
                Log_W_nk[:, 0:K] = mbar_solvers.mbar_log_W_nk(u_kn, self.N_k, f_k[0:K])
            # Pre-calculate the log denominator: Eqns 13, 14 in MBAR paper

            states_with_samples = self.N_k > 0
            log_denominator_n = logsumexp(
                f_k[0:K][states_with_samples] - u_kn[0:K][states_with_samples].T,
                b=self.N_k[states_with_samples],
                axis=1,
            )
            # Compute row of W_nk matrix for the extra states corresponding to u_ln
            # that the state list specifies
            for l in L_list:
                la = K + l  # l, augmented
                # Calculate log normalizing constants and log weights via Eqns 13, 14
                log_C_a = -logsumexp(
                    -u_ln[l, ri] - log_denominator_n
                )  # how do we change these numbers?
                Log_W_nk[:, la] = (
                    log_C_a - u_ln[l, ri] - log_denominator_n
                )  # how do we change these numbers?
                f_k[la] = log_C_a

            # Compute the remaining rows/columns of W_nk, and calculate
            # their normalizing constants c_k
            for s in range(S):
                sa = K + NL + s  # augmented s
                l = K + state_map[0, s]
                i = state_map[1, s]
                Log_W_nk[:, sa] = np.log(A_n[i, ri]) + Log_W_nk[:, l]
                f_k[sa] = -logsumexp(Log_W_nk[:, sa])
                Log_W_nk[:, sa] += f_k[sa]  # normalize this row

            # Compute estimates of A_i[s]
            A_i = np.zeros([S], np.float64)
            for s in range(S):
                A_i[s] = np.exp(-f_k[K + NL + s])

            # Now that covariances are computed, add the constants back to A_i that
            # were required to enforce positivity
            if n == 0:
                for s in range(S):
                    A_i[s] += A_min[state_map[1, s]] - logfactors[state_map[1, s]]
                    # expectations of the observables at these states
                    if S > 0:
                        result_vals["observables"] = A_i
                if return_theta:
                    Theta_ij = self._computeAsymptoticCovarianceMatrix(
                        np.exp(Log_W_nk), N_k, method=uncertainty_method
                    )

                # free energies at these new states.
                result_vals["f"] = f_k[K + state_list]
            else:
                for s in range(S):
                    A_i_bootstrap[n - 1, s] = A_i[s] + (
                        A_min[state_map[1, s]] - logfactors[state_map[1, s]]
                    )
                f_bootstrap[n - 1, :] = f_k[K + state_list]

        if uncertainty_method == "bootstrap":
            result_vals["bootstrapped_observables"] = A_i_bootstrap
            result_vals["bootstrapped_f"] = f_bootstrap

        # these values may be used outside the routine, so copy back.
        for i in A_list:
            A_n[i, :] = A_n[i, :] + (A_min[i] - logfactors[i])

        if return_theta:

            # Note: these variances will be the same whether or not we
            # subtract a different constant from each A_i
            # for efficency, output theta in block form
            #          K*K   K*S  K*NL
            # Theta =  K*S   S*S  NL*S
            #          K*NL  NL*S NL*NL

            # first the observables (S of them), then the free energies (also S of them)
            if S > 0:
                si = K + NL + np.arange(S)
            else:
                si = np.zeros(0, dtype=int)
            li = K + state_list
            i = np.concatenate((si, li))
            Theta = Theta_ij[np.ix_(i, i)]
            result_vals["Theta"] = Theta
            if S > 0:
                # we need to return the minimum A as well
                result_vals["Amin"] = (
                    A_min[state_map[1, np.arange(S)]] - logfactors[state_map[1, np.arange(S)]]
                )

        # Return expectations and uncertainties.
        return result_vals

        # For reference
        # Covariance of normalization constants is cov(ln A - ln a, ln B - ln b) = (Theta(c_A,c_B)-Theta(c_A,c_b)-Theta(c_B,c_a) + Theta(c_a,c_b))
        # Covariance of the differences of observables is cov(A-B)
        #   = Cov(A,A)+Cov(B,B)-2Cov(A,B) =   A^2 cov(ln A - ln a, ln A - ln a)
        #                                  +  B^2 cov(ln B - ln b, ln B - ln b)
        #                                  +  2AB cov(ln A - ln a, ln B - ln b)
        #                                 =   A^2 (Theta(c_A,c_A) + Theta(c_a,c_a) - 2Theta(c_A,c_a))
        #                                  +  B^2 (Theta(c_B,c_B) + Theta(c_b,c_b) - 2Theta(c_B,c_b))
        #                                  +  2AB (Theta(c_A,c_B) + Theta(c_a,c_b) - Theta(c_A,c_b) - Theta(c_B,c_a))
        #
        # Covariance in two observables = Cov(A,B)
        #                               = cov(exp(ln c_A - ln c_a),exp(ln c_B - ln c_b))
        #                               = AB cov(ln c_A - ln c_a, ln c_B - ln c_b)
        #                               = AB ((Theta(c_A,c_B) + Theta(c_a,c_b) - Theta(c_A,c_b) - Theta(c_B,c_a))
        #
        # Covariance of the differences of observables and a free energy (a could be b, or some other value)
        # is cov(A - ln c_b).
        #
        #   = Cov(exp(ln c_A - ln c_a), exp(ln c_A - ln c_a)) + Cov(c_b,c_b) - 2Cov(exp(ln c_A - ln c_a), c_b)
        #   = A^2 cov(ln c_A - ln c_a, ln c_A - ln c_a) + Cov(c_b,c_b) - 2A cov(ln c_A - ln c_a, ln c_b)
        #   =  A^2 ((Theta(c_A,c_A) + Theta(c_a,c_a) - 2Theta(c_A,c_a)) + Theta(c_b,c_b)
        #     - 2A Theta(c_A,c_b) + 2A Theta(c_a, c_b)
        #
        #   if A is sampled at the same free energy as the difference, then this will become:
        #   =  A^2 ((Theta(c_A,c_A) + Theta(c_a,c_a) - 2Theta(c_A,c_a)) + Theta(c_a,c_a)
        #     - 2A Theta(c_A,c_a) + 2A Theta(c_a, c_a)
        #   =  A^2 (Theta(c_A,c_A) + (A^2+2A+1)Theta(c_a,c_a) -(2A^2+2A)Theta(c_A,c_a)
        #

    # =========================================================================
    def compute_covariance_of_sums(self, d_ij, K, a):

        """
        We wish to calculate the variance of a weighted sum of free energy differences.
        for example ``var(\\sum a_i df_i)``.

        We explicitly lay out the calculations for four variables (where each variable
        is a logarithm of a partition function), then generalize.

        The uncertainty in the sum of two weighted differences is

        .. code-block:: none

            var(a1(f_i1 - f_j1) + a2(f_i2 - f_j2)) =
                a1^2 var(f_i1 - f_j1) +
                a2^2 var(f_i2 - f_j2) +
                2 a1 a2 cov(f_i1 - f_j1, f_i2 - f_j2)
            cov(f_i1 - f_j1, f_i2 - f_j2) =
                cov(f_i1,f_i2) -
                cov(f_i1,f_j2) -
                cov(f_j1,f_i2) +
                cov(f_j1,f_j2)

        call:

        .. code-block:: none

            f_i1 = a
            f_j1 = b
            f_i2 = c
            f_j2 = d
            a1^2 var(a-b) + a2^2 var(c-d) + 2a1a2 cov(a-b,c-d)

        we want ``2cov(a-b,c-d) = 2cov(a,c)-2cov(a,d)-2cov(b,c)+2cov(b,d)``,
        since ``var(x-y) = var(x) + var(y) - 2cov(x,y)``,
        then, ``2cov(x,y) = -var(x-y) + var(x) + var(y)``. So, we get

        .. code-block:: none

            2cov(a,c) = -var(a-c) + var(a) + var(c)
            -2cov(a,d) = +var(a-d) - var(a) - var(d)
            -2cov(b,c) = +var(b-c) - var(b) - var(c)
            2cov(b,d) = -var(b-d) + var(b) + var(d)

        adding up, finally :

        .. code-block:: none

            2cov(a-b,c-d) = 2cov(a,c)-2cov(a,d)-2cov(b,c)+2cov(b,d) =
                - var(a-c) + var(a-d) + var(b-c) - var(b-d)

            a1^2 var(a-b)+a2^2 var(c-d)+2a1a2cov(a-b,c-d) =
                a1^2 var(a-b)+a2^2 var(c-d)+a1a2 [-var(a-c)+var(a-d)+var(b-c)-var(b-d)]

            var(a1(f_i1 - f_j1) + a2(f_i2 - f_j2)) =
                a1^2 var(f_i1 - f_j1) + a2^2 var(f_i2 - f_j2) + 2a1 a2 cov(f_i1 - f_j1, f_i2 - f_j2)
            = a1^2 var(f_i1 - f_j1) + a2^2 var(f_i2 - f_j2) + a1 a2 [-var(f_i1 - f_i2) + var(f_i1 - f_j2) + var(f_j1-f_i2) - var(f_j1 - f_j2)]

        assume two arrays of free energy differences, and and array of constant vectors a.
        we want the variance ``var(\\sum_k a_k (f_i,k - f_j,k))`` Each set is separated from the other by an offset K
        same process applies with the sum, with the single var terms and the pair terms

        Parameters
        ----------
        d_ij : a matrix of standard deviations of the quantities f_i - f_j
        K : The number of states in each 'chunk', has to be constant
        outputs : KxK variance matrix for the sums or differences ``\\sum a_i df_i``
        """

        # todo: vectorize this, use for computing entropy and enthalpy.
        var_ij = np.square(d_ij)
        d2 = np.zeros([K, K], float)
        n = len(a)
        for i in range(K):
            for j in range(K):
                for k in range(n):
                    d2[i, j] += a[k] ** 2 * var_ij[i + k * K, j + k * K]
                    for l in range(n):
                        d2[i, j] += (
                            a[k]
                            * a[l]
                            * (
                                -var_ij[i + k * K, i + l * K]
                                + var_ij[i + k * K, j + l * K]
                                + var_ij[j + k * K, i + l * K]
                                - var_ij[j + k * K, j + l * K]
                            )
                        )

        return np.sqrt(d2)

    # =========================================================================
    def compute_expectations(
        self,
        A_n,
        u_kn=None,
        output="averages",
        state_dependent=False,
        compute_uncertainty=True,
        uncertainty_method=None,
        warning_cutoff=1.0e-10,
        return_theta=False,
    ):
        """Compute the expectation of an observable of a phase space function.

        Compute the expectation of an observable of a single phase space
        function A(x) at all states where potentials are generated.

        Parameters
        ----------
        A_n : np.ndarray, float
            A_n (N_max np float64 array) - A_n[n] = A(x_n)

        u_kn : np.ndarray
            u_kn (energies of state of interest length N)
            default is self.u_kn

        output : string, optional
            'averages' outputs expectations of observables and 'differences' outputs
            a matrix of differences in the observables.

        compute_uncertainty : bool, optional
            If False, the uncertainties will not be computed (default : True)

        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method,
            or None to use default See help for _computeAsymptoticCovarianceMatrix()
            for more information on various methods. if uncertainty_method = "bootstrap",
            then uncertainty over bootstrap samples is used. (default: None)

        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)

        state_dependent : bool, whether the expectations are state-dependent.

        Returns
        -------

        Results dictionary with the following keys

        'mu' : np.ndarray, float
            if output is 'averages'
            A_i  (K np float64 array) -  A_i[i] is the estimate for the expectation of A(x) for state i.
            if output is 'differences'
        'sigma' : np.ndarray, float
            dA_i  (K np float64 array) - dA_i[i] is uncertainty estimate (one standard deviation) for A_i[i]
            or
            dA_ij (K np float64 array) - dA_ij[i,j] is uncertainty estimate (one standard deviation) for the difference in A beteen i and j
            or None, if compute_uncertainty is False.
        'Theta' : ((KxK np float64 array): Covariance matrix of log weights

        References
        ----------

        See Section IV of [1].

        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> A_n = x_n
        >>> results = mbar.compute_expectations(A_n)
        >>> A_n = u_kn[0,:]
        >>> results = mbar.compute_expectations(A_n, output='differences')
        """

        if uncertainty_method == "bootstrap" and (
            self.n_bootstraps <= 0 or self.n_bootstraps is None
        ):
            raise ParameterError(
                "Cannot request bootstrap sampling of expectations without any bootstraps."
            )

        dims = len(np.shape(A_n))

        if dims > 2:
            logger.warning(
                "dim=3 for (state_dependent==True) matrices for observables and dim=2 for (state_dependent==False) observables are deprecated; we suggest you convert to NxK form instead of NxKxK form."
            )

        if not state_dependent:
            if dims == 2:
                A_n = kn_to_n(A_n, N_k=self.N_k)
                if u_kn is not None:
                    if len(np.shape(u_kn)) == 3:
                        u_kn = kln_to_kn(u_kn, N_k=self.N_k)
                    elif len(np.shape(u_kn)) == 2:
                        u_kn = kn_to_n(u_kn, N_k=self.N_k)
        else:
            if dims == 3:
                A_n = kln_to_kn(A_n, N_k=self.N_k)
                if u_kn is not None:
                    if len(np.shape(u_kn)) == 3:
                        u_kn = kln_to_kn(u_kn, N_k=self.N_k)
                    elif len(np.shape(u_kn)) == 2:
                        u_kn = kn_to_n(u_kn, N_k=self.N_k)

        if u_kn is None:
            u_kn = self.u_kn

        # Retrieve N and K for convenience.
        N = self.N
        ushape = np.shape(u_kn)
        if len(ushape) == 1:
            K = 1
        else:
            K = np.shape(u_kn)[0]  # number of potentials provided.

        state_map = np.zeros([2, K], int)
        if state_dependent:
            for k in range(K):
                # first property at the first state, 2nd property at the 2nd state
                state_map[0, k] = k
                state_map[1, k] = k
        else:
            # only one property, evaluate at K different states.
            for k in range(K):
                state_map[0, k] = k
                state_map[1, k] = 0

        inner_results = self.compute_expectations_inner(
            A_n,
            u_kn,
            state_map,
            return_theta=compute_uncertainty,
            uncertainty_method=uncertainty_method,
            warning_cutoff=warning_cutoff,
        )

        result_vals = dict()
        if (compute_uncertainty and uncertainty_method != "bootstrap") or return_theta:
            # we want the theta matrix for the exponentials of the
            # observables, which means we need to make the
            # transformation.
            Adiag = np.zeros([2 * K, 2 * K], dtype=np.float64)
            diag = np.ones(2 * K, dtype=np.float64)
            diag[0:K] = diag[K : 2 * K] = inner_results["observables"] - inner_results["Amin"]
            np.fill_diagonal(Adiag, diag)
            Theta = Adiag @ inner_results["Theta"] @ Adiag  # matrix multipliction . . .
            covA_ij = np.array(
                Theta[0:K, 0:K]
                + Theta[K : 2 * K, K : 2 * K]
                - Theta[0:K, K : 2 * K]
                - Theta[K : 2 * K, 0:K]
            )

        if output == "averages":
            result_vals["mu"] = inner_results["observables"]
            if compute_uncertainty:
                if uncertainty_method == "bootstrap":
                    result_vals["sigma"] = np.std(
                        inner_results["bootstrapped_observables"], axis=0
                    )
                else:
                    result_vals["sigma"] = np.sqrt(covA_ij[0:K, 0:K].diagonal())

        if output == "differences":
            A_im = inner_results["observables"]
            result_vals["mu"] = A_im - np.vstack(A_im)  # Cast to A_ij

            if compute_uncertainty:
                if uncertainty_method == "bootstrap":
                    bootstrap_differences = np.zeros([self.n_bootstraps, len(A_im), len(A_im)])
                    for b in range(self.n_bootstraps):
                        A_b = inner_results["bootstrapped_observables"][b]
                        bootstrap_differences[b, :, :] = A_b - np.vstack(A_b)
                    result_vals["sigma"] = np.std(bootstrap_differences, axis=0)
                else:
                    result_vals["sigma"] = self._ErrorOfDifferences(
                        covA_ij, warning_cutoff=warning_cutoff
                    )

        if return_theta:
            result_vals["Theta"] = Theta

        return result_vals

    # =========================================================================
    def compute_multiple_expectations(
        self,
        A_in,
        u_n,
        compute_uncertainty=True,
        compute_covariance=False,
        uncertainty_method=None,
        warning_cutoff=1.0e-10,
        return_theta=False,
    ):
        """Compute the expectations of multiple observables of phase space functions.

        Compute the expectations of multiple observables of phase
        space functions [A_0(x),A_1(x),...,A_i(x)] at a single state,
        along with the error in the estimates and the uncertainty in
        the estimates.  The state is specified by the choice of u_n,
        which is the energy of the n samples evaluated at a the chosen
        state.


        Parameters
        ----------
        A_in : np.ndarray, float, shape=(I, k, N)
            A_in[i,n] = A_i(x_n), the value of phase observable i for configuration n at state of interest
        u_n : np.ndarray, float, shape=(N)
            u_n[n] is the reduced potential of configuration n at the state of interest
        compute_uncertainty : bool, optional, default=True
            If True, calculate the uncertainty
        compute_covariance : bool, optional, default=False
            If True, calculate the covariance
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods.
            if method = "bootstrap" then uncertainty over bootstrap samples is used.
            with bootstraps. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default : 1.0e-10)

        Returns
        -------
        Results dictionary with the following keys

        'mu' : np.ndarray, float, shape=(I)
            result_vals['mu'] is the estimate for the expectation of A_i(x) at the state specified by u_kn
        'sigma' : np.ndarray, float, shape = (I)
            result_vals['sigma'] is the uncertainty in the expectation of A_state_map[i](x) at the state specified by u_n[state_map[i],:]
            or None if compute_uncertainty is False
        'covariances' : np.ndarray, float, shape=(I, I)
            result_vals['covariances'] is the COVARIANCE in the estimates of A_i[i] and A_i[j]: we can't actually take a square root
            or None if compute_covariance is False
        'Theta' : np.ndarray, float, shape=(I, I), covariances of the log weights, useful for some additional calculations.

        Examples
        --------

        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> A_in = np.array([x_n,x_n**2,x_n**3])
        >>> u_n = u_kn[0,:]
        >>> results = mbar.compute_multiple_expectations(A_in, u_kn)

        """

        # Retrieve N and K for convenience.
        I = A_in.shape[0]  # number of observables
        K = self.K
        N = self.N  # N is total number of samples

        if len(np.shape(A_in)) == 3:
            A_in_old = A_in.copy()  # convert to k by n format
            A_in = np.zeros([I, N], np.float64)
            for i in range(I):
                A_in[i, :] = kn_to_n(A_in_old[i, :, :], N_k=self.N_k)

        if len(np.shape(u_n)) == 2:
            u_n = kn_to_n(u_n, N_k=self.N_k)

        state_map = np.zeros([2, I], int)
        state_map[1, :] = np.arange(I)  # same (first) state for all variables.

        inner_results = self.compute_expectations_inner(
            A_in,
            u_n,
            state_map,
            return_theta=(compute_uncertainty or compute_covariance),
            uncertainty_method=uncertainty_method,
            warning_cutoff=warning_cutoff,
        )
        result_vals = dict()
        result_vals["mu"] = inner_results["observables"]

        if (
            (compute_uncertainty or compute_covariance) and compute_uncertainty != "bootstrap"
        ) or return_theta:
            Adiag = np.zeros([2 * I, 2 * I], dtype=np.float64)
            diag = np.ones(2 * I, dtype=np.float64)
            diag[0:I] = diag[I : 2 * I] = inner_results["observables"] - inner_results["Amin"]
            np.fill_diagonal(Adiag, diag)
            Theta = Adiag @ inner_results["Theta"] @ Adiag  # matrix multiplication

            if compute_uncertainty:
                covA_ij = np.array(
                    Theta[0:I, 0:I]
                    + Theta[I : 2 * I, I : 2 * I]
                    - Theta[0:I, I : 2 * I]
                    - Theta[I : 2 * I, 0:I]
                )
                result_vals["sigma"] = np.sqrt(covA_ij[0:I, 0:I].diagonal())

            if compute_covariance:
                # compute estimate of statistical covariance of the observables
                result_vals["covariances"] = inner_results["Theta"][0:I, 0:I]

            if return_theta:
                result_vals["Theta"] = Theta
        if uncertainty_method == "bootstrap":
            if compute_uncertainty:
                result_vals["sigma"] = np.std(inner_results["bootstrapped_observables"], axis=0)
            if compute_covariance:
                result_vals["covariances"] = np.cov(inner_results["bootstrapped_observables"].T)
        return result_vals

    # =========================================================================
    def compute_perturbed_free_energies(
        self, u_ln, compute_uncertainty=True, uncertainty_method=None, warning_cutoff=1.0e-10
    ):
        """Compute the free energies for a new set of states.

        Here, we desire the free energy differences among a set of new states, as well as the uncertainty estimates in these differences.

        Parameters
        ----------
        u_ln : np.ndarray, float, shape=(L, Nmax)
            u_ln[l,n] is the reduced potential energy of uncorrelated
            configuration n evaluated at new state k.  Can be completely indepednent of the original number of states.
        compute_uncertainty : bool, optional, default=True
            If False, the uncertainties will not be computed (default: True)
        uncertainty_method : string, optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods.
            if method = "bootstrap" then uncertainty over bootstrap samples is used.
            with bootstraps. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)

        Returns
        -------
        Results dictionary with the following entries.

        'Delta_f' : np.ndarray, float, shape=(L, L)
            result_vals['Delta_f'] = f_j - f_i, the dimensionless free energy difference between new states i and j
        'dDelta_f' : np.ndarray, float, shape=(L, L)
            result_vals['dDelta_f'] is the estimated statistical uncertainty in result_vals['Delta_f']
            or not included if `compute_uncertainty` is False

        Examples
        --------
        >>> from pymbar import testsystems
        >>> (x_n, u_kn, N_k, s_n) = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> results = mbar.compute_perturbed_free_energies(u_kn)
        """

        # Get the dimensions of the matrix of reduced potential energies, and convert if necessary
        if len(np.shape(u_ln)) == 3:
            u_ln = kln_to_kn(u_ln, N_k=self.N_k)

        L, N = u_ln.shape

        # Check dimensions.
        if N < self.N:
            raise DataError(
                "There seems to be too few samples in u_kn. You must evaluate at the new potential with all of the samples used originally."
            )

        state_list = np.arange(L)  # need to get it into the correct shape
        A_in = np.array([0])
        inner_results = self.compute_expectations_inner(
            A_in,
            u_ln,
            state_list,
            return_theta=compute_uncertainty,
            uncertainty_method=uncertainty_method,
            warning_cutoff=warning_cutoff,
        )

        Deltaf_ij, dDeltaf_ij = None, None

        f_k = inner_results["f"]

        result_vals = dict()
        result_vals["Delta_f"] = f_k - np.vstack(f_k)

        if compute_uncertainty:
            if uncertainty_method == "bootstrap":
                result_vals["dDelta_f"] = np.std(inner_results["bootstrapped_f"], axis=0)
            else:
                result_vals["dDelta_f"] = self._ErrorOfDifferences(
                    inner_results["Theta"], warning_cutoff=warning_cutoff
                )

        return result_vals

    # =====================================================================

    def compute_entropy_and_enthalpy(
        self, u_kn=None, uncertainty_method=None, verbose=False, warning_cutoff=1.0e-10
    ):
        """Decompose free energy differences into enthalpy and entropy differences.

        Compute the decomposition of the free energy difference between
        states 1 and N into reduced free energy differences, reduced potential
        (enthalpy) differences, and reduced entropy (S/k) differences.

        Parameters
        ----------
        u_kn : float, NxK array
            The energies of the state that are being used.
        uncertainty_method : str , optional
            Choice of method used to compute asymptotic covariance method, or None to use default
            See help for computeAsymptoticCovarianceMatrix() for more information on various methods.
            if method = "bootstrap" then uncertainty over bootstrap samples is used.
            with bootstraps. (default: None)
        warning_cutoff : float, optional
            Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)

        Returns
        -------
        Results dictionary with the following keys

        'Delta_f' : np.ndarray, float, shape=(K, K)
            results['Delta_f'] is the dimensionless free energy difference f_j - f_i
        'dDelta_f' : np.ndarray, float, shape=(K, K)
            uncertainty in results['Delta_f']
        'Delta_u' : np.ndarray, float, shape=(K, K)
            results['Delta_u'] is the reduced potential energy difference u_j - u_i
        'dDelta_u' : np.ndarray, float, shape=(K, K)
            uncertainty in results['Delta_u']
        'Delta_s' : np.ndarray, float, shape=(K, K)
            results['Delta_s'] is the reduced entropy difference S/k between states i and j (s_j - s_i)
        'dDelta_s' : np.ndarray, float, shape=(K, K)
            uncertainty in results['Delta_s']

        Examples
        --------

        >>> from pymbar import testsystems
        >>> x_n, u_kn, N_k, s_n = testsystems.HarmonicOscillatorsTestCase().sample(mode='u_kn')
        >>> mbar = MBAR(u_kn, N_k)
        >>> results = mbar.compute_entropy_and_enthalpy()

        """
        if verbose:
            logger.info("Computing average energy and entropy by MBAR.")

        dims = len(np.shape(u_kn))
        if dims == 3:
            u_kn = kln_to_kn(u_kn, N_k=self.N_k)

        if u_kn is None:
            u_kn = self.u_kn

        # Retrieve N and K for convenience.
        K, N = np.shape(u_kn)
        A_in = u_kn.copy()
        state_map = np.zeros([2, K], int)
        for k in range(K):
            state_map[0, k] = k
            state_map[1, k] = k

        inner_results = self.compute_expectations_inner(
            A_in,
            u_kn,
            state_map,
            return_theta=True,
            uncertainty_method=uncertainty_method,
            warning_cutoff=warning_cutoff,
        )

        # construct the covariance matrix of exp(ln c_Ua - ln c_a) - ln c_ca

        Theta = np.zeros([3 * K, 3 * K], dtype=np.float64)
        Theta[0 : 2 * K, 0 : 2 * K] = inner_results["Theta"]
        Theta[2 * K : 3 * K, :] = Theta[K : 2 * K, :]
        Theta[:, 2 * K : 3 * K] = Theta[:, K : 2 * K]
        diag = np.ones(3 * K, dtype=np.float64)
        diag[0:K] = diag[K : 2 * K] = inner_results["observables"] - inner_results["Amin"]
        Adiag = np.zeros([3 * K, 3 * K], dtype=np.float64)
        np.fill_diagonal(Adiag, diag)
        Theta = Adiag @ Theta @ Adiag

        result_vals = dict()
        # Compute reduced free energy difference.
        f_k = inner_results["f"]
        result_vals["Delta_f"] = f_k - np.vstack(f_k)
        # Compute reduced enthalpy difference.
        u_k = inner_results["observables"]
        result_vals["Delta_u"] = u_k - np.vstack(u_k)
        # Compute reduced entropy difference
        s_k = u_k - f_k
        result_vals["Delta_s"] = s_k - np.vstack(s_k)

        if uncertainty_method == "bootstrap":
            # bootstrapped free energy
            diffm = np.zeros([self.n_bootstraps, self.K, self.K])
            for b in range(self.n_bootstraps):
                # take the matrix of differences of f_ij
                f = self.f_k_boots[b, :]
                diffm[b, :, :] = f - np.vstack(f)
            result_vals["dDelta_f"] = np.std(diffm, axis=0)

            # bootstrapped enthalpy
            for b in range(self.n_bootstraps):
                # take the matrix of differences of f_ij
                u = inner_results["bootstrapped_observables"][b]
                diffm[b, :, :] = u - np.vstack(u)
            result_vals["dDelta_u"] = np.std(diffm, axis=0)
            # bootstrapped entropy
            for b in range(self.n_bootstraps):
                # take the matrix of differences of f_ij
                s = inner_results["bootstrapped_observables"][b] - self.f_k_boots[b, :]
                diffm[b, :, :] = s - np.vstack(s)
            result_vals["dDelta_s"] = np.std(diffm, axis=0)
        else:
            # compute uncertainty matrix in free energies:
            covf = Theta[2 * K : 3 * K, 2 * K : 3 * K]
            result_vals["dDelta_f"] = self._ErrorOfDifferences(covf, warning_cutoff=warning_cutoff)

            # compute uncertainty matrix in energies/enthalpies:
            covu = (
                Theta[0:K, 0:K]
                + Theta[K : 2 * K, K : 2 * K]
                - Theta[0:K, K : 2 * K]
                - Theta[K : 2 * K, 0:K]
            )
            result_vals["dDelta_u"] = self._ErrorOfDifferences(covu, warning_cutoff=warning_cutoff)

            # compute uncertainty matrix in entropies
            # s_i = u_i - f_i
            # cov(s_i) =   cov(u_i - f_i)
            #         =   cov(exp(ln C_a - ln c_a) + ln c_a)
            #         =   cov(exp(ln C_a - ln c_a), exp(ln C_a - ln c_a)) + cov(ln c_a, ln c_a)
            #           + cov(exp(ln C_a - ln c_a), ln c_a) + cov(ln c_a, exp(ln C_a - ln c_a))
            #         = cov(u,u) + cov(f,f)
            #             + A cov(ln C_a - ln c_a, ln c_a) + A cov(ln c_a, ln C_a - ln c_a)
            #         = cov(u,u) + cov(f,f)
            #             + A cov(ln C_a, ln c_a) - A cov(ln c_a, ln c_a) + A cov(ln c_a, ln C_a) - A cov(ln c_a, ln c_a)
            #         = cov(u,u) + cov(f,f) + A cov(ln C_a,ln c_a) + A cov(ln c_a, ln C_a) - 2A cov(ln_ca,ln_ca)
            #
            covs = (
                covu
                + covf
                + Theta[0:K, 2 * K : 3 * K]
                + Theta[2 * K : 3 * K, 0:K]
                - Theta[K : 2 * K, 2 * K : 3 * K]
                - Theta[2 * K : 3 * K, K : 2 * K]
            )
            # note: not clear that Theta[K:2*K,2*K:3*K] and Theta[K:2*K,2*K:3*K] are symmetric?
            result_vals["dDelta_s"] = self._ErrorOfDifferences(covs, warning_cutoff=warning_cutoff)

        return result_vals

    # =========================================================================
    # PRIVATE METHODS - INTERFACES ARE NOT expORTED
    # =========================================================================

    def _ErrorOfDifferences(self, cov, warning_cutoff=1.0e-10):
        """
        inputs:
        cov is the covariance matrix of A

        returns the statistical error matrix of A_i - A_j
        """

        diag = cov.diagonal()
        d2 = diag + np.vstack(diag) - 2 * cov

        # Cast warning_cutoff to compare a negative number
        cutoff = -abs(warning_cutoff)

        # check for any numbers below zero.
        if np.any(d2 < 0.0):
            if np.any(d2 < cutoff):
                logger.warning(
                    "A squared uncertainty is negative. Largest Magnitude = {0:f}".format(
                        abs(np.min(d2[d2 < cutoff]))
                    )
                )
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

    # =========================================================================

    def _zerosamestates(self, A):
        """
        zeros out states that should be identical

        Since this is in place update, must be normal NumPy ndarray instead of JAX DeviceArray

        REQUIRED ARGUMENTS

        A: np.ndarray
            The matrix whose entries are to be zeroed.

        """
        for pair in self.samestates:
            A[pair[0], pair[1]] = 0
            A[pair[1], pair[0]] = 0

    # =========================================================================
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
        Theta : np.ndarray, shape=(K, K), dtype='float'
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

        if uncertainty method = "bootstrap", we asked for uncertainties with bootstrap, so we probably don't care that
        much about the theta matrix, and we will just default to using svd-ew, the best method.


        """

        # Set 'svd-ew' as default if uncertainty method specified as None.
        if method == None or method == "bootstrap":
            method = "svd-ew"

        # Get dimensions of weight matrix.
        N, K = W.shape

        # Check dimensions
        if K != N_k.size:
            raise ParameterError("W must be NxK, where N_k is a K-dimensional array.")
        if np.sum(N_k) != N:
            raise ParameterError("W must be NxK, where N = sum_k N_k.")

        check_w_normalized(W, N_k)

        # Compute estimate of asymptotic covariance matrix using specified method.
        if method == "approximate":
            # Use fast approximate expression from Kong et al. -- this underestimates the true covariance, but may be a good approximation in some cases and requires no matrix inversions
            # Theta = P'P

            # Compute covariance
            Theta = W.T @ W

        elif method == "svd":
            # Use singular value decomposition based approach given in supplementary material to efficiently compute uncertainty
            # See Appendix D.1, Eq. D4 in [1].

            # Construct matrices
            Ndiag = np.diag(N_k)
            I = np.identity(K, dtype=np.float64)

            # Compute SVD of W
            # False Avoids O(N^2) memory allocation by only calculting the active subspace of U.
            U, S, Vt = linalg.svd(W, full_matrices=False)
            Sigma = np.diag(S)
            V = Vt.T

            # Compute covariance
            Theta = (
                V @ Sigma @ self._pseudoinverse(I - Sigma @ V.T @ Ndiag @ V @ Sigma) @ Sigma @ V.T
            )

        elif method == "svd-ew":
            # Use singular value decomposition based approach given in supplementary material to efficiently compute uncertainty
            # The eigenvalue decomposition of W'W is used to forego computing the SVD.
            # See Appendix D.1, Eqs. D4 and D5 of [1].

            # Construct matrices
            Ndiag = np.diag(N_k)
            I = np.identity(K, dtype=np.float64)

            # Compute singular values and right singular vectors of W without using SVD
            # Instead, we compute eigenvalues and eigenvectors of W'W.
            # Note W'W = (U S V')'(U S V') = V S' U' U S V' = V (S'S) V'
            S2, V = linalg.eigh(W.T @ W)
            # Set any slightly negative eigenvalues to zero.
            S2[np.where(S2 < 0.0)] = 0.0
            # Form matrix of singular values Sigma, and V.
            Sigma = np.diag(np.sqrt(S2))

            # Compute covariance
            Theta = (
                V @ Sigma @ self._pseudoinverse(I - Sigma @ V.T @ Ndiag @ V @ Sigma) @ Sigma @ V.T
            )

        else:
            # Raise an exception.
            raise ParameterError(f"Method {method} unrecognized.")

        return Theta

    # =========================================================================

    def _initializeFreeEnergies(self, verbose=False, method="zeros", f_k_init=None):
        """
        Compute an initial guess at the relative free energies.

        Parameters
        ----------
        verbose : bool, optional=False
            If True, will print debug information
        method : str, optional=zeros
            Method for initializing guess at free energies.
            * zeros : all free energies are initially set to zero
            * mean-reduced-potential : the mean reduced potential is used
            * 'BAR' : BAR is used to find the free energy difference between consecutive states

        """

        if method == "zeros":
            # Use zeros for initial free energies.
            if verbose:
                logger.info("Initializing free energies to zero.")
            self.f_k[:] = 0.0
        elif method == "mean-reduced-potential":
            # Compute initial guess at free energies from the mean reduced
            # potential from each state
            if verbose:
                logger.info(
                    "Initializing free energies with mean reduced potential for each state."
                )
            means = np.zeros([self.K], float)
            for k in self.states_with_samples:
                means[k] = self.u_kn[k, 0 : self.N_k[k]].mean()
            if np.max(np.abs(means)) < 0.000001:
                logger.warning(
                    "Warning: All mean reduced potentials are close to zero. "
                    "If you are using energy differences in the u_kln matrix, "
                    "then the mean reduced potentials will be zero, and this is expected behavoir."
                )
            self.f_k = means
        elif method == "BAR":
            self.f_k = self._initialize_with_bar(self.u_kn, f_k_init)
        else:
            # The specified method is not implemented.
            raise ParameterError("Method " + method + " unrecognized.")

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
          'log weights' here refers to \\log [ \\sum_{k=1}^K N_k exp[f_k - (u_k(x_n) - u(x_n)] ]
        """
        return -1.0 * logsumexp(self.f_k + u_n[:, np.newaxis] - self.u_kn.T, b=self.N_k, axis=1)

    def _initialize_with_bar(self, u_kn, f_k_init=None):

        """

        Internal method for intializing free energies simulations with BAR.
        Only useful to do when the states are in order.

        """

        from pymbar.other_estimators import bar

        initialization_order = np.where(self.N_k > 0)[0]
        # Initialize all f_k to zero.
        if f_k_init is None:
            f_k_init = np.zeros(len(self.f_k))

        starting_f_k_init = f_k_init.copy()
        for index in range(0, np.size(initialization_order) - 1):
            k = initialization_order[index]
            l = initialization_order[index + 1]
            # forward work
            # here, we actually need to distinguish which states are which
            w_F = u_kn[l, self.x_kindices == k] - u_kn[k, self.x_kindices == k]
            # self.u_kln[k, l, 0:self.N_k[k]] - self.u_kln[k, k, 0:self.N_k[k]])
            # reverse work
            w_R = u_kn[k, self.x_kindices == l] - u_kn[l, self.x_kindices == l]
            # self.u_kln[l, k, 0:self.N_k[l]] - self.u_kln[l, l, 0:self.N_k[l]])

            if len(w_F) > 0 and len(w_R) > 0:
                # bar solution doesn't need to be incredibly accurate to
                # kickstart NR.
                try:
                    f_k_init[l] = (
                        f_k_init[k]
                        + bar(
                            w_F,
                            w_R,
                            method="bisection",  # Fast, so be more certain.
                            DeltaF=starting_f_k_init[l] - starting_f_k_init[k],
                            relative_tolerance=0.00001,
                            verbose=False,
                            maximum_iterations=100,  # If it doesn't converge fast, probably not worth it.
                            compute_uncertainty=False,
                        )["Delta_f"]
                    )
                except ConvergenceError:
                    logger.warn("WARNING: BAR did not converge to within tolerance")
                    f_k_init[l] = f_k_init[k]  # just assume zero if didn't converge.
            else:
                # no states observed, so we don't need to initialize this free energy anyway, as
                # the solution is noniterative.
                f_k_init[l] = 0

        return f_k_init
