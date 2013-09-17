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

Notes
-----

If numerical precision / underflow is a problem, the following code could
be improved by replacing `logsumexp` with an implementation of `logsumexp`
that uses Kahan summation--as in `math.fsum()`.

"""

import numpy as np
import itertools
from pymbar.utils import ensure_type, ParameterError, validate_weight_matrix
from sklearn.utils.extmath import logsumexp
import scipy.optimize
import logging

def check_same_states(u_kn, relative_tolerance=1E-7):
    n_states, n_samples = u_kn.shape
    for (i, j) in itertools.combinations(xrange(n_states), 2):
        delta = u_kn[i] - u_kn[j]
        delta = np.linalg.norm(delta) ** 2. 
        if delta < relative_tolerance:
            raise(ValueError("Error: states %d and %d are likely identical: Delta U = %f" % (i, j, delta)))

def list_same_states(u_kn, N_k):
    logging.warn("WARNING, list_same_states has not been written yet!!!")
    return []


def set_zero_component(f_i, zero_component=None):
    if zero_component is not None:
        f_i -= f_i[zero_component]

class MBARSolver(object):
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
    def __init__(self, u_kn, N_k):
        """Create MBAR object from reduced energies.

        Parameters
        ----------
        u_in : np.ndarray, shape=(n_states, n_samples)
            Reduced potential energies in states k for samples i.
        N_k : np.ndarray, shape=(n_states)
            Number of samples taken from each thermodynamic state
        """        
        self.n_states, self.n_samples = u_kn.shape

        self.u_kn = ensure_type(u_kn, np.float64, 2, 'u_kn')
        
        self.q_ki = np.exp(-self.u_kn)
        self.q_ki /= self.q_ki.max(0)  # Divide for overflow.
        
        self.N_k = ensure_type(N_k, np.float64, 1, 'N_k', (self.n_states))
        self.log_N_k = np.log(self.N_k)
        
        self.N = self.N_k.sum()
        
        self.states = self.u_kn
        
        self.check_self_consistency()
    
    def check_self_consistency(self):
        check_same_states(self.u_kn)
        
class FixedPointMBARSolver(MBARSolver):
    def __init__(self, u_kn, N_k):
        MBARSolver.__init__(self, u_kn, N_k)

    def self_consistent_eqn_fast(self, f_i, zero_component=None):
        set_zero_component(f_i, zero_component)

        c_i = np.exp(f_i)
        denom_n = self.q_ki.T.dot(self.N_k * c_i)
        
        num = self.q_ki.dot(denom_n ** -1.)
        
        new_f_i = 1.0 * np.log(num)

        set_zero_component(new_f_i, zero_component)

        return f_i + new_f_i

    def self_consistent_eqn(self, f_i, zero_component=None):
        set_zero_component(f_i, zero_component)
      
        exp_args = self.log_N_k + f_i - self.u_kn.T
        L_n = logsumexp(exp_args, axis=1)
        
        exp_args = -L_n - self.u_kn
        q_i = logsumexp(exp_args, axis=1)
        
        set_zero_component(q_i, zero_component)
        
        return f_i + q_i

    def fixed_point_eqn(self, f_i, zero_component=None):
        set_zero_component(f_i, zero_component)    
        return self.self_consistent_eqn(f_i, zero_component=zero_component) + f_i

    def fixed_point_eqn_fast(self, f_i, zero_component=None):
        set_zero_component(f_i, zero_component)
        return self.self_consistent_eqn_fast(f_i, zero_component=zero_component) + f_i
        
    def solve(self, start=None, use_fast_first=True, zero_component=0):
        if start is None:
            f_i = np.zeros(self.n_states)

        if use_fast_first == True:
            eqn = lambda x: self.fixed_point_eqn_fast(x, zero_component=zero_component)
            f_i = scipy.optimize.fixed_point(eqn, f_i)
        
        eqn = lambda x: self.fixed_point_eqn_fast(x, zero_component=zero_component)
        f_i = scipy.optimize.fixed_point(eqn, f_i)
        
        return f_i
        

class BFGSMBARSolver(MBARSolver):
    def __init__(self, u_kn, N_k):
        MBARSolver.__init__(self, u_kn, N_k)


    def objective(self, f_i):
        F = self.N_k.dot(f_i)
        
        exp_arg = self.log_N_k + f_i - self.u_kn.T
        F -= logsumexp(exp_arg, axis=1).sum()
        return F * -1.        

    def gradient(self, f_i):   
        exp_args = self.log_N_k + f_i - self.u_kn.T
        L_n = logsumexp(exp_args, axis=1)
        
        exp_args = -L_n - self.u_kn
        q_i = logsumexp(exp_args, axis=1)
        
        grad = -1.0 * self.N_k * (1 - np.exp(f_i + q_i))
        
        return grad
        
    def solve(self, start=None, use_fast_first=True):
        if start is None:
            f_i = np.zeros(self.n_states)
        
        if use_fast_first == True:
            f_i, final_objective, convergence_parms = scipy.optimize.fmin_l_bfgs_b(self.objective_fast, f_i, self.gradient_fast, factr=1E-2, pgtol=1E-8)
        
        f_i, final_objective, convergence_parms = scipy.optimize.fmin_l_bfgs_b(self.objective, f_i, self.gradient, factr=1E-2, pgtol=1E-8)
        
        return f_i        

    def objective_fast(self, f_i):
        F = self.N_k.dot(f_i)

        c_i = np.exp(f_i)

        log_arg = self.q_ki.T.dot(self.N_k * c_i)
        F -= np.log(log_arg).sum()
        return F * -1.
        

    def gradient_fast(self, f_i):   
        c_i = np.exp(f_i)
        denom_n = self.q_ki.T.dot(self.N_k * c_i)
        
        num = self.q_ki.dot(denom_n ** -1.)

        grad = self.N_k * (1.0 - c_i * num)
        grad *= -1.

        return grad

def get_nonzero_indices(N_k):
    """Determine list of k indices for which N_k != 0"""
    nonzero_N_k_indices = np.where(N_k != 0)[0]
    nonzero_N_k_indices = nonzero_N_k_indices.astype(np.int32)
    return nonzero_N_k_indices

class MBAR(object):
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

    def __init__(self, u_kn, N_k, initial_f_k=None):

        """Initialize multistate Bennett acceptance ratio (MBAR) on a set of simulation data.

        Upon initialization, the dimensionless free energies for all states are computed.
        This may take anywhere from seconds to minutes, depending upon the quantity of data.
        After initialization, the computed free energies may be obtained by a call to 'getFreeEnergies()', or
        free energies or expectation at any state of interest can be computed by calls to 'computeFreeEnergy()' or
        'computeExpectations()'.

        Parameters
        ----------
        u_kn : np.ndarray, float, shape=(K, N)
            u_kn[k,n] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.
        N_k :  np.ndarray, int, shape=(K)
            N_k[k] is the number of uncorrelated snapshots sampled from state k
            This can be zero if the expectation or free energy of this
            state is desired but no samples were drawn from this state.
        initial_f_k : np.ndarray, float, shape=(K), optional
            Set to the initial dimensionless free energies to use as a 
            guess (default None, which sets all f_k = 0)

        Notes
        -----
        The reduced potential energy u_kln[k,n] = u_k(x_{n}), where the reduced potential energy u_k(x) is defined (as in the text) by:
        u_k(x) = beta_k [ U_k(x) + p_k V(x) + mu_k' n(x) ]
        where
        beta_k = 1/(kB T_k) is the inverse temperature of condition k, where kB is Boltzmann's constant
        U_k(x) is the potential energy function for state k
        p_k is the pressure at state k (if an isobaric ensemble is specified)
        V(x) is the volume of configuration x
        mu_k is the M-vector of chemical potentials for the various species, if a (semi)grand ensemble is specified, and ' denotes transpose
        n(x) is the M-vector of numbers of the various molecular species for configuration x, corresponding to the chemical potential components of mu_m.

        The configurations x_kn must be uncorrelated.  This can be ensured by subsampling a correlated timeseries with a period larger than the statistical inefficiency,
        which can be estimated from the potential energy timeseries {u_k(x_n)}_{n=1}^{N} using the provided utility function 'statisticalInefficiency()'.
        See the help for this function for more information.

        Examples
        --------
        To do
        """

        # Store local copies of necessary data.
        self.n_states, self.n_samples = u_kn.shape
        self.K = self.n_states

        self.u_kn = ensure_type(u_kn, np.float64, 2, 'u_kn')
        self.N_k = ensure_type(N_k, np.int, 1, 'N_k', (self.n_states))
        self.log_N_k = np.log(self.N_k)

        self.N = self.N_k.sum()  # N_k is the total number of uncorrelated configurations pooled across all states

        self.same_states = list_same_states(self.u_kn, self.N_k)

        self.nonzero_N_k_indices = get_nonzero_indices(self.N_k)

        # Store versions of variables nonzero indices file
        # Number of states with samples.
        self.K_nonzero = self.nonzero_N_k_indices.size
        logging.debug("There are %d states with samples." % self.K_nonzero)

        self.N_nonzero = self.N_k[self.nonzero_N_k_indices].copy()

        logging.debug("N_k = ")  # Print number of samples from each state.
        logging.debug(N_k)

        # Initialize estimate of relative dimensionless free energy of each state to zero.
        # Note that f_k[0] will be constrained to be zero throughout.
        # this is default
        if initial_f_k is None:       
            self.f_k = np.zeros((self.n_states), dtype=np.float64)
        else:
            logging.debug("Initializing f_k with provided initial guess.")
            self.f_k = ensure_type(initial_f_k, 'float', (1), "f_k", (self.n_states), can_be_none=True)
            logging.debug("Subtracting f_k[0]")
            self.f_k -= self.f_k[0]
            logging.debug(self.f_k)

        self.mbar_solver = FixedPointMBARSolver(u_kn, N_k)  # NEED to deal with zero count states!!!!!!
        self.f_k = self.mbar_solver.solve()

        logging.debug("Recomputing all free energies and log weights for storage")
       
        self.Log_W_nk, self.f_k = compute_log_weights(self.f_k, self.N_k, self.u_kn)

        # Print final dimensionless free energies.
        logging.debug("Final dimensionless free energies")
        logging.debug("f_k = ")
        logging.debug(self.f_k)
        logging.debug("MBAR initialization complete.")

    def _zero_same_states(self, A):
        """
        zeros out states that should be identical

        REQUIRED ARGUMENTS

        A: the matrix whose entries are to be zeroed.

        """

        for pair in self.same_states:
            A[pair[0], pair[1]] = 0
            A[pair[1], pair[0]] = 0



    def get_free_energy_differences(self, compute_uncertainty=True, uncertainty_method='svd-ew', warning_cutoff=1.0e-10, return_theta=False):
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

        """

        # Compute free energy differences.
        f_i = np.atleast_2d(self.f_k)
        
        Deltaf_ij = f_i - f_i.transpose()

        # zero out numerical error for thermodynamically identical states
        self._zero_same_states(Deltaf_ij)

        if compute_uncertainty or return_theta:
            # Compute asymptotic covariance matrix.
            Theta_ij = _compute_covariance_matrix(
                np.exp(self.Log_W_nk), self.N_k, method=uncertainty_method)

        if compute_uncertainty:
            # compute the covariance component without doing the double loop.
            # d2DeltaF = Theta_ij[i,i] + Theta_ij[j,j] - 2.0 * Theta_ij[i,j]

            diag = np.atleast_2d(Theta_ij.diagonal())
            d2DeltaF = diag + diag.transpose() - 2 * Theta_ij

            # zero out numerical error for thermodynamically identical states
            self._zero_same_states(d2DeltaF)

            # check for any numbers below zero.
            if (np.any(d2DeltaF < 0.0)):
                if(np.any(d2DeltaF) < warning_cutoff):
                    # Hmm.  Will this print correctly?
                    logging.warn("A squared uncertainty is negative.  d2DeltaF = %e" % d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)])
                else:
                    d2DeltaF[(np.any(d2DeltaF) < warning_cutoff)] = 0.0

            # take the square root of the matrix
            dDeltaf_ij = np.sqrt(d2DeltaF)
            dDeltaf_ij = np.array(d2DeltaF)
        
        if return_theta:
            return Deltaf_ij, dDeltaf_ij, Theta_ij
        elif compute_uncertainty:
            return Deltaf_ij, dDeltaf_ij
        else:
            return Deltaf_ij


    def compute_expectation(self, A_n, compute_uncertainty=True, uncertainty_method="svd-ew", warning_cutoff=1.0e-10):
        """Compute the expectation of an observable of phase space function.
         
        Compute the expectation of an observable of phase space function 
        A(x) at all K states, including states for which no samples 
        were drawn. A must NOT be a function of the state k.

        Parameters
        ----------
        A_n : np.ndarray, float, shape=(N)
            A_n = A(x_n)
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
            A_i  (K np float64 array) -  A_i[k] is the estimate for the expectation of A(x) for state k.
        dA : np.ndarray, float
            dA_i  (K np float64 array) - dA_i[k] is uncertainty estimate (one standard deviation) for A_k[k]

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

        """
        A_i, Log_W_nk_extended, N_k_extended = _compute_expectation(A_n, self.Log_W_nk, self.N_k)

        if compute_uncertainty or return_theta:
            dA_i, Theta_ij = _compute_uncertainty(Log_W_nk_extended, A_i, N_k_extended, A_n)
                
        if compute_uncertainty:
            return A_i, dA_i, Theta_ij
        else:
            return A_i


    def compute_expectation_difference(self, A_n, difference=False, compute_uncertainty=True, uncertainty_method="svd-ew", warning_cutoff=1.0e-10):
        """Compute the expectation of an observable of phase space function.
         
        Compute the expectation of an observable of phase space function 
        A(x) at all K states, including states for which no samples 
        were drawn. A must NOT be a function of the state k.

        Parameters
        ----------
        A_n : np.ndarray, float, shape=(N)
            A_n = A(x_n)
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
            A_i  (K np float64 array) -  A_i[k] is the estimate for the expectation of A(x) for state k.
        dA : np.ndarray, float
            dA_i  (K np float64 array) - dA_i[k] is uncertainty estimate (one standard deviation) for A_k[k]

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

        """
        A_i, Log_W_nk_extended, N_k_extended = _compute_expectation(A_n, self.Log_W_nk, self.N_k)

        if compute_uncertainty or return_theta:
            dA_i, Theta_ij = _compute_uncertainty(Log_W_nk_extended, A_i, N_k_extended, A_n)
            A_ij, dA_ij = _compute_expectation_differences(A_i, Theta_ij, A_n, compute_uncertainty=compute_uncertainty)
        else:
            A_ij = _compute_expectation_differences(A_i, Theta_ij, compute_uncertainty=compute_uncertainty)
        
        if compute_uncertainty:
            return A_ij, dA_ij, Theta_ij
        else:
            return A_ij

def compute_log_weights(f_k, N_k, u_kn):
    """
    Compute the normalized weights corresponding to samples for the given reduced potential.
    Also stores the all_log_denom array for reuse.


   """

    n_states = len(f_k)
    n_samples = int(sum(N_k))

    log_W_nk = np.zeros((n_samples, n_states))
    f_k_out = np.zeros(n_states, dtype=np.float64)

    log_weight_denom = compute_unnormalized_log_weights(u_kn, N_k, f_k)    

    for l in range(n_states):
        current_log_w_kn = -u_kn[l, :] + log_weight_denom + f_k[l]

        f_k_out[l] = f_k[l] - logsumexp(current_log_w_kn)
        # renormalize the weights, needed for nonzero states.
        current_log_w_kn += (f_k_out[l] - f_k[l])
        log_W_nk[:, l] = current_log_w_kn

    f_k_out = f_k_out - f_k_out[0]
    
    return log_W_nk, f_k_out

def compute_unnormalized_log_weights(u_kn, N_k, f_k):
    """Return unnormalized log weights.

    Returns
    -------
    log_w_kn : 
        (K x N np float64 array) - unnormalized log weights

    REFERENCE
      'log weights' here refers to \log [ \sum_{k=1}^K N_k exp[f_k - u_k(x_n)] ]
    """
    
    n_states = len(f_k)
    n_samples = int(sum(N_k))
    
    nonzero_N_k_indices = get_nonzero_indices(N_k)
    
    log_w_kn = np.zeros(n_samples)
    exp_arg = (np.log(N_k[nonzero_N_k_indices]) + f_k[nonzero_N_k_indices] - u_kn[nonzero_N_k_indices].T).T
    return -1.0 * logsumexp(exp_arg)


def _compute_expectation_differences(A_i, Theta_ij, A_n, compute_uncertainty=False):
    """Return differences of expectations and uncertainties."""

    n_states = len(A_i)
    K = n_states
    
    A_offset = np.min(A_n) - 1.
    A_i = A_i - A_offset    
    
    # compute expectation differences
    A_im = np.matrix(A_i)
    A_ij = A_im - A_im.transpose()

    # todo - vectorize the differences.
    if compute_uncertainty == False:
        return A_ij
    else:
        dA_ij = np.zeros([K, K], dtype=np.float64)

        for i in range(0, n_states):
            for j in range(0, n_states):
                try:
                    dA_ij[i, j] = np.sqrt(
                      + A_i[i] * Theta_ij[i, i] * A_i[i] - A_i[i] * Theta_ij[i, j] * A_i[j] - A_i[
                          i] * Theta_ij[i, K + i] * A_i[i] + A_i[i] * Theta_ij[i, K + j] * A_i[j]
                        - A_i[j] * Theta_ij[j, i] * A_i[i] + A_i[j] * Theta_ij[j, j] * A_i[j] + A_i[
                            j] * Theta_ij[j, K + i] * A_i[i] - A_i[j] * Theta_ij[j, K + j] * A_i[j]
                        - A_i[i] * Theta_ij[K + i, i] * A_i[i] + A_i[i] * Theta_ij[K + i, j] * A_i[j] + A_i[
                            i] * Theta_ij[K + i, K + i] * A_i[i] - A_i[i] * Theta_ij[K + i, K + j] * A_i[j]
                        + A_i[j] * Theta_ij[K + j, i] * A_i[i] - A_i[j] * Theta_ij[K + j, j] * A_i[j] - A_i[
                            j] * Theta_ij[K + j, K + i] * A_i[i] + A_i[j] * Theta_ij[K + j, K + j] * A_i[j]
                        )
                except ValueError:  # Not sure what exception this should be, but we need to catch a *specific* one, not all exceptions.
                    dA_ij[i, j] = 0.0
    
        A_ij = np.array(A_ij)
        return A_ij, dA_ij

def _compute_expectation(A_n, Log_W_nk, N_k):
    
    n_samples, n_states = Log_W_nk.shape
    
    Log_W_nk = ensure_type(Log_W_nk, np.float64, 2, 'Log_W_nk', shape=(n_samples, n_states))
    A_n = ensure_type(A_n, np.float64, 1, 'A_n', shape=(n_samples,))
    N_k = ensure_type(N_k, np.float64, 1, 'N_k', shape=(n_states,))

    # Augment W_nk, N_k, and c_k for q_A(x) for the observable, with one
    # extra row/column for each state (Eq. 13 of [1]).
    # log of weight matrix
    Log_W_nk_extended = np.zeros([n_samples, n_states * 2], np.float64)
    N_k_extended = np.zeros([n_states * 2], np.int32)  # counts
    # "free energies" of the new states
    f_k = np.zeros([n_states], np.float64)

    # Fill in first half of matrix with existing q_k(x) from states.
    Log_W_nk_extended[:, 0:n_states] = Log_W_nk
    N_k_extended[0:n_states] = N_k

    # Make A_n all positive so we can operate logarithmically for
    # robustness
    A_i = np.zeros([n_states], np.float64)
    A_min = np.min(A_n)
    A_n = A_n - (A_min - 1)

    # Compute the remaining rows/columns of W_nk and the rows c_k for the
    # observables.

    for l in range(n_states):
        # this works because all A_n are now positive; we took min at beginning
        Log_W_nk_extended[:, n_states + l] = np.log(A_n) + Log_W_nk[:, l]
        f_k[l] = -logsumexp(Log_W_nk_extended[:, n_states + l])
        Log_W_nk_extended[:, n_states + l] += f_k[l]              # normalize the row
        A_i[l] = np.exp(-f_k[l])

    # add back minima now now that uncertainties are computed.
    A_i += (A_min - 1)

    return A_i, Log_W_nk_extended, N_k_extended

def _compute_uncertainty(Log_W_nk_extended, A_i, N_k_extended, A_n, uncertainty_method="svd-ew"):
    n_samples, n_states = Log_W_nk_extended.shape
    n_states = n_states / 2  # Input from extended 

    Theta_ij = _compute_covariance_matrix(np.exp(Log_W_nk_extended), N_k_extended, method=uncertainty_method)
    
    A_offset = np.min(A_n) - 1.
    A_i = A_i - A_offset

    # Compute uncertainties.
    dA_i = np.zeros((n_states), np.float64)
    for k in range(n_states):
        dA_i[k] = np.abs(A_i[k]) * np.sqrt(
            Theta_ij[n_states + k, n_states + k] + Theta_ij[k, k] - 2.0 * Theta_ij[k, n_states + k])  # Eq. 16 of [1]

    return dA_i, Theta_ij

def _compute_covariance_matrix(W, N_k, method='svd-ew', pinv_tol=1E-12):
    """Compute estimate of the asymptotic covariance matrix.

    Parameters
    ----------
    W : np.ndarray, float, shape=(N, K)
        Normalized weights (see Eq. 9 of [1]) - W[n,k] is the weight of snapshot n (n = 1..N) in state k
        Note that sum(W(:,k)) = 1 for any k = 1..K, and sum(N_k(:) .* W(n,:)) = 1 for any n.
    N_k : np.ndarray, int, shape=(K)
        N_k[k] is the number of samples from state K
    method : string, optional
        if not None, specified method is used to compute asymptotic covariance method:
        method must be one of ['generalized-inverse', 'svd', 'svd-ew', 'inverse', 'tan-HGH', 'tan', 'approximate']
        If None is specified, 'svd-ew' is used.
    
    Returns
    -------
    Theta : np.ndarray, float, shape=(K, K)
        Asymptotic covariance matrix (see Eq. 8 of [1])

    Notes
    -----

    The computational costs of the various 'method' arguments varies:

    'generalized-inverse' currently requires computation of the pseudoinverse of an NxN matrix (where N is the total number of samples)
    'svd' computes the generalized inverse using the singular value decomposition -- this should be efficient yet accurate (faster)
    'svd-ev' is the same as 'svd', but uses the eigenvalue decomposition of W'W to bypass the need to perform an SVD (fastest)
    'inverse' only requires standard inversion of a KxK matrix (where K is the number of states), but requires all K states to be different
    'approximate' only requires multiplication of KxN and NxK matrices, but is an approximate underestimate of the uncertainty
    'tan' uses a simplified form that requires two pseudoinversions, but can be unstable
    'tan-HGH' makes weaker assumptions on 'tan' but can occasionally be unstable

    References
    ----------
    
    See Section II and Appendix D of [1].

    """

    [N, K] = W.shape

    W = ensure_type(W, np.float64, 2, 'W')
    N_k = ensure_type(N_k, np.int, 1, "N_k", shape=(K,))

    if(np.sum(N_k) != N):
        raise ParameterError('W must be NxK, where N = sum_k N_k.')

    # Check to make sure the weight matrix W is properly normalized.
    validate_weight_matrix(W, N_k)

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
        Theta = W.T * np.linalg.pinv(I - W * Ndiag * W.T) * W

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
        if (abs(np.linalg.det(W.T * W)) < tolerance):
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
        [U, S, Vt] = np.linalg.svd(W)
        Sigma = np.matrix(np.diag(S))
        V = np.matrix(Vt).T

        # Compute covariance
        Theta = V * Sigma * np.linalg.pinv(
            I - Sigma * V.T * Ndiag * V * Sigma, rcond=pinv_tol) * Sigma * V.T

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
        [S2, V] = np.linalg.eigh(W.T * W)
        # Set any slightly negative eigenvalues to zero.
        S2[np.where(S2 < 0.0)] = 0.0
        # Form matrix of singular values Sigma, and V.
        Sigma = np.matrix(np.diag(np.sqrt(S2)))
        V = np.matrix(V)

        # Compute covariance
        Theta = V * Sigma * np.linalg.pinv(
            I - Sigma * V.T * Ndiag * V * Sigma, rcond=pinv_tol) * Sigma * V.T

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
        Hinv = np.linalg.pinv(H)

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
        Oinv = np.linalg.pinv(O)
        Theta = np.linalg.pinv(Oinv - Lambda)

    else:
        # Raise an exception.
        raise ParameterError('Method ' + method + ' unrecognized.')

    return Theta
