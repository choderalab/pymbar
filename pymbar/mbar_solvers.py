from __future__ import division  # Ensure same division behavior in py2 and py3
import numpy as np
import math
import scipy.optimize
from pymbar.utils import ensure_type, logsumexp, check_w_normalized
import warnings

# Below are the recommended default protocols (ordered sequence of minimization algorithms / NLE solvers) for solving the MBAR equations.
# Note: we use tuples instead of lists to avoid accidental mutability.
DEFAULT_SUBSAMPLING_PROTOCOL = (dict(method="L-BFGS-B"), )  # First use BFGS on subsampled data.
DEFAULT_SOLVER_PROTOCOL = (dict(method="hybr"), )  # Then do fmin hybrid on full dataset.


def validate_inputs(u_kn, N_k, f_k):
    """Check types and return inputs for MBAR calculations.

    Parameters
    ----------
    u_kn or q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies or unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    u_kn or q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies or unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='float'
        The number of samples in each state.  Converted to float because this cast is required when log is calculated.
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state
    """
    n_states, n_samples = u_kn.shape

    u_kn = ensure_type(u_kn, 'float', 2, "u_kn or Q_kn", shape=(n_states, n_samples))
    N_k = ensure_type(N_k, 'float', 1, "N_k", shape=(n_states,), warn_on_cast=False)  # Autocast to float because will be eventually used in float calculations.
    f_k = ensure_type(f_k, 'float', 1, "f_k", shape=(n_states,))

    return u_kn, N_k, f_k


def self_consistent_update(u_kn, N_k, f_k):
    """Return an improved guess for the dimensionless free energies

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    f_k : np.ndarray, shape=(n_states), dtype='float'
        Updated estimate of f_k

    Notes
    -----
    Equation C3 in MBAR JCP paper.
    """

    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)
    
    states_with_samples = (N_k > 0)

    # Only the states with samples can contribute to the denominator term.
    log_denominator_n = logsumexp(f_k[states_with_samples] - u_kn[states_with_samples].T, b=N_k[states_with_samples], axis=1)
    
    # All states can contribute to the numerator term.
    return -1. * logsumexp(-log_denominator_n - u_kn, axis=1)



def mbar_gradient(u_kn, N_k, f_k):
    """Gradient of MBAR objective function.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of mbar_objective

    Notes
    -----
    This is equation C6 in the JCP MBAR paper.
    """
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    log_numerator_k = logsumexp(-log_denominator_n - u_kn, axis=1)
    return -1 * N_k * (1.0 - np.exp(f_k + log_numerator_k))


def mbar_objective_and_gradient(u_kn, N_k, f_k):
    """Calculates both objective function and gradient for MBAR.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state


    Returns
    -------
    obj : float
        Objective function
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of objective function

    Notes
    -----
    This objective function is essentially a doubly-summed partition function and is
    quite sensitive to precision loss from both overflow and underflow. For optimal
    results, u_kn can be preconditioned by subtracting out a `n` dependent
    vector.

    More optimal precision, the objective function uses math.fsum for the
    outermost sum and logsumexp for the inner sum.
    
    The gradient is equation C6 in the JCP MBAR paper; the objective
    function is its integral.
    """
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    log_numerator_k = logsumexp(-log_denominator_n - u_kn, axis=1)
    grad = -1 * N_k * (1.0 - np.exp(f_k + log_numerator_k))

    obj = math.fsum(log_denominator_n) - N_k.dot(f_k)

    return obj, grad


def mbar_hessian(u_kn, N_k, f_k):
    """Hessian of MBAR objective function.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    H : np.ndarray, dtype=float, shape=(n_states, n_states)
        Hessian of mbar objective function.

    Notes
    -----
    Equation (C9) in JCP MBAR paper.
    """
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)

    W = mbar_W_nk(u_kn, N_k, f_k)

    H = W.T.dot(W)
    H *= N_k
    H *= N_k[:, np.newaxis]
    H -= np.diag(W.sum(0) * N_k)

    return -1.0 * H


def mbar_log_W_nk(u_kn, N_k, f_k):
    """Calculate the log weight matrix.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    logW_nk : np.ndarray, dtype='float', shape=(n_samples, n_states)
        The normalized log weights.

    Notes
    -----
    Equation (9) in JCP MBAR paper.
    """
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    logW = f_k - u_kn.T - log_denominator_n[:, np.newaxis]
    return logW

def mbar_W_nk(u_kn, N_k, f_k):
    """Calculate the weight matrix.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    W_nk : np.ndarray, dtype='float', shape=(n_samples, n_states)
        The normalized weights.

    Notes
    -----
    Equation (9) in JCP MBAR paper.
    """
    return np.exp(mbar_log_W_nk(u_kn, N_k, f_k))


def precondition_u_kn(u_kn, N_k, f_k):
    """Subtract a sample-dependent constant from u_kn to improve precision

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities

    Notes
    -----
    Returns u_kn - x_n, where x_n is based on the current estimate of f_k.
    Upon subtraction of x_n, the MBAR objective function changes by an
    additive constant, but its derivatives remain unchanged.  We choose
    x_n such that the current objective function value is zero, which
    should give maximum precision in the objective function.
    """
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)
    u_kn = u_kn - u_kn.min(0)
    u_kn += (logsumexp(f_k - u_kn.T, b=N_k, axis=1)) - N_k.dot(f_k) / float(N_k.sum())
    return u_kn


def solve_mbar_once(u_kn_nonzero, N_k_nonzero, f_k_nonzero, method="hybr", tol=1E-20, options=None):
    """Solve MBAR self-consistent equations using some form of equation solver.

    Parameters
    ----------
    u_kn_nonzero : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
        for the nonempty states
    N_k_nonzero : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state for the nonempty states
    f_k_nonzero : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies for the nonempty states
    method : str, optional, default="hybr"
        The optimization routine to use.  This can be any of the methods
        available via scipy.optimize.minimize() or scipy.optimize.root().
    tol : float, optional, default=1E-20
        The convergance tolerance for minimize() or root()
    options: dict, optional, default=None
        Optional dictionary of algorithm-specific parameters.  See
        scipy.optimize.root or scipy.optimize.minimize for details.

    Returns
    -------
    f_k : np.ndarray
        The converged reduced free energies.
    results : dict
        Dictionary containing entire results of optimization routine, may
        be useful when debugging convergence.

    Notes
    -----
    This function requires that N_k_nonzero > 0--that is, you should have
    already dropped all the states for which you have no samples.
    Internally, this function works in a reduced coordinate system defined
    by subtracting off the first component of f_k and fixing that component
    to be zero.

    For fast but precise convergence, we recommend calling this function
    multiple times to polish the result.  `solve_mbar()` facilitates this.
    """
    u_kn_nonzero, N_k_nonzero, f_k_nonzero = validate_inputs(u_kn_nonzero, N_k_nonzero, f_k_nonzero)
    f_k_nonzero = f_k_nonzero - f_k_nonzero[0]  # Work with reduced dimensions with f_k[0] := 0
    u_kn_nonzero = precondition_u_kn(u_kn_nonzero, N_k_nonzero, f_k_nonzero)

    pad = lambda x: np.pad(x, (1, 0), mode='constant')  # Helper function inserts zero before first element
    unpad_second_arg = lambda obj, grad: (obj, grad[1:])  # Helper function drops first element of gradient

    # Create objective functions / nonlinear equations to send to scipy.optimize, fixing f_0 = 0
    grad = lambda x: mbar_gradient(u_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Objective function gradient
    grad_and_obj = lambda x: unpad_second_arg(*mbar_objective_and_gradient(u_kn_nonzero, N_k_nonzero, pad(x)))  # Objective function gradient and objective function
    hess = lambda x: mbar_hessian(u_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Hessian of objective function

    with warnings.catch_warnings(record=True) as w:
        if method in ["L-BFGS-B", "dogleg", "CG", "BFGS", "Newton-CG", "TNC", "trust-ncg", "SLSQP"]:
            if method in ["L-BFGS-B", "CG"]:
                hess = None  # To suppress warning from passing a hessian function.
            results = scipy.optimize.minimize(grad_and_obj, f_k_nonzero[1:], jac=True, hess=hess, method=method, tol=tol, options=options)
        else:
            results = scipy.optimize.root(grad, f_k_nonzero[1:], jac=hess, method=method, tol=tol, options=options)

    f_k_nonzero = pad(results["x"])

    #If there were runtime warnings, show the messages
    if len(w) > 0:
        for warn_msg in w:
            warnings.showwarning(warn_msg.message, warn_msg.category, warn_msg.filename, warn_msg.lineno, warn_msg.file, "") 
        #Ensure MBAR solved correctly
        W_nk_check = mbar_W_nk(u_kn_nonzero, N_k_nonzero, f_k_nonzero)
        check_w_normalized(W_nk_check, N_k_nonzero)
        print("MBAR weights converged within tolerance, despite the SciPy Warnings. Please validate your results.")
       
            
    return f_k_nonzero, results


def solve_mbar(u_kn_nonzero, N_k_nonzero, f_k_nonzero, solver_protocol=None, verbose=False):
    """Solve MBAR self-consistent equations using some sequence of equation solvers.

    Parameters
    ----------
    u_kn_nonzero : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
        for the nonempty states
    N_k_nonzero : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state for the nonempty states
    f_k_nonzero : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies for the nonempty states
    solver_protocol: tuple(dict()), optional, default=None
        Optional list of dictionaries of steps in solver protocol.
        If None, a default protocol will be used.

    Returns
    -------
    f_k : np.ndarray
        The converged reduced free energies.
    all_results : list(dict())
        List of results from each step of solver_protocol.  Each element in
        list contains the results dictionary from solve_mbar_single()
        for the corresponding step.

    Notes
    -----
    This function requires that N_k_nonzero > 0--that is, you should have
    already dropped all the states for which you have no samples.
    Internally, this function works in a reduced coordinate system defined
    by subtracting off the first component of f_k and fixing that component
    to be zero.

    This function calls `solve_mbar_once()` multiple times to achieve
    converged results.  Generally, a single call to solve_mbar_single()
    will not give fully converged answers because of limited numerical precision.
    Each call to `solve_mbar_once()` re-conditions the nonlinear
    equations using the current guess.
    """
    if solver_protocol is None:
        solver_protocol = DEFAULT_SOLVER_PROTOCOL

    all_results = []
    for k, options in enumerate(solver_protocol):
        f_k_nonzero, results = solve_mbar_once(u_kn_nonzero, N_k_nonzero, f_k_nonzero, **options)
        all_results.append(results)
    
    if verbose:
        print(("Final gradient norm: %.3g" % np.linalg.norm(mbar_gradient(u_kn_nonzero, N_k_nonzero, f_k_nonzero))))

    return f_k_nonzero, all_results


def subsample_data(u_kn0, N_k0, s_n, subsampling, rescale=False, replace=False):
    """Return a subsample from dataset.

    Parameters
    ----------
    u_kn0 : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k0 : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    s_n : np.ndarray, shape=(n_samples), dtype='int'
        State of origin of each sample x_n
    subsampling : int
        The factor by which to subsample (E.g. 10 for 10X).
    rescale : bool, optional, default=True
        If True, rescale and shift the subset to have same mean and variance
        as full dataset
    replace : bool, optional, default=False
        Subsample with replacement

    Returns
    -------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='float'
        The new number of samples in each state

    Notes
    -----
    In situations where N >> K and the overlap is good, one might use
    subsampling to solve MBAR on a smaller dataset as an initial guess.
    """    
    n_states = len(N_k0)
    N_k = N_k0 // subsampling
    N_k[(N_k == 0) & (N_k0 > 0)] = 1

    u_kn = np.zeros((n_states, N_k.sum()))

    if rescale:
        mu_k = np.array([u_kn0[:, s_n == k].mean(1) for k in range(n_states)])
        sigma_k = np.array([u_kn0[:, s_n == k].std(1) for k in range(n_states)])
        standardize = lambda x: (x - x.mean(1)[:, np.newaxis]) / x.std(1)[:, np.newaxis]
    else:
        mu_k = np.zeros((n_states, n_states))
        sigma_k = np.ones((n_states, n_states))
        standardize = lambda x: x

    start = 0
    for k in range(n_states):
        if N_k[k] <= 0:
            continue
        samples = np.random.choice(np.where(s_n == k)[0], size=(N_k[k].astype(int)), replace=replace)
        u_k = standardize(u_kn0[:, samples]) * sigma_k[k][:, np.newaxis] + mu_k[k][:, np.newaxis]
        num = N_k[k]
        u_kn[:, start:start + num] = u_k
        start += num

    return u_kn, N_k


def solve_mbar_with_subsampling(u_kn, N_k, f_k, solver_protocol, subsampling_protocol, subsampling, x_kindices=None):
    """Solve for free energies of states with samples, then calculate for
    empty states.  Optionally uses subsampling as a hot-start to speed up
    calculations.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. -log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state
    solver_protocol: tuple(dict()), optional, default=None
        Sequence of dictionaries of steps in solver protocol for final
        stage of refinement.
    subsampling_protocol: tuple(dict()), optional, default=None
        Sequence of dictionaries of steps in solver protocol for first
        stage of refinement with subsampled dataset.
    subsampling : int
        By what factor do we subsample the dataset for getting a first
        pass solution to MBAR.
    x_kindices : np.ndarray, optional, shape=(N_samples), dtype='int'
        The stage of origin for each sample.  This is required to use
        subsampling to use a fast guess as way to hot start and accelerate
        MBAR.

    Returns
    -------
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The free energies of states
    
    
    """
    states_with_samples = np.where(N_k > 0)[0]

    if len(states_with_samples) == 1:
        f_k_nonzero = np.array([0.0])
    else:
        if subsampling is not None and x_kindices is not None and subsampling > 1:
            s_n = np.unique(x_kindices, return_inverse=True)[1]
            u_kn_subsampled, N_k_subsampled = subsample_data(u_kn[states_with_samples], N_k[states_with_samples], s_n, subsampling=subsampling)
            f_k_nonzero, all_results = solve_mbar(u_kn_subsampled, N_k_subsampled, f_k[states_with_samples], solver_protocol=subsampling_protocol)
        else:
            f_k_nonzero, all_results = solve_mbar(u_kn[states_with_samples], N_k[states_with_samples], f_k[states_with_samples], solver_protocol=subsampling_protocol)

        f_k[states_with_samples] = f_k_nonzero
        f_k_nonzero, all_results = solve_mbar(u_kn[states_with_samples], N_k[states_with_samples], f_k[states_with_samples], solver_protocol=solver_protocol)
    
    f_k[states_with_samples] = f_k_nonzero

    # Update all free energies because those from states with zero samples are not correctly computed by Newton-Raphson.
    f_k = self_consistent_update(u_kn, N_k, f_k)
    f_k -= f_k[0]  # This is necessary because state 0 might have had zero samples, but we still want that state to be the reference with free energy 0.

    return f_k
