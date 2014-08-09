import numpy as np
import math
import scipy.optimize
from pymbar.utils import ensure_type


try:  # numexpr used in logsumexp when available.
    import numexpr
    HAVE_NUMEXPR = True
except ImportError:
    HAVE_NUMEXPR = False

# This is the default protocol (ordered list of minimization algorithms / NLE solvers) for solving the MBAR equations.
DEFAULT_SOLVER_PROTOCOL = [dict(method="L-BFGS-B", fast=True), dict(method="L-BFGS-B", fast=True), dict(method="hybr", fast=True), dict(method="hybr")]


def logsumexp(a, axis=None, b=None, use_numexpr=True):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int, optional, default=None
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`.
    use_numexpr : bool, optional, default=True
        If True, use the numexpr library to speed up the calculation, which
        can give a 2-4X speedup when working with large arrays.

    Returns
    -------
    res : ndarray
        The result, ``log(sum(exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``log(sum(b*exp(a)))``
        is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2, scipy.misc.logsumexp

    Notes
    -----
    This is based on scipy.misc.logsumexp but with optional numexpr
    support for improved performance.
    """

    a = np.asarray(a)

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        if use_numexpr and HAVE_NUMEXPR:
            out = np.log(numexpr.evaluate("b * exp(a - a_max)").sum(axis))
        else:
            out = np.log(np.sum(b * np.exp(a - a_max), axis=axis))
    else:
        if use_numexpr and HAVE_NUMEXPR:
            out = np.log(numexpr.evaluate("exp(a - a_max)").sum(axis))
        else:
            out = np.log(np.sum(np.exp(a - a_max), axis=axis))

    a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out


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

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    return -1. * logsumexp(-log_denominator_n - u_kn, axis=1)


def mbar_gradient_fast(Q_kn, N_k, f_k):
    """Gradient of mbar_objective()

    Parameters
    ----------
    Q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        Unnormalized probabilities.  Q_kn = exp(-u_kn)
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of objective function

    Notes
    -----
    This is equation C6 in the JCP MBAR paper.

    This "fast" version works by performing matrix operations using Q_kn.
    This may have slightly reduced precision as compared to the non-fast
    version, which uses logsumexp operations instead of matrix
    multiplication.
    """
    Q_kn, N_k, f_k = validate_inputs(Q_kn, N_k, f_k)

    c_k_inv = np.exp(f_k)
    denominator_n = Q_kn.T.dot(N_k * c_k_inv)

    numerator_k = Q_kn.dot(denominator_n ** -1.)

    grad = N_k * (1.0 - c_k_inv * numerator_k)
    grad *= -1.

    return grad


def mbar_objective_and_gradient_fast(Q_kn, N_k, f_k):
    """Calculates both objective function and gradient for MBAR.

    Parameters
    ----------
    Q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        Unnormalized probabilities.  Q_kn = exp(-u_kn)
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

    This "fast" version works by performing matrix operations using Q_kn.
    This may have slightly reduced precision as compared to the non-fast
    version, which uses logsumexp operations instead of matrix
    multiplication.
    
    The gradient is equation C6 in the JCP MBAR paper; the objective
    function is its integral.    
    """
    Q_kn, N_k, f_k = validate_inputs(Q_kn, N_k, f_k)

    c_k_inv = np.exp(f_k)
    denominator_n = Q_kn.T.dot(N_k * c_k_inv)

    numerator_k = Q_kn.dot(denominator_n ** -1.)

    grad = N_k * (1.0 - c_k_inv * numerator_k)
    grad *= -1.

    obj = -N_k.dot(f_k) + np.log(denominator_n).sum()

    return obj, grad


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
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    W = np.exp(f_k - u_kn.T - log_denominator_n[:, np.newaxis])
    return W


def mbar_hessian_fast(Q_kn, N_k, f_k):
    """Hessian of MBAR objective function (fast).

    Parameters
    ----------
    Q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        Unnormalized probabilities.  Q_kn = exp(-u_kn)
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

    This "fast" version works by performing matrix operations using Q_kn.
    This may have slightly reduced precision as compared to the non-fast
    version, which uses logsumexp operations instead of matrix
    multiplication.
    """
    Q_kn, N_k, f_k = validate_inputs(Q_kn, N_k, f_k)

    W = mbar_W_nk_fast(Q_kn, N_k, f_k)

    H = W.T.dot(W)
    H *= N_k
    H *= N_k[:, np.newaxis]
    H -= np.diag(W.sum(0) * N_k)

    return -1.0 * H


def mbar_W_nk_fast(Q_kn, N_k, f_k):
    """Calculate the weight matrix.

    Parameters
    ----------
    Q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        Unnormalized probabilities.  Q_kn = exp(-u_kn)
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
    Q_kn, N_k, f_k = validate_inputs(Q_kn, N_k, f_k)

    c_k_inv = np.exp(f_k)
    denominator_n = Q_kn.T.dot(N_k * c_k_inv)
    return c_k_inv * Q_kn.T / denominator_n[:, np.newaxis]


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
    should give maximal precision in the objective function.
    """
    u_kn, N_k, f_k = validate_inputs(u_kn, N_k, f_k)
    u_kn = u_kn - u_kn.min(0)
    u_kn += (logsumexp(f_k - u_kn.T, b=N_k, axis=1)) - N_k.dot(f_k) / float(N_k.sum())
    return u_kn


def solve_mbar_once(u_kn_nonzero, N_k_nonzero, f_k_nonzero, fast=False, method="hybr", tol=1E-20, options=None):
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
    fast : bool, optional, default=False
        If true, use matrix-vector products of Q_kn rather than logsumexp
        operations on u_kn.  This increases speed but reduces precision.
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
    if not fast:
        grad = lambda x: mbar_gradient(u_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Objective function gradient
        grad_and_obj = lambda x: unpad_second_arg(*mbar_objective_and_gradient(u_kn_nonzero, N_k_nonzero, pad(x)))  # Objective function gradient
        hess = lambda x: mbar_hessian(u_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Hessian of objective function
        eqns = grad
        jac = hess
    else:
        Q_kn_nonzero = np.exp(-u_kn_nonzero)
        grad = lambda x: mbar_gradient_fast(Q_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Objective function gradient
        grad_and_obj = lambda x: unpad_second_arg(*mbar_objective_and_gradient_fast(Q_kn_nonzero, N_k_nonzero, pad(x)))  # Objective function gradient
        hess = lambda x: mbar_hessian_fast(Q_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Hessian of objective function
        eqns = grad
        jac = hess

    if method in ["L-BFGS-B", "dogleg", "CG", "BFGS", "Newton-CG", "TNC", "trust-ncg", "SLSQP"]:
        results = scipy.optimize.minimize(grad_and_obj, f_k_nonzero[1:], jac=True, hess=hess, method=method, tol=tol, options=options)
    else:
        results = scipy.optimize.root(eqns, f_k_nonzero[1:], jac=jac, method=method, tol=tol, options=options)

    f_k_nonzero = pad(results["x"])
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
    solver_protocol: list(dict()), optional, default=None
        Optional list of dictionaries of steps in solver protocol.
        If None, the default protocol of [dict(method="L-BFGS-B", fast=True),
        dict(method="L-BFGS-B", fast=True),
        dict(method="hybr", fast=True), dict(method="hybr")]
        will be used.  This achieves optimal precision and convergence in
        the minimium amount of time.

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
        print("Final gradient norm: %.3g" % np.linalg.norm(mbar_gradient(u_kn_nonzero, N_k_nonzero, f_k_nonzero)))

    return f_k_nonzero, all_results
