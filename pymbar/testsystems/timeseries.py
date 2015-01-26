import numpy as np


def correlated_timeseries_example(N=10000, tau=5.0, seed=None):
    """Generate synthetic timeseries data with known correlation time.

    Parameters
    ----------
    N : int, optional
        length (in number of samples) of timeseries to generate
    tau : float, optional
        correlation time (in number of samples) for timeseries
    seed : int, optional
        If not None, specify the numpy random number seed.

    Returns
    -------
    dih : np.ndarray, shape=(num_dihedrals), dtype=float
        dih[i,j] gives the dihedral angle at traj[i] correponding to indices[j].

    Notes
    -----

    Synthetic timeseries generated using bivariate Gaussian process described
    by Janke (Eq. 41 of Ref. [1]).

    As noted in Eq. 45-46 of Ref. [1], the true integrated autocorrelation time will be given by
    tau_int = (1/2) coth(1 / 2 tau) = (1/2) (1+rho)/(1-rho)
    which, for tau >> 1, is approximated by
    tau_int = tau + 1/(12 tau) + O(1/tau^3)
    So for tau >> 1, tau_int is approximately the given exponential tau.

    References
    ----------
    .. [1] Janke W. Statistical analysis of simulations: Data correlations and error estimation.  In 'Quantum Simulations of Complex Many-Body Systems: From Theory to Algorithms'. NIC Series, VOl. 10, pages 423-445, 2002.

    Examples
    --------

    Generate a timeseries of length 10000 with correlation time of 10.

    >>> A_t = correlated_timeseries_example(N=10000, tau=10.0)

    Generate an uncorrelated timeseries of length 1000.

    >>> A_t = correlated_timeseries_example(N=1000, tau=1.0)

    Generate a correlated timeseries with correlation time longer than the length.

    >>> A_t = correlated_timeseries_example(N=1000, tau=2000.0)

    """

    # Set random number generator into a known state for reproducibility.
    random = np.random.RandomState(seed)

    # Compute correlation coefficient rho, 0 <= rho < 1.
    rho = np.exp(-1.0 / tau)
    sigma = np.sqrt(1.0 - rho * rho)

    # Generate uncorrelated Gaussian variates.
    e_n = random.randn(N)

    # Generate correlated signal from uncorrelated Gaussian variates using correlation coefficient.
    # NOTE: This will be slow.
    # TODO: Can we speed this up using vector operations?
    A_n = np.zeros([N], np.float32)
    A_n[0] = e_n[0]
    for n in range(1, N):
        A_n[n] = rho * A_n[n - 1] + sigma * e_n[n]

    return A_n
