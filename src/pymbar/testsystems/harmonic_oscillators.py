import numpy as np
from mdtraj.utils import ensure_type

DEFAULT_N_k = np.array([100., 100., 100.])
DEFAULT_O_k = np.array([  0.,   1.,   2.])
DEFAULT_K_k = np.array([  1.,   1.,   1.])


def harmonic_oscillators_example(N_k=DEFAULT_N_k, O_k=DEFAULT_O_k, K_k=DEFAULT_K_k, seed=None):
    """Generate samples from 1D harmonic oscillators with specified relative spacing (in units of std devs).

    Parameters
    ----------
    N_k : np.ndarray, int
        number of samples per state
    O_k : np.ndarray, float
        offsets of the harmonic oscillators in dimensionless units
    K_k : np.ndarray, float
        force constants of harmonic oscillators in dimensionless units
    seed : int, optional
        If not None, specify the numpy random number seed.

    Returns
    -------
    x_kn : np.ndarray, shape=(n_states, n_samples), dtype=float
        1D harmonic oscillator positions
    u_kln : np.ndarray, shape=(n_states, n_states, n_samples), dtype=float
        reduced potential
    N_k : np.ndarray, shape=(n_states), dtype=float
        number of samples per state

    Notes
    -----

    Examples
    --------

    Generate energy samples with default parameters.

    >>> [x_kn, u_kln, N_k] = harmonic_oscillators_example()

    Specify number of samples, specify the states of the harmonic oscillators

    >>> [x_kn, u_kln, N_k] = harmonic_oscillators_example(N_k=[10, 20, 30, 40, 50], O_k=[0, 1, 2, 3, 4], K_k=[1, 2, 4, 8, 16])

    """

    N_k = ensure_type(N_k, np.float64, 1, 'N_k')

    n_states = len(N_k)

    O_k = ensure_type(O_k, np.float64, 1, 'O_k', n_states)
    K_k = ensure_type(K_k, np.float64, 1, 'K_k', n_states)

    # Determine maximum number of samples.
    Nmax = N_k.max()

    if seed is not None:
        np.random.seed(seed)

    # calculate the standard deviation induced by the spring constants.
    sigma_k = (K_k) ** -0.5

    # generate space to store the energies
    u_kln = np.zeros([n_states, n_states, Nmax], np.float64)

    # Generate position samples.
    x_kn = np.zeros([n_states, Nmax], np.float64)
    for k in xrange(n_states):
        x_kn[k, 0:N_k[k]] = np.random.normal(O_k[k], sigma_k[k], N_k[k])
        for l in xrange(n_states):
            u_kln[k, l, 0:N_k[k]] = (K_k[l] / 2.0) * (x_kn[k, 0:N_k[k]] - O_k[l]) ** 2

    return x_kn, u_kln, N_k
