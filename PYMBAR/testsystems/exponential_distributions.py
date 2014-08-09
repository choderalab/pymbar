import numpy as np
from pymbar.utils import ensure_type


class ExponentialTestCase(object):

    """Test cases using exponential distributions.

    Examples
    --------

    Generate energy samples with default parameters.

    >>> testcase = ExponentialTestCase()
    >>> [x_kn, u_kln, N_k] = testcase.sample()

    Retrieve analytical properties.

    >>> analytical_means = testcase.analytical_means()
    >>> analytical_variances = testcase.analytical_variances()
    >>> analytical_standard_deviations = testcase.analytical_standard_deviations()
    >>> analytical_free_energies = testcase.analytical_free_energies()
    >>> analytical_x_squared = testcase.analytical_x_squared()

    Generate energy samples with default parameters in one line.

    >>> [x_kn, u_kln, N_k] = ExponentialTestCase().sample()

    Generate energy samples with specified parameters.

    >>> testcase = ExponentialTestCase(rates=[1., 2., 3., 4., 5.])
    >>> [x_kn, u_kln, N_k] = testcase.sample(N_k=[10, 20, 30, 40, 50])

    Test sampling in different output modes.

    >>> [x_kn, u_kln, N_k] = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kln')
    >>> [x_n, u_kn, N_k] = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kn')

    """

    def __init__(self, rates=[1, 2, 3, 4, 5], beta=1.0):
        """Generate test case with exponential distributions.

        Parameters
        ----------
        rates : np.ndarray, float, shape=(n_states)
            Rate parameters (e.g. lambda) for each state.

        beta : float, optional, default=1.0
            Inverse temperature.

        Notes
        -----
        We assume potentials of the form U(x) = lambda x.
        """
        rates = np.array(rates, np.float64)

        self.n_states = len(rates)
        self.rates = rates

        self.beta = beta

    def analytical_means(self):
        return self.rates ** -1.

    def analytical_variances(self):
        return self.rates ** -2.

    def analytical_standard_deviations(self):
        return np.sqrt(self.rates ** -2.)

    def analytical_free_energies(self):
        """Return the FE: -log(Z)"""
        return np.log(self.rates)

    def analytical_x_squared(self):
        return self.analytical_variances() + self.analytical_means() ** 2.

    def sample(self, N_k=[10, 20, 30, 40, 50], mode='u_kln'):
        """Draw samples from the distribution.

        Parameters
        ----------

        N_k : np.ndarray, int
            number of samples per state
        mode : str, optional, default='u_kln'
            If 'u_kln', return K x K x N_max matrix where u_kln[k,l,n] is reduced potential of sample n from state k evaluated at state l.
            If 'u_kn', return K x N_tot matrix where u_kn[k,n] is reduced potential of sample n (in concatenated indexing) evaluated at state k.

        Returns
        -------
        if mode == 'u_kn':

        x_n : np.ndarray, shape=(n_states*n_samples), dtype=float
           x_n[n] is sample n (in concatenated indexing)
        u_kn : np.ndarray, shape=(n_states, n_states*n_samples), dtype=float
           u_kn[k,n] is reduced potential of sample n (in concatenated indexing) evaluated at state k.
        N_k : np.ndarray, shape=(n_states), dtype=float
           N_k[k] is the number of samples generated from state k

        x_kn : np.ndarray, shape=(n_states, n_samples), dtype=float
            1D harmonic oscillator positions
        u_kln : np.ndarray, shape=(n_states, n_states, n_samples), dytpe=float, only if mode='u_kln'
           u_kln[k,l,n] is reduced potential of sample n from state k evaluated at state l.
        N_k : np.ndarray, shape=(n_states), dtype=float
           N_k[k] is the number of samples generated from state k

        """
        N_k = np.array(N_k, np.float64)
        if len(N_k) != self.n_states:
            raise Exception("N_k has %d states while self.n_states has %d states." % (len(N_k), self.n_states))

        states = ["state %d" % k for k in range(self.n_states)]

        N_max = N_k.max()  # maximum number of samples per state
        N_tot = N_k.sum()  # total number of samples

        x_kn = np.zeros([self.n_states, N_max], np.float64)
        u_kln = np.zeros([self.n_states, self.n_states, N_max], np.float64)
        x_n = np.zeros([N_tot], np.float64)
        u_kn = np.zeros([self.n_states, N_tot], np.float64)
        index = 0
        for k, N in enumerate(N_k):
            x = np.random.exponential(scale=self.rates[k] ** -1., size=N)
            x_kn[k, 0:N] = x
            x_n[index:(index + N)] = x
            for l in range(self.n_states):
                u = self.beta * self.rates[l] * x
                u_kln[k, l, 0:N] = u
                u_kn[l, index:(index + N)] = u
            index += N

        if (mode == 'u_kn'):
            return x_n, u_kn, N_k
        elif (mode == 'u_kln'):
            return x_kn, u_kln, N_k
        else:
            raise Exception("Unknown mode '%s'" % mode)

        return
