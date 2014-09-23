import numpy as np
from pymbar.utils import ensure_type


class HarmonicOscillatorsTestCase(object):

    """Test cases using harmonic oscillators.

    Examples
    --------

    Generate energy samples with default parameters.

    >>> testcase = HarmonicOscillatorsTestCase()
    >>> [x_kn, u_kln, N_k] = testcase.sample()

    Retrieve analytical properties.

    >>> analytical_means = testcase.analytical_means()
    >>> analytical_variances = testcase.analytical_variances()
    >>> analytical_standard_deviations = testcase.analytical_standard_deviations()
    >>> analytical_free_energies = testcase.analytical_free_energies()
    >>> analytical_x_squared = testcase.analytical_x_squared()

    Generate energy samples with default parameters in one line.

    >>> [x_kn, u_kln, N_k] = HarmonicOscillatorsTestCase().sample()

    Generate energy samples with specified parameters.

    >>> testcase = HarmonicOscillatorsTestCase(O_k=[0, 1, 2, 3, 4], K_k=[1, 2, 4, 8, 16])
    >>> [x_kn, u_kln, N_k] = testcase.sample(N_k=[10, 20, 30, 40, 50])

    Test sampling in different output modes.

    >>> [x_kn, u_kln, N_k] = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kln')
    >>> [x_n, u_kn, N_k] = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kn')

    """

    def __init__(self, O_k=[0, 1, 2, 3, 4], K_k=[1, 2, 4, 8, 16], beta=1.0):
        """Generate test case with exponential distributions.

        Parameters
        ----------
        O_k : np.ndarray, float, shape=(n_states)
            Offset parameters for each state.
        K_k : np.ndarray, float, shape=(n_states)
            Force constants for each state.
        beta : float, optional, default=1.0
            Inverse temperature.

        Notes
        -----
        We assume potentials of the form U(x) = (k / 2) * (x - o)^2
        Here, k and o are the corresponding entries of O_k and K_k.
        The equilibrium distribution is given analytically by
        p(x;beta,K) = sqrt[(beta K) / (2 pi)] exp[-beta K (x-x_0)**2 / 2]
        The dimensionless free energy is therefore
        f(beta,K) = - (1/2) * ln[ (2 pi) / (beta K) ]

        """
        self.O_k = np.array(O_k, np.float64)
        self.n_states = len(self.O_k)
        self.K_k = np.array(K_k, np.float64)

        self.beta = beta

    def analytical_means(self):
        return self.O_k

    def analytical_variances(self):
        return (self.beta * self.K_k) ** -1.

    def analytical_standard_deviations(self):
        return (self.beta * self.K_k) ** -0.5

    def analytical_free_energies(self, subtract_component=0):
        fe = -0.5 * np.log(2 * np.pi / (self.beta * self.K_k))
        if subtract_component is not None:
            fe -= fe[subtract_component]
        return fe

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
        N_k : np.ndarray, shape=(n_states), dtype=int32
           N_k[k] is the number of samples generated from state k

        """
        N_k = np.array(N_k, np.int32)
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
            x0 = self.O_k[k]
            sigma = (self.beta * self.K_k[k]) ** -0.5
            x = np.random.normal(loc=x0, scale=sigma, size=N)
            x_kn[k, 0:N] = x
            x_n[index:(index + N)] = x
            for l in range(self.n_states):
                u = self.beta * 0.5 * self.K_k[l] * (x - self.O_k[l]) ** 2.0
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
