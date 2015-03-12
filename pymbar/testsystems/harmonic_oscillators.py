import numpy as np
from pymbar.utils import ensure_type


class HarmonicOscillatorsTestCase(object):

    """Test cases using harmonic oscillators.

    Examples
    --------

    Generate energy samples with default parameters.

    >>> testcase = HarmonicOscillatorsTestCase()
    >>> [x_kn, u_kln, N_k, s_n] = testcase.sample()

    Retrieve analytical properties.

    >>> analytical_means = testcase.analytical_means()
    >>> analytical_variances = testcase.analytical_variances()
    >>> analytical_standard_deviations = testcase.analytical_standard_deviations()
    >>> analytical_free_energies = testcase.analytical_free_energies()
    >>> analytical_x_squared = testcase.analytical_observable('position^2')

    Generate energy samples with default parameters in one line.

    >>> (x_kn, u_kln, N_k, s_n) = HarmonicOscillatorsTestCase().sample()

    Generate energy samples with specified parameters.

    >>> testcase = HarmonicOscillatorsTestCase(O_k=[0, 1, 2, 3, 4], K_k=[1, 2, 4, 8, 16])
    >>> (x_kn, u_kln, N_k, s_n) = testcase.sample(N_k=[10, 20, 30, 40, 50])

    Test sampling in different output modes.

    >>> (x_kn, u_kln, N_k) = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kln')
    >>> (x_n, u_kn, N_k, s_n) = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kn')

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
        self.beta = beta
        self.O_k = np.array(O_k, np.float64)
        self.n_states = len(self.O_k)
        self.K_k = np.array(K_k, np.float64)

        if len(self.K_k) != self.n_states:
            raise ValueError('Lengths of K_k=%d and O_k=%d should be equal' % (len(self.O_k),len(self.K_k)))

    def analytical_means(self):
        return self.O_k

    def analytical_variances(self):
        return (self.beta * self.K_k) ** -1.

    def analytical_standard_deviations(self):
        return (self.beta * self.K_k) ** -0.5

    def analytical_observable(self, observable = 'position'):

        if observable == 'position':
            return self.analytical_means()
        if observable == 'potential energy':
            return (0.5/self.beta)*np.ones(self.n_states)
        if observable == 'position^2':
            return 1.0/(self.beta*self.K_k) + np.square(self.O_k)
        if observable == 'RMS displacement':
            return self.analytical_standard_deviations()

    def analytical_free_energies(self, subtract_component=0):
        fe = -0.5 * np.log(2 * np.pi / (self.beta * self.K_k))
        if subtract_component is not None:
            fe -= fe[subtract_component]
        return fe

    def analytical_entropies(self, subtract_component = 0):
        return self.analytical_observable(observable = 'potential energy') - self.analytical_free_energies(subtract_component)

    def sample(self, N_k=[10, 20, 30, 40, 50], mode='u_kn'):
        """Draw samples from the distribution.

        Parameters
        ----------

        N_k : np.ndarray, int
            number of samples per state
        mode : str, optional, default='u_kn'
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
        s_n : np.ndarray, shape=(n_samples), dtype='int'
            s_n is the state of origin of x_n

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

        N_max = N_k.max()  # maximum number of samples per state
        N_tot = N_k.sum()  # total number of samples

        x_kn = np.zeros([self.n_states, N_max], np.float64)
        u_kln = np.zeros([self.n_states, self.n_states, N_max], np.float64)
        x_n = np.zeros([N_tot], np.float64)
        s_n = np.zeros([N_tot], np.int)
        u_kn = np.zeros([self.n_states, N_tot], np.float64)
        index = 0
        for k, N in enumerate(N_k):
            x0 = self.O_k[k]
            sigma = (self.beta * self.K_k[k]) ** -0.5
            x = np.random.normal(loc=x0, scale=sigma, size=N)
            x_kn[k, 0:N] = x
            x_n[index:(index + N)] = x
            s_n[index:(index + N)] = k
            for l in range(self.n_states):
                u = self.beta * 0.5 * self.K_k[l] * (x - self.O_k[l]) ** 2.0
                u_kln[k, l, 0:N] = u
                u_kn[l, index:(index + N)] = u
            index += N

        if (mode == 'u_kn'):
            return x_n, u_kn, N_k, s_n
        elif (mode == 'u_kln'):
            return x_kn, u_kln, N_k
        else:
            raise Exception("Unknown mode '%s'" % mode)

        return

    @classmethod
    def evenly_spaced_oscillators(cls, n_states, n_samples_per_state, lower_O_k=1.0, upper_O_k=5.0, lower_k_k=1.0, upper_k_k=3.0):
        """Generate samples from evenly spaced harmonic oscillators.

        Parameters
        ----------
        n_states : np.ndarray, int
            number of states
        n_samples_per_state : np.ndarray, int
            number of samples per state.  The total number of samples
            n_samples will be equal to n_states * n_samples_per_state
        lower_O_k : float, optional, default=1.0
            Lower bound of O_k values
        upper_O_k : float, optional, default=5.0
            Upper bound of O_k values
        lower_k_k : float, optional, default=1.0
            Lower bound of O_k values
        upper_k_k : float, optional, default=3.0
            Upper bound of k_k values

        Returns
        -------
        name: str
            Name of testsystem
        testsystem : TestSystem
            The testsystem object
        x_n : np.ndarray, shape=(n_samples)
            Coordinates of the samples
        u_kn : np.ndarray, shape=(n_states, n_samples)
            Reduced potential energies
        N_k : np.ndarray, shape=(n_states)
            Number of samples drawn from each state
        s_n : np.ndarray, shape=(n_samples)
            State of origin of each sample
        """
        name = "%dx%d oscillators" % (n_states, n_samples_per_state)

        O_k = np.linspace(lower_O_k, upper_O_k, n_states)
        k_k = np.linspace(lower_k_k, upper_k_k, n_states)
        N_k = (np.ones(n_states) * n_samples_per_state).astype('int')

        testsystem = cls(O_k, k_k)
        x_n, u_kn, N_k_output, s_n = testsystem.sample(N_k, mode='u_kn')

        return name, testsystem, x_n, u_kn, N_k_output, s_n
