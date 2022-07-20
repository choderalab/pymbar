import numpy as np


class ExponentialTestCase(object):

    """Test cases using exponential distributions.

    Examples
    --------

    Generate energy samples with default parameters.

    >>> testcase = ExponentialTestCase()
    >>> x_kn, u_kln, N_k = testcase.sample()

    Retrieve analytical properties.

    >>> analytical_means = testcase.analytical_means()
    >>> analytical_variances = testcase.analytical_variances()
    >>> analytical_standard_deviations = testcase.analytical_standard_deviations()
    >>> analytical_free_energies = testcase.analytical_free_energies()
    >>> analytical_x_squared = testcase.analytical_x_squared()

    Generate energy samples with default parameters in one line.

    >>> x_kn, u_kln, N_k = ExponentialTestCase().sample()

    Generate energy samples with specified parameters.

    >>> testcase = ExponentialTestCase(rates=[1., 2., 3., 4., 5.])
    >>> x_kn, u_kln, N_k = testcase.sample(N_k=[10, 20, 30, 40, 50])

    Test sampling in different output modes.

    >>> x_kn, u_kln, N_k = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kln')
    >>> x_n, u_kn, N_k, s_n = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kn')
    >>> testcase = ExponentialTestCase(rates=[4., 5.])
    >>> w_F, w_R, N_k = testcase.sample(N_k=[40, 50], mode='wFwR')

    """

    def __init__(self, rates=(1, 2, 3, 4, 5), beta=1.0):
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
        self.rates = np.array(rates, np.float64)
        self.beta = beta

    def analytical_free_energies(self):
        """Return the FE: -log(Z)"""
        return np.log(self.rates)

    def analytical_means(self):
        return self.rates**-1.0

    def analytical_variances(self):
        return self.rates**-2.0

    def analytical_standard_deviations(self):
        return np.sqrt(self.rates**-2.0)

    def analytical_observable(self, observable="position"):

        if observable == "position":
            return self.analytical_means()
        if observable == "position^2":
            return 2.0 * self.analytical_variances()
        if observable == "RMS displacement":
            # <X^2> - <X>^2 = 2L^2-L^2 = L^2
            return self.analytical_variances()
        if observable == "potential energy":
            return np.ones(len(self.rates))

    def analytical_entropies(self):
        return (
            self.analytical_observable(observable="potential energy")
            - self.analytical_free_energies()
        )

    def analytical_x_squared(self):
        return self.analytical_variances() + self.analytical_means() ** 2.0

    def sample(self, N_k=(10, 20, 30, 40, 50), mode="u_kln", seed=None):
        """Draw samples from the distribution.

        Parameters
        ----------

        N_k : np.ndarray, int
            number of samples per state
        mode : str, optional, default='u_kln'
            If 'u_kln', return K x K x N_max matrix where u_kln[k,l,n] is reduced
            potential of sample n from state k evaluated at state l.
            If 'u_kn', return K x N_tot matrix where u_kn[k,n] is reduced potential
            of sample n (in concatenated indexing) evaluated at state k.
            If 'wFwR', check that len(N_k) involves only two states, and calculate
            the forward and reverse work distributions.

        seed: int, optional, default=None.
            Provides control over the random seed for replicability.

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

        if mode == 'u_kln':

        x_kn : np.ndarray, shape=(n_states, n_samples), dtype=float
            1D harmonic oscillator positions
        u_kln : np.ndarray, shape=(n_states, n_states, n_samples), dytpe=float, only if mode='u_kln'
           u_kln[k,l,n] is reduced potential of sample n from state k evaluated at state l.
        N_k : np.ndarray, shape=(n_states), dtype=float
           N_k[k] is the number of samples generated from state k


        if mode == 'wFwR':

        w_F : np.ndarray, shape=(N_k[0]), dtype=float
            Work generated switching from state 0 to 1
        w_R : np.ndaarry, shape=(N_k[1]), dtype=float
            Work generated switching from state 1 to 0
        N_k : np.ndarray, shape=(2), dtype=float
           N_k[k] is the number of samples generated from state k

        """

        np.random.seed(seed)

        N_k = np.array(N_k, np.int32)
        if len(N_k) != self.n_states:
            raise Exception(
                "N_k has {:d} states while self.n_states has {:d} states.".format(
                    len(N_k), self.n_states
                )
            )

        if mode == "wFwR":
            if len(N_k) != 2:
                raise Exception(
                    "N_k has {:d} states instead of 2, we cannot generate forward and reverse work distributions".format(
                        len(N_k)
                    )
                )

        N_max = N_k.max()  # maximum number of samples per state
        N_tot = N_k.sum()  # total number of samples
        x_kn = np.zeros([self.n_states, N_max], np.float64)
        u_kln = np.zeros([self.n_states, self.n_states, N_max], np.float64)
        x_n = np.zeros([N_tot], np.float64)
        s_n = np.zeros([N_tot], int)
        u_kn = np.zeros([self.n_states, N_tot], np.float64)
        index = 0
        for k, N in enumerate(N_k):
            x = np.random.exponential(scale=self.rates[k] ** -1.0, size=N)
            x_kn[k, 0:N] = x
            x_n[index : (index + N)] = x
            s_n[index : (index + N)] = k
            for l in range(self.n_states):
                u = self.beta * self.rates[l] * x
                u_kln[k, l, 0:N] = u
                u_kn[l, index : (index + N)] = u
            index += N

        if mode == "u_kn":
            return x_n, u_kn, N_k, s_n
        elif mode == "u_kln":
            return x_kn, u_kln, N_k
        elif mode == "wFwR":
            return (
                u_kln[0, 1, : N_k[0]] - u_kln[0, 0, : N_k[0]],
                u_kln[1, 0, : N_k[1]] - u_kln[1, 1, : N_k[1]],
                N_k,
            )
        else:
            raise Exception("Unknown mode '{}'".format(mode))

        return

    @classmethod
    def evenly_spaced_exponentials(
        cls, n_states, n_samples_per_state, lower_rate=1.0, upper_rate=3.0
    ):
        """Generate samples from evenly spaced exponential distributions.

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

        name = "{:d}x{:d} exponentials".format(n_states, n_samples_per_state)

        rates = np.linspace(lower_rate, upper_rate, n_states)
        N_k = (np.ones(n_states) * n_samples_per_state).astype("int")

        testsystem = cls(rates)
        x_n, u_kn, N_k_output, s_n = testsystem.sample(N_k, mode="u_kn")

        return name, testsystem, x_n, u_kn, N_k_output, s_n
