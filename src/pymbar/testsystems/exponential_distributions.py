import numpy as np
import pandas as pd
from pymbar.utils import ensure_type


class ExponentialTestCase(object):
    def __init__(self, rates):
        """Generate test case with exponential distributions.

        Parameters
        ----------
        rates : np.ndarray, float, shape=(n_states)
            Rate parameters (e.g. lambda) for each state.
            
        Notes
        -----
        We assume potentials of the form U(x) = lambda x.  
        """
        rates = ensure_type(rates, np.float64, 1, "rates")

        self.n_states = len(rates)
        self.rates = rates
    
    def analytical_means(self):
        return self.rates ** -1.
        
    def analytical_variances(self):
        return self.rates ** -2.
        
    def analytical_free_energies(self):
        """Return the FE: -log(Z)"""
        return np.log(self.rates)

    def analytical_x_squared(self):
        return self.analytical_variances() + self.analytical_means() ** 2.

    def sample(self, N_k):
        """Draw samples from the distribution.

        Parameters
        ----------

        N_k : np.ndarray, int
            number of samples per state
            
        Returns
        -------
        x_kn : np.ndarray, shape=(n_states, n_samples), dtype=float
            1D harmonic oscillator positions            
        """
        N_k = ensure_type(N_k, np.float64, 1, "N_k", self.n_states, warn_on_cast=False)

        states = ["state %d" % k for k in range(self.n_states)]
        
        x_n = []
        origin_and_frame = []
        for k, N in enumerate(N_k):
            x_n.extend(np.random.exponential(scale=self.rates[k] ** -1., size=N))
            origin_and_frame.extend([(states[k], i) for i in range(int(N))])
        
        origin_and_frame = pd.MultiIndex.from_tuples(origin_and_frame, names=["origin", "frame"])
        x_n = pd.Series(x_n, name="x", index=origin_and_frame)

        u_kn = np.outer(x_n, self.rates)
        u_kn = pd.DataFrame(u_kn, columns=states, index=origin_and_frame).T  # Note the transpose

        return x_n, u_kn, origin_and_frame
