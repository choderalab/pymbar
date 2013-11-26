import numpy as np
from pymbar.utils import ensure_type

import logging
logger = logging.getLogger(__name__)

class HarmonicOscillatorsTestCase(object):
    def __init__(self, O_k, K_k, beta_k):
        """Generate test case with harmonic oscillators.

        Parameters
        ----------
        O_k : np.ndarray, float, shape=(n_states)
            Offset parameters for each state.
        K_k : np.ndarray, float, shape=(n_states)
            Force constants for each state.            
        Notes
        -----
        We assume potentials of the form U(x) = (k / 2) * (x - o)^2
        Here, k and o are the corresponding entries of O_k and K_k.
        The equilibrium distribution is given analytically by
        p(x;beta,K) = sqrt[(beta K) / (2 pi)] exp[-beta K (x-x_0)**2 / 2]
        The dimensionless free energy is therefore
        f(beta,K) = - (1/2) * ln[ (2 pi) / (beta K) ]        
        
        """
        self.O_k = ensure_type(O_k, np.float64, 1, "O_k")
        self.n_states = len(self.O_k)
        
        self.K_k = ensure_type(K_k, np.float64, 1, "K_k", self.n_states)
        self.beta_k = ensure_type(beta_k, np.float64, 1, "beta_k", self.n_states)
    
    def analytical_means(self):
        return self.O_k
        
    def analytical_variances(self):
        return (self.beta_k * self.K_k) ** -1.
        
    def analytical_free_energies(self, subtract_component=0):
        fe = -0.5 * np.log( 2 * np.pi / (self.beta_k * self.K_k))
        if subtract_component is not None:
            fe -= fe[subtract_component]
        return fe

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
        x_n : np.ndarray, shape=(n_samples), dtype=float
            1D harmonic oscillator positions            
        u_kn : np.ndarray, shape=(n_states, n_samples), dtype=float
            1D harmonic oscillator reduced (unitless) potential energies
        origin : np.ndarray, shape=(n_states, n_samples), dtype=float
            State of origin of each sample
        """
        N_k = ensure_type(N_k, np.float64, 1, "N_k", self.n_states, warn_on_cast=False)

        states = range(self.n_states)
        
        x_n = []
        origin_and_frame = []
        for k, N in enumerate(N_k):
            x0 = self.O_k[k]
            sigma = (self.beta_k[k] * self.K_k[k]) ** -0.5
            x_n.extend(np.random.normal(loc=x0, scale=sigma, size=N))
            origin_and_frame.extend([(states[k], i) for i in range(int(N))])
        
        origin_and_frame = np.array(origin_and_frame)
        x_n = np.array(x_n)

        u_kn = np.array([x_n for state in states])

        u_kn = 0.5 * self.beta_k * self.K_k * (u_kn.T - self.O_k) ** 2.0        
        u_kn = u_kn.T

        return x_n, u_kn, origin_and_frame
