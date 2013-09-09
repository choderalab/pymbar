from pymbar.utils import ensure_type
import numpy as np

K_k = np.array([25.0, 16.0, 9.0, 4.0, 1.0, 1.0])
O_k = np.array([0.0, 1, 2, 3, 4, 5])
N_k = 100 * np.array([1000.0, 1000, 1000, 1000, 0, 1000])
Nk_ne_zero = (N_k != 0)
beta = 1.0
K_extra = np.array([20.0, 12, 6, 2, 1])
O_extra = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

class AnalyticalHarmonicOscillator(object):
    """
    For a harmonic oscillator with spring constant K,
    x ~ Normal(x_0, sigma^2), where sigma = 1/sqrt(beta K)
    """

    def __init__(self, beta, K, O):
        """Generate a set of harmonic oscillators.  
        
        Parameters
        ----------
        beta : np.ndarray, float
            number of samples per state
        K : np.ndarray, float
            force constants of harmonic oscillators in dimensionless units
        O : np.ndarray, float
            offsets of the harmonic oscillators in dimensionless units
        """
        self.n_states = len(K)

        K = ensure_type(K, np.float64, 1, 'K')
        O = ensure_type(O, np.float64, 1, 'O', self.n_states)

        self.sigma = (beta * K) ** (-0.5)

        self.f_k = -np.log(np.sqrt(2.0 * np.pi) * self.sigma)
        f_as_2D = np.array([self.f_k])
        self.f_ij = f_as_2D - f_as_2D.T

        # Calculate and store observables
        self.RMS_displacement = self.sigma
        self.potential_energy = 1.0 / (2.0 * beta) * np.ones(self.n_states)
        self.position = O.copy()
        self.position_squared = (1.0 + beta * K * O ** 2.0) / (beta * K)
        self.displacement = np.zeros(self.n_states)  # CHECK
        self.displacement_squared = self.sigma ** 2.0  # CHECK
