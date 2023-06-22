# Import the methods functionally
# This is admittedly non-standard, but solves the following use case:
# * Has JAX
# * Wants to use PyMBAR
# * Does NOT want JAX to be set to 64-bit mode
# Also solves the future use case of different accelerator,
# but want to selectively use them

# Fallback/default solver methods
# NOTE: ALL ACCELERATORS MUST SHADOW THIS NAMESPACE EXACTLY
import numpy as np
from numpy.linalg import lstsq
import scipy.optimize
from scipy.special import logsumexp

from pymbar.mbar_solvers.mbar_solver import MBARSolver


class MBARSolverNumpy(MBARSolver):
    """
    Solver methods for MBAR. Implementations use specific libraries/accelerators to solve the code paths.

    Default solver is the numpy solution
    """

    @property
    def exp(self):
        return np.exp

    @property
    def sum(self):
        return np.sum

    @property
    def diag(self):
        return np.diag

    @property
    def newaxis(self):
        return np.newaxis

    @property
    def dot(self):
        return np.dot

    @property
    def s_(self):
        return np.s_

    @property
    def pad(self):
        return np.pad

    @property
    def lstsq(self):
        return lstsq

    @property
    def optimize(self):
        return scipy.optimize

    @property
    def logsumexp(self):
        return logsumexp

    @staticmethod
    def _passthrough_jit(fn):
        return fn

    @property
    def jit(self):
        """Passthrough JIT"""
        return self._passthrough_jit

    def _adaptive_core(self, u_kn, N_k, f_k, g, gamma):
        """
        Core function to execute per iteration of a method.
        """
        H = self.mbar_hessian(u_kn, N_k, f_k)  # Objective function hessian
        Hinvg = np.linalg.lstsq(H, g, rcond=-1)[0]
        Hinvg -= Hinvg[0]
        f_nr = f_k - gamma * Hinvg

        # self-consistent iteration gradient norm and saved log sums.
        f_sci = self.self_consistent_update(u_kn, N_k, f_k)
        f_sci = f_sci - f_sci[0]  # zero out the minimum
        g_sci = self.mbar_gradient(u_kn, N_k, f_sci)
        gnorm_sci = self.dot(g_sci, g_sci)

        # newton raphson gradient norm and saved log sums.
        g_nr = self.mbar_gradient(u_kn, N_k, f_nr)
        gnorm_nr = self.dot(g_nr, g_nr)

        return f_sci, g_sci, gnorm_sci, f_nr, g_nr, gnorm_nr
