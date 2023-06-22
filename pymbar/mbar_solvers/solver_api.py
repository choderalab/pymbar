"""
API Definitions of the solver module to be consistent with PyMBAR 4.0
and for subclassing any solvers for implementation.
"""

from functools import wraps
from abc import ABC, abstractmethod


class MBARSolverAPI(ABC):
    """
    API for MBAR solvers
    """

    JITABLE_API_METHODS = (
        "mbar_gradient",
        "mbar_objective",
        "mbar_objective_and_gradient",
        "mbar_hessian",
        "mbar_log_W_nk",
        "mbar_W_nk",
        "precondition_u_kn"
    )

    @abstractmethod
    def self_consistent_update(self, u_kn, N_k, f_k, states_with_samples=None):
        pass

    @abstractmethod
    def mbar_gradient(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def mbar_objective(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def mbar_objective_and_gradient(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def mbar_hessian(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def mbar_log_W_nk(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def mbar_W_nk(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def adaptive(self, u_kn, N_k, f_k, tol=1.0e-8, options=None):
        pass

    @abstractmethod
    def precondition_u_kn(self, u_kn, N_k, f_k):
        pass

    @abstractmethod
    def solve_mbar_once(
            self,
            u_kn_nonzero,
            N_k_nonzero,
            f_k_nonzero,
            method="adaptive",
            tol=1e-12,
            continuation=None,
            options=None,
        ):
        pass

    @abstractmethod
    def solve_mbar(self, u_kn_nonzero, N_k_nonzero, f_k_nonzero, solver_protocol=None):
        pass

    @abstractmethod
    def solve_mbar_for_all_states(self, u_kn, N_k, f_k, states_with_samples, solver_protocol):
        pass


class MBARSolverAcceleratorMethods(ABC):
    """
    Methods which have to be implemented by MBAR solver accelerators
    """

    JITABLE_ACCELERATOR_METHODS = (
        "_adaptive_core",
    )

    @property
    @abstractmethod
    def exp(self):
        pass

    @property
    @abstractmethod
    def sum(self):
        pass

    @property
    @abstractmethod
    def diag(self):
        pass

    @property
    @abstractmethod
    def newaxis(self):
        pass

    @property
    @abstractmethod
    def dot(self):
        pass

    @property
    @abstractmethod
    def s_(self):
        pass

    @property
    @abstractmethod
    def pad(self):
        pass

    @property
    @abstractmethod
    def lstsq(self):
        pass

    @property
    @abstractmethod
    def optimize(self):
        pass

    @property
    @abstractmethod
    def logsumexp(self):
        pass

    @property
    @abstractmethod
    def jit(self):
        pass

    def _precondition_jit(self, jitable_fn):
        @wraps(jitable_fn)  # Helper to ensure the decorated function still registers for docs and inspection
        def wrapped_precog_jit(*args, **kwargs):
            # Uses "self" here as intercepted first arg for instance of MBARSolver
            jited_fn = self.jit(jitable_fn)
            return jited_fn(*args, **kwargs)
        return wrapped_precog_jit

    @abstractmethod
    def _adaptive_core(self, u_kn, N_k, f_k, g, options):
        pass

    def __hash__(self):
      return hash((self.exp,
                   self.sum,
                   self.diag,
                   self.newaxis,
                   self.dot,
                   self.s_,
                   self.pad,
                   self.lstsq,
                   self.optimize,
                   self.logsumexp,
                   self.jit
                   ))

    def __eq__(self, other):
        return isinstance(other, MBARSolverAcceleratorMethods) and self.__hash__ == other.__hash__
