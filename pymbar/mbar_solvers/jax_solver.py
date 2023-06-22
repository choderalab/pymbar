"""Set the imports for the JAX accelerated methods"""

import logging
from functools import wraps

try:
    from jax.config import config

    import jax.numpy as jnp
    from jax.numpy.linalg import lstsq
    import jax.scipy.optimize
    from jax.scipy.special import logsumexp

    from jax import jit
except ImportError:
    raise ImportError("JAX not found!")

from pymbar.mbar_solvers.mbar_solver import MBARSolver

logger = logging.getLogger(__name__)


class MBARSolverJAX(MBARSolver):
    """
    Solver methods for MBAR. Implementations use specific libraries/accelerators to solve the code paths.

    Default solver is the numpy solution
    """

    def __init__(self):
        # Throw warning only if the whole of JAX is found
        if not config.x64_enabled:
            # Warn that we're going to be setting 64 bit jax
            logger.warning(
                "\n"
                "****** PyMBAR will use 64-bit JAX! *******\n"
                "* JAX is currently set to 32-bit bitsize *\n"
                "* which is its default.                  *\n"
                "*                                        *\n"
                "* PyMBAR requires 64-bit mode and WILL   *\n"
                "* enable JAX's 64-bit mode when called.  *\n"
                "*                                        *\n"
                "* This MAY cause problems with other     *\n"
                "* Uses of JAX in the same code.          *\n"
                "*                                        *\n"
                "* If you want 32-bit JAX and PyMBAR      *\n"
                "* please set:                            *\n"
                "*           accelerator=numpy            *\n"
                "* when you instance the MBAR object      *\n"
                "******************************************\n"
            )
        # Double __ in middle name intentional here
        self._static__adaptive_core = generate_static_adaptive_core(self)
        super().__init__()

    @property
    def exp(self):
        return jnp.exp

    @property
    def sum(self):
        return jnp.sum

    @property
    def diag(self):
        return jnp.diag

    @property
    def newaxis(self):
        return jnp.newaxis

    @property
    def dot(self):
        return jnp.dot

    @property
    def s_(self):
        return jnp.s_

    @property
    def pad(self):
        return jnp.pad

    @property
    def lstsq(self):
        return lstsq

    @property
    def optimize(self):
        return jax.scipy.optimize

    @property
    def logsumexp(self):
        return logsumexp

    @property
    def jit(self):
        return jit

    @property
    def real_jit(self):
        return True

    def _precondition_jit(self, jitable_fn):
        @wraps(
            jitable_fn
        )  # Helper to ensure the decorated function still registers for docs and inspection
        def staggered_jit(*args, **kwargs):
            # This will only trigger if JAX is set
            if not config.x64_enabled:
                # Warn that JAX 64-bit will being turned on
                logger.warning(
                    "\n"
                    "******* JAX 64-bit mode is now on! *******\n"
                    "*     JAX is now set to 64-bit mode!     *\n"
                    "*   This MAY cause problems with other   *\n"
                    "*      uses of JAX in the same code.     *\n"
                    "******************************************\n"
                )
                config.update("jax_enable_x64", True)
            jited_fn = self.jit(jitable_fn)
            return jited_fn(*args, **kwargs)

        return staggered_jit

    def _adaptive_core(self, u_kn, N_k, f_k, g, gamma):
        """JAX version of adaptive inner loop.
        N_k must be float (should be cast at a higher level)

        """


def generate_static_adaptive_core(solver: MBARSolver):
    def _adaptive_core(u_kn, N_k, f_k, g, gamma):
        # Perform Newton-Raphson iterations (with sci computed on the way)
        g = solver.mbar_gradient(u_kn, N_k, f_k)  # Objective function gradient
        H = solver.mbar_hessian(u_kn, N_k, f_k)  # Objective function hessian
        Hinvg = lstsq(H, g, rcond=-1)[0]
        Hinvg -= Hinvg[0]
        f_nr = f_k - gamma * Hinvg

        # self-consistent iteration gradient norm and saved log sums.
        f_sci = solver.self_consistent_update(u_kn, N_k, f_k)
        f_sci = f_sci - f_sci[0]  # zero out the minimum
        g_sci = solver.mbar_gradient(u_kn, N_k, f_sci)
        gnorm_sci = solver.dot(g_sci, g_sci)

        # newton raphson gradient norm and saved log sums.
        g_nr = solver.mbar_gradient(u_kn, N_k, f_nr)
        gnorm_nr = solver.dot(g_nr, g_nr)

        return f_sci, g_sci, gnorm_sci, f_nr, g_nr, gnorm_nr

    return _adaptive_core
