
"""Set the imports for the JAX accelerated methods"""

import logging
from functools import partial, wraps

from jax.config import config

import jax.numpy as jnp
from jax.numpy.linalg import lstsq
import jax.scipy.optimize
from jax.scipy.special import logsumexp

from jax import jit

from pymbar.mbar_solvers.mbar_solver import MBARSolver

logger = logging.getLogger(__name__)

# hell: https://github.com/google/jax/discussions/16020


@jax.tree_util.register_pytree_node_class
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
                "******************************************\n"
            )
        super().__init__()

    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {
            "exp": self.exp,
            "sum": self.sum,
            "diag": self.diag,
            "newaxis": self.newaxis,
            "dot": self.dot,
            "s_": self._s,
            "pad": self.pad,
            "lstsq": self.lstsq,
            "optimize": self.optimize,
            "logsumexp": self.logsumexp
        }  # static values
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls()

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

    # def _precondition_jit(self, jitable_fn):
    #     @wraps(jitable_fn)  # Helper to ensure the decorated function still registers for docs and inspection
    #     def wrapped_precog_jit(self, *args, **kwargs):
    #         # Uses "self" here as intercepted first arg for instance of MBARSolver
    #         jited_fn = self.jit(jitable_fn)
    #         return jited_fn(*args, **kwargs)
    #     return wrapped_precog_jit

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
            # jited_fn = partial(jit, static_argnums=(0,))(jitable_fn)
            # breakpoint()
            # print(jited_fn._cache_size())
            return jited_fn(*args, **kwargs)
        return staggered_jit

    def _adaptive_core(self, u_kn, N_k, f_k, g, options):
        """JAX version of adaptive inner loop.
        N_k must be float (should be cast at a higher level)

        """
        gamma = options["gamma"]
        # Perform Newton-Raphson iterations (with sci computed on the way)
        g = self.mbar_gradient(u_kn, N_k, f_k)  # Objective function gradient
        H = self.mbar_hessian(u_kn, N_k, f_k)  # Objective function hessian
        Hinvg = lstsq(H, g, rcond=-1)[0]
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
