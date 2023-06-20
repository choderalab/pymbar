##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2017-2022 University of Colorado Boulder
# Copyright 2010-2017 Memorial Sloan-Kettering Cancer Center
# Portions of this software are Copyright (c) 2010-2016 University of Virginia
# Portions of this software are Copyright (c) 2006-2007 The Regents of the University of California.  All Rights Reserved.
# Portions of this software are Copyright (c) 2007-2008 Stanford University and Columbia University.
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################

"""
###########
pymbar.mbar_solvers
###########

A module implementing the solvers array operations for the MBAR solvers with various code bases for acceleration.

All methods have the same calls and returns, independent of their underlying codes for solution.

Please reference the following if you use this code in your research:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177

"""

import logging
from .mbar_solver import (
    validate_inputs,
    JAX_SOLVER_PROTOCOL,
    DEFAULT_SOLVER_PROTOCOL,
    ROBUST_SOLVER_PROTOCOL,
    BOOTSTRAP_SOLVER_PROTOCOL
)
from .numpy_solver import MBARSolverNumpy

logger = logging.getLogger(__name__)

default_solver = MBARSolverNumpy  # Set fallback solver
ACCELERATOR_MAP = {"numpy": MBARSolverNumpy}
try:
    from .jax_solver import MBARSolverJAX
    default_solver = MBARSolverJAX
    ACCELERATOR_MAP["jax"] = MBARSolverJAX
    logger.info("JAX detected. Using JAX acceleration by default.")
except ImportError:
    logger.warning(
        "\n"
        "********* JAX NOT FOUND *********\n"
        " PyMBAR can run faster with JAX  \n"
        " But will work fine without it   \n"
        "Either install with pip or conda:\n"
        "      pip install pybar[jax]     \n"
        "               OR                \n"
        "      conda install pymbar       \n"
        "*********************************"
    )


# Helper function for toggling the solver method
def get_accelerator(accelerator_name: str):
    """
    get the accelerator in the namespace for this module
    """
    # Saving accelerator to new tag does not change since we're saving the immutable string object
    accel = accelerator_name.lower()
    if accel not in ACCELERATOR_MAP:
        raise ValueError(
            f"Accelerator {accel} is not implemented or did not load correctly. Please use one of the following:\n"
            + "".join((f"* {a}\n" for a in ACCELERATOR_MAP.keys()))
            + f"(case-insentive)\n"
            + f"If you expected {accel} to load, please check the logs above for details."
        )
    logger.info(f"Getting accelerator {accel}...")
    return ACCELERATOR_MAP[accel]


# Imports done, handle initialization
module_solver = default_solver()

# Establish API methods for 4.x consistency
self_consistent_update = module_solver.self_consistent_update
mbar_gradient = module_solver.mbar_gradient
mbar_objective = module_solver.mbar_objective
mbar_objective_and_gradient = module_solver.mbar_objective_and_gradient
mbar_hessian = module_solver.mbar_hessian
mbar_log_W_nk = module_solver.mbar_log_W_nk
mbar_W_nk = module_solver.mbar_W_nk
adaptive = module_solver.adaptive
precondition_u_kn = module_solver.precondition_u_kn
solve_mbar_once = module_solver.solve_mbar_once
solve_mbar = module_solver.solve_mbar
solve_mbar_for_all_states = module_solver.solve_mbar_for_all_states
