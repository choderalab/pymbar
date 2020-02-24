##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2016-2017 University of Colorado Boulder
# Copyright 2010-2017 Memorial Sloan-Kettering Cancer Center
# Portions of this software are Copyright 2010-2016 University of Virginia
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_array_less,
    assert_array_max_ulp,
    assert_equal,
    assert_raises,
    assert_string_equal,
    assert_warns,
)
import warnings
import contextlib

from pymbar.testsystems import HarmonicOscillatorsTestCase, ExponentialTestCase

__all__ = [
    "assert_allclose",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_almost_equal",
    "assert_array_almost_equal_nulp",
    "assert_array_equal",
    "assert_array_less",
    "assert_array_max_ulp",
    "assert_equal",
    "assert_raises",
    "assert_string_equal",
    "assert_warns",
    "suppress_derivative_warnings_for_tests",
    "suppress_matrix_warnings_for_tests",
    "oscillators",
    "exponentials",
]

##############################################################################
# functions
##############################################################################


@contextlib.contextmanager
def suppress_derivative_warnings_for_tests():
    """
    Suppress specific warnings and then reset when done, used as a with suppress_warnings():
    """
    # Supress the warnings when Jacobian and Hessian information is not used in a specific solver
    warnings.filterwarnings("ignore", ".*does not use the Jacobian.*")
    warnings.filterwarnings("ignore", ".*does not use Hessian.*")
    warnings.filterwarnings(
        "ignore", ".*parameter will change to the default of machine precision.*"
    )
    yield
    # Clear warning filters
    warnings.resetwarnings()


@contextlib.contextmanager
def suppress_matrix_warnings_for_tests():
    """
    Suppress specific warnings and then reset when done, used as a with suppress_warnings():
    """
    # Supress the numpy matrix warnings
    warnings.filterwarnings("ignore", ".*the matrix subclass is not*")
    yield
    # Clear warning filters
    warnings.resetwarnings()


def oscillators(n_states, n_samples, provide_test=False):
    name = f"{n_states}x{n_samples} oscillators"
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype("int")
    test = HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
    returns = [name, u_kn, N_k_output, s_n]
    if provide_test:
        returns.append(test)
    return returns


def exponentials(n_states, n_samples, provide_test=False):
    name = f"{n_states}x{n_samples} exponentials"
    rates = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype("int")
    test = ExponentialTestCase(rates)
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
    returns = [name, u_kn, N_k_output, s_n]
    if provide_test:
        returns.append(test)
    return returns
