"""Test MBAR accelerators by ensuring they yield comperable results to the default (numpy) and can cycle between them
"""

import numpy as np
import pytest

from pymbar import MBAR
from pymbar.mbar_solvers import get_accelerator, default_solver
from pymbar.utils_for_testing import assert_equal, assert_allclose

# Pylint doesn't like the interplay between pytest and importing fixtures. disabled the one problem.
from pymbar.tests.test_mbar import (  # pylint: disable=unused-import
    system_generators,
    N_k,
    free_energies_almost_equal,
    fixed_harmonic_sample,
)

# Setup skip if conditions
has_jax = False
try:
    # pylint: disable=unused-import
    from jax import jit
    from jax.numpy import ndarray as jax_ndarray

    has_jax = True
except ImportError:
    pass

# Establish marks
needs_jax = pytest.mark.skipif(not has_jax, reason="Needs Jax Accelerator")


# Required test function for testing that the accelerator worked correctly.
def check_numpy(mbar: MBAR):
    assert isinstance(mbar.f_k, np.ndarray)


def check_jax(mbar: MBAR):
    assert isinstance(mbar.f_k, jax_ndarray)


# Setup accelerator list. Each parameter is (string_of_accelerator, accelerator_check)
numpy_accel = pytest.param(("numpy", check_numpy), id="numpy")
jax_accel = pytest.param(("jax", check_jax), marks=needs_jax, id="jax")
accelerators = [numpy_accel, jax_accel]


@pytest.fixture
def fallback_accelerator():
    return "numpy", check_numpy


@pytest.fixture(scope="module", params=system_generators)
def only_test_data(request):
    _, test = request.param()
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
    assert_equal(N_k, N_k_output)
    yield_bundle = {"test": test, "x_n": x_n, "u_kn": u_kn}
    yield yield_bundle


@pytest.fixture()
def static_ukn_nk(fixed_harmonic_sample):
    _, u_kn, N_k_output, _ = fixed_harmonic_sample.sample(N_k, mode="u_kn")
    assert_equal(N_k, N_k_output)
    return u_kn, N_k_output


@pytest.mark.parametrize("accelerator", accelerators)
def test_mbar_accelerators_are_accurate(only_test_data, accelerator):
    """Test that each accelerator is scientifically accurate"""
    accelerator_name, accelerator_check = accelerator
    test, x_n, u_kn = only_test_data["test"], only_test_data["x_n"], only_test_data["u_kn"]
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
    mbar = build_out_an_mbar(u_kn, N_k, accelerator_name, accelerator_check, boostraps=200)
    results = mbar.compute_free_energy_differences()
    fe = results["Delta_f"]
    fe_sigma = results["dDelta_f"]
    free_energies_almost_equal(fe, fe_sigma, test.analytical_free_energies())
    accelerator_check(mbar)


def build_out_an_mbar(u_kn, N_k, accelerator_name, accelerator_check, boostraps=0):
    """Helper function to build an MBAR object"""
    mbar = MBAR(u_kn, N_k, verbose=True, accelerator=accelerator_name, n_bootstraps=boostraps)
    assert mbar.solver == get_accelerator(accelerator_name)
    accelerator_check(mbar)
    return mbar


@pytest.mark.parametrize("accelerator", accelerators)
def test_mbar_accelerators_can_toggle(static_ukn_nk, accelerator, fallback_accelerator):
    """
    Test that accelerator can toggle and the act of doing so doesn't corrupt each other's output.
    """
    u_kn, N_k_output = static_ukn_nk
    # Setup and check the accelerator
    accelerator_name, accelerator_check = accelerator
    mbar = build_out_an_mbar(u_kn, N_k, accelerator_name, accelerator_check)
    # Setup and check the fallback
    fall_back_name, fall_back_check = fallback_accelerator
    mbar_fallback = build_out_an_mbar(u_kn, N_k, fall_back_name, fall_back_check)
    # Ensure fallback and accelerator match
    assert_allclose(mbar.f_k, mbar_fallback.f_k)
    # Rebuild the accelerated version again.
    mbar_rebuild = build_out_an_mbar(u_kn, N_k, accelerator_name, accelerator_check)
    assert_allclose(mbar.f_k, mbar_rebuild.f_k)


def test_default_acclerator_is_correct(static_ukn_nk):
    u_kn, N_k_output = static_ukn_nk

    def blank_check(*args):
        return True

    mbar = build_out_an_mbar(u_kn, N_k, default_solver, blank_check)
    assert mbar.solver == get_accelerator(default_solver)
