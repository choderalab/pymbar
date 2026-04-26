"""End-to-end tests for the chunked batch-evaluation path:
compute_expectations / compute_multiple_expectations /
compute_perturbed_free_energies with ``chunk_size`` set must match the dense
path bit-close for point estimates, and must reject ``compute_uncertainty=True``
and ``uncertainty_method='bootstrap'``.
"""
import numpy as np
import pytest

import pymbar
from pymbar.utils import ParameterError
from pymbar.utils_for_testing import assert_array_almost_equal, oscillators


@pytest.fixture(scope="module")
def small_oscillator():
    name, u_kn, N_k, s_n = oscillators(8, 300)
    m = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10)
    return m, u_kn, N_k


@pytest.mark.parametrize("chunk_size", [50, 137, 5000])
def test_compute_perturbed_free_energies_chunked(small_oscillator, chunk_size):
    m, u_kn, N_k = small_oscillator
    # u_ln = u_kn rows shifted slightly to define perturbed states.
    rng = np.random.default_rng(0)
    u_ln = u_kn + 0.1 * rng.standard_normal(u_kn.shape)
    dense = m.compute_perturbed_free_energies(u_ln, compute_uncertainty=False)
    chunked = m.compute_perturbed_free_energies(
        u_ln, compute_uncertainty=False, chunk_size=chunk_size)
    assert_array_almost_equal(chunked["Delta_f"], dense["Delta_f"], decimal=10)


@pytest.mark.parametrize("state_dependent", [False, True])
@pytest.mark.parametrize("chunk_size", [50, 137, 5000])
def test_compute_expectations_chunked(small_oscillator, chunk_size,
                                      state_dependent):
    # state_dependent=True hits the diagonal index pattern (state_map[0,k]=k,
    # state_map[1,k]=k) in the observable kernel; =False hits state_map[1,k]=0.
    m, u_kn, _ = small_oscillator
    rng = np.random.default_rng(1 + state_dependent)
    shape = (u_kn.shape[0], u_kn.shape[1]) if state_dependent else u_kn.shape[1]
    A_n = rng.uniform(0.5, 2.0, size=shape)
    kw = dict(state_dependent=state_dependent, compute_uncertainty=False)
    dense = m.compute_expectations(A_n, **kw)
    chunked = m.compute_expectations(A_n, **kw, chunk_size=chunk_size)
    assert_array_almost_equal(chunked["mu"], dense["mu"], decimal=10)


@pytest.mark.parametrize("chunk_size", [50, 137, 5000])
def test_compute_multiple_expectations_chunked(small_oscillator, chunk_size):
    m, u_kn, N_k = small_oscillator
    rng = np.random.default_rng(2)
    A_in = rng.uniform(0.5, 2.0, size=(4, u_kn.shape[1]))  # 4 observables
    u_n = u_kn[0, :]
    dense = m.compute_multiple_expectations(
        A_in, u_n, compute_uncertainty=False)
    chunked = m.compute_multiple_expectations(
        A_in, u_n, compute_uncertainty=False, chunk_size=chunk_size)
    assert_array_almost_equal(chunked["mu"], dense["mu"], decimal=10)


def test_chunked_rejects_compute_uncertainty(small_oscillator):
    m, u_kn, N_k = small_oscillator
    with pytest.raises(ParameterError, match="return_theta"):
        m.compute_perturbed_free_energies(
            u_kn, compute_uncertainty=True, chunk_size=50)
    with pytest.raises(ParameterError, match="return_theta"):
        m.compute_expectations(
            u_kn[0], compute_uncertainty=True, chunk_size=50)


def test_chunked_rejects_bootstrap(small_oscillator):
    m, u_kn, N_k = small_oscillator
    with pytest.raises(ParameterError, match="bootstrap"):
        m.compute_perturbed_free_energies(
            u_kn, compute_uncertainty=False,
            uncertainty_method="bootstrap", chunk_size=50)
