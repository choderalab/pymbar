"""Unit tests for pymbar._chunked: equivalence to dense scipy implementations."""
import numpy as np
import pytest
from scipy.special import logsumexp as sp_logsumexp

from pymbar import _chunked
from pymbar.utils_for_testing import assert_array_almost_equal


def _make_problem(K=10, N=2000, seed=0):
    """Synthetic MBAR problem with reasonable overlap and a near-stationary f_k."""
    rng = np.random.default_rng(seed)
    centers = np.linspace(-2.0, 2.0, K)
    u_kn = centers[:, None] + rng.standard_normal((K, N)) * 0.5
    N_k = np.full(K, N // K, dtype=np.float64)
    f_k = -np.array([np.log(np.mean(np.exp(-u_kn[k]))) for k in range(K)])
    return u_kn, N_k, f_k


def _dense_log_denom(u_kn, N_k, f_k):
    return sp_logsumexp(f_k - u_kn.T, b=N_k, axis=1)


def _dense_log_num_k(u_kn, log_denom):
    return sp_logsumexp(-log_denom - u_kn, axis=1)


def _dense_hessian(u_kn, N_k, f_k):
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    W = np.exp(f_k - u_kn.T - log_denom[:, None])
    H = (W.T @ W) * N_k * N_k[:, None] - np.diag(W.sum(0) * N_k)
    return -H


@pytest.mark.parametrize("chunk_size", [1, 7, 100, 999, 10_000])
def test_chunked_log_denominator(chunk_size):
    u_kn, N_k, f_k = _make_problem(K=10, N=2000)
    assert_array_almost_equal(
        _chunked.chunked_log_denominator(u_kn, N_k, f_k, chunk_size),
        _dense_log_denom(u_kn, N_k, f_k), decimal=12)


@pytest.mark.parametrize("chunk_size", [1, 13, 200, 5000])
def test_chunked_log_numerator_k(chunk_size):
    u_kn, N_k, f_k = _make_problem(K=8, N=1500)
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    assert_array_almost_equal(
        _chunked.chunked_log_numerator_k(u_kn, N_k, log_denom, chunk_size),
        _dense_log_num_k(u_kn, log_denom), decimal=12)


@pytest.mark.parametrize("chunk_size", [10, 250])
def test_chunked_mbar_objective(chunk_size):
    u_kn, N_k, f_k = _make_problem()
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    expected = float(np.sum(log_denom) - np.dot(N_k, f_k))
    got = _chunked.chunked_mbar_objective(u_kn, N_k, f_k, chunk_size)
    assert abs(got - expected) < 1e-9 * max(1.0, abs(expected))


@pytest.mark.parametrize("chunk_size", [10, 250])
def test_chunked_mbar_gradient(chunk_size):
    u_kn, N_k, f_k = _make_problem()
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    log_num = _dense_log_num_k(u_kn, log_denom)
    expected = -1.0 * N_k * (1.0 - np.exp(f_k + log_num))
    assert_array_almost_equal(
        _chunked.chunked_mbar_gradient(u_kn, N_k, f_k, chunk_size),
        expected, decimal=10)


@pytest.mark.parametrize("chunk_size", [10, 500])
def test_chunked_mbar_objective_and_gradient(chunk_size):
    u_kn, N_k, f_k = _make_problem()
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    log_num = _dense_log_num_k(u_kn, log_denom)
    obj_exp = float(np.sum(log_denom) - np.dot(N_k, f_k))
    grad_exp = -1.0 * N_k * (1.0 - np.exp(f_k + log_num))
    obj, grad = _chunked.chunked_mbar_objective_and_gradient(
        u_kn, N_k, f_k, chunk_size)
    assert abs(obj - obj_exp) < 1e-9 * max(1.0, abs(obj_exp))
    assert_array_almost_equal(grad, grad_exp, decimal=10)


@pytest.mark.parametrize("chunk_size", [10, 200])
def test_chunked_mbar_hessian(chunk_size):
    u_kn, N_k, f_k = _make_problem(K=6, N=1000)
    expected = _dense_hessian(u_kn, N_k, f_k)
    got = _chunked.chunked_mbar_hessian(u_kn, N_k, f_k, chunk_size)
    assert_array_almost_equal(got, expected, decimal=8)


@pytest.mark.parametrize("chunk_size", [10, 500])
def test_chunked_self_consistent_update(chunk_size):
    u_kn, N_k, f_k = _make_problem()
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    expected = -_dense_log_num_k(u_kn, log_denom)
    assert_array_almost_equal(
        _chunked.chunked_self_consistent_update(u_kn, N_k, f_k, chunk_size),
        expected, decimal=12)


@pytest.mark.parametrize("chunk_size", [10, 500])
def test_chunked_precondition_u_kn(chunk_size):
    u_kn, N_k, f_k = _make_problem()
    u_dense = u_kn - u_kn.min(0)
    u_dense = u_dense + (_dense_log_denom(u_dense, N_k, f_k)
                          - np.dot(N_k, f_k) / N_k.sum())
    u_chunked = _chunked.chunked_precondition_u_kn(
        u_kn.copy(), N_k, f_k, chunk_size)
    assert_array_almost_equal(u_chunked, u_dense, decimal=10)


@pytest.mark.parametrize("chunk_size", [10, 500])
def test_chunked_mbar_log_W_nk(chunk_size):
    u_kn, N_k, f_k = _make_problem()
    log_denom = _dense_log_denom(u_kn, N_k, f_k)
    expected = f_k - u_kn.T - log_denom[:, None]
    got = _chunked.chunked_mbar_log_W_nk(u_kn, N_k, f_k, chunk_size)
    assert_array_almost_equal(got, expected, decimal=12)


def test_zero_N_k_handled():
    """States with N_k=0 contribute nothing (log_N goes to -inf cleanly)."""
    u_kn, N_k, f_k = _make_problem(K=5, N=500)
    N_k = N_k.copy()
    N_k[2] = 0.0
    assert_array_almost_equal(
        _chunked.chunked_log_denominator(u_kn, N_k, f_k, chunk_size=100),
        _dense_log_denom(u_kn, N_k, f_k), decimal=12)
