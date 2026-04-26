"""Memory regression for the chunked helpers.

Compares peak Python allocations of pymbar._chunked vs an inline numpy
reference that materializes the (N, K) intermediate. JAX-independent --
operates directly on the helpers, not the dispatcher path.
"""
import tracemalloc

import numpy as np
import pytest

from pymbar import _chunked


def _make_problem(K, N, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.linspace(-1.5, 1.5, K)
    u_kn = centers[:, None] + rng.standard_normal((K, N)) * 0.3
    N_k = np.full(K, N // K, dtype=np.float64)
    return u_kn, N_k, np.zeros(K, dtype=np.float64)


def _peak(fn, *args):
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fn(*args)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


def _dense_log_denominator(u_kn, N_k, f_k):
    # Materializes a (K, N) intermediate -- the cost we're trying to avoid.
    log_N = np.log(N_k)
    a = (f_k + log_N)[:, None] - u_kn
    m = a.max(axis=0)
    return m + np.log(np.exp(a - m).sum(axis=0))


def _dense_hessian(u_kn, N_k, f_k):
    log_denom = _dense_log_denominator(u_kn, N_k, f_k)
    W = np.exp(f_k[None, :] - u_kn.T - log_denom[:, None])
    H = W.T @ W
    H *= N_k
    H *= N_k[:, None]
    H -= np.diag(W.sum(0) * N_k)
    return -H


def _dense_precondition(u_kn, N_k, f_k):
    u_kn = u_kn - u_kn.min(0)
    log_denom = _dense_log_denominator(u_kn, N_k, f_k)
    return u_kn + (log_denom - np.dot(N_k, f_k) / N_k.sum())


@pytest.mark.parametrize("op", ["log_denominator", "hessian", "precondition"])
def test_chunked_peak_below_dense(op):
    """At K=50, N=200k, dense (N, K) intermediate is 80 MB. Chunked at
    chunk_size=10k should peak well below that. Assert chunked < dense / 4.
    """
    K, N, chunk_size = 50, 200_000, 10_000
    u_kn, N_k, f_k = _make_problem(K, N)

    if op == "log_denominator":
        peak_dense = _peak(_dense_log_denominator, u_kn, N_k, f_k)
        peak_chunked = _peak(_chunked.chunked_log_denominator,
                              u_kn, N_k, f_k, chunk_size)
    elif op == "hessian":
        peak_dense = _peak(_dense_hessian, u_kn, N_k, f_k)
        peak_chunked = _peak(_chunked.chunked_mbar_hessian,
                              u_kn, N_k, f_k, chunk_size)
    elif op == "precondition":
        peak_dense = _peak(_dense_precondition, u_kn.copy(), N_k, f_k)
        peak_chunked = _peak(_chunked.chunked_precondition_u_kn,
                              u_kn.copy(), N_k, f_k, chunk_size)

    print(f"\n{op}: dense {peak_dense/1e6:.1f} MB, "
          f"chunked {peak_chunked/1e6:.1f} MB "
          f"({peak_dense/max(peak_chunked,1):.1f}x smaller)")
    assert peak_chunked < peak_dense / 4
