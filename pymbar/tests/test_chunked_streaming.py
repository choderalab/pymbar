"""Tests for the streaming u_kn path: MBAR(u_kn=callable, N_k, chunk_size=N).

Verifies that fitting with a chunk-yielding callable produces the same f_k
as the dense-ndarray path; that the API enforces required preconditions;
and that peak Python memory is bounded by O(chunk_size * K) when the
callable streams without holding the full matrix.
"""
import tracemalloc

import numpy as np
import pytest

import pymbar
from pymbar import _chunked
from pymbar.utils_for_testing import assert_array_almost_equal, oscillators


def _make_provider(u_kn_dense, chunk_size):
    """Build a callable that yields chunks of u_kn_dense without holding any
    other reference (caller can del the dense array after calling _make_provider)."""
    def provider():
        N = u_kn_dense.shape[1]
        for s in range(0, N, chunk_size):
            yield u_kn_dense[:, s:s + chunk_size]
    return provider


@pytest.fixture(scope="module")
def small_oscillator():
    name, u_kn, N_k, s_n = oscillators(8, 300)
    return u_kn, N_k


@pytest.mark.parametrize("chunk_size", [50, 137, 5000])
def test_streaming_matches_dense(small_oscillator, chunk_size):
    u_kn, N_k = small_oscillator
    m_dense = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10,
                          chunk_size=chunk_size)
    m_stream = pymbar.MBAR(_make_provider(u_kn, chunk_size), N_k,
                           relative_tolerance=1e-10,
                           chunk_size=chunk_size,
                           cache_log_W_nk=False)
    assert_array_almost_equal(np.asarray(m_stream.f_k) - m_stream.f_k[0],
                              np.asarray(m_dense.f_k) - m_dense.f_k[0],
                              decimal=7)


def test_streaming_requires_chunk_size(small_oscillator):
    u_kn, N_k = small_oscillator
    with pytest.raises(ValueError, match="chunk_size required"):
        pymbar.MBAR(_make_provider(u_kn, 100), N_k,
                    cache_log_W_nk=False)


def test_streaming_rejects_log_W_nk_cache(small_oscillator):
    u_kn, N_k = small_oscillator
    with pytest.raises(ValueError, match="cache_log_W_nk=True"):
        pymbar.MBAR(_make_provider(u_kn, 100), N_k, chunk_size=100)


def test_streaming_rejects_bootstrap(small_oscillator):
    u_kn, N_k = small_oscillator
    with pytest.raises(NotImplementedError, match="streaming"):
        pymbar.MBAR(_make_provider(u_kn, 100), N_k,
                    chunk_size=100, cache_log_W_nk=False, n_bootstraps=5)


def test_streaming_rejects_empty_states(small_oscillator):
    u_kn, N_k = small_oscillator
    N_k_with_zero = np.array(N_k, dtype=np.int64)
    N_k_with_zero[3] = 0
    with pytest.raises(NotImplementedError, match="N_k=0"):
        pymbar.MBAR(_make_provider(u_kn, 100), N_k_with_zero,
                    chunk_size=100, cache_log_W_nk=False)


def test_iter_chunks_dense():
    u_kn = np.arange(60, dtype=float).reshape(6, 10)
    chunks = list(_chunked._iter_chunks(u_kn, 4))
    starts = [s for s, _ in chunks]
    sizes = [c.shape[1] for _, c in chunks]
    assert starts == [0, 4, 8]
    assert sizes == [4, 4, 2]
    # last chunk smaller: dense slice of (6, 2) reaches the end
    assert chunks[-1][1].shape == (6, 2)


def test_iter_chunks_callable():
    u_kn = np.arange(60, dtype=float).reshape(6, 10)
    provider = _make_provider(u_kn, 4)
    chunks = list(_chunked._iter_chunks(provider, 4))
    starts = [s for s, _ in chunks]
    sizes = [c.shape[1] for _, c in chunks]
    assert starts == [0, 4, 8]
    assert sizes == [4, 4, 2]


def test_iter_chunks_rejects_other_types():
    with pytest.raises(TypeError, match="ndarray or zero-arg callable"):
        list(_chunked._iter_chunks([1, 2, 3], 1))


def test_streaming_helpers_match_dense():
    """Per-helper equivalence ndarray vs callable on the same data."""
    rng = np.random.default_rng(0)
    K, N = 8, 1600  # K divides N so N_k.sum() == N
    u_kn = rng.standard_normal((K, N))
    N_k = np.full(K, N // K, dtype=np.float64)
    f_k = np.zeros(K)
    chunk_size = 250

    log_denom_d = _chunked.chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    log_denom_s = _chunked.chunked_log_denominator(
        _make_provider(u_kn, chunk_size), N_k, f_k, chunk_size)
    assert_array_almost_equal(log_denom_s, log_denom_d, decimal=12)

    log_num_d = _chunked.chunked_log_numerator_k(u_kn, N_k, log_denom_d, chunk_size)
    log_num_s = _chunked.chunked_log_numerator_k(
        _make_provider(u_kn, chunk_size), N_k, log_denom_d, chunk_size)
    assert_array_almost_equal(log_num_s, log_num_d, decimal=12)

    H_d = _chunked.chunked_mbar_hessian(u_kn, N_k, f_k, chunk_size)
    H_s = _chunked.chunked_mbar_hessian(
        _make_provider(u_kn, chunk_size), N_k, f_k, chunk_size)
    assert_array_almost_equal(H_s, H_d, decimal=8)


def test_streaming_peak_memory_bounded():
    """Streaming path: peak Python allocation through chunked_log_denominator
    is bounded by O(chunk_size * K), independent of total N. Compares to a
    dense (N, K) reference materialisation.
    """
    K, N, chunk_size = 50, 200_000, 10_000
    u_kn = np.random.default_rng(0).standard_normal((K, N))
    N_k = np.full(K, N // K, dtype=np.float64)
    f_k = np.zeros(K)

    # Dense reference: one (K, N) intermediate, similar shape to scipy logsumexp
    def _dense_ref(u_kn, N_k, f_k):
        log_N = np.log(N_k)
        a = (f_k + log_N)[:, None] - u_kn
        m = a.max(axis=0)
        return m + np.log(np.exp(a - m).sum(axis=0))

    tracemalloc.start(); tracemalloc.reset_peak()
    _dense_ref(u_kn, N_k, f_k)
    _, peak_dense = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start(); tracemalloc.reset_peak()
    _chunked.chunked_log_denominator(_make_provider(u_kn, chunk_size),
                                      N_k, f_k, chunk_size)
    _, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nstreaming log_denominator: dense {peak_dense/1e6:.1f} MB, "
          f"stream {peak_stream/1e6:.1f} MB "
          f"({peak_dense/max(peak_stream,1):.1f}x smaller)")
    assert peak_stream < peak_dense / 4


def test_streaming_precondition_returns_provider():
    rng = np.random.default_rng(0)
    K, N = 6, 1200  # K divides N
    u_kn = rng.standard_normal((K, N))
    N_k = np.full(K, N // K, dtype=np.float64)
    f_k = np.zeros(K)
    cs = 200

    # Dense reference
    u_dense = u_kn.copy()
    u_dense = _chunked.chunked_precondition_u_kn(u_dense, N_k, f_k, cs)

    # Streaming: returns a callable
    out = _chunked.chunked_precondition_u_kn(_make_provider(u_kn, cs),
                                              N_k, f_k, cs)
    assert callable(out)
    streamed = np.concatenate([c for c in out()], axis=1)
    assert_array_almost_equal(streamed, u_dense, decimal=8)
