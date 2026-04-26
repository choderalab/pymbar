"""End-to-end tests for MBAR(..., chunk_size=N): same f_k as dense path."""
import numpy as np
import pytest

import pymbar
from pymbar import mbar_solvers
from pymbar.utils_for_testing import assert_array_almost_equal, oscillators


@pytest.fixture(scope="module")
def small_oscillator():
    name, u_kn, N_k, s_n = oscillators(8, 300)
    return u_kn, N_k


@pytest.mark.parametrize("chunk_size", [50, 137, 5000])
def test_mbar_chunked_matches_dense(small_oscillator, chunk_size):
    u_kn, N_k = small_oscillator
    m_dense = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10)
    m_chunked = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10,
                            chunk_size=chunk_size)
    assert_array_almost_equal(np.asarray(m_chunked.f_k) - m_chunked.f_k[0],
                              np.asarray(m_dense.f_k) - m_dense.f_k[0],
                              decimal=7)


def test_mbar_chunked_log_W_nk_matches(small_oscillator):
    u_kn, N_k = small_oscillator
    m_dense = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10)
    m_chunked = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10, chunk_size=70)
    assert_array_almost_equal(np.asarray(m_chunked.Log_W_nk),
                              np.asarray(m_dense.Log_W_nk), decimal=8)


def test_mbar_chunked_no_log_W_nk(small_oscillator):
    u_kn, N_k = small_oscillator
    m = pymbar.MBAR(u_kn, N_k, chunk_size=70, cache_log_W_nk=False)
    assert m.Log_W_nk is None


def test_mbar_chunked_rejects_bootstrap(small_oscillator):
    u_kn, N_k = small_oscillator
    with pytest.raises(NotImplementedError, match="chunk_size with n_bootstraps"):
        pymbar.MBAR(u_kn, N_k, chunk_size=50, n_bootstraps=5)


@pytest.mark.parametrize("chunk_size", [50, 5000])
def test_mbar_chunked_adaptive_solver(small_oscillator, chunk_size):
    """The 'adaptive' solver routes through jax_core_adaptive, which is
    `@jit_or_pass_after_bitsize`-decorated. The chunked path's Python
    loops with side-effecting numpy buffer writes cannot be traced inside
    that outer JIT (cold cache: ConcretizationTypeError on
    `int(np.sum(N_k))` against a traced N_k; warm cache: silently reuses
    the cached dense jaxpr because `_CHUNK_SIZE` is a Python global the
    JIT cache can't key on). The fix in `mbar_solvers.staggered_jit`
    bypasses the outer JIT when `_CHUNK_SIZE is not None`.

    Regression test: forces the adaptive method explicitly so the JIT
    path is exercised — DEFAULT_SOLVER_PROTOCOL prefers `hybr` first
    (scipy, not JIT'd) and only falls back to `adaptive` for harder
    problems, so a small test fixture would otherwise skip this path.
    `jax.clear_caches()` at entry ensures we hit the cold-cache trace
    path regardless of test ordering.
    """
    try:
        import jax
        jax.clear_caches()
    except ImportError:
        pass
    u_kn, N_k = small_oscillator
    adaptive_only = (dict(method="adaptive", options=dict(min_sc_iter=0)),)
    m_chunked = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10,
                            chunk_size=chunk_size,
                            solver_protocol=adaptive_only)
    # Independent dense reference on a clean cache so the cached chunked
    # jaxpr can't sneak in: separate import / fresh MBAR with no chunking.
    try:
        import jax
        jax.clear_caches()
    except ImportError:
        pass
    m_dense = pymbar.MBAR(u_kn, N_k, relative_tolerance=1e-10,
                          solver_protocol=adaptive_only)
    assert_array_almost_equal(np.asarray(m_chunked.f_k) - m_chunked.f_k[0],
                              np.asarray(m_dense.f_k) - m_dense.f_k[0],
                              decimal=6)


def test_chunk_size_setter_validates():
    with pytest.raises(ValueError, match=">= 1"):
        mbar_solvers.set_chunk_size(0)
    mbar_solvers.set_chunk_size(None)
    assert mbar_solvers.get_chunk_size() is None
    mbar_solvers.set_chunk_size(100)
    assert mbar_solvers.get_chunk_size() == 100
    mbar_solvers.set_chunk_size(None)


def test_chunk_size_context_restores():
    mbar_solvers.set_chunk_size(None)
    with mbar_solvers.chunk_size_context(500):
        assert mbar_solvers.get_chunk_size() == 500
        with mbar_solvers.chunk_size_context(None):
            assert mbar_solvers.get_chunk_size() is None
        assert mbar_solvers.get_chunk_size() == 500
    assert mbar_solvers.get_chunk_size() is None
